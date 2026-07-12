"""
Scalar response surrogate over a parametric unsteady DOE.

This trains a small surrogate over the scalar response metrics produced
by experiments/run_parametric_unsteady_doe.py. The features are the
design variables (q_offset, epsilon, frequency_hz, phase) and the targets
are time-averaged means and post-transient amplitudes (and optional probe
pressure amplitudes) extracted by response_metrics.

This is NOT a time-accurate reduced-order model. It is a scalar response
surrogate intended for trend interpolation and candidate screening only.

Acceptable model hierarchy (preferred first):
    1. GPSurrogate (ARD-RBF) from optimization.py when there are enough
       valid cases per target.
    2. Ridge regression with low-order polynomial features.
    3. Inverse-distance interpolation fallback for tiny DOEs.

Validation uses leave-one-out (LOO) when valid sample count permits, else
the script honestly writes a warning into surrogate_validation_summary.json
and skips accuracy reporting.
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.run_static_wall_sweep import (
    ARTIFACT_SCHEMA_VERSION,
    require_schema_v2,
    write_csv,
    write_json,
)
from optimization import GPSurrogate


FEATURE_KEYS = ("q_offset", "epsilon", "frequency_hz", "phase")

DEFAULT_TARGETS = (
    "exit_mach_mean",
    "exit_mach_amplitude",
    "tpr_mean",
    "tpr_amplitude",
    "shock_x_mean",
    "pressure_recovery_mean",
    "pressure_recovery_amplitude",
    "mdot_exit_mean",
    "mdot_exit_amplitude",
    "mass_defect_mean",
    "probe_throat_pressure_amplitude",
    "probe_combustor_pressure_amplitude",
    "probe_exit_pressure_amplitude",
)

MIN_LOO_SAMPLES = 5
MIN_FIT_SAMPLES = 3


def _parse_float(text):
    """Cast a CSV cell to float, returning None for empty/invalid cells."""
    if text is None:
        return None
    text = str(text).strip()
    if text == "" or text.lower() in {"nan", "null", "none"}:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not np.isfinite(value):
        return None
    return value


def load_doe_summary(doe_root):
    """Read DOE summary.csv and design_matrix.csv (if present)."""
    summary_path = Path(doe_root) / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing DOE summary.csv: {summary_path}")
    with summary_path.open() as f:
        summary_rows = list(csv.DictReader(f))

    design_path = Path(doe_root) / "design_matrix.csv"
    design_rows = []
    if design_path.is_file():
        with design_path.open() as f:
            design_rows = list(csv.DictReader(f))
    return summary_rows, design_rows


def _drop_failed(rows):
    """Drop summary rows that did not produce valid post-transient data."""
    cleaned = []
    for r in rows:
        status = (r.get("status") or "").strip().lower()
        if status and status != "ok":
            continue
        x = [_parse_float(r.get(k)) for k in FEATURE_KEYS]
        if any(v is None for v in x):
            continue
        cleaned.append(r)
    return cleaned


def feature_names_for_rows(rows):
    """Use circular features when the design actually varies phase."""
    phases = {_parse_float(row.get("phase")) for row in rows}
    phases.discard(None)
    if len(phases) > 1:
        return ("q_offset", "epsilon", "frequency_hz", "phase_sin", "phase_cos")
    return FEATURE_KEYS


def build_feature_matrix(rows, feature_names=None):
    """Return the feature matrix with optional circular phase encoding."""
    if feature_names is None:
        feature_names = feature_names_for_rows(rows)
    matrix = []
    for row in rows:
        phase = float(row["phase"])
        values = {
            "q_offset": float(row["q_offset"]),
            "epsilon": float(row["epsilon"]),
            "frequency_hz": float(row["frequency_hz"]),
            "phase": phase,
            "phase_sin": float(np.sin(phase)),
            "phase_cos": float(np.cos(phase)),
        }
        matrix.append([values[name] for name in feature_names])
    return np.asarray(matrix, dtype=float)


def _parse_bool(text):
    return str(text).strip().lower() in {"1", "true", "yes"}


def _support_key_for_target(key):
    if not key.endswith("_amplitude"):
        return None
    stem = key[:-len("_amplitude")]
    if stem.startswith("probe_") and stem.endswith("_pressure"):
        return f"{stem[:-len('_pressure')]}_supported"
    return f"{stem}_supported"


def collect_target_vector(rows, key):
    """Return (values_finite, mask) for one target metric."""
    parsed = [_parse_float(r.get(key)) for r in rows]
    support_key = _support_key_for_target(key)
    supported = ([True] * len(rows) if support_key is None else
                 [_parse_bool(row.get(support_key)) for row in rows])
    mask = np.array([v is not None and ok for v, ok in zip(parsed, supported)])
    values = np.array([v if v is not None else np.nan for v in parsed], dtype=float)
    return values, mask


def normalize_features(X):
    """Per-dimension min-max normalization; returns (X_norm, mins, ranges)."""
    if X.shape[0] == 0:
        return X.copy(), np.zeros(X.shape[1]), np.ones(X.shape[1])
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    rng = np.maximum(maxs - mins, 1e-12)
    return (X - mins) / rng, mins, rng


def fit_ridge(X_train, y_train, degree=2, alpha=1.0e-3):
    """Lightweight polynomial ridge regression up to ``degree``."""
    Phi = _polynomial_features(X_train, degree)
    n, k = Phi.shape
    A = Phi.T @ Phi + alpha * np.eye(k)
    beta = np.linalg.solve(A, Phi.T @ y_train)
    return ("ridge", degree, alpha, beta)


def predict_ridge(model, X_query):
    """Evaluate ridge model."""
    _, degree, _, beta = model
    Phi = _polynomial_features(X_query, degree)
    return Phi @ beta


def _polynomial_features(X, degree):
    """Build polynomial features up to a given total degree (no cross terms)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    cols = [np.ones(X.shape[0])]
    for d in range(1, degree + 1):
        cols.append(X**d)
    return np.column_stack(cols)


def fit_inverse_distance(X_train, y_train):
    """Trivial inverse-distance interpolator stored as (X, y)."""
    return ("inverse_distance", X_train.copy(), y_train.copy())


def predict_inverse_distance(model, X_query):
    """Evaluate inverse-distance model."""
    _, X_train, y_train = model
    if X_query.ndim == 1:
        X_query = X_query[np.newaxis, :]
    out = np.zeros(X_query.shape[0])
    for k, x in enumerate(X_query):
        d = np.linalg.norm(X_train - x, axis=1)
        if np.any(d <= 1.0e-12):
            out[k] = float(y_train[np.argmin(d)])
            continue
        w = 1.0 / d**2
        out[k] = float((w * y_train).sum() / w.sum())
    return out


def fit_gp(X_train, y_train, n_restarts=0):
    """Fit a GP cache, using stable fixed hyperparameters by default."""
    gp = GPSurrogate(ndim=X_train.shape[1])
    gp.train(X_train, y_train, n_restarts=n_restarts)
    return ("gp", gp)


def predict_gp(model, X_query):
    """Evaluate GP model."""
    _, gp = model
    mu, _ = gp.predict(X_query)
    return mu


def fit_model(X_train_norm, y_train):
    """Pick the most expressive model the sample count supports."""
    n = X_train_norm.shape[0]
    if n >= MIN_LOO_SAMPLES:
        try:
            return fit_gp(X_train_norm, y_train), "gp_fixed_hyperparameters"
        except Exception:
            pass
    if n >= MIN_FIT_SAMPLES:
        try:
            degree = 2 if n >= 6 else 1
            return fit_ridge(X_train_norm, y_train, degree=degree), f"ridge_deg{degree}"
        except Exception:
            pass
    return fit_inverse_distance(X_train_norm, y_train), "inverse_distance"


def predict_model(model, X_query_norm):
    """Dispatch prediction based on model kind."""
    kind = model[0]
    if kind == "gp":
        return predict_gp(model, X_query_norm)
    if kind == "ridge":
        return predict_ridge(model, X_query_norm)
    if kind == "inverse_distance":
        return predict_inverse_distance(model, X_query_norm)
    raise ValueError(f"unknown model kind: {kind}")


def leave_one_out(X_norm, y, full_model=None, full_kind=None):
    """Return arrays of (predicted, actual) for LOO over the dataset.

    GP hyperparameters are fit once on the full dataset and held fixed
    across the folds (each fold only refits the Cholesky/alpha caches on
    its training subset). Re-optimizing 6 hyperparameters per fold per
    target made the demo pipeline take tens of minutes for no accuracy
    benefit at these sample sizes.
    """
    n = X_norm.shape[0]
    preds = np.zeros(n)
    if full_model is None or full_kind is None:
        full_model, full_kind = fit_model(X_norm, y)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if str(full_kind).startswith("gp"):
            _, gp_full = full_model
            gp_i = GPSurrogate(ndim=X_norm.shape[1])
            gp_i.log_sigma_f = gp_full.log_sigma_f
            gp_i.log_lengthscales = gp_full.log_lengthscales.copy()
            gp_i.log_sigma_n = gp_full.log_sigma_n
            gp_i.train(X_norm[mask], y[mask], n_restarts=0)
            model = ("gp", gp_i)
        else:
            model, _ = fit_model(X_norm[mask], y[mask])
        preds[i] = predict_model(model, X_norm[i:i + 1])[0]
    return preds, y.copy()


def plot_predicted_vs_actual(predicted, actual, target_name, path):
    """Predicted-vs-actual scatter for one target."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual, predicted, s=40, color="#1f77b4", alpha=0.8)
    lo = float(min(actual.min(), predicted.min()))
    hi = float(max(actual.max(), predicted.max()))
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="y = x")
    ax.set_xlabel(f"actual {target_name}")
    ax.set_ylabel(f"predicted {target_name}")
    ax.set_title(f"Surrogate predicted vs actual\n{target_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def build_surrogate(doe_root, output_root=None, targets=None,
                    holdout_fraction=0.2, seed=42):
    """Train surrogate per target and emit reports + plots."""
    doe_root = Path(doe_root)
    if not doe_root.is_absolute():
        doe_root = REPO_ROOT / doe_root
    require_schema_v2(doe_root, "unsteady DOE")

    if output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "runs" / f"unsteady_response_surrogate_{stamp}"
    else:
        output_root = Path(output_root)
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root

    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if targets is None:
        targets = list(DEFAULT_TARGETS)

    summary_rows, design_rows = load_doe_summary(doe_root)
    valid_rows = _drop_failed(summary_rows)
    feature_names = feature_names_for_rows(valid_rows)
    rng = np.random.default_rng(seed)

    train_data_rows = []
    for r in valid_rows:
        record = {k: _parse_float(r.get(k)) for k in FEATURE_KEYS}
        record["case_id"] = r.get("case_id", "")
        for t in targets:
            record[t] = _parse_float(r.get(t))
        train_data_rows.append(record)
    write_csv(output_root / "surrogate_training_data.csv", train_data_rows,
              fieldnames=["case_id", *FEATURE_KEYS, *targets])

    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "doe_root": str(doe_root),
        "n_summary_rows": len(summary_rows),
        "n_valid_rows": len(valid_rows),
        "raw_design_features": list(FEATURE_KEYS),
        "features": list(feature_names),
        "circular_phase_encoding": "phase_sin" in feature_names,
        "targets": list(targets),
        "min_loo_samples": MIN_LOO_SAMPLES,
        "min_fit_samples": MIN_FIT_SAMPLES,
        "holdout_fraction": holdout_fraction,
        "seed": int(seed),
        "model_hierarchy": ["gp_fixed_hyperparameters", "ridge", "inverse_distance"],
        "gp_hyperparameters_optimized": False,
        "scalar_response_surrogate": True,
        "time_accurate_rom": False,
        "limitations": [
            "Scalar response surrogate only — predicts post-transient scalar metrics, not time histories.",
            "Trained on low-fidelity effective-area forcing data from the Python prototype.",
            "Not a substitute for high-fidelity unsteady CFD.",
        ],
    }

    if len(valid_rows) < MIN_FIT_SAMPLES:
        message = (
            f"only {len(valid_rows)} valid DOE rows available; need at least "
            f"{MIN_FIT_SAMPLES}. Surrogate training skipped."
        )
        write_json(output_root / "model_metadata.json", metadata)
        write_json(output_root / "surrogate_validation_summary.json", {
            "status": "insufficient_data",
            "message": message,
            "metadata": metadata,
        })
        print(message)
        return output_root, {"status": "insufficient_data"}

    X = build_feature_matrix(valid_rows, feature_names=feature_names)
    X_norm, x_mins, x_ranges = normalize_features(X)
    metadata["feature_normalization"] = {
        "min": x_mins.tolist(), "range": x_ranges.tolist(),
    }

    validation_summary = {"status": "ok", "metadata": metadata, "targets": {}}

    for target in targets:
        y, mask = collect_target_vector(valid_rows, target)
        n_valid_target = int(mask.sum())
        result = {
            "n_valid": n_valid_target,
            "n_dropped": int((~mask).sum()),
            "model_selected": None,
            "loo": None,
            "holdout": None,
            "warning": "",
        }

        if n_valid_target < MIN_FIT_SAMPLES:
            result["warning"] = (
                f"only {n_valid_target} valid cases for target; need "
                f"{MIN_FIT_SAMPLES}. Skipping training."
            )
            validation_summary["targets"][target] = result
            continue

        X_tgt = X_norm[mask]
        y_tgt = y[mask]
        model, model_name = fit_model(X_tgt, y_tgt)
        result["model_selected"] = model_name

        if n_valid_target >= MIN_LOO_SAMPLES:
            try:
                preds, actuals = leave_one_out(
                    X_tgt, y_tgt, full_model=model, full_kind=model_name,
                )
                err = preds - actuals
                rmse = float(np.sqrt(np.mean(err**2)))
                mae = float(np.mean(np.abs(err)))
                rel_rmse = float(rmse / max(np.std(actuals), 1.0e-12))
                result["loo"] = {
                    "rmse": rmse,
                    "mae": mae,
                    "rmse_over_std": rel_rmse,
                    "n_samples": int(n_valid_target),
                }
                plot_predicted_vs_actual(
                    preds, actuals, target,
                    plots_dir / f"predicted_vs_actual_{target}.png",
                )
            except Exception as exc:
                result["warning"] = f"LOO failed: {type(exc).__name__}: {exc}"

        try:
            n_train = max(int(round((1.0 - holdout_fraction) * n_valid_target)),
                          MIN_FIT_SAMPLES)
            n_train = min(n_train, n_valid_target - 1)
            if n_train >= MIN_FIT_SAMPLES and n_train < n_valid_target:
                idx = np.arange(n_valid_target)
                rng.shuffle(idx)
                train_idx = idx[:n_train]
                test_idx = idx[n_train:]
                model_h, _ = fit_model(X_tgt[train_idx], y_tgt[train_idx])
                pred_h = predict_model(model_h, X_tgt[test_idx])
                err_h = pred_h - y_tgt[test_idx]
                rmse_h = float(np.sqrt(np.mean(err_h**2)))
                result["holdout"] = {
                    "n_train": int(n_train),
                    "n_test": int(len(test_idx)),
                    "rmse": rmse_h,
                }
        except Exception as exc:
            result.setdefault("warnings", []).append(
                f"holdout failed: {type(exc).__name__}: {exc}",
            )

        validation_summary["targets"][target] = result

    write_json(output_root / "model_metadata.json", metadata)
    write_json(output_root / "surrogate_validation_summary.json", validation_summary)

    plotted_any = any(
        validation_summary["targets"][t].get("loo") is not None for t in targets
    )
    note = (
        "Trained scalar response surrogate. This is NOT a time-accurate "
        "reduced-order model."
    )
    print(note)
    print(f"Valid DOE rows used: {len(valid_rows)}/{len(summary_rows)}")
    print(f"Predicted-vs-actual plots: {'generated' if plotted_any else 'skipped (too few samples)'}")
    print(f"Surrogate output: {output_root}")
    return output_root, validation_summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train a scalar response surrogate on a parametric unsteady DOE. "
            "Predicts post-transient scalar metrics, not time histories."
        ),
    )
    parser.add_argument("--doe-root", required=True,
                        help="Path to a parametric unsteady DOE output directory.")
    parser.add_argument("--output-root", default=None,
                        help="Path for surrogate outputs (default: timestamped runs/).")
    parser.add_argument("--targets", default=None,
                        help=("Comma-separated target metric names. Defaults to "
                              "scalar means and amplitudes from DEFAULT_TARGETS."))
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    targets = None
    if args.targets:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    build_surrogate(
        doe_root=args.doe_root,
        output_root=args.output_root,
        targets=targets,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
