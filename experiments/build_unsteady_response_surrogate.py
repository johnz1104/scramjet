"""
Scalar response surrogate over a parametric unsteady DOE.

This trains a small surrogate over the response metrics produced by
experiments/run_parametric_unsteady_doe.py. New DOE artifacts use reduced
frequency ``k = 2*pi*f*L_ref/u_ref`` as the frequency coordinate; schema-v2
artifacts created before that additive field remain readable through a
dimensional-frequency fallback.

Periodic response is represented by the complex harmonic coefficient

    H = amplitude * exp(-i * phase_lag),

so lag is never regressed as a discontinuous raw angle.  Amplitudes are also
fit in log10 space.  Validation converts both representations back to physical
amplitude/phase and reports circular phase error.

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
from scipy.stats import rankdata

from experiments.run_static_wall_sweep import (
    ARTIFACT_SCHEMA_VERSION,
    require_schema_v2,
    write_csv,
    write_json,
)
from optimization import GPSurrogate


LEGACY_FEATURE_KEYS = ("q_offset", "epsilon", "frequency_hz", "phase")
RAW_FEATURE_KEYS = (
    "q_offset", "epsilon", "frequency_hz", "reduced_frequency", "phase",
)

SCALAR_TARGETS = (
    "exit_mach_mean",
    "exit_mach_amplitude",
    "tpr_mean",
    "tpr_amplitude",
    "shock_x_mean",
    "shock_x_amplitude",
    "pressure_recovery_mean",
    "pressure_recovery_amplitude",
    "mdot_exit_mean",
    "mdot_exit_amplitude",
    "mass_defect_mean",
    "probe_throat_pressure_amplitude",
    "probe_combustor_pressure_amplitude",
    "probe_exit_pressure_amplitude",
)

COMPLEX_RESPONSE_STEMS = (
    "exit_mach",
    "tpr",
    "shock_x",
    "pressure_recovery",
    "probe_throat_pressure",
    "probe_combustor_pressure",
    "probe_exit_pressure",
)


def complex_target_names(stem):
    """Names of the real/imaginary targets for one harmonic response."""
    return f"{stem}_response_real", f"{stem}_response_imag"


DEFAULT_TARGETS = SCALAR_TARGETS + tuple(
    name
    for stem in COMPLEX_RESPONSE_STEMS
    for name in complex_target_names(stem)
)


def default_targets():
    """Scalar targets plus complex-response components for supported lags."""
    return list(DEFAULT_TARGETS)

MIN_LOO_SAMPLES = 5
MIN_FIT_SAMPLES = 3
MAX_REPORTABLE_CIRCULAR_LOO_RMSE_RAD = 1.0
MIN_REPORTABLE_SUPPORTED_CASES = 20


def response_reportability(circular_loo_rmse_rad, supported_cases,
                           varying_features):
    """Evaluate the explicit phase-response reporting gate.

    ``status: ok`` elsewhere means that a model was constructed.  It does not
    imply that its phase predictions are accurate or sufficiently supported.
    The reporting policy deliberately counts the four physical design axes
    before any sine/cosine expansion of phase.
    """
    supported_cases = int(supported_cases)
    varying_features = int(varying_features)
    required_cases = max(
        MIN_REPORTABLE_SUPPORTED_CASES,
        4 * (varying_features + 1),
    )
    policy = {
        "name": "circular_loo_support_gate_v1",
        "circular_loo_rmse_rad_max": MAX_REPORTABLE_CIRCULAR_LOO_RMSE_RAD,
        "supported_cases_minimum_rule": "max(20, 4*(varying_features+1))",
        "varying_features": varying_features,
        "supported_cases_required": required_cases,
        "supported_cases_observed": supported_cases,
    }
    reasons = []
    rmse = _parse_float(circular_loo_rmse_rad)
    if rmse is None:
        reasons.append("circular_loo_rmse_unavailable")
    elif rmse > MAX_REPORTABLE_CIRCULAR_LOO_RMSE_RAD:
        reasons.append("circular_loo_rmse_exceeds_1_rad")
    if supported_cases < required_cases:
        reasons.append("insufficient_supported_cases")
    return {
        "reportable": not reasons,
        "reportability_reasons": reasons,
        "reportability_policy": policy,
    }


def _varying_physical_feature_count(rows, row_indices=None):
    """Count varying physical design axes before circular phase expansion."""
    if row_indices is not None:
        rows = [rows[int(index)] for index in row_indices]
    if not rows:
        return 0
    frequency_key = (
        "reduced_frequency"
        if all(_parse_float(row.get("reduced_frequency")) is not None
               for row in rows)
        else "frequency_hz"
    )
    keys = ("q_offset", "epsilon", frequency_key, "phase")
    varying = 0
    for key in keys:
        values = [_parse_float(row.get(key)) for row in rows]
        values = np.asarray([value for value in values if value is not None])
        if len(values) > 1 and float(np.ptp(values)) > 1.0e-14:
            varying += 1
    return varying


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
        base = [_parse_float(r.get(k)) for k in ("q_offset", "epsilon", "phase")]
        has_frequency = any(
            _parse_float(r.get(k)) is not None
            for k in ("reduced_frequency", "frequency_hz")
        )
        if any(v is None for v in base) or not has_frequency:
            continue
        cleaned.append(r)
    return cleaned


def feature_names_for_rows(rows):
    """Select reduced frequency when available and circularize design phase."""
    use_reduced = bool(rows) and all(
        _parse_float(row.get("reduced_frequency")) is not None for row in rows
    )
    frequency_key = "reduced_frequency" if use_reduced else "frequency_hz"
    phases = {_parse_float(row.get("phase")) for row in rows}
    phases.discard(None)
    if len(phases) > 1:
        return ("q_offset", "epsilon", frequency_key, "phase_sin", "phase_cos")
    return ("q_offset", "epsilon", frequency_key, "phase")


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
            "reduced_frequency": float(row.get(
                "reduced_frequency", row["frequency_hz"],
            )),
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


def _complex_target_spec(key):
    """Return (stem, component) for a derived complex target, else None."""
    for component in ("real", "imag"):
        suffix = f"_response_{component}"
        if key.endswith(suffix):
            return key[:-len(suffix)], component
    return None


def _value_for_target(row, key):
    """Extract a direct or derived target value from one summary row."""
    spec = _complex_target_spec(key)
    if spec is None:
        return _parse_float(row.get(key))
    stem, component = spec
    amplitude = _parse_float(row.get(f"{stem}_amplitude"))
    lag = _parse_float(row.get(f"{stem}_phase_lag_rad"))
    if amplitude is None or lag is None:
        return None
    if component == "real":
        return float(amplitude * np.cos(lag))
    # H = A exp(-i lag): positive lag has a negative imaginary component.
    return float(-amplitude * np.sin(lag))


def collect_target_vector(rows, key):
    """Return (values_finite, mask) for one target metric."""
    parsed = [_value_for_target(r, key) for r in rows]
    complex_spec = _complex_target_spec(key)
    support_key = (
        _support_key_for_target(f"{complex_spec[0]}_amplitude")
        if complex_spec is not None else _support_key_for_target(key)
    )
    supported = ([True] * len(rows) if support_key is None else
                 [_parse_bool(row.get(support_key)) for row in rows])
    positive_required = key.endswith("_amplitude")
    mask = np.array([
        v is not None and ok and (not positive_required or v > 0.0)
        for v, ok in zip(parsed, supported)
    ])
    values = np.array([v if v is not None else np.nan for v in parsed], dtype=float)
    return values, mask


def target_transform(key):
    """Amplitude scalars use log10 conditioning; other targets are identity."""
    return "log10" if key.endswith("_amplitude") else "identity"


def transform_target(values, transform):
    values = np.asarray(values, dtype=float)
    if transform == "log10":
        if np.any(values <= 0.0):
            raise ValueError("log-amplitude targets must be positive")
        return np.log10(values)
    return values.copy()


def inverse_target(values, transform):
    values = np.asarray(values, dtype=float)
    if transform == "log10":
        return np.power(10.0, values)
    return values.copy()


def wrap_phase(values):
    """Wrap scalar/array angles to [-pi, pi)."""
    return (np.asarray(values, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def zero_forcing_response_value(target, value, epsilon, atol=1.0e-15):
    """Apply the exact zero-input boundary to periodic response outputs."""
    is_periodic = target.endswith("_amplitude") or _complex_target_spec(target) is not None
    if is_periodic and abs(float(epsilon)) <= float(atol):
        return 0.0
    return float(value)


def feature_relevance(X, y, feature_names):
    """Exploratory per-feature absolute Spearman association.

    This is deliberately reported as association, not causal sensitivity.  It
    remains meaningful for the fixed-hyperparameter demo GP, whose nominal ARD
    lengthscales are otherwise all identical.
    """
    y_rank = rankdata(np.asarray(y, dtype=float))
    raw = {}
    for index, name in enumerate(feature_names):
        x = np.asarray(X[:, index], dtype=float)
        if np.ptp(x) <= 1.0e-14 or np.ptp(y_rank) <= 1.0e-14:
            rho = 0.0
        else:
            rho = float(np.corrcoef(rankdata(x), y_rank)[0, 1])
            if not np.isfinite(rho):
                rho = 0.0
        raw[name] = rho
    total = sum(abs(value) for value in raw.values())
    return {
        name: {
            "spearman_r": value,
            "absolute": abs(value),
            "normalized_absolute": (abs(value) / total if total > 0.0 else 0.0),
        }
        for name, value in raw.items()
    }


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


def plot_circular_errors(predicted, actual, stem, path):
    """Plot wrapped LOO phase errors for one complex response."""
    error = wrap_phase(np.asarray(predicted) - np.asarray(actual))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(actual, error, s=42, color="#7a3db8", alpha=0.85)
    ax.axhline(0.0, color="k", ls="--", lw=1.0)
    ax.set_xlabel("actual positive phase lag [rad]")
    ax.set_ylabel("wrapped predicted - actual [rad]")
    ax.set_title(f"Circular LOO error: {stem}")
    ax.set_ylim(-np.pi, np.pi)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _surface_query_rows(valid_rows, resolution):
    """Dense in-domain (epsilon, frequency) grid at observed q/phase levels."""
    q_levels = sorted({_parse_float(row.get("q_offset")) for row in valid_rows})
    phase_levels = sorted({_parse_float(row.get("phase")) for row in valid_rows})
    eps_values = np.asarray([
        _parse_float(row.get("epsilon")) for row in valid_rows
    ], dtype=float)
    frequency_values = np.asarray([
        _parse_float(row.get("frequency_hz")) for row in valid_rows
    ], dtype=float)
    eps_grid = np.linspace(float(eps_values.min()), float(eps_values.max()), resolution)
    frequency_grid = np.linspace(
        float(frequency_values.min()), float(frequency_values.max()), resolution,
    )

    ratios = []
    for row in valid_rows:
        frequency = _parse_float(row.get("frequency_hz"))
        reduced = _parse_float(row.get("reduced_frequency"))
        if frequency is not None and reduced is not None and abs(frequency) > 0.0:
            ratios.append(reduced / frequency)
    reduced_per_hz = float(np.median(ratios)) if ratios else None

    rows = []
    for q_offset in q_levels:
        for phase in phase_levels:
            for epsilon in eps_grid:
                for frequency in frequency_grid:
                    row = {
                        "q_offset": q_offset,
                        "epsilon": float(epsilon),
                        "frequency_hz": float(frequency),
                        "phase": phase,
                    }
                    if reduced_per_hz is not None:
                        row["reduced_frequency"] = float(reduced_per_hz * frequency)
                    rows.append(row)
    return rows, q_levels, phase_levels, eps_grid, frequency_grid


def generate_response_surface(valid_rows, feature_names, trained_models,
                              transforms, x_mins, x_ranges, output_root,
                              plots_dir, resolution=25):
    """Predict an in-domain response grid and render complex-response maps."""
    if not valid_rows or not trained_models:
        return {"status": "skipped", "reason": "no trained models"}
    rows, q_levels, phase_levels, eps_grid, frequency_grid = _surface_query_rows(
        valid_rows, resolution,
    )
    X_query = build_feature_matrix(rows, feature_names=feature_names)
    X_query_norm = (X_query - x_mins) / x_ranges

    for target, model in trained_models.items():
        prediction_model = predict_model(model, X_query_norm)
        prediction = inverse_target(prediction_model, transforms[target])
        for row, value in zip(rows, prediction):
            # A zero sinusoidal input has zero periodic response, while its
            # phase remains undefined.
            value = zero_forcing_response_value(
                target, value, row["epsilon"],
            )
            row[f"predicted_{target}"] = float(value)

    mapped_stems = []
    for stem in COMPLEX_RESPONSE_STEMS:
        real_name, imag_name = complex_target_names(stem)
        real_key = f"predicted_{real_name}"
        imag_key = f"predicted_{imag_name}"
        if not rows or real_key not in rows[0] or imag_key not in rows[0]:
            continue
        mapped_stems.append(stem)
        for row in rows:
            real = row[real_key]
            imag = row[imag_key]
            if abs(row["epsilon"]) <= 1.0e-15:
                row[f"predicted_{stem}_complex_amplitude"] = 0.0
                row[f"predicted_{stem}_phase_lag_rad"] = None
            else:
                row[f"predicted_{stem}_complex_amplitude"] = float(np.hypot(real, imag))
                row[f"predicted_{stem}_phase_lag_rad"] = float(
                    wrap_phase(np.arctan2(-imag, real)),
                )

    write_csv(output_root / "response_surface.csv", rows)

    n_per_map = resolution * resolution
    for stem in mapped_stems:
        for q_index, q_offset in enumerate(q_levels):
            for phase_index, phase in enumerate(phase_levels):
                block_index = q_index * len(phase_levels) + phase_index
                block = rows[block_index * n_per_map:(block_index + 1) * n_per_map]
                amplitude = np.asarray([
                    row[f"predicted_{stem}_complex_amplitude"] for row in block
                ]).reshape(resolution, resolution)
                lag = np.asarray([
                    (np.nan if row[f"predicted_{stem}_phase_lag_rad"] is None
                     else row[f"predicted_{stem}_phase_lag_rad"])
                    for row in block
                ], dtype=float).reshape(resolution, resolution)
                fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
                x_axis = np.asarray([
                    row.get("reduced_frequency", row["frequency_hz"])
                    for row in block[:resolution]
                ])
                Xp, Yp = np.meshgrid(x_axis, eps_grid)
                c0 = axes[0].pcolormesh(Xp, Yp, amplitude, shading="auto", cmap="viridis")
                c1 = axes[1].pcolormesh(
                    Xp, Yp, lag, shading="auto", cmap="twilight",
                    vmin=-np.pi, vmax=np.pi,
                )
                fig.colorbar(c0, ax=axes[0], label="complex-response amplitude")
                fig.colorbar(c1, ax=axes[1], label="positive phase lag [rad]")
                xlabel = ("reduced frequency k" if "reduced_frequency" in block[0]
                          else "frequency [Hz]")
                for ax in axes:
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("epsilon")
                axes[0].set_title(f"{stem}: amplitude")
                axes[1].set_title(f"{stem}: phase lag")
                fig.suptitle(f"q_offset={q_offset:g}, forcing phase={phase:g} rad")
                plt.tight_layout()
                safe_q = f"{q_offset:+.6f}".replace("+", "p").replace("-", "m").replace(".", "p")
                safe_phase = f"{phase:+.4f}".replace("+", "p").replace("-", "m").replace(".", "p")
                fig.savefig(
                    plots_dir / f"response_map_{stem}_q_{safe_q}_phase_{safe_phase}.png",
                    dpi=140,
                )
                plt.close(fig)
    return {
        "status": "ok",
        "resolution_per_axis": int(resolution),
        "n_rows": len(rows),
        "mapped_complex_responses": mapped_stems,
        "domain_only": True,
    }


def build_surrogate(doe_root, output_root=None, targets=None,
                    holdout_fraction=0.2, seed=42, surface_resolution=25):
    """Train surrogate per target and emit reports + plots."""
    doe_root = Path(doe_root)
    if not doe_root.is_absolute():
        doe_root = REPO_ROOT / doe_root
    require_schema_v2(doe_root, "unsteady DOE")
    doe_manifest_path = doe_root / "manifest.json"
    doe_manifest = (
        json.loads(doe_manifest_path.read_text())
        if doe_manifest_path.is_file() else {}
    )

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
        targets = default_targets()

    summary_rows, design_rows = load_doe_summary(doe_root)
    valid_rows = _drop_failed(summary_rows)
    feature_names = feature_names_for_rows(valid_rows)
    rng = np.random.default_rng(seed)

    train_data_rows = []
    for r in valid_rows:
        record = {k: _parse_float(r.get(k)) for k in RAW_FEATURE_KEYS}
        record["case_id"] = r.get("case_id", "")
        for t in targets:
            record[t] = _value_for_target(r, t)
        train_data_rows.append(record)
    write_csv(output_root / "surrogate_training_data.csv", train_data_rows,
              fieldnames=["case_id", *RAW_FEATURE_KEYS, *targets])

    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "doe_root": str(doe_root),
        "geometry_lineage_id": doe_manifest.get("geometry_lineage_id"),
        "reduced_frequency": doe_manifest.get("reduced_frequency"),
        "n_summary_rows": len(summary_rows),
        "n_valid_rows": len(valid_rows),
        "raw_design_features": list(RAW_FEATURE_KEYS),
        "features": list(feature_names),
        "frequency_coordinate": (
            "reduced_frequency" if "reduced_frequency" in feature_names
            else "frequency_hz_legacy_fallback"
        ),
        "circular_phase_encoding": "phase_sin" in feature_names,
        "targets": list(targets),
        "target_conditioning": {
            "amplitudes": "log10",
            "other_scalars": "identity",
        },
        "complex_response": {
            "definition": "H = amplitude*exp(-i*positive_phase_lag)",
            "stems": list(COMPLEX_RESPONSE_STEMS),
            "unsupported_lags_trained": False,
            "reportability_policy": {
                "name": "circular_loo_support_gate_v1",
                "circular_loo_rmse_rad_max": 1.0,
                "supported_cases_minimum_rule": "max(20, 4*(varying_features+1))",
                "status_ok_meaning": "model constructed successfully",
            },
        },
        "feature_relevance": {
            "method": "absolute Spearman association on valid target rows",
            "interpretation": "exploratory association, not causal sensitivity",
        },
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
            "Sparse-demo response maps are visualization aids unless LOO diagnostics are acceptable.",
        ],
    }

    if len(valid_rows) < MIN_FIT_SAMPLES:
        message = (
            f"only {len(valid_rows)} valid DOE rows available; need at least "
            f"{MIN_FIT_SAMPLES}. Surrogate training skipped."
        )
        complex_responses = {}
        for stem in COMPLEX_RESPONSE_STEMS:
            real_name, imag_name = complex_target_names(stem)
            supported_indices = [
                index for index, row in enumerate(valid_rows)
                if (_value_for_target(row, real_name) is not None
                    and _value_for_target(row, imag_name) is not None)
            ]
            complex_responses[stem] = {
                "status": "insufficient_supported_data",
                "n_samples": len(supported_indices),
                "circular_loo_rmse_rad": None,
                **response_reportability(
                    None, len(supported_indices),
                    _varying_physical_feature_count(
                        valid_rows, supported_indices,
                    ),
                ),
            }
        write_json(output_root / "model_metadata.json", metadata)
        write_json(output_root / "surrogate_validation_summary.json", {
            "status": "insufficient_data",
            "message": message,
            "metadata": metadata,
            "complex_responses": complex_responses,
        })
        print(message)
        return output_root, {"status": "insufficient_data"}

    X = build_feature_matrix(valid_rows, feature_names=feature_names)
    X_norm, x_mins, x_ranges = normalize_features(X)
    metadata["feature_normalization"] = {
        "min": x_mins.tolist(), "range": x_ranges.tolist(),
    }

    validation_summary = {
        "status": "ok", "metadata": metadata, "targets": {},
        "complex_responses": {},
    }
    trained_models = {}
    transforms = {}
    loo_cache = {}
    loo_rows = [
        {
            "case_id": row.get("case_id", ""),
            **{key: _parse_float(row.get(key)) for key in RAW_FEATURE_KEYS},
        }
        for row in valid_rows
    ]

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
            "transform": target_transform(target),
            "feature_relevance": None,
        }

        if n_valid_target < MIN_FIT_SAMPLES:
            result["warning"] = (
                f"only {n_valid_target} valid cases for target; need "
                f"{MIN_FIT_SAMPLES}. Skipping training."
            )
            validation_summary["targets"][target] = result
            continue

        X_tgt = X_norm[mask]
        y_physical = y[mask]
        transform = target_transform(target)
        y_tgt = transform_target(y_physical, transform)
        model, model_name = fit_model(X_tgt, y_tgt)
        result["model_selected"] = model_name
        result["feature_relevance"] = feature_relevance(
            X_tgt, y_tgt, feature_names,
        )
        trained_models[target] = model
        transforms[target] = transform

        if n_valid_target >= MIN_LOO_SAMPLES:
            try:
                preds_model, actuals_model = leave_one_out(
                    X_tgt, y_tgt, full_model=model, full_kind=model_name,
                )
                preds = inverse_target(preds_model, transform)
                actuals = inverse_target(actuals_model, transform)
                err = preds - actuals
                rmse = float(np.sqrt(np.mean(err**2)))
                mae = float(np.mean(np.abs(err)))
                rel_rmse = float(rmse / max(np.std(actuals), 1.0e-12))
                result["loo"] = {
                    "rmse": rmse,
                    "mae": mae,
                    "rmse_over_std": rel_rmse,
                    "n_samples": int(n_valid_target),
                    "model_space_rmse": float(np.sqrt(np.mean(
                        (preds_model - actuals_model)**2,
                    ))),
                    "model_space": transform,
                }
                row_indices = np.flatnonzero(mask)
                loo_cache[target] = {
                    "row_indices": row_indices,
                    "predicted": preds,
                    "actual": actuals,
                }
                for row_index, predicted, actual in zip(row_indices, preds, actuals):
                    loo_rows[int(row_index)][f"predicted_{target}"] = float(predicted)
                    loo_rows[int(row_index)][f"actual_{target}"] = float(actual)
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
                pred_h_model = predict_model(model_h, X_tgt[test_idx])
                pred_h = inverse_target(pred_h_model, transform)
                actual_h = inverse_target(y_tgt[test_idx], transform)
                err_h = pred_h - actual_h
                rmse_h = float(np.sqrt(np.mean(err_h**2)))
                result["holdout"] = {
                    "n_train": int(n_train),
                    "n_test": int(len(test_idx)),
                    "rmse": rmse_h,
                }
        except Exception as exc:
            message = f"holdout failed: {type(exc).__name__}: {exc}"
            result["warning"] = "; ".join(
                part for part in (result.get("warning", ""), message) if part
            )

        validation_summary["targets"][target] = result

    for stem in COMPLEX_RESPONSE_STEMS:
        real_name, imag_name = complex_target_names(stem)
        if real_name not in loo_cache or imag_name not in loo_cache:
            supported_indices = [
                index for index, row in enumerate(valid_rows)
                if (_value_for_target(row, real_name) is not None
                    and _value_for_target(row, imag_name) is not None)
            ]
            validation_summary["complex_responses"][stem] = {
                "status": "insufficient_supported_data",
                "n_samples": len(supported_indices),
                "circular_loo_rmse_rad": None,
                **response_reportability(
                    None,
                    len(supported_indices),
                    _varying_physical_feature_count(
                        valid_rows, supported_indices,
                    ),
                ),
            }
            continue
        real_cache = loo_cache[real_name]
        imag_cache = loo_cache[imag_name]
        real_by_row = {
            int(index): (float(predicted), float(actual))
            for index, predicted, actual in zip(
                real_cache["row_indices"], real_cache["predicted"], real_cache["actual"],
            )
        }
        imag_by_row = {
            int(index): (float(predicted), float(actual))
            for index, predicted, actual in zip(
                imag_cache["row_indices"], imag_cache["predicted"], imag_cache["actual"],
            )
        }
        common = sorted(set(real_by_row) & set(imag_by_row))
        pred_real = np.asarray([real_by_row[index][0] for index in common])
        pred_imag = np.asarray([imag_by_row[index][0] for index in common])
        actual_real = np.asarray([real_by_row[index][1] for index in common])
        actual_imag = np.asarray([imag_by_row[index][1] for index in common])
        predicted_amplitude = np.hypot(pred_real, pred_imag)
        actual_amplitude = np.hypot(actual_real, actual_imag)
        predicted_lag = wrap_phase(np.arctan2(-pred_imag, pred_real))
        actual_lag = wrap_phase(np.arctan2(-actual_imag, actual_real))
        circular_error = wrap_phase(predicted_lag - actual_lag)
        amplitude_error = predicted_amplitude - actual_amplitude
        circular_rmse = float(np.sqrt(np.mean(circular_error**2)))
        response_result = {
            "status": "ok",
            "n_samples": len(common),
            "circular_mae_rad": float(np.mean(np.abs(circular_error))),
            "circular_rmse_rad": circular_rmse,
            "circular_loo_rmse_rad": circular_rmse,
            "amplitude_rmse": float(np.sqrt(np.mean(amplitude_error**2))),
            "amplitude_rmse_over_std": float(
                np.sqrt(np.mean(amplitude_error**2))
                / max(float(np.std(actual_amplitude)), 1.0e-12)
            ),
            "phase_convention": "positive lag; H=A*exp(-i*lag)",
            **response_reportability(
                circular_rmse,
                len(common),
                _varying_physical_feature_count(valid_rows, common),
            ),
        }
        validation_summary["complex_responses"][stem] = response_result
        for offset, row_index in enumerate(common):
            loo_rows[row_index].update({
                f"predicted_{stem}_complex_amplitude": float(predicted_amplitude[offset]),
                f"actual_{stem}_complex_amplitude": float(actual_amplitude[offset]),
                f"predicted_{stem}_phase_lag_rad": float(predicted_lag[offset]),
                f"actual_{stem}_phase_lag_rad": float(actual_lag[offset]),
                f"{stem}_circular_error_rad": float(circular_error[offset]),
            })
        plot_circular_errors(
            predicted_lag, actual_lag, stem,
            plots_dir / f"circular_loo_{stem}.png",
        )

    loo_fieldnames = sorted({key for row in loo_rows for key in row})
    write_csv(
        output_root / "loo_predictions.csv", loo_rows,
        fieldnames=loo_fieldnames,
    )
    surface_summary = generate_response_surface(
        valid_rows, feature_names, trained_models, transforms,
        x_mins, x_ranges, output_root, plots_dir,
        resolution=max(int(surface_resolution), 2),
    )
    metadata["response_surface"] = surface_summary

    write_json(output_root / "manifest.json", {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": metadata["created_utc"],
        "study": "unsteady_response_surrogate",
        "doe_root": str(doe_root),
        "geometry_lineage_id": metadata.get("geometry_lineage_id"),
        "reduced_frequency": metadata.get("reduced_frequency"),
        "features": list(feature_names),
        "complex_response_stems": list(COMPLEX_RESPONSE_STEMS),
    })
    write_json(output_root / "model_metadata.json", metadata)
    write_json(output_root / "surrogate_validation_summary.json", validation_summary)

    plotted_any = any(
        validation_summary["targets"][t].get("loo") is not None for t in targets
    )
    note = (
        "Trained scalar and complex-response surrogate. This is NOT a "
        "time-accurate reduced-order model."
    )
    print(note)
    print(f"Valid DOE rows used: {len(valid_rows)}/{len(summary_rows)}")
    print(f"Predicted-vs-actual plots: {'generated' if plotted_any else 'skipped (too few samples)'}")
    print(f"Surrogate output: {output_root}")
    return output_root, validation_summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train a scalar/complex-response surrogate on a parametric "
            "unsteady DOE. Predicts post-transient metrics, not time histories."
        ),
    )
    parser.add_argument("--doe-root", required=True,
                        help="Path to a parametric unsteady DOE output directory.")
    parser.add_argument("--output-root", default=None,
                        help="Path for surrogate outputs (default: timestamped runs/).")
    parser.add_argument("--targets", default=None,
                        help=("Comma-separated target metric names. Defaults to "
                              "scalar means/log-amplitudes plus complex responses."))
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--surface-resolution", type=int, default=25,
                        help="Points per epsilon/frequency axis in emitted maps.")
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
        surface_resolution=args.surface_resolution,
    )


if __name__ == "__main__":
    main()
