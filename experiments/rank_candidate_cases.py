"""
Rank DOE (and optionally static-sweep) cases for high-fidelity follow-up.

This produces a ranked list of low-fidelity cases that look promising for
later OpenFOAM/FUN3D screening. The score is a transparent weighted sum of
normalized metrics. Cases marked failed, or carrying response-metric
warnings, are excluded.

These rankings are NOT validated physical optima. They are scaffolding
choices about which cases to run in a later high-fidelity workflow.
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

import numpy as np

from experiments.run_static_wall_sweep import (
    ARTIFACT_SCHEMA_VERSION,
    require_schema_v2,
    write_csv,
    write_json,
)


DEFAULT_WEIGHTS = {
    "tpr_mean": 1.5,
    "mass_conservation": 0.5,
    "response_informativeness": 1.0,
    "numerical_stability": 1.0,
    "warning_penalty": 0.1,
}

DESIGN_KEYS = (
    "q_offset", "epsilon", "frequency_hz", "reduced_frequency", "phase",
)


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
    """Read DOE summary.csv and design_matrix.csv if present."""
    summary_path = Path(doe_root) / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing DOE summary.csv: {summary_path}")
    with summary_path.open() as f:
        rows = list(csv.DictReader(f))
    return rows


def load_static_summary(sweep_root):
    """Read static sweep summary.csv if it exists, else return []."""
    if not sweep_root:
        return []
    path = Path(sweep_root) / "summary.csv"
    if not path.is_file():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def load_surrogate_audit(surrogate_root):
    """Load optional LOO predictions and circular validation diagnostics."""
    if not surrogate_root:
        return {}, None
    root = Path(surrogate_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    prediction_path = root / "loo_predictions.csv"
    validation_path = root / "surrogate_validation_summary.json"
    metadata_path = root / "model_metadata.json"
    if not prediction_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(
            f"surrogate audit requires {prediction_path.name} and "
            f"{validation_path.name} under {root}"
        )
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"response surrogate at {root} is schema_version="
            f"{metadata.get('schema_version')!r}; version "
            f"{ARTIFACT_SCHEMA_VERSION} is required"
        )
    with prediction_path.open() as handle:
        predictions = {
            row.get("case_id", ""): row for row in csv.DictReader(handle)
            if row.get("case_id")
        }
    validation = json.loads(validation_path.read_text())
    return predictions, validation


def normalized(values, higher_is_better):
    """Min-max normalize; higher_is_better picks polarity. Empty -> zeros."""
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    mask = np.isfinite(arr)
    out = np.zeros_like(arr)
    if mask.sum() == 0:
        return out
    lo, hi = float(arr[mask].min()), float(arr[mask].max())
    rng = hi - lo
    if rng <= 0.0:
        out[mask] = 0.5
        return out
    if higher_is_better:
        out[mask] = (arr[mask] - lo) / rng
    else:
        out[mask] = (hi - arr[mask]) / rng
    return out


def is_failed(row):
    """Return True if the row indicates a failed case."""
    status = (row.get("status") or "").strip().lower()
    return bool(status) and status != "ok"


def has_warnings(row):
    """Detect warning text in DOE summary rows."""
    warnings = row.get("warnings") or ""
    return bool(str(warnings).strip())


def score_doe_cases(rows, weights, require_finite_phase=True,
                    include_zero_eps=False, surrogate_predictions=None,
                    require_predicted_phase=False):
    """Score each DOE row. Returns a list of (row, score, breakdown)."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    eligible = []
    for row in rows:
        if is_failed(row):
            continue
        epsilon = _parse_float(row.get("epsilon"))
        zero_epsilon = epsilon is not None and abs(epsilon) <= 1.0e-15
        if not include_zero_eps and (epsilon is None or zero_epsilon):
            continue
        if (not zero_epsilon
                and str(row.get("exit_mach_supported", "")).lower()
                not in {"1", "true", "yes"}):
            continue
        if any(_parse_float(row.get(key)) is None for key in (
            "tpr_mean", "mass_defect_mean",
        )):
            continue
        amplitude_key = ("exit_mach_raw_amplitude" if zero_epsilon
                         else "exit_mach_amplitude")
        if _parse_float(row.get(amplitude_key)) is None:
            continue
        row = dict(row)
        row["_ranking_exit_mach_amplitude"] = row.get(amplitude_key)
        prediction = (surrogate_predictions or {}).get(row.get("case_id", ""))
        if require_predicted_phase:
            predicted_lag = _parse_float(
                (prediction or {}).get("predicted_exit_mach_phase_lag_rad"),
            )
            if predicted_lag is None:
                continue
        if prediction is not None:
            row["_surrogate_prediction"] = prediction
        eligible.append(row)
    tpr = [_parse_float(r.get("tpr_mean")) for r in eligible]
    mass_defect = [abs(_parse_float(r.get("mass_defect_mean")))
                   for r in eligible]
    em_amp = [_parse_float(r.get("_ranking_exit_mach_amplitude")) for r in eligible]

    n_tpr = normalized(tpr, higher_is_better=True)
    n_mass = normalized(mass_defect, higher_is_better=False)
    n_em_high = normalized(em_amp, higher_is_better=True)

    breakdowns = []
    scored = []
    for k, r in enumerate(eligible):
        b = {
            "tpr_mean": float(n_tpr[k]),
            "mass_conservation": float(n_mass[k]),
            "response_informativeness": float(n_em_high[k]),
            "numerical_stability": 1.0,  # already non-failed
            "warning_penalty": 0.0 if has_warnings(r) else 1.0,
        }
        if require_finite_phase:
            phase_lag = _parse_float(r.get("exit_mach_phase_lag_rad"))
            epsilon = _parse_float(r.get("epsilon"))
            if phase_lag is None and not (
                    include_zero_eps and epsilon is not None
                    and abs(epsilon) <= 1.0e-15):
                continue
        score = sum(weights[k] * b[k] for k in b)
        scored.append((r, float(score), b))
        breakdowns.append(b)

    return scored, weights


def score_static_cases(rows, weights):
    """Score static sweep rows using a simpler set of QoI."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    eligible = []
    for r in rows:
        if any(_parse_float(r.get(k)) is None for k in ("q", "tpr")):
            continue
        if str(r.get("converged", "")).lower() not in {"1", "true", "yes"}:
            continue
        eligible.append(r)
    tpr = [_parse_float(r.get("tpr")) for r in eligible]
    n_tpr = normalized(tpr, higher_is_better=True)
    scored = []
    for k, r in enumerate(eligible):
        b = {
            "tpr_mean": float(n_tpr[k]),
            "numerical_stability": 1.0,
            "warning_penalty": 1.0,
        }
        score = (weights.get("tpr_mean", 1.0) * b["tpr_mean"]
                 + weights.get("numerical_stability", 1.0) * b["numerical_stability"])
        scored.append((r, float(score), b))
    return scored


def to_csv_row(row, score, breakdown, source):
    """Flatten a (row, score, breakdown) tuple for ranked_cases.csv."""
    out = {"source": source, "score": score}
    for key in ("case_id", "q_offset", "epsilon", "frequency_hz", "phase",
                "reduced_frequency",
                "q", "exit_mach_mean", "exit_mach_amplitude",
                "tpr_mean", "tpr", "mass_defect_mean",
                "exit_mach_phase_lag_rad", "warnings",
                "status"):
        if key in row:
            out[key] = row.get(key)
    prediction = row.get("_surrogate_prediction") or {}
    for key in (
        "predicted_exit_mach_complex_amplitude",
        "predicted_exit_mach_phase_lag_rad",
        "exit_mach_circular_error_rad",
        "predicted_tpr_complex_amplitude",
        "predicted_tpr_phase_lag_rad",
        "tpr_circular_error_rad",
    ):
        if key in prediction:
            out[f"surrogate_loo_{key}"] = prediction.get(key)
    for k, v in breakdown.items():
        out[f"score_{k}"] = v
    return out


def write_selection_report(path, doe_root, static_root, weights,
                            top_k, selected_doe, selected_static,
                            surrogate_validation=None,
                            require_predicted_phase=False):
    """Write a human-readable selection_report.md."""
    lines = []
    lines.append("# Candidate ranking selection report\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
    lines.append("These rankings are low-fidelity selected cases for high-fidelity ")
    lines.append("OpenFOAM/FUN3D follow-up. They are NOT validated physical optima ")
    lines.append("and they were produced from a Python effective-area-forcing prototype.\n")
    lines.append("## Inputs\n")
    lines.append(f"- DOE root: `{doe_root}`")
    lines.append(f"- Static sweep root: `{static_root}`")
    lines.append(
        "- Surrogate-predicted phase gate: "
        f"`{bool(require_predicted_phase)}` (measured supported phase remains authoritative)"
    )
    lines.append("")
    lines.append("## Scoring weights\n")
    for k, v in weights.items():
        lines.append(f"- `{k}` = {v}")
    lines.append("")
    lines.append("## Top DOE candidates\n")
    if selected_doe:
        lines.append("| Rank | case_id | q_offset | epsilon | f [Hz] | k | measured lag [rad] | score |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, (row, score, _) in enumerate(selected_doe[:top_k], start=1):
            lines.append(
                f"| {i} | {row.get('case_id', '')} | {row.get('q_offset', '')} | "
                f"{row.get('epsilon', '')} | {row.get('frequency_hz', '')} | "
                f"{row.get('reduced_frequency', '')} | "
                f"{row.get('exit_mach_phase_lag_rad', '')} | {score:.4f} |"
            )
    else:
        lines.append("(no DOE candidates were eligible after filtering)")
    lines.append("")
    lines.append("## Top static-sweep candidates\n")
    if selected_static:
        lines.append("| Rank | q | TPR | score |")
        lines.append("|---|---|---|---|")
        for i, (row, score, _) in enumerate(selected_static[:top_k], start=1):
            lines.append(
                f"| {i} | {row.get('q', '')} | {row.get('tpr', '')} | "
                f"{score:.4f} |"
            )
    else:
        lines.append("(no static sweep summary provided or no eligible cases)")
    lines.append("")
    lines.append("## Response-surrogate audit\n")
    if surrogate_validation:
        complex_block = surrogate_validation.get("complex_responses", {})
        lines.append("Surrogate values below are leave-one-out diagnostics only; ")
        lines.append("they do not replace measured DOE metrics in the score.\n")
        lines.append("| response | supported LOO samples | circular MAE [rad] | circular RMSE [rad] |")
        lines.append("|---|---:|---:|---:|")
        for stem, result in complex_block.items():
            if result.get("status") != "ok":
                continue
            lines.append(
                f"| {stem} | {result.get('n_samples', '')} | "
                f"{result.get('circular_mae_rad', float('nan')):.4f} | "
                f"{result.get('circular_rmse_rad', float('nan')):.4f} |"
            )
    else:
        lines.append("(no response-surrogate audit supplied)")
    lines.append("")
    lines.append("## Limitations\n")
    lines.append("- Low-fidelity effective-area forcing only — not body-fitted CFD.")
    lines.append("- TPR uses the shared mass-flux-weighted definition but remains a ")
    lines.append("  low-fidelity prediction until an OpenFOAM case closes the loop.")
    lines.append("- Phase lag is reported only when post-transient sample count and ")
    lines.append("  cycle count are sufficient.")
    lines.append("- Rankings are intended to seed high-fidelity case selection, not ")
    lines.append("  to certify physical performance.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _greedy_diverse_selection(scored, top_k):
    """Choose high-scoring cases while spreading (q_offset, epsilon, f)."""
    if not scored or top_k <= 0:
        return []
    pool = scored[:max(top_k * 4, top_k)]
    use_reduced = all(
        _parse_float(row.get("reduced_frequency")) is not None
        for row, _, _ in pool
    )
    frequency_key = "reduced_frequency" if use_reduced else "frequency_hz"
    design = np.asarray([
        [float(row["q_offset"]), float(row["epsilon"]), float(row[frequency_key])]
        for row, _, _ in pool
    ])
    lo, hi = design.min(axis=0), design.max(axis=0)
    design = (design - lo) / np.maximum(hi - lo, 1.0e-30)
    scores = np.asarray([item[1] for item in pool], dtype=float)
    score_norm = ((scores - scores.min()) /
                  max(float(scores.max() - scores.min()), 1.0e-30))
    chosen = [int(np.argmax(scores))]
    while len(chosen) < min(top_k, len(pool)):
        candidates = [i for i in range(len(pool)) if i not in chosen]
        utilities = []
        for i in candidates:
            separation = min(
                float(np.linalg.norm(design[i] - design[j])) for j in chosen
            ) / np.sqrt(3.0)
            utilities.append(0.7 * score_norm[i] + 0.3 * separation)
        chosen.append(candidates[int(np.argmax(utilities))])
    return [pool[i] for i in chosen]


def rank_cases(doe_root, output_root=None, static_root=None, top_k=5,
               weights=None, require_finite_phase=True,
               include_zero_eps=False, surrogate_root=None,
               require_predicted_phase=False):
    """Rank DOE + optional static-sweep candidates; write outputs."""
    doe_root = Path(doe_root)
    if not doe_root.is_absolute():
        doe_root = REPO_ROOT / doe_root
    require_schema_v2(doe_root, "unsteady DOE")
    if static_root:
        static_root = Path(static_root)
        if not static_root.is_absolute():
            static_root = REPO_ROOT / static_root
        require_schema_v2(static_root, "static sweep")

    if output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "runs" / f"ranked_candidate_cases_{stamp}"
    else:
        output_root = Path(output_root)
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    doe_rows = load_doe_summary(doe_root)
    static_rows = load_static_summary(static_root) if static_root else []
    surrogate_predictions, surrogate_validation = load_surrogate_audit(
        surrogate_root,
    )
    if require_predicted_phase and not surrogate_predictions:
        raise ValueError(
            "require_predicted_phase=True requires --surrogate-root with LOO predictions"
        )

    scored_doe, effective_weights = score_doe_cases(
        doe_rows, weights,
        require_finite_phase=require_finite_phase,
        include_zero_eps=include_zero_eps,
        surrogate_predictions=surrogate_predictions,
        require_predicted_phase=require_predicted_phase,
    )
    scored_doe.sort(key=lambda triple: triple[1], reverse=True)
    scored_static = score_static_cases(static_rows, weights)
    scored_static.sort(key=lambda triple: triple[1], reverse=True)

    csv_rows = []
    for row, score, breakdown in scored_doe:
        csv_rows.append(to_csv_row(row, score, breakdown, "doe"))
    for row, score, breakdown in scored_static:
        csv_rows.append(to_csv_row(row, score, breakdown, "static_sweep"))
    fieldnames = sorted({k for r in csv_rows for k in r.keys()}) if csv_rows else []
    write_csv(output_root / "ranked_cases.csv", csv_rows, fieldnames=fieldnames)

    selected_doe = _greedy_diverse_selection(scored_doe, top_k)
    selected_static = scored_static[:top_k]

    selected_cases = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "low_fidelity_selected_cases_for_high_fidelity_followup": True,
        "validated_physical_optima": False,
        "doe_root": str(doe_root),
        "static_sweep_root": str(static_root) if static_root else None,
        "top_k": int(top_k),
        "weights": effective_weights,
        "surrogate_audit": {
            "root": str(surrogate_root) if surrogate_root else None,
            "used_for_scoring": False,
            "require_finite_predicted_phase": bool(require_predicted_phase),
            "measured_phase_authoritative": True,
            "complex_response_validation": (
                (surrogate_validation or {}).get("complex_responses")
            ),
        },
        "doe_candidates": [
            {
                "case_id": row.get("case_id", ""),
                "rank": i + 1,
                "score": float(score),
                "design": {k: _parse_float(row.get(k)) for k in DESIGN_KEYS},
                "measured_response": {
                    "exit_mach_amplitude": _parse_float(row.get(
                        "_ranking_exit_mach_amplitude",
                    )),
                    "exit_mach_phase_lag_rad": _parse_float(row.get(
                        "exit_mach_phase_lag_rad",
                    )),
                    "tpr_mean": _parse_float(row.get("tpr_mean")),
                },
                "surrogate_loo_audit": {
                    key: _parse_float(value)
                    for key, value in (row.get("_surrogate_prediction") or {}).items()
                    if key in {
                        "predicted_exit_mach_complex_amplitude",
                        "predicted_exit_mach_phase_lag_rad",
                        "exit_mach_circular_error_rad",
                        "predicted_tpr_complex_amplitude",
                        "predicted_tpr_phase_lag_rad",
                        "tpr_circular_error_rad",
                    }
                },
                "case_relpath": str(Path("cases") / (row.get("case_id") or "")),
                "score_breakdown": breakdown,
                "warnings": row.get("warnings", ""),
            }
            for i, (row, score, breakdown) in enumerate(selected_doe)
        ],
        "static_sweep_candidates": [
            {
                "rank": i + 1,
                "score": float(score),
                "q": _parse_float(row.get("q")),
                "tpr": _parse_float(row.get("tpr")),
                "exit_mach": _parse_float(row.get("exit_mach")),
                "case_relpath": str(Path("cases") / row.get("q", "")),
            }
            for i, (row, score, _) in enumerate(selected_static)
        ],
        "notes": [
            "Rankings derived from low-fidelity effective-area forcing data.",
            "Failed cases and warning-flagged cases are filtered or down-weighted.",
            "epsilon=0 baselines are excluded unless explicitly requested.",
            "DOE top-k uses a score/spread greedy selection in (q_offset, epsilon, frequency).",
            "Reduced frequency replaces dimensional frequency in spread when available.",
            "Measured supported lag remains authoritative; surrogate LOO lag is audit-only unless its optional finite-value gate is requested.",
            "These selections seed a later high-fidelity CFD workflow.",
        ],
    }
    write_json(output_root / "selected_cases.json", selected_cases)
    write_selection_report(
        output_root / "selection_report.md", doe_root, static_root,
        effective_weights, top_k, selected_doe, selected_static,
        surrogate_validation=surrogate_validation,
        require_predicted_phase=require_predicted_phase,
    )

    print(f"Ranked {len(scored_doe)} DOE candidates and {len(scored_static)} static cases.")
    print(f"Top-{top_k} selections written to: {output_root}")
    return output_root, selected_cases


def parse_weights(text):
    """Parse comma-separated key=value pairs into a dict."""
    if not text:
        return None
    out = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, value = part.partition("=")
        out[key.strip()] = float(value)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Rank low-fidelity DOE (and optional static sweep) cases for "
            "later high-fidelity CFD follow-up. Outputs ranked_cases.csv, "
            "selected_cases.json, and selection_report.md."
        ),
    )
    parser.add_argument("--doe-root", required=True,
                        help="Path to a parametric unsteady DOE output directory.")
    parser.add_argument("--static-sweep-root", default=None,
                        help="Optional path to a static wall sweep output directory.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--weights", default=None,
                        help="Comma-separated key=value pairs to override scoring weights.")
    parser.add_argument("--allow-missing-phase", action="store_true",
                        help="Allow cases without a finite exit-Mach phase lag.")
    parser.add_argument("--include-zero-eps", action="store_true",
                        help="Allow epsilon=0 labeled baselines in unsteady candidates.")
    parser.add_argument(
        "--surrogate-root", default=None,
        help="Optional response-surrogate output carrying circular LOO diagnostics.",
    )
    parser.add_argument(
        "--require-finite-predicted-phase", action="store_true",
        help=("Additionally require a finite LOO-predicted exit-Mach lag. "
              "Measured supported lag remains the scoring source."),
    )
    args = parser.parse_args(argv)

    rank_cases(
        doe_root=args.doe_root,
        output_root=args.output_root,
        static_root=args.static_sweep_root,
        top_k=args.top_k,
        weights=parse_weights(args.weights),
        require_finite_phase=not args.allow_missing_phase,
        include_zero_eps=args.include_zero_eps,
        surrogate_root=args.surrogate_root,
        require_predicted_phase=args.require_finite_predicted_phase,
    )


if __name__ == "__main__":
    main()
