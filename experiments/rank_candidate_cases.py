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

from experiments.run_static_wall_sweep import write_csv, write_json


DEFAULT_WEIGHTS = {
    "pressure_recovery_mean": 1.0,
    "mdot_stability": 1.0,
    "exit_mach_amplitude_low_is_good": 0.5,
    "exit_mach_amplitude_high_is_good": 0.0,
    "numerical_stability": 1.0,
    "warning_penalty": 0.5,
}

DESIGN_KEYS = ("q_offset", "epsilon", "frequency_hz", "phase")


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


def score_doe_cases(rows, weights, low_amplitude_preferred=True,
                    require_finite_phase=False):
    """Score each DOE row. Returns a list of (row, score, breakdown)."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    if not low_amplitude_preferred:
        weights["exit_mach_amplitude_low_is_good"] = 0.0
        weights["exit_mach_amplitude_high_is_good"] = max(
            weights.get("exit_mach_amplitude_high_is_good", 1.0), 1.0,
        )

    eligible = [r for r in rows if not is_failed(r)]
    p_rec = [_parse_float(r.get("pressure_recovery_mean")) for r in eligible]
    mdot_amp = [_parse_float(r.get("mdot_amplitude")) for r in eligible]
    em_amp = [_parse_float(r.get("exit_mach_amplitude")) for r in eligible]

    n_rec_norm = normalized(p_rec, higher_is_better=True)
    n_mdot_stab = normalized(mdot_amp, higher_is_better=False)
    n_em_low = normalized(em_amp, higher_is_better=False)
    n_em_high = normalized(em_amp, higher_is_better=True)

    breakdowns = []
    scored = []
    for k, r in enumerate(eligible):
        b = {
            "pressure_recovery_mean": float(n_rec_norm[k]),
            "mdot_stability": float(n_mdot_stab[k]),
            "exit_mach_amplitude_low_is_good": float(n_em_low[k]),
            "exit_mach_amplitude_high_is_good": float(n_em_high[k]),
            "numerical_stability": 1.0,  # already non-failed
            "warning_penalty": 0.0 if has_warnings(r) else 1.0,
        }
        if require_finite_phase:
            phase_lag = _parse_float(r.get("exit_mach_phase_lag_rad"))
            if phase_lag is None:
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
        if any(_parse_float(r.get(k)) is None for k in ("q", "pressure_recovery", "thrust")):
            continue
        eligible.append(r)
    p_rec = [_parse_float(r.get("pressure_recovery")) for r in eligible]
    thrust = [_parse_float(r.get("thrust")) for r in eligible]
    n_rec_norm = normalized(p_rec, higher_is_better=True)
    n_th_norm = normalized(thrust, higher_is_better=True)
    scored = []
    for k, r in enumerate(eligible):
        b = {
            "pressure_recovery_mean": float(n_rec_norm[k]),
            "thrust": float(n_th_norm[k]),
            "numerical_stability": 1.0,
            "warning_penalty": 1.0,
        }
        score = (weights.get("pressure_recovery_mean", 1.0) * b["pressure_recovery_mean"]
                 + 0.5 * b["thrust"]
                 + weights.get("numerical_stability", 1.0) * b["numerical_stability"])
        scored.append((r, float(score), b))
    return scored


def to_csv_row(row, score, breakdown, source):
    """Flatten a (row, score, breakdown) tuple for ranked_cases.csv."""
    out = {"source": source, "score": score}
    for key in ("case_id", "q_offset", "epsilon", "frequency_hz", "phase",
                "q", "exit_mach_mean", "exit_mach_amplitude",
                "pressure_recovery_mean", "pressure_recovery",
                "mdot_amplitude", "exit_mach_phase_lag_rad", "warnings",
                "status"):
        if key in row:
            out[key] = row.get(key)
    for k, v in breakdown.items():
        out[f"score_{k}"] = v
    return out


def write_selection_report(path, doe_root, static_root, weights,
                            top_k, selected_doe, selected_static):
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
    lines.append("")
    lines.append("## Scoring weights\n")
    for k, v in weights.items():
        lines.append(f"- `{k}` = {v}")
    lines.append("")
    lines.append("## Top DOE candidates\n")
    if selected_doe:
        lines.append("| Rank | case_id | q_offset | epsilon | f [Hz] | phase | score |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, (row, score, _) in enumerate(selected_doe[:top_k], start=1):
            lines.append(
                f"| {i} | {row.get('case_id', '')} | {row.get('q_offset', '')} | "
                f"{row.get('epsilon', '')} | {row.get('frequency_hz', '')} | "
                f"{row.get('phase', '')} | {score:.4f} |"
            )
    else:
        lines.append("(no DOE candidates were eligible after filtering)")
    lines.append("")
    lines.append("## Top static-sweep candidates\n")
    if selected_static:
        lines.append("| Rank | q | thrust | pressure_recovery | score |")
        lines.append("|---|---|---|---|---|")
        for i, (row, score, _) in enumerate(selected_static[:top_k], start=1):
            lines.append(
                f"| {i} | {row.get('q', '')} | {row.get('thrust', '')} | "
                f"{row.get('pressure_recovery', '')} | {score:.4f} |"
            )
    else:
        lines.append("(no static sweep summary provided or no eligible cases)")
    lines.append("")
    lines.append("## Limitations\n")
    lines.append("- Low-fidelity effective-area forcing only — not body-fitted CFD.")
    lines.append("- Pressure recovery is a Python QoI proxy, not a stagnation pressure ")
    lines.append("  ratio from a validated solver.")
    lines.append("- Phase lag is reported only when post-transient sample count and ")
    lines.append("  cycle count are sufficient.")
    lines.append("- Rankings are intended to seed high-fidelity case selection, not ")
    lines.append("  to certify physical performance.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def rank_cases(doe_root, output_root=None, static_root=None, top_k=5,
               weights=None, low_amplitude_preferred=True,
               require_finite_phase=False):
    """Rank DOE + optional static-sweep candidates; write outputs."""
    doe_root = Path(doe_root)
    if not doe_root.is_absolute():
        doe_root = REPO_ROOT / doe_root

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

    scored_doe, effective_weights = score_doe_cases(
        doe_rows, weights,
        low_amplitude_preferred=low_amplitude_preferred,
        require_finite_phase=require_finite_phase,
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

    selected_doe = scored_doe[:top_k]
    selected_static = scored_static[:top_k]

    selected_cases = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "low_fidelity_selected_cases_for_high_fidelity_followup": True,
        "validated_physical_optima": False,
        "doe_root": str(doe_root),
        "static_sweep_root": str(static_root) if static_root else None,
        "top_k": int(top_k),
        "weights": effective_weights,
        "doe_candidates": [
            {
                "case_id": row.get("case_id", ""),
                "rank": i + 1,
                "score": float(score),
                "design": {k: _parse_float(row.get(k)) for k in DESIGN_KEYS},
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
                "thrust": _parse_float(row.get("thrust")),
                "pressure_recovery": _parse_float(row.get("pressure_recovery")),
                "exit_mach": _parse_float(row.get("exit_mach")),
                "case_relpath": str(Path("cases") / row.get("q", "")),
            }
            for i, (row, score, _) in enumerate(selected_static)
        ],
        "notes": [
            "Rankings derived from low-fidelity effective-area forcing data.",
            "Failed cases and warning-flagged cases are filtered or down-weighted.",
            "These selections seed a later high-fidelity CFD workflow.",
        ],
    }
    write_json(output_root / "selected_cases.json", selected_cases)
    write_selection_report(
        output_root / "selection_report.md", doe_root, static_root,
        effective_weights, top_k, selected_doe, selected_static,
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
    parser.add_argument("--prefer-high-amplitude", action="store_true",
                        help="Reverse the default 'low response amplitude is good' bias.")
    parser.add_argument("--require-finite-phase", action="store_true",
                        help="Exclude cases without a finite exit-Mach phase lag.")
    args = parser.parse_args(argv)

    rank_cases(
        doe_root=args.doe_root,
        output_root=args.output_root,
        static_root=args.static_sweep_root,
        top_k=args.top_k,
        weights=parse_weights(args.weights),
        low_amplitude_preferred=not args.prefer_high_amplitude,
        require_finite_phase=args.require_finite_phase,
    )


if __name__ == "__main__":
    main()
