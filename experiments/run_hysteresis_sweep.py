"""Deterministic up/down back-pressure staircase with state carry-over.

This is a cheap H5 diagnostic for numerical path dependence in the quasi-1D
model.  It is not a validation of physical inlet hysteresis: the inlet remains
a prescribed supersonic Dirichlet boundary and cannot represent spillage or
variable mass capture.  Each pressure level starts from the final state of the
previous level; no state is reinitialized between the up and down legs.

Outputs:
    manifest.json, summary.csv, stage_run_status.json,
    hysteresis_assessment.json, plots/hysteresis_paths.png
"""
import argparse
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

from diagnostics import shock_diagnostics
from experiments.run_forced_shock_benchmark import (
    analytic_map,
    initialize_shock,
    make_duct,
)
from experiments.run_static_wall_sweep import (
    ARTIFACT_SCHEMA_VERSION,
    git_metadata,
    write_csv,
    write_json,
)
from rom import _compute_qoi
from solver import CombustionConfig, InletConfig, Solver, SolverConfig


def default_pressure_factors(A_in=0.05, A_ex=0.10, L=1.0, M1=2.0,
                             p1=20000.0, base_shock_x=0.5):
    """Monotone staircase spanning downstream capture to inlet expulsion."""
    p_base = analytic_map(base_shock_x, A_in, A_ex, L, M1, p1)[0]
    target_positions = (0.85, 0.70, 0.55, 0.40, 0.25, 0.12, 0.03)
    factors = [
        analytic_map(x, A_in, A_ex, L, M1, p1)[0] / p_base
        for x in target_positions
    ]
    # Above the equilibrium pressure for a shock at x=0.03: attempts to
    # expel the shock through the prescribed inlet boundary.
    factors.append(1.50)
    return sorted(float(value) for value in factors)


def build_staircase(pressure_factors):
    """Return deterministic up/down legs without repeating the peak."""
    up = sorted({float(value) for value in pressure_factors})
    if len(up) < 2 or up[0] <= 0.0:
        raise ValueError("need at least two distinct positive pressure factors")
    rows = [
        {"leg": "up", "leg_index": index, "pressure_factor": factor}
        for index, factor in enumerate(up)
    ]
    rows.extend(
        {"leg": "down", "leg_index": index, "pressure_factor": factor}
        for index, factor in enumerate(reversed(up[:-1]))
    )
    return rows


def classify_regime(qoi, shock, exit_mach):
    """Classify captured/start state using observables available to this model."""
    if shock.get("shock_detected", False):
        if qoi.get("shock_at_inlet", False):
            return "unstarted", "shock_at_inlet"
        return "started", "captured_internal_shock"
    if float(exit_mach) < 1.0 or qoi.get("min_centerline_mach_contraction", 2.0) < 1.0:
        return "unstarted", "subsonic_without_resolved_internal_shock"
    return "started", "supersonic_without_internal_shock"


def assess_hysteresis(rows, shock_position_tolerance_m, tpr_tolerance=0.01):
    """Compare matching up/down levels and return a resolution-aware result."""
    by_leg = {"up": {}, "down": {}}
    for row in rows:
        by_leg[row["leg"]][round(float(row["pressure_factor"]), 12)] = row
    comparisons = []
    for factor in sorted(set(by_leg["up"]) & set(by_leg["down"])):
        up = by_leg["up"][factor]
        down = by_leg["down"][factor]
        x_up = up.get("shock_x")
        x_down = down.get("shock_x")
        position_delta = None
        if x_up is not None and x_down is not None:
            position_delta = float(x_down - x_up)
        tpr_delta = None
        if up.get("tpr") is not None and down.get("tpr") is not None:
            tpr_delta = float(down["tpr"] - up["tpr"])
        class_changed = up.get("classification") != down.get("classification")
        shock_difference_exceeds_tolerance = bool(
            position_delta is not None
            and abs(position_delta) > shock_position_tolerance_m
        )
        tpr_difference_exceeds_tolerance = bool(
            tpr_delta is not None and abs(tpr_delta) > tpr_tolerance
        )
        corroborated_path_dependence = bool(
            shock_difference_exceeds_tolerance
            or tpr_difference_exceeds_tolerance
        )
        threshold_flip = bool(class_changed and not corroborated_path_dependence)
        comparisons.append({
            "pressure_factor": factor,
            "up_classification": up.get("classification"),
            "down_classification": down.get("classification"),
            "classification_changed": class_changed,
            "shock_position_delta_down_minus_up_m": position_delta,
            "tpr_delta_down_minus_up": tpr_delta,
            "shock_difference_exceeds_tolerance": shock_difference_exceeds_tolerance,
            "tpr_difference_exceeds_tolerance": tpr_difference_exceeds_tolerance,
            "corroborated_path_dependence": corroborated_path_dependence,
            "threshold_flip_at_resolution": threshold_flip,
            # Backward-readable summary field. A bare threshold flip is no
            # longer promoted to numerical path dependence.
            "path_dependent": corroborated_path_dependence,
        })

    all_complete = bool(rows) and all(row.get("status") == "ok" for row in rows)
    if not comparisons or not all_complete:
        classification = "indeterminate_incomplete"
    elif any(item["corroborated_path_dependence"] for item in comparisons):
        classification = "numerical_path_dependence_detected"
    elif any(item["threshold_flip_at_resolution"] for item in comparisons):
        classification = "threshold_flip_at_resolution"
    else:
        classification = "single_path_within_resolution"
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "classification": classification,
        "physical_hysteresis_validated": False,
        "all_stages_settled": all_complete,
        "shock_position_tolerance_m": float(shock_position_tolerance_m),
        "tpr_tolerance": float(tpr_tolerance),
        "comparisons": comparisons,
        "limitation": (
            "The prescribed-inlet quasi-1D model can diagnose deterministic "
            "path dependence but cannot validate spillage-driven physical "
            "restart/unstart hysteresis."
        ),
    }


def plot_paths(rows, path):
    """Plot shock position and TPR along the two pressure legs."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    styles = {"up": ("o-", "#c0392b"), "down": ("s--", "#2471a3")}
    for leg in ("up", "down"):
        leg_rows = [row for row in rows if row["leg"] == leg]
        factor = [row["pressure_factor"] for row in leg_rows]
        shock_x = [np.nan if row.get("shock_x") is None else row["shock_x"]
                   for row in leg_rows]
        tpr = [np.nan if row.get("tpr") is None else row["tpr"] for row in leg_rows]
        style, color = styles[leg]
        axes[0].plot(factor, shock_x, style, color=color, label=leg)
        axes[1].plot(factor, tpr, style, color=color, label=leg)
    axes[0].set_ylabel("captured shock position [m]")
    axes[1].set_ylabel("total-pressure recovery")
    for ax in axes:
        ax.set_xlabel("back pressure / base back pressure")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_title("Up/down shock path")
    axes[1].set_title("Up/down TPR path")
    fig.suptitle("Quasi-1D numerical hysteresis diagnostic")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def stage_completion_gate(stable_checks, required_checks,
                          elapsed_flowthrough_times,
                          min_flowthrough_times):
    """Return true only when both stability and physical-duration gates pass."""
    return bool(
        int(stable_checks) >= int(required_checks)
        and float(elapsed_flowthrough_times) >= float(min_flowthrough_times)
    )


def advance_stage_until_qoi_settled(solver, max_steps, check_interval,
                                    qoi_rtol=1.0e-3,
                                    required_checks=3,
                                    min_flowthrough_times=3.0):
    """Advance one staircase level until classification/shock/TPR settle.

    A captured stationary shock retains a nonzero raw-RHS floor in this
    shock-capturing scheme, so a research-facing shock staircase must not call
    that floor "unsettled" forever.  Completion instead requires consecutive
    agreement of regime classification, shock position within two cells, and
    relative TPR drift below ``qoi_rtol``.  The dimensionless RHS is still
    recorded for audit.
    """
    if int(max_steps) < 1 or int(check_interval) < 1:
        raise ValueError("max_steps and check_interval must be positive")
    if int(required_checks) < 1 or float(min_flowthrough_times) < 0.0:
        raise ValueError("required_checks must be positive and flow-through time nonnegative")

    history = []
    stable_checks = 0
    settled = False
    dx_tolerance = 2.0 * float(np.mean(solver.mesh.dx))
    start_time = float(solver.time)
    flowthrough_time = (
        float(solver.cfg.geometry.L_total)
        / max(abs(float(solver.cfg.inlet.u_inf)), 1.0e-30)
    )
    for local_step in range(1, int(max_steps) + 1):
        solver.advance_one_step()
        if local_step % int(check_interval) != 0 and local_step != int(max_steps):
            continue
        qoi = _compute_qoi(solver)
        shock = shock_diagnostics(solver)
        exit_mach = float(solver.state.mach()[-2, 0])
        classification, regime = classify_regime(qoi, shock, exit_mach)
        shock_x = (
            float(shock["shock_x"])
            if shock.get("shock_detected") and np.isfinite(shock.get("shock_x", np.nan))
            else None
        )
        residual = float(solver._normalized_residual())
        entry = {
            "local_step": local_step,
            "global_step": int(solver.step_count),
            "time_s": float(solver.time),
            "classification": classification,
            "regime_detail": regime,
            "shock_x": shock_x,
            "tpr": float(qoi["tpr"]),
            "residual_l2": residual,
            "elapsed_flowthrough_times": float(
                (solver.time - start_time) / flowthrough_time
            ),
        }
        history.append(entry)
        window_size = int(required_checks) + 1
        if len(history) >= window_size:
            window = history[-window_size:]
            class_stable = len({item["classification"] for item in window}) == 1
            shock_values = [item["shock_x"] for item in window]
            if all(value is None for value in shock_values):
                shock_stable = True
            elif all(value is not None for value in shock_values):
                shock_stable = float(np.ptp(shock_values)) <= dx_tolerance
            else:
                shock_stable = False
            tpr_values = np.asarray([item["tpr"] for item in window], dtype=float)
            tpr_scale = max(float(np.max(np.abs(tpr_values))), 1.0e-12)
            tpr_stable = float(np.ptp(tpr_values)) / tpr_scale <= qoi_rtol
            stable_checks = int(required_checks) if (
                class_stable and shock_stable and tpr_stable
            ) else 0
        elapsed_flowthrough_times = (solver.time - start_time) / flowthrough_time
        if stage_completion_gate(
            stable_checks, required_checks,
            elapsed_flowthrough_times, min_flowthrough_times,
        ):
            settled = True
            break
    return {
        "settled": settled,
        "completion_criterion": "classification + shock_x(two_cells) + relative_TPR",
        "qoi_rtol": float(qoi_rtol),
        "required_checks": int(required_checks),
        "min_flowthrough_times": float(min_flowthrough_times),
        "flowthrough_time_s": float(flowthrough_time),
        "check_interval_steps": int(check_interval),
        "steps": int(local_step),
        "final_residual": history[-1]["residual_l2"] if history else None,
        "check_history": history,
    }


def run_hysteresis(output_root=None, pressure_factors=None, nx=100,
                   stage_steps=8000, qoi_rtol=1.0e-3,
                   steady_check_interval=100, cfl=0.35,
                   initial_shock_x=0.85, base_shock_x=0.5,
                   A_in=0.05, A_ex=0.10, L=1.0,
                   M1=2.0, p1=20000.0, T1=300.0):
    """Run a state-carrying pressure staircase and write its assessment."""
    if pressure_factors is None:
        pressure_factors = default_pressure_factors(
            A_in, A_ex, L, M1, p1, base_shock_x,
        )
    staircase = build_staircase(pressure_factors)
    p_base = analytic_map(base_shock_x, A_in, A_ex, L, M1, p1)[0]

    if output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "runs" / f"hysteresis_sweep_{stamp}"
    else:
        output_root = Path(output_root)
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    geom = make_duct(A_in, A_ex, L)
    cfg = SolverConfig()
    cfg.inlet = InletConfig(mach=M1, T_inf=T1, p_inf=p1)
    cfg.geometry = geom
    cfg.mesh.nx = int(nx)
    cfg.mesh.ny = 1
    cfg.cfl = float(cfl)
    cfg.print_interval = 0
    cfg.residual_interval = int(steady_check_interval)
    cfg.combustion = CombustionConfig(enabled=False)
    cfg.outlet_type = "back_pressure"
    cfg.outlet_p_back = p_base * staircase[0]["pressure_factor"]
    solver = Solver(cfg)
    initialize_shock(
        solver, geom, initial_shock_x, A_in, A_ex, L, M1, p1, T1,
    )
    shock_tolerance = 2.0 * float(np.mean(solver.mesh.dx))

    write_json(output_root / "manifest.json", {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study": "quasi1d_back_pressure_hysteresis_diagnostic",
        "hypothesis": "H5 numerical restart/unstart path dependence",
        "deterministic": True,
        "random_seed": None,
        "state_carry_over": True,
        "physical_hysteresis_validation": False,
        "classification_policy": {
            "shock_position_tolerance_m": shock_tolerance,
            "tpr_tolerance": 0.01,
            "outcomes": [
                "indeterminate_incomplete",
                "numerical_path_dependence_detected",
                "threshold_flip_at_resolution",
                "single_path_within_resolution",
            ],
        },
        "pressure_factors_up": sorted({float(value) for value in pressure_factors}),
        "base_back_pressure_Pa": p_base,
        "base_shock_x_m": base_shock_x,
        "initial_shock_x_m": initial_shock_x,
        "config": {
            "nx": int(nx), "stage_steps": int(stage_steps), "cfl": float(cfl),
            "stage_completion": "classification + shock_x(two_cells) + relative_TPR",
            "qoi_rtol": float(qoi_rtol),
            "steady_check_interval": int(steady_check_interval),
            "A_in": A_in, "A_ex": A_ex, "L": L,
            "M1": M1, "p1_Pa": p1, "T1_K": T1,
        },
        "git": git_metadata(),
    })

    rows = []
    stage_statuses = []
    for stage_index, level in enumerate(staircase):
        factor = level["pressure_factor"]
        pressure = p_base * factor
        solver.cfg.outlet_p_back = pressure
        solver.bc.outlet_p_back = pressure
        start_step = solver.step_count
        start_time = solver.time
        print(
            f"[hysteresis {stage_index + 1}/{len(staircase)}] "
            f"{level['leg']} factor={factor:.6f} ..."
        )
        raw_status = advance_stage_until_qoi_settled(
            solver, max_steps=int(stage_steps),
            check_interval=int(steady_check_interval),
            qoi_rtol=float(qoi_rtol), required_checks=3,
        )
        qoi = _compute_qoi(solver)
        shock = shock_diagnostics(solver)
        exit_mach = float(solver.state.mach()[-2, 0])
        classification, regime = classify_regime(qoi, shock, exit_mach)
        shock_x = (
            float(shock["shock_x"])
            if shock.get("shock_detected") and np.isfinite(shock.get("shock_x", np.nan))
            else None
        )
        row = {
            "stage_index": stage_index,
            "leg": level["leg"],
            "leg_index": level["leg_index"],
            "pressure_factor": factor,
            "back_pressure_Pa": pressure,
            "status": "ok" if raw_status.get("settled") else "incomplete",
            "settled": bool(raw_status.get("settled")),
            "stage_steps": int(solver.step_count - start_step),
            "stage_duration_s": float(solver.time - start_time),
            "final_residual": raw_status.get("final_residual"),
            "classification": classification,
            "regime_detail": regime,
            "shock_x": shock_x,
            "shock_detected": bool(shock.get("shock_detected")),
            "shock_at_inlet": bool(qoi.get("shock_at_inlet")),
            "exit_mach": exit_mach,
            "tpr": float(qoi["tpr"]),
            "mass_defect": float(qoi["mass_defect"]),
        }
        rows.append(row)
        stage_statuses.append({
            "stage_index": stage_index,
            "leg": level["leg"],
            "pressure_factor": factor,
            "solver": {
                **raw_status,
            },
        })

    write_csv(output_root / "summary.csv", rows)
    write_json(output_root / "stage_run_status.json", {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "stages": stage_statuses,
    })
    assessment = assess_hysteresis(
        rows, shock_position_tolerance_m=shock_tolerance,
    )
    write_json(output_root / "hysteresis_assessment.json", assessment)
    plot_paths(rows, plots_dir / "hysteresis_paths.png")
    print(f"Hysteresis classification: {assessment['classification']}")
    print(f"Output: {output_root}")
    return output_root, rows, assessment


def parse_float_list(text):
    return [float(part) for part in text.split(",") if part.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a deterministic state-carrying back-pressure staircase.",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--pressure-factors", default=None,
                        help="Comma-separated factors relative to the base back pressure.")
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--stage-steps", type=int, default=8000)
    parser.add_argument(
        "--qoi-rtol", type=float, default=1.0e-3,
        help="Relative TPR stability tolerance in the per-level QoI gate.",
    )
    parser.add_argument("--steady-check-interval", type=int, default=100)
    parser.add_argument("--cfl", type=float, default=0.35)
    parser.add_argument("--initial-shock-x", type=float, default=0.85)
    parser.add_argument("--base-shock-x", type=float, default=0.5)
    args = parser.parse_args(argv)
    run_hysteresis(
        output_root=args.output_root,
        pressure_factors=(parse_float_list(args.pressure_factors)
                          if args.pressure_factors else None),
        nx=args.nx,
        stage_steps=args.stage_steps,
        qoi_rtol=args.qoi_rtol,
        steady_check_interval=args.steady_check_interval,
        cfl=args.cfl,
        initial_shock_x=args.initial_shock_x,
        base_shock_x=args.base_shock_x,
    )


if __name__ == "__main__":
    main()
