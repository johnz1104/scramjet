"""
Parametric unsteady design-of-experiments runner.

Sweeps combinations of (q_offset, epsilon, frequency_hz, phase) for the
reduced-fidelity effective-area forcing model

    q_total(t) = q_offset + epsilon * sin(2*pi*frequency_hz*t + phase)
    A(x, t)    = A_base(x) + q_total(t) * phi(x),

where phi(x) is the same Gaussian throat localization used by the static
and unsteady prototypes. This is an effective area-source DOE, not a
moving-wall or deforming-mesh CFD study.

Each successful case writes:
    config.json, forcing_history.csv, qoi_history.csv,
    probe_history.csv, response_metrics.json, diagnostics.json,
    residual.csv, timestep_history.csv, plots/

Aggregate outputs:
    design_matrix.csv, summary.csv, plots/

Failed cases are recorded as rows in summary.csv with status="failed"
and an error_message, so the DOE does not crash on a single bad case.
"""
import argparse
import itertools
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mesh import (
    LocalizedAreaPerturbation,
    SinusoidalAreaForcing,
    TimeDependentPerturbedGeometryProfile,
)
from solver import Solver
from diagnostics import all_case_diagnostics
from experiments.run_static_wall_sweep import (
    config_to_dict,
    git_metadata,
    residual_rows,
    write_csv,
    write_json,
)
from experiments.run_unsteady_area_breathing import (
    make_cold_flow_config,
    probe_locations,
    qoi_row,
    forcing_row,
    probe_row,
    timestep_rows,
    plot_forcing,
    plot_qoi,
    plot_probe_pressure,
)
from response_metrics import extract_response_metrics


def default_q_offsets():
    """Static offsets in m^2 — scaled to a 0.05 m^2 baseline throat area."""
    return [-0.025, 0.0, 0.025]


def default_epsilons():
    """Unsteady amplitudes in m^2."""
    return [0.0, 0.01, 0.02]


def default_frequencies_hz():
    """
    Two small frequencies appropriate for the prototype time scale.

    Default unsteady run uses ~1 kHz with t_final = 0.5 ms. To capture at
    least one full cycle per case at the DOE mesh and step budget, we
    select 500 Hz and 1500 Hz: a low-frequency sweep that resolves the
    breathing cleanly, and a higher one that exercises faster forcing.
    """
    return [500.0, 1500.0]


def default_phases():
    """Phase offsets in radians; one value keeps the default DOE small."""
    return [0.0]


def case_label(idx):
    """Filesystem-safe case folder name."""
    return f"case_{idx:03d}"


def design_matrix(q_offsets, epsilons, frequencies_hz, phases):
    """Cartesian product of the four DOE axes."""
    rows = []
    for idx, (q_offset, eps, freq, phi) in enumerate(
        itertools.product(q_offsets, epsilons, frequencies_hz, phases)
    ):
        rows.append({
            "case_id": case_label(idx),
            "q_offset": float(q_offset),
            "epsilon": float(eps),
            "frequency_hz": float(freq),
            "phase": float(phi),
        })
    return rows


def build_time_dependent_geometry(base_geometry, q_offset, epsilon,
                                   frequency_hz, phase, width=None,
                                   x_center=None, min_area=1.0e-4):
    """Construct A(x, t) = A_base(x) + (q_offset + epsilon*sin(...)) * phi(x)."""
    if width is None:
        width = 0.05 * base_geometry.L_total
    if x_center is None:
        x_center = base_geometry.x_throat

    perturbation = LocalizedAreaPerturbation(
        enabled=True,
        mode="throat_gaussian",
        amplitude=0.0,
        x_center=x_center,
        width=width,
        min_area=min_area,
    )
    forcing = SinusoidalAreaForcing(
        amplitude=float(epsilon),
        frequency_hz=float(frequency_hz),
        phase=float(phase),
        enabled=True,
        mean=float(q_offset),
    )
    return TimeDependentPerturbedGeometryProfile(base_geometry, perturbation, forcing)


def select_t_final(frequency_hz, cycles, t_final_static):
    """Pick t_final so cases capture enough physical time."""
    if frequency_hz <= 0.0:
        return float(t_final_static)
    return float(cycles) / float(frequency_hz)


def run_one_case(case_dir, design_row, baseline_state_U, baseline_summary,
                 nx, ny, cfl, mach, altitude, width, x_center, min_area,
                 cycles, t_final_static, unsteady_steps, sample_interval_steps,
                 discard_fraction, preset=None):
    """Run a single DOE case. Returns a summary row (status PASS/FAIL)."""
    plots_dir = case_dir / "plots"
    case_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_row = {
        "case_id": design_row["case_id"],
        "q_offset": design_row["q_offset"],
        "epsilon": design_row["epsilon"],
        "frequency_hz": design_row["frequency_hz"],
        "phase": design_row["phase"],
        "status": "failed",
        "error_message": "",
    }

    try:
        t_final = select_t_final(design_row["frequency_hz"], cycles, t_final_static)

        cfg = make_cold_flow_config(
            nx=nx, ny=ny, n_steps=unsteady_steps, cfl=cfl,
            mach=mach, altitude=altitude, preset=preset,
        )
        cfg.t_final = t_final
        cfg.geometry = build_time_dependent_geometry(
            base_geometry=cfg.geometry,
            q_offset=design_row["q_offset"],
            epsilon=design_row["epsilon"],
            frequency_hz=design_row["frequency_hz"],
            phase=design_row["phase"],
            width=width, x_center=x_center, min_area=min_area,
        )

        solver = Solver(cfg)
        if baseline_state_U is not None:
            solver.state.U = baseline_state_U.copy()

        probes = probe_locations(cfg.geometry)

        forcing_rows = []
        qoi_rows = []
        probe_rows = []

        def sample(current_solver):
            forcing_rows.append(forcing_row(current_solver))
            qoi_rows.append(qoi_row(current_solver))
            probe_rows.append(probe_row(current_solver, probes))

        sample(solver)

        def callback(current_solver):
            if current_solver.step_count % sample_interval_steps == 0:
                sample(current_solver)

        solver.run(n_steps=unsteady_steps, t_final=t_final, step_callback=callback)
        if forcing_rows[-1]["time"] != solver.time:
            sample(solver)

        if np.any(np.isnan(solver.state.U)):
            raise RuntimeError("NaN appeared in conservative state")

        write_json(case_dir / "config.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "study": "parametric_unsteady_doe",
            "model": "time_dependent_effective_area_forcing_with_offset",
            "not_true_moving_wall_cfd": True,
            "design": design_row,
            "baseline": baseline_summary,
            "config": config_to_dict(cfg),
            "probes": probes,
            "cycles_requested": float(cycles),
            "t_final": float(t_final),
        })
        write_csv(case_dir / "forcing_history.csv", forcing_rows)
        write_csv(case_dir / "qoi_history.csv", qoi_rows)
        write_csv(case_dir / "probe_history.csv", probe_rows)
        write_csv(case_dir / "residual.csv", residual_rows(solver),
                  fieldnames=["sample", "residual_l2"])
        write_csv(case_dir / "timestep_history.csv", timestep_rows(solver))
        write_json(case_dir / "diagnostics.json", all_case_diagnostics(solver))

        metrics = extract_response_metrics(
            qoi_rows=qoi_rows, forcing_rows=forcing_rows, probe_rows=probe_rows,
            frequency_hz=float(design_row["frequency_hz"]),
            discard_fraction=float(discard_fraction),
        )
        write_json(case_dir / "response_metrics.json", metrics)

        plot_forcing(forcing_rows, plots_dir / "forcing_history.png")
        plot_qoi(qoi_rows, plots_dir / "qoi_history.png")
        plot_probe_pressure(probe_rows, plots_dir / "probe_pressure_history.png")

        summary_row.update({
            "status": "ok",
            "samples": len(qoi_rows),
            "final_time": float(solver.time),
            "steps": int(solver.step_count),
            "min_throat_area": float(cfg.geometry.min_area_value(time=solver.time)),
        })
        _merge_metrics_into_summary(summary_row, metrics)
        return summary_row
    except Exception as exc:
        summary_row["error_message"] = f"{type(exc).__name__}: {exc}"
        summary_row["traceback"] = traceback.format_exc(limit=2)
        return summary_row


def _merge_metrics_into_summary(summary_row, metrics):
    """Flatten response metric dict into single-row form for summary.csv."""
    summary_row["transient_discard_fraction"] = metrics.get("transient_discard_fraction")
    summary_row["n_samples_after_transient"] = metrics.get("n_samples_after_transient")
    summary_row["n_cycles_after_transient"] = metrics.get("n_cycles_after_transient")
    summary_row["forcing_mean"] = metrics.get("forcing", {}).get("mean")
    summary_row["forcing_amplitude"] = metrics.get("forcing", {}).get("amplitude")
    qoi_block = metrics.get("qoi", {})
    for key, sub in qoi_block.items():
        summary_row[f"{key}_mean"] = sub.get("mean")
        summary_row[f"{key}_amplitude"] = sub.get("amplitude")
        summary_row[f"{key}_phase_lag_rad"] = sub.get("phase_lag_vs_q_rad")
    probe_block = metrics.get("probes", {})
    for probe_name, sub in probe_block.items():
        summary_row[f"probe_{probe_name}_pressure_amplitude"] = sub.get("pressure_amplitude")
        summary_row[f"probe_{probe_name}_pressure_mean"] = sub.get("pressure_mean")
    warnings = metrics.get("warnings", [])
    summary_row["warnings"] = "; ".join(warnings) if warnings else ""


def plot_response_amplitude_vs_epsilon(summary_rows, path):
    """Aggregate plot: response amplitude (exit Mach) vs epsilon, colored by frequency."""
    valid = [r for r in summary_rows if r.get("status") == "ok"
             and r.get("exit_mach_amplitude") is not None]
    if not valid:
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    freqs = sorted({r["frequency_hz"] for r in valid})
    cmap = plt.get_cmap("viridis")
    for k, f in enumerate(freqs):
        eps = [r["epsilon"] for r in valid if r["frequency_hz"] == f]
        amp = [r["exit_mach_amplitude"] for r in valid if r["frequency_hz"] == f]
        order = np.argsort(eps)
        ax.plot(np.array(eps)[order], np.array(amp)[order], "o-",
                color=cmap(k / max(len(freqs) - 1, 1)),
                label=f"f = {f:g} Hz")
    ax.set_xlabel("epsilon [m^2]")
    ax.set_ylabel("exit Mach amplitude")
    ax.set_title("Unsteady response amplitude vs forcing amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def plot_mean_qoi_vs_q_offset(summary_rows, path):
    """Aggregate plot: mean exit Mach and mean pressure recovery vs q_offset."""
    valid = [r for r in summary_rows if r.get("status") == "ok"
             and r.get("exit_mach_mean") is not None]
    if not valid:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter([r["q_offset"] for r in valid],
                    [r["exit_mach_mean"] for r in valid],
                    c=[r["epsilon"] for r in valid], cmap="plasma", s=40)
    axes[0].set_xlabel("q_offset [m^2]")
    axes[0].set_ylabel("mean exit Mach")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter([r["q_offset"] for r in valid],
                    [r["pressure_recovery_mean"] for r in valid],
                    c=[r["epsilon"] for r in valid], cmap="plasma", s=40)
    axes[1].set_xlabel("q_offset [m^2]")
    axes[1].set_ylabel("mean pressure recovery")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Mean QoI vs static q_offset (color = epsilon)")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def plot_frequency_response(summary_rows, path):
    """Aggregate plot: response amplitude vs frequency at fixed epsilon > 0."""
    valid = [r for r in summary_rows if r.get("status") == "ok"
             and r.get("exit_mach_amplitude") is not None
             and r.get("epsilon", 0.0) > 0.0]
    if not valid:
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    eps_levels = sorted({r["epsilon"] for r in valid})
    cmap = plt.get_cmap("magma")
    for k, eps in enumerate(eps_levels):
        rows = [r for r in valid if r["epsilon"] == eps]
        if not rows:
            continue
        freqs = [r["frequency_hz"] for r in rows]
        amps = [r["exit_mach_amplitude"] for r in rows]
        order = np.argsort(freqs)
        ax.plot(np.array(freqs)[order], np.array(amps)[order], "o-",
                color=cmap(0.3 + 0.6 * k / max(len(eps_levels) - 1, 1)),
                label=f"epsilon = {eps:g}")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("exit Mach amplitude")
    ax.set_title("Frequency response of effective-area forcing")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def warm_start_baseline(nx, ny, baseline_steps, cfl, mach, altitude, preset=None):
    """Run a short cold-flow baseline; supply its state as warm start."""
    cfg = make_cold_flow_config(nx=nx, ny=ny, n_steps=baseline_steps, cfl=cfl,
                                mach=mach, altitude=altitude, preset=preset)
    solver = Solver(cfg)
    solver.run()
    return solver


def run_doe(output_root=None, q_offsets=None, epsilons=None,
            frequencies_hz=None, phases=None, nx=30, ny=6,
            unsteady_steps=300, baseline_steps=80, cfl=0.35,
            mach=6.0, altitude=25000.0, width=None, x_center=None,
            min_area=1.0e-4, cycles=1.5, t_final_static=2.0e-4,
            sample_interval_steps=2, discard_fraction=0.25, preset=None):
    """Run the DOE and aggregate outputs."""
    q_offsets = list(default_q_offsets() if q_offsets is None else q_offsets)
    epsilons = list(default_epsilons() if epsilons is None else epsilons)
    frequencies_hz = list(default_frequencies_hz() if frequencies_hz is None else frequencies_hz)
    phases = list(default_phases() if phases is None else phases)

    if output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "runs" / f"parametric_unsteady_doe_{stamp}"
    else:
        output_root = Path(output_root)
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root

    cases_root = output_root / "cases"
    plots_dir = output_root / "plots"
    cases_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    design_rows = design_matrix(q_offsets, epsilons, frequencies_hz, phases)
    write_csv(output_root / "design_matrix.csv", design_rows)

    write_json(output_root / "manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study": "parametric_unsteady_doe",
        "model": "time_dependent_effective_area_forcing_with_offset",
        "not_true_moving_wall_cfd": True,
        "axes": {
            "q_offset": q_offsets,
            "epsilon": epsilons,
            "frequency_hz": frequencies_hz,
            "phase": phases,
        },
        "n_cases": len(design_rows),
        "config_defaults": {
            "nx": nx, "ny": ny, "cfl": cfl, "mach": mach, "altitude": altitude,
            "unsteady_steps": unsteady_steps, "baseline_steps": baseline_steps,
            "cycles": cycles, "t_final_static": t_final_static, "min_area": min_area,
            "discard_fraction": discard_fraction,
            "sample_interval_steps": sample_interval_steps,
        },
        "git": git_metadata(),
    })

    baseline_solver = warm_start_baseline(
        nx=nx, ny=ny, baseline_steps=baseline_steps, cfl=cfl,
        mach=mach, altitude=altitude, preset=preset,
    )
    baseline_summary = {
        "steps": int(baseline_solver.step_count),
        "final_time": float(baseline_solver.time),
    }
    baseline_state_U = baseline_solver.state.U

    summary_rows = []
    for idx, design_row in enumerate(design_rows):
        case_dir = cases_root / design_row["case_id"]
        print(f"[{idx + 1}/{len(design_rows)}] {design_row['case_id']}: "
              f"q_offset={design_row['q_offset']:+.4f} "
              f"epsilon={design_row['epsilon']:+.4f} "
              f"f={design_row['frequency_hz']:.0f} Hz "
              f"phase={design_row['phase']:.3f}")
        row = run_one_case(
            case_dir, design_row,
            baseline_state_U=baseline_state_U,
            baseline_summary=baseline_summary,
            nx=nx, ny=ny, cfl=cfl, mach=mach, altitude=altitude,
            width=width, x_center=x_center, min_area=min_area,
            cycles=cycles, t_final_static=t_final_static,
            unsteady_steps=unsteady_steps,
            sample_interval_steps=sample_interval_steps,
            discard_fraction=discard_fraction,
            preset=preset,
        )
        summary_rows.append(row)
        if row["status"] != "ok":
            print(f"    FAILED: {row.get('error_message')}")

    fieldnames = sorted({k for row in summary_rows for k in row.keys()})
    write_csv(output_root / "summary.csv", summary_rows, fieldnames=fieldnames)

    plot_response_amplitude_vs_epsilon(
        summary_rows, plots_dir / "response_amplitude_vs_epsilon.png",
    )
    plot_mean_qoi_vs_q_offset(
        summary_rows, plots_dir / "mean_qoi_vs_q_offset.png",
    )
    plot_frequency_response(
        summary_rows, plots_dir / "frequency_response.png",
    )

    n_ok = sum(1 for r in summary_rows if r["status"] == "ok")
    print(f"\nDOE complete: {n_ok}/{len(summary_rows)} cases succeeded.")
    print(f"Output: {output_root}")
    return output_root, summary_rows


def parse_float_list(text):
    """Parse comma-separated float list."""
    return [float(part) for part in text.split(",") if part.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run a small parametric DOE over (q_offset, epsilon, "
            "frequency_hz, phase) for the reduced-fidelity effective-area "
            "forcing model. Outputs per-case configs, histories, response "
            "metrics, and aggregate plots."
        ),
    )
    parser.add_argument("--output-root", default=None,
                        help="Output directory (default: timestamped runs/ subdir).")
    parser.add_argument("--q-offsets", default=None,
                        help="Comma-separated q_offset values [m^2].")
    parser.add_argument("--epsilons", default=None,
                        help="Comma-separated forcing amplitudes [m^2].")
    parser.add_argument("--frequencies-hz", default=None,
                        help="Comma-separated forcing frequencies [Hz].")
    parser.add_argument("--phases", default=None,
                        help="Comma-separated forcing phases [rad].")
    parser.add_argument("--nx", type=int, default=30)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--unsteady-steps", type=int, default=300)
    parser.add_argument("--baseline-steps", type=int, default=80)
    parser.add_argument("--cfl", type=float, default=0.35)
    parser.add_argument("--mach", type=float, default=6.0)
    parser.add_argument("--altitude", type=float, default=25000.0)
    parser.add_argument("--width", type=float, default=None)
    parser.add_argument("--x-center", type=float, default=None)
    parser.add_argument("--min-area", type=float, default=1.0e-4)
    parser.add_argument("--cycles", type=float, default=1.5,
                        help="Number of forcing cycles per case for f > 0.")
    parser.add_argument("--t-final-static", type=float, default=2.0e-4,
                        help="t_final used when frequency_hz == 0.")
    parser.add_argument("--sample-interval-steps", type=int, default=2)
    parser.add_argument("--discard-fraction", type=float, default=0.25,
                        help="Fraction of early samples treated as transient.")
    parser.add_argument("--preset", default=None,
                        help="Experiment-condition preset (e.g. configs/tusq_m585.json); "
                             "overrides --mach/--altitude.")
    args = parser.parse_args(argv)

    run_doe(
        output_root=args.output_root,
        q_offsets=parse_float_list(args.q_offsets) if args.q_offsets else None,
        epsilons=parse_float_list(args.epsilons) if args.epsilons else None,
        frequencies_hz=parse_float_list(args.frequencies_hz) if args.frequencies_hz else None,
        phases=parse_float_list(args.phases) if args.phases else None,
        nx=args.nx, ny=args.ny,
        unsteady_steps=args.unsteady_steps,
        baseline_steps=args.baseline_steps,
        cfl=args.cfl, mach=args.mach, altitude=args.altitude,
        width=args.width, x_center=args.x_center, min_area=args.min_area,
        cycles=args.cycles, t_final_static=args.t_final_static,
        sample_interval_steps=args.sample_interval_steps,
        discard_fraction=args.discard_fraction,
        preset=args.preset,
    )


if __name__ == "__main__":
    main()
