"""
Reduced-fidelity unsteady effective-area breathing runner.

This script applies a sinusoidal throat-area forcing,

    q(t) = epsilon * sin(2*pi*f*t + phase)
    A(x, t) = A_base(x) + q(t) * phi(x),

inside the existing quasi-1D area-source framework. It is not true
moving-wall CFD: there is no moving mesh, ALE boundary condition, turbulence
model, or combustion model.  It does include the quasi-1D moving-control-
volume source, including wall-pressure work in the energy equation.
Generic ducts use a Gaussian ``phi``; Config A uses the tabulated
model-assumed first cantilever shape.
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

from mesh import (
    CONFIG_A_GEOMETRY_LINEAGE_ID,
    LocalizedAreaPerturbation,
    SinusoidalAreaForcing,
    TimeDependentPerturbedGeometryProfile,
    config_a_cantilever_mode,
    config_a_normalized_to_raw,
    config_a_raw_to_normalized,
    config_a_ramp_area_law,
)
from solver import CombustionConfig, InletConfig, Solver, SolverConfig
from rom import _compute_qoi
from response_metrics import extract_response_metrics
from experiments.run_static_wall_sweep import (
    config_to_dict,
    git_metadata,
    residual_rows,
    write_csv,
    write_json,
)


def make_cold_flow_config(nx=30, ny=6, n_steps=120, cfl=0.35,
                          mach=6.0, altitude=25000.0, preset=None,
                          area_law="auto"):
    """Build a cold-flow config with combustion disabled."""
    cfg = SolverConfig()
    if preset:
        from experiments.presets import inlet_from_preset
        cfg.inlet = inlet_from_preset(preset)
    else:
        cfg.inlet = InletConfig(mach=mach, altitude=altitude, Yf_inlet=0.0)
    cfg.mesh.nx = nx
    cfg.mesh.ny = ny
    cfg.n_steps = n_steps
    cfg.cfl = cfl
    cfg.print_interval = max(n_steps + 1, 1)
    cfg.viscous = False
    cfg.wall_type = "slip"
    cfg.combustion = CombustionConfig(enabled=False)
    cfg.area_source = True
    from experiments.presets import preset_geometry, resolve_area_law
    if resolve_area_law(area_law, preset=preset) == "config_a":
        cfg.geometry = (
            preset_geometry(preset) if preset else config_a_ramp_area_law()
        )
        if cfg.geometry is None:
            cfg.geometry = config_a_ramp_area_law()
    return cfg


def attach_unsteady_area_breathing(cfg, amplitude=0.001, frequency_hz=1000.0,
                                   phase=0.0, width=None, x_center=None,
                                   min_area=1.0e-4, motion_mode="auto"):
    """Attach time-dependent effective-area forcing to a config."""
    base = cfg.geometry
    if width is None:
        width = 0.05 * base.L_total
    if x_center is None:
        x_center = base.x_throat

    if motion_mode not in {"auto", "gaussian", "config_a"}:
        raise ValueError(f"unsupported motion mode: {motion_mode}")
    resolved_motion = motion_mode
    if resolved_motion == "auto":
        resolved_motion = (
            "config_a"
            if getattr(base, "geometry_lineage_id", None) == CONFIG_A_GEOMETRY_LINEAGE_ID
            else "gaussian"
        )
    if resolved_motion == "config_a":
        perturbation = config_a_cantilever_mode(
            base, amplitude=0.0, min_area=min_area,
        )
    else:
        perturbation = LocalizedAreaPerturbation(
            enabled=True,
            mode="throat_gaussian",
            amplitude=0.0,
            x_center=x_center,
            width=width,
            min_area=min_area,
        )
    forcing = SinusoidalAreaForcing(
        amplitude=amplitude,
        frequency_hz=frequency_hz,
        phase=phase,
        enabled=True,
    )
    cfg.geometry = TimeDependentPerturbedGeometryProfile(base, perturbation, forcing)
    return cfg


def probe_locations(geometry):
    """Return named probe locations along the duct."""
    return {
        "inlet_side": 0.5 * geometry.x_throat,
        "throat": geometry.x_throat,
        "combustor": 0.5 * (geometry.x_throat + geometry.x_comb_exit),
        "exit": geometry.x_exit,
    }


def qoi_row(solver):
    """Extract time-dependent QoI row."""
    qoi = _compute_qoi(solver)
    rho, u, v, p, T, Yf = solver.state.primitives()
    M = solver.state.mach()
    return {
        "time": float(solver.time),
        "step": int(solver.step_count),
        "exit_mach": float(qoi["exit_mach"]),
        "max_mach": float(np.max(M)),
        "pressure_recovery": float(qoi["pressure_recovery"]),
        "tpr": float(qoi["tpr"]),
        "shock_x": float(qoi["shock_x"]),
        "mdot_prescribed": float(qoi["mdot_prescribed"]),
        "mdot_exit": float(qoi["mdot_exit"]),
        "mass_defect": float(qoi["mass_defect"]),
        "thrust": float(qoi["thrust"]),
        "pressure_min": float(np.min(p)),
        "pressure_max": float(np.max(p)),
    }


def forcing_row(solver):
    """Extract forcing diagnostics."""
    geom = solver.cfg.geometry
    t = float(solver.time)
    return {
        "time": t,
        "step": int(solver.step_count),
        "q": float(geom.current_amplitude(t)),
        "A_throat": float(geom.throat_area(t)),
        "min_area": float(geom.min_area_value(time=t)),
        "max_area": float(geom.max_area_value(time=t)),
    }


def probe_row(solver, probes):
    """Extract y-averaged probe data at named x locations."""
    rho, u, v, p, T, Yf = solver.state.primitives()
    M = solver.state.mach()
    row = {
        "time": float(solver.time),
        "step": int(solver.step_count),
    }
    for name, x_probe in probes.items():
        i = int(np.argmin(np.abs(solver.mesh.xc - x_probe)))
        prefix = name
        row[f"{prefix}_x"] = float(solver.mesh.xc[i])
        row[f"{prefix}_pressure"] = float(np.mean(p[i, :]))
        row[f"{prefix}_mach"] = float(np.mean(M[i, :]))
        row[f"{prefix}_temperature"] = float(np.mean(T[i, :]))
        row[f"{prefix}_density"] = float(np.mean(rho[i, :]))
        row[f"{prefix}_u"] = float(np.mean(u[i, :]))
    return row


def timestep_rows(solver):
    """Return time-step history rows from the solver dt history."""
    rows = []
    t = 0.0
    for idx, dt in enumerate(solver.dt_history, start=1):
        t += float(dt)
        rows.append({"step": idx, "time": t, "dt": float(dt)})
    return rows


def check_monotonic(rows, label):
    """Raise if row times are not monotonically increasing."""
    times = [float(row["time"]) for row in rows]
    if any(t2 < t1 for t1, t2 in zip(times, times[1:])):
        raise ValueError(f"{label} time history is not monotonically increasing")


def response_metrics(qoi_rows, forcing_rows, frequency_hz,
                     discard_fraction=0.25, probe_rows=None):
    """Robust response metrics (delegates to response_metrics.extract_response_metrics)."""
    return extract_response_metrics(
        qoi_rows=qoi_rows, forcing_rows=forcing_rows, probe_rows=probe_rows,
        frequency_hz=float(frequency_hz),
        discard_fraction=float(discard_fraction),
    )


def plot_forcing(forcing_rows, path):
    """Plot q(t) and throat area."""
    t = np.array([row["time"] for row in forcing_rows], dtype=float)
    q = np.array([row["q"] for row in forcing_rows], dtype=float)
    A = np.array([row["A_throat"] for row in forcing_rows], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, q, "b-", lw=1.4)
    axes[0].set_ylabel("q(t) [m^2]")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, A, "r-", lw=1.4)
    axes[1].set_ylabel("A_throat [m^2]")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_qoi(qoi_rows, path):
    """Plot selected QoI histories."""
    t = np.array([row["time"] for row in qoi_rows], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for ax, key in zip(axes, ["exit_mach", "pressure_recovery", "thrust"]):
        y = np.array([row[key] for row in qoi_rows], dtype=float)
        ax.plot(t, y, lw=1.4)
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_probe_pressure(probe_rows, path):
    """Plot pressure histories at all probes."""
    t = np.array([row["time"] for row in probe_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    for key in probe_rows[0]:
        if key.endswith("_pressure"):
            y = np.array([row[key] for row in probe_rows], dtype=float)
            ax.plot(t, y, lw=1.2, label=key.replace("_pressure", ""))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("pressure [Pa]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def validate_output_histories(forcing_rows, qoi_rows, probe_rows, forcing):
    """Validate monotonic time and sinusoidal forcing values."""
    check_monotonic(forcing_rows, "forcing_history")
    check_monotonic(qoi_rows, "qoi_history")
    check_monotonic(probe_rows, "probe_history")
    for row in forcing_rows:
        expected = forcing.value(row["time"])
        if abs(row["q"] - expected) > 1.0e-12:
            raise ValueError("forcing_history q(t) does not match forcing model")


def run_case(output_root=None, amplitude=None, epsilon_le_over_S=None,
             frequency_hz=1000.0,
             phase=0.0, cycles=0.5, t_final=None, width=None,
             x_center=None, min_area=1.0e-4, nx=30, ny=6,
             baseline_steps=80, unsteady_steps=160, cfl=0.35,
             mach=6.0, altitude=25000.0, sample_interval_steps=2,
             preset=None, area_law="auto", motion_mode="auto"):
    """Run a small unsteady effective-area breathing case."""
    if frequency_hz < 0.0:
        raise ValueError("frequency_hz must be non-negative")
    if t_final is None:
        if frequency_hz > 0.0:
            t_final = cycles / frequency_hz
        else:
            t_final = 2.0e-4

    if output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "runs" / f"unsteady_area_breathing_{stamp}"
    else:
        output_root = Path(output_root)
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_cfg = make_cold_flow_config(
        nx=nx, ny=ny, n_steps=baseline_steps, cfl=cfl,
        mach=mach, altitude=altitude, preset=preset, area_law=area_law,
    )
    base_geometry = baseline_cfg.geometry
    if amplitude is not None and epsilon_le_over_S is not None:
        raise ValueError("raw amplitude and epsilon_le_over_S are mutually exclusive")
    if epsilon_le_over_S is not None:
        if getattr(base_geometry, "geometry_lineage_id", None) != CONFIG_A_GEOMETRY_LINEAGE_ID:
            raise ValueError("epsilon_le_over_S requires Config-A geometry")
        amplitude = config_a_normalized_to_raw(epsilon_le_over_S, base_geometry)
    elif amplitude is None:
        amplitude = 0.001
    normalized_amplitude = (
        config_a_raw_to_normalized(amplitude, base_geometry)
        if getattr(base_geometry, "geometry_lineage_id", None) == CONFIG_A_GEOMETRY_LINEAGE_ID
        else None
    )
    baseline_solver = Solver(baseline_cfg)
    baseline_solver.run()

    cfg = make_cold_flow_config(
        nx=nx, ny=ny, n_steps=unsteady_steps, cfl=cfl,
        mach=mach, altitude=altitude, preset=preset, area_law=area_law,
    )
    cfg.t_final = t_final
    attach_unsteady_area_breathing(
        cfg,
        amplitude=amplitude,
        frequency_hz=frequency_hz,
        phase=phase,
        width=width,
        x_center=x_center,
        min_area=min_area,
        motion_mode=motion_mode,
    )

    solver = Solver(cfg)
    solver.state.U = baseline_solver.state.U.copy()
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

    validate_output_histories(forcing_rows, qoi_rows, probe_rows, cfg.geometry.forcing)

    write_json(output_root / "config.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study": "unsteady_area_breathing",
        "model": "time_dependent_effective_area_forcing",
        "not_true_moving_wall_cfd": True,
        "git": git_metadata(),
        "geometry_lineage_id": getattr(cfg.geometry, "geometry_lineage_id", None),
        "reduced_frequency_reference": {
            "length_ref_m": getattr(
                cfg.geometry, "reduced_frequency_length_ref_m",
                cfg.geometry.L_total,
            ),
            "length_source": (
                "published_deformable_surface"
                if getattr(cfg.geometry, "geometry_lineage_id", None)
                == CONFIG_A_GEOMETRY_LINEAGE_ID else "geometry.L_total"
            ),
        },
        "forcing_coordinates": {
            "epsilon": float(amplitude),
            "epsilon_le_over_S": normalized_amplitude,
            "input_representation": (
                "epsilon_le_over_S" if epsilon_le_over_S is not None else "raw_epsilon"
            ),
        },
        "baseline": {
            "steps": baseline_solver.step_count,
            "final_time": baseline_solver.time,
        },
        "config": config_to_dict(cfg),
        "probes": probes,
    })
    write_csv(output_root / "forcing_history.csv", forcing_rows)
    write_csv(output_root / "qoi_history.csv", qoi_rows)
    write_csv(output_root / "probe_history.csv", probe_rows)
    write_csv(output_root / "residual.csv", residual_rows(solver),
              fieldnames=["sample", "step", "time", "residual_l2"])
    write_csv(output_root / "timestep_history.csv", timestep_rows(solver))
    metrics = response_metrics(qoi_rows, forcing_rows, frequency_hz,
                               probe_rows=probe_rows)
    write_json(output_root / "response_metrics.json", metrics)

    plot_forcing(forcing_rows, plots_dir / "forcing_history.png")
    plot_qoi(qoi_rows, plots_dir / "qoi_history.png")
    plot_probe_pressure(probe_rows, plots_dir / "probe_pressure_history.png")

    return output_root, {
        "forcing_samples": len(forcing_rows),
        "qoi_samples": len(qoi_rows),
        "probe_samples": len(probe_rows),
        "final_time": solver.time,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run unsteady effective-area breathing")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--amplitude", type=float, default=None)
    parser.add_argument("--epsilon-le-over-S", type=float, default=None)
    parser.add_argument("--frequency-hz", type=float, default=1000.0)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--cycles", type=float, default=0.5)
    parser.add_argument("--t-final", type=float, default=None)
    parser.add_argument("--width", type=float, default=None)
    parser.add_argument("--x-center", type=float, default=None)
    parser.add_argument("--min-area", type=float, default=1.0e-4)
    parser.add_argument("--nx", type=int, default=30)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--baseline-steps", type=int, default=80)
    parser.add_argument("--unsteady-steps", type=int, default=160)
    parser.add_argument("--cfl", type=float, default=0.35)
    parser.add_argument("--mach", type=float, default=6.0)
    parser.add_argument("--altitude", type=float, default=25000.0)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--area-law", choices=("auto", "default", "config_a"),
                        default="auto")
    parser.add_argument("--motion-mode", choices=("auto", "gaussian", "config_a"),
                        default="auto")
    parser.add_argument("--sample-interval-steps", type=int, default=2)
    args = parser.parse_args(argv)

    output_root, summary = run_case(
        output_root=args.output_root,
        amplitude=args.amplitude,
        epsilon_le_over_S=args.epsilon_le_over_S,
        frequency_hz=args.frequency_hz,
        phase=args.phase,
        cycles=args.cycles,
        t_final=args.t_final,
        width=args.width,
        x_center=args.x_center,
        min_area=args.min_area,
        nx=args.nx,
        ny=args.ny,
        baseline_steps=args.baseline_steps,
        unsteady_steps=args.unsteady_steps,
        cfl=args.cfl,
        mach=args.mach,
        altitude=args.altitude,
        sample_interval_steps=args.sample_interval_steps,
        preset=args.preset,
        area_law=args.area_law,
        motion_mode=args.motion_mode,
    )
    print(f"Unsteady area-breathing run written to: {output_root}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
