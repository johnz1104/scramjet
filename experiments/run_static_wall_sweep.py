"""
Minimal static cold-flow throat-area sweep runner.

This script uses a localized effective-area perturbation,

    A(x; q) = A_base(x) + q * exp(-0.5 * ((x - x_center) / width)^2),

as a low-fidelity wall-position proxy inside the existing quasi-1D area-source
framework. It does not model true wall motion, a moving mesh, ALE, turbulence,
or combustion.
"""
import argparse
import csv
import json
import platform
import subprocess
import sys
import warnings
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
    PerturbedGeometryProfile,
    config_a_ramp_area_law,
    geometry_to_dict,
)
from solver import CombustionConfig, InletConfig, Solver, SolverConfig
from rom import _compute_qoi
from diagnostics import all_case_diagnostics


ARTIFACT_SCHEMA_VERSION = 2


def default_q_values():
    """Default perturbation amplitudes in m^2, scaled to the 0.05 m^2 throat."""
    # Includes every default DOE q_offset so strict exporter matching works.
    return [-0.025, -0.0125, 0.0, 0.0125, 0.025]


def make_config(q, nx=40, ny=8, n_steps=5000, cfl=0.35,
                mach=6.0, altitude=25000.0, width=None,
                x_center=None, min_area=1.0e-4, preset=None,
                steady_rtol=1.0e-6, steady_check_interval=50,
                area_law="default"):
    """Build a cold-flow solver config for one static perturbation value."""
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
    cfg.print_interval = 500
    cfg.steady_rtol = steady_rtol
    cfg.steady_check_interval = int(steady_check_interval)
    cfg.residual_interval = int(steady_check_interval)
    cfg.viscous = False
    cfg.wall_type = "slip"
    cfg.combustion = CombustionConfig(enabled=False)
    cfg.area_source = True

    if area_law == "config_a":
        warnings.warn(
            "Config-A area-law lengths/capture height are placeholders; "
            "calibrate them from the UNSW drawings before quantitative use.",
            RuntimeWarning,
        )
        cfg.geometry = config_a_ramp_area_law()
    elif area_law != "default":
        raise ValueError(f"unsupported area law: {area_law}")

    base = cfg.geometry
    if width is None:
        width = 0.05 * base.L_total
    if x_center is None:
        x_center = base.x_throat

    perturbation = LocalizedAreaPerturbation(
        enabled=True,
        mode="throat_gaussian",
        amplitude=q,
        x_center=x_center,
        width=width,
        min_area=min_area,
    )
    cfg.geometry = PerturbedGeometryProfile(base, perturbation)
    return cfg


def config_to_dict(cfg):
    """Return a JSON-serialisable config manifest."""
    return {
        "inlet": {
            "mach": cfg.inlet.mach,
            "altitude": cfg.inlet.altitude,
            "gamma": cfg.inlet.gamma,
            "R_gas": cfg.inlet.R_gas,
            "Yf_inlet": cfg.inlet.Yf_inlet,
            "T_inf": cfg.inlet.T_inf,
            "p_inf": cfg.inlet.p_inf,
            "rho_inf": cfg.inlet.rho_inf,
            "u_inf": cfg.inlet.u_inf,
        },
        "mesh": {
            "nx": cfg.mesh.nx,
            "ny": cfg.mesh.ny,
            "y_stretch": cfg.mesh.y_stretch,
        },
        "solver": {
            "cfl": cfg.cfl,
            "n_steps": cfg.n_steps,
            "print_interval": cfg.print_interval,
            "steady_rtol": cfg.steady_rtol,
            "steady_check_interval": cfg.steady_check_interval,
            "residual_interval": cfg.residual_interval,
        },
        "physics": {
            "cold_flow_only": True,
            "viscous": cfg.viscous,
            "wall_type": cfg.wall_type,
            "area_source": cfg.area_source,
            "combustion_enabled": cfg.combustion.enabled,
            "passive_scalar_enabled": cfg.passive_scalar_enabled,
            "heat_release_model": cfg.heat_release_model,
            "turbulence_model": cfg.turbulence_model,
        },
        "geometry": geometry_to_dict(cfg.geometry),
    }


def git_metadata():
    """Best-effort git state for reproducibility."""
    def _run(args):
        try:
            result = subprocess.run(
                args, cwd=REPO_ROOT, check=False, capture_output=True, text=True,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    versions = {"python": platform.python_version()}
    for module_name in ("numpy", "scipy", "numba", "matplotlib"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except (ImportError, AttributeError):
            versions[module_name] = None

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "status_short": _run(["git", "status", "--short"]),
        "dependencies": versions,
    }


def write_json(path, data):
    """Write pretty JSON."""
    def _safe(value):
        if isinstance(value, dict):
            return {key: _safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_safe(item) for item in value]
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value) if np.isfinite(value) else None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_safe(data), f, indent=2, allow_nan=False)
        f.write("\n")


def write_csv(path, rows, fieldnames=None):
    """Write dictionaries to CSV."""
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def require_schema_v2(artifact_root, artifact_name="artifact", warn_only=False):
    """Gate analyses against pre-fix artifacts lacking schema version 2."""
    root = Path(artifact_root)
    manifest_path = root / "manifest.json"
    version = None
    if manifest_path.is_file():
        try:
            version = json.loads(manifest_path.read_text()).get("schema_version")
        except (OSError, json.JSONDecodeError):
            version = None
    if version == ARTIFACT_SCHEMA_VERSION:
        return True
    message = (
        f"{artifact_name} at {root} is schema_version={version!r}; "
        f"version {ARTIFACT_SCHEMA_VERSION} is required because pre-fix "
        "artifacts used invalid completion/energy conventions"
    )
    if warn_only:
        warnings.warn(message, RuntimeWarning)
        return False
    raise ValueError(message)


def case_name(q):
    """Filesystem-safe q label."""
    return f"q_{q:+.6f}".replace("+", "p").replace("-", "m").replace(".", "p")


def centerline_rows(solver):
    """Extract centerline primitive data."""
    rho, u, v, p, T, Yf = solver.state.primitives()
    M = solver.state.mach()
    j_mid = solver.mesh.ny // 2
    rows = []
    for i, x in enumerate(solver.mesh.xc):
        rows.append({
            "x": float(x),
            "mach": float(M[i, j_mid]),
            "pressure": float(p[i, j_mid]),
            "temperature": float(T[i, j_mid]),
            "density": float(rho[i, j_mid]),
            "u": float(u[i, j_mid]),
            "v": float(v[i, j_mid]),
        })
    return rows


def residual_rows(solver):
    """Extract available residual samples."""
    return [
        {
            "sample": idx,
            "step": int(solver.residual_steps[idx]),
            "time": float(solver.residual_times[idx]),
            "residual_l2": float(res),
        }
        for idx, res in enumerate(solver.residual_history)
    ]


def qoi_with_diagnostics(solver, q):
    """Compute available QoI and simple field extrema."""
    qoi = _compute_qoi(solver)
    rho, u, v, p, T, Yf = solver.state.primitives()
    M = solver.state.mach()
    qoi.update({
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "q": float(q),
        "steps": int(solver.step_count),
        "time": float(solver.time),
        "dt_min": float(np.min(solver.dt_history)) if solver.dt_history else 0.0,
        "dt_max": float(np.max(solver.dt_history)) if solver.dt_history else 0.0,
        "mach_min": float(np.min(M)),
        "mach_max": float(np.max(M)),
        "pressure_min": float(np.min(p)),
        "pressure_max": float(np.max(p)),
        "temperature_min": float(np.min(T)),
        "temperature_max": float(np.max(T)),
        "throat_area": solver.cfg.geometry.throat_area(),
        "min_sampled_area": solver.cfg.geometry.min_area_value(),
        "converged": bool(solver.run_status and solver.run_status["converged"]),
        "final_residual": (
            solver.run_status.get("final_residual") if solver.run_status else None
        ),
    })

    history = (solver.run_status or {}).get("steady_qoi_history", [])
    drift = {}
    if len(history) == 2:
        old = history[0]["qoi"]
        new = history[1]["qoi"]
        for key in sorted(set(old) & set(new)):
            a, b = old[key], new[key]
            if (isinstance(a, (int, float, np.number))
                    and not isinstance(a, (bool, np.bool_))
                    and isinstance(b, (int, float, np.number))
                    and not isinstance(b, (bool, np.bool_))
                    and np.isfinite(a) and np.isfinite(b)):
                drift[key] = float(abs(b - a) / max(abs(b), 1.0e-30))
    qoi["qoi_drift_relative"] = drift
    qoi["qoi_drift_max_relative"] = max(drift.values(), default=None)
    return qoi


def run_case(q, case_dir, **config_kwargs):
    """Run one cold-flow case and write per-case outputs."""
    cfg = make_config(q, **config_kwargs)
    write_json(case_dir / "config.json", {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "q": float(q),
        "git": git_metadata(),
        "config": config_to_dict(cfg),
    })

    solver = Solver(cfg)
    solver.run(steady_qoi_fn=_compute_qoi)

    qoi = qoi_with_diagnostics(solver, q)
    write_json(case_dir / "qoi.json", qoi)
    write_json(case_dir / "diagnostics.json", all_case_diagnostics(solver))
    write_csv(case_dir / "centerline.csv", centerline_rows(solver))
    write_csv(case_dir / "residual.csv", residual_rows(solver),
              fieldnames=["sample", "step", "time", "residual_l2"])
    return solver, qoi


def plot_area_profiles(configs, q_values, output_path):
    """Plot A(x) for each sweep value."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for cfg, q in zip(configs, q_values):
        x = np.linspace(0.0, cfg.geometry.L_total, 500)
        ax.plot(x, cfg.geometry.area(x), lw=1.4, label=f"q={q:g}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("A(x) [m^2]")
    ax.set_title("Static Effective Area Profiles")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_qoi_vs_q(summary_rows, output_path, metric="tpr"):
    """Plot one QoI versus q."""
    q = np.array([row["q"] for row in summary_rows], dtype=float)
    y = np.array([row[metric] for row in summary_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(q, y, "o-", lw=1.5)
    ax.set_xlabel("q [m^2]")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs static throat-area perturbation")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def run_sweep(q_values=None, output_root=None, **config_kwargs):
    """Run the full static sweep and write aggregate outputs."""
    if q_values is None:
        q_values = default_q_values()
    q_values = [float(q) for q in q_values]

    if output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "runs" / f"static_wall_sweep_{stamp}"
    else:
        output_root = Path(output_root)
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root

    cases_dir = output_root / "cases"
    plots_dir = output_root / "plots"
    cases_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_root / "manifest.json", {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study": "static_wall_sweep",
        "model": "cold_flow_effective_area_perturbation",
        "q_values": q_values,
        "config_defaults": config_kwargs,
        "git": git_metadata(),
    })

    summary_rows = []
    configs = []
    for q in q_values:
        case_dir = cases_dir / case_name(q)
        solver, qoi = run_case(q, case_dir, **config_kwargs)
        summary_rows.append(qoi)
        configs.append(solver.cfg)

    fieldnames = sorted({key for row in summary_rows for key in row.keys()})
    write_csv(output_root / "summary.csv", summary_rows, fieldnames=fieldnames)
    plot_area_profiles(configs, q_values, plots_dir / "area_profiles.png")
    plot_qoi_vs_q(summary_rows, plots_dir / "qoi_vs_q.png", metric="tpr")
    return output_root, summary_rows


def parse_q_values(text):
    """Parse comma-separated q values."""
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a static cold-flow wall sweep")
    parser.add_argument("--q-values", default=",".join(str(q) for q in default_q_values()))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=5000,
                        help="Hard step cap for convergence-driven cases.")
    parser.add_argument("--steady-rtol", type=float, default=1.0e-6)
    parser.add_argument("--steady-check-interval", type=int, default=50)
    parser.add_argument("--cfl", type=float, default=0.35)
    parser.add_argument("--mach", type=float, default=6.0)
    parser.add_argument("--altitude", type=float, default=25000.0)
    parser.add_argument("--width", type=float, default=None)
    parser.add_argument("--x-center", type=float, default=None)
    parser.add_argument("--min-area", type=float, default=1.0e-4)
    parser.add_argument("--preset", default=None,
                        help="Experiment-condition preset (e.g. configs/tusq_m585.json); "
                             "overrides --mach/--altitude.")
    parser.add_argument("--area-law", choices=("default", "config_a"),
                        default="default")
    args = parser.parse_args(argv)

    output_root, _ = run_sweep(
        q_values=parse_q_values(args.q_values),
        output_root=args.output_root,
        nx=args.nx,
        ny=args.ny,
        n_steps=args.n_steps,
        cfl=args.cfl,
        mach=args.mach,
        altitude=args.altitude,
        width=args.width,
        x_center=args.x_center,
        min_area=args.min_area,
        preset=args.preset,
        steady_rtol=args.steady_rtol,
        steady_check_interval=args.steady_check_interval,
        area_law=args.area_law,
    )
    print(f"Static wall sweep written to: {output_root}")


if __name__ == "__main__":
    main()
