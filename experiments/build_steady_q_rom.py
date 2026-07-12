"""
Steady POD/ROM connection for the static q_offset wall-position sweep.

This script feeds the static throat-area perturbation parameter ``q``
through the existing steady POD/ROM in ``rom.py``. It is appropriate
because:

  * The static sweep produces converged cold-flow states for varying q.
  * The existing ROM already accepts ``q_throat`` via ``_apply_params``.
  * Snapshots are steady states, not time histories.

It is NOT a time-accurate or unsteady ROM. The unsteady scalar response
surrogate (experiments/build_unsteady_response_surrogate.py) handles
post-transient scalar metrics over the DOE; the two workflows are
intentionally separate.
"""
import argparse
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
    LocalizedAreaPerturbation,
    PerturbedGeometryProfile,
)
from solver import CombustionConfig, InletConfig, Solver, SolverConfig
from rom import ROMEvaluator
from experiments.run_static_wall_sweep import (
    ARTIFACT_SCHEMA_VERSION,
    require_schema_v2,
    write_csv,
    write_json,
)


def load_q_values_from_sweep(sweep_root):
    """Read the q values used by a static wall sweep manifest."""
    manifest_path = Path(sweep_root) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing static sweep manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    q_values = manifest.get("q_values")
    if not q_values:
        raise ValueError(f"manifest has no q_values: {manifest_path}")
    return [float(q) for q in q_values]


def make_base_config(nx=30, ny=6, n_steps=1400, cfl=0.35,
                     mach=6.0, altitude=25000.0, min_area=1.0e-4,
                     source_case_config=None):
    """Build a cold-flow base config that owns a PerturbedGeometryProfile."""
    cfg = SolverConfig()
    cfg.inlet = InletConfig(mach=mach, altitude=altitude, Yf_inlet=0.0)
    cfg.mesh.nx = nx
    cfg.mesh.ny = ny
    cfg.n_steps = n_steps
    cfg.cfl = cfl
    cfg.print_interval = 0
    cfg.steady_rtol = 1.0e-6
    cfg.steady_check_interval = 50
    cfg.residual_interval = 50
    cfg.viscous = False
    cfg.wall_type = "slip"
    cfg.combustion = CombustionConfig(enabled=False)
    cfg.area_source = True

    perturbation = None
    if source_case_config is not None:
        from experiments.export_high_fidelity_scaffold import geometry_from_dict
        inlet = source_case_config["inlet"]
        cfg.inlet = InletConfig(
            mach=float(inlet["mach"]), altitude=float(inlet["altitude"]),
            gamma=float(inlet["gamma"]), R_gas=float(inlet["R_gas"]),
            Yf_inlet=0.0, T_inf=float(inlet["T_inf"]),
            p_inf=float(inlet["p_inf"]),
        )
        source_geometry = geometry_from_dict(source_case_config["geometry"])
        base = getattr(source_geometry, "base_geometry", source_geometry).copy()
        source_perturbation = getattr(source_geometry, "perturbation", None)
        if source_perturbation is not None:
            perturbation = source_perturbation.copy()
            perturbation.amplitude = 0.0
    else:
        base = cfg.geometry
    if perturbation is None:
        perturbation = LocalizedAreaPerturbation(
            enabled=True, mode="throat_gaussian", amplitude=0.0,
            x_center=base.x_throat, width=0.05 * base.L_total,
            min_area=min_area,
        )
    cfg.geometry = PerturbedGeometryProfile(base, perturbation)
    return cfg


def plot_pod_energy(pod, path):
    """Save the standard POD energy spectrum plot."""
    fig = pod.plot_energy()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_q_vs_qoi(q_train, qoi_train, q_test, qoi_test_full, qoi_test_rom, key, path):
    """Plot full-solver and ROM predictions for one QoI versus q."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(q_train, [q[key] for q in qoi_train], "ko", ms=6, label="training (full)")
    if q_test:
        ax.plot(q_test, [q[key] for q in qoi_test_full], "bs", ms=8,
                label="held-out (full)")
        ax.plot(q_test, [q[key] for q in qoi_test_rom], "r^", ms=8,
                label="held-out (POD state QoI)")
        idw = [q.get("qoi_idw", {}).get(key, np.nan) for q in qoi_test_rom]
        ax.plot(q_test, idw, "gx", ms=8, label="held-out (direct QoI IDW)")
    ax.set_xlabel("q_throat [m^2]")
    ax.set_ylabel(key)
    ax.set_title(f"{key} vs q_throat")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def build_rom(sweep_root=None, output_root=None, nx=30, ny=6, n_steps=1400,
              cfl=0.35, mach=6.0, altitude=25000.0, min_area=1.0e-4,
              q_values=None, energy_threshold=0.999, holdout_fraction=0.25,
              seed=42):
    """Train the steady ROM over the static q axis and validate."""
    if q_values is None:
        if sweep_root is None:
            raise ValueError("Provide either --sweep-root or --q-values")
        q_values = load_q_values_from_sweep(sweep_root)
    source_case_config = None
    if sweep_root is not None:
        sweep_root = Path(sweep_root)
        if not sweep_root.is_absolute():
            sweep_root = REPO_ROOT / sweep_root
        require_schema_v2(sweep_root, "static sweep")
        case_configs = []
        for path in (sweep_root / "cases").glob("*/config.json"):
            data = json.loads(path.read_text())
            case_configs.append((abs(float(data.get("q", 0.0))), data["config"]))
        if case_configs:
            source_case_config = min(case_configs, key=lambda item: item[0])[1]

    q_values = [float(q) for q in q_values]
    if len(q_values) < 3:
        raise ValueError(
            f"Steady ROM needs at least 3 q values for POD; got {len(q_values)}."
        )

    if output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "runs" / f"steady_q_rom_{stamp}"
    else:
        output_root = Path(output_root)
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
    plots_dir = output_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = make_base_config(nx=nx, ny=ny, n_steps=n_steps, cfl=cfl,
                                mach=mach, altitude=altitude, min_area=min_area,
                                source_case_config=source_case_config)

    rng = np.random.default_rng(seed)
    q_sorted = sorted(q_values)
    n_total = len(q_sorted)
    n_test = max(int(round(holdout_fraction * n_total)), 1)
    n_test = min(n_test, max(n_total - 3, 0))
    train_q = list(q_sorted)
    test_q = []
    if n_test > 0 and n_total - n_test >= 3:
        idx = np.arange(n_total)
        rng.shuffle(idx)
        test_idx = sorted(idx[:n_test].tolist())
        test_q = [q_sorted[i] for i in test_idx]
        train_q = [q for q in q_sorted if q not in test_q]
        if len(train_q) < 3:
            train_q = list(q_sorted)
            test_q = []

    train_params = [{"q_throat": q} for q in train_q]
    test_params = [{"q_throat": q} for q in test_q]

    print("=" * 60)
    print("Steady q-axis POD/ROM")
    print(f"  Training q values: {train_q}")
    print(f"  Held-out q values: {test_q}")
    print("=" * 60)

    rom = ROMEvaluator(base_cfg, energy_threshold=energy_threshold)
    n_modes = rom.build(train_params)
    if n_modes < 1:
        raise RuntimeError("ROM build failed (no POD modes retained).")

    validation = {}
    qoi_train = list(rom.collector.qoi)
    qoi_test_full = []
    qoi_test_rom = []
    if test_params:
        errors = rom.validate(test_params)
        validation["errors"] = errors
        for params in test_params:
            qoi_test_rom.append(rom.evaluate(params))

        from rom import _clone_config, _apply_params, _compute_qoi
        for params in test_params:
            cfg = _clone_config(base_cfg)
            _apply_params(cfg, params)
            solver = Solver(cfg)
            solver.run()
            qoi_test_full.append(_compute_qoi(solver))

    plot_pod_energy(rom.pod, plots_dir / "pod_energy.png")
    for key in ("tpr", "exit_mach", "mass_defect"):
        plot_q_vs_qoi(train_q, qoi_train, test_q, qoi_test_full, qoi_test_rom, key,
                       plots_dir / f"{key}_vs_q.png")

    write_json(output_root / "manifest.json", {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study": "steady_q_rom",
        "source_sweep_root": str(sweep_root) if sweep_root else None,
        "model": "steady_pod_rom_over_q_throat",
        "time_accurate_rom": False,
        "config_defaults": {
            "nx": nx, "ny": ny, "n_steps": n_steps, "cfl": cfl,
            "mach": mach, "altitude": altitude, "min_area": min_area,
            "energy_threshold": energy_threshold,
        },
        "training_q": train_q,
        "holdout_q": test_q,
        "n_modes": int(n_modes),
        "build_time_s": float(rom.build_time),
        "mean_full_time_s": float(rom.mean_full_time),
    })

    pod_summary = {
        "n_modes": int(rom.pod.n_modes),
        "n_snapshots": int(len(rom.collector.snapshots)),
        "singular_values": rom.pod.singular_values.tolist(),
        "cumulative_energy": rom.pod.cumulative_energy.tolist(),
        "energy_threshold": float(rom.energy_threshold),
    }
    write_json(output_root / "pod_summary.json", pod_summary)

    train_rows = [
        {"q": q, **qoi} for q, qoi in zip(train_q, qoi_train)
    ]
    test_rows = [
        {"q": q, "kind": "rom", **qoi}
        for q, qoi in zip(test_q, qoi_test_rom)
    ] + [
        {"q": q, "kind": "full", **qoi}
        for q, qoi in zip(test_q, qoi_test_full)
    ]
    write_csv(output_root / "training_qoi.csv", train_rows)
    if test_rows:
        write_csv(output_root / "holdout_qoi.csv", test_rows)

    validation_summary = {
        "status": "ok",
        "training_q": train_q,
        "holdout_q": test_q,
        "validation": validation,
        "n_modes": int(rom.pod.n_modes),
        "notes": [
            "Steady POD/ROM trained over static throat-area perturbation q.",
            "Not a time-accurate ROM. Does not replace unsteady response surrogate.",
            "Snapshots are converged cold-flow states from the Python prototype.",
        ],
    }
    write_json(output_root / "validation_summary.json", validation_summary)

    print(f"Steady q-axis ROM written to: {output_root}")
    return output_root, validation_summary


def parse_q_values(text):
    """Parse comma-separated float list."""
    return [float(part) for part in text.split(",") if part.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build a steady POD/ROM over the static throat-area perturbation "
            "axis used by the wall-position workflow. This is a steady ROM "
            "demo, not a time-accurate unsteady ROM."
        ),
    )
    parser.add_argument("--sweep-root", default=None,
                        help="Static wall sweep root (uses its q_values).")
    parser.add_argument("--q-values", default=None,
                        help="Comma-separated q values [m^2] (overrides --sweep-root).")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--nx", type=int, default=30)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--n-steps", type=int, default=1400)
    parser.add_argument("--cfl", type=float, default=0.35)
    parser.add_argument("--mach", type=float, default=6.0)
    parser.add_argument("--altitude", type=float, default=25000.0)
    parser.add_argument("--min-area", type=float, default=1.0e-4)
    parser.add_argument("--energy-threshold", type=float, default=0.999)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    build_rom(
        sweep_root=args.sweep_root,
        output_root=args.output_root,
        nx=args.nx, ny=args.ny, n_steps=args.n_steps, cfl=args.cfl,
        mach=args.mach, altitude=args.altitude, min_area=args.min_area,
        q_values=parse_q_values(args.q_values) if args.q_values else None,
        energy_threshold=args.energy_threshold,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
