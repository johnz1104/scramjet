"""Assertion-bearing end-to-end verification for the research workflow."""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from solver import SolverConfig, Solver, InletConfig, CombustionConfig
from rom import ROMEvaluator, _compute_qoi
from optimization import DesignSpace, BayesianOptimizer


SCHEMA_VERSION = 2
results = {"schema_version": SCHEMA_VERSION}


def converged_config(nx, ny, n_steps=1800):
    cfg = SolverConfig()
    cfg.mesh.nx, cfg.mesh.ny = nx, ny
    cfg.n_steps = n_steps
    cfg.print_interval = 0
    cfg.residual_interval = 50
    cfg.steady_check_interval = 50
    cfg.steady_rtol = 1.0e-6
    cfg.cfl = 0.35
    return cfg


def finite_or_none(value):
    """Recursively make strict JSON while retaining absent-shock semantics."""
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


print("\n[1] Converged clean cold-flow run")
cfg = converged_config(64, 12, n_steps=2200)
cfg.inlet = InletConfig(mach=6.0, altitude=25000.0)
cfg.combustion = CombustionConfig(enabled=False)
t0 = time.time()
solver = Solver(cfg)
run_status = solver.run()
clean_wall = time.time() - t0
qoi = _compute_qoi(solver)
rho, u, v, p, T, Yf = solver.state.primitives()
M = solver.state.mach()
results["scramjet_clean"] = {
    "scope": "paper-1 cold-flow baseline",
    "wall_time_s": clean_wall,
    "n_cells": cfg.mesh.nx * cfg.mesh.ny,
    "run_status": run_status,
    "qoi": qoi,
    "mach_range": [float(M.min()), float(M.max())],
    "temperature_range_K": [float(T.min()), float(T.max())],
    "mass_balance_gate": {
        "absolute_mass_defect_limit": 0.03,
        "role": "coarse-grid regression guard, not a grid-convergence result",
        "research_grade_target_after_grid_refinement": 0.01,
    },
}
fig = solver.plot_mach()
fig.savefig(os.path.join(_HERE, "verify_mach.png"), dpi=140)
plt.close(fig)
fig = solver.plot_centerline()
fig.savefig(os.path.join(_HERE, "verify_centerline.png"), dpi=140)
plt.close(fig)


print("\n[2] Extended capability (combustion; out of paper scope)")
cfg2 = converged_config(48, 8, n_steps=500)
cfg2.steady_rtol = None
cfg2.inlet = InletConfig(mach=6.0, altitude=25000.0, Yf_inlet=0.02)
cfg2.combustion = CombustionConfig(
    enabled=True, A_pre=1.0e6, Ea=65000.0, Q_heat=1.5e6,
)
t0 = time.time()
solver2 = Solver(cfg2)
solver2.run()
comb_wall = time.time() - t0
rho2, u2, v2, p2, T2, Yf2 = solver2.state.primitives()
results["extended_combustion_capability"] = {
    "paper_scope": False,
    "decision": "parked; revisit after cold-flow multi-fidelity loop closes",
    "wall_time_s": comb_wall,
    "fuel_range": [float(Yf2.min()), float(Yf2.max())],
    "temperature_range_K": [float(T2.min()), float(T2.max())],
    "qoi": _compute_qoi(solver2),
}
fig = solver2.plot_field("temperature")
fig.savefig(os.path.join(_HERE, "verify_combustion_T.png"), dpi=140)
plt.close(fig)
fig = solver2.plot_field("fuel_fraction")
fig.savefig(os.path.join(_HERE, "verify_combustion_Yf.png"), dpi=140)
plt.close(fig)


print("\n[3] Coefficient-interpolated POD with state-derived QoIs")
rom_cfg = converged_config(32, 6, n_steps=1400)
train_params = [
    {"A_exit": 0.12, "L_nozzle": 0.34},
    {"A_exit": 0.14, "L_nozzle": 0.40},
    {"A_exit": 0.16, "L_nozzle": 0.46},
    {"A_exit": 0.18, "L_nozzle": 0.52},
    {"A_exit": 0.13, "L_nozzle": 0.49},
    {"A_exit": 0.19, "L_nozzle": 0.37},
]
test_params = [
    {"A_exit": 0.145, "L_nozzle": 0.43},
    {"A_exit": 0.172, "L_nozzle": 0.48},
]
rom = ROMEvaluator(rom_cfg, energy_threshold=0.999)
n_modes = rom.build(train_params)
errors = rom.validate(test_params)
results["rom"] = {
    "identity": "coefficient-interpolated POD state reconstruction",
    "comparison_baseline": "direct QoI IDW",
    "training_parameters": ["A_exit", "L_nozzle"],
    "n_training_snapshots": len(train_params),
    "n_modes": n_modes,
    "build_time_s": rom.build_time,
    "mean_full_solver_s": rom.mean_full_time,
    "validation_mean_relative_errors": errors,
}
fig = rom.pod.plot_energy()
fig.savefig(os.path.join(_HERE, "verify_pod_energy.png"), dpi=140)
plt.close(fig)


print("\n[4] GP adaptive sampling with ROM prescreen/full confirmation")
space = DesignSpace([
    ("A_exit", 0.12, 0.19),
    ("L_nozzle", 0.34, 0.52),
])
optimizer = BayesianOptimizer(
    space, rom_cfg, rom_evaluator=rom,
    objective_weights={"tpr": -1.0},
)
t0 = time.time()
best_params, best_qoi = optimizer.run(
    n_init=4, n_iter=5, n_candidates=120, rom_top_m=4, verbose=True,
)
bo_wall = time.time() - t0
results["adaptive_sampling"] = {
    "total_session_wall_s": bo_wall,
    "objective": "maximize tpr",
    "best_params": best_params,
    "best_qoi": best_qoi,
    "best_full_verified": optimizer.best_full_verified,
    "all_gp_observations_full": all(
        source == "full" for source in optimizer.source_eval
    ),
    "cost_report": optimizer.cost_report,
}
fig = optimizer.plot_convergence()
fig.savefig(os.path.join(_HERE, "verify_bo_convergence.png"), dpi=140)
plt.close(fig)


pod_tpr_error = errors.get("pod_state", {}).get("tpr")
idw_tpr_error = errors.get("idw", {}).get("tpr")
assertions = {
    "clean_run_converged": bool(run_status["converged"]),
    "clean_mass_balance_within_3pct": abs(qoi["mass_defect"]) < 0.03,
    "clean_tpr_physical": 0.0 < qoi["tpr"] <= 1.0 + 1.0e-8,
    "clean_state_admissible": bool(qoi["state_admissible"]),
    "rom_has_modes": n_modes >= 1,
    "pod_tpr_error_below_loose_bound": (
        pod_tpr_error is not None and pod_tpr_error < 0.35
    ),
    "idw_tpr_error_below_loose_bound": (
        idw_tpr_error is not None and idw_tpr_error < 0.35
    ),
    "bo_best_is_full_verified": bool(optimizer.best_full_verified),
    "gp_trained_on_full_only": all(
        source == "full" for source in optimizer.source_eval
    ),
    "bo_best_tpr_finite": bool(np.isfinite(best_qoi["tpr"])),
    "combustion_fuel_bounded": bool(Yf2.min() >= -1.0e-10 and Yf2.max() <= 1.0 + 1.0e-10),
}
results["assertions"] = {
    "passed": all(assertions.values()),
    "checks": assertions,
}

strict_results = finite_or_none(results)
with open(os.path.join(_HERE, "verify_results.json"), "w") as handle:
    json.dump(strict_results, handle, indent=2, allow_nan=False)
    handle.write("\n")

print("\n" + "=" * 60)
print("VERIFICATION ASSERTIONS")
print("=" * 60)
for name, passed in assertions.items():
    print(f"  {name}: {'PASS' if passed else 'FAIL'}")
if not all(assertions.values()):
    print("Verification failed; see verification/verify_results.json")
    sys.exit(1)
print("All end-to-end assertions passed.")
