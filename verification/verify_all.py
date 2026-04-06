"""
verify_all.py — end-to-end validation harness.

Runs every subsystem of the scramjet project and writes the numbers we need
for the report: a converged clean scramjet run (Mach 6, 25 km, inviscid
with variable-area source), a separate low-intensity combustion run to
exercise the heat-release model in flowing conditions, a POD ROM build +
held-out validation, and a Bayesian-optimisation loop that uses the ROM.
Results are dumped to verify_results.json.
"""
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver import SolverConfig, Solver, InletConfig, MeshConfig, CombustionConfig
from rom import ROMEvaluator, _compute_qoi
from optimization import DesignSpace, BayesianOptimizer

results = {}

# 1) clean scramjet run (Mach 6, 25 km, inviscid, variable-area on)
print("\n[1] Clean scramjet run: Mach 6, h=25km, inviscid + variable-area")
cfg = SolverConfig()
cfg.inlet = InletConfig(mach=6.0, altitude=25000.0)
cfg.mesh.nx = 80
cfg.mesh.ny = 16
cfg.n_steps = 1500
cfg.print_interval = 300
cfg.cfl = 0.4
cfg.viscous = False
cfg.combustion = CombustionConfig(enabled=False)
cfg.area_source = True

t0 = time.time()
solver = Solver(cfg)
solver.run()
wall_full = time.time() - t0

qoi = _compute_qoi(solver)
rho, u, v, p, T, Yf = solver.state.primitives()
M = solver.state.mach()

results['scramjet_clean'] = {
    'mach_freestream': 6.0,
    'altitude_m': 25000.0,
    'T_inf_K': float(cfg.inlet.T_inf),
    'p_inf_Pa': float(cfg.inlet.p_inf),
    'rho_inf_kg_m3': float(cfg.inlet.rho_inf),
    'u_inf_m_s': float(cfg.inlet.u_inf),
    'wall_time_s': wall_full,
    'n_cells': int(cfg.mesh.nx * cfg.mesh.ny),
    'n_steps': cfg.n_steps,
    'final_time_s': float(solver.time),
    'M_exit': float(qoi['exit_mach']),
    'M_max': float(M.max()),
    'M_min': float(M.min()),
    'p_max_Pa': float(p.max()),
    'p_min_Pa': float(p.min()),
    'T_max_K': float(T.max()),
    'T_min_K': float(T.min()),
    'thrust_N_per_m': float(qoi['thrust']),
    'Isp_s': float(qoi['Isp']),
    'pressure_recovery': float(qoi['pressure_recovery']),
    'mdot_kg_s_per_m': float(qoi['mdot']),
}
print(f"  wall={wall_full:.2f}s  thrust={qoi['thrust']:.1f}  Isp={qoi['Isp']:.1f}")
print(f"  M range=[{M.min():.3f},{M.max():.3f}]  T range=[{T.min():.0f},{T.max():.0f}]K")

fig = solver.plot_mach()
fig.savefig("verify_mach.png", dpi=140); plt.close(fig)
fig = solver.plot_centerline()
fig.savefig("verify_centerline.png", dpi=140); plt.close(fig)

# 2) combustion-coupled run (mild heat release for stability)
print("\n[2] Combustion-coupled run (mild Arrhenius)")
cfg2 = SolverConfig()
cfg2.inlet = InletConfig(mach=6.0, altitude=25000.0, Yf_inlet=0.02)
cfg2.mesh.nx = 60
cfg2.mesh.ny = 12
cfg2.n_steps = 600
cfg2.print_interval = 300
cfg2.cfl = 0.3
cfg2.viscous = False
cfg2.combustion = CombustionConfig(
    enabled=True,
    A_pre=1.0e6, Ea=65000.0, Q_heat=1.5e6,
)
cfg2.area_source = True

t0 = time.time()
solver2 = Solver(cfg2)
solver2.run()
wall_comb = time.time() - t0

qoi2 = _compute_qoi(solver2)
rho2, u2, v2, p2, T2, Yf2 = solver2.state.primitives()
# centreline Yf depletion
j_mid = solver2.mesh.ny // 2
Yf_inlet_mean = float(np.mean(Yf2[0, :]))
Yf_exit_mean = float(np.mean(Yf2[-1, :]))
T_exit_mean = float(np.mean(T2[-1, :]))
T_inlet_mean = float(np.mean(T2[0, :]))

results['scramjet_combustion'] = {
    'mach_freestream': 6.0,
    'altitude_m': 25000.0,
    'Yf_inlet': 0.02,
    'Arrhenius_A_pre': 1.0e6,
    'Arrhenius_Ea_J_mol': 65000.0,
    'Arrhenius_Q_heat_J_kg': 1.5e6,
    'wall_time_s': wall_comb,
    'n_cells': int(cfg2.mesh.nx * cfg2.mesh.ny),
    'n_steps': cfg2.n_steps,
    'Yf_inlet_mean': Yf_inlet_mean,
    'Yf_exit_mean': Yf_exit_mean,
    'Yf_consumed_frac': float(1.0 - Yf_exit_mean / max(Yf_inlet_mean, 1e-30)),
    'T_inlet_mean_K': T_inlet_mean,
    'T_exit_mean_K': T_exit_mean,
    'delta_T_K': T_exit_mean - T_inlet_mean,
    'thrust_N_per_m': float(qoi2['thrust']),
    'Isp_s': float(qoi2['Isp']),
}
print(f"  wall={wall_comb:.2f}s")
print(f"  Yf inlet={Yf_inlet_mean:.4f} -> exit={Yf_exit_mean:.4f} "
      f"({results['scramjet_combustion']['Yf_consumed_frac']*100:.1f}% consumed)")
print(f"  T inlet={T_inlet_mean:.1f}K -> exit={T_exit_mean:.1f}K  "
      f"(dT={results['scramjet_combustion']['delta_T_K']:.1f}K)")

fig = solver2.plot_field("temperature")
fig.savefig("verify_combustion_T.png", dpi=140); plt.close(fig)
fig = solver2.plot_field("fuel_fraction")
fig.savefig("verify_combustion_Yf.png", dpi=140); plt.close(fig)

# 3) POD ROM build + held-out validation
print("\n[3] POD ROM build")
rom_cfg = SolverConfig()
rom_cfg.mesh.nx = 40
rom_cfg.mesh.ny = 10
rom_cfg.n_steps = 400
rom_cfg.print_interval = 500
rom_cfg.cfl = 0.35

train_params = [
    {'A_exit': 0.12, 'L_nozzle': 0.35},
    {'A_exit': 0.14, 'L_nozzle': 0.40},
    {'A_exit': 0.16, 'L_nozzle': 0.45},
    {'A_exit': 0.18, 'L_nozzle': 0.50},
    {'A_exit': 0.13, 'L_nozzle': 0.45},
    {'A_exit': 0.15, 'L_nozzle': 0.35},
    {'A_exit': 0.17, 'L_nozzle': 0.40},
    {'A_exit': 0.19, 'L_nozzle': 0.50},
    {'A_exit': 0.20, 'L_nozzle': 0.55},
]
rom = ROMEvaluator(rom_cfg, energy_threshold=0.999)
r = rom.build(train_params)

test_params = [
    {'A_exit': 0.135, 'L_nozzle': 0.42},
    {'A_exit': 0.155, 'L_nozzle': 0.38},
    {'A_exit': 0.175, 'L_nozzle': 0.47},
]
errors = rom.validate(test_params)

t0 = time.time()
for p in test_params:
    rom.evaluate(p)
rom_eval_time = (time.time() - t0) / len(test_params)

mean_full = rom.collector.mean_wall_time()
speedup = mean_full / max(rom_eval_time, 1e-9)
# wall-time reduction when 80% of optimisation evaluations use ROM
wall_saving_pct = (1.0 - (0.2 * mean_full + 0.8 * rom_eval_time) / mean_full) * 100.0

results['rom'] = {
    'n_training_snapshots': len(train_params),
    'n_modes': int(r),
    'energy_threshold': 0.999,
    'mean_full_solver_s': float(mean_full),
    'mean_rom_eval_s': float(rom_eval_time),
    'speedup': float(speedup),
    'wall_saving_pct_at_80pct_rom': float(wall_saving_pct),
    'mean_rel_err_thrust': float(errors.get('thrust', 0.0)),
    'mean_rel_err_Isp': float(errors.get('Isp', 0.0)),
    'mean_rel_err_exit_mach': float(errors.get('exit_mach', 0.0)),
    'mean_rel_err_pressure_recovery': float(errors.get('pressure_recovery', 0.0)),
}

fig = rom.pod.plot_energy()
fig.savefig("verify_pod_energy.png", dpi=140); plt.close(fig)

# 4) Bayesian optimisation with ROM acceleration
print("\n[4] Bayesian optimisation with ROM acceleration")
bo_cfg = SolverConfig()
bo_cfg.mesh.nx = 40
bo_cfg.mesh.ny = 10
bo_cfg.n_steps = 400
bo_cfg.print_interval = 500
bo_cfg.cfl = 0.35

space = DesignSpace([
    ('A_exit', 0.10, 0.20),
    ('L_nozzle', 0.30, 0.55),
    ('L_combustor', 0.35, 0.60),
])

optimizer = BayesianOptimizer(
    space, bo_cfg, rom_evaluator=rom,
    objective_weights={'thrust': -1.0, 'Isp': -0.5},
)

t0 = time.time()
best_params, best_qoi = optimizer.run(
    n_init=6, n_iter=14,
    uncertainty_threshold=0.25,
    n_candidates=200,
    verbose=True,
)
bo_wall = time.time() - t0

results['bayes_opt'] = {
    'total_wall_s': float(bo_wall),
    'n_total_evals': int(len(optimizer.X_eval)),
    'n_full_evals': int(optimizer.n_full),
    'n_rom_evals': int(optimizer.n_rom),
    'full_solver_wall_s': float(optimizer.full_solver_time),
    'rom_wall_s': float(optimizer.rom_time),
    'pct_rom_evals': float(
        100.0 * optimizer.n_rom / max(optimizer.n_full + optimizer.n_rom, 1)
    ),
    'best_params': {k: float(v) for k, v in best_params.items()},
    'best_thrust': float(best_qoi['thrust']),
    'best_Isp': float(best_qoi['Isp']),
    'best_objective': float(optimizer.best_y),
}
if optimizer.n_full > 0:
    t_full_only = (optimizer.n_full + optimizer.n_rom) * (optimizer.full_solver_time / optimizer.n_full)
    savings = (1.0 - (optimizer.full_solver_time + optimizer.rom_time) / t_full_only) * 100.0
    results['bayes_opt']['estimated_wall_saving_pct'] = float(savings)

fig = optimizer.plot_convergence()
fig.savefig("verify_bo_convergence.png", dpi=140); plt.close(fig)

with open("verify_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for section, data in results.items():
    print(f"\n[{section}]")
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

print("\nAll end-to-end runs completed. Results written to verify_results.json")
