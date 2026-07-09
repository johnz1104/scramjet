"""
tests.py — Validation test suite for the scramjet CFD solver.

Validation cases, each targeting a different solver component:

    0. Area perturbation    — validates static throat-area wrapper
    1. Sod shock tube       — validates FVM (HLLC, MUSCL, RK3-SSP)
    2. Nozzle area–Mach     — validates the quasi-1D variable-area coupling
                              against the exact isentropic solution
                              (+ agreement with the standalone source form)
    3. Shock position       — validates the back-pressure outlet BC and
                              shock capture against the exact normal-shock-
                              in-diverging-duct solution
    4. Couette flow         — validates the FEMViscous implicit diffusion
                              operator (the actual class, moving top wall)
    5. Ignition delay       — validates Arrhenius combustion model

Usage:
    python tests.py              # run all tests
    python tests.py geometry     # area perturbation only
    python tests.py sod          # Sod shock tube only
    python tests.py nozzle       # isentropic area-Mach only
    python tests.py shock        # analytic shock position only
    python tests.py couette      # Couette flow only
    python tests.py ignition     # ignition delay only

Dependency: mesh.py, fvm.py, physics.py, solver.py, diagnostics.py
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mesh import (
    StructuredMesh2D,
    GeometryProfile,
    LocalizedAreaPerturbation,
    PerturbedGeometryProfile,
    SinusoidalAreaForcing,
    TabulatedAreaProfile,
    TimeDependentPerturbedGeometryProfile,
)
from fvm import StateVector, BoundaryConditions, FVMResidual, TimeIntegrator
from physics import (
    TransportProperties,
    FEMViscous,
    SingleStepArrhenius,
    SimpleHeatRelease,
    VariableAreaSource,
)
from solver import SolverConfig, Solver, InletConfig, CombustionConfig
from diagnostics import scalar_diagnostics, shock_diagnostics
from response_metrics import extract_response_metrics, fit_sinusoid


# Exact gas-dynamics references live in gasdynamics.py (shared with the
# forced-shock benchmark and the Busemann generator)
from gasdynamics import isentropic_area_ratio as _isentropic_area_ratio
from gasdynamics import mach_from_area_ratio as _mach_from_area_ratio
from gasdynamics import normal_shock as _normal_shock_full


def _normal_shock(M, gamma=1.4):
    """Return (M_downstream, p0_downstream/p0_upstream)."""
    ns = _normal_shock_full(M, gamma)
    return ns["M2"], ns["p0_ratio"]


def test_area_perturbation():
    """
    Localized throat-area perturbation sanity test.

    This verifies the static effective-area mode used by the cold-flow
    wall-position workflow. It does not represent moving-wall CFD.
    """
    print("=" * 60)
    print("TEST 0: Localized Area Perturbation")
    print("=" * 60)

    base = GeometryProfile.default()
    x = np.linspace(0.0, base.L_total, 200)
    x_center = base.x_throat
    x_c = np.array([x_center])

    zero = LocalizedAreaPerturbation(
        enabled=True,
        amplitude=0.0,
        x_center=x_center,
        width=0.05,
        min_area=1.0e-6,
    )
    geom_zero = PerturbedGeometryProfile(base, zero)
    zero_area_ok = np.array_equal(geom_zero.area(x), base.area(x))
    zero_grad_ok = np.array_equal(geom_zero.area_gradient(x), base.area_gradient(x))

    positive = LocalizedAreaPerturbation(
        enabled=True,
        amplitude=0.005,
        x_center=x_center,
        width=0.05,
        min_area=1.0e-6,
    )
    geom_positive = PerturbedGeometryProfile(base, positive)
    positive_ok = geom_positive.area(x_c)[0] > base.area(x_c)[0]

    negative = LocalizedAreaPerturbation(
        enabled=True,
        amplitude=-0.005,
        x_center=x_center,
        width=0.05,
        min_area=0.01,
    )
    geom_negative = PerturbedGeometryProfile(base, negative)
    A_negative = geom_negative.area(x_c)[0]
    negative_ok = A_negative < base.area(x_c)[0] and A_negative > negative.min_area

    invalid_raised = False
    try:
        invalid = LocalizedAreaPerturbation(
            enabled=True,
            amplitude=-0.2,
            x_center=x_center,
            width=0.05,
            min_area=0.01,
        )
        PerturbedGeometryProfile(base, invalid)
    except ValueError:
        invalid_raised = True

    grad_mode = LocalizedAreaPerturbation(
        enabled=True,
        amplitude=0.003,
        x_center=x_center,
        width=0.08,
        min_area=1.0e-6,
    )
    geom_grad = PerturbedGeometryProfile(base, grad_mode)
    x_fd = np.linspace(0.05, base.L_total - 0.05, 120)
    x_fd = x_fd[np.abs(x_fd - base.x_throat) > 1.0e-3]
    x_fd = x_fd[np.abs(x_fd - base.x_comb_exit) > 1.0e-3]
    h = 1.0e-6
    grad_exact = geom_grad.area_gradient(x_fd)
    grad_fd = (geom_grad.area(x_fd + h) - geom_grad.area(x_fd - h)) / (2.0 * h)
    grad_ok = np.max(np.abs(grad_exact - grad_fd)) < 5.0e-5

    disabled = LocalizedAreaPerturbation(
        enabled=False,
        amplitude=-1.0,
        x_center=x_center,
        width=0.05,
        min_area=0.01,
    )
    geom_disabled = PerturbedGeometryProfile(base, disabled)
    disabled_ok = np.array_equal(geom_disabled.area(x), base.area(x))

    forcing = SinusoidalAreaForcing(
        amplitude=0.002,
        frequency_hz=250.0,
        phase=0.3,
    )
    t_samples = np.array([0.0, 0.0005, 0.001, 0.0015])
    q_expected = 0.002 * np.sin(2.0 * np.pi * 250.0 * t_samples + 0.3)
    q_actual = np.array([forcing.value(t) for t in t_samples])
    forcing_ok = np.allclose(q_actual, q_expected, rtol=0.0, atol=1e-15)

    forcing_with_mean = SinusoidalAreaForcing(
        amplitude=0.002,
        frequency_hz=250.0,
        phase=0.3,
        mean=0.01,
    )
    q_with_mean_expected = 0.01 + q_expected
    q_with_mean_actual = np.array([forcing_with_mean.value(t) for t in t_samples])
    forcing_mean_ok = np.allclose(q_with_mean_actual, q_with_mean_expected,
                                   rtol=0.0, atol=1e-15)

    dyn_zero = TimeDependentPerturbedGeometryProfile(
        base,
        LocalizedAreaPerturbation(
            enabled=True,
            amplitude=0.0,
            x_center=x_center,
            width=0.05,
            min_area=1.0e-6,
        ),
        SinusoidalAreaForcing(amplitude=0.0, frequency_hz=100.0, phase=1.1),
    )
    dyn_zero.set_time(0.123)
    dyn_zero_ok = np.array_equal(dyn_zero.area(x), base.area(x))

    static_forcing = TimeDependentPerturbedGeometryProfile(
        base,
        LocalizedAreaPerturbation(
            enabled=True,
            amplitude=0.0,
            x_center=x_center,
            width=0.05,
            min_area=1.0e-6,
        ),
        SinusoidalAreaForcing(amplitude=0.004, frequency_hz=0.0, phase=np.pi / 2.0),
    )
    A_static_0 = static_forcing.area_at_time(x_c, 0.0)[0]
    A_static_1 = static_forcing.area_at_time(x_c, 99.0)[0]
    static_forcing_ok = (
        abs(A_static_0 - (base.area(x_c)[0] + 0.004)) < 1.0e-12
        and abs(A_static_1 - A_static_0) < 1.0e-12
    )

    dynamic_invalid_raised = False
    try:
        TimeDependentPerturbedGeometryProfile(
            base,
            LocalizedAreaPerturbation(
                enabled=True,
                amplitude=0.0,
                x_center=x_center,
                width=0.05,
                min_area=0.01,
            ),
            SinusoidalAreaForcing(amplitude=0.2, frequency_hz=100.0, phase=0.0),
        )
    except ValueError:
        dynamic_invalid_raised = True

    checks = {
        "q=0 area recovers baseline": zero_area_ok,
        "q=0 gradient recovers baseline": zero_grad_ok,
        "positive q increases throat area": positive_ok,
        "negative q decreases throat area above min_area": negative_ok,
        "invalid negative area raises ValueError": invalid_raised,
        "analytical gradient matches finite difference": grad_ok,
        "disabled perturbation recovers baseline": disabled_ok,
        "sinusoidal q(t) matches analytic value": forcing_ok,
        "sinusoidal q(t) with mean offset matches analytic": forcing_mean_ok,
        "epsilon=0 dynamic geometry recovers baseline": dyn_zero_ok,
        "frequency=0 dynamic forcing is static offset": static_forcing_ok,
        "dynamic area positivity is enforced": dynamic_invalid_raised,
    }

    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    passed = all(checks.values())
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_reduced_fidelity_extensions():
    """
    Passive-scalar, simple heat-release, and turbulence flag sanity tests.

    These are reduced-fidelity controls only. They do not validate realistic
    combustion or turbulence closure.
    """
    print("\n" + "=" * 60)
    print("TEST 0B: Reduced-Fidelity Extension Controls")
    print("=" * 60)

    cfg_cold = SolverConfig()
    cfg_cold.mesh.nx = 8
    cfg_cold.mesh.ny = 2
    cfg_cold.n_steps = 1
    cfg_cold.print_interval = 10
    solver_cold = Solver(cfg_cold)
    cold_ok = solver_cold.simple_heat_release is None and solver_cold.combustion is None

    cfg_passive_zero = SolverConfig()
    cfg_passive_zero.inlet = InletConfig(Yf_inlet=0.0)
    cfg_passive_zero.mesh.nx = 8
    cfg_passive_zero.mesh.ny = 2
    cfg_passive_zero.n_steps = 3
    cfg_passive_zero.print_interval = 10
    cfg_passive_zero.passive_scalar_enabled = True
    cfg_passive_zero.area_source = False
    solver_passive_zero = Solver(cfg_passive_zero)
    solver_passive_zero.run()
    diag_zero = scalar_diagnostics(solver_passive_zero)
    no_fuel_created_ok = abs(diag_zero["integrated_fuel_scalar"]) < 1.0e-14

    cfg_passive = SolverConfig()
    cfg_passive.inlet = InletConfig(Yf_inlet=0.1)
    cfg_passive.mesh.nx = 8
    cfg_passive.mesh.ny = 2
    cfg_passive.n_steps = 3
    cfg_passive.print_interval = 10
    cfg_passive.passive_scalar_enabled = True
    cfg_passive.area_source = False
    solver_passive = Solver(cfg_passive)
    solver_passive.run()
    diag_passive = scalar_diagnostics(solver_passive)
    scalar_bounded_ok = diag_passive["fuel_scalar_bounded"]

    U = np.zeros((5, 1, 1))
    U[0, 0, 0] = 1.0
    U[3, 0, 0] = 250000.0
    U[4, 0, 0] = 0.5
    heat_on = SimpleHeatRelease(heat_rate=1000.0, fuel_coupled=True).compute(U)
    heat_off = SimpleHeatRelease(heat_rate=0.0, fuel_coupled=True).compute(U)
    heat_on_ok = heat_on[3, 0, 0] > 0.0
    heat_off_ok = np.allclose(heat_off, 0.0)

    cfg_simple = SolverConfig()
    cfg_simple.heat_release_model = "simple"
    cfg_simple.simple_heat_release_rate = 1000.0
    cfg_simple.mesh.nx = 4
    cfg_simple.mesh.ny = 2
    cfg_simple.n_steps = 1
    cfg_simple.print_interval = 10
    simple_solver = Solver(cfg_simple)
    simple_model_ok = simple_solver.simple_heat_release is not None

    cfg_none = SolverConfig()
    cfg_none.turbulence_model = "none"
    turbulence_none_ok = Solver(cfg_none) is not None

    rans_raised = False
    try:
        cfg_rans = SolverConfig()
        cfg_rans.turbulence_model = "rans"
        Solver(cfg_rans)
    except NotImplementedError:
        rans_raised = True

    les_raised = False
    try:
        cfg_les = SolverConfig()
        cfg_les.turbulence_model = "les"
        Solver(cfg_les)
    except NotImplementedError:
        les_raised = True

    checks = {
        "cold-flow default has no heat or combustion source": cold_ok,
        "passive scalar with zero inlet creates no fuel": no_fuel_created_ok,
        "passive scalar remains bounded": scalar_bounded_ok,
        "simple heat release adds energy": heat_on_ok,
        "turning simple heat release off removes source": heat_off_ok,
        "simple heat-release config constructs source": simple_model_ok,
        "turbulence_model=none works": turbulence_none_ok,
        "turbulence_model=rans raises NotImplementedError": rans_raised,
        "turbulence_model=les raises NotImplementedError": les_raised,
    }

    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    passed = all(checks.values())
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_response_metrics():
    """
    Synthetic-signal validation for response_metrics.extract_response_metrics.

    Covers: clean amplitude/phase recovery, transient cutoff behavior,
    insufficient-cycle phase-lag rejection, flat-response phase-lag
    rejection, epsilon=0 reporting, and probe pressure aggregation.

    These checks do not validate a CFD solution; they validate the metric
    extractor itself.
    """
    print("\n" + "=" * 60)
    print("TEST 0C: Response Metrics (synthetic signals)")
    print("=" * 60)

    rng = np.random.default_rng(seed=12345)

    def build_rows(t, q, qoi_keys_values, probe_pressures=None):
        forcing_rows = [{"time": float(ti), "q": float(qi)}
                        for ti, qi in zip(t, q)]
        qoi_rows = []
        for k, ti in enumerate(t):
            row = {"time": float(ti)}
            for key, series in qoi_keys_values.items():
                row[key] = float(series[k])
            qoi_rows.append(row)
        probe_rows = None
        if probe_pressures is not None:
            probe_rows = []
            for k, ti in enumerate(t):
                row = {"time": float(ti)}
                for name, series in probe_pressures.items():
                    row[f"{name}_pressure"] = float(series[k])
                probe_rows.append(row)
        return forcing_rows, qoi_rows, probe_rows

    # 1. Clean sinusoid with a known phase lag.
    freq = 200.0
    omega = 2.0 * np.pi * freq
    n_samples = 600
    t_clean = np.linspace(0.0, 4.0 / freq, n_samples)
    forcing_amp = 0.02
    forcing_mean = -0.01
    forcing_phase = 0.0
    q_clean = forcing_mean + forcing_amp * np.sin(omega * t_clean + forcing_phase)
    qoi_amp = 1.5
    qoi_mean = 5.0
    expected_lag = 0.7  # rad
    response = qoi_mean + qoi_amp * np.sin(omega * t_clean + forcing_phase + expected_lag)
    flat_series = np.full_like(t_clean, 4.2)
    forcing_rows, qoi_rows, probe_rows = build_rows(
        t_clean, q_clean,
        {"exit_mach": response, "mdot": flat_series, "pressure_recovery": response,
         "max_mach": response, "thrust": response},
        probe_pressures={"inlet_side": response * 100.0, "throat": flat_series},
    )
    metrics_clean = extract_response_metrics(
        qoi_rows=qoi_rows, forcing_rows=forcing_rows, probe_rows=probe_rows,
        frequency_hz=freq, discard_fraction=0.25, min_cycles=1.0, min_samples=8,
    )
    forcing_amp_ok = abs(metrics_clean["forcing"]["amplitude"] - forcing_amp) < 1e-6
    forcing_mean_ok = abs(metrics_clean["forcing"]["mean"] - forcing_mean) < 1e-6
    response_amp_ok = abs(metrics_clean["qoi"]["exit_mach"]["amplitude"] - qoi_amp) < 1e-6
    response_mean_ok = abs(metrics_clean["qoi"]["exit_mach"]["mean"] - qoi_mean) < 1e-6
    response_phase_ok = abs(metrics_clean["qoi"]["exit_mach"]["phase_lag_vs_q_rad"]
                            - expected_lag) < 1e-3

    # Probe pressure amplitude reflects scaling.
    probe_amp_ok = abs(metrics_clean["probes"]["inlet_side"]["pressure_amplitude"]
                       - qoi_amp * 100.0) < 1e-4

    # Flat QoI must report null phase lag with a warning.
    flat_qoi_warning = metrics_clean["qoi"]["mdot"]["warning"]
    flat_qoi_null = metrics_clean["qoi"]["mdot"]["phase_lag_vs_q_rad"] is None
    flat_qoi_ok = flat_qoi_null and "flat" in flat_qoi_warning.lower()

    flat_probe_warning = metrics_clean["probes"]["throat"]["warning"]
    flat_probe_ok = (metrics_clean["probes"]["throat"]["pressure_phase_lag_vs_q_rad"] is None
                     and "flat" in flat_probe_warning.lower())

    # 2. Transient cutoff: corrupt the first 25% with a wild offset and check
    #    that the post-transient amplitude is still correct.
    contaminated = response.copy()
    n_corrupt = n_samples // 4
    contaminated[:n_corrupt] += 30.0 * rng.standard_normal(n_corrupt)
    forcing_rows_c, qoi_rows_c, _ = build_rows(
        t_clean, q_clean, {"exit_mach": contaminated},
    )
    metrics_transient = extract_response_metrics(
        qoi_rows=qoi_rows_c, forcing_rows=forcing_rows_c,
        frequency_hz=freq, discard_fraction=0.25, min_cycles=1.0, min_samples=8,
        qoi_keys=("exit_mach",),
    )
    transient_amp_ok = abs(metrics_transient["qoi"]["exit_mach"]["amplitude"] - qoi_amp) < 5e-2
    transient_phase_ok = abs(metrics_transient["qoi"]["exit_mach"]["phase_lag_vs_q_rad"]
                             - expected_lag) < 5e-2

    # 3. Insufficient cycles: only ~0.5 cycles of forcing, min_cycles = 1.0.
    n_short = 32
    t_short = np.linspace(0.0, 0.5 / freq, n_short)
    q_short = forcing_amp * np.sin(omega * t_short)
    y_short = qoi_amp * np.sin(omega * t_short + expected_lag) + qoi_mean
    forcing_rows_s, qoi_rows_s, _ = build_rows(
        t_short, q_short, {"exit_mach": y_short},
    )
    metrics_short = extract_response_metrics(
        qoi_rows=qoi_rows_s, forcing_rows=forcing_rows_s,
        frequency_hz=freq, discard_fraction=0.25, min_cycles=1.0, min_samples=8,
        qoi_keys=("exit_mach",),
    )
    insufficient_cycle_null = (
        metrics_short["qoi"]["exit_mach"]["phase_lag_vs_q_rad"] is None
    )
    insufficient_cycle_warning_ok = any(
        "insufficient cycles" in w.lower() for w in metrics_short["warnings"]
    )

    # 4. epsilon=0 (zero forcing): mean is preserved but phase lag must be null.
    n_zero = 200
    t_zero = np.linspace(0.0, 2.0 / freq, n_zero)
    q_zero = np.zeros_like(t_zero)
    y_zero = qoi_mean + 0.0 * t_zero
    forcing_rows_z, qoi_rows_z, _ = build_rows(
        t_zero, q_zero, {"exit_mach": y_zero},
    )
    metrics_zero = extract_response_metrics(
        qoi_rows=qoi_rows_z, forcing_rows=forcing_rows_z,
        frequency_hz=freq, discard_fraction=0.25, min_cycles=1.0, min_samples=8,
        qoi_keys=("exit_mach",),
    )
    zero_mean_ok = abs(metrics_zero["forcing"]["mean"] - 0.0) < 1e-12
    zero_amp_small = metrics_zero["forcing"]["amplitude"] < 1e-12
    zero_phase_null = metrics_zero["qoi"]["exit_mach"]["phase_lag_vs_q_rad"] is None

    # 5. Convenience utility round-trips.
    mean_fit, amp_fit, phase_fit = fit_sinusoid(t_clean, response, freq)
    fit_ok = (abs(mean_fit - qoi_mean) < 1e-6
              and abs(amp_fit - qoi_amp) < 1e-6
              and abs(phase_fit - (forcing_phase + expected_lag)) < 1e-3)

    checks = {
        "clean signal recovers forcing amplitude": forcing_amp_ok,
        "clean signal recovers forcing mean": forcing_mean_ok,
        "clean signal recovers response amplitude": response_amp_ok,
        "clean signal recovers response mean": response_mean_ok,
        "clean signal recovers response phase lag": response_phase_ok,
        "probe amplitude reflects scaling": probe_amp_ok,
        "flat QoI phase lag is null with warning": flat_qoi_ok,
        "flat probe phase lag is null with warning": flat_probe_ok,
        "transient is removed: amplitude still correct": transient_amp_ok,
        "transient is removed: phase still correct": transient_phase_ok,
        "insufficient cycles yields null phase lag": insufficient_cycle_null,
        "insufficient cycles emits warning": insufficient_cycle_warning_ok,
        "epsilon=0 mean preserved": zero_mean_ok,
        "epsilon=0 amplitude is ~zero": zero_amp_small,
        "epsilon=0 phase lag is null": zero_phase_null,
        "fit_sinusoid utility round-trips": fit_ok,
    }
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    passed = all(checks.values())
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_sod_shock_tube():
    """
    Sod shock tube problem (Sod 1978).

    IC:  left state  (x < 0.5): rho=1.0, p=1.0, u=0
         right state (x > 0.5): rho=0.125, p=0.1, u=0

    Exact solution at t=0.2 contains:
        - left rarefaction fan
        - contact discontinuity
        - right-moving shock

    This tests the FVM spatial discretisation (HLLC + MUSCL) and
    temporal integration (RK3-SSP) in 1D (ny=1 limit of the 2D solver).

    Pass criteria (nx=200, t=0.2):
        L1 error in rho < 0.020
        L1 error in u   < 0.035
        L1 error in p   < 0.015
    """
    print("=" * 60)
    print("TEST 1: Sod Shock Tube")
    print("=" * 60)

    gamma = 1.4
    nx = 200
    ny = 1
    x_min, x_max = 0.0, 1.0
    t_final = 0.2

    mesh = StructuredMesh2D.uniform(x_min, x_max, 0.0, 0.01, nx, ny)
    state = StateVector(nx, ny, gamma=gamma)

    # initial conditions
    rho = np.ones((nx, ny))
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    p = np.ones((nx, ny))

    x_mid = 0.5
    for i in range(nx):
        if mesh.xc[i] > x_mid:
            rho[i, :] = 0.125
            p[i, :] = 0.1

    state.set_primitive(rho, u, v, p)

    # boundary conditions: both sides extrapolation (outflow)
    # use left-state inflow but it won't matter because flow doesn't reach boundary
    bc = BoundaryConditions(state, rho[0, 0], 0.0, 0.0, p[0, 0])

    fvm_res = FVMResidual(mesh, gamma=gamma, eps2_scale=3.0)
    integrator = TimeIntegrator(cfl=0.5)

    # time march
    t = 0.0
    n = 0
    while t < t_final:
        dt = integrator.compute_dt(state, mesh)
        dt = min(dt, t_final - t)

        integrator.step(state, dt, fvm_res.compute, bc.apply)
        t += dt
        n += 1

    print(f"  Completed {n} steps, t_final = {t:.6f}")

    # exact solution (Toro, ch. 4)
    rho_ex, u_ex, p_ex = _sod_exact(mesh.xc, t_final, gamma)

    rho_num, u_num, v_num, p_num, T_num, Yf_num = state.primitives()

    # L1 errors
    err_rho = np.mean(np.abs(rho_num[:, 0] - rho_ex))
    err_u = np.mean(np.abs(u_num[:, 0] - u_ex))
    err_p = np.mean(np.abs(p_num[:, 0] - p_ex))

    print(f"  L1 errors: rho={err_rho:.4f}, u={err_u:.4f}, p={err_p:.4f}")

    pass_rho = err_rho < 0.020
    pass_u = err_u < 0.035
    pass_p = err_p < 0.015
    passed = pass_rho and pass_u and pass_p

    print(f"  rho: {'PASS' if pass_rho else 'FAIL'} (threshold 0.020)")
    print(f"  u:   {'PASS' if pass_u else 'FAIL'} (threshold 0.035)")
    print(f"  p:   {'PASS' if pass_p else 'FAIL'} (threshold 0.015)")

    # save plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = mesh.xc

    axes[0].plot(x, rho_ex, "k-", lw=1.5, label="Exact")
    axes[0].plot(x, rho_num[:, 0], "ro", ms=2, label="FVM")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].plot(x, u_ex, "k-", lw=1.5)
    axes[1].plot(x, u_num[:, 0], "ro", ms=2)
    axes[1].set_ylabel("Velocity")

    axes[2].plot(x, p_ex, "k-", lw=1.5)
    axes[2].plot(x, p_num[:, 0], "ro", ms=2)
    axes[2].set_ylabel("Pressure")

    for ax in axes:
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Sod Shock Tube (t = {t_final}, nx = {nx})")
    plt.tight_layout()
    fig.savefig("test_sod.png", dpi=150)
    print(f"  Plot saved: test_sod.png")

    return passed


def _sod_exact(x, t, gamma):
    """
    Exact solution to the Sod shock tube problem at time t.

    Uses the analytical Riemann solution (Toro, Riemann Solvers, ch. 4).
    Left state: (rho_L, u_L, p_L) = (1.0, 0.0, 1.0)
    Right state: (rho_R, u_R, p_R) = (0.125, 0.0, 0.1)
    """
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    # left and right states
    rho_L, u_L, p_L = 1.0, 0.0, 1.0
    rho_R, u_R, p_R = 0.125, 0.0, 0.1
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)

    # star region (iterative solve for p_star)
    # Newton iteration on the pressure function
    p_star = 0.30313  # known exact value for standard Sod
    u_star = 0.92745

    # post-shock density (right side)
    rho_star_R = rho_R * ((p_star / p_R + gm1 / gp1) / (gm1 / gp1 * p_star / p_R + 1.0))

    # post-expansion density (left side)
    rho_star_L = rho_L * (p_star / p_L) ** (1.0 / gamma)

    # sound speed in star-left region
    c_star_L = c_L * (p_star / p_L) ** (gm1 / (2.0 * gamma))

    # wave speeds
    # rarefaction head: x/t = u_L - c_L
    # rarefaction tail: x/t = u_star - c_star_L
    # contact: x/t = u_star
    # shock: x/t = u_R + c_R * sqrt((gp1/(2*gamma)) * (p_star/p_R) + gm1/(2*gamma))
    x_head = 0.5 + (u_L - c_L) * t
    x_tail = 0.5 + (u_star - c_star_L) * t
    x_contact = 0.5 + u_star * t
    S_shock = u_R + c_R * np.sqrt(gp1 / (2.0 * gamma) * p_star / p_R + gm1 / (2.0 * gamma))
    x_shock = 0.5 + S_shock * t

    rho_out = np.empty_like(x)
    u_out = np.empty_like(x)
    p_out = np.empty_like(x)

    for i in range(len(x)):
        xi = x[i]
        if xi < x_head:
            # undisturbed left
            rho_out[i] = rho_L
            u_out[i] = u_L
            p_out[i] = p_L
        elif xi < x_tail:
            # inside rarefaction fan (Toro eq 4.56)
            # self-similar variable: xi_s = (x - x0) / t
            xi_s = (xi - 0.5) / t
            # u = (2/(gamma+1)) * (c_L + (gamma-1)/2 * u_L + xi_s)
            u_out[i] = 2.0 / gp1 * (c_L + gm1 / 2.0 * u_L + xi_s)
            # c = (2/(gamma+1)) * (c_L + (gamma-1)/2 * (u_L - xi_s))
            c_fan = 2.0 / gp1 * (c_L + gm1 / 2.0 * (u_L - xi_s))
            rho_out[i] = rho_L * (c_fan / c_L) ** (2.0 / gm1)
            p_out[i] = p_L * (c_fan / c_L) ** (2.0 * gamma / gm1)
        elif xi < x_contact:
            # star-left region
            rho_out[i] = rho_star_L
            u_out[i] = u_star
            p_out[i] = p_star
        elif xi < x_shock:
            # star-right region
            rho_out[i] = rho_star_R
            u_out[i] = u_star
            p_out[i] = p_star
        else:
            # undisturbed right
            rho_out[i] = rho_R
            u_out[i] = u_R
            p_out[i] = p_R

    return rho_out, u_out, p_out


def test_nozzle_area_mach():
    """
    Steady quasi-1D nozzle flow against the exact isentropic area–Mach
    relation (Mach 6, default three-section duct, ny = 1).

    This is the test that pins the variable-area coupling — the physics
    the contraction/unstart and wall-motion studies rest on. A supersonic
    stream must DEcelerate and compress through the converging inlet and
    REaccelerate through the diverging sections.

    Checks:
        - centerline M(x) matches the exact isentropic solution
        - mass conservation: rho*u*A equal at inlet/exit stations
        - total pressure conserved (smooth flow, no entropy generation)
        - solution is steady (no reconstruction limit cycle)
        - the standalone source-vector form (physics.VariableAreaSource)
          agrees with the primary area-weighted conservative path

    Pass criteria (nx=200):
        mean |M - M_isentropic| / M < 0.5%, max < 2%
        |mdot_exit/mdot_inlet - 1| < 1%
        p0 spread < 1%
        unsteadiness (300-step rho fluctuation) < 1e-10
        paths agree on exit Mach within 1%
    """
    print("\n" + "=" * 60)
    print("TEST 2: Quasi-1D Nozzle vs Isentropic Area-Mach")
    print("=" * 60)

    nx, n_steps = 200, 8000
    gamma = 1.4

    def build_solver(area_source):
        cfg = SolverConfig()
        cfg.inlet = InletConfig(mach=6.0, altitude=25000.0)
        cfg.mesh.nx = nx
        cfg.mesh.ny = 1
        cfg.cfl = 0.35
        cfg.print_interval = 0
        cfg.combustion = CombustionConfig(enabled=False)
        cfg.area_source = area_source
        return Solver(cfg)

    # --- primary path: area-weighted conservative coupling ---
    solver = build_solver(area_source=True)
    for _ in range(n_steps):
        solver.advance_one_step()

    rho, u, v, p, T, Yf = solver.state.primitives()
    M = solver.state.mach()[:, 0]
    x = solver.mesh.xc
    A = solver.cfg.geometry.area(x)

    A_star = A[0] / _isentropic_area_ratio(M[0], gamma)
    M_isen = np.array([_mach_from_area_ratio(a / A_star, True, gamma) for a in A])

    interior = slice(2, nx - 2)
    err = np.abs(M[interior] - M_isen[interior]) / M_isen[interior]

    mdot = rho[:, 0] * u[:, 0] * A
    i_in = np.argmin(np.abs(x - 0.1))     # smooth stations away from
    i_ex = np.argmin(np.abs(x - 1.1))     # boundaries and the throat kink
    mdot_err = abs(mdot[i_ex] / mdot[i_in] - 1.0)

    p0 = p[:, 0] * (1.0 + 0.5 * (gamma - 1.0) * M**2) ** (gamma / (gamma - 1.0))
    core = slice(5, nx - 5)
    p0_spread = (p0[core].max() - p0[core].min()) / p0[5]

    hist = []
    for _ in range(300):
        solver.advance_one_step()
        hist.append(solver.state.U[0, :, 0].copy())
    hist = np.array(hist)
    unsteadiness = float((hist.std(axis=0) / hist.mean(axis=0)).max())

    # --- reference path: standalone corrected source vector ---
    solver_src = build_solver(area_source=False)
    src = VariableAreaSource(solver_src.mesh, solver.cfg.geometry.copy())
    base_rhs = solver_src._rhs

    def rhs_with_source(U, time=None):
        return base_rhs(U, time) + src.compute(U, gamma)

    solver_src._rhs = rhs_with_source
    for _ in range(n_steps):
        solver_src.advance_one_step()
    M_src = solver_src.state.mach()[:, 0]
    path_agreement = abs(M_src[-3] / M[-3] - 1.0)

    i_throat = int(np.argmin(np.abs(x - solver.cfg.geometry.x_throat)))
    print(f"  M: inlet {M[0]:.3f} -> throat {M[i_throat]:.3f} -> exit {M[-3]:.3f}")
    print(f"  isentropic:  {M_isen[0]:.3f} -> {M_isen[i_throat]:.3f} -> {M_isen[-3]:.3f}")
    print(f"  Mach error: mean {err.mean()*100:.3f}%, max {err.max()*100:.3f}%")
    print(f"  mass conservation |mdot_ex/mdot_in - 1|: {mdot_err*100:.3f}%")
    print(f"  total-pressure spread: {p0_spread*100:.3f}%")
    print(f"  unsteadiness: {unsteadiness:.2e}")
    print(f"  source-vector path exit-Mach agreement: {path_agreement*100:.3f}%")

    checks = {
        "mean Mach error < 0.5%": err.mean() < 0.005,
        "max Mach error < 2%": err.max() < 0.02,
        "mass conserved inlet->exit < 1%": mdot_err < 0.01,
        "total pressure spread < 1%": p0_spread < 0.01,
        "steady (no limit cycle) < 1e-10": unsteadiness < 1e-10,
        "source-vector path agrees < 1%": path_agreement < 0.01,
    }
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    passed = all(checks.values())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(x, M_isen, "k-", lw=2, label="exact isentropic")
    axes[0].plot(x, M, "b--", lw=1.5, label="solver (conservative)")
    axes[0].plot(x, M_src, "r:", lw=1.5, label="solver (source form)")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("Mach")
    axes[0].set_title("Quasi-1D nozzle, M$_\\infty$=6")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, mdot / mdot[i_in], "b-", lw=1.5)
    axes[1].axhline(1.0, color="k", lw=0.8)
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("$\\dot m(x)/\\dot m_{in}$")
    axes[1].set_title("Mass conservation")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig("test_nozzle.png", dpi=150)
    print("  Plot saved: test_nozzle.png")

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_shock_position():
    """
    Steady normal shock in a diverging duct against the exact solution.

    A linear diffuser (A: 0.05 -> 0.10 over 1 m) with a Mach 2 supersonic
    inlet and an imposed exit back pressure supports a normal shock whose
    position follows from isentropic + normal-shock relations. The solver
    is initialized with the shock deliberately misplaced (x = 0.42) and
    must migrate it to the analytic location (x = 0.50) selected by the
    imposed back pressure.

    Validates: the back-pressure outlet BC, shock capturing inside the
    quasi-1D area coupling, and diagnostics.shock_diagnostics (which the
    shock_x QoI uses).

    Pass criteria (nx=100):
        |x_shock - 0.5| <= 0.03 m (3 cells)
        measured p0 ratio across shock within 0.02 of analytic
        realized exit pressure within 2% of imposed
    """
    print("\n" + "=" * 60)
    print("TEST 3: Analytic Shock Position (back-pressure BC)")
    print("=" * 60)

    gamma = 1.4
    R_gas = 287.0
    A_in, A_ex, L = 0.05, 0.10, 1.0
    M1, p1, T1 = 2.0, 20000.0, 300.0
    x_target, x_init = 0.5, 0.42
    nx, n_steps = 100, 15000

    p01 = p1 * (1.0 + 0.2 * M1**2) ** 3.5
    T01 = T1 * (1.0 + 0.2 * M1**2)
    A_star1 = A_in / _isentropic_area_ratio(M1, gamma)

    def shock_solution(x_s):
        """Exit pressure and post-shock references for a shock at x_s."""
        A_s = A_in + (A_ex - A_in) * x_s / L
        M_su = _mach_from_area_ratio(A_s / A_star1, True, gamma)
        M_sd, r_p0 = _normal_shock(M_su, gamma)
        p02 = p01 * r_p0
        A_star2 = A_s / _isentropic_area_ratio(M_sd, gamma)
        M_e = _mach_from_area_ratio(A_ex / A_star2, False, gamma)
        p_e = p02 * (1.0 + 0.2 * M_e**2) ** -3.5
        return p_e, r_p0, A_star2, p02, M_e

    p_e, r_p0_exact, _, _, M_e_exact = shock_solution(x_target)
    _, _, A_star2_0, p02_0, _ = shock_solution(x_init)

    x_samp = np.linspace(0.0, L, 60)
    geom = TabulatedAreaProfile(x_samp, A_in + (A_ex - A_in) * x_samp / L,
                                name="linear_diffuser")

    cfg = SolverConfig()
    cfg.inlet = InletConfig(mach=M1, T_inf=T1, p_inf=p1)
    cfg.geometry = geom
    cfg.mesh.nx = nx
    cfg.mesh.ny = 1
    cfg.cfl = 0.35
    cfg.print_interval = 0
    cfg.combustion = CombustionConfig(enabled=False)
    cfg.outlet_type = "back_pressure"
    cfg.outlet_p_back = p_e
    solver = Solver(cfg)

    # initialize with the shock misplaced at x_init
    xc = solver.mesh.xc
    A = geom.area(xc)
    rho0 = np.empty(nx); u0 = np.empty(nx); p0_arr = np.empty(nx)
    for i in range(nx):
        if xc[i] < x_init:
            M_i = _mach_from_area_ratio(A[i] / A_star1, True, gamma)
            p_tot = p01
        else:
            M_i = _mach_from_area_ratio(A[i] / A_star2_0, False, gamma)
            p_tot = p02_0
        p_i = p_tot * (1.0 + 0.2 * M_i**2) ** -3.5
        T_i = T01 / (1.0 + 0.2 * M_i**2)
        rho0[i] = p_i / (R_gas * T_i)
        u0[i] = M_i * np.sqrt(gamma * R_gas * T_i)
        p0_arr[i] = p_i
    ones = np.ones((nx, 1))
    solver.state.set_primitive(rho0[:, None] * ones, u0[:, None] * ones,
                               np.zeros((nx, 1)), p0_arr[:, None] * ones)

    for _ in range(n_steps):
        solver.advance_one_step()

    diag = shock_diagnostics(solver)
    rho, u, v, p, T, Yf = solver.state.primitives()
    p_exit = float(np.mean(p[-2, :]))
    M_exit = float(solver.state.mach()[-2, 0])

    x_err = abs(diag["shock_x"] - x_target)
    p0_ratio_err = abs(diag["shock_p0_ratio"] - r_p0_exact)
    p_exit_err = abs(p_exit / p_e - 1.0)

    print(f"  analytic: shock at x={x_target}, p0 ratio {r_p0_exact:.4f}, "
          f"p_exit {p_e:.0f} Pa, M_exit {M_e_exact:.3f}")
    print(f"  solver:   shock at x={diag['shock_x']:.4f} (started at {x_init}), "
          f"p0 ratio {diag['shock_p0_ratio']:.4f}, "
          f"p_exit {p_exit:.0f} Pa, M_exit {M_exit:.3f}")

    checks = {
        "shock detected": diag["shock_detected"],
        "shock position within 3 cells": x_err <= 0.03,
        "shock p0 ratio within 0.02": p0_ratio_err <= 0.02,
        "exit pressure within 2%": p_exit_err <= 0.02,
    }
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    passed = all(checks.values())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    M_line = solver.state.mach()[:, 0]
    axes[0].plot(xc, M_line, "b-", lw=1.5)
    axes[0].axvline(x_target, color="k", ls="--", lw=1, label="analytic shock")
    axes[0].axvline(x_init, color="gray", ls=":", lw=1, label="initial guess")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("Mach")
    axes[0].set_title("Normal shock in diverging duct (M$_1$=2)")
    axes[0].legend(fontsize=8)
    axes[1].plot(xc, p[:, 0] / p1, "r-", lw=1.5)
    axes[1].axvline(x_target, color="k", ls="--", lw=1)
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("p / p$_1$")
    axes[1].set_title("Static pressure")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig("test_shock.png", dpi=150)
    print("  Plot saved: test_shock.png")

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_busemann_generator():
    """
    Busemann inlet generator self-consistency (Taylor–Maccoll integration).

    Exact checks with no tunable thresholds beyond integration accuracy:
        - the integration terminates exactly on the freestream Mach conoid
          (theta1 = pi - mu(M1)) — the entry surface of a Busemann flow is
          a characteristic of the uniform freestream
        - mass balance between the entry streamtube and exit tube using
          isentropic conical compression + terminal-shock relations
        - the wall contour contracts monotonically
        - theta-beta-M closure of the terminal shock
    """
    print("\n" + "=" * 60)
    print("TEST 0D: Busemann Generator (Taylor-Maccoll)")
    print("=" * 60)

    from busemann import generate_busemann_inlet

    checks = {}
    for M2, d2 in [(2.5, 12.0), (3.0, 15.0), (3.5, 20.0)]:
        res = generate_busemann_inlet(M2=M2, delta2_deg=d2)
        c = res["checks"]
        label = f"M2={M2}, d2={d2}"
        checks[f"{label}: lands on Mach conoid"] = (
            c["mach_conoid_residual_deg"] < 1e-6)
        checks[f"{label}: mass balance closes"] = (
            c["mass_balance_residual"] < 1e-8)
        checks[f"{label}: contour monotone"] = (
            c["contour_monotonic_fraction"] > 0.999)
        checks[f"{label}: shock closure exact"] = (
            c["shock_deflection_residual_deg"] < 1e-9)
        checks[f"{label}: M1 > M2 (compression)"] = res["M1"] > M2
        print(f"  {label}: M1={res['M1']:.3f}, CR={res['contraction_ratio']:.1f}, "
              f"p0rec={res['p0_ratio_overall']:.4f}, "
              f"conoid res={c['mach_conoid_residual_deg']:.1e} deg, "
              f"mass res={c['mass_balance_residual']:.1e}")

    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    passed = all(checks.values())
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_couette_flow():
    """
    Steady-state Couette flow between parallel plates.

    Setup:
        - Bottom wall: no-slip, u=0
        - Top wall:    moving at u=U_wall
        - No pressure gradient
        - Incompressible limit (low Mach)

    Exact solution: u(y) = U_wall * y / H  (linear profile)

    This exercises the ACTUAL physics.FEMViscous operator (implicit
    backward-Euler diffusion with Dirichlet wall velocities) on a real
    conservative state — not a reimplementation of the same math.

    Pass criterion: L2 error in u-profile < 0.05 * U_wall
    """
    print("\n" + "=" * 60)
    print("TEST 4: Couette Flow (FEMViscous operator)")
    print("=" * 60)

    gamma = 1.4
    R_gas = 287.0
    nx = 4
    ny = 30
    H = 0.01      # channel height [m]
    L = 0.02      # channel length [m]
    U_wall = 10.0  # top wall velocity [m/s]

    T_ref = 300.0
    p_ref = 101325.0
    rho_ref = p_ref / (R_gas * T_ref)

    mesh = StructuredMesh2D.uniform(0.0, L, 0.0, H, nx, ny)
    transport = TransportProperties(gamma=gamma, R_gas=R_gas)
    fem = FEMViscous(mesh, transport, gamma=gamma, R_gas=R_gas,
                     wall_u_bottom=0.0, wall_u_top=U_wall)

    mu = transport.viscosity(np.array([T_ref]))[0]
    nu = mu / rho_ref
    tau = H**2 / nu  # diffusion timescale

    dt = 0.5 * tau   # implicit solver: large steps to steady state
    n_steps = 20     # 10 diffusion timescales

    print(f"  nu = {nu:.4e} m^2/s, tau = {tau:.4e} s")
    print(f"  dt = {dt:.4e} s, n_steps = {n_steps}")

    # conservative state at rest
    state = StateVector(nx, ny, gamma=gamma, R_gas=R_gas)
    fill = np.full((nx, ny), 1.0)
    state.set_primitive(rho_ref * fill, 0.0 * fill, 0.0 * fill, p_ref * fill)

    for _ in range(n_steps):
        fem.step(state.U, dt)

    rho, u, v, p, T, Yf = state.primitives()
    u_profile = u[nx // 2, :]

    # exact solution at cell centers
    y = mesh.yc
    u_exact = U_wall * y / H

    err_L2 = np.sqrt(np.mean((u_profile - u_exact)**2))
    err_rel = err_L2 / U_wall

    print(f"  u(y) L2 error: {err_L2:.4e} ({err_rel*100:.2f}% of U_wall)")
    passed = err_rel < 0.05

    print(f"  {'PASS' if passed else 'FAIL'} (threshold 5%)")

    # save plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(u_exact, y * 1000, "k-", lw=2, label="Exact (linear)")
    ax.plot(u_profile, y * 1000, "ro", ms=5, label="FEMViscous")
    ax.set_xlabel("u [m/s]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Couette Flow Validation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig("test_couette.png", dpi=150)
    print(f"  Plot saved: test_couette.png")

    return passed


def test_ignition_delay():
    """
    0-D ignition delay test for the Arrhenius combustion model.

    Setup:
        - Homogeneous mixture at T_init, p_init, Yf_init
        - No flow (u=v=0), no spatial gradients
        - Only the combustion source term is active

    Expected behavior:
        - Temperature and pressure remain nearly constant until ignition
        - At ignition, rapid temperature rise as fuel is consumed
        - Final state approaches adiabatic flame temperature

    This tests the SingleStepArrhenius source term implementation
    and the stability of the chemistry coupling.

    Pass criteria:
        - Fuel is consumed (final Yf < 0.01 * initial Yf)
        - Temperature increases (final T > 1.5 * initial T)
        - Mass is conserved (|rho_final - rho_initial| / rho_initial < 1e-6)
    """
    print("\n" + "=" * 60)
    print("TEST 3: Ignition Delay (Combustion)")
    print("=" * 60)

    gamma = 1.4
    R_gas = 287.0
    nx = 1
    ny = 1

    # initial conditions: hot premixed gas
    T_init = 1500.0   # K (high enough for ignition)
    p_init = 101325.0  # Pa
    Yf_init = 0.05     # 5% fuel mass fraction
    rho_init = p_init / (R_gas * T_init)

    # Arrhenius parameters (tuned for fast ignition at ~1500 K)
    combustion = SingleStepArrhenius(
        A_pre=1.0e8, Ea=40000.0, Q_heat=2.5e6,
        nf=1.0, no=1.0, W_f=0.002,
        gamma=gamma, R_gas=R_gas,
    )

    # state vector (1x1 "mesh")
    U = np.zeros((5, nx, ny))
    E_init = p_init / ((gamma - 1.0) * rho_init)  # no kinetic energy
    U[0, 0, 0] = rho_init
    U[1, 0, 0] = 0.0   # rho*u = 0
    U[2, 0, 0] = 0.0
    U[3, 0, 0] = rho_init * E_init
    U[4, 0, 0] = rho_init * Yf_init

    # time integration with sub-stepping
    # use explicit Euler with small dt for stiff chemistry
    dt = 1.0e-7   # s
    n_steps = 10000
    t_max = dt * n_steps

    T_hist = []
    Yf_hist = []
    t_hist = []

    for n in range(n_steps):
        S = combustion.compute(U)

        # explicit Euler update (0-D, no spatial terms)
        U += dt * S

        # clamp fuel fraction
        rho_curr = U[0, 0, 0]
        Yf_curr = U[4, 0, 0] / max(rho_curr, 1e-30)
        if Yf_curr < 0.0:
            U[4, 0, 0] = 0.0
            Yf_curr = 0.0

        # extract T
        E_curr = U[3, 0, 0] / max(rho_curr, 1e-30)
        p_curr = (gamma - 1.0) * rho_curr * E_curr
        T_curr = max(p_curr / (rho_curr * R_gas), 50.0)

        T_hist.append(T_curr)
        Yf_hist.append(Yf_curr)
        t_hist.append((n + 1) * dt)

    T_final = T_hist[-1]
    Yf_final = Yf_hist[-1]
    rho_final = U[0, 0, 0]

    print(f"  T_init = {T_init:.1f} K, T_final = {T_final:.1f} K")
    print(f"  Yf_init = {Yf_init:.4f}, Yf_final = {Yf_final:.6f}")
    print(f"  rho_init = {rho_init:.4f}, rho_final = {rho_final:.4f}")

    # check criteria
    fuel_consumed = Yf_final < 0.01 * Yf_init
    temp_increased = T_final > 1.5 * T_init
    mass_conserved = abs(rho_final - rho_init) / rho_init < 1e-6

    print(f"  Fuel consumed:  {'PASS' if fuel_consumed else 'FAIL'} "
          f"(Yf_final/Yf_init = {Yf_final/Yf_init:.4e})")
    print(f"  Temp increased: {'PASS' if temp_increased else 'FAIL'} "
          f"(T_final/T_init = {T_final/T_init:.2f})")
    print(f"  Mass conserved: {'PASS' if mass_conserved else 'FAIL'} "
          f"(delta_rho/rho = {abs(rho_final-rho_init)/rho_init:.2e})")

    passed = fuel_consumed and temp_increased and mass_conserved

    # save plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    t_ms = np.array(t_hist) * 1000  # convert to ms

    axes[0].plot(t_ms, T_hist, "r-", lw=2)
    axes[0].set_xlabel("Time [ms]")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title("Ignition Delay — Temperature")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ms, Yf_hist, "b-", lw=2)
    axes[1].set_xlabel("Time [ms]")
    axes[1].set_ylabel("Fuel mass fraction Yf")
    axes[1].set_title("Ignition Delay — Fuel Consumption")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Arrhenius Combustion Validation")
    plt.tight_layout()
    fig.savefig("test_ignition.png", dpi=150)
    print(f"  Plot saved: test_ignition.png")

    return passed


if __name__ == "__main__":
    tests = {
        "geometry": test_area_perturbation,
        "reduced": test_reduced_fidelity_extensions,
        "metrics": test_response_metrics,
        "busemann": test_busemann_generator,
        "sod": test_sod_shock_tube,
        "nozzle": test_nozzle_area_mach,
        "shock": test_shock_position,
        "couette": test_couette_flow,
        "ignition": test_ignition_delay,
    }

    if len(sys.argv) > 1:
        name = sys.argv[1].lower()
        if name in tests:
            result = tests[name]()
            status = "PASSED" if result else "FAILED"
            print(f"\n{'='*60}")
            print(f"  {name.upper()}: {status}")
            print(f"{'='*60}")
        else:
            print(f"Unknown test: {name}")
            print(f"Available: {', '.join(tests.keys())}")
    else:
        # run all
        results = {}
        for name, fn in tests.items():
            results[name] = fn()

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        all_passed = True
        for name, passed in results.items():
            status = "PASSED" if passed else "FAILED"
            print(f"  {name:12s}: {status}")
            if not passed:
                all_passed = False

        print(f"{'='*60}")
        print(f"  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        print(f"{'='*60}")
