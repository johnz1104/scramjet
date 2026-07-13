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
    6. Breathing energy     — validates moving-wall pressure work and the
                              isentropic compression law
    7. Transient diffusion  — validates the dynamic-coefficient scaling
    8. Research workflow    — validates reduced-frequency coordinates,
                              complex response encoding, circular gates,
                              Culick--Rogers response, and hysteresis logic

Usage:
    python tests.py              # run all tests
    python tests.py geometry     # area perturbation only
    python tests.py sod          # Sod shock tube only
    python tests.py nozzle       # isentropic area-Mach only
    python tests.py shock        # analytic shock position only
    python tests.py couette      # Couette flow only
    python tests.py ignition     # ignition delay only
    python tests.py breathing    # moving-wall energy source only
    python tests.py diffusion    # transient diffusion only
    python tests.py config       # config clone/application invariants
    python tests.py research     # research-coordinate/workflow invariants
    python tests.py config_a     # source-reconstructed Config-A invariants

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
    geometry_to_dict,
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
from rom import PODBasis, ReducedSolver, _apply_params, _clone_config


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
    # Delayed response: the estimator convention must return +expected_lag.
    response = qoi_mean + qoi_amp * np.sin(
        omega * t_clean + forcing_phase - expected_lag,
    )
    flat_series = np.full_like(t_clean, 4.2)
    forcing_rows, qoi_rows, probe_rows = build_rows(
        t_clean, q_clean,
        {"exit_mach": response, "mdot_prescribed": flat_series,
         "pressure_recovery": response,
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
    flat_qoi_warning = metrics_clean["qoi"]["mdot_prescribed"]["warning"]
    flat_qoi_null = metrics_clean["qoi"]["mdot_prescribed"]["phase_lag_vs_q_rad"] is None
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
    insufficient_amplitude_null = (
        metrics_short["qoi"]["exit_mach"]["amplitude"] is None
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
              and abs(phase_fit - (forcing_phase - expected_lag)) < 1e-3)

    # 6. Different timestamp arrays and lengths are fit independently.
    t_force_sparse = np.linspace(0.0, 4.0 / freq, 337)
    forcing_sparse = [
        {"time": float(ti),
         "q": float(forcing_mean + forcing_amp * np.sin(omega * ti))}
        for ti in t_force_sparse
    ]
    metrics_mismatch = extract_response_metrics(
        qoi_rows=qoi_rows, forcing_rows=forcing_sparse,
        frequency_hz=freq, discard_fraction=0.25,
        qoi_keys=("exit_mach",),
    )
    own_time_alignment_ok = (
        abs(metrics_mismatch["qoi"]["exit_mach"]["phase_lag_vs_q_rad"]
            - expected_lag) < 1.0e-3
    )
    quality_ok = (
        metrics_clean["qoi"]["exit_mach"]["quality"]["supported"]
        and metrics_clean["qoi"]["exit_mach"]["quality"]["r_squared"] > 0.999
    )

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
        "insufficient cycles also gates amplitude": insufficient_amplitude_null,
        "insufficient cycles emits warning": insufficient_cycle_warning_ok,
        "epsilon=0 mean preserved": zero_mean_ok,
        "epsilon=0 amplitude is ~zero": zero_amp_small,
        "epsilon=0 phase lag is null": zero_phase_null,
        "fit_sinusoid utility round-trips": fit_ok,
        "mismatched histories use their own timestamps": own_time_alignment_ok,
        "fit quality block is populated": quality_ok,
    }
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    passed = all(checks.values())
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_research_workflow_coordinates():
    """Fast invariants for the post-remediation research workflow."""
    print("\n" + "=" * 60)
    print("TEST: Research Workflow Coordinates and Circular Response")
    print("=" * 60)

    from experiments.build_unsteady_response_surrogate import (
        _value_for_target,
        collect_target_vector,
        feature_names_for_rows,
        inverse_target,
        transform_target,
        zero_forcing_response_value,
    )
    from experiments.rank_candidate_cases import score_doe_cases
    from experiments.run_forced_shock_benchmark import (
        culick_rogers_frequency_response,
        culick_rogers_operating_point,
    )
    from experiments.run_hysteresis_sweep import (
        assess_hysteresis,
        build_staircase,
        stage_completion_gate,
    )
    from experiments.run_parametric_unsteady_doe import (
        design_matrix,
        reduced_frequency,
    )

    k_exact = 2.0 * np.pi * 500.0 * 1.2 / 600.0
    k_value = reduced_frequency(500.0, 1.2, 600.0)
    design = design_matrix(
        [0.0], [0.01], [500.0], [0.0],
        length_ref=1.2, velocity_ref=600.0,
    )
    reduced_frequency_ok = (
        abs(k_value - k_exact) < 1.0e-14
        and abs(design[0]["reduced_frequency"] - k_exact) < 1.0e-14
    )

    supported_row = {
        "case_id": "supported", "status": "ok",
        "q_offset": 0.0, "epsilon": 0.01,
        "frequency_hz": 500.0, "reduced_frequency": k_value, "phase": 0.0,
        "exit_mach_amplitude": 2.0,
        "exit_mach_phase_lag_rad": 0.7,
        "exit_mach_supported": True,
        "tpr_mean": 0.9, "mass_defect_mean": 0.01,
    }
    unsupported_row = {
        **supported_row,
        "case_id": "unsupported", "epsilon": 0.0,
        "exit_mach_amplitude": None,
        "exit_mach_phase_lag_rad": None,
        "exit_mach_supported": False,
    }
    real = _value_for_target(supported_row, "exit_mach_response_real")
    imag = _value_for_target(supported_row, "exit_mach_response_imag")
    real_values, real_mask = collect_target_vector(
        [supported_row, unsupported_row], "exit_mach_response_real",
    )
    complex_encoding_ok = (
        abs(real - 2.0 * np.cos(0.7)) < 1.0e-14
        and abs(imag + 2.0 * np.sin(0.7)) < 1.0e-14
        and real_mask.tolist() == [True, False]
        and abs(real_values[0] - real) < 1.0e-14
    )
    reduced_features_ok = (
        "reduced_frequency" in feature_names_for_rows([supported_row])
        and "frequency_hz" not in feature_names_for_rows([supported_row])
    )
    legacy_row = dict(supported_row)
    legacy_row.pop("reduced_frequency")
    legacy_features_ok = "frequency_hz" in feature_names_for_rows([legacy_row])
    log_conditioning_ok = np.allclose(
        inverse_target(transform_target([1.0e-3, 2.0], "log10"), "log10"),
        [1.0e-3, 2.0], rtol=1.0e-14, atol=0.0,
    )
    zero_boundary_ok = (
        zero_forcing_response_value("exit_mach_amplitude", 2.0, 0.0) == 0.0
        and zero_forcing_response_value("exit_mach_response_real", 2.0, 0.0) == 0.0
        and zero_forcing_response_value("exit_mach_mean", 2.0, 0.0) == 2.0
    )

    cr_point = culick_rogers_operating_point(
        0.5, 0.05, 0.10, 1.0, 2.0, 20000.0, 300.0,
    )
    cr_zero = culick_rogers_frequency_response(0.0, cr_point["tau_s"])
    cr_low = culick_rogers_frequency_response(20.0, cr_point["tau_s"])
    cr_high = culick_rogers_frequency_response(400.0, cr_point["tau_s"])
    culick_rogers_ok = (
        cr_point["C_m_per_Pa"] < 0.0 and cr_point["tau_s"] > 0.0
        and np.isclose(cr_point["C_m_per_Pa"], -1.7236066611310142e-5,
                       rtol=1.0e-12, atol=0.0)
        and np.isclose(cr_point["tau_s"], 3.0890012768257473e-3,
                       rtol=1.0e-12, atol=0.0)
        and cr_zero["normalized_gain"] == 1.0
        and cr_zero["phase_lag_rad"] == 0.0
        and cr_low["normalized_gain"] > cr_high["normalized_gain"]
        and 0.0 < cr_low["phase_lag_rad"] < cr_high["phase_lag_rad"] < np.pi / 2.0
    )

    predictions = {
        "supported": {"predicted_exit_mach_phase_lag_rad": "0.6"},
    }
    scored, _ = score_doe_cases(
        [supported_row, {**supported_row, "case_id": "missing_prediction"}],
        weights=None, require_finite_phase=True,
        surrogate_predictions=predictions, require_predicted_phase=True,
    )
    predicted_phase_gate_ok = len(scored) == 1 and scored[0][0]["case_id"] == "supported"

    staircase_a = build_staircase([1.1, 0.9, 1.0])
    staircase_b = build_staircase([1.1, 0.9, 1.0])
    comparison_rows = [
        {"leg": "up", "pressure_factor": 0.9, "classification": "started",
         "shock_x": 0.7, "tpr": 0.9, "status": "ok"},
        {"leg": "down", "pressure_factor": 0.9, "classification": "started",
         "shock_x": 0.705, "tpr": 0.899, "status": "ok"},
    ]
    assessment = assess_hysteresis(
        comparison_rows, shock_position_tolerance_m=0.02,
    )
    hysteresis_logic_ok = (
        staircase_a == staircase_b
        and assessment["classification"] == "single_path_within_resolution"
        and not assessment["physical_hysteresis_validated"]
        and not stage_completion_gate(3, 3, 2.99, 3.0)
        and stage_completion_gate(3, 3, 3.0, 3.0)
    )

    checks = {
        "reduced-frequency formula and design field": reduced_frequency_ok,
        "complex response uses H=A*exp(-i lag) and support gate": complex_encoding_ok,
        "new rows use reduced frequency": reduced_features_ok,
        "legacy schema-v2 rows retain dimensional-frequency fallback": legacy_features_ok,
        "log-amplitude transform round trip": log_conditioning_ok,
        "zero forcing zeros every periodic-response representation": zero_boundary_ok,
        "Culick--Rogers first-order gain/lag limits": culick_rogers_ok,
        "optional predicted-phase ranking gate": predicted_phase_gate_ok,
        "hysteresis sequence and resolution-aware classification deterministic": hysteresis_logic_ok,
    }
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return all(checks.values())


def test_config_a_reconstruction():
    """Source, motion, lineage, and topology gates for reconstructed Config A."""
    print("\n" + "=" * 60)
    print("TEST: Config-A Source Reconstruction and Research Gates")
    print("=" * 60)

    import tempfile
    from pathlib import Path

    from mesh import (
        CONFIG_A_GEOMETRY_LINEAGE_ID,
        config_a_cantilever_mode,
        config_a_normalized_to_raw,
        config_a_raw_to_normalized,
        geometry_from_dict,
        load_config_a_geometry,
    )
    from experiments.build_unsteady_response_surrogate import response_reportability
    from experiments.export_high_fidelity_scaffold import (
        config_a_external_wall_coordinates,
        wall_contour_rows,
    )
    from experiments.gmsh_openfoam_export import (
        write_gmsh_config_a_external,
        write_gmsh_duct,
    )
    from experiments.rank_candidate_cases import require_reportable_response
    from experiments.reconstruct_config_a_geometry import reconstruct_config_a
    from experiments.run_hysteresis_sweep import assess_hysteresis
    from experiments.run_static_wall_sweep import make_config
    from experiments.run_unsteady_area_breathing import make_cold_flow_config

    fresh_a = reconstruct_config_a()
    fresh_b = reconstruct_config_a()
    deterministic = (
        fresh_a == fresh_b
        and fresh_a["artifact_checksum_sha256"]
        == "0dfd7e0afbde75121973fa9ded4371dbd2e0a71720a277dfadc34556b491af95"
    )
    hard_constraints = (
        fresh_a["hard_constraints_within_published_rounding"] is True
        and all(
            (not result["hard"]) or result["within_published_rounding"]
            for result in fresh_a["constraint_residuals"].values()
        )
        and fresh_a["gas_dynamic_design"]["compression_segments"] == 50
        and fresh_a["gas_dynamic_design"]["segment_interpolation"]
        == "piecewise_linear_only"
    )

    geometry = load_config_a_geometry()
    station_ok = all(
        abs(geometry.nominal_stations[name] - expected) < 1.0e-14
        for name, expected in {
            "leading_edge_x_m_solver": 0.0,
            "cowl_x_m_solver": 0.1758,
            "support_x_m_solver": 0.2051,
            "outlet_x_m_solver": 0.2358,
        }.items()
    )
    dense_x = np.linspace(0.0, geometry.L_total, 5000)
    dense_area = geometry.area(dense_x)
    monotone_area = (
        np.all(dense_area > 0.0)
        and np.max(np.diff(dense_area)) <= 1.0e-12
        and dense_area.min() >= geometry.A_samples.min() - 1.0e-14
        and dense_area.max() <= geometry.A_samples.max() + 1.0e-14
    )
    rebuilt = geometry_from_dict(geometry_to_dict(geometry))
    serialization_ok = (
        rebuilt.geometry_lineage_id == CONFIG_A_GEOMETRY_LINEAGE_ID
        and geometry_to_dict(rebuilt) == geometry_to_dict(geometry)
        and np.array_equal(rebuilt.area(dense_x), geometry.area(dense_x))
    )

    mode = config_a_cantilever_mode(geometry, amplitude=0.0)
    support = geometry.nominal_stations["support_x_m_solver"]
    mode_points = np.array([0.0, support, geometry.L_total])
    mode_shape = mode.shape(mode_points, geometry)
    mode_gradient = mode.shape_gradient(np.array([support]), geometry)[0]
    zero_geometry = PerturbedGeometryProfile(geometry, mode)
    raw = config_a_normalized_to_raw(0.04, geometry)
    motion_ok = (
        np.allclose(mode_shape, [1.0, 0.0, 0.0], rtol=0.0, atol=1.0e-14)
        and abs(mode_gradient) < 1.0e-12
        and np.array_equal(zero_geometry.area(dense_x), geometry.area(dense_x))
        and abs(config_a_raw_to_normalized(raw, geometry) - 0.04) < 1.0e-15
        and mode.metadata["dic_calibrated"] is False
        and mode.metadata["positive_displacement"]
        == "downward ramp motion / increased gap"
    )

    preset = "configs/tusq_m585.json"
    auto_config = make_cold_flow_config(
        nx=8, ny=3, n_steps=1, preset=preset, area_law="auto",
    )
    bypass_config = make_cold_flow_config(
        nx=8, ny=3, n_steps=1, preset=preset, area_law="default",
    )
    generic_config = make_cold_flow_config(
        nx=8, ny=3, n_steps=1, area_law="auto",
    )
    rom_config = make_config(
        0.0, nx=8, ny=3, n_steps=1, preset=preset, area_law="auto",
    )
    rom_clone = _clone_config(rom_config)
    _apply_params(rom_clone, {"q_throat": raw})
    selection_ok = (
        auto_config.geometry.geometry_lineage_id == CONFIG_A_GEOMETRY_LINEAGE_ID
        and abs(auto_config.geometry.reduced_frequency_length_ref_m - 0.2111) < 1.0e-14
        and not hasattr(bypass_config.geometry, "geometry_lineage_id")
        and not hasattr(generic_config.geometry, "geometry_lineage_id")
        and rom_clone.geometry.geometry_lineage_id == CONFIG_A_GEOMETRY_LINEAGE_ID
        and rom_clone.geometry.perturbation.mode
        == "model_assumed_cantilever_first_mode"
        and abs(rom_clone.geometry.perturbation.amplitude - raw) < 1.0e-15
    )

    sparse_gate = response_reportability(0.5, 12, 4)
    supported_gate = response_reportability(0.5, 20, 4)
    legacy_rejected = False
    sparse_rejected = False
    try:
        require_reportable_response({
            "complex_responses": {"exit_mach": {"status": "ok"}},
        })
    except ValueError as exc:
        legacy_rejected = "unknown/missing" in str(exc)
    try:
        require_reportable_response({
            "complex_responses": {"exit_mach": sparse_gate},
        })
    except ValueError as exc:
        sparse_rejected = "not reportable" in str(exc)
    reportability_ok = (
        sparse_gate["reportable"] is False
        and "insufficient_supported_cases" in sparse_gate["reportability_reasons"]
        and supported_gate["reportable"] is True
        and legacy_rejected and sparse_rejected
        and require_reportable_response({
            "complex_responses": {"exit_mach": supported_gate},
        }) is supported_gate
    )

    flip_rows = [
        {"leg": "up", "pressure_factor": 1.0, "classification": "started",
         "shock_x": 0.50, "tpr": 0.90, "status": "ok"},
        {"leg": "down", "pressure_factor": 1.0, "classification": "unstarted",
         "shock_x": 0.505, "tpr": 0.899, "status": "ok"},
    ]
    path_rows = [dict(row) for row in flip_rows]
    path_rows[1]["shock_x"] = 0.54
    incomplete_rows = [dict(row) for row in flip_rows]
    incomplete_rows[1]["status"] = "incomplete"
    hysteresis_ok = (
        assess_hysteresis(flip_rows, 0.02)["classification"]
        == "threshold_flip_at_resolution"
        and assess_hysteresis(path_rows, 0.02)["classification"]
        == "numerical_path_dependence_detected"
        and assess_hysteresis(incomplete_rows, 0.02)["classification"]
        == "indeterminate_incomplete"
    )

    base, ramp_x, ramp_y, cowl_x, cowl_y = (
        config_a_external_wall_coordinates(geometry)
    )
    closed_rows = wall_contour_rows(geometry, n_points=len(ramp_x))
    topology_area_ok = (
        np.allclose(-ramp_y, geometry.area(ramp_x), rtol=0.0, atol=1.0e-12)
        and all(row["height_per_unit_depth"] > 0.0 for row in closed_rows)
    )
    legacy_lineage_rejected = False
    try:
        config_a_external_wall_coordinates(GeometryProfile.default())
    except ValueError as exc:
        legacy_lineage_rejected = "legacy schema-v2" in str(exc)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        closed_geo = write_gmsh_duct(
            tmp / "closed.geo", geometry.x_samples, geometry.A_samples,
        )
        external_geo = write_gmsh_config_a_external(
            tmp / "external.geo", ramp_x, ramp_y, cowl_x, cowl_y,
        )
        closed_text = closed_geo.read_text()
        external_text = external_geo.read_text()
        patches_ok = all(
            f'Physical Surface("{name}")' in external_text
            for name in (
                "inlet", "farfield", "rampWall", "cowlWall", "outlet",
                "frontAndBack",
            )
        )
        export_ok = (
            'Physical Surface("bottomWall")' in closed_text
            and patches_ok and "Spline(" not in external_text
            and legacy_lineage_rejected
        )

    smoke_cfg = make_config(
        0.0, nx=30, ny=4, n_steps=1200, cfl=0.25,
        preset=preset, area_law="auto", steady_rtol=2.0e-5,
        steady_check_interval=50,
    )
    smoke_cfg.print_interval = smoke_cfg.n_steps + 1
    smoke_solver = Solver(smoke_cfg)
    from rom import _compute_qoi
    smoke_solver.run(steady_qoi_fn=_compute_qoi)
    smoke_qoi = _compute_qoi(smoke_solver)
    smoke_ok = (
        smoke_solver.converged
        and smoke_qoi["state_admissible"]
        and all(np.isfinite(smoke_qoi[key]) for key in (
            "tpr", "exit_mach", "mdot_exit", "mass_defect",
        ))
        and 0.0 < smoke_qoi["tpr"] <= 1.0
    )

    checks = {
        "deterministic frozen checksum": deterministic,
        "hard source constraints and 50 MOC segments": hard_constraints,
        "solver stations use the published rebase": station_ok,
        "effective area is positive monotone without overshoot": monotone_area,
        "wall geometry serialization is exact": serialization_ok,
        "cantilever normalization and raw conversion": motion_ok,
        "preset auto/default geometry selection": selection_ok,
        "phase reportability and legacy enforcement gates": reportability_ok,
        "resolution-aware hysteresis outcomes": hysteresis_ok,
        "closed/external export topology and lineage gates": topology_area_ok and export_ok,
        "Config-A q=0 solver smoke converges with TPR <= 1": smoke_ok,
    }
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return all(checks.values())


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


class _UniformBreathingGeometry:
    """Spatially uniform A(t) used to isolate moving-volume physics."""

    is_time_dependent = True

    def __init__(self, area0=1.0, area_rate=-0.2):
        self.area0 = float(area0)
        self.area_rate = float(area_rate)
        self.time = 0.0

    def set_time(self, time):
        self.time = float(time)

    def area(self, x):
        x = np.asarray(x, dtype=float)
        return np.full_like(x, self.area0 + self.area_rate * self.time)

    def area_gradient(self, x):
        return np.zeros_like(np.asarray(x, dtype=float))

    def area_time_derivative(self, x):
        return np.full_like(np.asarray(x, dtype=float), self.area_rate)


def test_breathing_energy(legacy_breathing_energy=False):
    """Moving-control-volume pressure work and isentropic compression.

    The first check evaluates the production FVM residual in interior cells.
    The second integrates the standalone source ODE for a uniform static gas;
    this cleanly excludes boundary waves and verifies rho*A, p*A**gamma, and
    T*A**(gamma-1) invariants.
    """
    print("\n" + "=" * 60)
    print("TEST 6: Breathing-Wall Energy Source")
    print("=" * 60)

    gamma = 1.4
    mesh = StructuredMesh2D.uniform(0.0, 1.0, 0.0, 0.1, 8, 2)
    geom = _UniformBreathingGeometry(area0=1.0, area_rate=-0.2)
    state = StateVector(mesh.nx, mesh.ny, gamma=gamma, R_gas=287.0)
    rho0, p0 = 1.1, 1.0
    fill = np.ones((mesh.nx, mesh.ny))
    state.set_primitive(rho0 * fill, 0.0 * fill, 0.0 * fill, p0 * fill)

    residual = FVMResidual(
        mesh, gamma=gamma, geometry=geom,
        legacy_breathing_energy=legacy_breathing_energy,
    )
    rhs = residual.compute(state.U, time=0.0)
    rate = geom.area_rate / geom.area0
    expected = -rate * state.U
    expected[3] -= rate * p0
    interior = (slice(None), slice(2, -2), slice(None))
    discrete_err = float(np.max(np.abs(rhs[interior] - expected[interior])))

    static_geom = GeometryProfile.default()
    corrected_static = FVMResidual(mesh, gamma=gamma, geometry=static_geom)
    legacy_static = FVMResidual(
        mesh, gamma=gamma, geometry=static_geom,
        legacy_breathing_energy=True,
    )
    static_invariance = np.array_equal(
        corrected_static.compute(state.U), legacy_static.compute(state.U),
    )

    # Integrate only the spatially uniform source.  A short compression keeps
    # explicit-RK error negligible while strongly discriminating p~A^-gamma
    # from the legacy p~A^-1 law.
    geom_ode = _UniformBreathingGeometry(area0=1.0, area_rate=-0.4)
    src = VariableAreaSource(
        mesh, geom_ode,
        legacy_breathing_energy=legacy_breathing_energy,
    )
    U = state.U.copy()
    t_end = 0.5
    n_steps = 5000
    dt = t_end / n_steps
    for n in range(n_steps):
        t = n * dt

        def ode(U_stage, stage_time):
            return src.compute(U_stage, gamma, time=stage_time)

        k1 = ode(U, t)
        k2 = ode(U + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = ode(U + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = ode(U + dt * k3, t + dt)
        U += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    A_ratio = geom_ode.area(np.array([0.0]))[0] / geom_ode.area0
    rho = float(U[0, 3, 0])
    p = float((gamma - 1.0) * U[3, 3, 0])
    T_ratio = (p / rho) / (p0 / rho0)
    rho_exact = rho0 * A_ratio**-1.0
    p_exact = p0 * A_ratio**-gamma
    T_exact_ratio = A_ratio**(-(gamma - 1.0))
    rho_err = abs(rho / rho_exact - 1.0)
    p_err = abs(p / p_exact - 1.0)
    T_err = abs(T_ratio / T_exact_ratio - 1.0)

    checks = {
        "discrete full source near machine precision": discrete_err < 2.0e-13,
        "dA/dt=0 path is bitwise invariant": static_invariance,
        "rho follows A^-1": rho_err < 2.0e-9,
        "p follows A^-gamma": p_err < 2.0e-9,
        "T follows A^-(gamma-1)": T_err < 2.0e-9,
    }
    print(f"  discrete max error: {discrete_err:.3e}")
    print(f"  integrated relative errors: rho={rho_err:.3e}, "
          f"p={p_err:.3e}, T={T_err:.3e}")
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return all(checks.values())


def test_config_integrity():
    """Config cloning/application must preserve BCs and geometry identity."""
    print("\n" + "=" * 60)
    print("TEST 6B: Config Clone and Parameter Validation")
    print("=" * 60)
    x = np.linspace(0.0, 0.4, 12)
    area = 0.05 - 0.01 * np.sin(np.pi * x / x[-1])
    cfg = SolverConfig()
    cfg.geometry = TabulatedAreaProfile(x, area, name="roundtrip_table")
    cfg.mesh.nx, cfg.mesh.ny, cfg.mesh.y_stretch = 24, 5, 1.08
    cfg.outlet_type = "back_pressure"
    cfg.outlet_p_back = 42000.0
    cfg.outlet_p_back_amplitude = 0.12
    cfg.outlet_p_back_frequency_hz = 321.0
    cfg.outlet_p_back_phase = 0.7
    cfg.legacy_breathing_energy = False
    cfg.steady_rtol = 2.0e-7

    cloned = _clone_config(cfg)
    top_level_equal = all(
        getattr(cloned, key) == getattr(cfg, key)
        for key in (
            "outlet_type", "outlet_p_back", "outlet_p_back_amplitude",
            "outlet_p_back_frequency_hz", "outlet_p_back_phase",
            "legacy_breathing_energy", "steady_rtol",
        )
    )
    geometry_equal = geometry_to_dict(cloned.geometry) == geometry_to_dict(cfg.geometry)
    independent = cloned.geometry is not cfg.geometry
    from experiments.export_high_fidelity_scaffold import (
        geometry_from_dict,
        match_static_case,
    )
    exported_table = geometry_from_dict(geometry_to_dict(cfg.geometry))
    exporter_table_roundtrip = np.array_equal(
        exported_table.A_samples, cfg.geometry.A_samples,
    )

    forcing = SinusoidalAreaForcing(
        amplitude=0.002, frequency_hz=500.0, phase=0.4, mean=0.001,
    )
    dynamic = TimeDependentPerturbedGeometryProfile(
        cfg.geometry,
        LocalizedAreaPerturbation(
            amplitude=0.0, x_center=cfg.geometry.x_throat,
            width=0.03, min_area=1.0e-4,
        ),
        forcing,
    )
    exported_mean = geometry_from_dict(geometry_to_dict(dynamic))
    x_probe = np.linspace(0.0, cfg.geometry.L_total, 30)
    expected_mean = (cfg.geometry.area(x_probe)
                     + forcing.mean * dynamic.perturbation.shape(
                         x_probe, cfg.geometry,
                     ))
    exporter_dynamic_mean = np.allclose(
        exported_mean.area(x_probe), expected_mean, rtol=0.0, atol=1.0e-14,
    ) and exported_mean.exported_forcing_spec["amplitude"] == forcing.amplitude
    q_mismatch_rejected = False
    try:
        match_static_case({"q0": 0.0, "q1": 0.01}, 0.025,
                          q_match_tol=1.0e-4)
    except ValueError as exc:
        q_mismatch_rejected = "available static q values" in str(exc)

    inlet_only = _clone_config(cfg)
    _apply_params(inlet_only, {"mach": 3.8})
    inlet_preserves_type = isinstance(inlet_only.geometry, TabulatedAreaProfile)
    inlet_preserves_samples = np.array_equal(
        inlet_only.geometry.A_samples, cfg.geometry.A_samples,
    )

    perturbed = _clone_config(cfg)
    _apply_params(perturbed, {"q_throat": 5.0e-4})
    perturb_wraps_table = (
        isinstance(perturbed.geometry, PerturbedGeometryProfile)
        and isinstance(perturbed.geometry.base_geometry, TabulatedAreaProfile)
    )

    section_rejected = False
    try:
        bad = _clone_config(cfg)
        _apply_params(bad, {"A_exit": 0.2})
    except ValueError:
        section_rejected = True

    unknown_rejected = False
    try:
        _apply_params(_clone_config(cfg), {"L_combustor_typo": 0.5})
    except ValueError:
        unknown_rejected = True

    pod = PODBasis()
    reduced = ReducedSolver(
        pod, [{"q_throat": 0.0}], np.zeros((1, 1)), [{"tpr": 1.0}], cfg,
    )
    query_rejected = False
    try:
        reduced.evaluate({"q_throat": 0.0, "A_exit": 0.2})
    except ValueError:
        query_rejected = True

    from diagnostics import physical_exit_index, transverse_average
    stretched = StructuredMesh2D.stretched(0.0, 1.0, 0.0, 1.0, 4, 4,
                                           y_ratio=1.7)
    values = np.array([1.0, 2.0, 4.0, 8.0])
    dy_average_ok = abs(
        transverse_average(values, stretched)
        - float(np.sum(values * stretched.dy) / np.sum(stretched.dy))
    ) < 1.0e-15
    exit_convention_ok = physical_exit_index(stretched) == stretched.nx - 2

    checks = {
        "all outlet/convergence fields round-trip": top_level_equal,
        "tabulated geometry round-trips exactly": geometry_equal,
        "clone geometry is independent": independent,
        "exporter round-trips tabulated samples": exporter_table_roundtrip,
        "exporter emits time-dependent mean + forcing metadata": exporter_dynamic_mean,
        "exporter rejects out-of-tolerance q match": q_mismatch_rejected,
        "inlet-only change preserves geometry type": inlet_preserves_type,
        "inlet-only change preserves samples": inlet_preserves_samples,
        "perturbation wraps existing tabulated base": perturb_wraps_table,
        "three-section key rejected for tabulated base": section_rejected,
        "unknown parameter rejected": unknown_rejected,
        "ROM query requires exact training keys": query_rejected,
        "transverse averages include stretched-grid dy": dy_average_ok,
        "shared exit convention is nx-2": exit_convention_ok,
    }
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return all(checks.values())


class _ConstantTransport:
    """Constant dynamic transport coefficients for scaling tests."""

    def __init__(self, mu=1.0e-5, cp=1004.5, pr=0.72, sc=0.9):
        self.mu = float(mu)
        self.cp = float(cp)
        self.pr = float(pr)
        self.sc = float(sc)

    def viscosity(self, T):
        return np.full_like(np.asarray(T, dtype=float), self.mu)

    def thermal_conductivity(self, T):
        return np.full_like(np.asarray(T, dtype=float), self.mu * self.cp / self.pr)

    def species_diffusivity(self, T, rho):
        return self.mu / (np.asarray(rho, dtype=float) * self.sc)


def _couette_startup_state(mesh, rho, T=300.0, gamma=1.4, R_gas=287.0):
    state = StateVector(mesh.nx, mesh.ny, gamma=gamma, R_gas=R_gas)
    fill = np.ones((mesh.nx, mesh.ny))
    state.set_primitive(rho * fill, 0.0 * fill, 0.0 * fill,
                        rho * R_gas * T * fill)
    return state


def test_transient_diffusion():
    """Transient Couette solution and linear density-timescale scaling."""
    print("\n" + "=" * 60)
    print("TEST 7: Transient Diffusion Scaling")
    print("=" * 60)
    gamma, R_gas = 1.4, 287.0
    nx, ny = 2, 60
    H, L, U_wall = 0.01, 0.01, 10.0
    rho1 = 1.0
    transport = _ConstantTransport(mu=1.0e-5)
    mesh = StructuredMesh2D.uniform(0.0, L, 0.0, H, nx, ny)

    nu = transport.mu / rho1
    tau = H**2 / nu
    t_eval = 0.08 * tau
    n_steps = 160
    dt = t_eval / n_steps
    state = _couette_startup_state(mesh, rho1, gamma=gamma, R_gas=R_gas)
    fem = FEMViscous(mesh, transport, gamma=gamma, R_gas=R_gas,
                     wall_u_bottom=0.0, wall_u_top=U_wall)
    for _ in range(n_steps):
        fem.step(state.U, dt)
    u_num = state.primitives()[1][0]

    y = mesh.yc
    eta = y / H
    u_exact = U_wall * eta
    for mode in range(1, 400):
        u_exact += (2.0 * U_wall / np.pi) * ((-1.0) ** mode / mode) * np.sin(
            mode * np.pi * eta,
        ) * np.exp(-(mode * np.pi)**2 * nu * t_eval / H**2)
    startup_err = float(np.sqrt(np.mean((u_num - u_exact)**2)) / U_wall)

    # At fixed dynamic viscosity, rho=2 at time t must match rho=1 at t/2.
    def run_density(rho, duration):
        st = _couette_startup_state(mesh, rho, gamma=gamma, R_gas=R_gas)
        op = FEMViscous(mesh, transport, gamma=gamma, R_gas=R_gas,
                        wall_u_bottom=0.0, wall_u_top=U_wall)
        dt_local = duration / n_steps
        for _ in range(n_steps):
            op.step(st.U, dt_local)
        return st.primitives()[1][0]

    u_rho1_half = run_density(rho1, 0.5 * t_eval)
    u_rho2_full = run_density(2.0 * rho1, t_eval)
    scaling_err = float(
        np.sqrt(np.mean((u_rho2_full - u_rho1_half)**2)) / U_wall
    )
    checks = {
        "startup Fourier-series L2 error < 2%": startup_err < 0.02,
        "doubling rho doubles timescale": scaling_err < 2.0e-4,
    }
    print(f"  startup relative L2 error: {startup_err:.3e}")
    print(f"  rho-timescale profile mismatch: {scaling_err:.3e}")
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return all(checks.values())


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
        - Fuel remains bounded without test-side clipping
        - Heat release equals Q_heat times the mass of fuel consumed
        - Final temperature approaches T_init + Q_heat*Yf_init/cv

    This tests the SingleStepArrhenius source term implementation
    and the stability of the chemistry coupling.

    Pass criteria:
        - Fuel is consumed and remains in [0, 1]
        - Final temperature is within 2% of the caloric exact value
        - No unphysical high-temperature excursion occurs
        - Mass is conserved
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
    n_steps = 30000
    t_max = dt * n_steps

    T_hist = []
    Yf_hist = []
    t_hist = []

    for n in range(n_steps):
        S = combustion.compute(U, dt=dt)

        # explicit Euler update (0-D, no spatial terms)
        U += dt * S

        rho_curr = U[0, 0, 0]
        Yf_curr = U[4, 0, 0] / max(rho_curr, 1e-30)

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

    cv = R_gas / (gamma - 1.0)
    T_exact = T_init + combustion.Q_heat * Yf_init / cv

    # check criteria
    fuel_consumed = Yf_final < 0.01 * Yf_init
    fuel_bounded = min(Yf_hist) >= -1.0e-12 and max(Yf_hist) <= Yf_init + 1.0e-12
    energy_consistent = abs(T_final / T_exact - 1.0) < 0.02
    sane_temperature = max(T_hist) < 2500.0
    mass_conserved = abs(rho_final - rho_init) / rho_init < 1e-6

    print(f"  Fuel consumed:  {'PASS' if fuel_consumed else 'FAIL'} "
          f"(Yf_final/Yf_init = {Yf_final/Yf_init:.4e})")
    print(f"  Fuel bounded:   {'PASS' if fuel_bounded else 'FAIL'}")
    print(f"  Energy balance: {'PASS' if energy_consistent else 'FAIL'} "
          f"(T_exact = {T_exact:.1f} K)")
    print(f"  Temperature sane: {'PASS' if sane_temperature else 'FAIL'} "
          f"(T_max = {max(T_hist):.1f} K)")
    print(f"  Mass conserved: {'PASS' if mass_conserved else 'FAIL'} "
          f"(delta_rho/rho = {abs(rho_final-rho_init)/rho_init:.2e})")

    passed = (fuel_consumed and fuel_bounded and energy_consistent
              and sane_temperature and mass_conserved)

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
        "research": test_research_workflow_coordinates,
        "config_a": test_config_a_reconstruction,
        "busemann": test_busemann_generator,
        "sod": test_sod_shock_tube,
        "nozzle": test_nozzle_area_mach,
        "shock": test_shock_position,
        "breathing": test_breathing_energy,
        "config": test_config_integrity,
        "diffusion": test_transient_diffusion,
        "couette": test_couette_flow,
        "ignition": test_ignition_delay,
    }

    if len(sys.argv) > 1:
        name = sys.argv[1].lower()
        if name in tests:
            if name == "breathing":
                legacy = any(
                    arg in ("--legacy-breathing-energy",
                            "legacy_breathing_energy=True")
                    for arg in sys.argv[2:]
                )
                result = tests[name](legacy_breathing_energy=legacy)
            else:
                result = tests[name]()
            status = "PASSED" if result else "FAILED"
            print(f"\n{'='*60}")
            print(f"  {name.upper()}: {status}")
            print(f"{'='*60}")
            sys.exit(0 if result else 1)
        else:
            print(f"Unknown test: {name}")
            print(f"Available: {', '.join(tests.keys())}")
            sys.exit(2)
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
        sys.exit(0 if all_passed else 1)
