"""
Forced-shock response benchmark (Culick & Rogers / Sajben lineage).

A normal shock is held in a diverging duct (Mach-2 supersonic inlet, linear
diffuser) by an imposed exit back pressure; the back pressure is then
modulated sinusoidally,

    p_b(t) = p_b0 * (1 + amp * sin(2*pi*f*t)),

and the shock-position response x_s(t) is extracted (amplitude and phase
lag vs the forcing) over a sweep of forcing frequencies. This verifies the
unsteady response-extraction pipeline (response_metrics.py) on a problem
with established references BEFORE any wall-motion claim rests on it:

  - the steady base state is exact (isentropic + normal-shock relations,
    validated by tests.py::test_shock_position), and
  - in the quasi-steady limit f -> 0 the shock-motion amplitude must
    approach  amp * p_b0 / (dp_e/dx_s),  the slope of the exact
    back-pressure-vs-shock-position map. The script reports the measured/
    quasi-steady amplitude ratio per frequency and overlays the first-order
    Culick--Rogers transfer (their Eqs. 42--44),

        x_s' / p_e' = C / (1 + i*omega*tau)

    in the present positive-lag convention.  Their input p_e' is the acoustic
    pressure immediately downstream of the shock, whereas this benchmark
    imposes pressure at the finite-duct outlet.  The plotted exit-forcing curve
    is therefore explicitly labeled a hybrid: exact exit-pressure static gain
    times the published relaxation factor.  Discrepancy is reported, not used
    as a pass/fail assertion.

Outputs (per frequency + aggregate) under runs/:
    summary.csv, response curves, shock-position histories, plots/
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

from gasdynamics import (
    isentropic_area_ratio,
    mach_from_area_ratio,
    normal_shock,
)
from mesh import TabulatedAreaProfile
from solver import CombustionConfig, InletConfig, Solver, SolverConfig
from diagnostics import shock_diagnostics
from response_metrics import extract_response_metrics
from experiments.run_static_wall_sweep import (
    ARTIFACT_SCHEMA_VERSION,
    git_metadata,
    write_csv,
    write_json,
)


CULICK_ROGERS_TRANSCRIPTION_PROVENANCE = {
    "transcription_verified": True,
    "source_record_url": "https://authors.library.caltech.edu/records/xyxx9-h6n68",
    "source_file": "265_Culick_FEC_1981.pdf",
    "source_file_md5": "03fec8618b3d0c512a75acf7b89ff11e",
    "source_file_sha256": "1863479c77563eb6d0d7871186d8c8c4f7be4c05fe9cb8d63ff651da6ca45071",
    "journal_pages": [1387, 1388],
    "pdf_pages_one_based": [6, 7],
    "equations": [43, 44],
    "verification_method": (
        "visual comparison with the attached Caltech scan; variables and "
        "grouping independently checked for C and tau"
    ),
    "verified_utc_date": "2026-07-13",
}


# Benchmark duct: linear diffuser, Mach 2 inlet (same as tests.py shock test)

def make_duct(A_in=0.05, A_ex=0.10, L=1.0, n_samples=60):
    x = np.linspace(0.0, L, n_samples)
    return TabulatedAreaProfile(x, A_in + (A_ex - A_in) * x / L,
                                name="linear_diffuser")


def analytic_map(x_s, A_in, A_ex, L, M1, p1, gamma=1.4):
    """Exact exit pressure p_e for a steady shock at x_s."""
    p01 = p1 * (1.0 + 0.5 * (gamma - 1.0) * M1**2) ** (gamma / (gamma - 1.0))
    A_star1 = A_in / isentropic_area_ratio(M1, gamma)
    A_s = A_in + (A_ex - A_in) * x_s / L
    M_su = mach_from_area_ratio(A_s / A_star1, True, gamma)
    ns = normal_shock(M_su, gamma)
    p02 = p01 * ns["p0_ratio"]
    A_star2 = A_s / isentropic_area_ratio(ns["M2"], gamma)
    M_e = mach_from_area_ratio(A_ex / A_star2, False, gamma)
    p_e = p02 * (1.0 + 0.5 * (gamma - 1.0) * M_e**2) ** (-gamma / (gamma - 1.0))
    return p_e, M_su, ns, A_star2


def initialize_shock(solver, geom, x_s0, A_in, A_ex, L, M1, p1, T1,
                     gamma=1.4, R_gas=287.0):
    """Set the initial field to the exact solution with a shock at x_s0."""
    p01 = p1 * (1.0 + 0.5 * (gamma - 1.0) * M1**2) ** (gamma / (gamma - 1.0))
    T01 = T1 * (1.0 + 0.5 * (gamma - 1.0) * M1**2)
    A_star1 = A_in / isentropic_area_ratio(M1, gamma)
    _, _, ns0, A_star2_0 = analytic_map(x_s0, A_in, A_ex, L, M1, p1, gamma)
    p02_0 = p01 * ns0["p0_ratio"]

    nx = solver.mesh.nx
    xc = solver.mesh.xc
    A = geom.area(xc)
    rho = np.empty(nx); u = np.empty(nx); p = np.empty(nx)
    for i in range(nx):
        if xc[i] < x_s0:
            M_i = mach_from_area_ratio(A[i] / A_star1, True, gamma)
            p_tot = p01
        else:
            M_i = mach_from_area_ratio(A[i] / A_star2_0, False, gamma)
            p_tot = p02_0
        p[i] = p_tot * (1.0 + 0.5 * (gamma - 1.0) * M_i**2) ** (-gamma / (gamma - 1.0))
        T_i = T01 / (1.0 + 0.5 * (gamma - 1.0) * M_i**2)
        rho[i] = p[i] / (R_gas * T_i)
        u[i] = M_i * np.sqrt(gamma * R_gas * T_i)
    ones = np.ones((nx, 1))
    solver.state.set_primitive(rho[:, None] * ones, u[:, None] * ones,
                               np.zeros((nx, 1)), p[:, None] * ones)


def quasi_steady_slope(x_target, A_in, A_ex, L, M1, p1, gamma=1.4, dx=1e-4):
    """dp_e/dx_s at the operating point (exact, by central difference)."""
    pe_p, *_ = analytic_map(x_target + dx, A_in, A_ex, L, M1, p1, gamma)
    pe_m, *_ = analytic_map(x_target - dx, A_in, A_ex, L, M1, p1, gamma)
    return (pe_p - pe_m) / (2.0 * dx)


def culick_rogers_coefficients(M_shock, p_upstream, a_upstream,
                               dln_area_dx, gamma=1.4):
    """Return the published isentropic-flow ``(C, tau)`` coefficients.

    Implements Culick & Rogers (AIAA Journal 21(10), 1983), Eqs. 43--44.
    ``M_shock``, ``p_upstream`` and ``a_upstream`` are mean values immediately
    upstream of the shock, and ``dln_area_dx=(1/A)dA/dx`` is evaluated at the
    mean shock station.  A diverging duct gives ``C < 0`` and ``tau > 0``.
    """
    M_shock = float(M_shock)
    p_upstream = float(p_upstream)
    a_upstream = float(a_upstream)
    dln_area_dx = float(dln_area_dx)
    if M_shock <= 1.0:
        raise ValueError("Culick--Rogers normal-shock model requires M_shock > 1")
    if p_upstream <= 0.0 or a_upstream <= 0.0 or dln_area_dx <= 0.0:
        raise ValueError("positive p, a, and diverging dln(A)/dx are required")
    common = 1.0 + ((gamma**2 + 1.0) / (gamma - 1.0)) * M_shock**2
    length_scale = 1.0 / dln_area_dx
    C = (
        -(1.0 / p_upstream) * length_scale * (gamma + 1.0)**2
        / (2.0 * gamma * (gamma - 1.0) * common)
    )
    tau = (
        (1.0 / a_upstream) * length_scale * 2.0 * (gamma + 1.0) * M_shock
        / ((gamma - 1.0) * common)
    )
    return float(C), float(tau)


def culick_rogers_operating_point(x_target, A_in, A_ex, L, M_inlet,
                                  p_inlet, T_inlet, gamma=1.4,
                                  R_gas=287.0):
    """Evaluate the local mean quantities required by Eqs. 43--44."""
    _, M_shock, _, _ = analytic_map(
        x_target, A_in, A_ex, L, M_inlet, p_inlet, gamma,
    )
    stagnation_factor_inlet = 1.0 + 0.5 * (gamma - 1.0) * M_inlet**2
    p0 = p_inlet * stagnation_factor_inlet ** (gamma / (gamma - 1.0))
    T0 = T_inlet * stagnation_factor_inlet
    local_factor = 1.0 + 0.5 * (gamma - 1.0) * M_shock**2
    p_upstream = p0 * local_factor ** (-gamma / (gamma - 1.0))
    T_upstream = T0 / local_factor
    a_upstream = np.sqrt(gamma * R_gas * T_upstream)
    A_shock = A_in + (A_ex - A_in) * x_target / L
    dln_area_dx = ((A_ex - A_in) / L) / A_shock
    C, tau = culick_rogers_coefficients(
        M_shock, p_upstream, a_upstream, dln_area_dx, gamma,
    )
    return {
        "M_shock": float(M_shock),
        "p_upstream_Pa": float(p_upstream),
        "T_upstream_K": float(T_upstream),
        "a_upstream_m_s": float(a_upstream),
        "dln_area_dx_per_m": float(dln_area_dx),
        "C_m_per_Pa": C,
        "tau_s": tau,
    }


def culick_rogers_frequency_response(frequency_hz, tau):
    """Normalized gain and positive phase lag of ``1/(1+i*omega*tau)``."""
    omega_tau = 2.0 * np.pi * float(frequency_hz) * float(tau)
    return {
        "omega_tau": float(omega_tau),
        "normalized_gain": float(1.0 / np.sqrt(1.0 + omega_tau**2)),
        "phase_lag_rad": float(np.arctan(omega_tau)),
    }


def wrap_phase(value):
    """Wrap a phase difference to [-pi, pi)."""
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def run_frequency(freq, amp, cycles, settle_steps, nx, x_target,
                  A_in, A_ex, L, M1, p1, T1, sample_interval_steps=5):
    """Run one forced case; return (summary_row, histories)."""
    gamma, R_gas = 1.4, 287.0
    p_e0, *_ = analytic_map(x_target, A_in, A_ex, L, M1, p1, gamma)
    geom = make_duct(A_in, A_ex, L)

    cfg = SolverConfig()
    cfg.inlet = InletConfig(mach=M1, T_inf=T1, p_inf=p1)
    cfg.geometry = geom
    cfg.mesh.nx = nx
    cfg.mesh.ny = 1
    cfg.cfl = 0.35
    cfg.print_interval = 0
    cfg.combustion = CombustionConfig(enabled=False)
    cfg.outlet_type = "back_pressure"
    cfg.outlet_p_back = p_e0
    cfg.outlet_p_back_amplitude = amp
    cfg.outlet_p_back_frequency_hz = freq
    solver = Solver(cfg)
    initialize_shock(solver, geom, x_target, A_in, A_ex, L, M1, p1, T1)

    # settle at the mean back pressure first (forcing amplitude scales with
    # sin(2 pi f t); t starts at 0 so early transients see small forcing)
    for _ in range(settle_steps):
        solver.advance_one_step()

    t_final = solver.time + cycles / freq
    forcing_rows, qoi_rows = [], []
    slope = quasi_steady_slope(x_target, A_in, A_ex, L, M1, p1)
    gain_sign = 1.0 if slope >= 0.0 else -1.0
    cr_point = culick_rogers_operating_point(
        x_target, A_in, A_ex, L, M1, p1, T1, gamma, R_gas,
    )
    cr_response = culick_rogers_frequency_response(freq, cr_point["tau_s"])

    def sample(s):
        d = shock_diagnostics(s)
        forcing_rows.append({
            "time": float(s.time),
            "q": float(s.bc.back_pressure(s.time)),
        })
        qoi_rows.append({
            "time": float(s.time),
            "shock_x": float(d["shock_x"]),
            # Align the static gain with the forcing before defining lag.
            # If dp_back/dx_shock < 0, upstream shock motion is positive.
            "shock_response_aligned": float(gain_sign * d["shock_x"]),
        })

    n = 0
    while solver.time < t_final:
        solver.advance_one_step(t_final=t_final)
        n += 1
        if n % sample_interval_steps == 0:
            sample(solver)

    metrics = extract_response_metrics(
        qoi_rows=qoi_rows, forcing_rows=forcing_rows,
        frequency_hz=freq, discard_fraction=0.25,
        qoi_keys=("shock_response_aligned",), min_cycles=2.0,
    )
    shock_metric = metrics["qoi"]["shock_response_aligned"]

    qs_amplitude = amp * p_e0 / abs(slope)

    amp_meas = shock_metric["amplitude"]
    row = {
        "frequency_hz": freq,
        "forcing_amp_frac": amp,
        "p_back_mean": p_e0,
        "shock_x_mean": (gain_sign * shock_metric["mean"]
                         if shock_metric["mean"] is not None else None),
        "shock_x_amplitude": amp_meas,
        "shock_x_phase_lag_rad": shock_metric["phase_lag_vs_q_rad"],
        "response_supported": shock_metric.get("quality", {}).get("supported", False),
        "quasi_steady_amplitude": qs_amplitude,
        "amplitude_ratio_vs_quasi_steady": (
            amp_meas / qs_amplitude
            if amp_meas is not None and qs_amplitude > 0 else None),
        "n_cycles": metrics["n_cycles_after_transient"],
        "warnings": "; ".join(metrics["warnings"]),
        "cr_local_shock_mach": cr_point["M_shock"],
        "cr_upstream_pressure_Pa": cr_point["p_upstream_Pa"],
        "cr_local_sound_speed_m_s": cr_point["a_upstream_m_s"],
        "cr_dln_area_dx_per_m": cr_point["dln_area_dx_per_m"],
        "cr_C_m_per_Pa": cr_point["C_m_per_Pa"],
        "cr_tau_s": cr_point["tau_s"],
        "cr_omega_tau": cr_response["omega_tau"],
        "cr_normalized_gain": cr_response["normalized_gain"],
        "cr_phase_lag_rad": cr_response["phase_lag_rad"],
        "cr_hybrid_exit_amplitude": (
            qs_amplitude * cr_response["normalized_gain"]
        ),
        "solver_over_cr_hybrid_gain": (
            amp_meas / (qs_amplitude * cr_response["normalized_gain"])
            if amp_meas is not None and qs_amplitude > 0.0 else None
        ),
        "solver_minus_cr_hybrid_phase_wrapped_rad": (
            wrap_phase(shock_metric["phase_lag_vs_q_rad"]
                       - cr_response["phase_lag_rad"])
            if shock_metric["phase_lag_vs_q_rad"] is not None else None
        ),
    }
    return row, (forcing_rows, qoi_rows)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Forced-shock (back-pressure modulation) response benchmark")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--frequencies-hz", default="20,50,100,200,400,800")
    parser.add_argument("--amplitude", type=float, default=0.02,
                        help="Fractional back-pressure modulation amplitude.")
    parser.add_argument("--cycles", type=float, default=8.0)
    parser.add_argument("--settle-steps", type=int, default=15000)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--x-target", type=float, default=0.5)
    parser.add_argument("--mach", type=float, default=2.0)
    parser.add_argument("--p1", type=float, default=20000.0)
    parser.add_argument("--t1", type=float, default=300.0)
    args = parser.parse_args(argv)

    freqs = [float(s) for s in args.frequencies_hz.split(",") if s.strip()]
    A_in, A_ex, L = 0.05, 0.10, 1.0

    if args.output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "runs" / f"forced_shock_benchmark_{stamp}"
    else:
        output_root = Path(args.output_root)
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_root / "manifest.json", {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study": "forced_shock_benchmark",
        "duct": {"A_in": A_in, "A_ex": A_ex, "L": L},
        "inlet": {"mach": args.mach, "p1_Pa": args.p1, "T1_K": args.t1},
        "x_target": args.x_target,
        "amplitude_frac": args.amplitude,
        "frequencies_hz": freqs,
        "references": [
            "Culick & Rogers, AIAA J 21(10), 1983 (quasi-1D shock response theory)",
            "Sajben, Bogar & Kroutil forced-oscillation transonic diffuser data",
        ],
        "culick_rogers_comparison": {
            "source": "Culick & Rogers (1983), Eqs. 42--44, isentropic-flow limit",
            "doi": "10.2514/3.60147",
            "published_input": "acoustic pressure immediately downstream of shock",
            "numerical_input": "imposed pressure at finite-duct outlet",
            "overlay": (
                "hybrid only: exact exit-pressure static gain multiplied by "
                "the published first-order relaxation factor"
            ),
            "pass_fail_assertion": False,
            "transcription": CULICK_ROGERS_TRANSCRIPTION_PROVENANCE,
        },
        "git": git_metadata(),
    })

    rows = []
    for freq in freqs:
        print(f"[forced-shock] f = {freq:g} Hz ...")
        row, (forcing_rows, qoi_rows) = run_frequency(
            freq, args.amplitude, args.cycles, args.settle_steps,
            args.nx, args.x_target, A_in, A_ex, L,
            args.mach, args.p1, args.t1)
        rows.append(row)
        case_dir = output_root / f"f_{freq:g}Hz"
        write_csv(case_dir / "forcing_history.csv", forcing_rows)
        write_csv(case_dir / "shock_history.csv", qoi_rows)
        ratio = row["amplitude_ratio_vs_quasi_steady"]
        lag = row["shock_x_phase_lag_rad"]
        amplitude_text = ("unsupported" if row["shock_x_amplitude"] is None
                          else f"{row['shock_x_amplitude']:.4f} m")
        print(f"    amplitude {amplitude_text} "
              f"(quasi-steady {row['quasi_steady_amplitude']:.4f} m, "
              f"ratio {ratio if ratio is None else round(ratio, 3)}), "
              f"phase lag {lag if lag is None else round(lag, 3)} rad")

    phase_indices = [
        i for i, row in enumerate(rows)
        if row.get("shock_x_phase_lag_rad") is not None
    ]
    if phase_indices:
        unwrapped = np.unwrap([
            rows[i]["shock_x_phase_lag_rad"] for i in phase_indices
        ])
        for i, phase_value in zip(phase_indices, unwrapped):
            rows[i]["shock_x_phase_lag_unwrapped_rad"] = float(phase_value)
            rows[i]["solver_minus_cr_hybrid_phase_unwrapped_rad"] = float(
                phase_value - rows[i]["cr_phase_lag_rad"],
            )
    write_csv(output_root / "summary.csv", rows)

    valid = [
        r for r in rows
        if r["amplitude_ratio_vs_quasi_steady"] is not None
        and r.get("shock_x_phase_lag_rad") is not None
    ]
    if valid:
        omega_tau = [r["cr_omega_tau"] for r in valid]
        a_arr = [r["amplitude_ratio_vs_quasi_steady"] for r in valid]
        a_cr = [r["cr_normalized_gain"] for r in valid]
        l_arr = [r.get("shock_x_phase_lag_unwrapped_rad",
                       r["shock_x_phase_lag_rad"]) for r in valid]
        l_cr = [r["cr_phase_lag_rad"] for r in valid]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].semilogx(omega_tau, a_arr, "bo-", label="solver / exact static gain")
        axes[0].semilogx(omega_tau, a_cr, "k--", label="Culick--Rogers hybrid")
        axes[0].axhline(1.0, color="k", ls="--", lw=0.8,
                        label="quasi-steady limit")
        axes[0].set_xlabel("Culick--Rogers reduced frequency, omega*tau")
        axes[0].set_ylabel("shock amplitude / quasi-steady amplitude")
        axes[0].set_title("Forced-shock amplitude response")
        axes[0].legend(fontsize=9)
        axes[1].semilogx(omega_tau, [float(x) for x in l_arr], "rs-",
                         label="solver vs outlet pressure")
        axes[1].semilogx(omega_tau, l_cr, "k--", label="Culick--Rogers local model")
        axes[1].set_xlabel("Culick--Rogers reduced frequency, omega*tau")
        axes[1].set_ylabel("phase lag vs p_b(t) [rad]")
        axes[1].set_title("Forced-shock phase response")
        axes[1].legend(fontsize=9)
        for ax in axes:
            ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        fig.savefig(plots_dir / "shock_response.png", dpi=140)
        plt.close(fig)

        gain_error = np.asarray(a_arr) - np.asarray(a_cr)
        phase_error = np.asarray(l_arr, dtype=float) - np.asarray(l_cr)
        write_json(output_root / "culick_rogers_comparison.json", {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "comparison_kind": "hybrid_exit_pressure_not_strict_local_input",
            "pass_fail_assertion": False,
            "n_supported": len(valid),
            "normalized_gain_rmse": float(np.sqrt(np.mean(gain_error**2))),
            "unwrapped_phase_difference_rmse_rad": float(
                np.sqrt(np.mean(phase_error**2)),
            ),
            "interpretation": (
                "The analytic curve isolates local shock relaxation. Extra "
                "lag in the solver includes propagation through the finite "
                "post-shock duct and outlet-boundary response."
            ),
        })

    print(f"\nForced-shock benchmark written to: {output_root}")
    return output_root


if __name__ == "__main__":
    main()
