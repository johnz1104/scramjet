"""
busemann.py — Busemann inlet generator (Taylor–Maccoll conical flow).

Generates the axisymmetric Busemann diffuser flowfield and its wall
streamline for the research plan's Config B family: freestream enters
through the leading Mach conoid, compresses isentropically through the
converging conical field, and passes through the terminal conical shock
into a uniform axial exit stream.

Construction (Mölder & Szpiro-style, integrated backwards from the shock):
    1. Choose the standard two-parameter family (M2, delta2): the
       pre-terminal-shock conical Mach number and the flow deflection
       through the terminal shock (the conical flow converges toward the
       axis at angle delta2 and the shock turns it exactly axial).
    2. The weak-branch theta-beta-M relation gives the wave angle beta;
       the terminal shock cone half-angle (from the downstream axis,
       focus at the origin) is theta2 = beta - delta2, and oblique-shock
       relations give the exit Mach M3.
    3. Integrate the Taylor–Maccoll equation from theta2 outward in theta,
       carrying the wall streamline dr/dtheta = r * V_r / V_theta, until
       the cylindrical-radial velocity component vanishes — the leading
       (freestream) conical characteristic theta1. The freestream Mach M1
       falls out of the integration.

Self-checks performed by generate_busemann_inlet (returned in
result["checks"], all should be ~machine/integration accuracy):
    - theta1 equals the freestream Mach angle conoid, theta1 = pi - mu(M1)
    - mass balance between the entry streamtube and the exit tube
    - the wall contour is monotonically contracting

The contour is returned both as (x, R) arrays and as a
mesh.TabulatedAreaProfile (A = pi R^2 with an appended isolator), ready to
drive the quasi-1D solver, the q-sweep machinery, and the Gmsh exporter.

References: Busemann (1942); Mölder & Szpiro, "Busemann inlet for
hypersonic speeds", J. Spacecraft 3(8), 1966; Taylor & Maccoll (1933).
"""
import numpy as np
from scipy.integrate import solve_ivp

from gasdynamics import (
    beta_from_deflection,
    isentropic_area_ratio,
    normal_shock,
    oblique_deflection,
    stagnation_pressure_ratio,
    stagnation_temperature_ratio,
)


def _speed_from_mach(M, gamma):
    """V/V_max from Mach number (V_max = sqrt(2 h0))."""
    g2 = 0.5 * (gamma - 1.0)
    return np.sqrt(g2 * M * M / (1.0 + g2 * M * M))


def _mach_from_speed(V, gamma):
    """Mach number from V/V_max."""
    g2 = 0.5 * (gamma - 1.0)
    return np.sqrt(V * V / (g2 * (1.0 - V * V)))


def _taylor_maccoll_rhs(theta, y, gamma):
    """
    Taylor–Maccoll conical-flow ODE + wall-streamline carrier.

    y = [V_r, V_theta, ln r], velocities normalized by V_max.
    """
    V_r, V_t, _ = y
    a2 = 0.5 * (gamma - 1.0) * (1.0 - V_r * V_r - V_t * V_t)
    denom = a2 - V_t * V_t
    if abs(denom) < 1e-14:
        denom = np.copysign(1e-14, denom)
    dV_t = (V_r * V_t * V_t - a2 * (2.0 * V_r + V_t / np.tan(theta))) / denom
    dlnr = V_r / V_t
    return [V_t, dV_t, dlnr]


def generate_busemann_inlet(M2=3.0, delta2_deg=15.0, gamma=1.4,
                            n_theta=400, isolator_length_factor=2.0,
                            R_exit=1.0):
    """
    Generate a Busemann inlet flowfield, wall contour, and area law.

    Args:
        M2:            conical-flow Mach number just upstream of the
                       terminal shock (design knob; > 1)
        delta2_deg:    flow deflection through the terminal shock [deg]
                       (the conical flow approaches the shock at this angle
                       toward the axis; the standard second family knob)
        gamma:         ratio of specific heats
        n_theta:       contour sampling resolution in theta
        isolator_length_factor: isolator length appended downstream of the
                       terminal shock, in exit diameters
        R_exit:        exit-tube radius used to scale the geometry [m]

    Returns:
        dict with keys:
            M1, M2, M3           — freestream / pre-shock / exit Mach
            delta_deg, beta_deg  — terminal deflection and wave angle
            theta1_deg, theta2_deg
            contraction_ratio    — capture area / exit-tube area
            p0_ratio_overall     — total-pressure recovery (isentropic
                                   conical compression x terminal shock)
            x_wall, R_wall       — compression contour (leading edge at x=0)
            area_profile         — mesh.TabulatedAreaProfile (incl. isolator)
            checks               — dict of self-consistency residuals
    """
    delta = np.radians(float(delta2_deg))
    if M2 <= 1.0:
        raise ValueError("M2 must be supersonic")
    if delta <= 0.0:
        raise ValueError("delta2_deg must be positive")

    # terminal shock closure: conical flow at angle delta -> axial exit,
    # weak-branch wave angle from theta-beta-M
    beta = beta_from_deflection(M2, delta, gamma, weak=True)
    theta2 = beta - delta
    if theta2 <= np.radians(0.5):
        raise ValueError(
            f"terminal shock cone angle theta2={np.degrees(theta2):.3f} deg "
            f"is degenerate; reduce delta2 or increase M2")
    Mn2 = M2 * np.sin(beta)
    ns = normal_shock(Mn2, gamma)
    M3 = ns["M2"] / np.sin(beta - delta)

    # initial conditions at the shock surface (theta = theta2, r = 1)
    V2 = _speed_from_mach(M2, gamma)
    y0 = [V2 * np.cos(beta), -V2 * np.sin(beta), 0.0]

    def hits_freestream(theta, y, *_args):
        """Cylindrical-radial velocity component: zero when flow is axial."""
        return y[0] * np.sin(theta) + y[1] * np.cos(theta)

    hits_freestream.terminal = True
    hits_freestream.direction = 1.0

    sol = solve_ivp(_taylor_maccoll_rhs, [theta2, np.pi - 1e-6], y0,
                    args=(gamma,), events=hits_freestream, dense_output=True,
                    rtol=1e-11, atol=1e-12, max_step=np.radians(0.25))
    if not sol.t_events[0].size:
        raise RuntimeError(
            f"Taylor–Maccoll integration never reached uniform flow "
            f"(M2={M2}, delta2={delta2_deg} deg)")

    theta1 = float(sol.t_events[0][0])
    V_r1, V_t1, lnr1 = sol.y_events[0][0]
    V1 = float(np.hypot(V_r1, V_t1))
    M1 = float(_mach_from_speed(V1, gamma))

    # sample the wall streamline between the shock and the entry conoid
    thetas = np.linspace(theta2, theta1, int(n_theta))
    V_r_s, V_t_s, lnr_s = sol.sol(thetas)
    r_s = np.exp(lnr_s) * R_exit / np.sin(theta2)  # scale: exit tube radius
    x_s = r_s * np.cos(thetas)
    R_s = r_s * np.sin(thetas)
    M_s = _mach_from_speed(np.hypot(V_r_s, V_t_s), gamma)

    # flow runs from theta1 (entry, most-negative x) to theta2 (shock)
    order = np.argsort(x_s)
    x_wall = x_s[order]
    R_wall = R_s[order]
    M_wall = M_s[order]
    x_wall = x_wall - x_wall[0]  # leading edge at x = 0

    # self-checks -----------------------------------------------------------
    checks = {}
    mu1 = np.arcsin(1.0 / M1)
    checks["mach_conoid_residual_deg"] = float(
        np.degrees(abs(theta1 - (np.pi - mu1))))

    # mass balance: entry streamtube (uniform M1, radius R1) vs exit tube.
    # p0, T0 are conserved through the isentropic conical field; the terminal
    # shock multiplies p0 by ns['p0_ratio'].
    R1 = float(R_wall[0])
    A1 = np.pi * R1**2
    A3 = np.pi * R_exit**2

    def mass_flux(M, p0, T0):
        """rho * u for given Mach at stagnation state (p0, T0)."""
        T = T0 / stagnation_temperature_ratio(M, gamma)
        p = p0 / stagnation_pressure_ratio(M, gamma)
        rho = p / (287.0 * T)
        u = M * np.sqrt(gamma * 287.0 * T)
        return rho * u

    p0, T0 = 1.0e5, 1000.0  # arbitrary reference; ratios are what matter
    mdot1 = mass_flux(M1, p0, T0) * A1
    mdot3 = mass_flux(M3, p0 * ns["p0_ratio"], T0) * A3
    checks["mass_balance_residual"] = float(abs(mdot3 / mdot1 - 1.0))

    dR = np.diff(R_wall)
    checks["contour_monotonic_fraction"] = float(np.mean(dR < 0.0))

    checks["shock_deflection_residual_deg"] = float(np.degrees(abs(
        oblique_deflection(M2, beta, gamma) - delta)))

    # area law (with isolator) ---------------------------------------------
    from mesh import TabulatedAreaProfile

    A_wall = np.pi * R_wall**2
    L_iso = isolator_length_factor * 2.0 * R_exit
    x_iso = np.linspace(x_wall[-1], x_wall[-1] + L_iso, 24)[1:]
    x_all = np.concatenate([x_wall, x_iso])
    A_all = np.concatenate([A_wall, np.full(len(x_iso), A3)])
    # PCHIP needs strictly increasing x; drop numerically coincident points
    keep = np.concatenate([[True], np.diff(x_all) > 1e-12 * x_all[-1]])
    area_profile = TabulatedAreaProfile(
        x_all[keep], A_all[keep],
        name=f"busemann_M2_{M2:g}_d2_{delta2_deg:g}")

    return {
        "M1": M1, "M2": float(M2), "M3": float(M3),
        "delta_deg": float(np.degrees(delta)),
        "beta_deg": float(np.degrees(beta)),
        "theta1_deg": float(np.degrees(theta1)),
        "theta2_deg": float(np.degrees(theta2)),
        "contraction_ratio": float(A1 / A3),
        "p0_ratio_overall": float(ns["p0_ratio"]),
        "x_wall": x_wall, "R_wall": R_wall, "M_wall": M_wall,
        "area_profile": area_profile,
        "checks": checks,
    }


def busemann_family(M2_values, delta2_deg=15.0, gamma=1.4):
    """Generate a family of Busemann inlets over M2 (the q-family analog)."""
    return [generate_busemann_inlet(M2=m2, delta2_deg=delta2_deg, gamma=gamma)
            for m2 in M2_values]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=== Busemann inlet generator ===")
    res = generate_busemann_inlet(M2=3.0, delta2_deg=15.0)
    print(f"  design: M1 = {res['M1']:.4f} -> M2 = {res['M2']:.2f} "
          f"-> M3 = {res['M3']:.4f}")
    print(f"  terminal shock: beta = {res['beta_deg']:.2f} deg, "
          f"deflection = {res['delta_deg']:.2f} deg")
    print(f"  entry conoid theta1 = {res['theta1_deg']:.2f} deg")
    print(f"  contraction ratio = {res['contraction_ratio']:.3f}, "
          f"overall p0 recovery = {res['p0_ratio_overall']:.4f}")
    print("  checks:")
    for k, v in res["checks"].items():
        print(f"    {k}: {v:.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(res["x_wall"], res["R_wall"], "b-", lw=2)
    axes[0].plot(res["x_wall"], -res["R_wall"], "b-", lw=2)
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("R [m]")
    axes[0].set_title(f"Busemann contour (M1={res['M1']:.2f})")
    axes[0].set_aspect("equal")
    ap = res["area_profile"]
    xs = np.linspace(0, ap.L_total, 400)
    axes[1].plot(xs, ap.area(xs), "r-", lw=2)
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("A(x) [m$^2$]")
    axes[1].set_title("Area law (with isolator)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "figures", "busemann_demo.png")
    fig.savefig(out, dpi=140)
    print(f"  wrote {out}")
