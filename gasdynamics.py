"""
gasdynamics.py — Exact perfect-gas compressible-flow relations.

Shared by the validation tests (isentropic nozzle, analytic shock position),
the forced-shock benchmark, and the Busemann inlet generator. All functions
assume a calorically perfect gas; gamma defaults to 1.4.

Contains:
    isentropic relations   — area ratio, stagnation ratios, inversions
    normal_shock           — jump conditions across a normal shock
    oblique-shock helpers  — theta-beta-M and post-shock state
"""
import numpy as np


# Isentropic relations

def isentropic_area_ratio(M, gamma=1.4):
    """A/A* as a function of Mach number."""
    t = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * M * M)
    return (1.0 / M) * t ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def mach_from_area_ratio(area_ratio, supersonic, gamma=1.4):
    """Invert A/A* on the chosen branch by bisection."""
    ar = float(area_ratio)
    if ar < 1.0:
        raise ValueError(f"A/A* must be >= 1, got {ar}")
    lo, hi = (1.0 + 1e-12, 100.0) if supersonic else (1e-9, 1.0 - 1e-12)
    f_lo = isentropic_area_ratio(lo, gamma) - ar
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if (isentropic_area_ratio(mid, gamma) - ar) * f_lo <= 0.0:
            hi = mid
        else:
            lo = mid
            f_lo = isentropic_area_ratio(lo, gamma) - ar
    return 0.5 * (lo + hi)


def stagnation_pressure_ratio(M, gamma=1.4):
    """p0/p at Mach M."""
    return (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (gamma / (gamma - 1.0))


def stagnation_temperature_ratio(M, gamma=1.4):
    """T0/T at Mach M."""
    return 1.0 + 0.5 * (gamma - 1.0) * M * M


# Normal shock

def normal_shock(M, gamma=1.4):
    """
    Normal-shock jump conditions for upstream Mach M > 1.

    Returns:
        dict with M2, p_ratio (p2/p1), rho_ratio, T_ratio, p0_ratio (p02/p01)
    """
    if M <= 1.0:
        raise ValueError(f"normal shock requires M > 1, got {M}")
    g = gamma
    M2 = np.sqrt((1.0 + 0.5 * (g - 1.0) * M * M)
                 / (g * M * M - 0.5 * (g - 1.0)))
    p_ratio = 1.0 + (2.0 * g / (g + 1.0)) * (M * M - 1.0)
    rho_ratio = ((g + 1.0) * M * M) / ((g - 1.0) * M * M + 2.0)
    T_ratio = p_ratio / rho_ratio
    p0_ratio = ((0.5 * (g + 1.0) * M * M / (1.0 + 0.5 * (g - 1.0) * M * M))
                ** (g / (g - 1.0))
                * ((2.0 * g / (g + 1.0)) * M * M
                   - (g - 1.0) / (g + 1.0)) ** (-1.0 / (g - 1.0)))
    return {"M2": float(M2), "p_ratio": float(p_ratio),
            "rho_ratio": float(rho_ratio), "T_ratio": float(T_ratio),
            "p0_ratio": float(p0_ratio)}


# Oblique shock

def oblique_deflection(M, beta, gamma=1.4):
    """
    Flow deflection angle theta from the theta-beta-M relation.

    Args:
        M:    upstream Mach number
        beta: shock angle relative to the upstream flow [rad]

    Returns:
        theta [rad] (0 for a Mach wave; raises if beta below the Mach angle)
    """
    mu = np.arcsin(1.0 / M)
    if beta < mu - 1e-12:
        raise ValueError(f"beta={beta:.4f} below Mach angle {mu:.4f}")
    tan_theta = (2.0 / np.tan(beta)
                 * (M * M * np.sin(beta) ** 2 - 1.0)
                 / (M * M * (gamma + np.cos(2.0 * beta)) + 2.0))
    return float(np.arctan(tan_theta))


def oblique_shock_from_beta(M, beta, gamma=1.4):
    """
    Post-shock state for an oblique shock of angle beta (relative to the
    upstream flow) at upstream Mach M.

    Returns:
        dict with theta (deflection), M2, p_ratio, rho_ratio, T_ratio, p0_ratio
    """
    theta = oblique_deflection(M, beta, gamma)
    Mn1 = M * np.sin(beta)
    ns = normal_shock(Mn1, gamma)
    M2 = ns["M2"] / np.sin(beta - theta)
    return {"theta": float(theta), "M2": float(M2),
            "p_ratio": ns["p_ratio"], "rho_ratio": ns["rho_ratio"],
            "T_ratio": ns["T_ratio"], "p0_ratio": ns["p0_ratio"]}


def beta_from_deflection(M, theta, gamma=1.4, weak=True):
    """
    Invert theta-beta-M for the shock angle at deflection theta.

    Args:
        weak: pick the weak-shock branch (True) or strong branch (False)

    Returns:
        beta [rad]
    """
    mu = np.arcsin(1.0 / M)
    # find beta of maximum deflection by golden-section-ish scan
    betas = np.linspace(mu + 1e-9, 0.5 * np.pi - 1e-9, 2000)
    thetas = np.array([oblique_deflection(M, b, gamma) for b in betas])
    i_max = int(np.argmax(thetas))
    if theta > thetas[i_max]:
        raise ValueError(f"deflection {theta:.4f} exceeds max "
                         f"{thetas[i_max]:.4f} at M={M}")
    if weak:
        lo, hi = 0, i_max
    else:
        lo, hi = i_max, len(betas) - 1
    seg = thetas[lo:hi + 1]
    j = int(np.argmin(np.abs(seg - theta)))
    b0 = betas[lo + j]
    # Newton refinement
    beta = b0
    for _ in range(60):
        f = oblique_deflection(M, beta, gamma) - theta
        df = (oblique_deflection(M, beta + 1e-7, gamma)
              - oblique_deflection(M, beta - 1e-7, gamma)) / 2e-7
        if abs(df) < 1e-14:
            break
        step = f / df
        beta -= np.clip(step, -0.05, 0.05)
        beta = float(np.clip(beta, mu + 1e-9, 0.5 * np.pi - 1e-9))
        if abs(f) < 1e-12:
            break
    return float(beta)


if __name__ == "__main__":
    print("=== gasdynamics sanity ===")
    print(f"  A/A*(M=2)      = {isentropic_area_ratio(2.0):.5f} (expect 1.68750)")
    print(f"  M(A/A*=1.6875) = {mach_from_area_ratio(1.6875, True):.5f} (expect 2.0)")
    ns = normal_shock(2.0)
    print(f"  normal shock M=2: M2={ns['M2']:.5f} (0.57735), "
          f"p2/p1={ns['p_ratio']:.4f} (4.5), p02/p01={ns['p0_ratio']:.5f} (0.72087)")
    th = oblique_deflection(3.0, np.deg2rad(30.0))
    print(f"  theta(M=3, beta=30deg) = {np.rad2deg(th):.3f} deg")
    b = beta_from_deflection(3.0, th)
    print(f"  beta round-trip: {np.rad2deg(b):.3f} deg (expect 30)")
    print("gasdynamics module OK")
