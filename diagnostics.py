"""
Diagnostics for reduced-fidelity scramjet studies.

These utilities report quantities useful for workflow screening and later
high-fidelity case selection. They do not implement turbulence or combustion
models.
"""
import numpy as np

from physics import TransportProperties


def physical_exit_index(mesh):
    """Index of the last physical station (the final column is a BC image)."""
    return mesh.nx - 2 if mesh.nx > 2 else mesh.nx - 1


def transverse_average(values, mesh, weights=None):
    """Wall-normal average with cell-width quadrature on stretched meshes."""
    values = np.asarray(values, dtype=float)
    quadrature = np.asarray(mesh.dy, dtype=float)
    if weights is not None:
        quadrature = quadrature * np.asarray(weights, dtype=float)
    denominator = float(np.sum(quadrature))
    if denominator <= 0.0:
        return float(np.mean(values))
    return float(np.sum(values * quadrature) / denominator)


def primitives_from_state(U, gamma=1.4, R_gas=287.0):
    """Extract primitive arrays from a conservative state tensor."""
    rho = np.maximum(np.asarray(U[0], dtype=float), 1.0e-30)
    u = np.asarray(U[1], dtype=float) / rho
    v = np.asarray(U[2], dtype=float) / rho
    E = np.asarray(U[3], dtype=float) / rho
    Yf = np.asarray(U[4], dtype=float) / rho
    p = (gamma - 1.0) * rho * (E - 0.5 * (u**2 + v**2))
    p = np.maximum(p, 1.0e-30)
    T = p / (rho * R_gas)
    M = np.sqrt(u**2 + v**2) / np.sqrt(gamma * p / rho)
    return rho, u, v, p, T, Yf, M


def total_pressure_recovery_from_state(U, mesh, cfg):
    """State-array implementation shared by full and reconstructed states."""
    gamma = cfg.inlet.gamma
    rho, u, v, p, T, Yf, M = primitives_from_state(
        U, gamma=gamma, R_gas=cfg.inlet.R_gas,
    )
    i_exit = physical_exit_index(mesh)
    factor = 1.0 + 0.5 * (gamma - 1.0) * M[i_exit, :]**2
    p0_exit = p[i_exit, :] * factor ** (gamma / (gamma - 1.0))
    mass_flux = np.maximum(rho[i_exit, :] * u[i_exit, :], 0.0)
    p0_exit_avg = transverse_average(
        p0_exit, mesh, weights=mass_flux,
    )
    inlet = cfg.inlet
    f_inf = 1.0 + 0.5 * (gamma - 1.0) * inlet.mach**2
    p0_inf = inlet.p_inf * f_inf ** (gamma / (gamma - 1.0))
    return p0_exit_avg / max(p0_inf, 1.0e-30)


def total_pressure_recovery(solver):
    """
    Total-pressure recovery: mass-flux-weighted exit p0 over freestream p0.

    This is the experiment-matched TPR (isolator/exit stagnation pressure
    ratio), distinct from the legacy static ratio p_exit/p_inf. The exit
    station is the last physical column (nx-2); column nx-1 is the outlet
    boundary-condition image.
    """
    return total_pressure_recovery_from_state(
        solver.state.U, solver.mesh, solver.cfg,
    )


def shock_diagnostics_from_state(U, mesh, cfg, p0_loss_threshold=0.03):
    """State-array shock detector shared by FOM and POD reconstructions."""
    rho, u, v, p, T, Yf, M = primitives_from_state(
        U, gamma=cfg.inlet.gamma, R_gas=cfg.inlet.R_gas,
    )
    return _shock_diagnostics_arrays(
        p, M, mesh, cfg.inlet.gamma, p0_loss_threshold,
    )


def shock_diagnostics(solver, p0_loss_threshold=0.03):
    """
    Locate the dominant shock (if any) on the duct centerline.

    Discriminator: local total-pressure destruction. A captured shock
    drops p0 across a few cells; smooth isentropic compression (e.g. the
    converging inlet) raises static pressure but conserves p0, so a
    static-pressure-rise detector would false-positive there. A shock is
    reported when the strongest 4-cell p0 drop exceeds
    ``p0_loss_threshold`` (fractional). The location is refined by the
    supersonic->subsonic crossing when one sits nearby (the
    normal-shock-in-duct signature used by the back-pressure studies).

    Returns:
        dict(shock_detected, shock_x, shock_index, shock_p0_ratio)
        shock_x is NaN when no shock is detected.
    """
    rho, u, v, p, T, Yf = solver.state.primitives()
    M = solver.state.mach()
    return _shock_diagnostics_arrays(
        p, M, solver.mesh, solver.cfg.inlet.gamma, p0_loss_threshold,
    )


def _shock_diagnostics_arrays(p, M, mesh, gamma, p0_loss_threshold):
    """Implementation for :func:`shock_diagnostics_from_state`."""
    j_mid = mesh.ny // 2
    x = mesh.xc
    nx = mesh.nx

    M_line = M[:, j_mid]
    p0_line = (p[:, j_mid]
               * (1.0 + 0.5 * (gamma - 1.0) * M_line**2) ** (gamma / (gamma - 1.0)))

    # exclude BC image cells at both ends
    lo, hi = 1, nx - 1
    if hi - lo < 6:
        return {"shock_detected": False, "shock_x": float("nan"),
                "shock_index": -1, "shock_p0_ratio": 1.0}

    dp0 = np.diff(p0_line[lo:hi])
    i_rel = int(np.argmin(dp0))          # strongest single-face p0 drop
    i_star = lo + i_rel                  # drop from cell i_star to i_star+1

    i0 = max(i_star - 1, lo)
    i1 = min(i_star + 3, hi - 1)
    p0_ratio = float(p0_line[i1] / max(p0_line[i0], 1e-30))

    detected = p0_ratio < 1.0 - p0_loss_threshold
    x_shock = float("nan")
    if detected:
        x_shock = float(0.5 * (x[i_star] + x[i_star + 1]))
        # refine with the sonic crossing when one exists near the drop
        crossings = np.where((M_line[lo:hi - 1] > 1.0)
                             & (M_line[lo + 1:hi] <= 1.0))[0]
        if len(crossings) > 0:
            ic = lo + int(crossings[0])
            if abs(ic - i_star) <= 4:
                x_shock = float(0.5 * (x[ic] + x[ic + 1]))
                i_star = ic

    return {
        "shock_detected": bool(detected),
        "shock_x": x_shock,
        "shock_index": int(i_star) if detected else -1,
        "shock_p0_ratio": p0_ratio,
    }


def scalar_diagnostics(solver):
    """Return passive fuel-scalar boundedness and inventory diagnostics."""
    rho, u, v, p, T, Yf = solver.state.primitives()
    scalar_mass = float(np.sum(rho * Yf * solver.mesh.vol))
    y_min = float(np.min(Yf))
    y_max = float(np.max(Yf))
    tol = 1.0e-10
    return {
        "fuel_scalar_min": y_min,
        "fuel_scalar_max": y_max,
        "integrated_fuel_scalar": scalar_mass,
        "fuel_scalar_bounded": bool(y_min >= -tol and y_max <= 1.0 + tol),
    }


def heat_release_diagnostics(solver):
    """Return simple heat-release proxy and thermodynamic extrema."""
    rho, u, v, p, T, Yf = solver.state.primitives()
    total_heat_release_proxy = 0.0
    if getattr(solver, "simple_heat_release", None) is not None:
        S = solver.simple_heat_release.compute(solver.state.U)
        total_heat_release_proxy = float(np.sum(S[3] * solver.mesh.vol))

    return {
        "heat_release_model": getattr(solver.cfg, "heat_release_model", "none"),
        "total_heat_release_proxy": total_heat_release_proxy,
        "temperature_min": float(np.min(T)),
        "temperature_max": float(np.max(T)),
        "pressure_min": float(np.min(p)),
        "pressure_max": float(np.max(p)),
    }


def grid_diagnostics(solver):
    """Return grid spacing and aspect-ratio diagnostics."""
    dx = solver.mesh.dx
    dy = solver.mesh.dy
    aspect = dx[:, np.newaxis] / dy[np.newaxis, :]
    return {
        "nx": int(solver.mesh.nx),
        "ny": int(solver.mesh.ny),
        "n_cells": int(solver.mesh.n_cells),
        "dx_min": float(np.min(dx)),
        "dx_max": float(np.max(dx)),
        "dy_min": float(np.min(dy)),
        "dy_max": float(np.max(dy)),
        "aspect_ratio_min": float(np.min(aspect)),
        "aspect_ratio_max": float(np.max(aspect)),
    }


def flow_diagnostics(solver):
    """Return flow and CFL diagnostics for later high-fidelity case screening."""
    rho, u, v, p, T, Yf = solver.state.primitives()
    M = solver.state.mach()
    transport = TransportProperties(
        gamma=solver.cfg.inlet.gamma,
        R_gas=solver.cfg.inlet.R_gas,
    )
    mu = transport.viscosity(T)
    velocity = np.sqrt(u**2 + v**2)
    L_ref = solver.cfg.geometry.L_total
    reynolds = rho * velocity * L_ref / np.maximum(mu, 1e-30)

    return {
        "mach_min": float(np.min(M)),
        "mach_max": float(np.max(M)),
        "mach_mean": float(np.mean(M)),
        "reynolds_min": float(np.min(reynolds)),
        "reynolds_max": float(np.max(reynolds)),
        "reynolds_mean": float(np.mean(reynolds)),
        "dt_min": float(np.min(solver.dt_history)) if solver.dt_history else 0.0,
        "dt_max": float(np.max(solver.dt_history)) if solver.dt_history else 0.0,
        "dt_mean": float(np.mean(solver.dt_history)) if solver.dt_history else 0.0,
    }


def turbulence_readiness_diagnostics(solver):
    """Return diagnostics without pretending to provide RANS/LES closure."""
    return {
        "turbulence_model": getattr(solver.cfg, "turbulence_model", "none"),
        "python_rans_les_supported": False,
        "note": (
            "The Python solver provides inviscid and molecular-diffusion "
            "prototype modes only; RANS/LES is deferred to OpenFOAM/FUN3D."
        ),
        "grid": grid_diagnostics(solver),
        "flow": flow_diagnostics(solver),
    }


def all_case_diagnostics(solver):
    """Combined diagnostics suitable for case JSON output."""
    return {
        "scalar": scalar_diagnostics(solver),
        "heat_release": heat_release_diagnostics(solver),
        "turbulence_readiness": turbulence_readiness_diagnostics(solver),
    }
