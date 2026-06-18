"""
Diagnostics for reduced-fidelity scramjet studies.

These utilities report quantities useful for workflow screening and later
high-fidelity case selection. They do not implement turbulence or combustion
models.
"""
import numpy as np

from physics import TransportProperties


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
