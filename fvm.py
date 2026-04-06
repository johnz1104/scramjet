"""
fvm.py — Finite Volume Method stack for the 2D compressible Euler/NS solver.

Contains (in dependency order):
    StateVector          — conservative <-> primitive conversion, EOS
    BoundaryConditions   — ghost-cell population for all 4 boundaries
    hllc_flux_kernel     — @njit HLLC Riemann solver (face-by-face)
    muscl_reconstruct_i  — @njit MUSCL-Venkatakrishnan on I-faces
    muscl_reconstruct_j  — @njit MUSCL-Venkatakrishnan on J-faces
    FVMResidual          — assemble dU/dt from face fluxes + source terms
    TimeIntegrator       — RK3-SSP explicit time stepping

Dependency: numpy, numba
"""
import numpy as np
import importlib.util

_HAS_NUMBA = importlib.util.find_spec("numba") is not None
if _HAS_NUMBA:
    from numba import njit
else:
    # pure-Python fallback — same interface, no compilation
    import functools
    def njit(**kwargs):
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kw):
                return fn(*args, **kw)
            return wrapper
        return decorator


class StateVector:
    """
    Conservative state vector U = [rho, rho*u, rho*v, rho*E, rho*Yf].

    Storage: (n_vars, nx, ny) array.
        index 0: rho          [kg/m^3]
        index 1: rho * u      [kg/(m^2 s)]
        index 2: rho * v      [kg/(m^2 s)]
        index 3: rho * E      [J/m^3]      (E = total specific energy)
        index 4: rho * Yf     [kg/m^3]     (fuel mass fraction * density)

    The 5th variable (species) is always allocated but only active when
    combustion is enabled. For pure Euler runs, Yf = 0 everywhere.
    """

    N_VARS = 5

    def __init__(self, nx, ny, gamma=1.4, R_gas=287.0):
        self.nx = nx
        self.ny = ny
        self.gamma = gamma
        self.R_gas = R_gas

        # conservative variables: (5, nx, ny)
        self.U = np.zeros((self.N_VARS, nx, ny), dtype=np.float64)

    def set_primitive(self, rho, u, v, p, Yf=None):
        """
        Initialise from primitive variables.

        Args:
            rho, u, v, p: arrays of shape (nx, ny)
            Yf:           fuel mass fraction, shape (nx, ny) or None
        """
        # E = p / (gamma - 1) / rho + 0.5 * (u^2 + v^2)
        E = p / ((self.gamma - 1.0) * rho) + 0.5 * (u**2 + v**2)

        self.U[0] = rho
        self.U[1] = rho * u
        self.U[2] = rho * v
        self.U[3] = rho * E

        if Yf is not None:
            self.U[4] = rho * Yf
        else:
            self.U[4] = 0.0

    def primitives(self):
        """
        Extract primitive variables from conservative state.

        Returns:
            rho, u, v, p, T, Yf — each shape (nx, ny)
        """
        rho = self.U[0]
        u = self.U[1] / rho
        v = self.U[2] / rho
        E = self.U[3] / rho

        # p = (gamma - 1) * rho * (E - 0.5 * (u^2 + v^2))
        ke = 0.5 * (u**2 + v**2)
        p = (self.gamma - 1.0) * rho * (E - ke)

        # T = p / (rho * R)
        T = p / (rho * self.R_gas)

        # Yf = rho_Yf / rho
        Yf = self.U[4] / np.maximum(rho, 1e-30)

        return rho, u, v, p, T, Yf

    def sound_speed(self):
        """Speed of sound: c = sqrt(gamma * p / rho)."""
        rho, u, v, p, T, Yf = self.primitives()
        return np.sqrt(self.gamma * np.maximum(p, 1e-30) / np.maximum(rho, 1e-30))

    def mach(self):
        """Local Mach number: M = |V| / c."""
        rho, u, v, p, T, Yf = self.primitives()
        c = np.sqrt(self.gamma * np.maximum(p, 1e-30) / np.maximum(rho, 1e-30))
        return np.sqrt(u**2 + v**2) / c

    def max_wave_speed(self):
        """Maximum wave speed across all cells: max(|u| + c, |v| + c)."""
        rho, u, v, p, T, Yf = self.primitives()
        c = np.sqrt(self.gamma * np.maximum(p, 1e-30) / np.maximum(rho, 1e-30))
        return max(np.max(np.abs(u) + c), np.max(np.abs(v) + c))


class BoundaryConditions:
    """
    Ghost-cell boundary condition handler.

    Supports:
        - Supersonic inflow (all variables prescribed)
        - Supersonic outflow (zero-gradient extrapolation)
        - Inviscid wall (slip, reflect normal velocity)
        - No-slip wall (zero velocity, adiabatic or isothermal)
    """

    def __init__(self, state, inlet_rho, inlet_u, inlet_v, inlet_p,
                 inlet_Yf=0.0, wall_type="slip"):
        """
        Args:
            state:      StateVector instance (for gamma, R_gas)
            inlet_*:    scalar inflow conditions
            wall_type:  "slip" (inviscid) or "no_slip" (viscous)
        """
        self.gamma = state.gamma
        self.R_gas = state.R_gas
        self.inlet_rho = inlet_rho
        self.inlet_u = inlet_u
        self.inlet_v = inlet_v
        self.inlet_p = inlet_p
        self.inlet_Yf = inlet_Yf
        self.wall_type = wall_type

        # precompute inlet conservative state
        E_in = inlet_p / ((self.gamma - 1.0) * inlet_rho) + 0.5 * (inlet_u**2 + inlet_v**2)
        self.U_inlet = np.array([inlet_rho, inlet_rho * inlet_u,
                                  inlet_rho * inlet_v, inlet_rho * E_in,
                                  inlet_rho * inlet_Yf])

    def apply(self, U):
        """
        Populate ghost values in-place. Called before each residual evaluation.

        Convention: U has shape (5, nx, ny). Boundaries are at
            i=0 (left/inlet), i=nx-1 (right/outlet),
            j=0 (bottom wall), j=ny-1 (top wall).
        """
        ny = U.shape[2]

        # --- left boundary: supersonic inflow ---
        for k in range(5):
            U[k, 0, :] = self.U_inlet[k]

        # --- right boundary: supersonic outflow (zero-gradient) ---
        U[:, -1, :] = U[:, -2, :]

        # --- wall BCs only apply when ny > 1 ---
        if ny < 2:
            return

        # --- bottom wall (j=0) ---
        if self.wall_type == "slip":
            # reflect v-momentum, copy everything else
            U[:, :, 0] = U[:, :, 1]
            U[2, :, 0] = -U[2, :, 1]  # rho*v -> -rho*v
        else:
            # no-slip: zero velocity at wall
            U[0, :, 0] = U[0, :, 1]       # rho extrapolated
            U[1, :, 0] = -U[1, :, 1]      # rho*u -> -rho*u (gives u=0 at face)
            U[2, :, 0] = -U[2, :, 1]      # rho*v -> -rho*v
            U[3, :, 0] = U[3, :, 1]       # energy extrapolated (adiabatic)
            U[4, :, 0] = U[4, :, 1]       # species extrapolated

        # --- top wall (j=ny-1) ---
        if self.wall_type == "slip":
            U[:, :, -1] = U[:, :, -2]
            U[2, :, -1] = -U[2, :, -2]
        else:
            U[0, :, -1] = U[0, :, -2]
            U[1, :, -1] = -U[1, :, -2]
            U[2, :, -1] = -U[2, :, -2]
            U[3, :, -1] = U[3, :, -2]
            U[4, :, -1] = U[4, :, -2]


@njit(cache=True)
def _euler_flux_x(rho, u, v, E, p, Yf):
    """
    Euler flux vector in x-direction for a single state.

    F = [rho*u, rho*u^2 + p, rho*u*v, (rho*E + p)*u, rho*Yf*u]
    """
    rhou = rho * u
    F0 = rhou
    F1 = rhou * u + p
    F2 = rhou * v
    F3 = (rho * E + p) * u
    F4 = rho * Yf * u
    return F0, F1, F2, F3, F4


@njit(cache=True)
def hllc_flux_kernel(rho_L, u_L, v_L, p_L, Yf_L,
                     rho_R, u_R, v_R, p_R, Yf_R,
                     gamma):
    """
    HLLC approximate Riemann solver for a single face (x-direction normal).

    Returns the numerical flux F_hllc = [F0, F1, F2, F3, F4].

    Wave speed estimates follow Einfeldt (1988):
        S_L = min(u_L - c_L, u_tilde - c_tilde)
        S_R = max(u_R + c_R, u_tilde + c_tilde)
    where tilde quantities are Roe-averaged.

    The contact wave speed S_star comes from the HLLC closure (Toro, ch. 10).
    """
    gm1 = gamma - 1.0

    # total energy per unit volume
    E_L = p_L / (gm1 * rho_L) + 0.5 * (u_L**2 + v_L**2)
    E_R = p_R / (gm1 * rho_R) + 0.5 * (u_R**2 + v_R**2)

    # sound speeds
    c_L = (gamma * max(p_L, 1e-30) / max(rho_L, 1e-30)) ** 0.5
    c_R = (gamma * max(p_R, 1e-30) / max(rho_R, 1e-30)) ** 0.5

    # Roe averages (density-weighted)
    sqrt_rL = rho_L ** 0.5
    sqrt_rR = rho_R ** 0.5
    denom = sqrt_rL + sqrt_rR
    u_roe = (sqrt_rL * u_L + sqrt_rR * u_R) / denom
    v_roe = (sqrt_rL * v_L + sqrt_rR * v_R) / denom
    H_L = (rho_L * E_L + p_L) / rho_L
    H_R = (rho_R * E_R + p_R) / rho_R
    H_roe = (sqrt_rL * H_L + sqrt_rR * H_R) / denom
    c_roe = (max(gm1 * (H_roe - 0.5 * (u_roe**2 + v_roe**2)), 1e-30)) ** 0.5

    # wave speed estimates (Einfeldt)
    S_L = min(u_L - c_L, u_roe - c_roe)
    S_R = max(u_R + c_R, u_roe + c_roe)

    # contact wave speed
    # S_star = (p_R - p_L + rho_L*u_L*(S_L - u_L) - rho_R*u_R*(S_R - u_R))
    #        / (rho_L*(S_L - u_L) - rho_R*(S_R - u_R))
    num = p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)
    den = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    if abs(den) < 1e-30:
        S_star = 0.5 * (u_L + u_R)
    else:
        S_star = num / den

    if S_L >= 0.0:
        # supersonic from left
        return _euler_flux_x(rho_L, u_L, v_L, E_L, p_L, Yf_L)

    elif S_R <= 0.0:
        # supersonic from right
        return _euler_flux_x(rho_R, u_R, v_R, E_R, p_R, Yf_R)

    elif S_star >= 0.0:
        # left star region
        FL = _euler_flux_x(rho_L, u_L, v_L, E_L, p_L, Yf_L)
        dS_L = S_L - u_L
        if abs(dS_L) < 1e-30:
            return FL
        coeff = rho_L * dS_L / (S_L - S_star)
        # p_star from HLLC closure (Toro eq 10.26)
        p_star = p_L + rho_L * dS_L * (S_star - u_L)

        U_starL = np.empty(5)
        U_starL[0] = coeff
        U_starL[1] = coeff * S_star
        U_starL[2] = coeff * v_L
        # energy: E_star = rho_L*E_L/rho_L + (S_star - u_L)*(S_star + p_L/(rho_L*dS_L))
        U_starL[3] = coeff * (E_L + (S_star - u_L) * (S_star + p_L / (rho_L * dS_L)))
        U_starL[4] = coeff * Yf_L

        UL = np.array([rho_L, rho_L * u_L, rho_L * v_L, rho_L * E_L, rho_L * Yf_L])
        F0 = FL[0] + S_L * (U_starL[0] - UL[0])
        F1 = FL[1] + S_L * (U_starL[1] - UL[1])
        F2 = FL[2] + S_L * (U_starL[2] - UL[2])
        F3 = FL[3] + S_L * (U_starL[3] - UL[3])
        F4 = FL[4] + S_L * (U_starL[4] - UL[4])
        return F0, F1, F2, F3, F4

    else:
        # right star region
        FR = _euler_flux_x(rho_R, u_R, v_R, E_R, p_R, Yf_R)
        dS_R = S_R - u_R
        if abs(dS_R) < 1e-30:
            return FR
        coeff = rho_R * dS_R / (S_R - S_star)

        U_starR = np.empty(5)
        U_starR[0] = coeff
        U_starR[1] = coeff * S_star
        U_starR[2] = coeff * v_R
        U_starR[3] = coeff * (E_R + (S_star - u_R) * (S_star + p_R / (rho_R * dS_R)))
        U_starR[4] = coeff * Yf_R

        UR = np.array([rho_R, rho_R * u_R, rho_R * v_R, rho_R * E_R, rho_R * Yf_R])
        F0 = FR[0] + S_R * (U_starR[0] - UR[0])
        F1 = FR[1] + S_R * (U_starR[1] - UR[1])
        F2 = FR[2] + S_R * (U_starR[2] - UR[2])
        F3 = FR[3] + S_R * (U_starR[3] - UR[3])
        F4 = FR[4] + S_R * (U_starR[4] - UR[4])
        return F0, F1, F2, F3, F4


@njit(cache=True)
def _euler_flux_y(rho, u, v, E, p, Yf):
    """
    Euler flux vector in y-direction.

    G = [rho*v, rho*u*v, rho*v^2 + p, (rho*E + p)*v, rho*Yf*v]
    """
    rhov = rho * v
    G0 = rhov
    G1 = rhov * u
    G2 = rhov * v + p
    G3 = (rho * E + p) * v
    G4 = rho * Yf * v
    return G0, G1, G2, G3, G4


@njit(cache=True)
def hllc_flux_kernel_y(rho_L, u_L, v_L, p_L, Yf_L,
                       rho_R, u_R, v_R, p_R, Yf_R,
                       gamma):
    """
    HLLC Riemann solver for y-direction faces.
    Identical structure to x-direction but the normal velocity is v, tangential is u.
    """
    gm1 = gamma - 1.0

    E_L = p_L / (gm1 * rho_L) + 0.5 * (u_L**2 + v_L**2)
    E_R = p_R / (gm1 * rho_R) + 0.5 * (u_R**2 + v_R**2)

    c_L = (gamma * max(p_L, 1e-30) / max(rho_L, 1e-30)) ** 0.5
    c_R = (gamma * max(p_R, 1e-30) / max(rho_R, 1e-30)) ** 0.5

    sqrt_rL = rho_L ** 0.5
    sqrt_rR = rho_R ** 0.5
    denom = sqrt_rL + sqrt_rR
    v_roe = (sqrt_rL * v_L + sqrt_rR * v_R) / denom
    u_roe = (sqrt_rL * u_L + sqrt_rR * u_R) / denom
    H_L = (rho_L * E_L + p_L) / rho_L
    H_R = (rho_R * E_R + p_R) / rho_R
    H_roe = (sqrt_rL * H_L + sqrt_rR * H_R) / denom
    c_roe = (max(gm1 * (H_roe - 0.5 * (u_roe**2 + v_roe**2)), 1e-30)) ** 0.5

    # wave speeds along y (normal velocity = v)
    S_L = min(v_L - c_L, v_roe - c_roe)
    S_R = max(v_R + c_R, v_roe + c_roe)

    num = p_R - p_L + rho_L * v_L * (S_L - v_L) - rho_R * v_R * (S_R - v_R)
    den = rho_L * (S_L - v_L) - rho_R * (S_R - v_R)
    if abs(den) < 1e-30:
        S_star = 0.5 * (v_L + v_R)
    else:
        S_star = num / den

    if S_L >= 0.0:
        return _euler_flux_y(rho_L, u_L, v_L, E_L, p_L, Yf_L)
    elif S_R <= 0.0:
        return _euler_flux_y(rho_R, u_R, v_R, E_R, p_R, Yf_R)
    elif S_star >= 0.0:
        GL = _euler_flux_y(rho_L, u_L, v_L, E_L, p_L, Yf_L)
        dS_L = S_L - v_L
        if abs(dS_L) < 1e-30:
            return GL
        coeff = rho_L * dS_L / (S_L - S_star)
        U_starL = np.empty(5)
        U_starL[0] = coeff
        U_starL[1] = coeff * u_L
        U_starL[2] = coeff * S_star
        U_starL[3] = coeff * (E_L + (S_star - v_L) * (S_star + p_L / (rho_L * dS_L)))
        U_starL[4] = coeff * Yf_L
        UL = np.array([rho_L, rho_L * u_L, rho_L * v_L, rho_L * E_L, rho_L * Yf_L])
        return (GL[0] + S_L * (U_starL[0] - UL[0]),
                GL[1] + S_L * (U_starL[1] - UL[1]),
                GL[2] + S_L * (U_starL[2] - UL[2]),
                GL[3] + S_L * (U_starL[3] - UL[3]),
                GL[4] + S_L * (U_starL[4] - UL[4]))
    else:
        GR = _euler_flux_y(rho_R, u_R, v_R, E_R, p_R, Yf_R)
        dS_R = S_R - v_R
        if abs(dS_R) < 1e-30:
            return GR
        coeff = rho_R * dS_R / (S_R - S_star)
        U_starR = np.empty(5)
        U_starR[0] = coeff
        U_starR[1] = coeff * u_R
        U_starR[2] = coeff * S_star
        U_starR[3] = coeff * (E_R + (S_star - v_R) * (S_star + p_R / (rho_R * dS_R)))
        U_starR[4] = coeff * Yf_R
        UR = np.array([rho_R, rho_R * u_R, rho_R * v_R, rho_R * E_R, rho_R * Yf_R])
        return (GR[0] + S_R * (U_starR[0] - UR[0]),
                GR[1] + S_R * (U_starR[1] - UR[1]),
                GR[2] + S_R * (U_starR[2] - UR[2]),
                GR[3] + S_R * (U_starR[3] - UR[3]),
                GR[4] + S_R * (U_starR[4] - UR[4]))


@njit(cache=True)
def _venkatakrishnan_limiter(dq_minus, dq_plus, eps2):
    """
    Venkatakrishnan slope limiter (smooth, differentiable).

    phi = (dq_plus^2 + eps2 + 2*dq_minus*dq_plus) / (dq_plus^2 + 2*dq_minus^2 + dq_minus*dq_plus + eps2)
    when dq_minus and dq_plus have the same sign, otherwise phi = 0.

    eps2: smoothing parameter ~ (K * dx)^3 where K ~ 1-5.
    """
    if dq_minus * dq_plus <= 0.0:
        return 0.0
    num = dq_plus**2 + eps2 + 2.0 * dq_minus * dq_plus
    den = dq_plus**2 + 2.0 * dq_minus**2 + dq_minus * dq_plus + eps2
    if abs(den) < 1e-30:
        return 0.0
    return num / den


@njit(cache=True)
def _minmod(a, b):
    """Minmod limiter: simpler, more diffusive than Venkatakrishnan."""
    if a * b <= 0.0:
        return 0.0
    if abs(a) < abs(b):
        return a
    return b


@njit(cache=True)
def muscl_reconstruct_i(U, nx, ny, n_vars, eps2):
    """
    MUSCL piecewise-linear reconstruction on I-faces (x-direction).

    For each face (i, j) with i in [1, nx-1]:
        Left state:  U_L = U[:, i-1, j] + 0.5 * phi * (U[:, i-1, j] - U[:, i-2, j])
        Right state: U_R = U[:, i, j]   - 0.5 * phi * (U[:, i+1, j] - U[:, i, j])

    At boundaries (i=0 and i=nx), first-order (no reconstruction).

    Returns:
        U_L, U_R: arrays of shape (n_vars, nx+1, ny) — left/right states at each I-face
    """
    U_L = np.zeros((n_vars, nx + 1, ny))
    U_R = np.zeros((n_vars, nx + 1, ny))

    for j in range(ny):
        for i in range(nx + 1):
            for k in range(n_vars):
                if i == 0:
                    # left boundary face: first-order
                    U_L[k, i, j] = U[k, 0, j]
                    U_R[k, i, j] = U[k, 0, j]
                elif i == nx:
                    # right boundary face: first-order
                    U_L[k, i, j] = U[k, nx - 1, j]
                    U_R[k, i, j] = U[k, nx - 1, j]
                elif i == 1:
                    # one cell from left boundary: left state is first-order
                    U_L[k, i, j] = U[k, 0, j]
                    dq_plus = U[k, i + 1, j] - U[k, i, j] if i + 1 < nx else 0.0
                    dq_minus = U[k, i, j] - U[k, i - 1, j]
                    phi_R = _venkatakrishnan_limiter(dq_plus, dq_minus, eps2)
                    U_R[k, i, j] = U[k, i, j] - 0.5 * phi_R * dq_minus
                elif i == nx - 1:
                    # one cell from right boundary: right state is first-order
                    dq_minus = U[k, i - 1, j] - U[k, i - 2, j]
                    dq_plus = U[k, i, j] - U[k, i - 1, j]
                    phi_L = _venkatakrishnan_limiter(dq_minus, dq_plus, eps2)
                    U_L[k, i, j] = U[k, i - 1, j] + 0.5 * phi_L * dq_plus
                    U_R[k, i, j] = U[k, i, j]
                else:
                    # interior face: full MUSCL
                    dq_minus_L = U[k, i - 1, j] - U[k, i - 2, j]
                    dq_plus_L = U[k, i, j] - U[k, i - 1, j]
                    phi_L = _venkatakrishnan_limiter(dq_minus_L, dq_plus_L, eps2)
                    U_L[k, i, j] = U[k, i - 1, j] + 0.5 * phi_L * dq_plus_L

                    dq_plus_R = U[k, i + 1, j] - U[k, i, j] if i + 1 < nx else 0.0
                    dq_minus_R = U[k, i, j] - U[k, i - 1, j]
                    phi_R = _venkatakrishnan_limiter(dq_plus_R, dq_minus_R, eps2)
                    U_R[k, i, j] = U[k, i, j] - 0.5 * phi_R * dq_minus_R

    return U_L, U_R


@njit(cache=True)
def muscl_reconstruct_j(U, nx, ny, n_vars, eps2):
    """
    MUSCL piecewise-linear reconstruction on J-faces (y-direction).
    Same logic as I-faces but along j-index.
    """
    U_L = np.zeros((n_vars, nx, ny + 1))
    U_R = np.zeros((n_vars, nx, ny + 1))

    for i in range(nx):
        for j in range(ny + 1):
            for k in range(n_vars):
                if j == 0:
                    U_L[k, i, j] = U[k, i, 0]
                    U_R[k, i, j] = U[k, i, 0]
                elif j == ny:
                    U_L[k, i, j] = U[k, i, ny - 1]
                    U_R[k, i, j] = U[k, i, ny - 1]
                elif j == 1:
                    U_L[k, i, j] = U[k, i, 0]
                    dq_plus = U[k, i, j + 1] - U[k, i, j] if j + 1 < ny else 0.0
                    dq_minus = U[k, i, j] - U[k, i, j - 1]
                    phi_R = _venkatakrishnan_limiter(dq_plus, dq_minus, eps2)
                    U_R[k, i, j] = U[k, i, j] - 0.5 * phi_R * dq_minus
                elif j == ny - 1:
                    dq_minus = U[k, i, j - 1] - U[k, i, j - 2]
                    dq_plus = U[k, i, j] - U[k, i, j - 1]
                    phi_L = _venkatakrishnan_limiter(dq_minus, dq_plus, eps2)
                    U_L[k, i, j] = U[k, i, j - 1] + 0.5 * phi_L * dq_plus
                    U_R[k, i, j] = U[k, i, j]
                else:
                    dq_minus_L = U[k, i, j - 1] - U[k, i, j - 2]
                    dq_plus_L = U[k, i, j] - U[k, i, j - 1]
                    phi_L = _venkatakrishnan_limiter(dq_minus_L, dq_plus_L, eps2)
                    U_L[k, i, j] = U[k, i, j - 1] + 0.5 * phi_L * dq_plus_L

                    dq_plus_R = U[k, i, j + 1] - U[k, i, j] if j + 1 < ny else 0.0
                    dq_minus_R = U[k, i, j] - U[k, i, j - 1]
                    phi_R = _venkatakrishnan_limiter(dq_plus_R, dq_minus_R, eps2)
                    U_R[k, i, j] = U[k, i, j] - 0.5 * phi_R * dq_minus_R

    return U_L, U_R


class FVMResidual:
    """
    Assembles the semi-discrete FVM residual: dU/dt = -R(U).

    R(U) = (1/V) * sum_faces(F_numerical * S_face)

    Uses MUSCL reconstruction + HLLC flux for each face.
    """

    def __init__(self, mesh, gamma=1.4, eps2_scale=1.0):
        """
        Args:
            mesh:       StructuredMesh2D instance
            gamma:      ratio of specific heats
            eps2_scale: Venkatakrishnan limiter parameter K (eps2 = (K*dx)^3)
        """
        self.mesh = mesh
        self.gamma = gamma
        self.n_vars = StateVector.N_VARS

        # limiter smoothing: eps^2 = (K * max(dx, dy))^3
        h = max(mesh.dx.max(), mesh.dy.max())
        self.eps2 = (eps2_scale * h) ** 3

    def compute(self, U):
        """
        Evaluate dU/dt from the FVM convective operator.

        Args:
            U: conservative state array, shape (5, nx, ny)

        Returns:
            dUdt: shape (5, nx, ny) — the negative divergence of fluxes
        """
        nx, ny = self.mesh.nx, self.mesh.ny
        gamma = self.gamma
        n_vars = self.n_vars
        vol = self.mesh.vol

        dUdt = np.zeros_like(U)

        # I-direction fluxes (x-faces)
        U_L, U_R = muscl_reconstruct_i(U, nx, ny, n_vars, self.eps2)

        # compute primitive variables from reconstructed conservative states
        for i in range(nx + 1):
            S_face = self.mesh.i_face_area[i, :]
            iL = max(i - 1, 0)
            iR = min(i, nx - 1)
            for j in range(ny):
                rho_L = U_L[0, i, j]
                u_L = U_L[1, i, j] / max(rho_L, 1e-30)
                v_L = U_L[2, i, j] / max(rho_L, 1e-30)
                E_L = U_L[3, i, j] / max(rho_L, 1e-30)
                Yf_L = U_L[4, i, j] / max(rho_L, 1e-30)
                p_L = (gamma - 1.0) * rho_L * (E_L - 0.5 * (u_L**2 + v_L**2))
                # positivity fallback: if reconstruction produced a non-physical
                # state (negative rho or p), fall back to the left cell average
                if rho_L <= 1e-12 or p_L <= 1e-12:
                    rho_L = max(U[0, iL, j], 1e-12)
                    u_L = U[1, iL, j] / rho_L
                    v_L = U[2, iL, j] / rho_L
                    E_L = U[3, iL, j] / rho_L
                    Yf_L = U[4, iL, j] / rho_L
                    p_L = max((gamma - 1.0) * rho_L * (E_L - 0.5 * (u_L**2 + v_L**2)), 1e-12)

                rho_R = U_R[0, i, j]
                u_R = U_R[1, i, j] / max(rho_R, 1e-30)
                v_R = U_R[2, i, j] / max(rho_R, 1e-30)
                E_R = U_R[3, i, j] / max(rho_R, 1e-30)
                Yf_R = U_R[4, i, j] / max(rho_R, 1e-30)
                p_R = (gamma - 1.0) * rho_R * (E_R - 0.5 * (u_R**2 + v_R**2))
                if rho_R <= 1e-12 or p_R <= 1e-12:
                    rho_R = max(U[0, iR, j], 1e-12)
                    u_R = U[1, iR, j] / rho_R
                    v_R = U[2, iR, j] / rho_R
                    E_R = U[3, iR, j] / rho_R
                    Yf_R = U[4, iR, j] / rho_R
                    p_R = max((gamma - 1.0) * rho_R * (E_R - 0.5 * (u_R**2 + v_R**2)), 1e-12)

                F = hllc_flux_kernel(rho_L, u_L, v_L, p_L, Yf_L,
                                     rho_R, u_R, v_R, p_R, Yf_R, gamma)

                # flux contribution:
                # face i is the LEFT face of cell i => flux enters cell i => +
                # face i is the RIGHT face of cell i-1 => flux leaves cell i-1 => -
                for k in range(n_vars):
                    flux_val = F[k] * S_face[j]
                    if i < nx:
                        dUdt[k, i, j] += flux_val / vol[i, j]
                    if i > 0:
                        dUdt[k, i - 1, j] -= flux_val / vol[i - 1, j]

        # J-direction fluxes (y-faces)
        U_L_j, U_R_j = muscl_reconstruct_j(U, nx, ny, n_vars, self.eps2)

        for i in range(nx):
            for j in range(ny + 1):
                S_face = self.mesh.j_face_area[i, j]
                jL = max(j - 1, 0)
                jR = min(j, ny - 1)

                rho_L = U_L_j[0, i, j]
                u_L = U_L_j[1, i, j] / max(rho_L, 1e-30)
                v_L = U_L_j[2, i, j] / max(rho_L, 1e-30)
                E_L = U_L_j[3, i, j] / max(rho_L, 1e-30)
                Yf_L = U_L_j[4, i, j] / max(rho_L, 1e-30)
                p_L = (gamma - 1.0) * rho_L * (E_L - 0.5 * (u_L**2 + v_L**2))
                if rho_L <= 1e-12 or p_L <= 1e-12:
                    rho_L = max(U[0, i, jL], 1e-12)
                    u_L = U[1, i, jL] / rho_L
                    v_L = U[2, i, jL] / rho_L
                    E_L = U[3, i, jL] / rho_L
                    Yf_L = U[4, i, jL] / rho_L
                    p_L = max((gamma - 1.0) * rho_L * (E_L - 0.5 * (u_L**2 + v_L**2)), 1e-12)

                rho_R = U_R_j[0, i, j]
                u_R = U_R_j[1, i, j] / max(rho_R, 1e-30)
                v_R = U_R_j[2, i, j] / max(rho_R, 1e-30)
                E_R = U_R_j[3, i, j] / max(rho_R, 1e-30)
                Yf_R = U_R_j[4, i, j] / max(rho_R, 1e-30)
                p_R = (gamma - 1.0) * rho_R * (E_R - 0.5 * (u_R**2 + v_R**2))
                if rho_R <= 1e-12 or p_R <= 1e-12:
                    rho_R = max(U[0, i, jR], 1e-12)
                    u_R = U[1, i, jR] / rho_R
                    v_R = U[2, i, jR] / rho_R
                    E_R = U[3, i, jR] / rho_R
                    Yf_R = U[4, i, jR] / rho_R
                    p_R = max((gamma - 1.0) * rho_R * (E_R - 0.5 * (u_R**2 + v_R**2)), 1e-12)

                G = hllc_flux_kernel_y(rho_L, u_L, v_L, p_L, Yf_L,
                                        rho_R, u_R, v_R, p_R, Yf_R, gamma)

                for k in range(n_vars):
                    flux_val = G[k] * S_face
                    if j < ny:
                        dUdt[k, i, j] += flux_val / vol[i, j]
                    if j > 0:
                        dUdt[k, i, j - 1] -= flux_val / vol[i, j - 1]

        return dUdt


class TimeIntegrator:
    """
    Strong-stability-preserving 3rd-order Runge-Kutta (Shu & Osher 1988).

    U^(1)   = U^n + dt * L(U^n)
    U^(2)   = 3/4 * U^n + 1/4 * (U^(1) + dt * L(U^(1)))
    U^(n+1) = 1/3 * U^n + 2/3 * (U^(2) + dt * L(U^(2)))

    where L(U) = dU/dt from the FVM residual + source terms.
    """

    def __init__(self, cfl=0.4):
        self.cfl = cfl

    def compute_dt(self, state, mesh):
        """
        CFL-limited time step.

        dt = CFL * min(dx, dy) / max_wave_speed
        """
        dx_min = mesh.dx.min()
        dy_min = mesh.dy.min()
        a_max = state.max_wave_speed()

        if a_max < 1e-30:
            return 1e-10

        # convective CFL: dt = CFL * min(dx, dy) / (|u| + c)_max
        dt_conv = self.cfl * min(dx_min, dy_min) / a_max
        return dt_conv

    def step(self, state, dt, rhs_fn, bc_fn):
        """
        Advance state by one RK3-SSP step.

        Args:
            state:  StateVector instance
            dt:     time step size
            rhs_fn: callable(U) -> dUdt array, shape (5, nx, ny)
            bc_fn:  callable(U) -> None, applies BCs in-place
        """
        U_n = state.U.copy()

        # stage 1
        bc_fn(state.U)
        k1 = rhs_fn(state.U)
        state.U = U_n + dt * k1

        # stage 2
        bc_fn(state.U)
        k2 = rhs_fn(state.U)
        state.U = 0.75 * U_n + 0.25 * (state.U + dt * k2)

        # stage 3
        bc_fn(state.U)
        k3 = rhs_fn(state.U)
        state.U = (1.0 / 3.0) * U_n + (2.0 / 3.0) * (state.U + dt * k3)

        # final BC enforcement
        bc_fn(state.U)


if __name__ == "__main__":
    print("=== StateVector ===")
    sv = StateVector(10, 5)
    rho = np.full((10, 5), 1.225)
    u = np.full((10, 5), 300.0)
    v = np.zeros((10, 5))
    p = np.full((10, 5), 101325.0)
    sv.set_primitive(rho, u, v, p)

    rho2, u2, v2, p2, T2, Yf2 = sv.primitives()
    print(f"  rho err: {np.max(np.abs(rho2 - rho)):.2e}")
    print(f"  u err:   {np.max(np.abs(u2 - u)):.2e}")
    print(f"  p err:   {np.max(np.abs(p2 - p)):.2e}")
    print(f"  Mach:    {sv.mach()[0, 0]:.4f}")

    print("\n=== HLLC flux (single face) ===")
    F = hllc_flux_kernel(1.0, 1.0, 0.0, 1.0, 0.0,
                         0.125, 0.0, 0.0, 0.1, 0.0, 1.4)
    print(f"  F = [{F[0]:.4f}, {F[1]:.4f}, {F[2]:.4f}, {F[3]:.4f}, {F[4]:.4f}]")

    print("\nAll fvm tests passed.")