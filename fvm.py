"""
fvm.py — Finite Volume Method stack for the 2D compressible Euler/NS solver.

Contains (in dependency order):
    StateVector          — conservative <-> primitive conversion, EOS
    BoundaryConditions   — boundary-cell population (inlet/outlet/walls,
                           incl. optional modulated back-pressure outlet)
    hllc_flux_kernel     — @njit HLLC Riemann solver (single face)
    muscl_reconstruct_i  — @njit MUSCL-Venkatakrishnan on I-faces
    muscl_reconstruct_j  — @njit MUSCL-Venkatakrishnan on J-faces
    accumulate_fluxes_*  — @njit flux assembly (area-weighted in x)
    FVMResidual          — dU/dt incl. quasi-1D variable-area coupling
    TimeIntegrator       — RK3-SSP explicit time stepping with stage times

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
    Boundary-cell condition handler.

    Note: this scheme has no ghost layer. The outermost *solution* cells are
    overwritten every stage (i=0 Dirichlet inlet, i=nx-1 outlet image,
    j=0 / j=ny-1 wall images), so the first/last rows and columns are
    boundary-condition cells, not accurate solution cells.

    Supports:
        - Supersonic inflow (all variables prescribed)
        - Supersonic outflow (zero-gradient extrapolation)
        - Back-pressure outflow (imposed static pressure, optionally
          sinusoidally modulated in time — for shock-in-duct and
          Culick-Rogers/Sajben-type forced-response studies)
        - Inviscid wall (slip, reflect normal velocity)
        - No-slip wall (zero velocity, adiabatic or isothermal)
    """

    def __init__(self, state, inlet_rho, inlet_u, inlet_v, inlet_p,
                 inlet_Yf=0.0, wall_type="slip", outlet_type="supersonic",
                 outlet_p_back=None, outlet_p_amplitude=0.0,
                 outlet_p_frequency_hz=0.0, outlet_p_phase=0.0):
        """
        Args:
            state:      StateVector instance (for gamma, R_gas)
            inlet_*:    scalar inflow conditions
            wall_type:  "slip" (inviscid) or "no_slip" (viscous)
            outlet_type: "supersonic" (zero-gradient) or "back_pressure"
            outlet_p_back: imposed outlet static pressure [Pa]
                        (required for outlet_type="back_pressure")
            outlet_p_amplitude: fractional sinusoidal modulation of p_back
            outlet_p_frequency_hz, outlet_p_phase: modulation parameters
        """
        self.gamma = state.gamma
        self.R_gas = state.R_gas
        self.inlet_rho = inlet_rho
        self.inlet_u = inlet_u
        self.inlet_v = inlet_v
        self.inlet_p = inlet_p
        self.inlet_Yf = inlet_Yf
        self.wall_type = wall_type

        if outlet_type not in ("supersonic", "back_pressure"):
            raise ValueError(f"Unsupported outlet_type: {outlet_type}")
        if outlet_type == "back_pressure" and outlet_p_back is None:
            raise ValueError("outlet_type='back_pressure' requires outlet_p_back")
        self.outlet_type = outlet_type
        self.outlet_p_back = outlet_p_back
        self.outlet_p_amplitude = float(outlet_p_amplitude)
        self.outlet_p_frequency_hz = float(outlet_p_frequency_hz)
        self.outlet_p_phase = float(outlet_p_phase)

        # precompute inlet conservative state
        E_in = inlet_p / ((self.gamma - 1.0) * inlet_rho) + 0.5 * (inlet_u**2 + inlet_v**2)
        self.U_inlet = np.array([inlet_rho, inlet_rho * inlet_u,
                                  inlet_rho * inlet_v, inlet_rho * E_in,
                                  inlet_rho * inlet_Yf])

    def back_pressure(self, time):
        """Instantaneous imposed outlet pressure p_b(t)."""
        p_b = float(self.outlet_p_back)
        if self.outlet_p_amplitude != 0.0:
            omega = 2.0 * np.pi * self.outlet_p_frequency_hz
            p_b *= 1.0 + self.outlet_p_amplitude * np.sin(
                omega * float(time) + self.outlet_p_phase)
        return p_b

    def apply(self, U, time=None):
        """
        Populate boundary cells in-place. Called before each residual evaluation.

        Convention: U has shape (5, nx, ny). Boundary cells are at
            i=0 (left/inlet), i=nx-1 (right/outlet),
            j=0 (bottom wall), j=ny-1 (top wall).
        """
        ny = U.shape[2]

        # --- left boundary: supersonic inflow ---
        for k in range(5):
            U[k, 0, :] = self.U_inlet[k]

        # --- right boundary ---
        # zero-gradient extrapolation for rho, momentum, species
        U[:, -1, :] = U[:, -2, :]
        if self.outlet_type == "back_pressure":
            # impose static pressure on the extrapolated state (throttle-style
            # outlet; drives the standard subsonic/shock-in-duct response)
            p_b = self.back_pressure(0.0 if time is None else time)
            rho_e = np.maximum(U[0, -1, :], 1e-30)
            ke = 0.5 * (U[1, -1, :]**2 + U[2, -1, :]**2) / rho_e
            U[3, -1, :] = p_b / (self.gamma - 1.0) + ke

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
def _limited_slope(dq_minus, dq_plus, eps2):
    """
    Cell-centered limited slope (van Albada average with Venkatakrishnan-style
    smoothing eps^2, which must scale with the variable's magnitude squared).

        slope = [(dq_plus^2 + eps2) dq_minus + (dq_minus^2 + eps2) dq_plus]
                / (dq_plus^2 + dq_minus^2 + 2 eps2)

    when the one-sided differences agree in sign (0 at extrema). In smooth
    flow slope -> the central difference average, giving second order; the
    same slope serves both faces of the cell, which is what makes the
    MUSCL + Riemann pairing upwind-dissipative and steady-state convergent.

    History: an earlier formulation limited each face separately using the
    downwind difference as the extrapolation slope; at limiter value 1 that
    is a fully central (zero-dissipation) reconstruction, and steady
    supersonic duct runs settled into a sustained ~1-2% limit cycle
    instead of converging.
    """
    if dq_minus * dq_plus <= 0.0:
        return 0.0
    den = dq_plus**2 + dq_minus**2 + 2.0 * eps2
    if den < 1e-300:
        return 0.0
    return ((dq_plus**2 + eps2) * dq_minus + (dq_minus**2 + eps2) * dq_plus) / den


@njit(cache=True)
def _minmod(a, b):
    """Minmod limiter: simpler, more diffusive alternative (kept for reference)."""
    if a * b <= 0.0:
        return 0.0
    if abs(a) < abs(b):
        return a
    return b


@njit(cache=True)
def muscl_reconstruct_i(U, nx, ny, n_vars, eps2):
    """
    MUSCL piecewise-linear reconstruction on I-faces (x-direction).

    Each interior cell m carries one limited slope s_m (van Albada/Venkat
    average of its two one-sided differences); face states are

        U_L(face i) = U[:, i-1, :] + 0.5 * s_{i-1}
        U_R(face i) = U[:, i,   :] - 0.5 * s_i

    Boundary cells (m = 0, nx-1) use zero slope: they are BC image cells,
    so the faces they touch are first-order.

    Args:
        eps2: per-variable smoothing, shape (n_vars,), each entry scaling
              with that variable's magnitude squared so limiting behaves
              identically across rho / momentum / energy.

    Returns:
        U_L, U_R: arrays of shape (n_vars, nx+1, ny)
    """
    U_L = np.zeros((n_vars, nx + 1, ny))
    U_R = np.zeros((n_vars, nx + 1, ny))
    slope = np.zeros((n_vars, nx, ny))

    for j in range(ny):
        for m in range(1, nx - 1):
            for k in range(n_vars):
                dq_minus = U[k, m, j] - U[k, m - 1, j]
                dq_plus = U[k, m + 1, j] - U[k, m, j]
                slope[k, m, j] = _limited_slope(dq_minus, dq_plus, eps2[k])

    for j in range(ny):
        for i in range(nx + 1):
            iL = i - 1 if i > 0 else 0
            iR = i if i < nx else nx - 1
            for k in range(n_vars):
                U_L[k, i, j] = U[k, iL, j] + 0.5 * slope[k, iL, j]
                U_R[k, i, j] = U[k, iR, j] - 0.5 * slope[k, iR, j]

    return U_L, U_R


@njit(cache=True)
def muscl_reconstruct_j(U, nx, ny, n_vars, eps2):
    """
    MUSCL piecewise-linear reconstruction on J-faces (y-direction).
    Same cell-slope logic as I-faces but along the j-index.
    """
    U_L = np.zeros((n_vars, nx, ny + 1))
    U_R = np.zeros((n_vars, nx, ny + 1))
    slope = np.zeros((n_vars, nx, ny))

    for i in range(nx):
        for m in range(1, ny - 1):
            for k in range(n_vars):
                dq_minus = U[k, i, m] - U[k, i, m - 1]
                dq_plus = U[k, i, m + 1] - U[k, i, m]
                slope[k, i, m] = _limited_slope(dq_minus, dq_plus, eps2[k])

    for i in range(nx):
        for j in range(ny + 1):
            jL = j - 1 if j > 0 else 0
            jR = j if j < ny else ny - 1
            for k in range(n_vars):
                U_L[k, i, j] = U[k, i, jL] + 0.5 * slope[k, i, jL]
                U_R[k, i, j] = U[k, i, jR] - 0.5 * slope[k, i, jR]

    return U_L, U_R


@njit(cache=True)
def accumulate_fluxes_i(U, U_L, U_R, dy, vol, A_face, A_cell, gamma, dUdt):
    """
    Assemble I-face (x-direction) HLLC flux contributions into dUdt.

    Quasi-1D duct coupling: fluxes are weighted by the face-interpolated
    duct area A_face and deposits divided by the cell area A_cell, i.e.
    the scheme integrates d(A U)/dt = -d(A F)/dx - A dG/dy + S.
    Pass A_face = A_cell = 1 for a plain 2D channel.

    Includes the positivity fallback: reconstructed face states with
    non-physical rho or p revert to the adjacent cell average.
    """
    n_vars, nxp1, ny = U_L.shape
    nx = nxp1 - 1

    for i in range(nx + 1):
        iL = i - 1 if i > 0 else 0
        iR = i if i < nx else nx - 1
        for j in range(ny):
            rho_L = U_L[0, i, j]
            u_L = U_L[1, i, j] / max(rho_L, 1e-30)
            v_L = U_L[2, i, j] / max(rho_L, 1e-30)
            E_L = U_L[3, i, j] / max(rho_L, 1e-30)
            Yf_L = U_L[4, i, j] / max(rho_L, 1e-30)
            p_L = (gamma - 1.0) * rho_L * (E_L - 0.5 * (u_L**2 + v_L**2))
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

            F0, F1, F2, F3, F4 = hllc_flux_kernel(
                rho_L, u_L, v_L, p_L, Yf_L,
                rho_R, u_R, v_R, p_R, Yf_R, gamma)

            w = A_face[i] * dy[j]
            if i < nx:
                inv = 1.0 / (A_cell[i] * vol[i, j])
                dUdt[0, i, j] += F0 * w * inv
                dUdt[1, i, j] += F1 * w * inv
                dUdt[2, i, j] += F2 * w * inv
                dUdt[3, i, j] += F3 * w * inv
                dUdt[4, i, j] += F4 * w * inv
            if i > 0:
                inv = 1.0 / (A_cell[i - 1] * vol[i - 1, j])
                dUdt[0, i - 1, j] -= F0 * w * inv
                dUdt[1, i - 1, j] -= F1 * w * inv
                dUdt[2, i - 1, j] -= F2 * w * inv
                dUdt[3, i - 1, j] -= F3 * w * inv
                dUdt[4, i - 1, j] -= F4 * w * inv


@njit(cache=True)
def accumulate_fluxes_j(U, U_L, U_R, dx, vol, gamma, dUdt):
    """
    Assemble J-face (y-direction) HLLC flux contributions into dUdt.

    The duct area A(x) is constant within a column, so it cancels from
    the y-direction terms and no area weighting is needed here.
    """
    n_vars, nx, nyp1 = U_L.shape
    ny = nyp1 - 1

    for i in range(nx):
        for j in range(ny + 1):
            jL = j - 1 if j > 0 else 0
            jR = j if j < ny else ny - 1

            rho_L = U_L[0, i, j]
            u_L = U_L[1, i, j] / max(rho_L, 1e-30)
            v_L = U_L[2, i, j] / max(rho_L, 1e-30)
            E_L = U_L[3, i, j] / max(rho_L, 1e-30)
            Yf_L = U_L[4, i, j] / max(rho_L, 1e-30)
            p_L = (gamma - 1.0) * rho_L * (E_L - 0.5 * (u_L**2 + v_L**2))
            if rho_L <= 1e-12 or p_L <= 1e-12:
                rho_L = max(U[0, i, jL], 1e-12)
                u_L = U[1, i, jL] / rho_L
                v_L = U[2, i, jL] / rho_L
                E_L = U[3, i, jL] / rho_L
                Yf_L = U[4, i, jL] / rho_L
                p_L = max((gamma - 1.0) * rho_L * (E_L - 0.5 * (u_L**2 + v_L**2)), 1e-12)

            rho_R = U_R[0, i, j]
            u_R = U_R[1, i, j] / max(rho_R, 1e-30)
            v_R = U_R[2, i, j] / max(rho_R, 1e-30)
            E_R = U_R[3, i, j] / max(rho_R, 1e-30)
            Yf_R = U_R[4, i, j] / max(rho_R, 1e-30)
            p_R = (gamma - 1.0) * rho_R * (E_R - 0.5 * (u_R**2 + v_R**2))
            if rho_R <= 1e-12 or p_R <= 1e-12:
                rho_R = max(U[0, i, jR], 1e-12)
                u_R = U[1, i, jR] / rho_R
                v_R = U[2, i, jR] / rho_R
                E_R = U[3, i, jR] / rho_R
                Yf_R = U[4, i, jR] / rho_R
                p_R = max((gamma - 1.0) * rho_R * (E_R - 0.5 * (u_R**2 + v_R**2)), 1e-12)

            G0, G1, G2, G3, G4 = hllc_flux_kernel_y(
                rho_L, u_L, v_L, p_L, Yf_L,
                rho_R, u_R, v_R, p_R, Yf_R, gamma)

            w = dx[i]
            if j < ny:
                inv = 1.0 / vol[i, j]
                dUdt[0, i, j] += G0 * w * inv
                dUdt[1, i, j] += G1 * w * inv
                dUdt[2, i, j] += G2 * w * inv
                dUdt[3, i, j] += G3 * w * inv
                dUdt[4, i, j] += G4 * w * inv
            if j > 0:
                inv = 1.0 / vol[i, j - 1]
                dUdt[0, i, j - 1] -= G0 * w * inv
                dUdt[1, i, j - 1] -= G1 * w * inv
                dUdt[2, i, j - 1] -= G2 * w * inv
                dUdt[3, i, j - 1] -= G3 * w * inv
                dUdt[4, i, j - 1] -= G4 * w * inv


class FVMResidual:
    """
    Assembles the semi-discrete FVM residual: dU/dt = -R(U).

    Plain 2D channel (geometry=None):
        R(U) = (1/V) * sum_faces(F_numerical * S_face)

    Quasi-1D duct (geometry given): the x-direction fluxes are weighted by
    the duct area A(x) evaluated at faces and the equations are solved in
    the area-weighted conservative form

        d(A U)/dt + d(A F)/dx + A dG/dy = S_geo,
        S_geo = [0, p dA/dx, 0, 0, 0] (+ the -U dA/dt term for breathing walls),

    discretised so that (i) rho*u*A telescopes exactly at steady state
    (discrete mass conservation) and (ii) a stagnant uniform-pressure state
    is an exact equilibrium (well-balanced: the momentum source uses the
    same face areas as the fluxes).

    Uses MUSCL reconstruction + HLLC flux for each face. The Venkatakrishnan
    smoothing eps^2 is computed per variable from the current solution scale,
    eps2_k = (K * h_rel)^3 * max|U_k|^2, with h_rel the grid spacing relative
    to the domain extent in the reconstruction direction.
    """

    def __init__(self, mesh, gamma=1.4, eps2_scale=1.0, geometry=None):
        """
        Args:
            mesh:       StructuredMesh2D instance
            gamma:      ratio of specific heats
            eps2_scale: Venkatakrishnan limiter parameter K
            geometry:   optional area-law object with area(x)/area_gradient(x)
                        (GeometryProfile-compatible). Enables the quasi-1D
                        variable-area coupling.
        """
        self.mesh = mesh
        self.gamma = gamma
        self.n_vars = StateVector.N_VARS
        self.eps2_scale = float(eps2_scale)

        # relative grid spacings for the limiter smoothing
        Lx = float(mesh.x_nodes[-1] - mesh.x_nodes[0])
        Ly = float(mesh.y_nodes[-1] - mesh.y_nodes[0])
        self._hx_rel = float(mesh.dx.max()) / max(Lx, 1e-30)
        self._hy_rel = float(mesh.dy.max()) / max(Ly, 1e-30)

        self.geometry = geometry
        self._geometry_time_dependent = bool(
            getattr(geometry, "is_time_dependent", False)) if geometry is not None else False
        self._refresh_area(time=None)

    def _refresh_area(self, time=None):
        """(Re)sample duct area at faces and cells; zero dA/dt if static."""
        mesh = self.mesh
        if self.geometry is None:
            self.A_face = np.ones(mesh.nx + 1)
            self.A_cell = np.ones(mesh.nx)
            self.dAdt_cell = np.zeros(mesh.nx)
            return

        if time is not None and hasattr(self.geometry, "set_time"):
            self.geometry.set_time(time)
        self.A_face = np.asarray(self.geometry.area(mesh.x_nodes), dtype=np.float64)
        self.A_cell = np.asarray(self.geometry.area(mesh.xc), dtype=np.float64)
        if hasattr(self.geometry, "area_time_derivative"):
            self.dAdt_cell = np.asarray(
                self.geometry.area_time_derivative(mesh.xc), dtype=np.float64)
        else:
            self.dAdt_cell = np.zeros(mesh.nx)

    def _eps2_vectors(self, U):
        """Per-variable Venkatakrishnan eps^2 for each sweep direction."""
        q_ref = np.empty(self.n_vars)
        for k in range(self.n_vars):
            q_ref[k] = np.max(np.abs(U[k]))
        eps2_i = (self.eps2_scale * self._hx_rel) ** 3 * q_ref**2
        eps2_j = (self.eps2_scale * self._hy_rel) ** 3 * q_ref**2
        return eps2_i, eps2_j

    def compute(self, U, time=None):
        """
        Evaluate dU/dt from the FVM convective operator (+ geometric sources).

        Args:
            U:    conservative state array, shape (5, nx, ny)
            time: physical time [s]; used to refresh a time-dependent
                  area law (breathing geometry). Ignored for static ducts.

        Returns:
            dUdt: shape (5, nx, ny)
        """
        nx, ny = self.mesh.nx, self.mesh.ny
        gamma = self.gamma
        n_vars = self.n_vars

        if self._geometry_time_dependent and time is not None:
            self._refresh_area(time)

        eps2_i, eps2_j = self._eps2_vectors(U)
        dUdt = np.zeros_like(U)

        U_L, U_R = muscl_reconstruct_i(U, nx, ny, n_vars, eps2_i)
        accumulate_fluxes_i(U, U_L, U_R, self.mesh.dy, self.mesh.vol,
                            self.A_face, self.A_cell, gamma, dUdt)

        U_L_j, U_R_j = muscl_reconstruct_j(U, nx, ny, n_vars, eps2_j)
        accumulate_fluxes_j(U, U_L_j, U_R_j, self.mesh.dx, self.mesh.vol,
                            gamma, dUdt)

        if self.geometry is not None:
            # well-balanced momentum source: p * (A_e - A_w) / (A_c * dx),
            # the discrete counterpart of p dA/dx in the area-weighted form
            rho_safe = np.maximum(U[0], 1e-30)
            ke = 0.5 * (U[1]**2 + U[2]**2) / rho_safe
            p = (gamma - 1.0) * (U[3] - ke)
            coeff = (self.A_face[1:] - self.A_face[:-1]) / (self.A_cell * self.mesh.dx)
            dUdt[1] += coeff[:, np.newaxis] * p

            # breathing-wall volume term: d(AU)/dt = A dU/dt + U dA/dt
            if np.any(self.dAdt_cell != 0.0):
                rate = (self.dAdt_cell / self.A_cell)[:, np.newaxis]
                dUdt -= rate * U

        return dUdt


class TimeIntegrator:
    """
    Strong-stability-preserving 3rd-order Runge-Kutta (Shu & Osher 1988).

    U^(1)   = U^n + dt * L(U^n)
    U^(2)   = 3/4 * U^n + 1/4 * (U^(1) + dt * L(U^(1)))
    U^(n+1) = 1/3 * U^n + 2/3 * (U^(2) + dt * L(U^(2)))

    where L(U) = dU/dt from the FVM residual + source terms. Stage times
    (t, t + dt, t + dt/2) are passed to the RHS and BC callables so
    time-dependent forcing (breathing area, modulated back pressure) is
    integrated at the correct stage instants.
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

    def step(self, state, dt, rhs_fn, bc_fn, t=0.0):
        """
        Advance state by one RK3-SSP step.

        Args:
            state:  StateVector instance
            dt:     time step size
            rhs_fn: callable(U, time) -> dUdt array, shape (5, nx, ny)
            bc_fn:  callable(U, time) -> None, applies BCs in-place
            t:      physical time at the start of the step
        """
        U_n = state.U.copy()

        # stage 1 (at t)
        bc_fn(state.U, t)
        k1 = rhs_fn(state.U, t)
        state.U = U_n + dt * k1

        # stage 2 (at t + dt)
        bc_fn(state.U, t + dt)
        k2 = rhs_fn(state.U, t + dt)
        state.U = 0.75 * U_n + 0.25 * (state.U + dt * k2)

        # stage 3 (at t + dt/2)
        bc_fn(state.U, t + 0.5 * dt)
        k3 = rhs_fn(state.U, t + 0.5 * dt)
        state.U = (1.0 / 3.0) * U_n + (2.0 / 3.0) * (state.U + dt * k3)

        # final BC enforcement at the new time level
        bc_fn(state.U, t + dt)


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