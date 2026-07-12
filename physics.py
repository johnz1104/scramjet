"""
physics.py — Source terms and physical models for the scramjet solver.

Contains:
    TransportProperties  — Sutherland viscosity, thermal conductivity, species diffusion
    VariableAreaSource   — quasi-1D area variation source term
    SingleStepArrhenius  — single-step fuel-oxidiser combustion model
    FEMViscous           — implicit viscous/thermal/species diffusion (FEM-style)

Dependency: numpy, scipy.sparse (FEM only)
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


class TransportProperties:
    """
    Temperature-dependent transport coefficients.

    Viscosity via Sutherland's law:
        mu(T) = mu_ref * (T / T_ref)^(3/2) * (T_ref + S) / (T + S)

    Thermal conductivity from constant Prandtl number:
        k = mu * cp / Pr

    Species diffusivity from constant Schmidt number:
        D = mu / (rho * Sc)
    """

    def __init__(self, mu_ref=1.716e-5, T_ref=273.15, S=110.4,
                 Pr=0.72, Sc=0.9, gamma=1.4, R_gas=287.0):
        self.mu_ref = mu_ref
        self.T_ref = T_ref
        self.S = S
        self.Pr = Pr
        self.Sc = Sc
        self.gamma = gamma
        self.R_gas = R_gas

        # cp = gamma * R / (gamma - 1)
        self.cp = gamma * R_gas / (gamma - 1.0)

    def viscosity(self, T):
        """
        Dynamic viscosity [Pa s] via Sutherland's law.

        mu(T) = mu_ref * (T/T_ref)^1.5 * (T_ref + S) / (T + S)
        """
        T_safe = np.maximum(T, 50.0)  # floor to avoid division issues
        return self.mu_ref * (T_safe / self.T_ref)**1.5 * (self.T_ref + self.S) / (T_safe + self.S)

    def thermal_conductivity(self, T):
        """k = mu * cp / Pr  [W/(m K)]"""
        return self.viscosity(T) * self.cp / self.Pr

    def species_diffusivity(self, T, rho):
        """D = mu / (rho * Sc)  [m^2/s]"""
        return self.viscosity(T) / (np.maximum(rho, 1e-30) * self.Sc)


class VariableAreaSource:
    """
    Quasi-1D variable-area source term (standalone reference form).

    For a duct of area A(x), the quasi-1D equations written for the
    non-area-weighted state U = [rho, rho u, rho v, rho E, rho Yf] carry a
    geometric source in EVERY equation:

        S = -(1/A) (dA/dx) * u * [rho, rho u, rho v, rho E + p, rho Yf]
            -(1/A) (dA/dt) * [rho, rho u, rho v, rho E + p, rho Yf]

    which yields the correct area-Mach behavior du/u = (dA/A)/(M^2 - 1):
    a converging duct compresses and decelerates a supersonic stream.

    NOTE: the Solver's primary path is the area-weighted conservative
    coupling inside fvm.FVMResidual (discretely mass-conserving and
    well-balanced). This class is retained as an independent reference
    implementation of the same physics for cross-checks; the two agree
    to discretization error (see tests.py: nozzle test).

    History: an earlier version applied only S1 = -(A'/A) p to x-momentum,
    which is the mirror image of the correct relation (flow accelerated
    through contractions, and rho*u*A drifted ~2x along the duct).
    """

    def __init__(self, mesh, geometry, legacy_breathing_energy=False):
        """
        Args:
            mesh:     StructuredMesh2D instance
            geometry: GeometryProfile-compatible area law
        """
        self.mesh = mesh
        self.geometry = geometry
        self.legacy_breathing_energy = bool(legacy_breathing_energy)

        # precompute area and gradient at cell centroids
        # A(x_c[i]) and dA/dx(x_c[i]) for each column i
        self.A_cell = None
        self.dAdx_cell = None
        self.dAdt_cell = None
        self.update(0.0)

    def update(self, time):
        """Refresh area and gradient fields for static or time-dependent geometry."""
        if hasattr(self.geometry, "set_time"):
            self.geometry.set_time(time)
        self.A_cell = self.geometry.area(self.mesh.xc)          # (nx,)
        self.dAdx_cell = self.geometry.area_gradient(self.mesh.xc)  # (nx,)
        if hasattr(self.geometry, "area_time_derivative"):
            self.dAdt_cell = self.geometry.area_time_derivative(self.mesh.xc)
        else:
            self.dAdt_cell = np.zeros_like(self.A_cell)

    def compute(self, U, gamma, time=None):
        """
        Evaluate the variable-area source term.

        Args:
            U:     conservative state, shape (5, nx, ny)
            gamma: ratio of specific heats
            time:  optional physical time for time-dependent area forcing

        Returns:
            S: source term array, shape (5, nx, ny)
        """
        if time is not None or getattr(self.geometry, "is_time_dependent", False):
            self.update(0.0 if time is None else time)

        rho_safe = np.maximum(U[0], 1e-30)
        u = U[1] / rho_safe
        ke = 0.5 * (U[1]**2 + U[2]**2) / rho_safe
        p = (gamma - 1.0) * (U[3] - ke)

        ratio = (self.dAdx_cell / np.maximum(self.A_cell, 1e-30))[:, np.newaxis]
        S = np.empty_like(U)
        S[0] = -ratio * U[1]                # mass:      -(A'/A) rho u
        S[1] = -ratio * U[1] * u            # x-mom:     -(A'/A) rho u^2
        S[2] = -ratio * U[2] * u            # y-mom:     -(A'/A) rho u v
        S[3] = -ratio * (U[3] + p) * u      # energy:    -(A'/A) (rho E + p) u
        S[4] = -ratio * U[4] * u            # species:   -(A'/A) rho Yf u

        if np.any(self.dAdt_cell != 0.0):
            rate = (self.dAdt_cell / np.maximum(self.A_cell, 1e-30))[:, np.newaxis]
            S -= rate * U
            if not self.legacy_breathing_energy:
                S[3] -= rate * p

        return S


class SingleStepArrhenius:
    """
    Single-step global Arrhenius combustion model.

    Reaction:  Fuel + nu_ox * Oxidiser -> Products

    Rate:
        omega_dot = A_pre * rho^(nf + no) * Yf^nf * Yo^no * exp(-Ea / (Ru * T))
                    [mol/(m^3 s)]

    Source terms:
        S_Yf  = -W_f * omega_dot        (fuel depletion)
        S_E   =  Q_heat * W_f * omega_dot  (heat release)

    The oxidiser mass fraction Yo = 1 - Yf (binary mixture assumption).
    """

    def __init__(self, A_pre=1.0e10, Ea=80000.0, Q_heat=3.0e6,
                 nf=1.0, no=1.0, W_f=0.002, Ru=8.314, gamma=1.4, R_gas=287.0):
        """
        Args:
            A_pre:  pre-exponential factor with units chosen so omega_dot is
                    mol/(m^3 s) for the stated density/reaction orders
            Ea:     activation energy [J/mol]
            Q_heat: heat of reaction per unit mass of fuel [J/kg]
            nf, no: fuel and oxidiser reaction orders
            W_f:    fuel molecular weight [kg/mol]
            Ru:     universal gas constant [J/(mol K)]
        """
        self.A_pre = A_pre
        self.Ea = Ea
        self.Q_heat = Q_heat
        self.nf = nf
        self.no = no
        self.W_f = W_f
        self.Ru = Ru
        self.gamma = gamma
        self.R_gas = R_gas

    def compute(self, U, dt=None):
        """
        Evaluate combustion source terms.

        Args:
            U:  conservative state, shape (5, nx, ny)
            dt: current integration step [s].  When supplied, the reaction
                rate is limited so one step cannot consume more fuel than
                is locally available.  No arbitrary limiter is used when
                the caller does not know the time step.

        Returns:
            S: source array, shape (5, nx, ny)
        """
        S = np.zeros_like(U)

        rho = U[0]
        rhou = U[1]
        rhov = U[2]
        rhoE = U[3]
        rhoYf = U[4]

        # primitive extraction
        rho_safe = np.maximum(rho, 1e-30)
        u = rhou / rho_safe
        v = rhov / rho_safe
        Yf = np.clip(rhoYf / rho_safe, 0.0, 1.0)
        Yo = np.clip(1.0 - Yf, 0.0, 1.0)

        # temperature: T = p / (rho * R)
        ke = 0.5 * (u**2 + v**2)
        E = rhoE / rho_safe
        p = (self.gamma - 1.0) * rho * (E - ke)
        p = np.maximum(p, 1e-30)
        T = p / (rho_safe * self.R_gas)
        T = np.maximum(T, 50.0)

        # Arrhenius rate: omega = A * rho^(nf+no) * Yf^nf * Yo^no * exp(-Ea/(Ru*T))
        omega = (self.A_pre
                 * rho_safe**(self.nf + self.no)
                 * np.power(np.maximum(Yf, 0.0), self.nf)
                 * np.power(np.maximum(Yo, 0.0), self.no)
                 * np.exp(-self.Ea / (self.Ru * T)))

        # Maximum depletion: one numerical step cannot consume more fuel
        # than is locally available.  The old hard-coded 1 microsecond
        # limiter made chemistry depend on an unrelated hidden time scale.
        if dt is not None:
            dt = float(dt)
            if dt <= 0.0:
                raise ValueError("combustion dt must be positive")
            omega = np.minimum(
                omega, rho_safe * Yf / (self.W_f * dt + 1e-30),
            )

        # source terms
        S[3] = self.Q_heat * self.W_f * omega  # energy release [J/(m^3 s)]
        S[4] = -self.W_f * omega               # fuel consumption [kg/(m^3 s)]

        return S


class SimpleHeatRelease:
    """
    Reduced-fidelity prescribed heat-release source.

    This is a sensitivity-model source, not a combustion model. It adds energy
    at a prescribed volumetric rate, optionally weighted by the local passive
    fuel scalar. It does not consume fuel or model ignition chemistry.
    """

    def __init__(self, heat_rate=0.0, fuel_coupled=True):
        self.heat_rate = float(heat_rate)
        self.fuel_coupled = bool(fuel_coupled)

    def compute(self, U):
        """Return conservative source terms with energy-only heating."""
        S = np.zeros_like(U)
        if self.heat_rate == 0.0:
            return S

        if self.fuel_coupled:
            rho = np.maximum(U[0], 1e-30)
            Yf = np.clip(U[4] / rho, 0.0, 1.0)
            S[3] = self.heat_rate * Yf
        else:
            S[3] = self.heat_rate
        return S


class FEMViscous:
    """
    Implicit viscous/thermal/species diffusion operator.

    Solves the parabolic sub-problem over a half-step (Strang splitting):
        d(rho*u)/dt = div(mu * grad(u))        (x-momentum diffusion)
        d(rho*v)/dt = div(mu * grad(v))        (y-momentum diffusion)
        d(rho*E)/dt = div(k * grad(T))         (thermal diffusion)
        d(rho*Yf)/dt = div(rho*D * grad(Yf))   (species diffusion)

    Uses cell-centered finite differences with implicit backward Euler and
    direct sparse solve — functionally equivalent to a Q1 FEM with mass
    lumping on the structured mesh.

    No-slip walls at j=0 and j=ny-1 (zero normal velocity, adiabatic
    dT/dn=0). The walls may translate in x: `wall_u_bottom`/`wall_u_top`
    set the tangential wall speed (Couette-type driving).
    """

    def __init__(self, mesh, transport, gamma=1.4, R_gas=287.0,
                 wall_u_bottom=0.0, wall_u_top=0.0):
        """
        Args:
            mesh:      StructuredMesh2D instance
            transport: TransportProperties instance
            wall_u_bottom, wall_u_top: tangential wall velocities [m/s]
        """
        self.mesh = mesh
        self.transport = transport
        self.gamma = gamma
        self.R_gas = R_gas
        self.wall_u_bottom = float(wall_u_bottom)
        self.wall_u_top = float(wall_u_top)

    def step(self, U, dt):
        """
        Advance the diffusive terms by dt using implicit backward Euler.

        Modifies U in-place.

        The diffusion equation for a generic scalar phi is written with a
        mass coefficient m and dynamic flux coefficient Gamma:
            m d(phi)/dt = div(Gamma * grad(phi))

        Discretised on the structured mesh as:
            m (phi^{n+1} - phi^n) / dt =
                sum_faces Gamma * (phi_nb - phi_P) * S_f / d_f / V_P

        Rearranged into a linear system A * phi^{n+1} = b.
        """
        nx, ny = self.mesh.nx, self.mesh.ny
        n_cells = nx * ny
        dx, dy = self.mesh.dx, self.mesh.dy
        vol = self.mesh.vol

        # extract primitives for transport coefficients
        rho = U[0]
        rho_safe = np.maximum(rho, 1e-30)
        u_vel = U[1] / rho_safe
        v_vel = U[2] / rho_safe
        ke = 0.5 * (u_vel**2 + v_vel**2)
        E = U[3] / rho_safe
        p = (self.gamma - 1.0) * rho * (E - ke)
        p = np.maximum(p, 1e-30)
        T = p / (rho_safe * self.R_gas)
        T = np.maximum(T, 50.0)
        Yf = np.clip(U[4] / rho_safe, 0.0, 1.0)

        # transport coefficients at cell centers
        mu = self.transport.viscosity(T)          # (nx, ny)
        k_th = self.transport.thermal_conductivity(T)
        D_sp = self.transport.species_diffusivity(T, rho_safe)

        def _flat(i, j):
            """Map (i,j) to flat index."""
            return i * ny + j

        def _build_diffusion_system(flux_coeff, phi, mass_coeff,
                                    bc_type="neumann",
                                    wall_bot_val=0.0, wall_top_val=0.0):
            """
            Build the implicit diffusion linear system for scalar phi.

            Args:
                flux_coeff:   dynamic diffusion coefficient Gamma
                mass_coeff:   transient coefficient m
                phi:          scalar field to diffuse, shape (nx, ny)
                bc_type:      "neumann" (zero-gradient) or "dirichlet"
                wall_bot_val: Dirichlet value at j=0 wall (used if bc_type="dirichlet")
                wall_top_val: Dirichlet value at j=ny-1 wall

            Returns:
                phi_new: diffused field, shape (nx, ny)
            """
            rows = []
            cols = []
            vals = []
            rhs = np.zeros(n_cells)

            for i in range(nx):
                for j in range(ny):
                    idx = _flat(i, j)
                    V = vol[i, j]
                    diag_coeff = mass_coeff[i, j] * V / dt
                    rhs[idx] = mass_coeff[i, j] * phi[i, j] * V / dt

                    # east face (i+1)
                    if i < nx - 1:
                        alpha_f = 2.0 * flux_coeff[i, j] * flux_coeff[i + 1, j] / max(flux_coeff[i, j] + flux_coeff[i + 1, j], 1e-30)
                        d_f = 0.5 * (dx[i] + dx[i + 1])
                        S_f = dy[j]
                        coeff = alpha_f * S_f / d_f
                        diag_coeff += coeff
                        rows.append(idx)
                        cols.append(_flat(i + 1, j))
                        vals.append(-coeff)
                    # west face (i-1)
                    if i > 0:
                        alpha_f = 2.0 * flux_coeff[i, j] * flux_coeff[i - 1, j] / max(flux_coeff[i, j] + flux_coeff[i - 1, j], 1e-30)
                        d_f = 0.5 * (dx[i] + dx[i - 1])
                        S_f = dy[j]
                        coeff = alpha_f * S_f / d_f
                        diag_coeff += coeff
                        rows.append(idx)
                        cols.append(_flat(i - 1, j))
                        vals.append(-coeff)

                    # north face (j+1)
                    if j < ny - 1:
                        alpha_f = 2.0 * flux_coeff[i, j] * flux_coeff[i, j + 1] / max(flux_coeff[i, j] + flux_coeff[i, j + 1], 1e-30)
                        d_f = 0.5 * (dy[j] + dy[j + 1])
                        S_f = dx[i]
                        coeff = alpha_f * S_f / d_f
                        diag_coeff += coeff
                        rows.append(idx)
                        cols.append(_flat(i, j + 1))
                        vals.append(-coeff)
                    # south face (j-1)
                    if j > 0:
                        alpha_f = 2.0 * flux_coeff[i, j] * flux_coeff[i, j - 1] / max(flux_coeff[i, j] + flux_coeff[i, j - 1], 1e-30)
                        d_f = 0.5 * (dy[j] + dy[j - 1])
                        S_f = dx[i]
                        coeff = alpha_f * S_f / d_f
                        diag_coeff += coeff
                        rows.append(idx)
                        cols.append(_flat(i, j - 1))
                        vals.append(-coeff)
                    else:
                        # wall BC at j=0
                        if bc_type == "dirichlet":
                            # ghost: phi_ghost = 2*wall_val - phi_P
                            # flux = alpha * (phi_ghost - phi_P) * S / d
                            #      = alpha * (2*wall_val - 2*phi_P) * S / dy
                            alpha_f = flux_coeff[i, j]
                            d_f = 0.5 * dy[j]
                            S_f = dx[i]
                            coeff = alpha_f * S_f / d_f
                            diag_coeff += coeff
                            rhs[idx] += coeff * wall_bot_val

                    if j == ny - 1:
                        # wall BC at j=ny-1
                        if bc_type == "dirichlet":
                            alpha_f = flux_coeff[i, j]
                            d_f = 0.5 * dy[j]
                            S_f = dx[i]
                            coeff = alpha_f * S_f / d_f
                            diag_coeff += coeff
                            rhs[idx] += coeff * wall_top_val

                    # diagonal entry
                    rows.append(idx)
                    cols.append(idx)
                    vals.append(diag_coeff)

            A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))
            phi_flat = spsolve(A, rhs)
            return phi_flat.reshape(nx, ny)

        # --- diffuse u-velocity (tangential wall speed as Dirichlet value) ---
        u_new = _build_diffusion_system(mu, u_vel, rho_safe,
                                        bc_type="dirichlet",
                                        wall_bot_val=self.wall_u_bottom,
                                        wall_top_val=self.wall_u_top)

        # --- diffuse v-velocity ---
        v_new = _build_diffusion_system(
            mu, v_vel, rho_safe, bc_type="dirichlet",
        )

        # --- diffuse temperature: rho*cp dT/dt = div(k grad(T)) ---
        T_new = _build_diffusion_system(
            k_th, T, rho_safe * self.transport.cp, bc_type="neumann",
        )
        T_new = np.maximum(T_new, 50.0)

        # --- diffuse species ---
        Yf_new = _build_diffusion_system(
            rho_safe * D_sp, Yf, rho_safe, bc_type="neumann",
        )
        Yf_new = np.clip(Yf_new, 0.0, 1.0)

        # reconstruct conservative variables from updated primitives
        # keep rho unchanged (continuity is handled by FVM)
        p_new = rho_safe * self.R_gas * T_new
        E_new = p_new / ((self.gamma - 1.0) * rho_safe) + 0.5 * (u_new**2 + v_new**2)

        U[1] = rho * u_new
        U[2] = rho * v_new
        U[3] = rho * E_new
        U[4] = rho * Yf_new


if __name__ == "__main__":
    print("=== TransportProperties ===")
    tp = TransportProperties()
    T_test = np.array([300.0, 1000.0, 3000.0])
    mu_test = tp.viscosity(T_test)
    k_test = tp.thermal_conductivity(T_test)
    print(f"  T = {T_test}")
    print(f"  mu = {mu_test}")
    print(f"  k  = {k_test}")

    print("\n=== SingleStepArrhenius ===")
    comb = SingleStepArrhenius()
    # create a simple test state
    nx, ny = 5, 3
    U_test = np.zeros((5, nx, ny))
    U_test[0] = 1.0                        # rho = 1 kg/m^3
    U_test[1] = 1.0 * 100.0                # rho*u = 100 m/s
    U_test[2] = 0.0                        # rho*v = 0
    p_test = 101325.0
    E_test = p_test / (0.4 * 1.0) + 0.5 * 100.0**2
    U_test[3] = 1.0 * E_test               # rho*E
    U_test[4] = 1.0 * 0.05                 # rho*Yf (5% fuel)

    S = comb.compute(U_test)
    print(f"  S_energy max: {S[3].max():.2e}")
    print(f"  S_Yf min:     {S[4].min():.2e}")

    print("\nAll physics tests passed.")
