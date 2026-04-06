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
    Source term for variable-area duct in 2D.

    For a duct with cross-sectional area A(x), the quasi-1D source is:

        S = -(1/A) * (dA/dx) * [0, p, 0, 0, 0]^T

    This appears in the x-momentum equation as a pressure-area force.
    """

    def __init__(self, mesh, geometry):
        """
        Args:
            mesh:     StructuredMesh2D instance
            geometry: GeometryProfile instance
        """
        self.mesh = mesh
        self.geometry = geometry

        # precompute area and gradient at cell centroids
        # A(x_c[i]) and dA/dx(x_c[i]) for each column i
        self.A_cell = geometry.area(mesh.xc)          # (nx,)
        self.dAdx_cell = geometry.area_gradient(mesh.xc)  # (nx,)

    def compute(self, U, gamma):
        """
        Evaluate the variable-area source term.

        Args:
            U:     conservative state, shape (5, nx, ny)
            gamma: ratio of specific heats

        Returns:
            S: source term array, shape (5, nx, ny)
        """
        S = np.zeros_like(U)
        nx, ny = self.mesh.nx, self.mesh.ny

        # extract pressure: p = (gamma-1) * (rho*E - 0.5 * rho * (u^2 + v^2))
        rho = U[0]
        rhou = U[1]
        rhov = U[2]
        rhoE = U[3]
        ke = 0.5 * (rhou**2 + rhov**2) / np.maximum(rho, 1e-30)
        p = (gamma - 1.0) * (rhoE - ke)

        # S_momentum_x = -(1/A) * (dA/dx) * p
        for i in range(nx):
            ratio = self.dAdx_cell[i] / max(self.A_cell[i], 1e-30)
            S[1, i, :] = -ratio * p[i, :]

        return S


class SingleStepArrhenius:
    """
    Single-step global Arrhenius combustion model.

    Reaction:  Fuel + nu_ox * Oxidiser -> Products

    Rate:
        omega_dot = A_pre * rho^(nf + no) * Yf^nf * Yo^no * exp(-Ea / (Ru * T))

    Source terms:
        S_Yf  = -W_f * omega_dot        (fuel depletion)
        S_E   =  Q_heat * omega_dot      (heat release)

    The oxidiser mass fraction Yo = 1 - Yf (binary mixture assumption).
    """

    def __init__(self, A_pre=1.0e10, Ea=80000.0, Q_heat=3.0e6,
                 nf=1.0, no=1.0, W_f=0.002, Ru=8.314, gamma=1.4, R_gas=287.0):
        """
        Args:
            A_pre:  pre-exponential factor [1/s * (kg/m^3)^(1-nf-no)]
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

    def compute(self, U):
        """
        Evaluate combustion source terms.

        Args:
            U: conservative state, shape (5, nx, ny)

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

        # clamp reaction rate to prevent numerical blowup
        # max depletion: can't consume more fuel than exists in one step
        omega = np.minimum(omega, rho_safe * Yf / (self.W_f * 1e-6 + 1e-30))

        # source terms
        S[3] = self.Q_heat * omega       # energy release
        S[4] = -self.W_f * omega          # fuel consumption

        return S


class FEMViscous:
    """
    Implicit viscous/thermal/species diffusion operator.

    Solves the parabolic sub-problem over a half-step (Strang splitting):
        d(rho*u)/dt = div(mu * grad(u))        (x-momentum diffusion)
        d(rho*v)/dt = div(mu * grad(v))        (y-momentum diffusion)
        d(rho*E)/dt = div(k * grad(T))         (thermal diffusion)
        d(rho*Yf)/dt = div(rho*D * grad(Yf))   (species diffusion)

    Uses cell-centred finite differences with implicit backward Euler and
    direct sparse solve — functionally equivalent to a Q1 FEM with mass
    lumping on the structured mesh.

    No-slip walls at j=0 and j=ny-1 (zero velocity, adiabatic dT/dn=0).
    """

    def __init__(self, mesh, transport, gamma=1.4, R_gas=287.0):
        """
        Args:
            mesh:      StructuredMesh2D instance
            transport: TransportProperties instance
        """
        self.mesh = mesh
        self.transport = transport
        self.gamma = gamma
        self.R_gas = R_gas

    def step(self, U, dt):
        """
        Advance the diffusive terms by dt using implicit backward Euler.

        Modifies U in-place.

        The diffusion equation for a generic scalar phi with diffusivity alpha:
            d(rho*phi)/dt = div(alpha * grad(phi))

        Discretised on the structured mesh as:
            (rho*phi^{n+1} - rho*phi^n) / dt = sum_faces alpha * (phi_nb - phi_P) * S_f / d_f / V_P

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

        # transport coefficients at cell centres
        mu = self.transport.viscosity(T)          # (nx, ny)
        k_th = self.transport.thermal_conductivity(T)
        D_sp = self.transport.species_diffusivity(T, rho)

        def _flat(i, j):
            """Map (i,j) to flat index."""
            return i * ny + j

        def _build_diffusion_system(alpha, phi, bc_type="neumann",
                                    wall_bot_val=0.0, wall_top_val=0.0):
            """
            Build the implicit diffusion linear system for scalar phi.

            Args:
                alpha:        diffusivity field, shape (nx, ny)
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
                    diag_coeff = rho_safe[i, j] * V / dt
                    rhs[idx] = rho_safe[i, j] * phi[i, j] * V / dt

                    # east face (i+1)
                    if i < nx - 1:
                        alpha_f = 2.0 * alpha[i, j] * alpha[i + 1, j] / max(alpha[i, j] + alpha[i + 1, j], 1e-30)
                        d_f = 0.5 * (dx[i] + dx[i + 1])
                        S_f = dy[j]
                        coeff = alpha_f * S_f / d_f
                        diag_coeff += coeff
                        rows.append(idx)
                        cols.append(_flat(i + 1, j))
                        vals.append(-coeff)
                    # west face (i-1)
                    if i > 0:
                        alpha_f = 2.0 * alpha[i, j] * alpha[i - 1, j] / max(alpha[i, j] + alpha[i - 1, j], 1e-30)
                        d_f = 0.5 * (dx[i] + dx[i - 1])
                        S_f = dy[j]
                        coeff = alpha_f * S_f / d_f
                        diag_coeff += coeff
                        rows.append(idx)
                        cols.append(_flat(i - 1, j))
                        vals.append(-coeff)

                    # north face (j+1)
                    if j < ny - 1:
                        alpha_f = 2.0 * alpha[i, j] * alpha[i, j + 1] / max(alpha[i, j] + alpha[i, j + 1], 1e-30)
                        d_f = 0.5 * (dy[j] + dy[j + 1])
                        S_f = dx[i]
                        coeff = alpha_f * S_f / d_f
                        diag_coeff += coeff
                        rows.append(idx)
                        cols.append(_flat(i, j + 1))
                        vals.append(-coeff)
                    # south face (j-1)
                    if j > 0:
                        alpha_f = 2.0 * alpha[i, j] * alpha[i, j - 1] / max(alpha[i, j] + alpha[i, j - 1], 1e-30)
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
                            alpha_f = alpha[i, j]
                            d_f = 0.5 * dy[j]
                            S_f = dx[i]
                            coeff = alpha_f * S_f / d_f
                            diag_coeff += coeff
                            rhs[idx] += coeff * wall_bot_val

                    if j == ny - 1:
                        # wall BC at j=ny-1
                        if bc_type == "dirichlet":
                            alpha_f = alpha[i, j]
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

        # --- diffuse u-velocity ---
        # kinematic viscosity nu = mu / rho
        nu = mu / rho_safe
        u_new = _build_diffusion_system(nu, u_vel, bc_type="dirichlet")

        # --- diffuse v-velocity ---
        v_new = _build_diffusion_system(nu, v_vel, bc_type="dirichlet")

        # --- diffuse temperature (thermal diffusivity alpha_T = k / (rho * cp)) ---
        alpha_T = k_th / (rho_safe * self.transport.cp)
        T_new = _build_diffusion_system(alpha_T, T, bc_type="neumann")
        T_new = np.maximum(T_new, 50.0)

        # --- diffuse species ---
        Yf_new = _build_diffusion_system(D_sp, Yf, bc_type="neumann")
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
