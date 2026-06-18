"""
solver.py — Configuration and top-level orchestrator for the scramjet solver.

Contains:
    InletConfig          — freestream / atmospheric conditions
    MeshConfig           — mesh resolution and stretching
    CombustionConfig     — Arrhenius parameters
    SolverConfig         — top-level container for all sub-configs
    Solver               — time-marching loop with Strang splitting

Dependency: mesh.py, fvm.py, physics.py
"""
import numpy as np
import matplotlib.pyplot as plt

from mesh import StructuredMesh2D, GeometryProfile
from fvm import StateVector, BoundaryConditions, FVMResidual, TimeIntegrator
from physics import (
    TransportProperties,
    VariableAreaSource,
    SingleStepArrhenius,
    SimpleHeatRelease,
    FEMViscous,
)


class InletConfig:
    """
    Freestream / inlet conditions, parameterised for altitude sweeps.

    Standard atmosphere model (simplified):
        T(h) = T_SL - L * h           (troposphere, h < 11 km)
        T(h) = T_trop                  (stratosphere, 11-25 km)
        T(h) = T_trop + L2 * (h - 25) (upper strat, 25-47 km)
        p(h) = p_SL * (T/T_SL)^(g/(L*R))
    """

    def __init__(self, mach=6.0, altitude=25000.0, gamma=1.4, R_gas=287.0,
                 Yf_inlet=0.0):
        self.mach = mach
        self.altitude = altitude
        self.gamma = gamma
        self.R_gas = R_gas
        self.Yf_inlet = Yf_inlet

        self._compute_freestream()

    def _compute_freestream(self):
        """Compute freestream rho, u, v, p, T from Mach and altitude."""
        h = self.altitude
        g = 9.80665
        R = self.R_gas
        gamma = self.gamma

        # simplified standard atmosphere
        T_SL = 288.15   # sea level temperature [K]
        p_SL = 101325.0  # sea level pressure [Pa]
        L = 0.0065       # lapse rate [K/m] (troposphere)

        if h <= 11000.0:
            T = T_SL - L * h
            p = p_SL * (T / T_SL) ** (g / (L * R))
        elif h <= 25000.0:
            # tropopause: isothermal at T_trop
            T_trop = T_SL - L * 11000.0  # ~216.65 K
            p_trop = p_SL * (T_trop / T_SL) ** (g / (L * R))
            T = T_trop
            p = p_trop * np.exp(-g * (h - 11000.0) / (R * T_trop))
        else:
            # upper stratosphere: slight warming
            T_trop = T_SL - L * 11000.0
            p_trop = p_SL * (T_trop / T_SL) ** (g / (L * R))
            p_25 = p_trop * np.exp(-g * 14000.0 / (R * T_trop))
            L2 = 0.003  # upper stratospheric lapse rate [K/m]
            T = T_trop + L2 * (h - 25000.0)
            p = p_25 * (T / T_trop) ** (-g / (L2 * R))

        self.T_inf = T
        self.p_inf = p
        self.rho_inf = p / (R * T)
        self.c_inf = np.sqrt(gamma * R * T)
        self.u_inf = self.mach * self.c_inf
        self.v_inf = 0.0


class MeshConfig:
    """Mesh resolution and geometry parameters."""

    def __init__(self, nx=100, ny=20, y_stretch=1.0):
        """
        Args:
            nx, ny:     number of cells in x, y
            y_stretch:  geometric stretching ratio (1.0 = uniform)
        """
        self.nx = nx
        self.ny = ny
        self.y_stretch = y_stretch


class CombustionConfig:
    """Arrhenius combustion parameters."""

    def __init__(self, enabled=False, A_pre=1.0e10, Ea=80000.0,
                 Q_heat=3.0e6, nf=1.0, no=1.0, W_f=0.002):
        self.enabled = enabled
        self.A_pre = A_pre
        self.Ea = Ea
        self.Q_heat = Q_heat
        self.nf = nf
        self.no = no
        self.W_f = W_f


class SolverConfig:
    """Top-level configuration container."""

    def __init__(self):
        self.inlet = InletConfig()
        self.mesh = MeshConfig()
        self.geometry = GeometryProfile.default()
        self.combustion = CombustionConfig()

        # time stepping
        self.cfl = 0.4
        self.n_steps = 2000
        self.t_final = None
        self.print_interval = 200

        # physics toggles
        self.viscous = False
        self.wall_type = "slip"   # "slip" or "no_slip"
        self.passive_scalar_enabled = False
        self.heat_release_model = "none"  # "none", "passive", or "simple"
        self.simple_heat_release_rate = 0.0
        self.simple_heat_release_fuel_coupled = True
        self.turbulence_model = "none"

        # variable-area source
        self.area_source = True


class Solver:
    """
    Top-level solver orchestrator.

    Time-marching loop with optional Strang operator splitting:

        If inviscid only (viscous=False):
            U^{n+1} = L_FVM(dt) U^n  +  dt * S_area  +  dt * S_comb

        If viscous (viscous=True), Strang splitting:
            U^{n+1} = L_FEM(dt/2) . L_FVM(dt) . L_FEM(dt/2) . U^n
            where L_FVM includes area and combustion sources.
    """

    def __init__(self, config):
        """
        Args:
            config: SolverConfig instance
        """
        self.cfg = config
        self._validate_model_flags()

        # build mesh
        geom = config.geometry
        if config.mesh.y_stretch > 1.001:
            self.mesh = StructuredMesh2D.stretched(
                0.0, geom.L_total, 0.0, geom.A_exit,
                config.mesh.nx, config.mesh.ny,
                y_ratio=config.mesh.y_stretch,
            )
        else:
            self.mesh = StructuredMesh2D.uniform(
                0.0, geom.L_total, 0.0, geom.A_exit,
                config.mesh.nx, config.mesh.ny,
            )

        # state vector
        self.state = StateVector(self.mesh.nx, self.mesh.ny,
                                  gamma=config.inlet.gamma,
                                  R_gas=config.inlet.R_gas)

        # initial condition: uniform freestream
        inlet = config.inlet
        rho_init = np.full((self.mesh.nx, self.mesh.ny), inlet.rho_inf)
        u_init = np.full((self.mesh.nx, self.mesh.ny), inlet.u_inf)
        v_init = np.zeros((self.mesh.nx, self.mesh.ny))
        p_init = np.full((self.mesh.nx, self.mesh.ny), inlet.p_inf)
        Yf_init = np.full((self.mesh.nx, self.mesh.ny), inlet.Yf_inlet)
        self.state.set_primitive(rho_init, u_init, v_init, p_init, Yf_init)

        # boundary conditions
        self.bc = BoundaryConditions(
            self.state, inlet.rho_inf, inlet.u_inf, inlet.v_inf,
            inlet.p_inf, inlet.Yf_inlet,
            wall_type=config.wall_type,
        )

        # FVM residual
        self.fvm_residual = FVMResidual(self.mesh, gamma=inlet.gamma)

        # source terms
        self.area_source = None
        if config.area_source:
            self.area_source = VariableAreaSource(self.mesh, config.geometry)

        self.combustion = None
        if config.combustion.enabled:
            cc = config.combustion
            self.combustion = SingleStepArrhenius(
                A_pre=cc.A_pre, Ea=cc.Ea, Q_heat=cc.Q_heat,
                nf=cc.nf, no=cc.no, W_f=cc.W_f,
                gamma=inlet.gamma, R_gas=inlet.R_gas,
            )

        self.simple_heat_release = None
        if getattr(config, "heat_release_model", "none") == "simple":
            self.simple_heat_release = SimpleHeatRelease(
                heat_rate=getattr(config, "simple_heat_release_rate", 0.0),
                fuel_coupled=getattr(config, "simple_heat_release_fuel_coupled", True),
            )

        # viscous operator
        self.fem_viscous = None
        if config.viscous:
            transport = TransportProperties(gamma=inlet.gamma, R_gas=inlet.R_gas)
            self.fem_viscous = FEMViscous(self.mesh, transport,
                                           gamma=inlet.gamma, R_gas=inlet.R_gas)

        # time integrator
        self.integrator = TimeIntegrator(cfl=config.cfl)

        # diagnostics
        self.time = 0.0
        self.step_count = 0
        self.dt_history = []
        self.residual_history = []

    def _validate_model_flags(self):
        """Validate reduced-fidelity feature flags."""
        heat_release_model = getattr(self.cfg, "heat_release_model", "none")
        turbulence_model = getattr(self.cfg, "turbulence_model", "none")
        if heat_release_model not in ("none", "passive", "simple"):
            raise ValueError(
                "heat_release_model must be 'none', 'passive', or 'simple'"
            )
        if turbulence_model != "none":
            raise NotImplementedError(
                f"turbulence_model='{turbulence_model}' is not implemented "
                "in the Python prototype; use OpenFOAM/FUN3D for RANS/LES."
            )
        if self.cfg.combustion.enabled and heat_release_model != "none":
            raise ValueError(
                "Use either CombustionConfig.enabled or heat_release_model, not both"
            )

    def _rhs(self, U):
        """
        Full right-hand-side: FVM convective residual + source terms.

        This is the L_FVM operator in the Strang splitting.
        """
        dUdt = self.fvm_residual.compute(U)

        if self.area_source is not None:
            dUdt += self.area_source.compute(U, self.cfg.inlet.gamma, time=self.time)

        if self.combustion is not None:
            dUdt += self.combustion.compute(U)

        if self.simple_heat_release is not None:
            dUdt += self.simple_heat_release.compute(U)

        return dUdt

    def _bc(self, U):
        """Apply boundary conditions in-place."""
        self.bc.apply(U)

    def compute_dt(self, t_final=None):
        """Compute the next CFL-limited time step."""
        dt = self.integrator.compute_dt(self.state, self.mesh)

        # viscous CFL constraint (explicit diffusive stability)
        if self.fem_viscous is not None:
            # for implicit FEM, the viscous step is unconditionally stable
            # but we limit dt to avoid large temporal splitting error
            rho, u, v, p, T, Yf = self.state.primitives()
            mu_max = self.fem_viscous.transport.viscosity(T).max()
            rho_min = np.maximum(rho.min(), 1e-30)
            dx_min = min(self.mesh.dx.min(), self.mesh.dy.min())
            # diffusive CFL: dt_diff = 0.5 * rho * dx^2 / mu
            dt_diff = 0.5 * rho_min * dx_min**2 / max(mu_max, 1e-30)
            dt = min(dt, dt_diff)

        if t_final is not None:
            remaining = float(t_final) - self.time
            if remaining <= 0.0:
                return 0.0
            dt = min(dt, remaining)

        return max(dt, 1e-15)

    def advance_one_step(self, dt=None, t_final=None):
        """Advance by one solver step and return the step size used."""
        if dt is None:
            dt = self.compute_dt(t_final=t_final)
        if dt <= 0.0:
            return 0.0

        self.dt_history.append(dt)

        if self.area_source is not None:
            self.area_source.update(self.time)

        # --- Strang splitting ---
        if self.fem_viscous is not None:
            # half-step viscous (L_FEM(dt/2))
            self.fem_viscous.step(self.state.U, 0.5 * dt)

        # full-step convective + sources (L_FVM(dt))
        self.integrator.step(self.state, dt, self._rhs, self._bc)

        if self.fem_viscous is not None:
            # half-step viscous (L_FEM(dt/2))
            self.fem_viscous.step(self.state.U, 0.5 * dt)

        self.time += dt
        self.step_count += 1
        return dt

    def run(self, n_steps=None, t_final=None, step_callback=None):
        """Execute the time-marching loop."""
        cfg = self.cfg
        if n_steps is None:
            n_steps = cfg.n_steps
        if t_final is None:
            t_final = cfg.t_final

        print(f"Scramjet CFD Solver")
        print(f"  Mach = {cfg.inlet.mach:.1f}, alt = {cfg.inlet.altitude:.0f} m")
        print(f"  Mesh: {self.mesh.nx} x {self.mesh.ny} = {self.mesh.n_cells} cells")
        print(f"  Viscous: {cfg.viscous}, Combustion: {cfg.combustion.enabled}")
        if t_final is None:
            print(f"  Steps: {n_steps}, CFL: {cfg.cfl}")
        else:
            print(f"  Steps max: {n_steps}, t_final: {t_final:.4e} s, CFL: {cfg.cfl}")
        print("-" * 60)

        for n in range(n_steps):
            if t_final is not None and self.time >= t_final:
                break

            dt = self.advance_one_step(t_final=t_final)
            if dt <= 0.0:
                break

            # residual monitoring
            if cfg.print_interval and n % cfg.print_interval == 0:
                rho, u, v, p, T, Yf = self.state.primitives()
                M = self.state.mach()
                res = np.sqrt(np.mean(self._rhs(self.state.U)**2))
                self.residual_history.append(res)

                print(f"  step {n:5d} | t = {self.time:.4e} s | dt = {dt:.3e} s "
                      f"| M_max = {M.max():.3f} | p_max = {p.max():.1f} Pa "
                      f"| res = {res:.3e}")

            if step_callback is not None:
                step_callback(self)

        print("-" * 60)
        print(f"Done. {self.step_count} steps, t_final = {self.time:.4e} s")

    def plot_mach(self):
        """2D contour plot of Mach number."""
        M = self.state.mach()
        fig, ax = plt.subplots(figsize=(12, 4))
        xc, yc = np.meshgrid(self.mesh.xc, self.mesh.yc, indexing="ij")
        c = ax.contourf(xc, yc, M, levels=50, cmap="jet")
        plt.colorbar(c, ax=ax, label="Mach")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Mach number (t = {self.time:.4e} s)")
        ax.set_aspect("equal")
        plt.tight_layout()
        return fig

    def plot_field(self, field_name="pressure"):
        """2D contour plot of a chosen field."""
        rho, u, v, p, T, Yf = self.state.primitives()
        fields = {"density": rho, "pressure": p, "temperature": T,
                  "u_velocity": u, "v_velocity": v, "fuel_fraction": Yf}
        data = fields.get(field_name, p)
        label = field_name

        fig, ax = plt.subplots(figsize=(12, 4))
        xc, yc = np.meshgrid(self.mesh.xc, self.mesh.yc, indexing="ij")
        c = ax.contourf(xc, yc, data, levels=50, cmap="viridis")
        plt.colorbar(c, ax=ax, label=label)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{field_name} (t = {self.time:.4e} s)")
        ax.set_aspect("equal")
        plt.tight_layout()
        return fig

    def plot_centerline(self):
        """Line plots of key variables along the domain centerline (j = ny//2)."""
        rho, u, v, p, T, Yf = self.state.primitives()
        M = self.state.mach()
        j_mid = self.mesh.ny // 2
        x = self.mesh.xc

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(x, M[:, j_mid], "b-")
        axes[0, 0].set_ylabel("Mach")
        axes[0, 0].set_title("Centerline Mach")

        axes[0, 1].plot(x, p[:, j_mid], "r-")
        axes[0, 1].set_ylabel("Pressure [Pa]")
        axes[0, 1].set_title("Centerline Pressure")

        axes[1, 0].plot(x, T[:, j_mid], "g-")
        axes[1, 0].set_ylabel("Temperature [K]")
        axes[1, 0].set_title("Centerline Temperature")

        axes[1, 1].plot(x, rho[:, j_mid], "k-")
        axes[1, 1].set_ylabel("Density [kg/m³]")
        axes[1, 1].set_title("Centerline Density")

        for ax in axes.flat:
            ax.set_xlabel("x [m]")
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Centerline profiles (t = {self.time:.4e} s)")
        plt.tight_layout()
        return fig

    def plot_residual(self):
        """Plot residual history."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(range(len(self.residual_history)), self.residual_history, "b-o", ms=3)
        ax.set_xlabel("Print interval")
        ax.set_ylabel("L2 residual")
        ax.set_title("Convergence history")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    print("=== InletConfig ===")
    inlet = InletConfig(mach=6.0, altitude=25000.0)
    print(f"  T_inf = {inlet.T_inf:.2f} K")
    print(f"  p_inf = {inlet.p_inf:.2f} Pa")
    print(f"  rho_inf = {inlet.rho_inf:.4f} kg/m^3")
    print(f"  u_inf = {inlet.u_inf:.1f} m/s")
    print(f"  c_inf = {inlet.c_inf:.1f} m/s")

    print("\n=== Quick solver test (inviscid, no combustion, 50 steps) ===")
    cfg = SolverConfig()
    cfg.mesh.nx = 40
    cfg.mesh.ny = 8
    cfg.n_steps = 50
    cfg.print_interval = 25

    solver = Solver(cfg)
    solver.run()

    M = solver.state.mach()
    print(f"  Final Mach range: [{M.min():.3f}, {M.max():.3f}]")

    print("\nSolver standalone test passed.")
