"""
rom.py — Proper Orthogonal Decomposition (POD) Reduced-Order Model.

Contains:
    SnapshotCollector    — runs the full solver at sampled parameter points, stores state fields
    PODBasis             — SVD decomposition, mode truncation, energy diagnostics
    ReducedSolver        — Galerkin-projected online evaluation (r x r system)
    ROMEvaluator         — end-to-end: basis + online solve + QoI extraction

The ROM operates in **parametric** mode: POD is taken over snapshots at
different design parameters (not over timesteps of a single run). This
maps the steady-state QoI landscape cheaply for the optimiser.

Dependency: solver.py (full-order model), numpy, scipy
"""
import time
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

from mesh import GeometryProfile
from solver import SolverConfig, Solver


class SnapshotCollector:
    """
    Runs the full solver at sampled parameter configurations and stores
    the converged state fields as columns of a snapshot matrix.

    Each snapshot is the flattened conservative state vector U.ravel(),
    so the snapshot matrix has shape (n_dof, n_snapshots) where
    n_dof = 5 * nx * ny.
    """

    def __init__(self, base_config):
        """
        Args:
            base_config: SolverConfig instance used as the template.
                         Geometry and inlet parameters will be overridden
                         per sample point.
        """
        self.base_config = base_config
        self.snapshots = []        # list of 1-D arrays (n_dof,)
        self.params = []           # list of parameter dicts
        self.qoi = []              # list of QoI dicts (thrust, Isp, ...)
        self.wall_times = []       # seconds per full solve

    def sample_and_run(self, param_list):
        """
        Run the full solver at each parameter configuration.

        Args:
            param_list: list of dicts, each with keys matching the
                        design variables. Supported keys:
                        - 'L_inlet', 'L_combustor', 'L_nozzle'
                        - 'A_inlet', 'A_throat', 'A_comb_exit', 'A_exit'
                        - 'mach', 'altitude'

        Returns:
            n_collected: number of successful snapshots
        """
        n_collected = 0
        for idx, params in enumerate(param_list):
            print(f"  Snapshot {idx+1}/{len(param_list)}: {params}")

            cfg = _clone_config(self.base_config)
            _apply_params(cfg, params)

            solver = Solver(cfg)
            t0 = time.time()
            solver.run()
            wall = time.time() - t0

            # check for NaN blowup
            if np.any(np.isnan(solver.state.U)):
                print(f"    SKIPPED (NaN blowup)")
                continue

            # store snapshot (flattened conservative state)
            self.snapshots.append(solver.state.U.ravel().copy())
            self.params.append(params)
            self.wall_times.append(wall)

            # extract QoI
            qoi = _compute_qoi(solver)
            self.qoi.append(qoi)
            n_collected += 1

            print(f"    thrust={qoi['thrust']:.2f} N, "
                  f"Isp={qoi['Isp']:.1f} s, wall={wall:.2f} s")

        return n_collected

    def snapshot_matrix(self):
        """
        Assemble snapshot matrix Phi of shape (n_dof, n_snapshots).
        Each column is a converged state at one parameter configuration.
        """
        return np.column_stack(self.snapshots)

    def mean_wall_time(self):
        """Average full-solver wall time per snapshot [s]."""
        if len(self.wall_times) == 0:
            return 0.0
        return np.mean(self.wall_times)


class PODBasis:
    """
    Proper Orthogonal Decomposition via truncated SVD.

    Given snapshot matrix Phi = [u_1 | u_2 | ... | u_N]:
        1. Subtract mean: Phi_fluct = Phi - mean(Phi)
        2. SVD: Phi_fluct = W * Sigma * V^T
        3. Retain r modes capturing >= energy_threshold of total energy
        4. POD basis: Psi_r = W[:, :r]

    Energy fraction of mode k: sigma_k^2 / sum(sigma^2)
    """

    def __init__(self, energy_threshold=0.999):
        """
        Args:
            energy_threshold: cumulative energy fraction to retain (default 99.9%)
        """
        self.energy_threshold = energy_threshold
        self.mean_state = None    # (n_dof,)
        self.basis = None         # (n_dof, r) — the POD modes Psi_r
        self.singular_values = None  # (r,)
        self.n_modes = 0
        self.n_dof = 0
        self.cumulative_energy = None  # (n_snapshots,)

    def build(self, snapshot_matrix):
        """
        Compute POD basis from snapshot matrix.

        Args:
            snapshot_matrix: shape (n_dof, n_snapshots)

        Returns:
            r: number of retained modes
        """
        n_dof, n_snap = snapshot_matrix.shape
        self.n_dof = n_dof

        # subtract mean
        self.mean_state = np.mean(snapshot_matrix, axis=1)
        fluct = snapshot_matrix - self.mean_state[:, np.newaxis]

        # SVD: fluct = W * Sigma * V^T
        # use full_matrices=False for efficiency (W is n_dof x n_snap)
        W, sigma, Vt = svd(fluct, full_matrices=False)

        # cumulative energy fraction
        energy = sigma**2
        total_energy = np.sum(energy)
        self.cumulative_energy = np.cumsum(energy) / total_energy

        # determine number of modes to retain
        r = np.searchsorted(self.cumulative_energy, self.energy_threshold) + 1
        r = min(r, len(sigma))
        self.n_modes = r

        self.basis = W[:, :r].copy()
        self.singular_values = sigma[:r].copy()

        pct = self.cumulative_energy[r - 1] * 100
        print(f"  POD: {r} modes capture {pct:.2f}% energy "
              f"(threshold {self.energy_threshold*100:.1f}%)")

        return r

    def project(self, state_flat):
        """
        Project a full-order state onto the reduced basis.

        a = Psi_r^T * (u - u_mean)

        Args:
            state_flat: full state vector, shape (n_dof,)

        Returns:
            a: reduced coefficients, shape (r,)
        """
        return self.basis.T @ (state_flat - self.mean_state)

    def reconstruct(self, a):
        """
        Reconstruct full-order state from reduced coefficients.

        u_approx = u_mean + Psi_r * a

        Args:
            a: reduced coefficients, shape (r,)

        Returns:
            state_flat: reconstructed state, shape (n_dof,)
        """
        return self.mean_state + self.basis @ a

    def reconstruction_error(self, state_flat):
        """
        Relative L2 reconstruction error for a single state.

        err = ||u - u_approx|| / ||u||
        """
        a = self.project(state_flat)
        u_approx = self.reconstruct(a)
        return np.linalg.norm(state_flat - u_approx) / max(np.linalg.norm(state_flat), 1e-30)

    def plot_energy(self):
        """Plot singular value spectrum and cumulative energy."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        n = len(self.singular_values)
        axes[0].semilogy(range(1, n + 1), self.singular_values, "bo-", ms=4)
        axes[0].axvline(self.n_modes, color="r", ls="--", label=f"r = {self.n_modes}")
        axes[0].set_xlabel("Mode index")
        axes[0].set_ylabel("Singular value")
        axes[0].set_title("POD Singular Value Spectrum")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        n_all = len(self.cumulative_energy)
        axes[1].plot(range(1, n_all + 1), self.cumulative_energy * 100, "go-", ms=4)
        axes[1].axhline(self.energy_threshold * 100, color="r", ls="--",
                        label=f"threshold = {self.energy_threshold*100:.1f}%")
        axes[1].axvline(self.n_modes, color="r", ls="--")
        axes[1].set_xlabel("Number of modes")
        axes[1].set_ylabel("Cumulative energy [%]")
        axes[1].set_title("Cumulative Energy Capture")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class ReducedSolver:
    """
    Galerkin-projected reduced-order solver.

    Given the POD basis Psi_r and the training snapshot coefficients,
    the online evaluation for a new parameter point mu_new uses
    interpolation in coefficient space:

        a(mu_new) = sum_k w_k(mu_new) * a_k

    where w_k are inverse-distance weights from the training parameter
    points. This avoids re-deriving the projected ODE (which requires
    DEIM for nonlinear terms) and gives O(r) online cost.

    For this scramjet application — where the solver runs to steady state
    and we only need QoI (thrust, Isp) — coefficient interpolation is
    the pragmatic choice. Full Galerkin projection + DEIM would be needed
    for unsteady time-accurate ROM, which is a Phase 4 extension.
    """

    def __init__(self, pod_basis, train_params, train_coeffs, train_qoi):
        """
        Args:
            pod_basis:    PODBasis instance (with mean_state and basis)
            train_params: list of N parameter dicts
            train_coeffs: (N, r) array of POD coefficients at training points
            train_qoi:    list of N QoI dicts
        """
        self.pod = pod_basis
        self.train_params = train_params
        self.train_coeffs = train_coeffs
        self.train_qoi = train_qoi

        # build parameter matrix for distance computation
        # extract a consistent ordered feature vector from each param dict
        self.param_keys = sorted(train_params[0].keys())
        self.X_train = np.array([
            [p[k] for k in self.param_keys] for p in train_params
        ])

        # normalise parameter space for distance computation
        self.X_min = self.X_train.min(axis=0)
        self.X_max = self.X_train.max(axis=0)
        self.X_range = np.maximum(self.X_max - self.X_min, 1e-30)

    def evaluate(self, params):
        """
        Fast ROM evaluation at a new parameter point.

        Uses inverse-distance weighted interpolation in both the
        coefficient space (for state reconstruction) and the QoI space
        (for direct thrust/Isp prediction).

        Args:
            params: dict of design parameters (same keys as training)

        Returns:
            qoi_pred: dict with predicted thrust, Isp, etc.
            state_flat: reconstructed full-order state (n_dof,)
        """
        # normalise query point
        x_new = np.array([params[k] for k in self.param_keys])
        x_norm = (x_new - self.X_min) / self.X_range
        X_norm = (self.X_train - self.X_min) / self.X_range

        # inverse-distance weights with Shepard exponent p=2
        # w_k = 1 / d_k^2, normalised
        dists = np.linalg.norm(X_norm - x_norm, axis=1)
        dists = np.maximum(dists, 1e-12)  # avoid division by zero

        weights = 1.0 / dists**2
        weights /= weights.sum()

        # interpolate coefficients
        a_pred = weights @ self.train_coeffs  # (r,)

        # reconstruct full state
        state_flat = self.pod.reconstruct(a_pred)

        # interpolate QoI directly (more accurate than extracting from reconstruction)
        qoi_pred = {}
        for key in self.train_qoi[0].keys():
            vals = np.array([q[key] for q in self.train_qoi])
            qoi_pred[key] = float(weights @ vals)

        return qoi_pred, state_flat


class ROMEvaluator:
    """
    End-to-end ROM pipeline: collect snapshots, build basis, evaluate.

    Workflow:
        1. Create with a base SolverConfig
        2. Call build(param_list) to collect snapshots and build ROM
        3. Call evaluate(params) for fast QoI prediction
        4. Call validate(test_params) to measure accuracy
    """

    def __init__(self, base_config, energy_threshold=0.999):
        self.base_config = base_config
        self.energy_threshold = energy_threshold

        self.collector = None
        self.pod = None
        self.reduced_solver = None

        # timing
        self.build_time = 0.0
        self.mean_full_time = 0.0

    def build(self, param_list):
        """
        Offline phase: run full solver at training points, build POD basis.

        Args:
            param_list: list of parameter dicts for training

        Returns:
            n_modes: number of retained POD modes
        """
        print("=" * 60)
        print("ROM OFFLINE PHASE")
        print("=" * 60)

        t0 = time.time()

        # collect snapshots
        self.collector = SnapshotCollector(self.base_config)
        n = self.collector.sample_and_run(param_list)
        print(f"\n  Collected {n} snapshots")

        if n < 2:
            print("  ERROR: need at least 2 snapshots for POD")
            return 0

        # build POD basis
        Phi = self.collector.snapshot_matrix()
        self.pod = PODBasis(energy_threshold=self.energy_threshold)
        r = self.pod.build(Phi)

        # project training snapshots onto basis
        train_coeffs = np.array([
            self.pod.project(s) for s in self.collector.snapshots
        ])

        # build reduced solver
        self.reduced_solver = ReducedSolver(
            self.pod, self.collector.params,
            train_coeffs, self.collector.qoi,
        )

        self.build_time = time.time() - t0
        self.mean_full_time = self.collector.mean_wall_time()

        print(f"\n  Build time: {self.build_time:.2f} s")
        print(f"  Mean full-solver time: {self.mean_full_time:.2f} s")
        print(f"  POD modes: {r}")
        print("=" * 60)

        return r

    def evaluate(self, params):
        """
        Online phase: fast ROM evaluation.

        Args:
            params: dict of design parameters

        Returns:
            qoi: dict with thrust, Isp, etc.
        """
        qoi, _ = self.reduced_solver.evaluate(params)
        return qoi

    def evaluate_batch(self, param_list):
        """Evaluate ROM at multiple parameter points."""
        return [self.evaluate(p) for p in param_list]

    def validate(self, test_params):
        """
        Run full solver and ROM at test points, compare QoI.

        Args:
            test_params: list of parameter dicts

        Returns:
            errors: dict of relative errors per QoI
        """
        print("\n--- ROM VALIDATION ---")
        rom_qois = []
        full_qois = []
        rom_times = []
        full_times = []

        for idx, params in enumerate(test_params):
            # ROM evaluation
            t0 = time.time()
            qoi_rom = self.evaluate(params)
            rom_times.append(time.time() - t0)

            # full solver
            cfg = _clone_config(self.base_config)
            _apply_params(cfg, params)
            solver = Solver(cfg)
            t0 = time.time()
            solver.run()
            full_times.append(time.time() - t0)

            qoi_full = _compute_qoi(solver)

            rom_qois.append(qoi_rom)
            full_qois.append(qoi_full)

            print(f"  Test {idx+1}: thrust ROM={qoi_rom['thrust']:.2f} "
                  f"full={qoi_full['thrust']:.2f} | "
                  f"Isp ROM={qoi_rom['Isp']:.1f} full={qoi_full['Isp']:.1f}")

        # compute relative errors
        errors = {}
        for key in rom_qois[0].keys():
            rom_vals = np.array([q[key] for q in rom_qois])
            full_vals = np.array([q[key] for q in full_qois])
            denom = np.maximum(np.abs(full_vals), 1e-30)
            errors[key] = np.mean(np.abs(rom_vals - full_vals) / denom)

        speedup = np.mean(full_times) / max(np.mean(rom_times), 1e-30)

        print(f"\n  Mean relative errors:")
        for key, err in errors.items():
            print(f"    {key}: {err*100:.2f}%")
        print(f"  Speedup: {speedup:.0f}x")

        return errors


def _clone_config(cfg):
    """Deep-copy a SolverConfig by rebuilding it."""
    new = SolverConfig()
    new.inlet = InletConfig(
        mach=cfg.inlet.mach, altitude=cfg.inlet.altitude,
        gamma=cfg.inlet.gamma, R_gas=cfg.inlet.R_gas,
        Yf_inlet=cfg.inlet.Yf_inlet,
    )
    new.mesh = MeshConfig(
        nx=cfg.mesh.nx, ny=cfg.mesh.ny, y_stretch=cfg.mesh.y_stretch,
    )
    new.geometry = GeometryProfile(
        L_inlet=cfg.geometry.L_inlet,
        L_combustor=cfg.geometry.L_combustor,
        L_nozzle=cfg.geometry.L_nozzle,
        A_inlet=cfg.geometry.A_inlet,
        A_throat=cfg.geometry.A_throat,
        A_comb_exit=cfg.geometry.A_comb_exit,
        A_exit=cfg.geometry.A_exit,
    )
    cc = cfg.combustion
    new.combustion = CombustionConfig(
        enabled=cc.enabled, A_pre=cc.A_pre, Ea=cc.Ea,
        Q_heat=cc.Q_heat, nf=cc.nf, no=cc.no, W_f=cc.W_f,
    )
    new.cfl = cfg.cfl
    new.n_steps = cfg.n_steps
    new.print_interval = cfg.print_interval
    new.viscous = cfg.viscous
    new.wall_type = cfg.wall_type
    new.area_source = cfg.area_source
    return new


def _apply_params(cfg, params):
    """Override config fields from a parameter dict."""
    # geometry parameters
    geom_keys = ['L_inlet', 'L_combustor', 'L_nozzle',
                 'A_inlet', 'A_throat', 'A_comb_exit', 'A_exit']
    geom_vals = {}
    for k in geom_keys:
        geom_vals[k] = params.get(k, getattr(cfg.geometry, k))

    cfg.geometry = GeometryProfile(**geom_vals)

    # inlet parameters
    if 'mach' in params:
        cfg.inlet = InletConfig(
            mach=params['mach'],
            altitude=params.get('altitude', cfg.inlet.altitude),
            gamma=cfg.inlet.gamma, R_gas=cfg.inlet.R_gas,
            Yf_inlet=cfg.inlet.Yf_inlet,
        )
    elif 'altitude' in params:
        cfg.inlet = InletConfig(
            mach=cfg.inlet.mach,
            altitude=params['altitude'],
            gamma=cfg.inlet.gamma, R_gas=cfg.inlet.R_gas,
            Yf_inlet=cfg.inlet.Yf_inlet,
        )


def _compute_qoi(solver):
    """
    Extract performance quantities of interest from a converged solver.

    Thrust = (mdot_exit * u_exit - mdot_inlet * u_inlet) + (p_exit - p_inf) * A_exit
    Isp = thrust / (mdot_fuel * g0)

    For inviscid flow without combustion, Isp is based on the ram-drag
    thrust and a notional fuel flow rate.
    """
    rho, u, v, p, T, Yf = solver.state.primitives()
    cfg = solver.cfg
    g0 = 9.80665

    # inlet conditions (first column of cells)
    rho_in = np.mean(rho[0, :])
    u_in = np.mean(u[0, :])
    p_in = np.mean(p[0, :])
    A_in = cfg.geometry.A_inlet

    # exit conditions (last column)
    rho_ex = np.mean(rho[-1, :])
    u_ex = np.mean(u[-1, :])
    p_ex = np.mean(p[-1, :])
    A_ex = cfg.geometry.A_exit

    # mass flow rates [kg/s per unit depth]
    mdot_in = rho_in * u_in * A_in
    mdot_ex = rho_ex * u_ex * A_ex

    # momentum thrust
    F_momentum = mdot_ex * u_ex - mdot_in * u_in

    # pressure thrust
    p_inf = cfg.inlet.p_inf
    F_pressure = (p_ex - p_inf) * A_ex

    thrust = F_momentum + F_pressure

    # Isp: use inlet mass flow as reference (no separate fuel injection yet)
    # this gives the "airbreathing Isp" or specific thrust
    Isp = thrust / max(mdot_in * g0, 1e-30)

    # exit Mach
    c_ex = np.sqrt(cfg.inlet.gamma * max(p_ex, 1e-30) / max(rho_ex, 1e-30))
    M_ex = np.sqrt(u_ex**2 + np.mean(v[-1, :])**2) / max(c_ex, 1e-30)

    # pressure recovery
    p_recovery = np.mean(p[-1, :]) / max(cfg.inlet.p_inf, 1e-30)

    return {
        'thrust': thrust,
        'Isp': Isp,
        'exit_mach': M_ex,
        'pressure_recovery': p_recovery,
        'mdot': mdot_in,
    }


# need these imports for _clone_config
from solver import InletConfig, MeshConfig, CombustionConfig


if __name__ == "__main__":
    print("=== ROM standalone test ===\n")

    # small mesh for fast testing
    cfg = SolverConfig()
    cfg.mesh.nx = 30
    cfg.mesh.ny = 6
    cfg.n_steps = 100
    cfg.print_interval = 100  # suppress output

    # training points: vary A_exit
    train_params = [
        {'A_exit': 0.12},
        {'A_exit': 0.13},
        {'A_exit': 0.14},
        {'A_exit': 0.15},
        {'A_exit': 0.16},
    ]

    rom = ROMEvaluator(cfg, energy_threshold=0.999)
    r = rom.build(train_params)

    # test at intermediate point
    test_params = [{'A_exit': 0.135}, {'A_exit': 0.145}]
    errors = rom.validate(test_params)

    print("\nROM standalone test complete.")
