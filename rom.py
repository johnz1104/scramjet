"""
rom.py — Proper Orthogonal Decomposition (POD) Reduced-Order Model.

Contains:
    SnapshotCollector    — runs the full solver at sampled parameter points, stores state fields
    PODBasis             — SVD decomposition, mode truncation, energy diagnostics
    ReducedSolver        — coefficient-interpolated POD state reconstruction
    ROMEvaluator         — end-to-end: basis + online solve + QoI extraction

The ROM operates in **parametric** mode: POD is taken over snapshots at
different design parameters (not over timesteps of a single run). This
maps the steady-state QoI landscape cheaply for the optimizer.

Dependency: solver.py (full-order model), numpy, scipy
"""
import copy
import time
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

from mesh import (
    GeometryProfile,
    LocalizedAreaPerturbation,
    PerturbedGeometryProfile,
    StructuredMesh2D,
)
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
            run_status = solver.run()
            wall = time.time() - t0

            # check for NaN blowup
            if np.any(np.isnan(solver.state.U)):
                print(f"    SKIPPED (NaN blowup)")
                continue
            if (run_status.get("steady_detection_enabled")
                    and not run_status.get("converged")):
                print("    SKIPPED (steady solve did not converge)")
                continue

            # store snapshot (flattened conservative state)
            self.snapshots.append(solver.state.U.ravel().copy())
            self.params.append(params)
            self.wall_times.append(wall)

            # extract QoI
            qoi = _compute_qoi(solver)
            self.qoi.append(qoi)
            n_collected += 1

            print(f"    TPR={qoi['tpr']:.5f}, "
                  f"shock_x={qoi['shock_x']:.4g} m, wall={wall:.2f} s")

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
    Coefficient-interpolated POD reduced-order solver.

    Given the POD basis Psi_r and the training snapshot coefficients,
    the online evaluation for a new parameter point mu_new uses
    interpolation in coefficient space:

        a(mu_new) = sum_k w_k(mu_new) * a_k

    where w_k are inverse-distance weights from the training parameter
    points. This avoids re-deriving the projected ODE (which requires
    DEIM for nonlinear terms) and gives O(r) online cost.

    QoIs are extracted from the reconstructed conservative state.  Direct
    inverse-distance interpolation of training QoIs is returned separately
    as ``qoi_idw`` so the paper can compare the two honestly.  This is not a
    Galerkin or time-accurate ROM; projection/DEIM remains future work.
    """

    def __init__(self, pod_basis, train_params, train_coeffs, train_qoi,
                 base_config):
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
        self.base_config = base_config

        # build parameter matrix for distance computation
        # extract a consistent ordered feature vector from each param dict
        self.param_keys = sorted(train_params[0].keys())
        self.X_train = np.array([
            [p[k] for k in self.param_keys] for p in train_params
        ])

        # normalize parameter space for distance computation
        self.X_min = self.X_train.min(axis=0)
        self.X_max = self.X_train.max(axis=0)
        self.X_range = np.maximum(self.X_max - self.X_min, 1e-30)

    def evaluate(self, params):
        """
        Fast ROM evaluation at a new parameter point.

        Uses inverse-distance interpolation in coefficient space and derives
        the primary QoIs from the resulting POD state reconstruction.

        Args:
            params: dict of design parameters (same keys as training)

        Returns:
            qoi_pred: state-derived QoI dict with an additional ``qoi_idw``
                      comparison-baseline block
            state_flat: reconstructed full-order state (n_dof,)
        """
        if set(params) != set(self.param_keys):
            missing = sorted(set(self.param_keys) - set(params))
            extra = sorted(set(params) - set(self.param_keys))
            raise ValueError(
                "ROM query keys must exactly match training keys; "
                f"missing={missing}, extra={extra}"
            )

        # normalize query point
        x_new = np.array([params[k] for k in self.param_keys])
        x_norm = (x_new - self.X_min) / self.X_range
        X_norm = (self.X_train - self.X_min) / self.X_range

        # inverse-distance weights with Shepard exponent p=2
        # w_k = 1 / d_k^2, normalized
        dists = np.linalg.norm(X_norm - x_norm, axis=1)
        dists = np.maximum(dists, 1e-12)  # avoid division by zero

        weights = 1.0 / dists**2
        weights /= weights.sum()

        # interpolate coefficients
        a_pred = weights @ self.train_coeffs  # (r,)

        # reconstruct full state
        state_flat = self.pod.reconstruct(a_pred)

        cfg = _clone_config(self.base_config)
        _apply_params(cfg, params)
        mesh = _mesh_from_config(cfg)
        U_reconstructed = state_flat.reshape(5, mesh.nx, mesh.ny)
        qoi_pred = compute_qoi_from_state(U_reconstructed, mesh, cfg)

        qoi_idw = {}
        for key in self.train_qoi[0].keys():
            values = [q.get(key) for q in self.train_qoi]
            if not all(isinstance(v, (int, float, np.number))
                       and not isinstance(v, (bool, np.bool_))
                       for v in values):
                continue
            vals = np.asarray(values, dtype=float)
            finite = np.isfinite(vals)
            if not np.any(finite):
                qoi_idw[key] = float("nan")
                continue
            finite_weights = weights[finite]
            finite_weights /= np.sum(finite_weights)
            qoi_idw[key] = float(finite_weights @ vals[finite])
        qoi_pred["qoi_idw"] = qoi_idw

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
            self.base_config,
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
            qoi: state-derived QoIs plus the ``qoi_idw`` baseline
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
        errors: dict with ``pod_state`` and ``idw`` relative-error blocks
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

            print(f"  Test {idx+1}: TPR POD={qoi_rom['tpr']:.5f} "
                  f"IDW={qoi_rom['qoi_idw'].get('tpr', float('nan')):.5f} "
                  f"full={qoi_full['tpr']:.5f} | "
                  f"shock_x POD={qoi_rom['shock_x']:.4g} "
                  f"full={qoi_full['shock_x']:.4g}")

        # compute relative errors
        errors = {"pod_state": {}, "idw": {}}
        numeric_keys = [
            key for key, value in full_qois[0].items()
            if isinstance(value, (int, float, np.number))
            and not isinstance(value, (bool, np.bool_))
        ]
        for key in numeric_keys:
            full_vals = np.asarray([q[key] for q in full_qois], dtype=float)
            finite_full = np.isfinite(full_vals)
            for label, getter in (
                ("pod_state", lambda q, k=key: q.get(k)),
                ("idw", lambda q, k=key: q.get("qoi_idw", {}).get(k)),
            ):
                vals = np.asarray([getter(q) for q in rom_qois], dtype=float)
                mask = finite_full & np.isfinite(vals)
                if not np.any(mask):
                    errors[label][key] = None
                    continue
                denom = np.maximum(np.abs(full_vals[mask]), 1.0e-30)
                errors[label][key] = float(np.mean(
                    np.abs(vals[mask] - full_vals[mask]) / denom,
                ))

        speedup = np.mean(full_times) / max(np.mean(rom_times), 1e-30)

        print(f"\n  Mean relative errors:")
        for label, block in errors.items():
            print(f"    {label}:")
            for key, err in block.items():
                if err is not None:
                    print(f"      {key}: {err*100:.2f}%")
        print(f"  Speedup: {speedup:.0f}x")

        return errors


def _clone_config(cfg):
    """Deep-copy a SolverConfig while preserving every current scalar flag."""
    new = SolverConfig()
    explicit = getattr(cfg.inlet, "explicit_conditions", False)
    new.inlet = InletConfig(
        mach=cfg.inlet.mach, altitude=cfg.inlet.altitude,
        gamma=cfg.inlet.gamma, R_gas=cfg.inlet.R_gas,
        Yf_inlet=cfg.inlet.Yf_inlet,
        T_inf=cfg.inlet.T_inf if explicit else None,
        p_inf=cfg.inlet.p_inf if explicit else None,
    )
    new.mesh = MeshConfig(
        nx=cfg.mesh.nx, ny=cfg.mesh.ny, y_stretch=cfg.mesh.y_stretch,
    )
    if hasattr(cfg.geometry, "copy"):
        new.geometry = cfg.geometry.copy()
    else:
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
    # Constructor-rebuild the structured members above, then copy every
    # top-level field (including all outlet controls and future additions).
    for name, value in vars(cfg).items():
        if name in {"inlet", "mesh", "geometry", "combustion"}:
            continue
        setattr(new, name, copy.deepcopy(value))
    return new


def _apply_params(cfg, params):
    """Apply a validated parameter dict without changing geometry identity."""
    params = dict(params)
    geom_keys = {
        'L_inlet', 'L_combustor', 'L_nozzle',
        'A_inlet', 'A_throat', 'A_comb_exit', 'A_exit',
    }
    perturbation_keys = {
        'q_throat', 'area_amplitude', 'area_enabled', 'area_mode',
        'area_x_center', 'area_width', 'min_area',
    }
    inlet_keys = {'mach', 'altitude'}
    unknown = set(params) - geom_keys - perturbation_keys - inlet_keys
    if unknown:
        raise ValueError(f"Unknown parameter keys: {sorted(unknown)}")

    geom_existing = cfg.geometry
    perturbation_existing = getattr(geom_existing, "perturbation", None)
    min_area_existing = getattr(geom_existing, "min_area", 1.0e-6)
    base_existing = getattr(geom_existing, "base_geometry", geom_existing)

    requested_geom = set(params) & geom_keys
    requested_perturbation = set(params) & perturbation_keys
    if requested_geom:
        if type(base_existing) is not GeometryProfile:
            raise ValueError(
                "Three-section geometry parameters cannot be applied to "
                f"{type(base_existing).__name__}; preserve/calibrate that "
                "geometry or use perturbation parameters instead."
            )
        geom_vals = {
            key: params.get(key, getattr(base_existing, key))
            for key in geom_keys
        }
        base_geometry = GeometryProfile(**geom_vals)
    else:
        base_geometry = base_existing.copy()

    if requested_geom or requested_perturbation:
        if (getattr(geom_existing, "is_time_dependent", False)
                and requested_perturbation):
            raise ValueError(
                "Applying steady perturbation parameters to a time-dependent "
                "geometry is ambiguous"
            )
        use_perturbation = (
            perturbation_existing is not None or bool(requested_perturbation)
        )
    else:
        use_perturbation = False

    if use_perturbation:
        if perturbation_existing is not None:
            perturbation = perturbation_existing.copy()
        else:
            perturbation = LocalizedAreaPerturbation(
                enabled=True,
                mode='throat_gaussian',
                amplitude=0.0,
                x_center=base_geometry.x_throat,
                width=max(0.05 * base_geometry.L_total, 1e-6),
            )

        if 'area_enabled' in params:
            perturbation.enabled = bool(params['area_enabled'])
        if 'area_mode' in params:
            if params['area_mode'] != 'throat_gaussian':
                raise ValueError(f"Unsupported area perturbation mode: {params['area_mode']}")
            perturbation.mode = params['area_mode']
        if 'q_throat' in params:
            perturbation.amplitude = float(params['q_throat'])
        if 'area_amplitude' in params:
            perturbation.amplitude = float(params['area_amplitude'])
        if 'area_x_center' in params:
            perturbation.x_center = float(params['area_x_center'])
        if 'area_width' in params:
            perturbation.width = float(params['area_width'])
            if perturbation.width <= 0.0:
                raise ValueError("area_width must be positive")

        min_area = float(params.get('min_area', min_area_existing))
        perturbation.min_area = min_area
        cfg.geometry = PerturbedGeometryProfile(base_geometry, perturbation)
    elif requested_geom:
        cfg.geometry = base_geometry

    # inlet parameters. Explicit tunnel conditions (presets) survive a Mach
    # sweep; asking for an 'altitude' switches back to the atmosphere model.
    explicit = getattr(cfg.inlet, "explicit_conditions", False) and 'altitude' not in params
    if 'mach' in params:
        cfg.inlet = InletConfig(
            mach=params['mach'],
            altitude=params.get('altitude', cfg.inlet.altitude),
            gamma=cfg.inlet.gamma, R_gas=cfg.inlet.R_gas,
            Yf_inlet=cfg.inlet.Yf_inlet,
            T_inf=cfg.inlet.T_inf if explicit else None,
            p_inf=cfg.inlet.p_inf if explicit else None,
        )
    elif 'altitude' in params:
        cfg.inlet = InletConfig(
            mach=cfg.inlet.mach,
            altitude=params['altitude'],
            gamma=cfg.inlet.gamma, R_gas=cfg.inlet.R_gas,
            Yf_inlet=cfg.inlet.Yf_inlet,
        )


def _mesh_from_config(cfg):
    """Build the structured mesh associated with a query configuration."""
    geom = cfg.geometry
    if cfg.mesh.y_stretch > 1.001:
        return StructuredMesh2D.stretched(
            0.0, geom.L_total, 0.0, geom.A_exit,
            cfg.mesh.nx, cfg.mesh.ny, y_ratio=cfg.mesh.y_stretch,
        )
    return StructuredMesh2D.uniform(
        0.0, geom.L_total, 0.0, geom.A_exit,
        cfg.mesh.nx, cfg.mesh.ny,
    )


def compute_qoi_from_state(U, mesh, cfg):
    """
    Extract shared QoIs from a conservative state tensor.

    Research QoIs (experiment-matched, see research plan §3.1):
        tpr         — mass-flux-weighted total-pressure recovery p0_exit/p0_inf
        shock_x     — dominant shock location on the centerline [m] (NaN if none)

    ``mdot_prescribed`` is deliberately named: the first column is fixed by
    the supersonic Dirichlet inlet and therefore cannot represent spillage or
    mass-capture collapse.  ``mdot_exit`` and ``mass_defect`` diagnose
    numerical conservation.  Unstart is observable only through shock
    expulsion/TPR collapse in this boundary model.
    """
    from diagnostics import (
        physical_exit_index,
        primitives_from_state,
        shock_diagnostics_from_state,
        total_pressure_recovery_from_state,
        transverse_average,
    )

    gamma = cfg.inlet.gamma
    R_gas = cfg.inlet.R_gas
    U = np.asarray(U, dtype=float)
    rho_raw = U[0]
    rho_safe = np.maximum(rho_raw, 1.0e-30)
    u_raw = U[1] / rho_safe
    v_raw = U[2] / rho_safe
    p_raw = (gamma - 1.0) * (
        U[3] - 0.5 * (U[1]**2 + U[2]**2) / rho_safe
    )
    state_admissible = bool(
        np.all(np.isfinite(U)) and np.all(rho_raw > 0.0) and np.all(p_raw > 0.0)
    )
    rho, u, v, p, T, Yf, M = primitives_from_state(
        U, gamma=gamma, R_gas=R_gas,
    )
    g0 = 9.80665

    i_in = 0
    i_exit = physical_exit_index(mesh)
    rho_in = transverse_average(rho[i_in, :], mesh)
    u_in = transverse_average(u[i_in, :], mesh)
    p_in = transverse_average(p[i_in, :], mesh)
    rho_ex = transverse_average(rho[i_exit, :], mesh)
    u_ex = transverse_average(u[i_exit, :], mesh)
    v_ex = transverse_average(v[i_exit, :], mesh)
    p_ex = transverse_average(p[i_exit, :], mesh)
    M_ex = transverse_average(M[i_exit, :], mesh)
    A_in = float(cfg.geometry.area(np.array([mesh.xc[i_in]]))[0])
    A_ex = float(cfg.geometry.area(np.array([mesh.xc[i_exit]]))[0])

    # mass flow rates [kg/s per unit depth]
    mdot_in = transverse_average(rho[i_in, :] * u[i_in, :], mesh) * A_in
    mdot_ex = transverse_average(rho[i_exit, :] * u[i_exit, :], mesh) * A_ex
    mass_defect = (mdot_ex - mdot_in) / max(abs(mdot_in), 1.0e-30)

    # momentum thrust
    F_momentum = mdot_ex * u_ex - mdot_in * u_in

    # pressure thrust
    p_inf = cfg.inlet.p_inf
    F_pressure = (p_ex - p_inf) * A_ex

    thrust = F_momentum + F_pressure

    # Isp: use inlet mass flow as reference (no separate fuel injection yet)
    # this gives the "airbreathing Isp" or specific thrust
    Isp = thrust / max(mdot_in * g0, 1e-30)

    # legacy static pressure ratio (NOT total-pressure recovery)
    p_recovery = p_ex / max(cfg.inlet.p_inf, 1e-30)

    # experiment-matched QoIs
    tpr = total_pressure_recovery_from_state(U, mesh, cfg)
    shock = shock_diagnostics_from_state(U, mesh, cfg)
    j_mid = mesh.ny // 2
    i_throat = int(np.argmin(np.abs(mesh.xc - cfg.geometry.x_throat)))
    contraction_hi = max(min(i_throat + 1, mesh.nx), 1)
    min_mach_contraction = float(np.min(M[:contraction_hi, j_mid]))
    inlet_shock_limit = max(2, int(np.ceil(0.1 * mesh.nx)))
    shock_at_inlet = bool(
        shock["shock_detected"] and shock["shock_index"] <= inlet_shock_limit
    )
    started = not shock_at_inlet

    return {
        'tpr': float(tpr),
        'shock_x': float(shock['shock_x']),
        'shock_detected': bool(shock['shock_detected']),
        'shock_at_inlet': shock_at_inlet,
        'started': bool(started),
        'unstart_classification': 'started' if started else 'unstarted',
        'min_centerline_mach_contraction': min_mach_contraction,
        'exit_mach': float(M_ex),
        'mdot_prescribed': float(mdot_in),
        'mdot_exit': float(mdot_ex),
        'mass_defect': float(mass_defect),
        'pressure_recovery': float(p_recovery),
        'thrust': float(thrust),
        'Isp': float(Isp),
        'state_admissible': state_admissible,
    }


def _compute_qoi(solver):
    """Solver wrapper around :func:`compute_qoi_from_state`."""
    return compute_qoi_from_state(solver.state.U, solver.mesh, solver.cfg)


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
