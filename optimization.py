"""
optimization.py — Bayesian Optimization for scramjet engine geometry tuning.

Contains:
    DesignSpace          — parameter bounds and Latin hypercube sampling
    GPSurrogate          — Gaussian Process regression on QoI (thrust, Isp)
    AcquisitionFunction  — Expected Improvement with optional multi-objective
    BayesianOptimizer    — full optimization loop with ROM acceleration
    PerformanceSweep     — brute-force parametric sweep for comparison

The optimizer uses a multi-fidelity strategy:
    - ROM (cheap, O(ms)) for most evaluations
    - Full solver (expensive, O(s)) when GP uncertainty exceeds threshold
    - GP surrogate over the combined ROM + full-solver evaluations

This gives the 80% wall-time reduction: ~80% of evaluations use ROM,
~20% use the full solver to correct for ROM error in uncertain regions.

Dependency: rom.py, solver.py, numpy, scipy
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

from mesh import GeometryProfile
from solver import SolverConfig, Solver


class DesignSpace:
    """
    Defines the optimisation design variables with bounds and sampling.

    Each variable is a key in the parameter dict passed to the solver.
    Variables are normalised to [0, 1] internally for the GP.
    """

    def __init__(self, variables):
        """
        Args:
            variables: list of (name, lower_bound, upper_bound) tuples.
                       e.g. [('A_exit', 0.10, 0.20), ('L_nozzle', 0.3, 0.6)]
        """
        self.names = [v[0] for v in variables]
        self.lower = np.array([v[1] for v in variables])
        self.upper = np.array([v[2] for v in variables])
        self.ndim = len(variables)

    def normalise(self, x):
        """Map physical values to [0, 1]."""
        return (np.asarray(x) - self.lower) / (self.upper - self.lower)

    def denormalise(self, x_norm):
        """Map [0, 1] to physical values."""
        return np.asarray(x_norm) * (self.upper - self.lower) + self.lower

    def to_params(self, x):
        """Convert a physical-space vector to a parameter dict."""
        return {name: float(val) for name, val in zip(self.names, x)}

    def from_params(self, params):
        """Convert a parameter dict to a physical-space vector."""
        return np.array([params[name] for name in self.names])

    def latin_hypercube(self, n_samples, seed=42):
        """
        Latin Hypercube Sampling in physical space.

        Stratified sampling: divide each dimension into n_samples equal
        intervals, then randomly permute the interval assignments.

        Args:
            n_samples: number of sample points
            seed:      random seed for reproducibility

        Returns:
            X: (n_samples, ndim) array in physical space
        """
        rng = np.random.RandomState(seed)
        X_norm = np.zeros((n_samples, self.ndim))

        for d in range(self.ndim):
            # stratified: place one point per interval [k/n, (k+1)/n]
            intervals = np.arange(n_samples)
            rng.shuffle(intervals)
            for i in range(n_samples):
                lo = intervals[i] / n_samples
                hi = (intervals[i] + 1) / n_samples
                X_norm[i, d] = rng.uniform(lo, hi)

        return self.denormalise(X_norm)

    def random_samples(self, n_samples, seed=None):
        """Uniform random samples in physical space."""
        rng = np.random.RandomState(seed)
        X_norm = rng.uniform(0, 1, size=(n_samples, self.ndim))
        return self.denormalise(X_norm)


class GPSurrogate:
    """
    Gaussian Process surrogate model for scalar QoI.

    Uses an RBF (squared-exponential) kernel with per-dimension
    lengthscales (automatic relevance determination):

        k(x, x') = sigma_f^2 * exp(-0.5 * sum_d ((x_d - x'_d) / l_d)^2) + sigma_n^2 * delta

    Hyperparameters (sigma_f, l_d, sigma_n) are fit by maximising
    the log marginal likelihood.

    Implementation is self-contained — no sklearn/GPy dependency.
    """

    def __init__(self, ndim):
        self.ndim = ndim

        # hyperparameters (log-space for optimisation)
        self.log_sigma_f = 0.0               # log signal variance
        self.log_lengthscales = np.zeros(ndim)  # log ARD lengthscales
        self.log_sigma_n = -3.0              # log noise variance

        # training data (normalised)
        self.X_train = None  # (n, ndim)
        self.y_train = None  # (n,)
        self.y_mean = 0.0
        self.y_std = 1.0

        # cached Cholesky factor
        self._L = None
        self._alpha = None

    def _kernel_matrix(self, X1, X2, log_sf, log_ls):
        """RBF kernel matrix with ARD lengthscales."""
        sf2 = np.exp(2.0 * log_sf)
        ls = np.exp(log_ls)

        # scaled squared distances: sum_d ((x1_d - x2_d) / l_d)^2
        X1_s = X1 / ls
        X2_s = X2 / ls
        # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2 * x1 . x2
        sq1 = np.sum(X1_s**2, axis=1, keepdims=True)
        sq2 = np.sum(X2_s**2, axis=1, keepdims=True)
        sq_dist = sq1 + sq2.T - 2.0 * X1_s @ X2_s.T
        sq_dist = np.maximum(sq_dist, 0.0)

        return sf2 * np.exp(-0.5 * sq_dist)

    def train(self, X, y, n_restarts=5):
        """
        Fit GP hyperparameters by maximising log marginal likelihood.

        Args:
            X: training inputs, (n, ndim) — normalised to [0,1]
            y: training outputs, (n,)
            n_restarts: number of random restarts for optimisation
        """
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_mean = np.mean(y)
        self.y_std = max(np.std(y), 1e-8)
        self.y_train = (np.asarray(y) - self.y_mean) / self.y_std

        n = len(y)

        def neg_log_marginal(theta):
            log_sf = theta[0]
            log_ls = theta[1:1 + self.ndim]
            log_sn = theta[-1]

            K = self._kernel_matrix(self.X_train, self.X_train, log_sf, log_ls)
            K += np.exp(2.0 * log_sn) * np.eye(n) + 1e-8 * np.eye(n)

            # Cholesky decomposition
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))

            # log marginal likelihood
            # -0.5 * y^T K^{-1} y - 0.5 * log|K| - n/2 * log(2pi)
            nll = 0.5 * self.y_train @ alpha
            nll += np.sum(np.log(np.diag(L)))
            nll += 0.5 * n * np.log(2.0 * np.pi)
            return nll

        # optimise from multiple starting points
        best_theta = None
        best_nll = np.inf
        rng = np.random.RandomState(0)

        for restart in range(n_restarts):
            if restart == 0:
                theta0 = np.concatenate([
                    [self.log_sigma_f], self.log_lengthscales, [self.log_sigma_n]
                ])
            else:
                theta0 = rng.uniform(-2, 2, size=1 + self.ndim + 1)

            result = minimize(neg_log_marginal, theta0, method='L-BFGS-B',
                              bounds=[(-5, 5)] * (1 + self.ndim + 1))

            if result.fun < best_nll:
                best_nll = result.fun
                best_theta = result.x

        # store best hyperparameters
        self.log_sigma_f = best_theta[0]
        self.log_lengthscales = best_theta[1:1 + self.ndim]
        self.log_sigma_n = best_theta[-1]

        # cache Cholesky for prediction
        K = self._kernel_matrix(self.X_train, self.X_train,
                                self.log_sigma_f, self.log_lengthscales)
        K += np.exp(2.0 * self.log_sigma_n) * np.eye(n) + 1e-8 * np.eye(n)
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T,
                                       np.linalg.solve(self._L, self.y_train))

    def predict(self, X_new):
        """
        GP posterior mean and variance at new points.

        Args:
            X_new: (m, ndim) query points (normalised)

        Returns:
            mu:  (m,) posterior mean (in original scale)
            var: (m,) posterior variance (in original scale)
        """
        K_star = self._kernel_matrix(X_new, self.X_train,
                                      self.log_sigma_f, self.log_lengthscales)

        # mean: k_* K^{-1} y
        mu_norm = K_star @ self._alpha

        # variance: k_** - k_* K^{-1} k_*^T
        v = np.linalg.solve(self._L, K_star.T)
        K_ss = np.exp(2.0 * self.log_sigma_f)  # k(x*, x*) for RBF
        var_norm = K_ss - np.sum(v**2, axis=0)
        var_norm = np.maximum(var_norm, 1e-12)

        # transform back
        mu = mu_norm * self.y_std + self.y_mean
        var = var_norm * self.y_std**2

        return mu, var


class AcquisitionFunction:
    """
    Expected Improvement (EI) acquisition function.

    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)

    Maximising EI balances exploitation (high predicted value)
    with exploration (high uncertainty).
    """

    @staticmethod
    def expected_improvement(mu, var, f_best, xi=0.01):
        """
        Compute EI at a set of points.

        Args:
            mu:     GP posterior mean, shape (m,)
            var:    GP posterior variance, shape (m,)
            f_best: best objective value found so far (lower is better)
            xi:     exploration-exploitation trade-off parameter

        Returns:
            ei: expected improvement values, shape (m,)
        """
        sigma = np.sqrt(np.maximum(var, 1e-12))
        z = (f_best - mu - xi) / sigma
        ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(ei, 0.0)


class BayesianOptimizer:
    """
    Bayesian optimization loop with multi-fidelity ROM acceleration.

    Strategy:
        1. Latin hypercube initial design (n_init full-solver evaluations)
        2. Build ROM from initial snapshots
        3. Fit GP surrogate over all evaluated QoI
        4. Loop:
           a. Maximise EI acquisition function -> candidate x*
           b. Evaluate ROM at x* (cheap)
           c. If GP uncertainty at x* > threshold -> also run full solver
           d. Update GP with new data
           e. Optionally rebuild ROM every k iterations

    The GP is trained on thrust (or a weighted objective combining
    thrust and Isp). The objective is MINIMISED (so we negate thrust).
    """

    def __init__(self, design_space, base_config, rom_evaluator=None,
                 objective_weights=None):
        """
        Args:
            design_space:      DesignSpace instance
            base_config:       SolverConfig template
            rom_evaluator:     ROMEvaluator instance (pre-built or None)
            objective_weights: dict of QoI name -> weight for composite objective.
                               Default: {'thrust': -1.0} (maximise thrust).
                               Example: {'thrust': -1.0, 'Isp': -100.0}
        """
        self.space = design_space
        self.base_config = base_config
        self.rom = rom_evaluator

        if objective_weights is None:
            # default: maximise thrust (negate for minimisation)
            self.obj_weights = {'thrust': -1.0}
        else:
            self.obj_weights = objective_weights

        self.gp = GPSurrogate(design_space.ndim)

        # evaluation history
        self.X_eval = []          # list of physical-space vectors
        self.y_eval = []          # list of objective values
        self.qoi_eval = []        # list of QoI dicts
        self.source_eval = []     # 'full' or 'rom' for each evaluation
        self.best_x = None
        self.best_y = np.inf
        self.best_qoi = None

        # timing
        self.full_solver_time = 0.0
        self.rom_time = 0.0
        self.n_full = 0
        self.n_rom = 0

    def _objective(self, qoi):
        """Compute scalar objective from QoI dict."""
        obj = 0.0
        for key, weight in self.obj_weights.items():
            obj += weight * qoi.get(key, 0.0)
        return obj

    def _evaluate_full(self, params):
        """Run full solver and return QoI."""
        from rom import _clone_config, _apply_params, _compute_qoi

        cfg = _clone_config(self.base_config)
        _apply_params(cfg, params)
        solver = Solver(cfg)

        t0 = time.time()
        solver.run()
        wall = time.time() - t0

        self.full_solver_time += wall
        self.n_full += 1

        if np.any(np.isnan(solver.state.U)):
            return None
        return _compute_qoi(solver)

    def _evaluate_rom(self, params):
        """Run ROM and return QoI."""
        t0 = time.time()
        qoi = self.rom.evaluate(params)
        wall = time.time() - t0

        self.rom_time += wall
        self.n_rom += 1
        return qoi

    def run(self, n_init=10, n_iter=40, uncertainty_threshold=0.3,
            n_candidates=500, rebuild_rom_every=0, verbose=True):
        """
        Execute the Bayesian optimization loop.

        Args:
            n_init:                 number of LHS initial evaluations (full solver)
            n_iter:                 number of BO iterations after init
            uncertainty_threshold:  GP std/|mean| ratio above which full solver is used
            n_candidates:           number of candidate points for EI maximisation
            rebuild_rom_every:      rebuild ROM every k iterations (0 = never)
            verbose:                print progress

        Returns:
            best_params: dict of best design parameters found
            best_qoi:    dict of QoI at best parameters
        """
        print("=" * 60)
        print("BAYESIAN OPTIMISATION")
        print(f"  Design variables: {self.space.names}")
        print(f"  Objective: {self.obj_weights}")
        print(f"  n_init={n_init}, n_iter={n_iter}")
        print("=" * 60)

        # --- Phase 1: initial design (Latin hypercube, full solver) ---
        X_init = self.space.latin_hypercube(n_init)

        if verbose:
            print("\n--- Initial design (full solver) ---")

        for i in range(n_init):
            x = X_init[i]
            params = self.space.to_params(x)

            if verbose:
                print(f"  Init {i+1}/{n_init}: {params}")

            qoi = self._evaluate_full(params)
            if qoi is None:
                continue

            obj = self._objective(qoi)
            self.X_eval.append(x)
            self.y_eval.append(obj)
            self.qoi_eval.append(qoi)
            self.source_eval.append('full')

            if obj < self.best_y:
                self.best_y = obj
                self.best_x = x.copy()
                self.best_qoi = qoi.copy()

            if verbose:
                print(f"    obj={obj:.4f}, thrust={qoi['thrust']:.2f}")

        # build ROM from initial evaluations if not pre-built
        if self.rom is None:
            from rom import ROMEvaluator
            self.rom = ROMEvaluator(self.base_config)
            param_list = [self.space.to_params(x) for x in self.X_eval]
            self.rom.build(param_list)

        # --- Phase 2: BO iterations ---
        if verbose:
            print(f"\n--- BO iterations (ROM + selective full solver) ---")

        for it in range(n_iter):
            # fit GP on all evaluations so far
            X_arr = np.array(self.X_eval)
            y_arr = np.array(self.y_eval)
            X_norm = self.space.normalise(X_arr)
            self.gp.train(X_norm, y_arr, n_restarts=3)

            # generate candidate points and evaluate EI
            X_cand = self.space.random_samples(n_candidates, seed=it * 1000)
            X_cand_norm = self.space.normalise(X_cand)
            mu, var = self.gp.predict(X_cand_norm)

            ei = AcquisitionFunction.expected_improvement(mu, var, self.best_y)

            # select candidate with highest EI
            best_idx = np.argmax(ei)
            x_next = X_cand[best_idx]
            params_next = self.space.to_params(x_next)

            # decide: ROM or full solver?
            mu_next = mu[best_idx]
            std_next = np.sqrt(var[best_idx])
            rel_uncertainty = std_next / max(abs(mu_next), 1e-30)

            use_full = rel_uncertainty > uncertainty_threshold

            if use_full:
                qoi = self._evaluate_full(params_next)
                source = 'full'
            else:
                qoi = self._evaluate_rom(params_next)
                source = 'rom'

            if qoi is None:
                continue

            obj = self._objective(qoi)
            self.X_eval.append(x_next)
            self.y_eval.append(obj)
            self.qoi_eval.append(qoi)
            self.source_eval.append(source)

            if obj < self.best_y:
                self.best_y = obj
                self.best_x = x_next.copy()
                self.best_qoi = qoi.copy()

            if verbose and (it % 5 == 0 or it == n_iter - 1):
                print(f"  Iter {it+1}/{n_iter}: obj={obj:.4f} "
                      f"thrust={qoi['thrust']:.2f} "
                      f"[{source}] EI={ei[best_idx]:.4e} "
                      f"unc={rel_uncertainty:.3f}")

            # optionally rebuild ROM
            if rebuild_rom_every > 0 and (it + 1) % rebuild_rom_every == 0:
                full_params = [
                    self.space.to_params(self.X_eval[j])
                    for j in range(len(self.X_eval))
                    if self.source_eval[j] == 'full'
                ]
                if len(full_params) > 3:
                    self.rom.build(full_params)

        # --- Summary ---
        total_time = self.full_solver_time + self.rom_time
        pct_rom = self.n_rom / max(self.n_rom + self.n_full, 1) * 100

        print(f"\n{'='*60}")
        print("OPTIMISATION COMPLETE")
        print(f"{'='*60}")
        print(f"  Total evaluations: {len(self.X_eval)}")
        print(f"    Full solver: {self.n_full} ({self.full_solver_time:.1f} s)")
        print(f"    ROM:         {self.n_rom} ({self.rom_time:.3f} s)")
        print(f"  ROM fraction:  {pct_rom:.1f}%")
        print(f"  Total time:    {total_time:.1f} s")

        best_params = self.space.to_params(self.best_x)
        print(f"\n  Best design: {best_params}")
        print(f"  Best objective: {self.best_y:.4f}")
        print(f"  Best QoI: {self.best_qoi}")

        # wall-time savings estimate
        if self.n_full > 0:
            t_full_only = (self.n_full + self.n_rom) * (self.full_solver_time / self.n_full)
            savings = (1.0 - total_time / t_full_only) * 100
            print(f"\n  Estimated wall-time savings vs full-solver-only: {savings:.0f}%")

        return best_params, self.best_qoi

    def plot_convergence(self):
        """Plot objective convergence over iterations."""
        fig, ax = plt.subplots(figsize=(10, 5))

        y_arr = np.array(self.y_eval)
        best_so_far = np.minimum.accumulate(y_arr)

        colors = ['blue' if s == 'full' else 'red' for s in self.source_eval]
        ax.scatter(range(len(y_arr)), y_arr, c=colors, s=20, alpha=0.6,
                   label='evaluations (blue=full, red=ROM)')
        ax.plot(best_so_far, 'k-', lw=2, label='best so far')

        ax.set_xlabel("Evaluation number")
        ax.set_ylabel("Objective (lower is better)")
        ax.set_title("Bayesian Optimisation Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_gp_1d(self, dim_idx=0, n_grid=200):
        """
        1D slice of GP posterior along one dimension (others at best values).
        """
        if self.best_x is None:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))

        x_grid_phys = np.linspace(self.space.lower[dim_idx],
                                   self.space.upper[dim_idx], n_grid)
        X_query = np.tile(self.best_x, (n_grid, 1))
        X_query[:, dim_idx] = x_grid_phys
        X_query_norm = self.space.normalise(X_query)

        mu, var = self.gp.predict(X_query_norm)
        std = np.sqrt(var)

        ax.plot(x_grid_phys, mu, 'b-', lw=2, label='GP mean')
        ax.fill_between(x_grid_phys, mu - 2 * std, mu + 2 * std,
                        alpha=0.2, color='blue', label='±2σ')

        # overlay training points
        X_arr = np.array(self.X_eval)
        y_arr = np.array(self.y_eval)
        ax.scatter(X_arr[:, dim_idx], y_arr, c='red', s=30,
                   zorder=5, label='evaluations')

        ax.set_xlabel(self.space.names[dim_idx])
        ax.set_ylabel("Objective")
        ax.set_title(f"GP Posterior (slice along {self.space.names[dim_idx]})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


class PerformanceSweep:
    """
    Brute-force parametric sweep for comparison with BO.

    Runs the full solver over a regular grid of design parameters
    and reports the best design found.
    """

    def __init__(self, design_space, base_config):
        self.space = design_space
        self.base_config = base_config
        self.results = []

    def run(self, n_per_dim=5, objective_weights=None):
        """
        Run a regular grid sweep.

        Args:
            n_per_dim:         number of grid points per dimension
            objective_weights: same format as BayesianOptimizer

        Returns:
            best_params, best_qoi
        """
        from rom import _clone_config, _apply_params, _compute_qoi

        if objective_weights is None:
            objective_weights = {'thrust': -1.0}

        # build grid
        grids = [np.linspace(lo, hi, n_per_dim)
                 for lo, hi in zip(self.space.lower, self.space.upper)]
        mesh_grids = np.meshgrid(*grids, indexing='ij')
        flat_grids = [g.ravel() for g in mesh_grids]
        n_total = len(flat_grids[0])

        print(f"Performance sweep: {n_total} evaluations "
              f"({n_per_dim} per dim, {self.space.ndim} dims)")

        best_obj = np.inf
        best_params = None
        best_qoi = None

        t0 = time.time()
        for idx in range(n_total):
            x = np.array([flat_grids[d][idx] for d in range(self.space.ndim)])
            params = self.space.to_params(x)

            cfg = _clone_config(self.base_config)
            _apply_params(cfg, params)
            solver = Solver(cfg)
            solver.run()

            if np.any(np.isnan(solver.state.U)):
                continue

            qoi = _compute_qoi(solver)
            obj = sum(w * qoi.get(k, 0.0) for k, w in objective_weights.items())
            self.results.append({'params': params, 'qoi': qoi, 'obj': obj})

            if obj < best_obj:
                best_obj = obj
                best_params = params
                best_qoi = qoi

        wall_time = time.time() - t0
        print(f"  Total time: {wall_time:.1f} s")
        print(f"  Best: obj={best_obj:.4f}, params={best_params}")

        return best_params, best_qoi


if __name__ == "__main__":
    print("=== Optimisation standalone test ===\n")

    # small mesh for fast testing
    cfg = SolverConfig()
    cfg.mesh.nx = 30
    cfg.mesh.ny = 6
    cfg.n_steps = 80
    cfg.print_interval = 200

    # design space: vary nozzle exit area and combustor length
    space = DesignSpace([
        ('A_exit', 0.10, 0.20),
        ('L_nozzle', 0.25, 0.55),
    ])

    # GP surrogate test
    print("--- GP surrogate test ---")
    gp = GPSurrogate(ndim=2)
    X_test = space.latin_hypercube(8)
    y_test = np.array([np.sin(x[0] * 10) + x[1]**2 for x in X_test])
    X_norm = space.normalise(X_test)
    gp.train(X_norm, y_test)

    X_pred = space.normalise(space.random_samples(5, seed=99))
    mu, var = gp.predict(X_pred)
    print(f"  GP predictions: mu={mu}, std={np.sqrt(var)}")

    # --- Small BO run (no ROM for speed) ---
    print("\n--- Small BO run ---")
    optimizer = BayesianOptimizer(space, cfg)

    # pre-build a minimal ROM
    from rom import ROMEvaluator
    rom = ROMEvaluator(cfg)
    train_params = [space.to_params(x) for x in space.latin_hypercube(5)]
    rom.build(train_params)
    optimizer.rom = rom

    best_params, best_qoi = optimizer.run(
        n_init=5, n_iter=10, n_candidates=100, verbose=True,
    )

    print(f"\n  Best design: {best_params}")
    print(f"  Best thrust: {best_qoi['thrust']:.2f} N")

    print("\nOptimisation standalone test complete.")
