"""
mesh.py — Spatial domain for the 2D scramjet CFD solver.

Contains:
    StructuredMesh2D   — 2D structured quadrilateral mesh with face geometry
    GeometryProfile    — Parameterised scramjet duct area profile A(x)

Dependency: numpy, matplotlib (plotting only)
"""
import numpy as np
import matplotlib.pyplot as plt


# StructuredMesh2D

class StructuredMesh2D:
    """
    2-D structured quadrilateral mesh.

    Cells are indexed (i, j): i in [0, nx-1] along x, j in [0, ny-1] along y.
    All cell arrays are (nx, ny) shaped.

    Face conventions (outward normal from the *left* or *lower* cell):
        I-faces: separate cell (i-1,j) from (i,j), normal ~ +x.  Shape (nx+1, ny).
        J-faces: separate cell (i,j-1) from (i,j), normal ~ +y.  Shape (nx, ny+1).

    Units: metres. Cell volumes = areas per unit depth [m^2].
    """

    def __init__(self, x_nodes, y_nodes):
        """
        Args:
            x_nodes: 1-D node x-coordinates, shape (nx+1,)  [m]
            y_nodes: 1-D node y-coordinates, shape (ny+1,)  [m]
        """
        self.x_nodes = np.asarray(x_nodes, dtype=np.float64)
        self.y_nodes = np.asarray(y_nodes, dtype=np.float64)

        self.nx = len(self.x_nodes) - 1
        self.ny = len(self.y_nodes) - 1
        self.n_cells = self.nx * self.ny

        self._compute_geometry()

    # Factory constructors

    @staticmethod
    def uniform(x_min, x_max, y_min, y_max, nx, ny):
        """Uniform Cartesian mesh."""
        return StructuredMesh2D(
            np.linspace(x_min, x_max, nx + 1),
            np.linspace(y_min, y_max, ny + 1),
        )

    @staticmethod
    def stretched(x_min, x_max, y_min, y_max, nx, ny, y_ratio=1.08):
        """
        Cartesian mesh with geometric wall-normal stretching in y.
        y_ratio > 1 gives finer spacing near y_min (the wall).
        Typical values: 1.05-1.15 for boundary layer resolution.
        """
        x_nodes = np.linspace(x_min, x_max, nx + 1)

        # geometric series: dy_j = dy_0 * y_ratio^j
        if abs(y_ratio - 1.0) < 1e-12:
            y_nodes = np.linspace(y_min, y_max, ny + 1)
        else:
            # sum of geometric series: L = dy_0 * (r^n - 1) / (r - 1)
            L = y_max - y_min
            dy_0 = L * (y_ratio - 1.0) / (y_ratio**ny - 1.0)
            y_nodes = np.zeros(ny + 1)
            y_nodes[0] = y_min
            for j in range(ny):
                y_nodes[j + 1] = y_nodes[j] + dy_0 * y_ratio**j

        return StructuredMesh2D(x_nodes, y_nodes)
    
    # Geometry computation

    def _compute_geometry(self):
        """Compute cell centroids, volumes, face areas, and face normals."""
        nx, ny = self.nx, self.ny
        xn, yn = self.x_nodes, self.y_nodes

        # cell centroids: average of bounding node coordinates
        # xc[i] = 0.5 * (x_nodes[i] + x_nodes[i+1])
        self.xc = 0.5 * (xn[:-1] + xn[1:])                    # (nx,)
        self.yc = 0.5 * (yn[:-1] + yn[1:])                    # (ny,)

        # cell sizes
        self.dx = np.diff(xn)                                   # (nx,)
        self.dy = np.diff(yn)                                   # (ny,)

        # cell volumes (= area per unit depth for 2D)
        # vol[i,j] = dx[i] * dy[j]
        self.vol = np.outer(self.dx, self.dy)                   # (nx, ny)

        # I-face areas (length of face in y-direction)
        # i_face_area[i, j] = dy[j]   for face between cell (i-1,j) and (i,j)
        self.i_face_area = np.tile(self.dy, (nx + 1, 1))       # (nx+1, ny)

        # J-face areas (length of face in x-direction)
        # j_face_area[i, j] = dx[i]   for face between cell (i,j-1) and (i,j)
        self.j_face_area = np.tile(self.dx, (ny + 1, 1)).T     # (nx, ny+1)

        # I-face normals point in +x: n = [1, 0]
        # J-face normals point in +y: n = [0, 1]
        # For a Cartesian mesh these are trivially axis-aligned.

    def plot(self, title="Mesh"):
        """Visualise the mesh grid lines."""
        fig, ax = plt.subplots(figsize=(12, 4))
        xn, yn = self.x_nodes, self.y_nodes

        # vertical lines (constant x)
        for i in range(len(xn)):
            ax.plot([xn[i], xn[i]], [yn[0], yn[-1]], "k-", lw=0.3)
        # horizontal lines (constant y)
        for j in range(len(yn)):
            ax.plot([xn[0], xn[-1]], [yn[j], yn[j]], "k-", lw=0.3)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.tight_layout()
        return fig


# GeometryProfile

class GeometryProfile:
    """
    Parameterised scramjet duct cross-sectional area profile A(x).

    Three sections:
        1. Inlet (converging):   quadratic ramp from A_inlet to A_throat
        2. Combustor (slight divergence): linear from A_throat to A_comb_exit
        3. Nozzle (strong divergence): power-law from A_comb_exit to A_exit

    The area profile is normalized by A_throat so that A(x_throat) = 1.
    """

    def __init__(self, L_inlet, L_combustor, L_nozzle,
                 A_inlet, A_throat, A_comb_exit, A_exit):
        """
        Args:
            L_inlet:     length of inlet section [m]
            L_combustor: length of combustor section [m]
            L_nozzle:    length of nozzle section [m]
            A_inlet:     inlet area [m^2]
            A_throat:    throat area (minimum, isolator entrance) [m^2]
            A_comb_exit: combustor exit area [m^2]
            A_exit:      nozzle exit area [m^2]
        """
        self.L_inlet = L_inlet
        self.L_combustor = L_combustor
        self.L_nozzle = L_nozzle
        self.L_total = L_inlet + L_combustor + L_nozzle

        self.A_inlet = A_inlet
        self.A_throat = A_throat
        self.A_comb_exit = A_comb_exit
        self.A_exit = A_exit

        # section boundaries
        self.x_throat = L_inlet
        self.x_comb_exit = L_inlet + L_combustor
        self.x_exit = self.L_total

    @staticmethod
    def default():
        """
        Default scramjet duct geometry.
        Representative of a Mach 6-8 hydrogen-fueled scramjet isolator-combustor-nozzle.
        """
        return GeometryProfile(
            L_inlet=0.3, L_combustor=0.5, L_nozzle=0.4,
            A_inlet=0.10, A_throat=0.05, A_comb_exit=0.07, A_exit=0.15,
        )

    def copy(self):
        """Return an independent copy of this geometry profile."""
        return GeometryProfile(
            L_inlet=self.L_inlet,
            L_combustor=self.L_combustor,
            L_nozzle=self.L_nozzle,
            A_inlet=self.A_inlet,
            A_throat=self.A_throat,
            A_comb_exit=self.A_comb_exit,
            A_exit=self.A_exit,
        )

    def area(self, x):
        """
        Cross-sectional area at position x.

        Args:
            x: scalar or ndarray of x-positions [m]

        Returns:
            A(x): same shape as x [m^2]
        """
        x = np.asarray(x, dtype=np.float64)
        A = np.empty_like(x)

        # section 1: inlet (quadratic converging)
        # A(x) = A_inlet + (A_throat - A_inlet) * (x / L_inlet)^2
        mask_inlet = x < self.x_throat
        xi = x[mask_inlet] / self.L_inlet
        A[mask_inlet] = self.A_inlet + (self.A_throat - self.A_inlet) * xi**2

        # section 2: combustor (linear divergence)
        # A(x) = A_throat + (A_comb_exit - A_throat) * (x - x_throat) / L_combustor
        mask_comb = (x >= self.x_throat) & (x < self.x_comb_exit)
        xc = (x[mask_comb] - self.x_throat) / self.L_combustor
        A[mask_comb] = self.A_throat + (self.A_comb_exit - self.A_throat) * xc

        # section 3: nozzle (power-law divergence, exponent 1.5)
        # A(x) = A_comb_exit + (A_exit - A_comb_exit) * ((x - x_comb_exit) / L_nozzle)^1.5
        mask_nozzle = x >= self.x_comb_exit
        xn = (x[mask_nozzle] - self.x_comb_exit) / self.L_nozzle
        xn = np.clip(xn, 0.0, 1.0)
        A[mask_nozzle] = self.A_comb_exit + (self.A_exit - self.A_comb_exit) * xn**1.5

        return A

    def area_gradient(self, x):
        """
        dA/dx at position x, computed analytically from the piecewise profile.
        """
        x = np.asarray(x, dtype=np.float64)
        dAdx = np.empty_like(x)

        # inlet: dA/dx = 2 * (A_throat - A_inlet) * x / L_inlet^2
        mask_inlet = x < self.x_throat
        dAdx[mask_inlet] = (2.0 * (self.A_throat - self.A_inlet)
                            * x[mask_inlet] / self.L_inlet**2)

        # combustor: dA/dx = (A_comb_exit - A_throat) / L_combustor
        mask_comb = (x >= self.x_throat) & (x < self.x_comb_exit)
        dAdx[mask_comb] = (self.A_comb_exit - self.A_throat) / self.L_combustor

        # nozzle: dA/dx = 1.5 * (A_exit - A_comb_exit) / L_nozzle * ((x - x_comb_exit) / L_nozzle)^0.5
        mask_nozzle = x >= self.x_comb_exit
        xn = (x[mask_nozzle] - self.x_comb_exit) / self.L_nozzle
        xn = np.clip(xn, 1e-12, 1.0)  # avoid sqrt(0)
        dAdx[mask_nozzle] = (1.5 * (self.A_exit - self.A_comb_exit)
                             / self.L_nozzle * xn**0.5)

        return dAdx

    def plot(self, n_pts=500):
        """Plot the area profile A(x)."""
        x = np.linspace(0, self.L_total, n_pts)
        A = self.area(x)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(x, A, "b-", lw=2)
        axes[0].set_ylabel("A(x) [m²]")
        axes[0].set_title("Scramjet Duct Area Profile")
        axes[0].axvline(self.x_throat, color="r", ls="--", lw=0.8, label="throat")
        axes[0].axvline(self.x_comb_exit, color="g", ls="--", lw=0.8, label="comb exit")
        axes[0].legend()

        dAdx = self.area_gradient(x)
        axes[1].plot(x, dAdx, "r-", lw=2)
        axes[1].set_ylabel("dA/dx [m]")
        axes[1].set_xlabel("x [m]")

        plt.tight_layout()
        return fig


class LocalizedAreaPerturbation:
    """
    Localized effective-area perturbation for wall-position prototypes.

    The perturbation is intentionally static and one-dimensional:

        A(x; q) = A_base(x) + q * phi(x)

    where phi(x) is a Gaussian mode. This models an effective throat-area
    displacement, not body-fitted wall motion or a deforming mesh.
    """

    def __init__(self, enabled=True, mode="throat_gaussian", amplitude=0.0,
                 x_center=None, width=0.05, min_area=1.0e-6):
        if mode != "throat_gaussian":
            raise ValueError(f"Unsupported area perturbation mode: {mode}")
        if width <= 0.0:
            raise ValueError("Area perturbation width must be positive")
        if min_area <= 0.0:
            raise ValueError("min_area must be positive")

        self.enabled = bool(enabled)
        self.mode = mode
        self.amplitude = float(amplitude)
        self.x_center = x_center
        self.width = float(width)
        self.min_area = float(min_area)

    @property
    def active(self):
        """Whether the perturbation changes the base geometry."""
        return self.enabled and abs(self.amplitude) > 0.0

    def copy(self):
        """Return an independent copy of this perturbation."""
        return LocalizedAreaPerturbation(
            enabled=self.enabled,
            mode=self.mode,
            amplitude=self.amplitude,
            x_center=self.x_center,
            width=self.width,
            min_area=self.min_area,
        )

    def center(self, base_geometry):
        """Resolve the perturbation center, defaulting to the throat."""
        if self.x_center is None:
            return base_geometry.x_throat
        return float(self.x_center)

    def shape(self, x, base_geometry):
        """Gaussian localization function phi(x)."""
        x = np.asarray(x, dtype=np.float64)
        xi = (x - self.center(base_geometry)) / self.width
        return np.exp(-0.5 * xi**2)

    def shape_gradient(self, x, base_geometry):
        """Analytical derivative dphi/dx for the Gaussian mode."""
        x = np.asarray(x, dtype=np.float64)
        phi = self.shape(x, base_geometry)
        return -((x - self.center(base_geometry)) / self.width**2) * phi

    def to_dict(self):
        """JSON-serialisable representation."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "amplitude": self.amplitude,
            "x_center": self.x_center,
            "width": self.width,
            "min_area": self.min_area,
            "active": self.active,
        }


class PerturbedGeometryProfile:
    """
    Geometry wrapper adding one localized effective-area perturbation.

    The wrapper mirrors the `GeometryProfile` interface used by the solver:
    `area(x)` and `area_gradient(x)`. When disabled or when q = 0, it returns
    the base geometry exactly.
    """

    def __init__(self, base_geometry, perturbation):
        self.base_geometry = base_geometry.copy()
        self.perturbation = perturbation.copy()

        # Mirror base fields used elsewhere in the project.
        self.L_inlet = self.base_geometry.L_inlet
        self.L_combustor = self.base_geometry.L_combustor
        self.L_nozzle = self.base_geometry.L_nozzle
        self.L_total = self.base_geometry.L_total
        self.A_inlet = self.base_geometry.A_inlet
        self.A_throat = self.base_geometry.A_throat
        self.A_comb_exit = self.base_geometry.A_comb_exit
        self.A_exit = self.base_geometry.A_exit
        self.x_throat = self.base_geometry.x_throat
        self.x_comb_exit = self.base_geometry.x_comb_exit
        self.x_exit = self.base_geometry.x_exit

        self._validate_profile()

    def copy(self):
        """Return an independent copy of this perturbed geometry."""
        return PerturbedGeometryProfile(self.base_geometry, self.perturbation)

    @property
    def min_area(self):
        """Minimum allowable effective area."""
        return self.perturbation.min_area

    def _perturbed_area_unchecked(self, x):
        """Return the perturbed area without validation."""
        A_base = self.base_geometry.area(x)
        if not self.perturbation.active:
            return A_base
        return (A_base
                + self.perturbation.amplitude
                * self.perturbation.shape(x, self.base_geometry))

    def _validate_area_values(self, A):
        """Raise if any sampled area violates the positivity floor."""
        min_value = float(np.min(A))
        if min_value <= self.min_area:
            raise ValueError(
                f"Perturbed area violates min_area: min(A)={min_value:.6e}, "
                f"min_area={self.min_area:.6e}"
            )

    def _validate_profile(self, n_pts=2000):
        """Best-effort whole-profile positivity check."""
        if not self.perturbation.active:
            return
        x = np.linspace(0.0, self.L_total, n_pts)
        self._validate_area_values(self._perturbed_area_unchecked(x))

    def area(self, x):
        """Return A(x; q), preserving the base geometry exactly when inactive."""
        A = self._perturbed_area_unchecked(x)
        if self.perturbation.active:
            self._validate_area_values(A)
        return A

    def area_gradient(self, x):
        """Return dA/dx, including q * dphi/dx for active perturbations."""
        dAdx = self.base_geometry.area_gradient(x)
        if not self.perturbation.active:
            return dAdx
        return (dAdx
                + self.perturbation.amplitude
                * self.perturbation.shape_gradient(x, self.base_geometry))

    def min_area_value(self, n_pts=1000):
        """Sample the current minimum area for validation and reporting."""
        x = np.linspace(0.0, self.L_total, n_pts)
        return float(np.min(self.area(x)))

    def throat_area(self):
        """Return the effective area at the nominal throat location."""
        return float(self.area(np.array([self.x_throat]))[0])

    def to_dict(self):
        """JSON-serialisable representation."""
        return {
            "type": "PerturbedGeometryProfile",
            "base": geometry_to_dict(self.base_geometry),
            "perturbation": self.perturbation.to_dict(),
            "min_area": self.min_area,
            "throat_area": self.throat_area(),
            "min_sampled_area": self.min_area_value(),
        }

    def plot(self, n_pts=500):
        """Plot the active area profile and gradient."""
        x = np.linspace(0.0, self.L_total, n_pts)
        A = self.area(x)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(x, self.base_geometry.area(x), "k--", lw=1.2, label="base")
        axes[0].plot(x, A, "b-", lw=2, label="perturbed")
        axes[0].set_ylabel("A(x) [m²]")
        axes[0].set_title("Perturbed Effective Area Profile")
        axes[0].axvline(self.x_throat, color="r", ls="--", lw=0.8, label="throat")
        axes[0].legend()

        dAdx = self.area_gradient(x)
        axes[1].plot(x, self.base_geometry.area_gradient(x), "k--", lw=1.2, label="base")
        axes[1].plot(x, dAdx, "r-", lw=2, label="perturbed")
        axes[1].set_ylabel("dA/dx [m]")
        axes[1].set_xlabel("x [m]")
        axes[1].legend()

        plt.tight_layout()
        return fig


class SinusoidalAreaForcing:
    """
    Time-dependent scalar throat-area forcing.

        q(t) = mean + amplitude * sin(2*pi*f*t + phase)

    `mean` is an optional static offset (`q_offset`) used by the
    parametric DOE workflow. With `mean = 0.0` this reduces to the
    pure-sinusoidal form `q(t) = amplitude * sin(...)` used by the
    earlier unsteady prototype.

    `frequency_hz = 0` is allowed and yields the static value
    `mean + amplitude * sin(phase)`.
    """

    def __init__(self, amplitude=0.0, frequency_hz=0.0, phase=0.0,
                 enabled=True, mean=0.0):
        if frequency_hz < 0.0:
            raise ValueError("frequency_hz must be non-negative")
        self.amplitude = float(amplitude)
        self.frequency_hz = float(frequency_hz)
        self.phase = float(phase)
        self.enabled = bool(enabled)
        self.mean = float(mean)

    def copy(self):
        """Return an independent copy of this forcing."""
        return SinusoidalAreaForcing(
            amplitude=self.amplitude,
            frequency_hz=self.frequency_hz,
            phase=self.phase,
            enabled=self.enabled,
            mean=self.mean,
        )

    def value(self, time):
        """Return q(t) = mean + amplitude * sin(2*pi*f*t + phase)."""
        if not self.enabled:
            return 0.0
        omega = 2.0 * np.pi * self.frequency_hz
        return self.mean + self.amplitude * np.sin(omega * float(time) + self.phase)

    def rate(self, time):
        """Return dq/dt = amplitude * omega * cos(2*pi*f*t + phase)."""
        if not self.enabled or self.frequency_hz == 0.0:
            return 0.0
        omega = 2.0 * np.pi * self.frequency_hz
        return self.amplitude * omega * np.cos(omega * float(time) + self.phase)

    def to_dict(self):
        """JSON-serialisable representation."""
        return {
            "type": "SinusoidalAreaForcing",
            "enabled": self.enabled,
            "amplitude": self.amplitude,
            "frequency_hz": self.frequency_hz,
            "phase": self.phase,
            "mean": self.mean,
        }


class TimeDependentPerturbedGeometryProfile:
    """
    Time-dependent effective-area wrapper for reduced-fidelity forcing.

    This reuses `LocalizedAreaPerturbation` for the Gaussian shape and
    `SinusoidalAreaForcing` for q(t):

        A(x, t) = A_base(x) + q(t) * phi(x)

    It is an effective area-source model, not moving-wall CFD.
    """

    def __init__(self, base_geometry, perturbation, forcing):
        self.base_geometry = base_geometry.copy()
        self.perturbation = perturbation.copy()
        self.forcing = forcing.copy()
        self.time = 0.0

        # Mirror base fields used elsewhere in the project.
        self.L_inlet = self.base_geometry.L_inlet
        self.L_combustor = self.base_geometry.L_combustor
        self.L_nozzle = self.base_geometry.L_nozzle
        self.L_total = self.base_geometry.L_total
        self.A_inlet = self.base_geometry.A_inlet
        self.A_throat = self.base_geometry.A_throat
        self.A_comb_exit = self.base_geometry.A_comb_exit
        self.A_exit = self.base_geometry.A_exit
        self.x_throat = self.base_geometry.x_throat
        self.x_comb_exit = self.base_geometry.x_comb_exit
        self.x_exit = self.base_geometry.x_exit

        self._validate_profile_over_cycle()

    @property
    def is_time_dependent(self):
        """Marker used by the area source to refresh each time step."""
        return True

    @property
    def min_area(self):
        """Minimum allowable effective area."""
        return self.perturbation.min_area

    def copy(self):
        """Return an independent copy of this geometry."""
        new = TimeDependentPerturbedGeometryProfile(
            self.base_geometry,
            self.perturbation,
            self.forcing,
        )
        new.time = self.time
        return new

    def set_time(self, time):
        """Set the active physical time used by area(x)."""
        self.time = float(time)

    def current_amplitude(self, time=None):
        """Return q(t) at a supplied or current time."""
        t = self.time if time is None else float(time)
        if not self.perturbation.enabled:
            return 0.0
        return self.forcing.value(t)

    def _area_unchecked(self, x, time):
        """Return A(x, t) without positivity validation."""
        A_base = self.base_geometry.area(x)
        q = self.current_amplitude(time)
        if abs(q) == 0.0:
            return A_base
        return A_base + q * self.perturbation.shape(x, self.base_geometry)

    def _validate_area_values(self, A):
        """Raise if any sampled area violates the positivity floor."""
        min_value = float(np.min(A))
        if min_value <= self.min_area:
            raise ValueError(
                f"Time-dependent perturbed area violates min_area: "
                f"min(A)={min_value:.6e}, min_area={self.min_area:.6e}"
            )

    def _sample_validation_times(self, n_phase=33):
        """Sample one forcing cycle, or one representative static time."""
        if self.forcing.frequency_hz == 0.0:
            return np.array([0.0])
        period = 1.0 / self.forcing.frequency_hz
        return np.linspace(0.0, period, n_phase)

    def _validate_profile_over_cycle(self, n_x=1000):
        """Best-effort positivity check over one forcing cycle."""
        x = np.linspace(0.0, self.L_total, n_x)
        for t in self._sample_validation_times():
            self._validate_area_values(self._area_unchecked(x, t))

    def area_at_time(self, x, time):
        """Return A(x, t)."""
        A = self._area_unchecked(x, time)
        self._validate_area_values(A)
        return A

    def area_gradient_at_time(self, x, time):
        """Return dA/dx at a supplied physical time."""
        dAdx = self.base_geometry.area_gradient(x)
        q = self.current_amplitude(time)
        if abs(q) == 0.0:
            return dAdx
        return dAdx + q * self.perturbation.shape_gradient(x, self.base_geometry)

    def area_time_derivative_at_time(self, x, time):
        """Return dA/dt(x, t) = (dq/dt) * phi(x) for the breathing mode."""
        x = np.asarray(x, dtype=np.float64)
        if not self.perturbation.enabled:
            return np.zeros_like(x)
        dqdt = self.forcing.rate(time)
        if dqdt == 0.0:
            return np.zeros_like(x)
        return dqdt * self.perturbation.shape(x, self.base_geometry)

    def area_time_derivative(self, x):
        """Return dA/dt at the currently active time (wall-velocity term)."""
        return self.area_time_derivative_at_time(x, self.time)

    def area(self, x):
        """Return A(x) at the currently active time."""
        return self.area_at_time(x, self.time)

    def area_gradient(self, x):
        """Return dA/dx at the currently active time."""
        return self.area_gradient_at_time(x, self.time)

    def min_area_value(self, n_pts=1000, time=None):
        """Sample the minimum area at the supplied or current time."""
        t = self.time if time is None else float(time)
        x = np.linspace(0.0, self.L_total, n_pts)
        return float(np.min(self.area_at_time(x, t)))

    def max_area_value(self, n_pts=1000, time=None):
        """Sample the maximum area at the supplied or current time."""
        t = self.time if time is None else float(time)
        x = np.linspace(0.0, self.L_total, n_pts)
        return float(np.max(self.area_at_time(x, t)))

    def throat_area(self, time=None):
        """Return the effective throat area at the supplied or current time."""
        t = self.time if time is None else float(time)
        return float(self.area_at_time(np.array([self.x_throat]), t)[0])

    def to_dict(self):
        """JSON-serialisable representation."""
        return {
            "type": "TimeDependentPerturbedGeometryProfile",
            "base": geometry_to_dict(self.base_geometry),
            "perturbation": self.perturbation.to_dict(),
            "forcing": self.forcing.to_dict(),
            "min_area": self.min_area,
            "current_time": self.time,
            "current_amplitude": self.current_amplitude(),
            "throat_area": self.throat_area(),
            "min_sampled_area": self.min_area_value(),
        }


class TabulatedAreaProfile:
    """
    Area law defined by sampled (x, A) pairs with monotone PCHIP interpolation.

    Mirrors the `GeometryProfile` interface used by the solver (`area`,
    `area_gradient`, `copy`, section attributes), so calibrated experiment
    geometries (Config A ramp/isolator, Busemann ducts) can drive the same
    quasi-1D machinery as the parametric three-section duct.

    Section attributes are derived: x_throat is the area minimum; the
    "combustor" is collapsed to zero length (x_comb_exit = x_throat) since
    tabulated experiment ducts have no meaningful three-section split.
    """

    def __init__(self, x_samples, A_samples, name="tabulated"):
        from scipy.interpolate import PchipInterpolator

        x = np.asarray(x_samples, dtype=np.float64)
        A = np.asarray(A_samples, dtype=np.float64)
        if x.ndim != 1 or x.shape != A.shape or len(x) < 4:
            raise ValueError("need matching 1-D x/A samples (>= 4 points)")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("x samples must be strictly increasing")
        if np.any(A <= 0.0):
            raise ValueError("all sampled areas must be positive")
        if abs(float(x[0])) > 1e-12:
            raise ValueError("x samples must start at 0")

        self.name = str(name)
        self.x_samples = x.copy()
        self.A_samples = A.copy()
        self._interp = PchipInterpolator(x, A, extrapolate=True)
        self._interp_deriv = self._interp.derivative()

        # GeometryProfile-compatible attributes
        self.L_total = float(x[-1])
        i_min = int(np.argmin(A))
        self.x_throat = float(x[i_min])
        self.A_inlet = float(A[0])
        self.A_throat = float(A[i_min])
        self.A_exit = float(A[-1])
        self.A_comb_exit = self.A_throat
        self.x_comb_exit = self.x_throat
        self.x_exit = self.L_total
        self.L_inlet = self.x_throat
        self.L_combustor = 0.0
        self.L_nozzle = self.L_total - self.x_throat

    def copy(self):
        """Return an independent copy of this profile."""
        return TabulatedAreaProfile(self.x_samples, self.A_samples, name=self.name)

    def area(self, x):
        """Interpolated A(x); clamped to the sampled range at the ends."""
        x = np.clip(np.asarray(x, dtype=np.float64), 0.0, self.L_total)
        return self._interp(x)

    def area_gradient(self, x):
        """Interpolated dA/dx from the PCHIP derivative."""
        x = np.clip(np.asarray(x, dtype=np.float64), 0.0, self.L_total)
        return self._interp_deriv(x)

    def to_dict(self):
        """JSON-serialisable representation (samples included)."""
        return {
            "type": "TabulatedAreaProfile",
            "name": self.name,
            "L_total": self.L_total,
            "x_throat": self.x_throat,
            "A_inlet": self.A_inlet,
            "A_throat": self.A_throat,
            "A_exit": self.A_exit,
            "n_samples": int(len(self.x_samples)),
            "x_samples": [float(v) for v in self.x_samples],
            "A_samples": [float(v) for v in self.A_samples],
        }


def config_a_ramp_area_law(L_ramp=0.100, L_isolator=0.060,
                           H_capture=0.050, incidence_deg=8.33,
                           total_turn_deg=18.51, n_samples=200):
    """
    Approximate effective-area law for the UNSW Config A ramp + isolator.

    Models the captured streamtube of the cantilevered concave (isentropic)
    compression ramp of Bhattrai et al. (JPP 38(1), 2022): the local flow
    turning grows linearly from `incidence_deg` at the leading edge to
    `total_turn_deg` at the ramp end, contracting the captured height
    dH/dx = -tan(theta(x)); the isolator then holds the area constant.

    IMPORTANT: the turning schedule and angles are documented in the paper,
    but the default lengths/capture height here are PLACEHOLDERS chosen at
    model scale — calibrate L_ramp, L_isolator, H_capture against the paper
    drawings / UNSW database before quantitative comparisons. Areas are per
    unit depth (2-D planar), consistent with the rest of the repo.

    Returns:
        TabulatedAreaProfile
    """
    th0 = np.deg2rad(float(incidence_deg))
    th1 = np.deg2rad(float(total_turn_deg))
    x_ramp = np.linspace(0.0, L_ramp, max(n_samples // 2, 8))
    theta = th0 + (th1 - th0) * (x_ramp / L_ramp)

    # captured-height contraction: dH/dx = -tan(theta(x))
    H = np.empty_like(x_ramp)
    H[0] = H_capture
    for i in range(1, len(x_ramp)):
        dx = x_ramp[i] - x_ramp[i - 1]
        H[i] = H[i - 1] - np.tan(0.5 * (theta[i] + theta[i - 1])) * dx
    if H[-1] <= 0.0:
        raise ValueError("Config A ramp closes the duct: reduce L_ramp or "
                         "increase H_capture")

    x_iso = np.linspace(L_ramp, L_ramp + L_isolator, max(n_samples // 4, 6))[1:]
    x = np.concatenate([x_ramp, x_iso])
    A = np.concatenate([H, np.full(len(x_iso), H[-1])])
    return TabulatedAreaProfile(x, A, name="config_a_ramp_isolator")


def geometry_to_dict(geometry):
    """JSON-serialisable representation of a geometry object."""
    if hasattr(geometry, "to_dict"):
        return geometry.to_dict()
    return {
        "type": "GeometryProfile",
        "L_inlet": geometry.L_inlet,
        "L_combustor": geometry.L_combustor,
        "L_nozzle": geometry.L_nozzle,
        "L_total": geometry.L_total,
        "A_inlet": geometry.A_inlet,
        "A_throat": geometry.A_throat,
        "A_comb_exit": geometry.A_comb_exit,
        "A_exit": geometry.A_exit,
        "x_throat": geometry.x_throat,
        "x_comb_exit": geometry.x_comb_exit,
        "x_exit": geometry.x_exit,
    }


# Standalone test

if __name__ == "__main__":
    print("=== StructuredMesh2D: uniform ===")
    mesh = StructuredMesh2D.uniform(0.0, 1.2, 0.0, 0.1, 60, 10)
    print(f"  nx={mesh.nx}, ny={mesh.ny}, n_cells={mesh.n_cells}")
    print(f"  dx range: [{mesh.dx.min():.4f}, {mesh.dx.max():.4f}]")
    print(f"  dy range: [{mesh.dy.min():.4f}, {mesh.dy.max():.4f}]")
    print(f"  vol sum:  {mesh.vol.sum():.6f} m^2  (expected: {1.2*0.1:.6f})")

    print("\n=== StructuredMesh2D: stretched ===")
    mesh_s = StructuredMesh2D.stretched(0.0, 1.2, 0.0, 0.1, 60, 10, y_ratio=1.15)
    print(f"  dy[0]={mesh_s.dy[0]:.6f}, dy[-1]={mesh_s.dy[-1]:.6f}")
    print(f"  ratio dy[-1]/dy[0] = {mesh_s.dy[-1]/mesh_s.dy[0]:.2f}")

    print("\n=== GeometryProfile: default ===")
    gp = GeometryProfile.default()
    x_test = np.array([0.0, gp.x_throat, gp.x_comb_exit, gp.x_exit])
    A_test = gp.area(x_test)
    print(f"  A(0)       = {A_test[0]:.4f}  (expect {gp.A_inlet:.4f})")
    print(f"  A(throat)  = {A_test[1]:.4f}  (expect {gp.A_throat:.4f})")
    print(f"  A(comb)    = {A_test[2]:.4f}  (expect {gp.A_comb_exit:.4f})")
    print(f"  A(exit)    = {A_test[3]:.4f}  (expect {gp.A_exit:.4f})")

    print("\nAll mesh tests passed.")
