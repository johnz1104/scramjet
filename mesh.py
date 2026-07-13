"""
mesh.py — Spatial domain for the 2D scramjet CFD solver.

Contains:
    StructuredMesh2D   — 2D structured quadrilateral mesh with face geometry
    GeometryProfile    — Parameterised scramjet duct area profile A(x)

Dependency: numpy, matplotlib (plotting only)
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


CONFIG_A_GEOMETRY_LINEAGE_ID = "unsw_ramp1018_source_reconstructed_v1"
CONFIG_A_GEOMETRY_ARTIFACT = (
    Path(__file__).resolve().parent
    / "configs" / "geometry" / "config_a_source_reconstructed_v1.json"
)


def _mirror_optional_geometry_metadata(target, source):
    """Copy additive research metadata without changing the solver interface."""
    for name in (
        "geometry_lineage_id",
        "reconstruction_status",
        "source_artifact_checksum_sha256",
        "reduced_frequency_length_ref_m",
        "positive_displacement_convention",
        "nominal_stations",
    ):
        if hasattr(source, name):
            setattr(target, name, copy.deepcopy(getattr(source, name)))


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
            "type": "LocalizedAreaPerturbation",
            "enabled": self.enabled,
            "mode": self.mode,
            "amplitude": self.amplitude,
            "x_center": self.x_center,
            "width": self.width,
            "min_area": self.min_area,
            "active": self.active,
        }


class TabulatedAreaPerturbation:
    """Dimensionless perturbation mode tabulated along solver ``x``.

    The scalar amplitude retains the existing raw effective-area units.  The
    table supplies only ``phi(x)`` in ``A=A_base+q*phi`` and therefore works
    with both static and sinusoidal wrappers without special solver branches.
    PCHIP preserves the monotone cantilever shape and the explicit zero tail.
    """

    def __init__(self, x_samples, shape_samples, enabled=True,
                 mode="tabulated", amplitude=0.0, min_area=1.0e-6,
                 metadata=None):
        from scipy.interpolate import PchipInterpolator

        x = np.asarray(x_samples, dtype=np.float64)
        shape = np.asarray(shape_samples, dtype=np.float64)
        if x.ndim != 1 or x.shape != shape.shape or len(x) < 4:
            raise ValueError("need matching 1-D x/shape samples (>= 4 points)")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("tabulated perturbation x samples must increase")
        if min_area <= 0.0:
            raise ValueError("min_area must be positive")
        self.x_samples = x.copy()
        self.shape_samples = shape.copy()
        self.enabled = bool(enabled)
        self.mode = str(mode)
        self.amplitude = float(amplitude)
        self.min_area = float(min_area)
        self.metadata = copy.deepcopy(metadata or {})
        self._interp = PchipInterpolator(x, shape, extrapolate=False)
        self._interp_deriv = self._interp.derivative()

    @property
    def active(self):
        return self.enabled and abs(self.amplitude) > 0.0

    def copy(self):
        return TabulatedAreaPerturbation(
            self.x_samples, self.shape_samples,
            enabled=self.enabled, mode=self.mode,
            amplitude=self.amplitude, min_area=self.min_area,
            metadata=self.metadata,
        )

    def shape(self, x, base_geometry):
        del base_geometry
        values = np.asarray(x, dtype=np.float64)
        clipped = np.clip(values, self.x_samples[0], self.x_samples[-1])
        result = np.asarray(self._interp(clipped), dtype=np.float64)
        result = np.where(
            (values < self.x_samples[0]) | (values > self.x_samples[-1]),
            0.0, result,
        )
        return result

    def shape_gradient(self, x, base_geometry):
        del base_geometry
        values = np.asarray(x, dtype=np.float64)
        clipped = np.clip(values, self.x_samples[0], self.x_samples[-1])
        result = np.asarray(self._interp_deriv(clipped), dtype=np.float64)
        return np.where(
            (values < self.x_samples[0]) | (values > self.x_samples[-1]),
            0.0, result,
        )

    def to_dict(self):
        return {
            "type": "TabulatedAreaPerturbation",
            "enabled": self.enabled,
            "mode": self.mode,
            "amplitude": self.amplitude,
            "min_area": self.min_area,
            "active": self.active,
            "x_samples": [float(value) for value in self.x_samples],
            "shape_samples": [float(value) for value in self.shape_samples],
            "metadata": copy.deepcopy(self.metadata),
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
        _mirror_optional_geometry_metadata(self, self.base_geometry)

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
        data = {
            "type": "PerturbedGeometryProfile",
            "base": geometry_to_dict(self.base_geometry),
            "perturbation": self.perturbation.to_dict(),
            "min_area": self.min_area,
            "throat_area": self.throat_area(),
            "min_sampled_area": self.min_area_value(),
        }
        if getattr(self, "geometry_lineage_id", None) is not None:
            data["geometry_lineage_id"] = self.geometry_lineage_id
            data["reconstruction_status"] = getattr(
                self, "reconstruction_status", None,
            )
            data["reduced_frequency_length_ref_m"] = getattr(
                self, "reduced_frequency_length_ref_m", None,
            )
        return data

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
        _mirror_optional_geometry_metadata(self, self.base_geometry)

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
        data = {
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
        if getattr(self, "geometry_lineage_id", None) is not None:
            data["geometry_lineage_id"] = self.geometry_lineage_id
            data["reconstruction_status"] = getattr(
                self, "reconstruction_status", None,
            )
            data["reduced_frequency_length_ref_m"] = getattr(
                self, "reduced_frequency_length_ref_m", None,
            )
        return data


class TabulatedAreaProfile:
    """
    Area law defined by sampled (x, A) pairs with monotone PCHIP interpolation.

    Mirrors the `GeometryProfile` interface used by the solver (`area`,
    `area_gradient`, `copy`, section attributes), so sourced/reconstructed
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


class WallContourGeometryProfile(TabulatedAreaProfile):
    """Tabulated area law retaining the physical lower and upper walls.

    The solver still consumes only ``area(x)`` and ``area_gradient(x)``.  The
    additional contours and lineage prevent an effective closed duct from
    being confused with the reconstructed external-inlet topology during
    export.
    """

    def __init__(self, x_samples, A_samples, ramp_x, ramp_y,
                 cowl_x, cowl_y, name="wall_contour", metadata=None):
        super().__init__(x_samples, A_samples, name=name)
        ramp_x = np.asarray(ramp_x, dtype=np.float64)
        ramp_y = np.asarray(ramp_y, dtype=np.float64)
        cowl_x = np.asarray(cowl_x, dtype=np.float64)
        cowl_y = np.asarray(cowl_y, dtype=np.float64)
        if ramp_x.ndim != 1 or ramp_x.shape != ramp_y.shape or len(ramp_x) < 4:
            raise ValueError("need matching physical ramp coordinates")
        if cowl_x.ndim != 1 or cowl_x.shape != cowl_y.shape or len(cowl_x) < 2:
            raise ValueError("need matching physical cowl coordinates")
        if np.any(np.diff(ramp_x) <= 0.0) or np.any(np.diff(cowl_x) <= 0.0):
            raise ValueError("physical wall x coordinates must increase")
        self.ramp_x = ramp_x.copy()
        self.ramp_y = ramp_y.copy()
        self.cowl_x = cowl_x.copy()
        self.cowl_y = cowl_y.copy()
        self.metadata = copy.deepcopy(metadata or {})
        self.geometry_lineage_id = self.metadata.get("geometry_lineage_id")
        self.reconstruction_status = self.metadata.get("reconstruction_status")
        self.source_artifact_checksum_sha256 = self.metadata.get(
            "source_artifact_checksum_sha256",
        )
        self.reduced_frequency_length_ref_m = float(self.metadata.get(
            "reduced_frequency_length_ref_m", self.L_total,
        ))
        self.positive_displacement_convention = self.metadata.get(
            "positive_displacement_convention",
            "positive q increases effective gap",
        )
        self.nominal_stations = copy.deepcopy(
            self.metadata.get("nominal_stations", {}),
        )
        self.section_break_indices = copy.deepcopy(
            self.metadata.get("section_break_indices", {}),
        )

    def copy(self):
        return WallContourGeometryProfile(
            self.x_samples, self.A_samples,
            self.ramp_x, self.ramp_y, self.cowl_x, self.cowl_y,
            name=self.name, metadata=self.metadata,
        )

    def surface_arc_coordinate(self, x):
        """Arc coordinate from the leading edge, clamped at the support."""
        from scipy.interpolate import PchipInterpolator

        support_index = int(self.section_break_indices.get(
            "support", len(self.ramp_x) - 1,
        ))
        wall_x = self.ramp_x[:support_index + 1]
        wall_y = self.ramp_y[:support_index + 1]
        s_polyline = np.r_[0.0, np.cumsum(np.hypot(
            np.diff(wall_x), np.diff(wall_y),
        ))]
        if s_polyline[-1] <= 0.0:
            raise ValueError("physical ramp arc length is zero")
        s_polyline *= self.reduced_frequency_length_ref_m / s_polyline[-1]
        interp = PchipInterpolator(wall_x, s_polyline, extrapolate=False)
        values = np.asarray(x, dtype=np.float64)
        clipped = np.clip(values, wall_x[0], wall_x[-1])
        return np.asarray(interp(clipped), dtype=np.float64)

    def to_dict(self):
        return {
            "type": "WallContourGeometryProfile",
            "name": self.name,
            "L_total": self.L_total,
            "x_throat": self.x_throat,
            "A_inlet": self.A_inlet,
            "A_throat": self.A_throat,
            "A_exit": self.A_exit,
            "n_samples": int(len(self.x_samples)),
            "x_samples": [float(value) for value in self.x_samples],
            "A_samples": [float(value) for value in self.A_samples],
            "physical_walls": {
                "ramp_x": [float(value) for value in self.ramp_x],
                "ramp_y": [float(value) for value in self.ramp_y],
                "cowl_x": [float(value) for value in self.cowl_x],
                "cowl_y": [float(value) for value in self.cowl_y],
            },
            "geometry_lineage_id": self.geometry_lineage_id,
            "reconstruction_status": self.reconstruction_status,
            "source_artifact_checksum_sha256": self.source_artifact_checksum_sha256,
            "reduced_frequency_length_ref_m": self.reduced_frequency_length_ref_m,
            "positive_displacement_convention": self.positive_displacement_convention,
            "nominal_stations": copy.deepcopy(self.nominal_stations),
            "section_break_indices": copy.deepcopy(self.section_break_indices),
            "metadata": copy.deepcopy(self.metadata),
        }


def _canonical_artifact_checksum(data):
    payload = dict(data)
    payload.pop("artifact_checksum_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_config_a_geometry(artifact_path=None):
    """Load and verify the frozen Config-A reconstruction artifact."""
    path = CONFIG_A_GEOMETRY_ARTIFACT if artifact_path is None else Path(artifact_path)
    data = json.loads(path.read_text())
    expected = data.get("artifact_checksum_sha256")
    actual = _canonical_artifact_checksum(data)
    if expected != actual:
        raise ValueError(
            f"Config-A geometry artifact checksum mismatch: expected {expected}, got {actual}"
        )
    if data.get("geometry_lineage_id") != CONFIG_A_GEOMETRY_LINEAGE_ID:
        raise ValueError("unsupported Config-A geometry lineage")
    if data.get("reconstruction_status") != "source_reconstructed_v1":
        raise ValueError("Config-A artifact is not source_reconstructed_v1")
    if data.get("hard_constraints_within_published_rounding") is not True:
        raise ValueError("Config-A artifact failed a hard source constraint")
    area = data["effective_area_per_unit_depth"]
    walls = data["physical_wall_coordinates"]
    ramp = walls["ramp"]
    cowl = walls["cowl"]
    return WallContourGeometryProfile(
        area["x_m_solver"], area["area_m2_per_m_depth"],
        ramp["x_m_solver"], ramp["y_m_original"],
        cowl["x_m_solver"], cowl["y_m_original"],
        name="config_a_source_reconstructed_v1",
        metadata={
            "geometry_lineage_id": data["geometry_lineage_id"],
            "reconstruction_status": data["reconstruction_status"],
            "source_artifact_checksum_sha256": expected,
            "source_artifact": str(path),
            "reduced_frequency_length_ref_m": data["derived_values"][
                "reduced_frequency_reference_length_m"
            ],
            "positive_displacement_convention": (
                "positive displacement is downward ramp motion / increased gap"
            ),
            "nominal_stations": data["nominal_stations"],
            "section_break_indices": ramp["section_break_indices"],
            "derived_values": data["derived_values"],
            "provenance": data["provenance"],
            "comparison_topologies": [
                "closed_effective_duct", "config_a_external_inlet",
            ],
        },
    )


def config_a_cantilever_mode(base_geometry, amplitude=0.0,
                             min_area=1.0e-6):
    """Return the assumed classical first cantilever mode for Config A."""
    if getattr(base_geometry, "geometry_lineage_id", None) != CONFIG_A_GEOMETRY_LINEAGE_ID:
        raise ValueError("Config-A cantilever mode requires reconstructed geometry lineage")
    beta = 1.875104
    support = float(base_geometry.nominal_stations["support_x_m_solver"])
    x = base_geometry.ramp_x.copy()
    s = base_geometry.surface_arc_coordinate(x)
    length = float(base_geometry.reduced_frequency_length_ref_m)
    xi = np.clip((length - s) / length, 0.0, 1.0)
    sigma = ((np.cosh(beta) + np.cos(beta))
             / (np.sinh(beta) + np.sin(beta)))
    shape = (
        np.cosh(beta * xi) - np.cos(beta * xi)
        - sigma * (np.sinh(beta * xi) - np.sin(beta * xi))
    )
    shape /= float(shape[0])
    shape[x >= support] = 0.0
    return TabulatedAreaPerturbation(
        x, shape, enabled=True,
        mode="model_assumed_cantilever_first_mode",
        amplitude=amplitude, min_area=min_area,
        metadata={
            "beta_1": beta,
            "surface_coordinate": "reconstructed ramp arc length",
            "normalization": "unit gap increase at leading edge",
            "support_boundary": "zero displacement and slope",
            "downstream_of_support": "zero",
            "positive_displacement": "downward ramp motion / increased gap",
            "dic_calibrated": False,
            "sampling_guidance_hz": {
                "manufactured_ramp_impulse_test": 55.5,
                "fem_first_mode": 60.0,
                "dic_response": 52.0,
            },
        },
    )


def config_a_normalized_to_raw(value, geometry):
    """Convert Delta-Y_LE/S to raw per-unit-depth effective-area amplitude."""
    length = float(geometry.reduced_frequency_length_ref_m)
    return float(value) * length


def config_a_raw_to_normalized(value, geometry):
    """Convert raw per-unit-depth effective-area amplitude to Delta-Y_LE/S."""
    length = float(geometry.reduced_frequency_length_ref_m)
    return float(value) / length


def config_a_ramp_area_law(artifact_path=None, **legacy_overrides):
    """Load the frozen source-constrained Ramp-1018 reconstruction.

    The former tunable placeholder has intentionally been removed.  Supplying
    its old scale/shape arguments would create an untraceable Config-A claim,
    so callers must use a separate generic ``TabulatedAreaProfile`` instead.
    """
    if legacy_overrides:
        names = ", ".join(sorted(legacy_overrides))
        raise ValueError(
            f"Config-A reconstruction is frozen; unsupported overrides: {names}"
        )
    return load_config_a_geometry(artifact_path)


def perturbation_from_dict(data):
    """Rebuild either legacy Gaussian or new tabulated perturbation metadata."""
    if data.get("type") == "TabulatedAreaPerturbation" or "x_samples" in data:
        return TabulatedAreaPerturbation(
            data["x_samples"], data["shape_samples"],
            enabled=data.get("enabled", True),
            mode=data.get("mode", "tabulated"),
            amplitude=data.get("amplitude", 0.0),
            min_area=data.get("min_area", 1.0e-6),
            metadata=data.get("metadata"),
        )
    return LocalizedAreaPerturbation(
        enabled=data.get("enabled", True),
        mode=data.get("mode", "throat_gaussian"),
        amplitude=data.get("amplitude", 0.0),
        x_center=data.get("x_center"),
        width=data.get("width", 0.05),
        min_area=data.get("min_area", 1.0e-6),
    )


def geometry_from_dict(data):
    """Rebuild every geometry type used by solver, ROM, and export paths."""
    geometry_type = data["type"]
    if geometry_type == "GeometryProfile":
        return GeometryProfile(
            data["L_inlet"], data["L_combustor"], data["L_nozzle"],
            data["A_inlet"], data["A_throat"],
            data["A_comb_exit"], data["A_exit"],
        )
    if geometry_type == "TabulatedAreaProfile":
        return TabulatedAreaProfile(
            data["x_samples"], data["A_samples"],
            name=data.get("name", "tabulated"),
        )
    if geometry_type == "WallContourGeometryProfile":
        walls = data["physical_walls"]
        metadata = copy.deepcopy(data.get("metadata") or {})
        for key in (
            "geometry_lineage_id", "reconstruction_status",
            "source_artifact_checksum_sha256", "reduced_frequency_length_ref_m",
            "positive_displacement_convention", "nominal_stations",
            "section_break_indices",
        ):
            if key in data:
                metadata[key] = copy.deepcopy(data[key])
        return WallContourGeometryProfile(
            data["x_samples"], data["A_samples"],
            walls["ramp_x"], walls["ramp_y"],
            walls["cowl_x"], walls["cowl_y"],
            name=data.get("name", "wall_contour"), metadata=metadata,
        )
    if geometry_type == "PerturbedGeometryProfile":
        return PerturbedGeometryProfile(
            geometry_from_dict(data["base"]),
            perturbation_from_dict(data["perturbation"]),
        )
    if geometry_type == "TimeDependentPerturbedGeometryProfile":
        forcing_data = data.get("forcing", {})
        forcing = SinusoidalAreaForcing(
            amplitude=forcing_data.get("amplitude", 0.0),
            frequency_hz=forcing_data.get("frequency_hz", 0.0),
            phase=forcing_data.get("phase", 0.0),
            enabled=forcing_data.get("enabled", True),
            mean=forcing_data.get("mean", 0.0),
        )
        geometry = TimeDependentPerturbedGeometryProfile(
            geometry_from_dict(data["base"]),
            perturbation_from_dict(data["perturbation"]),
            forcing,
        )
        geometry.time = float(data.get("current_time", 0.0))
        return geometry
    raise ValueError(f"Unsupported geometry type: {geometry_type}")


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
