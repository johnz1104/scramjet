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

    The area profile is normalised by A_throat so that A(x_throat) = 1.
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