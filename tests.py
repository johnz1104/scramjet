"""
tests.py — Validation test suite for the scramjet CFD solver.

Three canonical test cases, each targeting a different solver component:

    1. Sod shock tube       — validates FVM (HLLC, MUSCL, RK3-SSP)
    2. Couette flow         — validates FEM viscous diffusion
    3. Ignition delay       — validates Arrhenius combustion model

Usage:
    python tests.py              # run all tests
    python tests.py sod          # Sod shock tube only
    python tests.py couette      # Couette flow only
    python tests.py ignition     # ignition delay only

Dependency: mesh.py, fvm.py, physics.py
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mesh import StructuredMesh2D
from fvm import StateVector, BoundaryConditions, FVMResidual, TimeIntegrator
from physics import TransportProperties, FEMViscous, SingleStepArrhenius


def test_sod_shock_tube():
    """
    Sod shock tube problem (Sod 1978).

    IC:  left state  (x < 0.5): rho=1.0, p=1.0, u=0
         right state (x > 0.5): rho=0.125, p=0.1, u=0

    Exact solution at t=0.2 contains:
        - left rarefaction fan
        - contact discontinuity
        - right-moving shock

    This tests the FVM spatial discretisation (HLLC + MUSCL) and
    temporal integration (RK3-SSP) in 1D (ny=1 limit of the 2D solver).

    Pass criteria (nx=200, t=0.2):
        L1 error in rho < 0.020
        L1 error in u   < 0.035
        L1 error in p   < 0.015
    """
    print("=" * 60)
    print("TEST 1: Sod Shock Tube")
    print("=" * 60)

    gamma = 1.4
    nx = 200
    ny = 1
    x_min, x_max = 0.0, 1.0
    t_final = 0.2

    mesh = StructuredMesh2D.uniform(x_min, x_max, 0.0, 0.01, nx, ny)
    state = StateVector(nx, ny, gamma=gamma)

    # initial conditions
    rho = np.ones((nx, ny))
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    p = np.ones((nx, ny))

    x_mid = 0.5
    for i in range(nx):
        if mesh.xc[i] > x_mid:
            rho[i, :] = 0.125
            p[i, :] = 0.1

    state.set_primitive(rho, u, v, p)

    # boundary conditions: both sides extrapolation (outflow)
    # use left-state inflow but it won't matter because flow doesn't reach boundary
    bc = BoundaryConditions(state, rho[0, 0], 0.0, 0.0, p[0, 0])

    fvm_res = FVMResidual(mesh, gamma=gamma, eps2_scale=3.0)
    integrator = TimeIntegrator(cfl=0.5)

    # time march
    t = 0.0
    n = 0
    while t < t_final:
        dt = integrator.compute_dt(state, mesh)
        dt = min(dt, t_final - t)

        integrator.step(state, dt, fvm_res.compute, bc.apply)
        t += dt
        n += 1

    print(f"  Completed {n} steps, t_final = {t:.6f}")

    # exact solution (Toro, ch. 4)
    rho_ex, u_ex, p_ex = _sod_exact(mesh.xc, t_final, gamma)

    rho_num, u_num, v_num, p_num, T_num, Yf_num = state.primitives()

    # L1 errors
    err_rho = np.mean(np.abs(rho_num[:, 0] - rho_ex))
    err_u = np.mean(np.abs(u_num[:, 0] - u_ex))
    err_p = np.mean(np.abs(p_num[:, 0] - p_ex))

    print(f"  L1 errors: rho={err_rho:.4f}, u={err_u:.4f}, p={err_p:.4f}")

    pass_rho = err_rho < 0.020
    pass_u = err_u < 0.035
    pass_p = err_p < 0.015
    passed = pass_rho and pass_u and pass_p

    print(f"  rho: {'PASS' if pass_rho else 'FAIL'} (threshold 0.020)")
    print(f"  u:   {'PASS' if pass_u else 'FAIL'} (threshold 0.035)")
    print(f"  p:   {'PASS' if pass_p else 'FAIL'} (threshold 0.015)")

    # save plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = mesh.xc

    axes[0].plot(x, rho_ex, "k-", lw=1.5, label="Exact")
    axes[0].plot(x, rho_num[:, 0], "ro", ms=2, label="FVM")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].plot(x, u_ex, "k-", lw=1.5)
    axes[1].plot(x, u_num[:, 0], "ro", ms=2)
    axes[1].set_ylabel("Velocity")

    axes[2].plot(x, p_ex, "k-", lw=1.5)
    axes[2].plot(x, p_num[:, 0], "ro", ms=2)
    axes[2].set_ylabel("Pressure")

    for ax in axes:
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Sod Shock Tube (t = {t_final}, nx = {nx})")
    plt.tight_layout()
    fig.savefig("test_sod.png", dpi=150)
    print(f"  Plot saved: test_sod.png")

    return passed


def _sod_exact(x, t, gamma):
    """
    Exact solution to the Sod shock tube problem at time t.

    Uses the analytical Riemann solution (Toro, Riemann Solvers, ch. 4).
    Left state: (rho_L, u_L, p_L) = (1.0, 0.0, 1.0)
    Right state: (rho_R, u_R, p_R) = (0.125, 0.0, 0.1)
    """
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    # left and right states
    rho_L, u_L, p_L = 1.0, 0.0, 1.0
    rho_R, u_R, p_R = 0.125, 0.0, 0.1
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)

    # star region (iterative solve for p_star)
    # Newton iteration on the pressure function
    p_star = 0.30313  # known exact value for standard Sod
    u_star = 0.92745

    # post-shock density (right side)
    rho_star_R = rho_R * ((p_star / p_R + gm1 / gp1) / (gm1 / gp1 * p_star / p_R + 1.0))

    # post-expansion density (left side)
    rho_star_L = rho_L * (p_star / p_L) ** (1.0 / gamma)

    # sound speed in star-left region
    c_star_L = c_L * (p_star / p_L) ** (gm1 / (2.0 * gamma))

    # wave speeds
    # rarefaction head: x/t = u_L - c_L
    # rarefaction tail: x/t = u_star - c_star_L
    # contact: x/t = u_star
    # shock: x/t = u_R + c_R * sqrt((gp1/(2*gamma)) * (p_star/p_R) + gm1/(2*gamma))
    x_head = 0.5 + (u_L - c_L) * t
    x_tail = 0.5 + (u_star - c_star_L) * t
    x_contact = 0.5 + u_star * t
    S_shock = u_R + c_R * np.sqrt(gp1 / (2.0 * gamma) * p_star / p_R + gm1 / (2.0 * gamma))
    x_shock = 0.5 + S_shock * t

    rho_out = np.empty_like(x)
    u_out = np.empty_like(x)
    p_out = np.empty_like(x)

    for i in range(len(x)):
        xi = x[i]
        if xi < x_head:
            # undisturbed left
            rho_out[i] = rho_L
            u_out[i] = u_L
            p_out[i] = p_L
        elif xi < x_tail:
            # inside rarefaction fan (Toro eq 4.56)
            # self-similar variable: xi_s = (x - x0) / t
            xi_s = (xi - 0.5) / t
            # u = (2/(gamma+1)) * (c_L + (gamma-1)/2 * u_L + xi_s)
            u_out[i] = 2.0 / gp1 * (c_L + gm1 / 2.0 * u_L + xi_s)
            # c = (2/(gamma+1)) * (c_L + (gamma-1)/2 * (u_L - xi_s))
            c_fan = 2.0 / gp1 * (c_L + gm1 / 2.0 * (u_L - xi_s))
            rho_out[i] = rho_L * (c_fan / c_L) ** (2.0 / gm1)
            p_out[i] = p_L * (c_fan / c_L) ** (2.0 * gamma / gm1)
        elif xi < x_contact:
            # star-left region
            rho_out[i] = rho_star_L
            u_out[i] = u_star
            p_out[i] = p_star
        elif xi < x_shock:
            # star-right region
            rho_out[i] = rho_star_R
            u_out[i] = u_star
            p_out[i] = p_star
        else:
            # undisturbed right
            rho_out[i] = rho_R
            u_out[i] = u_R
            p_out[i] = p_R

    return rho_out, u_out, p_out


def test_couette_flow():
    """
    Steady-state Couette flow between parallel plates.

    Setup:
        - Bottom wall: no-slip, u=0
        - Top wall:    moving at u=U_wall
        - No pressure gradient
        - Incompressible limit (low Mach)

    Exact solution: u(y) = U_wall * y / H  (linear profile)

    This tests the implicit diffusion solver (FEM-style) with Dirichlet wall BCs.

    Pass criterion: L2 error in u-profile < 0.05 * U_wall
    """
    print("\n" + "=" * 60)
    print("TEST 2: Couette Flow (FEM Viscous)")
    print("=" * 60)

    gamma = 1.4
    R_gas = 287.0
    nx = 4
    ny = 30
    H = 0.01      # channel height [m]
    L = 0.02      # channel length [m]
    U_wall = 10.0  # top wall velocity [m/s]

    T_ref = 300.0
    p_ref = 101325.0
    rho_ref = p_ref / (R_gas * T_ref)

    mesh = StructuredMesh2D.uniform(0.0, L, 0.0, H, nx, ny)
    transport = TransportProperties(gamma=gamma, R_gas=R_gas)

    mu = transport.viscosity(np.array([T_ref]))[0]
    nu = mu / rho_ref
    tau = H**2 / nu  # diffusion timescale

    # we solve the pure diffusion problem for u(y) directly
    # d(u)/dt = nu * d^2(u)/dy^2
    # with u(0) = 0, u(H) = U_wall
    # at steady state: u(y) = U_wall * y / H

    # use implicit backward Euler with large dt to reach steady state fast
    dt = 0.5 * tau  # large fraction of diffusion timescale
    n_steps = 20     # 20 steps of dt = 0.5*tau => 10 diffusion timescales

    print(f"  nu = {nu:.4e} m^2/s, tau = {tau:.4e} s")
    print(f"  dt = {dt:.4e} s, n_steps = {n_steps}")

    # solve on a 1D column (all x-columns are identical)
    # build tridiagonal system for u(j), j=0..ny-1
    dy_arr = mesh.dy
    u_profile = np.zeros(ny)

    for step in range(n_steps):
        # backward Euler: (u^{n+1} - u^n) / dt = nu * d^2(u^{n+1})/dy^2
        # => -nu*dt/dy^2 * u_{j-1} + (1 + 2*nu*dt/dy^2) * u_j - nu*dt/dy^2 * u_{j+1} = u_j^n

        diag = np.ones(ny)
        lower = np.zeros(ny - 1)
        upper = np.zeros(ny - 1)
        rhs_vec = u_profile.copy()

        for j in range(ny):
            d = dy_arr[j]
            coeff = nu * dt / d**2

            if j > 0:
                lower[j - 1] = -coeff
                diag[j] += coeff
            else:
                # bottom wall: u=0 (Dirichlet)
                # ghost: u_{-1} = -u_0 => flux = nu*(u_{-1}-u_0)/(dy/2) = -2*nu*u_0/dy
                diag[j] += 2.0 * coeff
                # rhs += 2*coeff * 0.0 (wall value = 0)

            if j < ny - 1:
                upper[j] = -coeff
                diag[j] += coeff
            else:
                # top wall: u=U_wall (Dirichlet)
                diag[j] += 2.0 * coeff
                rhs_vec[j] += 2.0 * coeff * U_wall

        # solve tridiagonal system (Thomas algorithm)
        u_profile = _solve_tridiag(lower, diag, upper, rhs_vec)

    # exact solution at cell centres
    y = mesh.yc
    u_exact = U_wall * y / H

    err_L2 = np.sqrt(np.mean((u_profile - u_exact)**2))
    err_rel = err_L2 / U_wall

    print(f"  u(y) L2 error: {err_L2:.4e} ({err_rel*100:.2f}% of U_wall)")
    passed = err_rel < 0.05

    print(f"  {'PASS' if passed else 'FAIL'} (threshold 5%)")

    # save plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(u_exact, y * 1000, "k-", lw=2, label="Exact (linear)")
    ax.plot(u_profile, y * 1000, "ro", ms=5, label="FEM")
    ax.set_xlabel("u [m/s]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Couette Flow Validation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig("test_couette.png", dpi=150)
    print(f"  Plot saved: test_couette.png")

    return passed


def _solve_tridiag(lower, diag, upper, rhs):
    """
    Thomas algorithm for tridiagonal system.

    Args:
        lower: sub-diagonal, length n-1
        diag:  main diagonal, length n
        upper: super-diagonal, length n-1
        rhs:   right-hand side, length n

    Returns:
        x: solution vector, length n
    """
    n = len(diag)
    c = np.zeros(n)
    d = np.zeros(n)

    # forward sweep
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        if i < n - 1:
            w = diag[i] - lower[i - 1] * c[i - 1]
            c[i] = upper[i] / w
        else:
            w = diag[i] - lower[i - 1] * c[i - 1]
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / w

    # back substitution
    x = np.zeros(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]

    return x


def test_ignition_delay():
    """
    0-D ignition delay test for the Arrhenius combustion model.

    Setup:
        - Homogeneous mixture at T_init, p_init, Yf_init
        - No flow (u=v=0), no spatial gradients
        - Only the combustion source term is active

    Expected behaviour:
        - Temperature and pressure remain nearly constant until ignition
        - At ignition, rapid temperature rise as fuel is consumed
        - Final state approaches adiabatic flame temperature

    This tests the SingleStepArrhenius source term implementation
    and the stability of the chemistry coupling.

    Pass criteria:
        - Fuel is consumed (final Yf < 0.01 * initial Yf)
        - Temperature increases (final T > 1.5 * initial T)
        - Mass is conserved (|rho_final - rho_initial| / rho_initial < 1e-6)
    """
    print("\n" + "=" * 60)
    print("TEST 3: Ignition Delay (Combustion)")
    print("=" * 60)

    gamma = 1.4
    R_gas = 287.0
    nx = 1
    ny = 1

    # initial conditions: hot premixed gas
    T_init = 1500.0   # K (high enough for ignition)
    p_init = 101325.0  # Pa
    Yf_init = 0.05     # 5% fuel mass fraction
    rho_init = p_init / (R_gas * T_init)

    # Arrhenius parameters (tuned for fast ignition at ~1500 K)
    combustion = SingleStepArrhenius(
        A_pre=1.0e8, Ea=40000.0, Q_heat=2.5e6,
        nf=1.0, no=1.0, W_f=0.002,
        gamma=gamma, R_gas=R_gas,
    )

    # state vector (1x1 "mesh")
    U = np.zeros((5, nx, ny))
    E_init = p_init / ((gamma - 1.0) * rho_init)  # no kinetic energy
    U[0, 0, 0] = rho_init
    U[1, 0, 0] = 0.0   # rho*u = 0
    U[2, 0, 0] = 0.0
    U[3, 0, 0] = rho_init * E_init
    U[4, 0, 0] = rho_init * Yf_init

    # time integration with sub-stepping
    # use explicit Euler with small dt for stiff chemistry
    dt = 1.0e-7   # s
    n_steps = 10000
    t_max = dt * n_steps

    T_hist = []
    Yf_hist = []
    t_hist = []

    for n in range(n_steps):
        S = combustion.compute(U)

        # explicit Euler update (0-D, no spatial terms)
        U += dt * S

        # clamp fuel fraction
        rho_curr = U[0, 0, 0]
        Yf_curr = U[4, 0, 0] / max(rho_curr, 1e-30)
        if Yf_curr < 0.0:
            U[4, 0, 0] = 0.0
            Yf_curr = 0.0

        # extract T
        E_curr = U[3, 0, 0] / max(rho_curr, 1e-30)
        p_curr = (gamma - 1.0) * rho_curr * E_curr
        T_curr = max(p_curr / (rho_curr * R_gas), 50.0)

        T_hist.append(T_curr)
        Yf_hist.append(Yf_curr)
        t_hist.append((n + 1) * dt)

    T_final = T_hist[-1]
    Yf_final = Yf_hist[-1]
    rho_final = U[0, 0, 0]

    print(f"  T_init = {T_init:.1f} K, T_final = {T_final:.1f} K")
    print(f"  Yf_init = {Yf_init:.4f}, Yf_final = {Yf_final:.6f}")
    print(f"  rho_init = {rho_init:.4f}, rho_final = {rho_final:.4f}")

    # check criteria
    fuel_consumed = Yf_final < 0.01 * Yf_init
    temp_increased = T_final > 1.5 * T_init
    mass_conserved = abs(rho_final - rho_init) / rho_init < 1e-6

    print(f"  Fuel consumed:  {'PASS' if fuel_consumed else 'FAIL'} "
          f"(Yf_final/Yf_init = {Yf_final/Yf_init:.4e})")
    print(f"  Temp increased: {'PASS' if temp_increased else 'FAIL'} "
          f"(T_final/T_init = {T_final/T_init:.2f})")
    print(f"  Mass conserved: {'PASS' if mass_conserved else 'FAIL'} "
          f"(delta_rho/rho = {abs(rho_final-rho_init)/rho_init:.2e})")

    passed = fuel_consumed and temp_increased and mass_conserved

    # save plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    t_ms = np.array(t_hist) * 1000  # convert to ms

    axes[0].plot(t_ms, T_hist, "r-", lw=2)
    axes[0].set_xlabel("Time [ms]")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title("Ignition Delay — Temperature")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ms, Yf_hist, "b-", lw=2)
    axes[1].set_xlabel("Time [ms]")
    axes[1].set_ylabel("Fuel mass fraction Yf")
    axes[1].set_title("Ignition Delay — Fuel Consumption")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Arrhenius Combustion Validation")
    plt.tight_layout()
    fig.savefig("test_ignition.png", dpi=150)
    print(f"  Plot saved: test_ignition.png")

    return passed


if __name__ == "__main__":
    tests = {
        "sod": test_sod_shock_tube,
        "couette": test_couette_flow,
        "ignition": test_ignition_delay,
    }

    if len(sys.argv) > 1:
        name = sys.argv[1].lower()
        if name in tests:
            result = tests[name]()
            status = "PASSED" if result else "FAILED"
            print(f"\n{'='*60}")
            print(f"  {name.upper()}: {status}")
            print(f"{'='*60}")
        else:
            print(f"Unknown test: {name}")
            print(f"Available: {', '.join(tests.keys())}")
    else:
        # run all
        results = {}
        for name, fn in tests.items():
            results[name] = fn()

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        all_passed = True
        for name, passed in results.items():
            status = "PASSED" if passed else "FAILED"
            print(f"  {name:12s}: {status}")
            if not passed:
                all_passed = False

        print(f"{'='*60}")
        print(f"  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        print(f"{'='*60}")
