# Scramjet Unstart and Wall Motion Study

## Overview

A **scramjet** (supersonic combustion ramjet) is an air-breathing engine for
sustained flight above roughly Mach 5. It carries no rotating compressor: at
hypersonic speed the inlet geometry alone compresses and decelerates the
captured air, which is then mixed with fuel and burned in a still-supersonic
stream to produce thrust. Because the inlet performs the compression that a
turbomachine would otherwise supply, the engine operates only while the inlet is
*started*, delivering a steady, high-pressure flow to the combustor.

Two phenomena threaten that condition. **Unstart** occurs when excess internal
contraction or combustor back-pressure forces the shock train out of the duct:
mass capture collapses, and inlet pressure recovery, thrust, and flame-holding
collapse with it. **Aeroelastic deformation** arises because the compression
surfaces are thin, hot, and structurally loaded, so they deflect and oscillate
in flight and continuously perturb the geometry that fixes the compression.

This project studies both phenomena in a hypersonic scramjet duct, combining a
compressible-flow model of the inlet/isolator/combustor/nozzle with the
experimental geometries of Schram and Narayanaswamy [1] (streamtraced/Busemann
inlet unstart at angle of attack) and Bhattrai et al. [2] (aeroelastic response
of a compression-ramp intake), as well as general parametrized geometries: sweeping 
the static wall position spans a family of inlet contractions, and varying the 
oscillatory wall motion spans a family of unsteady regimes. High-fidelity, 
body-fitted simulations are run in OpenFOAM; the model in this repository resolves 
the quasi-1D compressible dynamics and the geometry and forcing parameter space 
that frame those simulations.

## Research direction

Two coupled questions drive the work.

1. **Inlet-contraction sensitivity and the approach to unstart.** How does a
   change in throat area or wall position shift the duct performance (exit Mach,
   pressure recovery, mass capture), and how much contraction can the duct
   sustain before the started solution is lost? This is the reduced-order
   counterpart of the inlet-unstart behavior measured for streamtraced and
   Busemann-type intakes at non-zero angle of attack [1], where unstart is
   governed by shock-train motion, isolator margin, and large-scale separation
   [3].

2. **Aeroelastic-intake response to a moving wall.** How does a small, periodic
   oscillation of the throat area drive the unsteady duct response, how large is
   that response, and how far does it lag the wall motion? This
   represents the aeroelastic deformation of a hypersonic intake ramp measured
   experimentally in [2], where ramp deflection correlated directly with
   isolator total-pressure loss, and connects to the broader hypersonic
   aerothermoelasticity literature [4].

Mapping a wide parameter space with the full model would be expensive, so two
reduced representations carry most of the load: a proper-orthogonal-decomposition
(POD) model [5], and a multi-fidelity Bayesian optimizer with a response
surrogate [6]. These concentrate the costly, body-fitted OpenFOAM cases on the
configurations the study flags as significant.

> **Geometries of interest.** The streamtraced/Busemann inlet of [1] and the
> cantilevered compression-ramp intake of [2] are the initial reference
> geometries. They enter the model through their effective-area signatures: a
> static contraction parameter `q` and a localized breathing mode
> `eps * phi(x)`. These signatures are general. Sweeping `q` defines a family of
> inlet wall positions, and varying the breathing amplitude, frequency, and
> phase defines a family of wall-motion regimes, so the study identifies further
> geometries and forcing conditions of interest beyond the two reference cases.
> Any selected configuration, whether a reference geometry or one surfaced by the
> study, is reconstructed body-fitted in OpenFOAM, for which the model exports
> the area profile, wall-contour suggestion, and freestream conditions.

![Effective-area parameterization](figures/geometry_parameterization.png)

*The two geometry controls in the effective-area model. (a) a static throat-area
parameter `q`, where `q < 0` contracts the inlet toward unstart, matching the
study geometry of [1]; (b) an unsteady area-breathing mode
`eps * phi(x) * sin(2 pi f t)`, a reduced representation of intake-ramp
aeroelasticity [2]. The model uses a quasi-1D area law, not a body-fitted wall.*

## Motivation

Because the compression is set entirely by the duct geometry and the shock
system it supports, a scramjet inlet is unusually sensitive to small changes in
that geometry. A modest change in throat area, whether a design choice or an
in-flight deformation of a compression surface, can move the shock train, erode
the margin against unstart, and, past a limit, collapse the inlet [3]. The
surfaces that hold this geometry are simultaneously primary load paths and heat
sinks, so under aerothermal load they deflect and oscillate, feeding the
structural response back into the aerodynamics [2, 4]. The two effects are
coupled: contraction sets how close the inlet runs to its unstart limit, and
wall motion modulates that margin in time.

Resolving these phenomena directly requires body-fitted, turbulent, and
eventually moving-mesh CFD, which is costly enough that sweeping geometry and
forcing parameters by brute force is impractical. A reduced-order compressible
model that captures the leading-order area dynamics can map the parameter space
first, locate the regimes worth resolving, and supply consistent geometry and
run conditions to the high-fidelity solver.

## Theory

The duct flow is modeled by the two-dimensional compressible conservation laws
with a quasi-1D variable-area source, written for the conservative state
`U = [rho, rho u, rho v, rho E, rho Y_f]`:

```math
\partial_t U + \partial_x F(U) + \partial_y G(U)
= S_{\text{area}}(U) + S_{\text{chem}}(U)
+ \nabla \cdot (\mu \nabla u,\; k \nabla T,\; \rho D \nabla Y_f).
```

The area enters as a pressure-area source that redistributes axial momentum with
the local area gradient:

```math
S_{\text{area}} = -\frac{1}{A}\frac{\mathrm{d}A}{\mathrm{d}x}\,[0,\; p,\; 0,\; 0,\; 0]^\top .
```

For a supersonic stream this term reproduces the area-Mach behavior that
governs the inlet: a decreasing area (`dA/dx < 0`) decelerates and compresses
the flow, while an increasing area accelerates and expands it. The inlet
contraction therefore sets the compression delivered to the isolator, and there
is a contraction limit beyond which a started, fully supersonic solution no
longer exists and the inlet unstarts, expelling the shock train upstream [3].

The duct geometry is a three-section area law `A_base(x)` (a contracting inlet,
a slightly diverging combustor, and an expanding nozzle). The two research
controls are localized perturbations of that law about the throat, using a
Gaussian shape `phi(x)`:

```math
A(x; q) = A_{\text{base}}(x) + q\,\phi(x), \qquad
\phi(x) = \exp\!\left[-\tfrac{1}{2}\left(\tfrac{x - x_{\text{throat}}}{\sigma}\right)^{2}\right],
```

```math
A(x, t) = A_{\text{base}}(x) + \big(q_{\text{offset}} + \varepsilon \sin(2\pi f t + \psi)\big)\,\phi(x).
```

The static parameter `q` represents an effective throat-area or wall-position
change and is the axis for the contraction and unstart study. The time-periodic
mode is a reduced representation of a compression surface oscillating under
aerothermoelastic load; its response amplitude and its phase lag relative to the
wall motion quantify the aeroelastic coupling reported in [2]. Heat release,
when enabled, uses a single-step Arrhenius source
`omega = A rho^(n_f+n_o) Y_f^(n_f) Y_o^(n_o) exp(-E_a / R_u T)`; cold-flow
studies set it to zero so that the geometry effect is isolated from
chemistry-model uncertainty.

## Model and methods

The convective fluxes use an HLLC approximate Riemann solver on MUSCL-
reconstructed states with the Venkatakrishnan limiter, advanced in time by a
third-order strong-stability-preserving Runge-Kutta scheme. An optional implicit
FEM-style diffusion step for velocity, temperature, and fuel fraction is applied
by Strang operator splitting, giving second-order accuracy overall. Unsteady
cases march in physical time from a converged steady baseline and record probe
histories at the inlet, throat, combustor, and exit.

The compressible model is paired with two reduced representations:

- A **POD reduced-order model** over the geometry axis. Converged snapshots at
  several `q` (and, optionally, exit area, nozzle length, and combustor length)
  are assembled into a snapshot matrix, truncated by SVD at a cumulative-energy
  threshold, and interpolated in coefficient space. This lowers the per-
  evaluation cost by roughly four orders of magnitude [5].
- A **multi-fidelity Bayesian optimizer and response surrogate**. A Gaussian
  process with an ARD-RBF kernel and Expected-Improvement acquisition searches
  the geometry space, routing most evaluations through the reduced-order model
  and falling back to the full solver when the predictive uncertainty is high
  [6]. The unsteady design of experiments over `(q_offset, eps, f, phase)` is
  summarized by a separate scalar response surrogate, with ridge-regression and
  inverse-distance fallbacks when samples are scarce.

Candidate configurations are ranked by a transparent weighted score over
normalized quantities of interest, and the selected geometries are exported for
OpenFOAM and FUN3D as an area profile, a suggested wall contour, freestream and
derived stagnation conditions, shared quantity-of-interest definitions, and mesh
and turbulence notes.

**Scope.** The model resolves inviscid, quasi-1D compressible duct dynamics with
the variable-area source, plus optional molecular diffusion and a single-step
heat-release source. Effects that require a body-fitted, turbulent, or
moving-mesh treatment, namely resolved boundary layers and shock/boundary-layer
interaction, turbulent separation, wall heat transfer, true wall-motion pressure
work and dynamic-mesh conservation, angle-of-attack three-dimensionality, and
finite-rate chemistry, are computed in the high-fidelity OpenFOAM cases. The
reduced-order results are read as trends and as a guide to where those cases are
needed.

## Validation

Two canonical cases pin down the numerics, both non-reacting with exact
references.

| Test | Measured error | Threshold | Result |
|---|---|---|---|
| Sod shock tube, density (L1) | 0.0053 | 0.020 | 3.8x under |
| Sod shock tube, velocity (L1) | 0.0088 | 0.035 | 4.0x under |
| Sod shock tube, pressure (L1) | 0.0039 | 0.015 | 3.8x under |
| Couette flow, `u(y)` (L2) | 8.7e-15 | 5% of U_wall | machine epsilon |
| 0-D ignition delay, fuel depletion and mass conservation | PASS | n/a | heat-release source only |

![Sod shock tube](test_sod.png)
*Sod shock tube against the exact Riemann solution. This exercises the full
HLLC, MUSCL, and RK3-SSP stack in the `ny = 1` limit.*

![Couette flow](test_couette.png)
*Couette flow. The implicit FEM-style diffusion operator reproduces the analytic
linear profile to machine precision.*

The 0-D ignition test confirms fuel depletion, mass conservation, and thermal
coupling of the Arrhenius source. It uses an uncapped heat release, so its
product temperature is not a physical flame temperature; combustion is kept off
in the cold-flow studies.

### Representative baseline run (Mach 6, 25 km, inviscid with variable area)

Freestream: T_inf = 216.65 K, p_inf = 2487 Pa, rho_inf = 0.0400 kg/m^3,
u_inf = 1770 m/s. Mesh 80 by 16, 1500 steps at CFL 0.4.

| Quantity | Value |
|---|---|
| Exit Mach (area-averaged) | 5.59 |
| Peak Mach (nozzle exit) | 7.18 |
| Pressure range | 1771 to 3187 Pa |
| Temperature range | 157.6 to 265.0 K |
| Thrust and Isp (per unit depth) | 6016 N/m, 86.6 s |

![Mach field](verification/verify_mach.png)
![Centerline profiles](verification/verify_centerline.png)

The flow decelerates through the converging inlet and reaccelerates through the
diverging combustor and nozzle, consistent with supersonic area-Mach behavior,
and no cell carries negative density, pressure, or temperature.

### Reduced-order model and multi-fidelity optimization

A snapshot POD model trained on 9 full-solver evaluations and truncated at 99.9%
cumulative energy (8 modes) reproduces held-out quantities of interest to a few
percent (exit Mach 2.2%, thrust and pressure recovery about 6%) at roughly
2.8e4 times lower per-evaluation cost, which gives an 80% reduction in wall time
at the 80%-reduced-order-fraction operating point.

![POD energy spectrum](verification/verify_pod_energy.png)
*POD singular-value spectrum and cumulative energy. The truncation at 8 modes
(99.9% energy) sets the size of the reduced model.*

A three-variable Bayesian optimizer (Gaussian process with Expected Improvement,
reduced-order and full-solver routing) explores an exit-area, nozzle-length, and
combustor-length space in about 44 s of wall time, increasing the objective by
driving exit area toward its bound.

![BO convergence](verification/verify_bo_convergence.png)
*Multi-fidelity Bayesian-optimization convergence. Blue points are full-solver
evaluations, red points are reduced-order evaluations.*

The unsteady design of experiments feeds a scalar response surrogate over the
post-transient quantities of interest. Its held-out predictions track the
computed values closely, which supports using the surrogate to rank candidate
configurations before committing high-fidelity resources.

![Surrogate parity](figures/surrogate_parity.png)
*Response-surrogate predicted against actual mean pressure recovery on held-out
cases; the dashed line is perfect agreement. Coarse-mesh demonstration.*

Static `q`-sweeps, the `(q_offset, eps, f)` design-of-experiments frequency and
amplitude responses, and candidate rankings are written under `runs/` when the
`experiments/` scripts are run.

## Repository contents

| File | Description |
|---|---|
| `mesh.py` | Structured 2-D mesh; `GeometryProfile` (three-section area law); `LocalizedAreaPerturbation`, `PerturbedGeometryProfile`, `TimeDependentPerturbedGeometryProfile`. |
| `fvm.py` | `StateVector`, boundary conditions, HLLC solver, MUSCL and Venkatakrishnan reconstruction, RK3-SSP integrator. |
| `physics.py` | Sutherland transport, quasi-1D area source, single-step Arrhenius, implicit FEM diffusion. |
| `solver.py` | Configuration dataclasses and the Strang-split `Solver` with physical-time stepping. |
| `rom.py` | Snapshot collection, `PODBasis` (SVD truncation), `ReducedSolver`, `ROMEvaluator`. |
| `optimization.py` | `DesignSpace`, `GPSurrogate`, Expected-Improvement acquisition, multi-fidelity `BayesianOptimizer`. |
| `response_metrics.py` | Amplitude, phase, and transient extraction with explicit undersampling warnings. |
| `diagnostics.py` | Scalar-boundedness and heat-release diagnostics. |
| `experiments/` | Static sweep, unsteady run, design of experiments, surrogate and reduced-order builders, candidate ranking, OpenFOAM and FUN3D exporter. |
| `figures/make_geometry_figure.py` | Regenerates the parameterization figure above. |
| `tests.py` | Sod, Couette, ignition-delay, geometry, and synthetic-response-metric tests. |
| `verification/verify_all.py` | End-to-end harness writing `verify_results.json` and `verify_*.png`. |

## Quick start

```bash
pip install numpy scipy matplotlib numba    # numba optional (JIT for hot loops)
python3 tests.py                            # validation suite (about 10 s), writes test_*.png
python3 verification/verify_all.py          # baseline, reduced-order, optimization (about 3 min)
python3 figures/make_geometry_figure.py     # regenerate the geometry figure
```

Parameter study (writes to `runs/`, which is git-ignored):

```bash
python3 experiments/run_static_wall_sweep.py        --output-root runs/static_demo
python3 experiments/run_parametric_unsteady_doe.py  --output-root runs/doe_demo
python3 experiments/build_unsteady_response_surrogate.py --doe-root runs/doe_demo --output-root runs/surrogate_demo
python3 experiments/build_steady_q_rom.py           --sweep-root runs/static_demo --output-root runs/rom_demo
python3 experiments/rank_candidate_cases.py         --doe-root runs/doe_demo --output-root runs/ranked_demo --top-k 5
python3 experiments/export_high_fidelity_scaffold.py --sweep-root runs/static_demo \
    --selected-cases runs/ranked_demo/selected_cases.json --output-root runs/export_demo
```

The exporter writes geometry, area, and wall-contour files, freestream and
stagnation conditions, shared quantity-of-interest definitions, and mesh and
turbulence notes. It does not generate a runnable OpenFOAM or FUN3D case;
body-fitted meshing, boundary tagging, turbulence-model selection, and, for
moving-wall studies, dynamic-mesh setup are carried out in the high-fidelity
solver.

## Reproducing the numbers

```bash
python3 tests.py                       # validation tests, writes test_*.png
python3 verification/verify_all.py     # writes verification/verify_results.json (all cited numbers)
```

## References

> Bibliographic details are auto-compiled below; confirm in a reference manager
> before any publication use.

[1] M. Schram and V. Narayanaswamy, "Unstart dynamics of a hypersonic
streamtraced (Busemann-derived) inlet at non-zero angles of attack,"
Experiments in Fluids, Vol. 67, 2026, Art. 64. doi:10.1007/s00348-026-04215-0.
Time-resolved unstart at angles of attack of -5, 0, and +3 degrees,
distinguishing "weak" and "strong" unstart responses by shock-foot and
shock-train tracking. (See also the same group's companion studies:
"High-Bandwidth Pressure Field Imaging of Stream-Traced Inlet Unstart Dynamics,"
AIAA Journal, doi:10.2514/1.J064324; and "Unstart Sensitivity of Hypersonic
Streamtraced Inlets During Angle-of-Attack Operation," AIAA Journal,
doi:10.2514/1.J064532.)

[2] S. Bhattrai, L. P. McQuellin, G. M. D. Currao, A. J. Neely, and D. R.
Buttsworth, "Experimental Study of Aeroelastic Response and Performance of a
Hypersonic Intake Ramp," Journal of Propulsion and Power, Vol. 38, No. 1, 2022.
doi:10.2514/1.B38348.

[3] Review of inlet and isolator unstart and shock-train dynamics: "A review of
the shock-dominated flow in a hypersonic inlet/isolator," Progress in Aerospace
Sciences, 2023. doi:10.1016/j.paerosci.2023.100952.

[4] J. J. McNamara and P. P. Friedmann, "Aeroelastic and Aerothermoelastic
Analysis in Hypersonic Flow: Past, Present, and Future," AIAA Journal, Vol. 49,
No. 6, 2011, pp. 1089-1122. doi:10.2514/1.J050882.

[5] Proper orthogonal decomposition for reduced-order modeling: G. Berkooz, P.
Holmes, and J. L. Lumley, "The Proper Orthogonal Decomposition in the Analysis
of Turbulent Flows," Annual Review of Fluid Mechanics, Vol. 25, 1993, pp.
539-575; K. C. Hall, J. P. Thomas, and E. H. Dowell, "Proper Orthogonal
Decomposition Technique for Transonic Unsteady Aerodynamic Flows," AIAA Journal,
Vol. 38, No. 10, 2000, doi:10.2514/2.867; and, for hypersonic inlets, "Reduced-
Order Modeling of Hypersonic Inlet Flowfield Based on Autoencoder and Proper
Orthogonal Decomposition," Journal of Spacecraft and Rockets,
doi:10.2514/1.A36194.

[6] Multi-fidelity surrogate optimization: D. R. Jones, M. Schonlau, and W. J.
Welch, "Efficient Global Optimization of Expensive Black-Box Functions," Journal
of Global Optimization, Vol. 13, 1998, pp. 455-492; A. I. J. Forrester, A.
Sobester, and A. J. Keane, "Multi-fidelity Optimization via Surrogate
Modelling," Proceedings of the Royal Society A, Vol. 463, 2007, pp. 3251-3269;
and the review "Multi-fidelity Bayesian Optimization: A Review,"
arXiv:2311.13050, 2023.
