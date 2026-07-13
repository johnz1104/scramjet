# Eleven-concern remediation and forward research assessment

Date: 2026-07-13
Scope: quasi-1D scramjet effective-area repository, Paper-1 cold-flow workflow

## Executive assessment

All eleven review concerns were reproduced against the code and have now been
addressed in the implementation, tests, workflow gates, or documented research
scope. The most consequential correction is the moving-wall energy source:
the non-area-weighted energy equation requires
`-(A_t/A)(rhoE+p)`, not `-(A_t/A)rhoE`. All pre-schema-v2 unsteady artifacts are
therefore excluded from new analysis.

The repaired repository is suitable for its intended role as a low-fidelity,
trend-screening and case-selection layer. It is not yet evidence of a closed
multi-fidelity result: Config-A dimensions remain placeholders, the default
demo uses the generic area law, and no emitted OpenFOAM case has been meshed,
run, and postprocessed. Those three facts define the critical path to Paper 1.

The 2026-07-13 forward pass completed the remaining small repo work: explicit
reduced-frequency coordinates, log/complex-response surrogates with circular
validation, surrogate-aware ranking audits, the quantitative Culick--Rogers
reference, per-case DoE run status, a deterministic hysteresis driver, and a
tighter mass-balance regression guard. None of these substitutes for the first
Config-A/OpenFOAM anchor.

The paper scope remains cold flow. Combustion units and testing are fixed, but
combustion is parked until the first low/high-fidelity loop closes; that scope
decision should be revisited after Phase 4 rather than allowed to distract the
current paper.

## Judgement calls and disagreements with the proposed remediation

Six adjustments were made rather than following the proposed mechanics
literally:

1. The integrated breathing thermodynamics check uses the spatially uniform
   moving-control-volume ODE, plus a separate production-FVM discrete-source
   check. A full duct compression test mixes the target thermodynamics with
   boundary waves and flux discretization. The isolated ODE more cleanly
   distinguishes `p proportional to A^-gamma` from the legacy
   `p proportional to A^-1` law; the FVM check proves the production path uses
   the same source.
2. ROM prescreening is not described as saving cost relative to standard BO.
   It still confirms one full case per iteration and adds ROM-training cost.
   The report gives two comparisons: overhead relative to standard
   one-candidate BO, and savings relative to evaluating every top-m shortlisted
   candidate at full fidelity.
3. The OpenFOAM handoff is called an executable comparison design, not a closed
   loop. It emits sampling and QoI code, but compatibility and numerical parity
   cannot be claimed before a case actually runs in the pinned image.
4. Surrogate-predicted lag does not replace a supported lag measured in an
   evaluated DoE case during ranking. The complex surrogate is carried as an
   audit and optional finite-value gate; measured evidence remains stronger.
5. The Culick--Rogers curve is not called a strict exit-pressure validation.
   Their transfer input is local post-shock acoustic pressure, whereas the
   numerical input is imposed at the end of a finite subsonic duct. The emitted
   hybrid curve isolates the published relaxation from that propagation gap.
6. Reduced frequency is not hardcoded to an unnamed duct scale, and the
   hysteresis output is not called physical validation. Both dimensional
   references are recorded/overridable; the staircase reports numerical path
   dependence under the prescribed-inlet limitation.

## Implemented theory

### Moving variable-area conservation

For `U = [rho, rho u, rho v, rho E, rho Yf]`, the area-weighted equations are

```text
d(AU)/dt + d(AF)/dx + A dG/dy
  = p A_x [0,1,0,0,0]^T
    - A_t [0,0,0,p,0]^T + A S_other.
```

After division by `A`, the complete moving-volume contribution is

```text
-(A_t/A) [rho, rho u, rho v, rho E + p, rho Yf]^T.
```

The pressure-work term follows directly from the moving-control-volume energy
balance. For a uniform static calorically perfect gas, mass and energy give

```text
rho A = constant,
p A^gamma = constant,
T A^(gamma-1) = constant.
```

The new breathing test closes all three invariants at approximately machine
precision. Enabling `legacy_breathing_energy` makes the same test fail with an
8.54% pressure/temperature error for the chosen compression. Static geometries
are bitwise identical across the corrected and legacy paths because `A_t=0`.

Although `p` is only a few percent of `rhoE+p` in a Mach-6 total-energy
budget, that ratio understates its thermodynamic importance: for a static gas
the missing term changes the pressure-rate coefficient from 1 to `gamma`, a
40% error for `gamma=1.4`. Post-shock/subsonic regions have both a larger
energy-source fraction and direct control over shock motion.

### Dynamic diffusion and chemistry

The implicit scalar operator now solves

```text
m d(phi)/dt = div(Gamma grad(phi))
```

with `(m,Gamma) = (rho,mu)` for velocity, `(rho cp,k)` for temperature,
and `(rho,rho D)` for fuel fraction. This removes the former extra density
division. Transient Couette start-up agrees with its Fourier series to
`5.65e-4` relative L2 error, and doubling density at fixed dynamic viscosity
exactly doubles the approach timescale in the discrete comparison.

The Arrhenius rate is explicitly molar. Energy is
`Q_heat W_f omega_dot`, fuel is `-W_f omega_dot`, and depletion limiting uses
the actual integration step. The isolated ignition test consumes fuel without
test-side clipping and reaches 1674.1 K versus the 1674.2 K caloric balance;
the pre-fix path reached about 88,600 K because it released 500 times too much
energy for `W_f=0.002 kg/mol`.

### POD identity and adaptive sampling

The reduced model is now described exactly as implemented:

1. form a POD basis from converged parametric snapshots;
2. inverse-distance interpolate modal coefficients;
3. reconstruct the conservative state;
4. derive TPR, shock, exit, mass-balance, and legacy QoIs from that state.

Direct IDW interpolation of training QoIs is retained in `qoi_idw` as a
comparison baseline. No Galerkin, DEIM, or online reduced ODE is claimed.
Queries must have exactly the training parameter keys.

The adaptive-sampling GP contains full-solver observations only. Expected
Improvement ranks a candidate pool, the POD state screens the top `m`, and one
candidate is fully evaluated. Best tracking is over full results, so the
returned optimum is verified by construction.

### Response estimator and quality model

Forcing, QoIs, and probes are each fitted against their own timestamps after a
physical-time transient cut. The basis is `[sin(omega t), cos(omega t), 1,
t-centered]`. A positive reported lag means a delayed response:

```text
lag = wrap(forcing phase - response phase).
```

Amplitude and phase share the minimum-cycle, minimum-sample,
samples-per-cycle, flat-forcing, and flat-response gates. Each fit records R2,
residual RMS, drift fraction, and signal-to-residual ratio. Unsupported raw
amplitudes remain diagnostic only and do not enter surrogate training.
Multiple design-phase values are encoded as sine/cosine features.

### Reduced-frequency and complex-response surrogate

The post-remediation follow-up now records

```text
k = 2*pi*f*L_ref/u_ref
```

in every new DoE design row, case summary, and manifest. `L_ref` and `u_ref`
are recorded, not inferred later. The demo default is full duct length and
freestream velocity; CLI overrides are required when Config A is calibrated so
the ramp/motion length can be the intended scale. Earlier schema-v2 artifacts
remain readable through a documented `frequency_hz` fallback; no pre-v2
artifact is admitted.

Phase is not treated as an ordinary scalar target. For each supported response
the surrogate fits

```text
H = amplitude * exp(-i*positive_lag)
```

through its real and imaginary components. This makes phase wrapping explicit
and makes the zero-response limit well defined. Scalar amplitude models use a
log10 target; dense response-surface output enforces exactly zero amplitude and
undefined phase at `epsilon=0`. Validation reconstructs amplitude and lag and
reports wrapped circular MAE/RMSE. Each target also records an exploratory
absolute-Spearman feature association; it is not described as causal
sensitivity. Unsupported lags never enter complex-response training.

Ranking keeps the stronger evidence source authoritative: a supported lag
measured in a completed DoE run controls eligibility and scoring. Optional LOO
surrogate lag is carried into the selection audit and may be required to be
finite, but it never replaces the measured value. Every new DoE case also
writes `run_status.json` with requested/achieved time, cycles, baseline status,
and the underlying solver completion record.

### Deterministic H5 diagnostic

`experiments/run_hysteresis_sweep.py` implements the planned up/down
back-pressure staircase with state carried continuously between levels. A
captured shock has a finite raw-RHS floor in the shock-capturing discretization,
so per-level completion uses three consecutive classification/shock-position/
TPR checks after at least three flow-through times; the residual is still
recorded. Matching up/down levels are compared at a two-cell shock-position
tolerance and a 0.01 TPR tolerance. Its output deliberately says **numerical
path dependence**, not physical hysteresis: the prescribed inlet still cannot
model spillage or changing capture.

### QoI and unstart observability

All transverse averages use `dy`, including mass-flux TPR weighting on
stretched grids. The shared exit convention is the last physical Python
column, `i=nx-2`. The QoI registry now includes:

- `tpr` and `shock_x` as primary experiment-facing quantities;
- `mdot_prescribed`, `mdot_exit`, and signed `mass_defect`;
- minimum centerline Mach through the contraction;
- shock-at-inlet and started/unstarted trend classification.

The inlet value is deliberately called prescribed: a supersonic Dirichlet
boundary cannot model spillage or mass-capture collapse. This model can flag
shock expulsion and TPR collapse, not reproduce the external capture process.

For context at Config B's transcribed `M=3.7`, `gamma=1.4`, a normal shock
gives `M2=0.44395`. Choking the post-shock subsonic stream gives a conservative
Kantrowitz capture-to-throat ratio of 1.4638, whereas the isentropic
supersonic-to-sonic area ratio is 8.1691. The nominal 3:1 geometry lies between
these bounds: above the conservative normal-shock swallowing limit but below
the isentropic limit. That is a useful regime classification, not a start
prediction for a viscous three-dimensional streamtraced inlet. The derivation
matches the normal-shock/sonic-throat construction in NASA's discussion of the
Kantrowitz maximum starting contraction ratio:
<https://ntrs.nasa.gov/api/citations/20100001729/downloads/20100001729.pdf>.

## Concern-by-concern disposition

| # | Disposition | Research consequence |
|---|---|---|
| 1 | Dimensionless residual checks, two-check steady gate, per-offset baselines, physical-time completion, and `incomplete` status added. | No exception-only “ok”; epsilon-zero transient amplitudes collapse. |
| 2 | Added wall-pressure work in both production and reference area sources; audit-only legacy switch. | All pre-v2 unsteady response results invalidated and regenerated. |
| 3 | Removed Galerkin/mixed-GP claims; state-derived POD QoIs, IDW comparison, and full-only GP prescreen implemented. | Defensible adaptive case selection; full-verified optimum. |
| 4 | Shared exit/dy conventions; prescribed/exit mass flow, mass defect, shock-at-inlet, and start proxies added; ranking uses TPR and measurable response. | Unstart is observable by shock/TPR trends, not capture collapse. |
| 5 | Corrected molar heat-release conversion and dt limiter; ignition test closes energy. | Capability fixed but remains outside Paper 1. |
| 6 | Correct dynamic diffusion coefficients and transient mass terms; new analytic tests. | Couette verification now tests transient physics, not only its steady shape. |
| 7 | Future-safe config cloning, outlet-field preservation, geometry-type preservation, wrapping of tabulated bases, and unknown-key rejection. | Config-A/Busemann and back-pressure lineages no longer silently mutate. |
| 8 | Own-time fits, positive-delay convention, drift/quality model, and amplitude gating. | Only supported response metrics enter surrogates; phase sign changed versus old artifacts. |
| 9 | Test exits fixed; assertion-bearing verification, dependencies, packaging, and GitHub Actions added. | Automation can fail reliably. |
| 10 | Tabulated/time-dependent mean export, strict q matching, `pipefail`, sampling objects, and executable QoI bridge added. | Handoff is apples-to-apples by design, but still unexecuted. |
| 11 | M3.7 preset transcription, Config-A CLI wiring/warning, dependency provenance, and schema-v2 gates added. | New analyses reject pre-fix artifacts; direct PDF/dimension calibration is still required. |

Nuances from the original review are retained: phase had partial gating before
the fix, provenance already included git metadata, `dy` omission was harmless
on uniform grids, and `nx-1`/`nx-2` could agree under a supersonic outlet. The
correct remediation nevertheless makes each convention explicit and robust to
the cases where those nuances stop being harmless.

## Quantified reassessment

### Completion and baseline correction

The regenerated demo has five of five static cases converged and 18 of 18
unsteady cases at exactly three requested cycles. All three unique
`q_offset` baselines converged independently. Across six epsilon-zero cases,
raw exit-Mach amplitudes range from `9.5e-16` to `7.7e-10` and are correctly
marked unsupported. Nonzero-forcing cases are supported. This directly removes
the former false-amplitude mechanism.

### Breathing energy old versus corrected

For the audit case `q_offset=0`, `epsilon=0.001 m2`, `f=1500 Hz`, three cycles,
30 by 6 cells:

| QoI | Corrected | Legacy | Corrected change |
|---|---:|---:|---:|
| Exit-Mach amplitude | 0.004094 | 0.018215 | -77.5% |
| Exit-Mach lag | 0.6845 rad | 0.2764 rad | +0.4081 rad |
| TPR amplitude | 0.002301 | 0.012971 | -82.3% |
| TPR lag | -0.1688 rad | 0.1110 rad | -0.2798 rad |
| Static-pressure-ratio amplitude | 0.001873 | 0.003182 | -41.1% |
| Static-pressure-ratio lag | -1.8270 rad | -2.4234 rad | +0.5964 rad |

These large response changes are a warning, not a publication result. The run
is coarse and only 2.25 post-discard cycles. It shows that the missing term
cannot be dismissed from wall-motion studies even when its fraction of total
energy is small. A research-grade reassessment needs Config-A geometry,
flow-through-based discard, frequency refinement, and at least 5-10 settled
cycles.

The forced-shock benchmark is invariant to this correction because `A_t=0`.
The regenerated eight-cycle gain-aligned amplitude ratio falls from 1.004 at
20 Hz to 0.171 at 400 Hz; unwrapped lag grows from 0.449 to 6.095 rad. This preserves
the low-pass/lag verification lineage while separating it from wall breathing.

The benchmark now evaluates the Culick--Rogers isentropic-flow coefficients
from their Eqs. 42--44. At the present mean shock station, the local upstream
Mach number is 2.456 and `tau=3.089e-3 s`; the published first-order relaxation
therefore predicts normalized gains from 0.932 at 20 Hz to 0.128 at 400 Hz and
positive lag from 0.370 to 1.443 rad. This is not presented as a strict
point-by-point validation against the imposed exit pressure. Their input is
the acoustic pressure immediately behind the shock, while the numerical input
is the boundary pressure after a finite post-shock duct. The overlay uses the
exact exit-pressure static gain times the published relaxation factor, calls
it a hybrid, and reports rather than passes/fails the extra propagation lag.
The resulting normalized-gain RMSE is 0.160 and the unwrapped phase-difference
RMSE is 2.41 rad; these values quantify the mismatch instead of disguising it.

The 60-cell deterministic hysteresis demo settles all 15 staircase levels. At
back-pressure factor 1.323, the up leg classifies the shock at the inlet while
the down leg retains a captured internal shock, so the driver reports
`numerical_path_dependence_detected`. Repeating the same configuration produces
byte-identical summary and assessment files. This result is a useful H5 target
for grid refinement, but it is not physical hysteresis validation because
inlet capture and spillage cannot vary in this model.

### Verification, ROM, and cost

`python3 tests.py` passes all 13 groups. The legacy breathing switch makes the
breathing group fail, as intended. `verification/verify_all.py` passes every
assertion and writes strict JSON without NaNs.

The clean-run mass-balance regression limit is now 3% (measured 2.03%), not
the earlier 8%. It is labeled as a coarse-grid gross-error guard; research
results retain a 1% target after an explicit grid-refinement study.

The verification POD uses six full snapshots and four modes. Mean held-out TPR
error is 1.21% for state-derived POD QoIs and 0.25% for direct IDW. The much
larger state-derived mass-defect and legacy thrust errors are reported rather
than hidden; Paper 1 should show POD-versus-IDW error by QoI.

The adaptive-sampling verification uses nine optimizer full solves, six
ROM-training full solves, and 20 ROM prescreen calls. Including training, the
regenerated run is 68.6% slower than standard BO at the same nine-full-solve
budget, but 36.8% cheaper than fully evaluating an equivalent top-four
shortlist. The scientific
benefit to test is improved case-selection quality per confirmed run, not a
blanket BO speedup.

The demo surrogate remains intentionally honest. Log conditioning improves
the physical-space exit-Mach-amplitude LOO RMSE from the earlier 0.255 to about
0.128 (`RMSE/std` from 1.04 to about 0.52), but the complex-response lag has
order-one circular error at only 12 supported samples. This coarse 18-case map
is pipeline validation, not a reportable surrogate. More samples--not cosmetic
hyperparameter tuning--remain the research requirement.

## Position against Paper-1 hypotheses and schedule

The current evidence supports a methods paper only after one OpenFOAM loop is
closed:

1. **Static effective-area sensitivity:** in the generic demo, TPR changes
   from 0.9815 at `q=0` to 0.8666 at `q=-0.025` (11.7% relative loss), with a
   shock detected in the contraction. This is directionally compatible with a
   sub-20% Config-A loss hypothesis, but it is not a Config-A comparison.
2. **Shock dynamics:** the forced-shock benchmark shows clear low-pass
   amplitude and increasing unwrapped lag. The Culick--Rogers first-order
   curve now provides a quantitative local-relaxation reference, with the
   finite-duct input mismatch kept explicit. This is the strongest current
   unsteady anchor.
3. **Wall-motion response:** corrected breathing dynamics produce measurable,
   quality-gated amplitude and lag, but the large old/new shift requires a new
   Config-A response map before any physical trend is quoted.
4. **Multi-fidelity value:** the software path now supports case selection and
   a shared QoI bridge; novelty must be demonstrated by error reduction after
   adding a small number of OpenFOAM anchors.

Suggested Paper-1 figure sequence:

1. Config-A/effective-area framework and fidelity ladder.
2. Calibrated Config-A area law and prescribed motion mode.
3. Canonical verification, including breathing thermodynamics and forced shock.
4. Static TPR/shock response versus measured ramp deflection.
5. POD-state versus IDW versus held-out full-model QoI errors.
6. Response estimator quality map and supported DoE region.
7. Amplitude/lag response surface with ranked CFD anchors.
8. Low/high-fidelity parity and surrogate-error reduction after anchoring.

From 2026-07-12 to a November 2026 abstract clock, the critical path is not
more low-fidelity feature work. It is Config-A dimensional calibration, one
validated OpenFOAM canonical duct, and the first static Config-A comparison.

## Forward hypotheses and expected findings

- **H1 — shock response is low-pass:** amplitude decays and unwrapped lag grows
  with reduced frequency. The forced-shock data already support this lineage.
- **H2 — static deflection primarily trades TPR for contraction:** the generic
  model predicts an approximately 12% loss at its strongest default
  contraction. Expect qualitative agreement with the UNSW less-than-20% trend,
  not amplitude agreement before viscous/body-fitted calibration.
- **H3 — wall-pressure work matters most near subsonic post-shock states:** the
  correction should increasingly alter shock response with reduced frequency
  and time spent in post-shock subsonic flow.
- **H4 — forcing amplitude modulates unstart margin:** larger motion should
  cause earlier shock approach to the inlet and a nonlinear TPR penalty, with
  the transition depending on mean `q_offset`.
- **H5 — hysteresis is a cheap discriminator:** up/down amplitude or
  back-pressure sweeps should give a clear yes/no on whether the low-fidelity
  dynamics reproduce restart/unstart path dependence. A classification target
  may be more defensible than amplitude regression near regime boundaries. The
  60-cell diagnostic gives a reproducible classification difference at one
  pressure level; the next test is whether it persists under grid and step-
  tolerance refinement. It remains a numerical-model diagnostic until a
  spillage-capable or high-fidelity comparison exists.

## Next actions

1. Calibrate Config-A ramp length, isolator length, capture height, and motion
   mode from the paper drawings/UNSW database. Keep the current CLI warning
   until those values are sourced.
2. Verify `configs/ncsu_m37.json` directly against the 2026 paper PDF; keep the
   current review-note status label until then.
3. Pin ESI/OpenFOAM v2606, run a canonical wedge and repo-matched duct, and
   execute `postprocess_qoi.py`. Record mesh and solver-version compatibility
   fixes in the generator, not hand-edited case copies.
4. Run the static Config-A deflection ladder with grid refinement and compare
   TPR/wall pressure against the experiment.
5. Run a research-grade corrected DoE: at least 5-10 settled cycles, discard
   tied to flow-through time, a calibrated reduced-frequency grid, and explicit
   convergence/quality acceptance thresholds.
6. Add the first OpenFOAM QoIs as anchors and measure error reduction versus
   full-only sampling. This is the claim Paper 1 still needs.
7. After that loop closes, decide whether combustion belongs in the next paper
   or remains a Phase-5 direction.

## Deliberately not claimed

- No variable inlet mass capture or spillage model.
- No Galerkin/DEIM or time-accurate POD solver.
- No fused multi-fidelity GP.
- No completed OpenFOAM comparison.
- No combustion result in Paper-1 scope.
- No quantitative Config-A/B statement before geometry/preset verification.

Every new experiment manifest carries schema version 2, git state, Python and
dependency versions. New DoE artifacts add reduced frequency without breaking
schema-v2 readers. Surrogate, ranking, ROM-from-sweep, and exporter stages
reject pre-v2 artifacts. Generated demo outputs live under the git-ignored
`runs/` tree; the committed evidence is the code, tests, verification JSON and
figures, and this report.
