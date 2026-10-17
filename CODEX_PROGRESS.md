# Native Taylor–Peierls repair progress

## Actual solved-interface R05 completion, 16 October 2026

- FEM calculation commit `2987e5b` preserves the actual-consumer repair and
  connects both public bilinear descriptions to a trial interface assembled by
  the current atomic-path-corridor specimen. Evidence commit `e2e84aa` retains
  direct classical and activated events plus exact fresh-process active-state
  restart.
- Runtime acceptance now requires provider construction/advance, cohesive
  assembly binding and an actual or restored trial interface. The regression
  checks residual/tangent participation, repeated mechanical feedback, bounded
  irreversible evolution, interface work, one accepted source/time update and
  byte-identical direct/restart histories. The inherited EXP-floor opening
  clock remains suppressed.
- A separate source-bound active-tip replay advances the persistent
  accommodation/source owner once per accepted interval using exactly
  `3.5e-10 s` and `1.0975208997081333e-10 s`; its emission increments and
  emission work are nonzero, with no one-second callback artifact.
- The older `r05_solved_native_moving_tip_20261016` accommodation/work values
  are retained only as invalidated historical evidence because their producer
  used one second per equilibrium callback and incorrect barriers. Unaffected
  transaction/serialization scope remains preserved.
- PF calculation/campaign commit `dd0c1953` binds all 36 fitted P9 rows to that
  actual driver/provider. The 699-row disabled campaign audit has no binding
  errors. Final applicable suites pass: PF 1008 passed with 16 explicit
  external-evidence deselections; FEM 759 passed, 2 skipped, plus 16 subtests.
- R07 and R09 resumptions remain prepared but were not launched: the latest
  measurement records about 50 MiB immediately free, 873 MiB swap free and
  28 GiB Data free, which does not admit their estimated peak plus 12 GiB
  reserve. Production, manuscript work, remote writes and scientific-data
  deletion remain disabled.

Controlling request: finish the 13 October 2026 native Taylor–Peierls repair in
the actual coupled solver.  This is a readiness qualification only: no
production campaign, manuscript edit, coefficient change, evidence deletion,
or remote action is authorized.

## Durable state

- Branch: `codex/shared-fracture-fatigue-options-v1`
- Starting archive head: `5ada17acf910024b6cea9b49ec254ebccc153774`
- Committed repair: `da8869f6` (`Repair coupled source fixed point and ledger ownership`)
- The committed repair corrects the false outer coefficient residual, restores
  configurable 64-iteration coupled allowance, performs exact rollback,
  checks accepted-state coefficients/state/source/affine-direct consistency,
  and routes the subsystem runner through the true outer ledger owner.
- The prescribed 5 GPa/16-bin runner is explicitly labeled a synthetic-kernel
  native subsystem fixture.  Its working rerun conserved the population ledger
  to `2.776e-17` and produced nonzero physical escape, gross exchange, and
  mobile-distance ledgers.

## Current uncommitted numerical-accuracy repair

- The full-attempt scale regression exposed loss of the slow source/transport
  depletion when a bounded base exponential is composed beside a
  `5.69e10 s^-1` release mode.
- The escape-row base-map correction did not repair the distributional error.
  The current implementation instead detects the unresolved scale separation
  and retries the identical augmented generator with error-controlled BDF.
- The accepted full augmented state includes populations, escape/recovery,
  gross trapping/release, and distance from that one solve.  The discarded
  affine/BDF discrepancy is reported separately; actual invariant and
  positivity corrections remain separately quantified in population units.
- A high-accuracy Radau solve is the independent oracle.  The troublesome-scale
  test covers populations, escape, gross exchange, distance, conservation,
  retry selection, and correction size.
- Focused status: 39 tests passed across coupled transport, persistent source,
  native Taylor–Peierls kinetics/release contract, and signed outer ownership.

## Exact next actions

1. Add and run a preserved-option test proving the installed reversal wrapper
   uses signed direction and a nonzero native recovery coefficient/ledger.
2. Review and commit the numerical-accuracy/reversal changes.
3. Rerun the labeled subsystem fixture under that committed head.
4. Run the actual 950 K full-PF native sequence from a fresh authenticated
   initial state: interrupted prefix, fresh-process continuation to step 64,
   uninterrupted comparison, then a distinct time/state refinement in the
   difficult active interval.  Do not reuse the provisional failed checkpoint
   as qualified-prefix evidence and do not force a topology event.
5. Run the complete test suite, update readiness records with scope-correct
   identities/counts, package the source/test patch and compact accepted states,
   then create the requested verified backup without deleting local evidence.

## Full-native diagnostic, 13 October 2026

- Commit `2253257c` completed 23 fresh full-PF accepted states through the
  normal checkpoint before an affine invariant defect of `3.714389e-09`
  exceeded the mixed population-unit limit `1.801674e-11` during step 24.
  This is a numerical conditioning failure, not a physical-domain event.
- The failed prefix remains diagnostic only at
  `integration_evidence/f1_native_taylor_peierls_qualification_2253257c`.
- The actual active-state rates are all fast (encounter about `2.43e12 s^-1`,
  Taylor release `5.69e10 s^-1`, hop about `4.75e12 s^-1`), so their ratio is
  only about 84.  The earlier scale-ratio selector therefore correctly did not
  choose BDF; the error instead came from repeated multiplication of a tiny
  base-map conservation bias over an activity near `2e13`.
- Current repair restores the invariant correction on the bounded base map
  before binary affine composition, while retaining BDF for the independently
  demonstrated extreme fast/slow separation.  Raw affine defect, accepted
  base-map correction, positivity, final invariant correction, and discarded
  affine/BDF discrepancy remain distinct diagnostics.  The 15 operator tests
  pass.  Exact next action: commit this numerical repair, replay step 24 from
  the diagnostic complete checkpoint, and only then recompute a clean qualified
  prefix under the final committed source.

The base-map replay at `c6edd790` demonstrated that the correction did not
reduce the step-24 defect, so it remains rejected as a solution.  The current
working repair instead evaluates the complete stable `2*n` mobile/retained
population block with its matrix exponential and obtains escape, recovery,
gross exchange, and distance by integrating that same generator.  It neither
removes retained state nor changes a rate.  Extreme fast/slow blocks continue
to use the independently checked BDF path.  A stiff continuous-source
partition test now explicitly exercises the stable-population method; the
focused 55-test set passes.  Next action is a committed step-24 checkpoint
replay.

The `d5b056c2` clean recomputation then reached accepted step 45 and exposed the
original source residual at step 46 with coefficient/state residuals exactly
zero.  An 8-iteration diagnostic reproduced the same result as 64 iterations:
activation rates `[1.0983862824, 1.0580603359] s^-1` remained about
`7.85e-4` below their targets while both sources were fully unblocked
(~`1.1 kPa` backstress versus ~`6.2 GPa` drive).  The cause was a false source
root bracket: the zero-activation target was used as an upper bound even though
blunting raises persistent-site multiplicity and hence raises the target after
activation is installed.  The working repair expands the bracket until the
unchanged rate equation has a true sign change or the original mechanical gate
blocks it.  A synthetic positive-feedback regression and the exact retained
step-46 replay both pass.  Next action: commit, then recompute the clean prefix
under the final source before restart/uninterrupted comparison.

## Completed native qualification, 13 October 2026

- Commit `2276745b` repairs the positive-feedback source bracket without
  changing the source law: the interval expands until the actual residual has a
  sign change or the original mechanical gate blocks.
- A clean actual 950 K full-PF run reached step 48, a fresh-process restart and
  an independent uninterrupted run reached step 64, and all 11 stored state
  arrays plus normalized `outer.json` and `kinetic.json` were exactly equal.
- The actual full-PF step-46-to-48 interval was refined from 8.4 s to 4.2 s and
  2.1 s. Kinetic and important ledger errors contracted by about 0.50,
  mechanics were preserved to `1e-12` relative L2, and population ledgers
  closed to `5.684e-14`.
- The current working checkout passed 913 tests in 200.43 s; the final focused
  source/operator/recovery set passed 56 tests in 16.07 s.
- Final remaining action is administrative only: commit the corrected readiness
  records, build one compact source/evidence package, copy it to the independent
  backup location, verify hashes, and retain all local and historical evidence.

## Recovery wiring qualification, 13 October 2026

- Commit `15b98bbf` makes the base, unit-source, and direct-source evaluations
  consume one frozen set of velocity, timestep, bin width, mobile recovery,
  and retained recovery arguments. Predictors remain ledger-pure; the accepted
  outer update owns recovery and escape once, and rejected evaluations roll
  back exactly.
- Regression coverage spans zero, mobile-only, retained-only, and combined
  recovery for empty and populated states, nonzero continuous injection,
  source/backstress feedback, affine amplitudes 0.25/1/2.5, full mass ledgers,
  and forced-failure rollback. Explicit zero recovery is exactly neutral.
- Commit `9f73dbff` supports an explicitly recorded prescribed diagnostic stress.
  The compact 950 K, 0.5 GPa, 16-bin numerical recovery overlay uses unchanged
  material laws and the error-controlled BDF route. Its fresh-process restart
  and uninterrupted terminal states are byte-identical across 28 arrays and
  scalars; recovered content is `1.335756144352698e-12`, and the maximum
  population-ledger balance error is `5.169878828456423e-26`.
- The accepted `2276745b` 5 GPa full-PF result remains the qualified
  zero-recovery specimen baseline. It was not rerun because the repaired
  coefficients are exactly zero in that configuration and regression proves
  numerical equivalence.
- Current status: 61/61 affected tests and 925/925 full-suite tests pass.
- Remaining readiness scopes are native source geography, tensor/constraint,
  FEM/CZM, branch/cavity, capability-wide restart, and long-growth analysis.

## Native source-geography qualification, 13 October 2026

- Commits `1192ee92`, `c6b98e09`, and `9d9f3edf` add fail-closed post-run
  verification and a directly executable compact qualification for the two
  persistent bulk geographies.
- `bulk_uniform` executed homogeneous persistent full-field bulk plasticity
  with tip emission disabled. `tip_plus_bulk` executed the same bulk mechanics
  with tip emission enabled. All six initial/restarted/uninterrupted native
  model-audit bindings matched the requested option exactly.
- Each fresh-process step-1-to-2 restart matched an independent uninterrupted
  step-2 state across all 11 checkpoint arrays; the paired NPZ files also have
  identical SHA-256 hashes.
- This closes execution/restart readiness for native source geography only. It
  is not a production or scientific sensitivity result.
- Focused status: 11/11 tests pass; the complete suite passes 929/929 in
  242.42 s.
- Remaining readiness scopes are tensor/constraint, FEM/CZM, branch/cavity,
  capability-wide restart, and long-growth analysis.

## Tensor and transverse-constraint qualification, 13 October 2026

- Commit `2111155b` corrects the full returned strain state for prescribed
  transverse stress and defines the reduced elastic potential appropriate to
  that dead-load control. No elastic parameter or constitutive law changed.
- A two-element affine triangular patch passes plane strain, plane stress,
  generalized prescribed `eps_zz`, and generalized prescribed `sigma_zz`.
  Each case checks the full tensor, transverse condition, rotation-invariant
  potential, and a finite-difference derivative against in-plane stress.
- The largest derivative relative error is `8.975e-12`; the prescribed-stress
  residual is `1.49e-8 Pa`. Twenty-three related mechanics/FEM and manifest
  tests pass; the complete checkout passes 934/934 tests in 212.51 s.
- The scope is compact affine-patch readiness, not a cracked-body, nonlinear,
  or 3-D qualification. Remaining scopes are FEM/CZM, branch/cavity,
  capability-wide restart, and long-growth analysis.

