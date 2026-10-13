# Native Taylor–Peierls repair progress

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

