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

