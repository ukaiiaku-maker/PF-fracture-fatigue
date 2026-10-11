# R3 integrated matched restoration

## Decision

R3 restores an executable hierarchy across the previously separate components.
The result remains `PARTIAL_RESTORATION`, because the retained bounded coupled
run has no fracture event at its measured fixed load, the archived spatial
reference has no unload/reload rows, and nonpositive full-order cavity/contact
loading remains outside the qualified V5 contract.  Those limits are reported
rather than converted into fitted effects or forced events.

All four task-owned executions passed:

| Capability | Native result | Restart/result |
|---|---:|---:|
| 900 K recorded operator replay | flags `false,true,true`; first two stress differences 0.0205% and 0.000793% | exact complete active state |
| Autonomous reduced outer load/unload/reload | 4 checkpoints, 20 um committed extension | completed 10/20/30/20/30 um outer history |
| Coupled reduced fracture + optional voiding | 368 native void/lifecycle events; 42.206 nm terminal subgrid cavity | clean Git donor, one global scheduler |
| Complete V2 fracture adapter | 1 native accepted renewal through signed history | byte-equivalent direct/adapter closure |
| Ligament veto | margin -3.73878e-6 J/m | input fingerprint unchanged |
| Accepted ligament | margin +7.16411e-3 J/m | exact checkpoint restore |
| Downstream child and continuation | margins +4.08404e-3 and +2.09576e-3 J/m | exact restart continuation |
| Developed fatigue | 4 checkpoints by cycle 38; 20 um extension | all waveform time consumed; exact midpoint restart |

## Resolved contracts

- The selected spatial local strength is the independently serialized 30 GPa
  cohesive ceiling from `run_args.sigma_cap_GPa`. It is not a K-shield cap and
  is inactive in the three archived 900 K rows.
- The post-install selected event law is deterministic unit-mark/fixed reward
  with continuous MPZ translation. The older wrapper's exponential/threshold-
  correlated declaration is stale for this execution; that stochastic law is
  retained as a separate available strategy.
- A checkpoint is an outer geometry synchronization, not another MPZ
  translation. The fatigue ledger closes after every event.
- The selected Taylor consumer is
  `MEAN_FORWARD_MINUS_ZERO_RATE_OVER_DENSITY_ORDER`. Its step-three source
  population divergence from the historical spatial consumer is retained and
  explicitly labelled.

## Matched execution

`matched_campaign/matched_campaign.json` contains both comparisons:

1. Recorded K, tensor-drive and time replay checks equations and units. It
   reproduces spatial tip stress to 2.055e-4, 7.931e-6 and 3.073e-7 relative
   difference across the three rows. The third-row `N_em` difference is not
   hidden.
2. The autonomous reduced arm receives only the external archived K history,
   uses its declared fixed reduced tensor shape, and executes load/unload/reload.
   In the selected monotonic outer controller, a checkpoint ends the current
   displacement increment; unused inner time is not relabelled as a hold.

The archived spatial evidence has only three monotonic increments, so R3 does
not claim a spatial unload/reload comparison. A new full-order structural
reversal remains an open qualification.

## Coupled void execution

`coupled_void/coupled_void_r3.json` launches the exact clean donor commit
`47a5a4bae048e5a8b1c6178efb6406325005f7ef` (tree
`bbd9a4b5432ab0c262a94fe8201bb810e6a9ecc3`) from `git archive`. The dirty
donor checkout is ignored.

The bounded aligned driver owns the complete V2 fracture closure and void state
in one global first-passage scheduler. It runs identical 1200 K, 0.4 um-opening
no-void and void-enabled arms; preserves losing clocks; consumes remaining
intervals after events; and refreshes cavity-dependent mechanics. The active
arm reaches `STABLE_SUBGRID_VOID`, radius 4.22064e-8 m, after direct birth,
embryo, residual-defect stabilization and cavity-surface emission packets. The
same fixed-load combined trajectory has zero fracture renewals. A separate
exact execution of the same complete adapter records one accepted renewal and
direct/adapter complete-state parity; it is not misreported as an event in the
fixed-load coupled trajectory.

This is a functional aligned fixture, not natural-nucleation or calibrated 2-D
transfer evidence. Its mechanics binding is the donor's bounded aligned
history/radius map and its direct row remains uncalibrated.

## Connection and continuation

`connection_recovery/connection_recovery.json` retains both decisions from the
unchanged native energy gate. The centered/native-load path is vetoed for
`insufficient_whole_topology_energy_release` and rolls back exactly. The
historical offset path uses the retained crack geometry and documented doubled
symmetric Dirichlet opening. It fractures a 73.1195 um ligament, preserves the
108.562 um pre-existing cavity chord as a separate free-span ledger, remeshes,
certifies the connected topology, creates `void-front-1`, and accepts one
ordinary continuation. Selected binary checkpoint payloads are included.

## Developed fatigue

`developed_fatigue/result.json` uses the already event-bearing retained
mean-cycle provider at 1800 K. It records raw a(N), individual events, source
state, unused time, and restart state. Four 5 um checkpoints are reached by
cycle 38. Every cycle consumes its complete waveform duration, every queued
geometry event is consumed once, and the MPZ/checkpoint identity closes without
a second translation. This is not a Paris-law fit.

## Reproduction

Run from the R3 worktree with the project environment:

```text
python scripts/run_r3_matched_campaign.py --output <out>/matched_campaign
python scripts/run_r3_coupled_void.py --output <out>/coupled_void
python scripts/run_r3_connection_recovery.py --output <out>/connection_recovery
python scripts/run_r3_developed_fatigue.py --output <out>/developed_fatigue --target-events 4 --phase-count 24 --maximum-cycles 80
python -m pytest -q tests/test_matched_persistent_execution_r3.py tests/test_developed_fatigue_r3.py tests/test_connection_recovery_r3.py tests/test_coupled_void_execution_r3.py
```

The focused R3 suite reports 8 passed tests. Before the final import-neutrality
guard was added, the complete repository suite reported 798 passed tests in
164.21 seconds; the guard itself is also covered by the focused suite.

## Primary artifact hashes

```text
a06935b1be5c7cd63b3c6cb530e932d37e40d54de3e195617508f111bae09c43  matched_campaign/matched_campaign.json
9cf12fb7d74330833db8590fa28703afc1334b44721cf85495c964c0f863a902  coupled_void/coupled_void_r3.json
70ddd9de3b6ee64286bc5aa6527d54926bb299be9982882b5f00326a43dda85d  connection_recovery/connection_recovery.json
1962cec7c765ca38d73938c412a877c04895ed08fe0d3a023f99e4b1bc29dfb3  developed_fatigue/result.json
```

