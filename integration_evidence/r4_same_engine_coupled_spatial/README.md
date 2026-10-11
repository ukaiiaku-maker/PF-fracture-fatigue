# R4 same-engine coupled spatial restoration

## Decision

R4 closes the requested supported-scope gaps on top of R3. The tested portable
calculation tree is commit `d77f387c2b4b88d25429e5cc31099107516b170f`
(tree `1e2909e2fa30ae3e4cdf8add9cc7f9c187ada331`), descended directly from
R3 commit `b156fb78b3bdebebe10c352a723801beef33382b`.

The supported R4 scope is complete. Global restoration remains partial in the
larger capability sense: the mechanically active reduced fixture uses a
documented **static prescribed off-axis cavity**, not a qualified moving-void
response map; nonpositive full-spatial cavity contact remains unsupported; and
the independent reduced opening-to-K closure is not claimed as a finite-
specimen mechanical prediction.

## Results

| Outcome | Result |
|---|---|
| Current backend through optional voiding | Direct/current and void-off adapter states are bitwise equal through events and restart |
| Active combined cavity–fracture path | Fixed-void V5 observation raises consumed energy-equivalent K by 4.2543%; five native fracture events occur with downstream cavity ownership and exact restart |
| Passive control | Cavity-owned/feedback-disabled arm equals void-disabled complete state; split-call diagnostic is labeled as a different event-terminated experiment |
| Fresh current-law spatial path | 900 K 10/20/30/20/30 µm opening history, four native geometry events, 14.527 µm projected extension, unload/reload completed |
| Matched reduced/spatial law | Same v9.13 material, persistent/reversible state strategy, named Taylor consumer, exponential marks and threshold-scaled rewards |
| Resolved stochastic fatigue | Seven rewards, 22.0512 µm by cycle 23; exact owned-state, pending-mark, phase-cursor, event and continuation restart equality |
| R3 topology | Retained unchanged; affected combined R3/R4 regression passed |
| Exact-tree tests | 804 passed, 0 failed, 0 skipped in 631.67 s |

## Backend identity and coupling

The R3 clean V3 donor remains a useful historical control, but its selected
`BoundedPreconnectionSubgridClosure` returns the no-void far-field G for every
subgrid cavity radius. It therefore does not prove mechanically active coupling
to the current backend.

R4's adapter owns
`ReducedPersistentReversibleTipEngine` and its `UnifiedMPZState`. The live
receipt records persistent reusable tip-surface sources, reversible signed
transport, the existing complementarity solution, the state-resolved signed
kernel, a 30 GPa spatial cohesive ceiling, and
`MEAN_FORWARD_MINUS_ZERO_RATE_OVER_DENSITY_ORDER`. Trial evolution is a private
deep clone containing the complete engine and RNG; commit installs that accepted
owner and rejects stale generation or parent-state tokens.

The active mechanical input is the existing V5 exact fixed-void marginal
observation for the `(100)` candidate in the `above` prescribed-cavity case:

- no-void G: 19.3863336980 J/m²
- fixed-void G: 21.0709346430 J/m²
- normalized G change: +8.68963%
- energy-equivalent K ratio: 1.04254319538

The repository's V5 provider defines energy-equivalent K as `sqrt(E' G)`; R4
uses the void/no-void ratio from that provider and does not fit an event time or
alter a barrier. At the six common external stages, the active arm receives
4.2543% higher K. The first-stage cleavage rate changes strongly because it is
below the later 10⁶ s⁻¹ ceiling; later rates share that ceiling. Source
populations diverge causally even though the event-terminated deterministic
fixture advances once in each event-bearing stage. The cavity remains a static
55 µm prescribed mechanics fixture throughout and is not relabeled as natural
nucleation or moving-cavity reduction evidence.

## Fresh spatial comparison

The spatial entry installs the same persistent/reversible engine and Taylor
consumer in the qualified v10.2.27 FEM driver. An absolute-opening controller
executes 10/20/30/20/30 µm without adaptive subdivision. The spatial arm obtains
its own J-derived K at each evolved geometry; the prescribed-drive reduced
replay uses those K rows only as inputs. A second reduced arm uses its independent
virgin 10 µm linear opening-to-K closure and does not ingest later spatial state.

The spatial run records four accepted renewals and 15.1592 µm path length
(14.5270 µm projected extension). Its 30 GPa ceiling is inactive in all five
rows. Replay stress differences are 0.0205%, 0.0110%, and below 3e-5% thereafter,
while post-event source populations differ substantially. Those source
differences are retained; no complete-state equality between reduced and
spatial models is claimed.

## Resolved stochastic fatigue

The retained R3 mean-cycle four-checkpoint result is unchanged. R4 separately
executes the current engine with phase-resolved native state evolution,
exponential mark seed 1733, threshold-scaled bounded reward, continuous MPZ
translation, and same-phase remainder continuation. The run reaches the
historical 20 µm target with seven rewards and terminates at 22.0512 µm in cycle
23. More than one event occurs in a cycle.

The disk checkpoint is taken immediately after event two at cycle 18, phase 1,
with 0.00275474232 s left in that phase. Owning arrays, ledgers, geometry, RNG,
pending mark, cursor, later events and terminal continuation match exactly.
Recomputable last-call diagnostic caches are intentionally not serialized and
are reported separately as non-identical; they do not affect the exact
continuation.

The included one-cycle mean-cycle comparison starts from the same marked state.
Neither arm fires in that affordable interval. It is a state comparison, not a
claim that changing frozen-rate phase quadrature is phase-resolved evolution.

## Evidence and tests

Primary files:

- `current_coupled_run3/current_coupled_r4.json`
- `current_law_spatial/steps_0900K.csv` and the associated raw spatial outputs
- `current_law_spatial_comparison_portable.json`
- `resolved_stochastic_fatigue_run3/result.json` and complete checkpoint
- `r4_combined_results.png`, generated only from the raw arrays above
- `final_full_suite_portable.xml`
- `dependencies/`, containing the exact fixed-void observation and signed kernel
- `preflight/r3_source_and_evidence_b156fb78.tar.gz`

The authoritative test log reports 804 tests, zero failures/errors/skips, in
631.645 seconds. The earlier `final_full_suite.xml`, `current_coupled_run2`, and
`resolved_stochastic_fatigue_run2` remain as pre-portability provenance. The
untracked first-attempt diagnostic directories are excluded from the review
archive.

Primary SHA-256 values:

```text
40a08d3b1618afc98607b63930db284143fc0ee69b020a21074ae893ff35ddf3  current_coupled_run3/current_coupled_r4.json
79ca1e7870ce6cdfb172345da701696d6f31b5d4aa0c906d36edcedb4ed22af4  current_law_spatial_comparison_portable.json
2fb76405d0f54388143acb01f78814f7ab8408a66dbbf8b43544ee0eb70e1870  resolved_stochastic_fatigue_run3/result.json
bf5174fa0137e7290b6d0494246102bdfeab2a4a5280dcbcdbe6a0f35cadfb7e  final_full_suite_portable.xml
a3178a12a1ac3227095d861e040f65318e1676b1c85c4afc298cbf35817994d4  dependencies/off_axis_prescribed_void_benchmark_v2.json
5aa74370e6419104684c52bbcf93323f905d2752d1bba59252f9b0b35c77e07c  dependencies/signed_kernel_family.json
71600cc5f803348ce759976f98ddb0164ab022b16adedd94e07548639e41fc88  r4_combined_results.png
d5df6e09013bfc3ca78199925c35feefd8f98d6310c54272cd35122dc97bb929  preflight/r3_source_and_evidence_b156fb78.tar.gz
```

See `commands.txt` for exact reproduction commands. No remote push was made.
