# R7 current-state restoration result

## Decision

R7 closes the R6 accepted-update, fatigue-accuracy, low-level remap trace, and
files-only restart defects.  It also executes the original full-spatial
current-state bulk/void scheduler and a pre-existing resolved-cavity cycle.

It does **not** claim that the final governing objective is fully closed.  The
short full-spatial run uses its native `UnifiedMPZFrontEngine`; the separate
dynamic current-tip run still obtains bulk/site fields from the recovered V3
history closure.  Therefore the supplied tree does not yet contain one
continuous trajectory that combines the selected persistent/reversible current
tip backend with the live full-spatial bulk producer and a fracture event.
Those two valid executions are retained as separate evidence and are not
combined into a synthetic history.

## Implemented production changes

- `advance_current_source_to_earliest_event` compares private full and two-half
  source/transport states, including signed/bin-resolved arrays, source/return
  ledgers, permanent slip, radius, rates, stress, action, event identity and
  event time.  The accepted split stops at the first event; the coupled caller
  owns the remainder.
- The dynamic adapter now uses that controller above a qualified 1 microsecond
  cap and a private native update at or below the cap.  Every receipt identifies
  the cap and its real-engine full/two-half qualification.  A half-local versus
  interval-relative event-time defect found during execution was corrected.
- The composite checkpoint manifest contains adapter generation/counters,
  mechanics binding, geometry descriptors, backend identity, native event
  count, dependency identities, every native checkpoint-file hash, the donor
  payload hash, configuration identity, and pending-scheduler state.  A new
  interpreter restores from the manifest and files alone, executes the native
  fracture event, and continues through the later interval exactly.
- Mechanical correspondence records 234 actual original moving-frame
  translation/wake-remap operations with bin-resolved pre/post hashes, totals,
  and native receipts.  Path length, projected endpoint, tangent, and
  material-fixed wake semantics are distinct.  No target population or future
  event time is imported.
- Fatigue output tolerances are prospective.  The 48-to-96 phase comparison
  passes: 0.279653% first-event-time error, 0.708064% terminal continuous
  micro-advance error, and 0.025266% terminal population error.  Native-action
  and source-refresh spreads are 4.52e-7 and 3.52e-6.  This qualifies the
  three-cycle one-event transient, not a developed rate.
- The altered recurrent stochastic path was rerun at 96 phases to its original
  20 micrometre/four-event minimum target.  It reached 22.0512 micrometres at
  cycle 22 with seven events and exact owned-state restart continuation.

## Executed evidence

| Evidence | Result | Interpretation |
|---|---|---|
| `live_spatial_pair/shared_v5_cavity_audit.json` | completed 1e-8 s in both arms; native restart exact | live full-field bulk update, current FEM observation, vacancy/void scheduler; no natural cavity or fracture event |
| `preexisting_cavity_cycle/result.json` | 3 micrometre resolved cavity, load/unload/reload, restart exact | supported current spatial cavity feedback fixture; no fracture event |
| `dynamic_current_tip/dynamic_void_r5.json` | pass; one native fracture event; radius changed; later continuation | current persistent/reversible tip plus dynamic recovered void owner; bulk/site inputs remain prescribed-history control |
| `state_correspondence.json` | pass; 234 remap records | actual reduced moving-frame operations are traced; autonomous 6.97449 degree spatial turning mismatch remains |
| `fatigue_accuracy.json` | `ACCURACY_QUALIFIED` | prospective three-cycle transient output accuracy |
| `recurrent_fatigue_96phase/result.json` | pass; 22.0512 micrometres at cycle 22 | recurrent target rerun with exact restart and multiple events |

## Preserved controls and limits

- The 42 nm zero-far-field-correction control remains exactly zero.
- The old 55 micrometre fixed-response multiplier is not used by any new
  dynamic or spatial run.
- The cavity-free incubation fixture is not relabeled natural birth.
- The full-spatial short pair and pre-existing-cavity cycle are not relabeled a
  continuous fracture trajectory.
- The turning-path residual remains an autonomous reduction error; the new
  trace establishes the exact accepted reduced operation but does not make the
  straight reduced path acquire the spatial 6.97449 degree tangent.
- A fully integrated selected-current-tip/full-spatial event trajectory and a
  matched no/passive/active full-spatial schedule through that event remain the
  unresolved R7 completion items.

## Verification

- Focused changed-path suite: 12 passed in 815.07 s (`focused-tests.xml`).
- Final exact-tree full suite: 820 passed in 2176.22 s (`full-suite.xml`).
- Calculation base: R6 packaging commit
  `6e26299ceae65d7e480b6382e09ddbf2e9d1a4d8`.
- Tested R7 calculation commit:
  `56ac7f1808bbda173bd4c1baef777aa7d4dc71d0`, tree
  `5473192ddda359ae68aff71e40d20f816d77132a`.

