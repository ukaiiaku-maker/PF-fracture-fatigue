# V12 mechanically separating sharp-wake prerequisite

This branch is based on V11 production release `2b5e535` and introduces the
separate model identity `sharp_wake_mechanically_separating_v12`. It does not
install the model in the V11 production transaction path and does not change
`sharp_wake_causal_v11`.

The V12 candidate derives support from the complete accepted crack graph. It
classifies physical roots, active tips, inactive terminals, degree-two interior
vertices, branch junctions, and merged vertices. Every exact graph-interior
support node must have its complete P1 element star disabled; an unresolved
star, unresolved graph edge, independent intact crossing path, nonlocal retained
support, premature mechanical coalescence, non-O(h) width, or non-O(h) tip
leakage fails closed. Only a graph vertex whose sole role is active degree-one
tip can be exempted, so a former or mixed-role tip closes.

## Reproduction

```bash
python scripts/qualify_v12_mechanically_separating_wake.py
python -m pytest -q tests/test_v12_mechanically_separating_wake.py \
  tests/test_v11_causal_sharp_wake.py tests/test_crack_network_v11.py \
  tests/test_v11_live_topology_multitip.py tests/test_topology_transaction_v11.py
```

The committed geometry matrix covers four nested resolutions, structured and
deterministically perturbed meshes, seven orientations, and two endpoint
phases. Unit tests separately cover graph partition and insertion-order
invariance, monotone sequential growth, kinks, branching, near-coalescence,
explicit vertex classes, fail-closed empty graphs, and exact rollback isolation.

## Gate ledger

| Gate | State | Evidence |
|---|---|---|
| `GRAPH_AWARE_NODE_STAR_CONSTRUCTION_SCREEN` | PASS | 112/112 preserved construction cases |
| `SYNTHETIC_ORIENTATION_AND_PHASE_SCREEN` | PASS | Four resolutions, two mesh families, seven angles, two phases |
| `SYNTHETIC_INDEPENDENT_INTACT_PATH_SCREEN` | PASS | Nonvacuous independent search passes the 112 synthetic cases; exact-only support is detected |
| `INDEPENDENT_INTACT_PATH_SEPARATION_CERTIFIED` | OPEN | Broad adaptive/branched and junction-sector certificate remains outstanding |
| `LOCAL_H_AND_FULL_SUPPORT_OH_OBJECTIVITY` | OPEN | Edge-local metrics and far-field grading factors 1–16 pass; local-refinement convergence remains outstanding |
| `NO_PREMATURE_MECHANICAL_COALESCENCE` | OPEN | Near-branch counterexample now fails closed; full separation-distance matrix remains outstanding |
| `ACTIVE_TIP_AND_EVENT_RESOLUTION_QUALIFIED` | OPEN | Off-grid production tips and unresolved subcell events fail closed; complete event sweep remains outstanding |
| `ACCEPTED_STATE_NONMUTATION_OR_TRIAL_ISOLATION` | PASS | Pure trial leaves accepted arrays and ownership unchanged |
| `PRODUCTION_TRANSACTION_ROLLBACK_QUALIFIED` | NOT_RUN | Production wiring remains blocked |
| `V12_CLEAN_WORKER_SCOPED_CI` | EXTERNAL_PASS | Run 33344440023 passed at tested head fdc47b614b54e8b03988a0909ca71da1b249c900; the new head must rerun |
| `MECHANICALLY_SEPARATING_WAKE_GEOMETRY_QUALIFIED` | OPEN | Broad arbitrary-graph/adaptive claim is withdrawn |
| `MECHANICALLY_SEPARATING_WAKE_PRIMAL_MECHANICS_QUALIFIED` | OPEN | Matched conforming-crack displacement/compliance comparison not yet committed |
| `MECHANICALLY_SEPARATING_WAKE_ABSOLUTE_K_QUALIFIED` | NOT_RUN | Interaction-integral work is forbidden until primal mechanics passes |
| `V12_SHARP_WAKE_PRODUCTION_PREREQUISITE_QUALIFIED` | OPEN | Requires both primal mechanics and absolute-K gates |

The result is therefore a hardened synthetic construction/separation screen,
not a broad geometry qualification, production approval, or authorization to
resume the void-nucleation mission.

## Test result

The scoped V12 plus V11 topology regression command passes `61 passed`. The
full repository suite reports `704 passed, 1 skipped, 7 failed`; all seven
failures are in pre-existing V10.2 compatibility/status tests outside this
branch's additive two-file implementation diff (legacy model-ID assertions,
a sandbox-denied `ps` call, zero-event summary compatibility, and a historical
fatigue delegate contract). No full-suite PASS is claimed.
