# V12 mechanically separating sharp-wake prerequisite

This branch is based on V11 production release `2b5e535` and introduces the
separate model identity `sharp_wake_mechanically_separating_v12`. It does not
install the model in the V11 production transaction path and does not change
`sharp_wake_causal_v11`.

The V12 candidate derives support from the complete accepted crack graph. It
classifies physical roots, active tips, inactive terminals, degree-two interior
vertices, branch junctions, and merged vertices. Every exact graph-interior
support node must have its complete P1 element star disabled; an unresolved
star, unresolved graph edge, non-O(h) width, or non-O(h) tip leakage fails
closed. Only a node exactly coincident with a currently active tip is exempted,
so a former tip closes when it becomes an interior vertex.

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
| `MECHANICALLY_SEPARATING_WAKE_GEOMETRY_QUALIFIED` | PASS | 112/112 matrix cases certified; no unresolved node-star bridges |
| `MECHANICALLY_SEPARATING_WAKE_PRIMAL_MECHANICS_QUALIFIED` | OPEN | Matched conforming-crack displacement/compliance comparison not yet committed |
| `MECHANICALLY_SEPARATING_WAKE_ABSOLUTE_K_QUALIFIED` | NOT_RUN | Interaction-integral work is forbidden until primal mechanics passes |
| `V12_SHARP_WAKE_PRODUCTION_PREREQUISITE_QUALIFIED` | OPEN | Requires both primal mechanics and absolute-K gates |

The result is therefore a geometry-qualified draft prerequisite, not production
approval and not authorization to resume the void-nucleation mission.

## Test result

The scoped V12 plus V11 topology regression command passes `52 passed`. The
full repository suite reports `704 passed, 1 skipped, 7 failed`; all seven
failures are in pre-existing V10.2 compatibility/status tests outside this
branch's additive two-file implementation diff (legacy model-ID assertions,
a sandbox-denied `ps` call, zero-event summary compatibility, and a historical
fatigue delegate contract). No full-suite PASS is claimed.
