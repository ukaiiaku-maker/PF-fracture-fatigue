# V12 mechanically separating sharp-wake qualification

This draft branch is based on V11 production commit `2b5e5351add0bf0db67f2cda35a1480c3e7efc91`
and defines the separate model identity `sharp_wake_mechanically_separating_v12`.
V12 is not installed in the production transaction path and PR #57 is unchanged.

## Attested geometry baseline

The final geometry evidence at branch head
`1b426ef04b3f371d16059a370f71883625bb53e1` reports
`MECHANICALLY_SEPARATING_WAKE_GEOMETRY_QUALIFIED = PASS`. Scoped Python 3.12
workflow run `33446100847` tested that exact head and passed 88 scoped tests,
two deterministic evidence regenerations, `git diff --check`, and the clean-worker check.

The Phase 0 integrity patch replaces literal scientific PASS values with
evidence-derived Booleans, adds bidirectional graph/support component incidence,
detects nonadjacent-arc support short circuits, calculates legal junction
overlap geometrically, adds Y/T defective controls and exact physical
coalescence, and adds an explicit no-mechanical-novelty event control. Its local
focused suite passes 60 tests; its exact implementation source is recorded in
the regenerated qualification provenance.

## Geometry evidence requirements

The runner derives every scientific geometry gate from its rows:

- uniform fixed-domain refinement requires decreasing physical width and area,
  convergent signed tip footprint, bounded `width/h`, and bounded `area/(L h)`;
- the fixed crack-local patch must remain invariant under independent far-field
  remeshing by physical-coordinate support fingerprint;
- event classification requires production-valid acceptance, alignment-remesh,
  and no-mechanical-novelty controls, correct classifications, a stiffness
  fingerprint change for every accepted event, and sequential equivalence;
- every graph component and node-connected support component must have a
  one-to-one incidence outside declared junction/coalescence neighborhoods;
- nonadjacent certification arcs in one graph component may not become
  node-connected through their support;
- kink, Y, T, mixed-role, merged-terminal, and exact-coalescence neighborhoods
  use geometric overlap and annular sector certificates; deliberately defective
  kink, Y, and T supports must be rejected.

## Reproduction

```bash
python scripts/qualify_v12_mechanically_separating_wake.py
python -m pytest -q \
  tests/test_v12_mechanically_separating_wake.py \
  tests/test_v11_causal_sharp_wake.py \
  tests/test_crack_network_v11.py \
  tests/test_v11_live_topology_multitip.py \
  tests/test_topology_transaction_v11.py
```

## Gate ledger

Authoritative geometry states are generated in
`artifacts/v12_mechanically_separating_wake/qualification.json`; they are not
manually asserted here. At the attested baseline all construction, partition,
local-objectivity, component-topology, coalescence, junction-sector,
active-tip/event-resolution, and overall geometry gates are PASS.

| Later gate | State |
|---|---|
| `PRODUCTION_TRANSACTION_ROLLBACK_QUALIFIED` | NOT_RUN |
| `MECHANICALLY_SEPARATING_WAKE_PRIMAL_MECHANICS_QUALIFIED` | OPEN |
| `MECHANICALLY_SEPARATING_WAKE_ABSOLUTE_K_QUALIFIED` | NOT_RUN |
| `V12_SHARP_WAKE_PRODUCTION_PREREQUISITE_QUALIFIED` | OPEN |

Geometry PASS does not authorize production wiring, absolute-K evaluation, or
voiding. The next bounded gate is matched conforming-versus-V12 primal mechanics.
