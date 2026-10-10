# Mechanics delta regression

Reviewed range: `9e884fb0b0845da621d2612bdf1042e481b8df49..current`.

| File | Semantic change | Activated evidence | Classification |
|---|---|---|---|
| `coalescence.py` | detects a collinear segment spanning the complete proposed increment | disjoint, endpoint, partial overlap, reversed orientation, full-span, transverse crossing | intentional topology correction |
| `mesh.py` | stores optional elementwise P0 damage; restores checkpoint mesh metrics | ULP perturbation accepted and authoritative values restored; 1e-7 relative mismatch rejected | intentional state/restart correction |
| `fem.py` | uses P0 element damage when present, nodal average otherwise | element damage 0.5 produces the exact degradation-ratio change; fallback matches prior nodal mechanics; assembly/stress agree | intentional mechanics correction |
| `crack_backend.py` | advances P0 damage together with nodal visualization damage | existing backend/topology suites plus P0 assembly fixture | intentional state-consistency correction |
| `plasticity.py` | exports named rate/accepted-state diagnostic arrays | existing plasticity suites | diagnostic, no state equation change |
| `sharp_front.py` | combined restart, terminal accepted-state synchronization, and forensic output | existing restart/event suites and public two-step full-PF run | execution/restart correction; not asserted numerically neutral |

The element-damage test deliberately differs from the nodal average, so the
corrected mechanics is expected to differ from the base in that activated
case. No parameter was retuned to recover the old response. The historical
schema has nodal damage only; the explicit native mapping is therefore absent
`element_damage_gp` to the current nodal-average fallback. The activated P0
fixture remains intentionally different at the exact expected degradation
ratio.

The immutable `9e884fb0` base and current `1cd1738` were also run through the
same v10.2.21 full-spatial entry, material row, kernel family, seed, mesh, and
loading. The one-step fallback case had byte-identical step, front, and event
histories. A two-step event case was likewise byte-identical through one
accepted geometry transaction: reactions, displacement, native K/J, front
state, event geometry, and terminal topology all matched. Historical v10.2.21
does not emit the later work-ledger columns; those absent values are not
reconstructed. Exact values and classifications are retained in
`integration_evidence/option_completion_v2/mechanics_base_current/comparison.json`.

The unchanged 54 MB 1002.903 µm trajectory remains evidence for commit
`baedb069`; it is not presented as a base-versus-integration equivalence test.
