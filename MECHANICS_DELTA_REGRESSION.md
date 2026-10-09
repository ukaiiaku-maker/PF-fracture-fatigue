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
case. No parameter was retuned to recover the old response. The unchanged
54 MB 1002.903 µm trajectory remains evidence for commit `baedb069`; it is not
presented as a base-versus-integration equivalence test. A controlled immutable
base/current coupled trajectory at matched accepted state remains open; the
current evidence establishes unchanged fallback behavior and the intended
local consequences of the two reported mechanics corrections.

