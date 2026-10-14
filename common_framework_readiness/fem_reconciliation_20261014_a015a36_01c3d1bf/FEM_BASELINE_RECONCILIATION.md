# FEM baseline reconciliation

Date: 2026-10-14

## Finding

The established elastic FEM/J solver is not the source of the recently reported convergence failure. The v10.0.5.11 reference was recovered at commit `23463cac72ebbc3be9eb53ea9fd86229102e1e98`, rerun in an isolated temporary checkout, and rerun from the current FEM head `a015a36b07a8887e7bc42f63f2bc39f42e3470f6`. The scientific payloads are identical at full precision.

At 1 micrometre total grip opening the recovered result is:

- same-mesh fixed-grip energy derivative: 28.789526830110503 J/m2;
- median accepted-contour J: 28.468695078083666 J/m2;
- J/G: 0.9888559560592957;
- derivative consistency error: 0.015401631559229462.

The 2 micrometre result is exactly four times the 1 micrometre result and has the same J/G ratio. The historical and current blobs for `mesh.py`, `fem.py`, `j_integral.py`, `physical_refinement_mesh_v100510.py`, `production_j_same_mesh_energy_v100511.py`, and the v10.0.5.11 audit runner are identical.

## First demonstrated divergence

The retained failed generic receipt varied the requested `tip_h_fine`, but the physical path-envelope provider selected support spacings of 12.25, 12.25, and 11.025 micrometres. Thus the first two nominal levels did not refine the controlling support. In addition, the quantity named as the common J observer was `KJ^2/Eprime`, derived algebraically from native J/K rather than evaluated independently from the field. The 12.81599292814948% medium-to-fine change is real for those runs, but it is not evidence of convergence or failure under a controlled refinement axis.

The PF-side qualifier now fails this retained receipt as `FAIL_INVALID_REFINEMENT_AXIS` before considering the 2% threshold and records that its common observer is not independent. The regression uses the retained receipt, so it will distinguish this protocol defect from a solver change.

## Corrected controlled qualification

The corrected benchmark uses the actual v10.0.5.18.3.4 physical reachable-envelope mesh provider and refines the selected path-envelope support ratio at fixed geometry, constitutive tensor, loading, process-zone length, and nominal tip input:

| support h/Lpz | nodes | reaction (N/unit thickness) | stored energy (J/m) | fixed-grip G (J/m2) | median full-energy J (J/m2) |
|---:|---:|---:|---:|---:|---:|
| 0.245 | 1539 | 39064.350742148235 | 0.003906435074214766 | 2.183237212997412 | 1.9317922455092986 |
| 0.180 | 2653 | 39044.008119218386 | 0.003904400811921725 | 2.391424579018917 | 1.9453330899522177 |
| 0.120 | 5724 | 38955.33174892638 | 0.003895533174891940 | 2.3540644513243905 | 1.9372500338123593 |

Medium-to-fine relative changes are 0.227636% for reaction and stored energy, 1.587048% for independently evaluated fixed-grip G, and 0.417244% for native full-energy J. All satisfy the declared 2% criterion. Matrix/stored-energy closure is below 2e-13.

This does not establish historical observer parity for the current path-envelope representation: the fine-level J/G ratio is 0.8229384, versus 0.988856 in the recovered production-mesh benchmark. That difference is now isolated from numerical support convergence and remains an observer/representation question. No fitted factor was introduced, and no constitutive or kinetic law changed.

## Lifecycle result

Current native tests separately exercise exact endpoint insertion, cohesive remeshing, rollback, bulk-state transfer after integration-point-count change, same-load post-event equilibrium, moving-tip state translation without repeated advance at checkpoint, persistent signed-state restart, and autonomous cohesive active-damage restart. The combined targeted group passes 78 tests in the dedicated FEM environment. The complete FEM suite passes 701 tests and 16 subtests.

The retained matched FEM evidence contains four successful atomic physical events with exact endpoint and direction preservation and complete-event remeshing before cohesive commit. That historical run subsequently stops because physical-refinement metadata is unavailable to the terminal attestation. It is therefore evidence for the four committed transitions, not a completed-run claim.

The native state serializers and restart transitions are preserved. The current generic FEM entry still rejects `--restart-from` because its production specimen driver has no deserializer at that entry. This is classified as native capability preserved but not wired through the shared interface, not as a demonstrated native restart regression.

