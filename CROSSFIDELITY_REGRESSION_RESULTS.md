# Cross-fidelity regression results

## Unchanged-contract recovery

The strongest fresh R1 reduced/spatial check is
`runs/r1_v1023_crossfidelity_reload`. The full spatial v10.2.3 engine generated
a 39-row 1200 K schedule. It accepted 5 um events at the first event and again
after reload, for 10 um total extension. The reduced replay consumes the exact
recorded `K`, time, temperature, channel factors, expected event flags and
state targets—there is no fitted conversion.

| Metric | Result |
|---|---:|
| maximum array relative error | 2.7035439210412937e-14 |
| maximum array absolute error | 1.4551915228366852e-11 |
| maximum scalar absolute error | 2.223532646894455e-07 |
| event flags | all match |
| raw/effective shielding invariant | pass |

This is deterministic same-law/state correspondence. It validates state
evolution, geometry-linked `r_eff`, source capacity, mobile/retained history,
shielding, event ownership and post-event reload against the recorded spatial
trajectory. It does not establish independent rounded-tip shape parity.

## Correction impact

R1 exposes the reduced integrator's internal-step budget as a numerical control;
the default physics is unchanged. Lower-budget active attempts stopped at the
existing fail-closed internal-step limit. Raising only that budget allowed the
same laws to complete the active event/reload schedule. No barrier or source
parameter changed.

The V5 loading correction makes the accepted structural factor explicit and
checkpoint-owned. A public `1.0,0.2,1.0` run completes and restart round-trips
exactly. Nonpositive applied opening is rejected because native fixed-load V5
equilibrium requires positive opening; this replaces an ambiguous runtime
failure with an honest model boundary.

## Preserved evidence

The inherited qualified long-growth spatial result remains 1002.9028827375072
um projected extension with 201 events and 2,085 accepted rows. Its stage-aware
observer preserves repeated lengths and distinguishes equilibrated/pre-event,
committed-geometry and later re-equilibrated states. The V4.10 cold censors and
five-event seam result remain diagnostics and are not promoted into this
cross-fidelity family.

Compact raw tables are in `integration_evidence/r1_restoration/`.
