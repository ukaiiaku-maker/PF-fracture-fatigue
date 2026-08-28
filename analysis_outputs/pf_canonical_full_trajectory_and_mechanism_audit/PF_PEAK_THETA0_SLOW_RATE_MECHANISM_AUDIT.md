# PF Peak Theta0 Slow-Rate Mechanism Audit

## Decision

The Peak slow-rate onset elevation is **not a structural KJ/opening effect**.
Before initial fracture, `KJ/U` is identical across the matched 0.01x, 1x,
and 100x cases to a maximum relative spread below `1e-12` at every
temperature. The entire exact KJ difference is therefore the greater opening
required after rate-dependent local-state evolution.

Across the deep 900–1200 K audit set, slow minus 1x onset KJ averages
**25.76 MPa sqrt(m)**. Median slow/1x ratios are **1.12**
for radius, **3.20** for mobile population, and
**1.19** for backstress; the median signed-shielding change is
**0.346 MPa sqrt(m)**. Retained population and multiplicity vary
non-monotonically with temperature, so neither supports a single-variable
explanation.

The supported classification is **TIME_AVAILABLE_FOR_EMISSION +
MIXED_LOCAL_STATE_EFFECT**, with important emission/blunting and backstress
changes and temperature-dependent shielding/multiplicity. A unique additive
partition among radius, backstress, shielding, multiplicity, and cleavage
action is not supported because the nonlinear evolved-state component swaps
cannot be reconstructed exactly from the default-off observer archive.

## Evidence and limits

- Common random numbers are preserved across rates at fixed class and
  temperature (identical seed triplets).
- The accepted-state archive supplies scalar radius, width, populations,
  backstress, signed shielding, multiplicity, rates, actions, resolved shears,
  and channel-resolved transport.
- Complete tensor matrices were not archived. No tensor was interpolated or
  inferred from a later state.
- The counterfactual table is fail-closed: actual states are recorded, while
  unavailable one-at-a-time evolved-state injections are explicitly marked
  unavailable rather than fabricated.
- Model-native KJ is not applied K or a conventional R-curve.
