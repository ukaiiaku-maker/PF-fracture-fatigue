# V2.3 exact-row cycle-hazard fatigue protocol

This bounded follow-up starts from sealed V2.2 commit
`f3ae18503c243b2afa6f5af7f323c69c6aa681ca`. It tests only
`P25_TJBSV2_S_002987` at 300 K, R=0.1, and 1000 Hz through the qualified
reduced v9.14 event-to-event fatigue driver at external immutable commit
`a74c4c38aad706209c7c9a8157971c847d2bf9bd`.

Load selection is based on the exact-row fresh-state cycle-integrated cleavage
hazard. A fixed 1--30 MPa sqrt(m) Kmax grid in 0.25 MPa sqrt(m) increments is
evaluated with 64 phase points. Three distinct grid points nearest projected
median first-passage targets of 100, 1,000, and 10,000 cycles are selected in
log-cycle space. The model-native monotonic ramp onset is reported as context
and is not imposed as a cyclic load cap.

Each physical trajectory uses seed 1720, the exact complete candidate row,
independent unit-exponential event thresholds, the existing threshold-scaled
bounded mean-preserving event-length law with 5 micrometre base length, a
25 micrometre developed-extension target, and a 1e8-cycle physical censor.
The middle load receives an explicit-cycle overlap calculation. No barrier,
renewal rule, state parameter, or event length may be retuned after selection.
No spatial or two-dimensional calculation is authorized.

The V2.2 interpretation is retained: the 900 K response is
`1D_RELOAD_SEPARATED_EFFECTIVE_RESISTANCE`, not an ASTM or conventional
R-curve. Canonical production promotion remains withheld. Successful exact
row loading supports only an additive prospective cross-code registry record.
