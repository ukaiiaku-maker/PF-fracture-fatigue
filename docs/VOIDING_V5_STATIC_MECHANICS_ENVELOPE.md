# V5 static-mechanics envelope

> **STATUS: SUPERSEDED_INITIAL_FINALIZATION_ATTEMPT.** Results below are
> retained for provenance only. No PASS claim in this document is authoritative.

The structured campaign contains 156 attempted production-FEM cases covering
five ligament ratios, five signed offsets, three nominal resolutions, a
15-case three-level convergence subset, opening sensitivity, fixed-mesh
orientations, four far-limit cases, four Kirsch controls, and centered energy
perturbations.

Every retained pass uses the body-fitted cavity mesh and exact V12 support; no
toy solver or centroid-band fallback is used. Failed exact-support certificates
are preserved. The prospective recovery ladder retries once with four additional
radial layers; successful retries are classified separately.

The demonstrated envelope is 2-D plane strain, one crack and one void. Cases
whose exact support lacks opposite-side seeds or whose certified arc is too
short are outside the current geometry/resolution envelope. This is not repaired
by changing physical parameters.

Energy output distinguishes `-d(U/B)/da` and `-d(U/B)/dR`, both in J/m², from
`d(U/B)/dA`, whose actual units are J/m³. Fixed-opening release signs are used.
