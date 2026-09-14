# Frozen V2.2 reduced 1-D response protocol

The reported monotonic observable is `1D_RELOAD_SEPARATED_EFFECTIVE_RESISTANCE`.
It is not an ASTM R-curve, J-R curve, conventional tearing resistance, or a
material K_R curve. `J_equiv = K_reinit^2/E_prime` is only an equivalent plotting
coordinate. Model-native first passage is not ASTM K_IC, and every new row remains
prospective and not experimentally calibrated.

All eight sealed V2.1 rows are screened at 900 K before selection. The principal
threshold is `Xi=ln(2)`, the load rate is 0.005 MPa sqrt(m)/s, the censor is 80
MPa sqrt(m), and each qualified reduced event commits the inherited fixed 5 um
advance. After each single event, applied K returns to zero with zero dwell while
the complete persistent state and moving-frame renewal are retained. Full state
and the existing `plasticity_enabled=False` opening-only diagnostic are matched.
Ten points (50 um) are requested. Class gates are copied verbatim into the JSON
protocol and cannot be changed after execution.

The monotonic controller delegates first passage, continuous 5 um translation,
and persistent-state renewal to the existing v10.2.9/v10.2.30 state-resolved
reduced engine. The cyclic campaign, if admitted, delegates waveform integration,
same-cycle continuation, stochastic thresholds, 5 um threshold-correlated event
lengths, checkpoint/restart, and qualified acceleration to the accepted v9.14
reduced driver. No spatial energy-gated wrapper or two-dimensional solve is used.

The archived reduced monotonic sentinel and finite v9.14 fatigue case must match
their immutable records before execution. Candidate selection and fatigue loads
remain disabled until every eight-row screen is terminal and the K-coordinate
audit returns `COMMON_FATIGUE_AND_MONOTONIC_K_COORDINATE_QUALIFIED`.
