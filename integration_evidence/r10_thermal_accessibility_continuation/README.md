# R10 thermal-accessibility continuation

This package continues the existing family-A search in the saved P3/P4 coordinates. It adds a certified, channel-local F0 opening bound; preserves exact 300 K process surfaces; enforces positive raw-domain endpoints on prospective rows; and records complete-state thermal and fatigue evidence without changing the barrier, event, or mechanics laws.

Current selections are `R10T_PEAK_P40_G2600_S0P25GPA_P4_LOW_TAYLOR_V1` (resolved Peak-T and two-seed fatigue pass), `R10T_DBTT_P40_G3075_S0P25GPA_P4_EP400_PP040_V1` (rising DBTT with increasing hot current-state contribution and exact-contract fatigue reuse), and `R10T_WEAKT_P40_G1962_S4P5GPA_P4_EP055_PP010_V1` (bounded compensated Weak-T fracture response). Weak-T fatigue remains unresolved because the mean-cycle accelerator failed its state-overlap check; this is a numerical censor, not a material failure. The ceramic rows remain controls without a Paris requirement.

The selected DBTT row was passed through a narrow exact-row spatial binding that retains the qualified non-fracture owners; its current status is `PASS_EXACT_COMPLETE_ROW_CURRENT_BACKEND`. The sealed historical same-row controls are also copied under `spatial/`.

Machine-readable dispositions are in `results/thermal_accessibility_result.json`. The result schema is `r10.thermal-accessibility-result/1`.
