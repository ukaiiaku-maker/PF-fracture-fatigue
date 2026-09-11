# Bounded cooperative-renewal contract audit

The previous calculation used generic m=3 and tau=1e-6 s. It is preserved byte-for-byte as GENERIC_MONOTONIC_RENEWAL_SCREEN. The 42 prospective fatigue launch records also used these defaults (this does not establish the historical A_NATIVE anchor settings); there is no established intentional monotonic-versus-fatigue distinction. The exact row renewal fields were not applied by that launcher.

The new analytical calculation reads m=3.2732414351776242 and tau=6.992153587194454e-7 s from every complete row. Only these two numerical inputs differ; barriers, conditional F1 reduction, and prospective gates remain unchanged. This corrects renewal provenance, not the completed physical trajectories. F2 remains unavailable.

All four rows, 37 temperatures, three ramp rates, and all three thresholds were evaluated: 1,332 conditions. The table uses the reference ramp 0.005 MPa sqrt(m)/s; all KFP values are in MPa sqrt(m). Accessible intervals describe sampled 25 K grid points, not localized transition temperatures. The available tier can vary by condition; the tables identify every F0-only substitution.

| Row | Threshold | KFP 300 K | KFP 1200 K | Accessible intervals (K) |
|---|---|---:|---:|---|
| A_NATIVE | SEED_1720 | 10.44186059 | 0.003673924904 | [[300.0, 1100.0]] |
| A_NATIVE | MEDIAN_LN2 | 10.7588778 | 0.03391208716 | [[300.0, 1125.0]] |
| A_NATIVE | UNIT_ACTION | 10.81711653 | 0.04892461946 | [[300.0, 1125.0]] |
| P25_TRANSFER_V1_RANK1 | SEED_1720 | 0.0009892767656 | 2.625314553e-10 | [] |
| P25_TRANSFER_V1_RANK1 | MEDIAN_LN2 | 0.009131222237 | 2.423295773e-09 | [] |
| P25_TRANSFER_V1_RANK1 | UNIT_ACTION | 0.01317323531 | 3.496076794e-09 | [] |
| P40_TRANSFER_CALIBRATED_GEN2 | SEED_1720 | 0.00504673935 | 2.625314553e-10 | [] |
| P40_TRANSFER_CALIBRATED_GEN2 | MEDIAN_LN2 | 0.04656883173 | 2.423295773e-09 | [] |
| P40_TRANSFER_CALIBRATED_GEN2 | UNIT_ACTION | 0.06716646382 | 3.496076794e-09 | [] |
| P55_TRANSFER_V1_RANK1 | SEED_1720 | 0.02717525829 | 2.625314553e-10 | [] |
| P55_TRANSFER_V1_RANK1 | MEDIAN_LN2 | 0.2495755529 | 2.423295773e-09 | [] |
| P55_TRANSFER_V1_RANK1 | UNIT_ACTION | 0.3585145392 | 3.496076794e-09 | [] |

Threshold actions are Xi=0.07509316036236147 (frozen seed 1720), ln(2)=0.6931471805599453, and 1. The renewal law is Lambda=P(m_c,lambda_raw*tau_c)/tau_c with the noninteger row m_c preserved.

Direct old/new comparison at 300 K, reference ramp, frozen Xi (available F0/F1 tier):

| Row | Generic KFP | Row-renewal KFP | Corrected/generic |
|---|---:|---:|---:|
| A_NATIVE | 10.10084721 | 10.44186059 | 1.03376 |
| P25_TRANSFER_V1_RANK1 | 0.0001155727876 | 0.0009892767656 | 8.55977 |
| P40_TRANSFER_CALIBRATED_GEN2 | 0.0005131906114 | 0.00504673935 | 9.83404 |
| P55_TRANSFER_V1_RANK1 | 0.002397223641 | 0.02717525829 | 11.3361 |

The [complete comparison](generic_vs_row_renewal_comparison.csv) includes all frozen-threshold roots, AK, AT, statuses and accessibility classifications. [Threshold robustness](threshold_robustness.csv) includes all nine threshold/rate combinations for every row. P25/P40 retain their fatigue-control roles, and P55 its boundary/falsification role; none becomes a qualified DBTT-, Peak-T-, weak-T-, or ceramic-like fracture archetype.

A_NATIVE is reported separately as an accessible low-temperature branch with high-temperature loss of accessibility. Its unchanged full-interval gate is not used to erase the low-temperature branch. P25/P40/P55 are zero-load dominated already at 300 K.

The saturated limit is KFP=Xi*Kdot*tau. The corrected plateau is 0.6992153587194454 times the old generic plateau at the same threshold and rate; neither is a numerical lower bound. Fixed-load D1/D2/DT surfaces are unchanged, but action-weighted descriptors, AK, AT and first-passage derivatives were recomputed.

F1 admitted: 539; F1 unavailable: 793. Unavailable F1/F2 states are not represented as full-state predictions. No physical trajectory was launched.

The deterministic analytical budget rejected 19 F1 conditions at 100,000 state evaluations per solve. These are included in the unavailable count. No accuracy or classification tolerance was relaxed; the stopped pre-budget analytical attempt and all completed condition records are retained under runs/row_renewal_monotonic_forward_v1.
