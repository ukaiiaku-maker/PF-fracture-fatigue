# Prospective Paris-slope candidate campaign

The original P40 row transferred sensitivity-predicted rates but missed the target local-slope profile. Its row and original predictions remain unchanged. One analysis-only transfer update was frozen before deriving the three new single-EXP candidates.

| Target | Exact candidate | Classification | Global slope |
|---|---|---|---|
| P40 | P40_TRANSFER_CALIBRATED_GEN2 | EFFECTIVE_GLOBAL_SLOPE_ONLY | 3.69873195352322 |
| P25 | P25_TRANSFER_V1_RANK1 | EFFECTIVE_GLOBAL_SLOPE_ONLY | 2.4715673454590834 |
| P55 | P55_TRANSFER_V1_RANK1 | TARGET_NOT_TRANSFERRED_WITH_SINGLE_EXP_FLOOR | not qualified |

Only the separately recorded retained rows are admitted as fatigue-response controls. The monotonic side check reveals substantial side effects and does not establish material archetypes.

All raw trajectories remain under runs/. Each launch was fresh, committed, hash-qualified, and capped by the original 1e12-cycle censor. No trajectory was resumed.

The first 20 µm are excluded using whole events. Local slopes, interval rates, event-size/waiting decomposition, seed transfer, and full nominal ΔK are available in the accompanying tables. No closure-corrected ΔK_eff is reported.

Window diagnostics (P55 is a three-point pilot only):

| Target | Target fit | Predicted fit | Physical fit | Physical local range | RMS prediction residual (decade) |
|---|---:|---:|---:|---|---:|
| P40 | 3.936529 | 3.936589 | 3.698732 | 2.818142–4.204155 | 0.028671 |
| P25 | 2.460330 | 2.460365 | 2.471567 | 1.944638–2.861646 | 0.011955 |
| P55 | 5.401636 | 5.402084 | 4.946011 | 4.000011–5.360128 | 0.068003 |

Second-seed comparisons use the same Kmax = 15, 18, 21 subset for both seeds.

| Candidate | Seed 1720 slope | Seed 1001723 slope | Median absolute prediction residual | Passed |
|---|---:|---:|---:|---|
| P40_TRANSFER_CALIBRATED_GEN2 | 3.5193164681573235 | 3.538695236009793 | 0.012370103954394831 | True |
| P25_TRANSFER_V1_RANK1 | 2.3453343947413186 | 2.3572093435850916 | 0.01747786826184503 | True |

R-transfer results hold every material field fixed. Full applied DeltaK is (1−R)Kmax.

| Candidate | R | Kmax | Full DeltaK | Physical da/dN (m/cycle) |
|---|---:|---:|---:|---:|
| P40_TRANSFER_CALIBRATED_GEN2 | -0.95 | 15.0 | 29.25 | 1.4322919131928629e-07 |
| P40_TRANSFER_CALIBRATED_GEN2 | 0.5 | 15.0 | 7.5 | 3.081733559313685e-07 |
| P40_TRANSFER_CALIBRATED_GEN2 | -0.95 | 18.0 | 35.1 | 2.9261681864032724e-07 |
| P40_TRANSFER_CALIBRATED_GEN2 | 0.5 | 18.0 | 9.0 | 6.293549762711213e-07 |
| P40_TRANSFER_CALIBRATED_GEN2 | -0.95 | 21.0 | 40.949999999999996 | 4.6505043279673775e-07 |
| P40_TRANSFER_CALIBRATED_GEN2 | 0.5 | 21.0 | 10.5 | 9.96709612382511e-07 |
| P25_TRANSFER_V1_RANK1 | -0.95 | 15.0 | 29.25 | 1.8940225777112593e-07 |
| P25_TRANSFER_V1_RANK1 | 0.5 | 15.0 | 7.5 | 4.1405471218378447e-07 |
| P25_TRANSFER_V1_RANK1 | -0.95 | 18.0 | 35.1 | 3.0684345336896135e-07 |
| P25_TRANSFER_V1_RANK1 | 0.5 | 18.0 | 9.0 | 6.775440507499524e-07 |
| P25_TRANSFER_V1_RANK1 | -0.95 | 21.0 | 40.949999999999996 | 4.156332112061093e-07 |
| P25_TRANSFER_V1_RANK1 | 0.5 | 21.0 | 10.5 | 9.19476941726198e-07 |

Monotonic first-passage K values below are reduced no-feedback screening values, not state-resolved fracture toughness.

| Row | 300 K | 600 K | 900 K | 1200 K |
|---|---:|---:|---:|---:|
| A_NATIVE | 10.1981 | 6.93046 | 4.70762 | 0.00511085 |
| P40_TRANSFER_CALIBRATED_GEN2 | 0.00683396 | 5e-09 | 5e-09 | 5e-09 |
| P25_TRANSFER_V1_RANK1 | 0.00153906 | 5e-09 | 5e-09 | 5e-09 |
| P55_TRANSFER_V1_RANK1 | 0.0319185 | 5e-09 | 5e-09 | 5e-09 |

Exact retained rows are in final_candidate_parameter_rows.csv; all tested frozen rows remain in transfer_candidate_registry_v1.csv.
First-passage accounting: 738 passages, 737 committed geometry events, 1 consumed zero-length energy-gated attempts. These are physical non-advancing passages, not numerical exclusions or censored trajectories.
Complete event actions come from checked event transactions. The legacy block and summary increments are preserved but are not substituted for whole-event hazard action.
The qualified physical source snapshot is f2d692263518b64a9bbef5619fea9fe359dee037. Later producer commits change analysis only; exact source-tree equivalence is recorded in physical_source_equivalence.json.
Terminal verification command: `python scripts/verify_v10_2_30_prospective_campaign.py` using the qualified environment.

Authoritative physical run roots: runs/prospective_paris_p40_pilot_v1 and runs/prospective_paris_transfer_v1.
