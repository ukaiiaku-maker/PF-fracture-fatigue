# B1 Exploratory Sweep Postmortem

Sweep records SHA-256: `0ef9f564471828782749d186a9647896e6c3e1c755bbb3391b5bcc4658e60b7b` (24 records).

## Classifications

- `B1_EXPLORATORY_SWEEP_V1_COMPLETE`
- `B1_NUMERICALLY_UNRESOLVED_AT_K12`
- `B1_CONVERGED_BUT_RATE_TRANSFER_FAILED_AT_K15_K18_K24`
- `B1_TRANSFER_NOT_QUALIFIED_FOR_CANDIDATE_SELECTION`
- `A0_RATE_EQUIVALENT_RADIUS_NONPHYSICAL_OR_UNDEFINED`

## Same-seed transfer (seed 1720 B1 vs seed 1720 physical, 160-sample level)

| Kmax | B1 da/dN | physical da/dN | phys/B1 ratio | log10 err (dec) | B1 dr (m) | phys dr (m) | phys/B1 dr |
|---|---|---|---|---|---|---|---|
| 12 | nan | 2.4482e-08 | NaN | NaN | 4.3661e-09 | 1.0532e-08 | 2.412 |
| 15 | 3.1599e-08 | 3.0035e-07 | 9.505 | 0.9780 | 8.2276e-09 | 4.1289e-08 | 5.018 |
| 18 | 5.3519e-08 | 4.4734e-07 | 8.359 | 0.9221 | 1.7819e-08 | 1.0047e-07 | 5.639 |
| 24 | 6.1626e-08 | 5.5806e-07 | 9.056 | 0.9569 | 4.8029e-08 | 3.4316e-07 | 7.145 |

## Adjacent log-log slopes

| interval | physical | B1 same-seed | error |
|---|---|---|---|
| m_12_15 | 11.2350 | NaN | NaN |
| m_15_18 | 2.1850 | 2.8900 | +0.7050 |
| m_18_24 | 0.7687 | 0.4903 | -0.2784 |

## K=12

- cycles = 1000000 (== maximum_cycles = 1,000,000)
- events reached = 86, sampled = 56
- `fixed_point_converged = false` -> `da_dN = NaN`
- **NaN is maximum-cycle exhaustion. It is not zero, not a physical censor, and not a converged rate.**

## Radius-only impossibility

Increasing r_eff lowers the opening stress sigma = (K - K_shield)/sqrt(2*pi*r), which lowers the predicted rate. B1 already predicts rates ~8-10x BELOW the physical rates at K=15/18/24. Scaling B1's blunting increment up to match the larger physical radius therefore moves the predicted rate further DOWN, away from the physical rate. A one-scalar radius-increment correction cannot repair the rate transfer: the missing transfer is not primarily a tip-radius amplitude correction.

- alpha_r (same-seed, K=18, radius only) = 5.6386
- alpha_r (three-seed mean, K=18, radius only) = 5.4151
- Both are **diagnostic only** and never enter candidate scoring.

## A0 rate-equivalent radius

| Kmax | A0 ceiling (r->0) | physical | root? | root (m) | classification |
|---|---|---|---|---|---|
| 12 | 2.6986e-07 | 2.4482e-08 | True | 8.6087e-07 | `A0_RATE_EQUIVALENT_RADIUS_NONPHYSICAL_BELOW_R0` |
| 15 | 2.6986e-07 | 3.0035e-07 | False | n/a | `A0_RATE_EQUIVALENT_RADIUS_UNDEFINED_PHYSICAL_RATE_EXCEEDS_A0_CEILING` |
| 18 | 2.6986e-07 | 4.4734e-07 | False | n/a | `A0_RATE_EQUIVALENT_RADIUS_UNDEFINED_PHYSICAL_RATE_EXCEEDS_A0_CEILING` |
| 24 | 2.6986e-07 | 5.5806e-07 | False | n/a | `A0_RATE_EQUIVALENT_RADIUS_UNDEFINED_PHYSICAL_RATE_EXCEEDS_A0_CEILING` |

At K=15/18/24 the physical rate **exceeds the limiting A0 opening-only rate as r -> 0**, so no positive radius can reproduce it. This is a physical statement about the reduced model's ceiling, not a root-finder failure.

## Outstanding semantics gap

B1 r_eff_m is a mean post-event radius after burn-in; the physical table exposes a single tip_radius_m whose exact statistic (terminal vs developed-window mean) is not confirmed from a surviving producer definition. All radius comparisons here are therefore labeled semantics_matched=False.

