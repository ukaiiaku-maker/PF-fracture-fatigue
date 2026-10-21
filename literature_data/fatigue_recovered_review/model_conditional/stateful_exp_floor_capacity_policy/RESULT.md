# QSTE340TM capacity-policy joint refit

Date: 21 October 2026

## Decision

With birth-phase-resolved retention, the primary **uncapped** joint fit converged at 0.10072 decades. Fixed 10,000 converged at 0.10108; fixed 1,000 at 0.18220. The historical 115.338-unit cap remains an unchanged control and is not the default.

| R | RMSE (decades) | observed slope | predicted slope |
|---:|---:|---:|---:|
| 0.1 | 0.1194 | 3.709 | 3.404 |
| 0.3 | 0.1107 | 3.378 | 3.793 |
| 0.5 | 0.0762 | 3.762 | 3.526 |
| 0.7 | 0.0637 | 2.920 | 1.969 |

The uncapped maximum retained count is 701.47, total density 7.0647e+14 m^-2, maximum backstress 3.164 GPa, and minimum raw opening barrier 16.561 kBT. No zero-barrier clipping is used.

## Retention timing

The native full-cycle approximation undercounted phase-resolved surviving births by 6.7x to 4.8e3x at audited states. The final selected fits use explicit birth-phase survival and preserve emission, Peierls, Taylor, recovery, backstress, blunting, stored-energy, and opening physics. Observation-interval refinement from 1 to 8 substeps changed legacy-mode RMSE by less than 0.0002 decades; assigning all pre-measurement cycles to the first measured load worsened it to about 0.189 decades.

## Complete-R holdouts (fixed-1,000 converged reference)

| held R | training RMSE | held-out RMSE | optimizer converged |
|---:|---:|---:|:---:|
| 0.1 | 0.1481 | 0.3056 | False |
| 0.3 | 0.1583 | 0.1767 | True |
| 0.5 | 0.1380 | 0.3726 | True |
| 0.7 | 0.1612 | 0.2372 | True |

R=0.5 remains the principal transfer weakness. The R=0.1 holdout is an optimizer-limited bound.

## Limits

Frequency is assumed, not measured: same-parameter checks at 1 and 20 Hz materially worsen RMSE, so absolute kinetic parameters remain frequency-conditional. The dataset is isothermal at 298.15 K; thermal coefficients are retained but not identified. The primary fit reaches the profiled initial-retained upper bound (500), so pre-measurement state is not identified. Its same-parameter frozen-state RMSE is 0.824 decades versus 0.101 full-state, confirming consequential state evolution. The released-curvature sensitivity reached 0.14880 decades but did not converge and is not selected. Overload and closure histories were not fitted.
