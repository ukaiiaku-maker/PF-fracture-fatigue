# QSTE340TM corrected continuous-birth joint refit

Date: 21 October 2026

## Decision

The within-panel birth-survival defect is corrected and numerically qualified.
For frozen source and escape coefficients the implementation evaluates

`a_j * [-expm1(-E_j)/E_j] * exp(-Hafter_j)`

with its stable zero-escape limit. The earlier `1bb0928e` full-current-panel
exposure result and exports remain unchanged as historical controls.

The corrected uncapped joint refit is a **bounded negative for curve shape**, not
an improved QSTE340TM calibration. It converged at 0.436238 log10-rate RMSE, but
its four global within-R slopes are 0.933, 0.366, 0.142 and 0.153, versus measured
3.709, 3.378, 3.762 and 2.920. The common-axis and local-slope plots show a sharp
onset followed by an unphysical plateau. This result must not replace the
historical fit as accepted experimental calibration evidence.

## Selected corrected model-A vector

The selection is the simpler single smooth-broad opening basis with the original
shared EXP_floor exponent `n=1`. It is selected as the most qualified corrected
reference, not because it fits the experimental shape.

| parameter | value |
|---|---:|
| opening floor | 19.9141961883 kBT |
| opening amplitude | 13.7443005708 kBT |
| opening stress scale | 0.1154922871 GPa |
| opening exponent | 4.4558876617 |
| emission G0 at 298.15 K | 2.6626382073 eV |
| shared plastic stress scale | 0.1000000000 GPa |
| escape G0 at 298.15 K | 2.1473665256 eV |
| direct opening-backstress fraction | 0.0634665238 |
| recovery rate | 0.0152837134 s^-1 |
| common initial retained count | 391.6543276468 |

Several values are at or close to search bounds, most visibly both stress scales.
The maximum retained count is 481.91, total density is 4.8691e14 m^-2, maximum
source backstress is 2.174 GPa, and the minimum raw opening barrier remains
positive at 17.695 kBT. Same-parameter frozen-state RMSE is 0.76971, versus
0.43624 with state evolution, so state feedback is consequential even though its
predicted curve shape is inadequate.

## Numerical qualification

- Waveform refinement from 32 to 512 panels changes RMSE from 0.4362382 to
  0.4362053 decades.
- At 128 waveform panels, outer-history refinement from 1 to 8 substeps changes
  RMSE from 0.4362072 to 0.4361801 decades.
- Tests cover exact constant coefficients, zero limits, positive survivor bounds,
  arbitrary phase subdivision, late births in a high-escape panel, exact state
  balance, and event-free restart/continuation.
- The import verifier checks 11 source files, 7,732 paired records and 23,207
  cell/conversion assertions. The corrected export has all 760 chronological rows.

## Nested comparisons

The additive two-scale smooth-broad opening comparison converged to RMSE
0.43623817. Its secondary amplitude shrank to 0.06895 kBT and did not change the
slopes or scientific conclusion.

Releasing one shared EXP_floor exponent lowered RMSE to 0.432649 at `n=3.1103`,
but predicted slopes remained only 0.684, 0.516, 0.288 and 0.785. It drove the
emission energy, opening exponent, and plastic stress scale close to bounds and
raised maximum density to 1.537e15 m^-2. It is retained as a sensitivity, not
selected.

## Initial state and transfer

The fixed-common-initial-state profile has a broad optimizer-dependent valley:
RMSE is 0.43619 at 400, 0.43629 at 1,000, and 0.43507 at 2,000 retained units.
The corresponding stored-energy barrier offsets are -0.04684, -0.11711, and
-0.23421 eV. Only the 1,000-unit profile converged within its 90-evaluation
budget; the remaining profile points are explicitly optimizer-limited. The
initial state and barrier level are therefore not uniquely identified.

Fresh training-only complete-R holdouts give held-out RMSEs of 0.647, 0.405,
0.294 and 0.816 decades for R=0.1, 0.3, 0.5 and 0.7. All four reached their
50-evaluation qualification budget; they are optimizer-limited transfer bounds.
The R=0.7 bias is +0.800 decades and R=0.1 bias is -0.588 decades, confirming
poor cross-R transfer.

## Scope and files

Temperature is 298.15 K from the user's room-temperature clarification.
Frequency remains the provisional 10 Hz assumption. These isothermal data do not
identify new thermal parameters. No mechanics, event reward, source geometry,
temperature law, empirical R correction, per-R initial state, or retained-count
cap was introduced.

- `fit_uncapped_continuous_birth.json`: selected vector, full/frozen states and
  phase/history convergence.
- `predictions_uncapped_continuous_birth.csv`: all 760 observations and channels.
- `fit_uncapped_two_scale_opening.json` and
  `fit_uncapped_exp_floor_curvature.json`: nested sensitivities.
- `initial_state_profile.json` and `complete_R_holdouts.json`: validation.
- `common_axis_data_fit.*`, `shape_and_residual_diagnostics.*`,
  `Gstar_vs_stress.*`, `retained_count_density_histories.png`, and
  `separate_process_rates.png`: figures.

The source data SHA-256 is
`99f869f5fb4963d37791aceff144c56150fd27ab51c59735588446efba4e36cd`.
Evaluator and analysis source hashes are recorded by Git in the preservation
commit; `analysis_summary.json` also binds the selected fit and source dataset.
