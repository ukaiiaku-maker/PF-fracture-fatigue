# Conditional frequency assumption and opening-reference result

Date: 20 October 2026

Temperature amendment: the user subsequently supplied the test condition as
25 °C (298.15 K). `room_temperature_binding.json` records that provenance and
converts the effective dimensionless barrier to eV without changing the fit.

## Decision

Use **10 Hz** as the provisional nominal fatigue frequency, with **1 Hz and
20 Hz** as sensitivity cases. This is an authorized modeling assumption, not
recovered source metadata. The existing cosine/sinusoidal `FatigueWaveform` is
also used provisionally and remains explicitly unresolved in the source record.

The frequency assumption is not the present scientific limiter. After refitting
the same effective opening reference, 1, 10, and 20 Hz give indistinguishable
fit error and almost identical shape parameters. Frequency primarily shifts the
dimensionless barrier floor:

| assumed frequency | fitted floor, G*/kBT | fitted exponent | fit RMSE, log10 rate |
|---:|---:|---:|---:|
| 1 Hz | 17.1972 | 0.2373 | 0.63244 |
| 10 Hz | 16.4871 | 0.2370 | 0.63244 |
| 20 Hz | 16.2724 | 0.2368 | 0.63245 |

Thus the predicted fitted curves are insensitive over this bracket, while the
reported barrier offset is conditional on frequency. The 1-to-10 Hz shift is
0.710 G*/kBT and the 10-to-20 Hz shift is 0.215 G*/kBT. These offsets must be
revisited if the source frequency is recovered.

## What was fitted

The calculation uses the existing:

- K/sqrt(2*pi*r0) local-drive map with r0 = 1 micrometre;
- cosine waveform and the measured DeltaK/R pairs;
- smooth normalized broad alternative-G* basis;
- cooperative completion law (`hits = 3.273241435`, `tau = 6.9921536e-7 s`);
- 5 micrometre renewal reward.

The fit is performed as an effective `G*/kBT` surface and then bound to the
user-supplied 298.15 K for the reported eV scale. Optimization uses 160 evenly
spaced, R-balanced support
points to avoid treating dense neighboring author-reduced points as independent;
all 760 points are evaluated. The objective includes both absolute log-rate and
within-R curve shape.

## Transfer result

The opening-only reference does **not** provide an adequate common fit across all
four R conditions:

| R | observed slope | predicted slope | log10-rate bias | RMSE |
|---:|---:|---:|---:|---:|
| 0.1 | 3.709 | 3.384 | -0.778 | 0.779 |
| 0.3 | 3.378 | 3.173 | -0.372 | 0.373 |
| 0.5 | 3.762 | 2.951 | +0.171 | 0.183 |
| 0.7 | 2.920 | 2.563 | +0.980 | 0.980 |

The sign-changing R bias means the single opening reference underpredicts the
low-R blocks and overpredicts the high-R blocks. Complete-R holdouts confirm the
transfer problem: held-out RMSE ranges from 0.249 to 1.360 log10 units, with the
R=0.7 block having a +1.360 log10 bias when excluded from calibration.

The fitted stress scale also reaches its declared lower search boundary. The
result is therefore a diagnostic boundary fit, not a promoted QSTE340TM barrier.
It must not be registered as a complete material or used to claim an identified
activation volume.

## Consequence

The 10 Hz assumption is adequate for continuing the analysis. The next limiter
is the stateful R dependence and the absence of a source-bound QSTE340TM
EXP_floor plasticity parameterization, not frequency. Native stateful execution
has not been launched with a synthetic tungsten plasticity vector. The overload
and closure histories remain reserved for validation after a steel-specific
plasticity prior/uncertainty model and physical closure observation map are
defined.

Machine-readable parameters, boundary flags, block metrics, holdouts, exact
predictions, and temperature provenance are in `conditional_opening_reference.json`,
`leave_one_R_out.csv`, `opening_reference_predictions.csv`, and
`room_temperature_binding.json`.
