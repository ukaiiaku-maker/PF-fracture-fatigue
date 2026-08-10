# v10.2.30 four-class fatigue driving-force ladder

## Scope and provenance

This campaign maps fixed-local-`DeltaK` stochastic Arrhenius fatigue through the measured high-rate endpoint for Peak, DBTT, Weak-T, and ceramic-like parameterizations. No material row, hazard, threshold distribution, event-length law, MPZ physics, energy gate, DMD tolerance, or deterministic/athermal fracture rule was changed. The common conditions are `R=0.1`, 1000 Hz, 300 K, `theta=30 deg`, and `da_phys=5 um`; the target is approximately 100 um projected extension and `1e12` cycles remains only a censor.

The final consolidated artifacts are under `runs/v10_2_30_four_class_rate_endpoint_final_20260810`. They contain 54 run records, 818 committed event intervals, 46 completed trajectories, five physical censors, and no physical numerical failure. Duplicate historical points are resolved by class and normalized driving fraction in `ladder_analysis/four_class_driving_force_ladder.{csv,json}`.

High-rate production roots:

- `runs/v10_2_30_weakt_ceramic_high_rate_7e12237_20260810`
- `runs/v10_2_30_weakt_ceramic_adaptive_high_rate_320c498_20260810`
- `runs/v10_2_30_weakt_ceramic_target_refinement_c33a95e_20260810`
- `runs/v10_2_30_peak_dbtt_target_extension_a173419_20260810`
- `runs/v10_2_30_peak_target_refinement_4ad7dee_20260810_run3`

Two earlier Peak refinement roots are zero-cycle launch diagnostics: one was denied `/dev/fd` process substitution by the sandbox and one used the incompatible base Python/SciPy. Neither sampled or consumed a production RNG stream or committed physical state. Run 3 is authoritative.

## Measured rate endpoint

Rates are developed projected `da/dN` in m/cycle. Every endpoint row reached the physical extension target using the canonical seed and complete stochastic/geometry transactions.

| class | measured lower `(f, rate)` | measured upper `(f, rate)` | nearest f | DeltaK | cycles first event | cycles target | events | projected/path um |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Peak | 1.135, 9.5754e-6 | 1.140, 1.11469e-5 | 1.135 | 24.1636 | 0.002896 | 10.2725 | 18 | 102.668 / 107.778 |
| DBTT | 1.100, 6.22744e-6 | 1.105, 1.23468e-5 | 1.105 | 23.2330 | 0.008841 | 8.08341 | 15 | 101.976 / 108.587 |
| Weak-T | 1.145, 8.01347e-6 | 1.150, 1.08769e-5 | 1.150 | 14.6084 | 0.008147 | 8.92066 | 16 | 111.501 / 119.405 |
| ceramic-like | 1.200, 8.73812e-6 | 1.205, 1.09372e-5 | 1.205 | 14.7727 | 0.985238 | 9.17442 | 20 | 108.149 / 121.110 |

These are actual neighboring trajectories, not extrapolated intercepts. All four endpoints occur after the event-spacing distribution has crossed into near-monotonic cyclic failure: the median spacing is below one cycle and at least half of intervals are sub-cycle. They still contain multiple independent exponential renewals and energy-gated geometry commits, so they are stochastic cyclic trajectories rather than an inserted monotonic fracture rule.

## Event spacing, stability, and scatter

| class / nearest endpoint | median spacing cycles | sub-cycle fraction | late/early | mean tortuosity | event-rate CV | locator max trials | stability |
|---|---:|---:|---:|---:|---:|---:|---|
| Peak f1.135 | 0.06478 | 0.778 | 0.971 | 1.0549 | 3.61 | 95 | stable |
| DBTT f1.105 | 0.03699 | 0.800 | 1.037 | 1.0612 | 0.965 | 49 | stable |
| Weak-T f1.150 | 0.00985 | 0.750 | 2.232 | 1.0577 | 3.49 | 96 | not stable by existing criterion |
| ceramic f1.205 | 0.03525 | 0.750 | 1.150 | 1.1656 | 1.87 | 94 | stable |

The complete CSV records mean/median/min/max event spacing, sub-cycle fraction, event-rate min/max/CV, late/early ratio, tortuosity, four 25 um moving-window rates, cycles to first event/target, extensions, restart generation, and locator maximum. Weak-T is retained honestly as non-stable because its existing late/early criterion is not met; it is nevertheless a complete measured rate endpoint.

## Regime transitions

Regimes are assigned from developed rate together with the event-spacing distribution. Near-monotonic status specifically requires both a median interval below one cycle and at least 50% sub-cycle intervals; total life alone is not used.

- Peak leaves HCF after f0.950, passes through accelerated fatigue at f0.975-f1.025, enters LCF at f1.050, and crosses from LCF to near-monotonic between f1.100 and f1.135.
- DBTT leaves HCF after f0.975, passes through accelerated fatigue at f1.000-f1.025, enters LCF at f1.050, and crosses to near-monotonic between f1.050 and f1.100.
- Weak-T leaves HCF after f0.950, passes through accelerated fatigue at f0.975-f1.025, enters LCF at f1.050, and crosses to near-monotonic between f1.100 and f1.145.
- Ceramic-like leaves HCF after f0.950, remains accelerated through f1.050, enters LCF by f1.100, and crosses to near-monotonic between f1.150 and f1.200.

Thus the requested `~1e-5 m/cycle` endpoint is physically meaningful as the measured high-rate boundary of the stochastic cyclic model, but it lies on the near-monotonic side of the event-spacing crossover for every class. It should not be presented as a long-life fatigue operating point.

## Numerical qualification

All current high-rate endpoint localizers remained within the `<100` candidate-evaluation gate (maximum 96). No case returned to repeated `1e-6`-cycle near-threshold marching. Weak-T f1.150 exposed a real final-waveform bracket defect after event 7; the physics-neutral fix commits the bracket's low prefix with the exact phase-resolved integrator before localizing the final cycle. Restart from the authoritative event-7 checkpoint preserved threshold, RNG, MPZ, geometry, cycles, and event history and completed the identical trajectory.

## Artifacts

The final directory contains:

- `four_class_fatigue_summary.json`, `four_class_fatigue_cases.csv`, and `four_class_event_intervals.csv`
- `ladder_analysis/four_class_driving_force_ladder.json` and `.csv`
- developed `da/dN` versus `DeltaK` and normalized `f`
- cycles-to-first-event and cycles-to-target versus both `f` and `DeltaK`
- event spacing versus `DeltaK`
- the four-class regime map
- a `1e-8` to `1e-4 m/cycle` high-rate zoom

All supervisors terminated normally, no worker remains live, disk stayed above the 10 GiB launch reserve, and no queued matrix was restarted.

## Extension to `da/dN ~ 1e-3 m/cycle`

The fixed-local-`DeltaK` ladder was subsequently extended without changing any physics or parameterization. The computational domain configured by this model is `2 mm x 4 mm`, rather than 1 cm, so the qualified ~100 um developed trajectories were retained; they remain comfortably inside the modeled specimen.

| class | closest measured f | DeltaK | developed da/dN | cycles target | events | projected/path um | measured neighbor |
|---|---:|---:|---:|---:|---:|---:|---|
| Peak | 1.175 | 25.0152 | 1.23793e-3 | 0.07893 | 18 | 102.668 / 107.778 | f1.180, 1.53938e-3 |
| DBTT | 1.125 | 23.6535 | 1.01319e-3 | 0.09989 | 15 | 101.976 / 108.587 | f1.128, 1.27269e-3 |
| Weak-T | 1.185 | 15.0530 | 9.83017e-4 | 0.10286 | 16 | 111.501 / 119.405 | f1.1875, 1.08987e-3 |
| ceramic-like | 1.264 | 15.4960 | 1.09493e-3 | 0.09589 | 20 | 108.149 / 121.110 | f1.250, 7.00857e-4 |

Peak and DBTT were refined below an initial above-target pair; Weak-T and ceramic-like have measured below/above brackets. Every listed trajectory reached the extension target, retained multiple stochastic renewals and energy-gated geometry commits, and passed the existing developed-growth stability criterion. At these rates all four are firmly in the near-monotonic cyclic regime: the values are useful as the upper measured boundary of this stochastic cyclic model, not as long-life fatigue points.

Production roots:

- `runs/v10_2_30_four_class_1e3_rate_4146c9e_20260810`
- `runs/v10_2_30_four_class_1e3_rate_refinement_eb6d7ae_20260810`
- `runs/v10_2_30_ceramic_1e3_rate_final_90e2065_20260810`

The consolidated 68-point machine-readable dataset and regenerated plots are under `runs/v10_2_30_four_class_1e3_rate_complete_20260810`.
