# v10.2.30 fatigue driving-force ladder

## Scope and provenance

This campaign maps fixed-local-`DeltaK` stochastic Arrhenius fatigue from VHCF through HCF, accelerated fatigue, LCF, and the near-monotonic cyclic-failure crossover. No material row, Arrhenius hazard, stochastic threshold distribution, event-length law, MPZ physics, energy gate, DMD tolerance, or deterministic/athermal fracture rule was changed.

- Branch: `codex/v10.2.30-fatigue-da-dN`
- Final production HEAD: `f1858cc4776c7931cbec7c7bc4583a24f403fef8` (the report and analysis tooling are committed separately)
- Qualified environment: Python 3.12.13, NumPy 2.5.1, SciPy 1.18.0, Matplotlib 3.11.0
- Common loading: `R=0.1`, `frequency=1000 Hz`, `temperature=300 K`, `theta=30 deg`, `da_phys=5 um`
- Canonical seeds: Peak `1720`, DBTT `1001723`, weak-T `2001726`, ceramic `3001729`
- Target: approximately 100 um projected extension; `1e12` cycles is a censor, not a required endpoint

Primary production roots:

- `runs/v10_2_30_dense_deltaK_cc1bf6f_20260804` (598 MiB)
- `runs/v10_2_30_driving_force_ladder_15e2650_20260809` (141 MiB)
- `runs/v10_2_30_above_one_ladder_1127b7f_20260810_run3` (88 MiB)
- `runs/v10_2_30_f1p100_ladder_f1858cc_20260810` (42 MiB)
- consolidated analysis: `runs/v10_2_30_fatigue_driving_force_ladder_final_20260810`

The two earlier above-unity staging roots are diagnostic zero-cycle launch failures only: run 1 was denied `/dev/fd` process substitution by the sandbox and run 2 selected an incompatible base Python/SciPy. Neither created a checkpoint, sampled/consumed a production RNG stream, or changed physical state. Run 3 is authoritative.

## Peak and DBTT results

All dimensional stress-intensity quantities are in MPa sqrt(m), rates in m/cycle. `Kmin=0.1 Kmax` throughout.

| class | f | DeltaK | Kmax | cycles first event | cycles target | events | projected/path um | developed da/dN | locator max | regime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Peak | 0.875 | 18.6284 | 20.6982 | 8.896e10 | 2.654e13 | 18 | 102.668/107.778 | 3.682e-18 | n/a | VHCF |
| Peak | 0.900 | 19.1606 | 21.2895 | 2.897e9 | 8.642e11 | 18 | 102.668/107.778 | 1.131e-16 | 95 | HCF |
| Peak | 0.925 | 19.6928 | 21.8809 | 1.100e8 | 3.282e10 | 18 | 102.668/107.778 | 2.977e-15 | n/a | HCF |
| Peak | 0.950 | 20.2251 | 22.4723 | 4.849e6 | 1.446e9 | 18 | 102.668/107.778 | 6.756e-14 | 144* | ACCELERATED_FATIGUE |
| Peak | 0.975 | 20.7573 | 23.0637 | 2.468e5 | 7.362e7 | 18 | 102.668/107.778 | 1.327e-12 | 96 | ACCELERATED_FATIGUE |
| Peak | 1.000 | 21.2895 | 23.6551 | 1.445e4 | 4.311e6 | 18 | 102.668/107.778 | 2.267e-11 | 96 | LCF |
| Peak | 1.025 | 21.8218 | 24.2464 | 971.0 | 2.897e5 | 18 | 102.668/107.778 | 3.373e-10 | 96 | LCF |
| Peak | 1.050 | 22.3540 | 24.8378 | 75.0 | 2.235e4 | 18 | 102.668/107.778 | 4.372e-9 | 96 | LCF |
| Peak | 1.100 | 23.4185 | 26.0206 | 0.980 | 208.0 | 18 | 102.668/107.778 | 4.696e-7 | 96 | NEAR_MONOTONIC_CYCLIC_FAILURE |
| DBTT | 0.900 | 18.9228 | 21.0253 | 5.931e11 | 3.901e13 | 15 | 101.976/108.587 | 2.595e-18 | n/a | VHCF |
| DBTT | 0.925 | 19.4484 | 21.6093 | 1.162e10 | 7.642e11 | 15 | 101.976/108.587 | 1.325e-16 | 46 | HCF |
| DBTT | 0.950 | 19.9740 | 22.1934 | 2.433e8 | 1.600e10 | 15 | 101.976/108.587 | 6.325e-15 | 144* | HCF |
| DBTT | 0.975 | 20.4997 | 22.7774 | 5.525e6 | 3.634e8 | 15 | 101.976/108.587 | 2.786e-13 | 96 | ACCELERATED_FATIGUE |
| DBTT | 1.000 | 21.0253 | 23.3615 | 1.379e5 | 9.070e6 | 15 | 101.976/108.587 | 1.116e-11 | 96 | LCF |
| DBTT | 1.025 | 21.5509 | 23.9455 | 3.847e3 | 2.530e5 | 15 | 101.976/108.587 | 4.001e-10 | 96 | LCF |
| DBTT | 1.050 | 22.0766 | 24.5295 | 123.0 | 8.088e3 | 15 | 101.976/108.587 | 1.252e-8 | 96 | LCF |
| DBTT | 1.100 | 23.1278 | 25.6976 | 0.0149 | 16.02 | 15 | 101.976/108.587 | 6.227e-6 | 96 | NEAR_MONOTONIC_CYCLIC_FAILURE |

`*` The 144 maximum belongs to the pre-prefix-cache process image/early history. It was diagnosed as redundant evaluation of a `[1,2]` bracket, not recursive `1e-6`-cycle marching. The same copied checkpoint reproduced the identical event and RNG renewal with 96 evaluations after prefix reuse; subsequent current-HEAD DBTT f0.950 events used at most 95 evaluations. All f>=0.975 current-HEAD trajectories remain below the 100-evaluation production gate.

## Developed-growth stability and scatter

All new 100 um Peak/DBTT cases satisfy the existing stability rule (at least 10 events, at least 50 um developed extension, late/early cumulative-rate ratio between 0.5 and 2). Peak has 107.778 um final path length and mean tortuosity 1.0549; DBTT has 108.587 um and mean tortuosity 1.0612. The preserved common-RNG geometry sequence makes these values invariant across `f` within each class.

- Peak late/early ratios are approximately 0.892 from f0.95 through f1.05 and 0.901 at f1.10.
- DBTT late/early ratios are approximately 1.103 from f0.95 through f1.05 and 1.570 at f1.10.
- Event-rate CV is approximately 3.39 for Peak and 0.248 for DBTT through the cyclic ladder; it rises at f1.10 as the target is traversed within only a few cycles.
- Event-level rates, four common 25 um windows, tortuosity, and event ranges are in `ladder_analysis/peak_dbtt_driving_force_ladder.csv`.

Representative 25 um moving-window rates:

| case | 0-25 um | 25-50 um | 50-75 um | 75-100 um |
|---|---:|---:|---:|---:|
| Peak f0.975 | 1.570e-12 | 1.227e-12 | 1.473e-12 | 1.397e-12 |
| DBTT f0.975 | 2.530e-13 | 2.848e-13 | 2.883e-13 | 3.178e-13 |
| Peak f1.000 | 2.681e-11 | 2.095e-11 | 2.515e-11 | 2.386e-11 |
| DBTT f1.000 | 1.014e-11 | 1.141e-11 | 1.155e-11 | 1.273e-11 |

## Regime and crossover interpretation

- VHCF/low growth: f0.75 Peak/DBTT/weak-T/ceramic completed `1e12` cycles without an event; projected additional growth at the censor is therefore bounded at zero for those realized trajectories. DBTT f0.90 and Peak f0.875 reach the target only after more than `1e13` cycles.
- HCF: Peak f0.90-f0.925 and DBTT f0.925-f0.95 show many distinct stochastic waiting/reload intervals and target lives from `3e10` to `9e11` cycles.
- Accelerated fatigue: Peak f0.95-f0.975 and DBTT f0.975 traverse developed growth in `7e7`-`1e9` cycles.
- LCF: f1.00 remains unequivocally cyclic (Peak 4.31e6 cycles, DBTT 9.07e6); f1.025 and f1.05 retain finite, threshold-controlled event-to-event waits down to tens/hundreds of cycles.
- Near-monotonic cyclic failure: f1.10 is the crossover. Peak reaches 48.81 um by cycle 99.95 and the target by cycle 208; DBTT reaches 48.25 um by cycle 8.01 and the target by cycle 16.02. Multiple stochastic renewals and energy-gated transactions remain present, but reload intervals are no longer a meaningful long-life fatigue trajectory.

Thus the approximate brackets are:

- Peak `f_HCF->LCF`: between f0.975 and f1.000; `f_LCF->near-monotonic`: between f1.050 and f1.100.
- DBTT `f_HCF->LCF`: between f0.975 and f1.000; `f_LCF->near-monotonic`: between f1.050 and f1.100.

No f above 1.10 is warranted.

## Descriptive power-law check

Local log-log fits over the developed f0.875-f0.975 subset give:

- Peak: `m=118.26`, `R^2=0.99987`, DeltaK 18.628-20.757 MPa sqrt(m), five points.
- DBTT: `m=144.74`, `R^2=0.99999`, DeltaK 18.923-20.500 MPa sqrt(m), four points.

These very large exponents describe a narrow Arrhenius transition and must not be interpreted as a global Paris law. No Paris relation was used in the simulation. VHCF censors, f1 LCF, and f1.10 near-monotonic points were excluded.

## Existing-case inventory and disposition

The recovered Weak-T f0.900 case reached its physical target at `5.382637251e9` cycles with 16 events, 111.501 um projected extension, 119.405 um path extension, and developed `da/dN=1.80361e-14` m/cycle. Its late/early rate ratio is 2.449, so it is correctly retained as not provisionally stable. The authoritative event-16 checkpoint and developed-growth output were complete before the wrapper later returned 124 during terminal post-processing; the aggregate therefore records `completed` with diagnostic reason `post_target_wrapper_exit_124`, rather than misclassifying a complete physical trajectory as failed.

- `COMPLETE_TARGET_GROWTH`: Peak f0.875-1.10 (where run), DBTT f0.900-1.10, ceramic f0.875/f0.900/f0.925, weak-T f0.850/f0.875/f0.925, plus Weak-T f0.900 after the final validated restart recorded below.
- `COMPLETE_1E12` / `CENSORED_PHYSICAL`: ceramic f0.850 (13 events, 57.255 um, developed rate `5.818e-17`), and the f0.750 no-event cases.
- `VHCF_LOW_GROWTH`: f0.750 cases; no additional compute is justified by their realized zero-event `1e12` bound.
- `PENDING_FRESH`: Peak/DBTT/weak-T f0.775, f0.800, f0.825 and excluded lower ceramic rows. These are lower priority than the now-resolved Peak/DBTT transition and are not launched automatically.
- Historical f0.850/f0.875 Peak/DBTT partial `1e14` outputs are censored/undeveloped where a later higher-f completed trajectory provides the useful curve point; they are preserved, not overwritten.
- No authoritative physical case is `FAILED_NUMERICAL` or `FAILED_RESTART`. Zero-cycle launch-environment diagnostics are not physical trajectories.

## Figures and machine-readable data

- `ladder_analysis/developed_da_dN_vs_f.png`
- `ladder_analysis/developed_da_dN_vs_deltaK.png`
- `ladder_analysis/cycles_to_first_event_vs_f.png`
- `ladder_analysis/cycles_to_target_vs_f.png`
- `ladder_analysis/peak_dbtt_driving_force_ladder.csv`
- `ladder_analysis/peak_dbtt_driving_force_ladder.json`
- `four_class_fatigue_cases.csv` and `four_class_event_intervals.csv`

## Terminal state

All Peak/DBTT ladder supervisors terminated atomically with completed case status, no live PID, and no stale `qualification_supervisor.lock.json`, `launcher.json`, or `active_workers.json`. Disk use stayed above the 10 GiB reserve. The full queued matrix was not restarted, and no branch was pushed.
