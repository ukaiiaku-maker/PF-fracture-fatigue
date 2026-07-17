# v10.1.7.2 deterministic versus stochastic cleavage-hazard pilot

## Scope

This branch is based directly on `v10.1.7.1-final-production-temperature-sweep`.
It preserves the finalized campaign source budget, local Taylor back stress,
active shielding, blunting, Peierls--Taylor transport, source refresh, material
barriers, mesh, and loading protocol.

The only optional change is the integrated cleavage-hazard threshold.

## Formulation

The deterministic model completes every renewal at integrated physical hazard

\[
H=\int \lambda_c\,dt=1.
\]

The stochastic mode draws after every completed event

\[
\Xi=-\ln U,\qquad U\sim\mathcal U(0,1),
\]

so that `Xi` is exponential with unit mean.  The normalized checkpoint progress
is

\[
B=H/\Xi,
\]

and therefore evolves at

\[
\dot B=\lambda_c/\Xi.
\]

Each renewal still advances the fixed checkpoint length.  Its stochastic waiting
time is `Xi/lambda_c`; the long-time mean event rate remains `lambda_c`.  The
deterministic control is exactly `Xi=1`.

No random perturbation is applied to K, J, stress, barriers, source capacity,
back stress, shielding, blunting, or the reported R-curve.

## Pilot

The default pilot uses:

- material class: DBTT;
- temperature: 700 K;
- deterministic control: one run;
- stochastic ensemble: seeds 1 through 10;
- target crack extension: 200 micrometres;
- checkpoint length: 5 micrometres;
- branching: disabled;
- wake shielding: disabled;
- all campaign scales fixed at one.

Run with:

```bash
OUTROOT=runs/v10_1_7_2_stochastic_hazard_DBTT_700K_200um_v1 \
SEEDS="1 2 3 4 5 6 7 8 9 10" \
TARGET_EXT_UM=200 \
bash scripts/run_v10_1_7_2_stochastic_hazard_pilot.sh
```

## Outputs

- `stochastic_hazard_case_summary.csv`
- `stochastic_hazard_event_curves.json`
- `stochastic_hazard_assessment.json`
- `stochastic_hazard_R_curve_ensemble.png`
- `stochastic_hazard_threshold_distribution.png`
- `stochastic_hazard_initiation_scatter.png`

## Acceptance

The pilot passes only if:

1. the stochastic ensemble mean differs from the deterministic curve by no more
   than 5 percent;
2. the 10--90 percent band is nonzero;
3. the seed trajectories are not identical;
4. repeated use of the same seed is exactly reproducible;
5. the sampled integrated-hazard thresholds are positive and statistically
   consistent with a unit-mean exponential distribution.

A visually noisy result with a biased ensemble mean is a failure, not an
improvement.

## Limitations

The pilot is monotonic, single-front, and branch-free.  Branch RNG partitioning,
restart serialization of RNG state, and cyclic stochastic hazard integration
must be audited before this option is promoted to fatigue or branching runs.
The deterministic v10.1.7.1 production results remain the reference until the
pilot passes.
