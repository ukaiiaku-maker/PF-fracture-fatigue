# QSTE340TM active restored-model calibration

This directory is the executed continuation from commit `f91c2fd1`. It preserves
the previously completed active cache, the newer chronological calculations,
all restart roots, fitted opening receipts, tabular exports, and figures.

## Scientific assessment

The qualified fixed-local-window result is a bounded negative calibration
result with a real near-frontier improvement. The transport-dominated EXP_floor
material is the best absolute active result: pooled RMSE is 0.439765 decades,
versus 0.444883 for the pooled-optimal common constant and 0.445274 for the
sustained-active control. Its fitted slopes are 0.90, 0.84, 2.62, and 1.45,
versus observed 3.71, 3.38, 3.76, and 2.92. Its history biases remain -0.428,
-0.276, +0.182, and +0.523 decades. The same-parameter frozen RMSE is 1.90862,
so this material has a large, consequential state effect; that effect still
does not yield a satisfactory joint calibration. See
`active_calibration_assessment.json` for all metrics, controls, coordinates,
and claim limits.

The separately completed trapping-dominated material is an informative shape
frontier. Its fitted slopes are 2.46, 2.64, 3.10, and 3.43 and its centered
shape error is 0.10777 decades. It does not solve the joint problem: the four
history biases are -0.590, -0.304, +0.095, and +0.800 decades and its overall
shape-fit RMSE is 0.50648 decades. The common-axis figure shows both frontiers.

This is not a negative result for the missing full crack-translation problem.
The signed kernel is qualified through 1.175 mm, while the measured histories
extend by 6--13 mm. No kernel extrapolation, population reset, or unqualified
periodic remap was used.

## Important numerical decisions

- Experimental histories retain original row order and piecewise-linear
  DeltaK(N). Long intervals use midpoint quadrature and integer cycle
  boundaries with exact observed endpoints.
- The native high-cycle wall/work limits are scheduling intervals. Outer and
  native checkpoints preserve the experimental cursor and complete state.
- A zero-opening, state-only observer is locally one-cycle exact, but a common
  two-support comparison showed a consequential long-projective difference in
  mobile content. It is diagnostic-only and is not used for production claims.
- Production continuation uses a real fitted opening surface as the DMD hazard
  observer seed. Candidate-specific opening coordinates are refit afterward;
  final receipts are captured only after installing those surfaces.

## Main files

- `active_calibration_assessment.json`: machine-readable scientific decision.
- `active_cache_provenance.json`: exact reconciliation of the completed and
  censored legacy cache identities.
- `restart_inventory.json`: every completed and interrupted root and its exact
  continuation status.
- `completed_candidates/`: complete active material results and final receipts.
- `qualifications/`: local observer/state qualification receipts.
- `tables/chronological_predictions.csv`: all 760 source-bound observations.
- `tables/genuine_support_state_history.csv`: genuine evolved support states.
- `tables/opening_surface_actual_sampled_stresses.csv`: G*, derivatives,
  inherited/adjustable contributions, raw rate, and cooperative completed rate.
- `tables/exact_final_support_opening_checks.csv`: final installed-surface
  production-consumer checks.
- `figures/`: common-axis fit/residual/state plots, G*, and matched-Kmax R
  comparison. Figures contain no grids and each subplot has an x-axis label.

## Reproduce analysis and package

From the repository root, using the accepted environment:

```bash
export MPLCONFIGDIR=/private/tmp/qste-mpl
/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python \
  scripts/analyze_qste340tm_active_calibration.py
/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python \
  -m pytest -q tests/test_qste340tm_active_completion.py \
  tests/test_qste340tm_restored_binding.py
/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python \
  scripts/package_qste340tm_active_calibration.py
```

The exact candidate restart commands are recorded in `restart_inventory.json`.
No rerun is required to inspect or plot the completed evidence.
