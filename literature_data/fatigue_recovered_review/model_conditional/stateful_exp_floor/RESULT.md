# Conditional QSTE340TM state-informed EXP_floor refinement

Date: 20 October 2026

## Result

The analytical state-informed fit lowers the all-point log-rate RMSE from 0.632
decades for the opening-only reference to 0.336 decades. It substantially
reduces the between-R offset, especially at R=0.7, but it does not reproduce
every within-R curve shape. This is a partial refinement, not an accepted
complete-material calibration.

| R | opening-only RMSE | stateful RMSE | observed slope | stateful slope |
|---:|---:|---:|---:|---:|
| 0.1 | 0.779 | 0.394 | 3.709 | 2.572 |
| 0.3 | 0.373 | 0.217 | 3.378 | 1.672 |
| 0.5 | 0.183 | 0.444 | 3.762 | -0.020 |
| 0.7 | 0.980 | 0.163 | 2.920 | 0.003 |

The R=0.5 block becomes worse and the R=0.3, 0.5 and 0.7 fitted slopes flatten.
The model is using a strong transition in retained plastic state to repair the
between-R displacement. That transition is physically interpretable through
the existing emission/escape/retention equations, but it is not yet supported
well enough to call the plasticity vector identified.

## Model scope

- One opening and plasticity material is shared by all four R histories.
- Each block is replayed in source order using cycle coordinates from the paired
  crack history; observations are not reset to a virgin state.
- Opening uses the existing smooth-broad barrier and cooperative completion law.
- Plasticity uses existing EXP_floor emission, series Peierls/Taylor escape,
  escape-limited retention, saturation, recovery, blunting, back stress and
  stored-energy coupling.
- The nonzero EXP_floor temperature slopes are retained. Only their effective
  298.15 K coordinates are fitted.
- Frequency remains the explicitly provisional 10 Hz assumption.

## Validation limit

All four leave-one-R-out least-squares refinements reached their 350-evaluation
limit. Their held-out RMSE values are recorded in `fit.json`; they are useful
bounds, not converged validation optima. R=0.3 transfers best, while R=0.5
remains the principal failure. The overload and closure histories were not
included in the likelihood and remain genuine validation targets.

This analytical result does not authorize a native full-state, thermal, or
complete QSTE340TM material claim.
