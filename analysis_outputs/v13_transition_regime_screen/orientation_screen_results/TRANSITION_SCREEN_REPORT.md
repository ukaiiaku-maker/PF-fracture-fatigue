# V13 transition-regime screen

`BRANCHING_KINETICS_MODEL_UNCALIBRATED`

This is a branch-disabled counterfactual screen. A positive row is the first exact frozen state predicting an admissible mark, not an actually committed branch. No observed incidence probability is estimated.

| Parent | First predicted event | Extension (µm) | 75 µm censored control | Terminal |
|---|---:|---:|---|---|
| theta15_rate1x_Peak_300K_seed3621 | 3 | 10.9534 | False | FIRST_PREDICTED_BRANCH_STATE_CAPTURED |
| theta15_rate1x_Peak_1000K_seed3621 | 3 | 10.9534 | False | FIRST_PREDICTED_BRANCH_STATE_CAPTURED |
| theta15_rate1x_weakT_300K_seed3621 | None | 77.2741 | True | target_reached |
| theta15_rate1x_weakT_1000K_seed3621 | None | 77.2741 | True | FIRST_PREDICTED_BRANCH_STATE_CAPTURED |
| theta30_rate1x_Peak_300K_seed3621 | 4 | 13.6603 | False | FIRST_PREDICTED_BRANCH_STATE_CAPTURED |
| theta30_rate1x_Peak_1000K_seed3621 | 2 | 6.83013 | False | FIRST_PREDICTED_BRANCH_STATE_CAPTURED |
| theta30_rate1x_weakT_300K_seed3621 | 4 | 15.4904 | False | FIRST_PREDICTED_BRANCH_STATE_CAPTURED |
| theta30_rate1x_weakT_1000K_seed3621 | 2 | 6.83013 | False | FIRST_PREDICTED_BRANCH_STATE_CAPTURED |

## Gate

Useful conditions: theta15_rate1x.

The gate requires a positive prediction and a completed 75 µm negative parent across the four groups, chi on both sides of one, a nonsaturated channel, and positive exact pair margin for positive predictions. Early legitimate gates are neither 75 µm negatives nor evidence of a transition.

The fixed 5 µm physical event is not shortened to hit a reporting boundary. An accepted endpoint may overshoot 75 µm; its frozen diagnostic is preserved but excluded from the common-window transition gate. The common censor is 75 µm in accepted maximum-forward-reach coordinates, not branch-junction position.

A positive pair margin is necessary. Its distance from the boundary is reported as margin/release and margin/cost rather than silently treating a large positive margin as a near-boundary observation.

Seed 3621 is paired across the four material–temperature groups at a fixed orientation. Candidate identities depend on orientation; the same numerical seed does not imply identical threshold streams across different orientations. This screen locates candidate conditions and is not an isolated causal estimate of orientation effects.
