# v10.1.7.3 stochastic avalanche-length pilot

## Purpose

The v10.1.7.2 stochastic first-passage pilot changed the integrated cleavage-hazard threshold but retained one fixed 5 micrometre crack-growth reward for every completed event. Two pilot seeds therefore visited essentially the same sequence of crack geometries and retained a detrended correlation close to one.

v10.1.7.3 tests a bounded correlated renewal-reward model. The same stochastic threshold controls both the event waiting time and the total crack extension associated with that event. This is an exploratory crack-growth intermittency model; it is not part of the finalized v10.1.7.1 production constitutive model.

## Event law

For each cleavage renewal, draw

\[
\Xi_i\sim\operatorname{Exponential}(1).
\]

The event completes when the integrated physical cleavage hazard reaches \(\Xi_i\). Its total crack-growth reward is

\[
L_i=L_0\,
\frac{\operatorname{clip}(\Xi_i,q_{\min},q_{\max})}
{E[\operatorname{clip}(\Xi,q_{\min},q_{\max})]}.
\]

For \(\Xi\sim\operatorname{Exponential}(1)\),

\[
E[\operatorname{clip}(\Xi,a,b)]
=a+e^{-a}-e^{-b}.
\]

The default values are

```text
L0 = 5 micrometres
q_min = 0.2
q_max = 4.0
```

so individual event lengths are bounded at approximately 1--20 micrometres before normalization, while the ensemble mean remains 5 micrometres. Under constant cleavage rate, using the same threshold for waiting time and reward preserves the long-time mean velocity \(L_0\lambda_c\).

No random perturbation is added to K, J, the cleavage or emission barriers, source capacity, back stress, shielding, blunting, or any material parameter.

## Geometry subsegments

Each completed event is realized through equal sharp-wake geometry subsegments. The default subsegment fraction is 0.1, giving ten subsegments per event regardless of its total length.

The existing kinetic MPZ integration remains finer than this geometry subdivision:

```text
maximum cleavage-action substep = 0.01
maximum MPZ translation substep = 5e-8 m = 0.05 micrometres
```

The ten geometry subsegments are committed during one outer geometry transaction. The FEM field is **not** re-equilibrated between these subsegments in this initial pilot. The next outer solve sees the completed event geometry. This limitation is written explicitly to the audit and geometry-event files.

## Controls

The runner executes three cases:

1. `fixed_original`: v10.1.7.2 deterministic threshold and one fixed 5 micrometre geometry event.
2. `segmented_deterministic`: deterministic threshold and fixed 5 micrometre reward, realized through ten geometry subsegments.
3. `stochastic_avalanche`: exponential threshold and threshold-scaled variable reward, also realized through ten geometry subsegments.

The segmented deterministic case isolates any response caused solely by geometry subdivision.

## Scope restrictions

This pilot is intentionally restricted to:

```text
single front
branching disabled
sharp_wake backend
wake shielding disabled
monotonic loading
```

The event descriptor queue is single-front/FIFO and is not branch-safe. Geometry-veto rollback for a variable event has not been generalized to all topology backends. Do not use this branch for adaptive CZM, branching, fatigue, or production temperature sweeps.

## Running

```bash
OUTROOT=runs/v10_1_7_3_stochastic_avalanche_DBTT_700K_200um_v1 \
CLASS=DBTT \
TEMP_K=700 \
SEEDS="1 2" \
TARGET_EXT_UM=200 \
EVENT_MIN_FACTOR=0.2 \
EVENT_MAX_FACTOR=4.0 \
EVENT_SUBSEGMENT_FRACTION=0.1 \
bash scripts/run_v10_1_7_3_stochastic_avalanche_pilot.sh
```

The runner is resume-safe and validates completed cases before skipping them.

## Outputs

```text
stochastic_avalanche_pilot_manifest.tsv
stochastic_avalanche_case_summary.csv
stochastic_avalanche_event_curves.json
stochastic_avalanche_assessment.json
stochastic_avalanche_R_curve_ensemble.png
stochastic_avalanche_event_lengths.png
stochastic_avalanche_event_length_distribution.png
stochastic_avalanche_threshold_vs_length.png
```

Each segmented run also writes

```text
stochastic_avalanche_geometry_events.json
```

with the event length, threshold, number of subsegments, and whether mechanics was re-equilibrated between subsegments.

## Acceptance checks

The analyzer separates discretization effects from stochastic event-size effects. The default gates are:

- segmented deterministic response differs from the original fixed response by no more than 2% of the deterministic K range;
- stochastic ensemble mean bias is no more than 5%;
- sampled mean event length lies within 20% of 5 micrometres;
- event-length coefficient of variation is at least 0.30;
- mean 10--90% K band is at least 0.25 MPa sqrt(m);
- mean detrended correlation to the segmented deterministic curve is at most 0.98.

Failure of these gates is informative. A large segmented-control shift means geometry subdivision itself changed the response. A large ensemble bias means the renewal-reward coupling is not mean-preserving in the evolving system. Persistently high correlation means the geometry/compliance waveform still dominates despite variable event sizes.
