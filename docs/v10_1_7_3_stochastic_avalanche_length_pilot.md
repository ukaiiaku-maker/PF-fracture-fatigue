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

The corrected pilot defaults are

```text
L0 = 5 micrometres
q_min = 0.5
q_max = 4.0
```

so individual event lengths remain broad while avoiding requested geometry events below the reliable sharp-wake realization scale. The ensemble mean remains 5 micrometres. Under constant cleavage rate, using the same threshold for waiting time and reward preserves the long-time mean velocity \(L_0\lambda_c\).

No random perturbation is added to K, J, the cleavage or emission barriers, source capacity, back stress, shielding, blunting, or any material parameter.

## Geometry realization

The first implementation attempted to call the sharp-wake backend ten times inside one outer FEM solve. That was invalid because the backend has a finite minimum realizable damage increment: nominal 0.5 micrometre requests were promoted to approximately 2 micrometres, creating a 20 micrometre event from a nominal 5 micrometre event.

The corrected pilot performs one checked geometry commit per sampled event. The requested and realized lengths must agree within 5% or 1 nm, whichever is larger. A mismatch vetoes the geometry transaction rather than silently changing the crack-growth reward.

The input `EVENT_SUBSEGMENT_FRACTION` is retained only as a record of the desired future driver-level subdivision. It is not used to claim that mechanics was re-equilibrated at 10% increments.

True 10% subincrements require:

1. commit one fraction of the event;
2. reassemble and solve the FEM field;
3. recompute J/K and the tip state;
4. commit the next fraction at unchanged external time/load unless the avalanche arrests.

That driver-level lifecycle is outside this initial pilot.

## Controls

The runner executes three cases:

1. `fixed_original`: v10.1.7.2 deterministic threshold and the original fixed 5 micrometre geometry event.
2. `segmented_deterministic`: retained directory name for the deterministic variable-event-backend control; it must realize one checked 5 micrometre commit and match the original response.
3. `stochastic_avalanche`: exponential threshold and threshold-scaled variable reward, realized as one checked geometry commit per event.

The deterministic wrapper control isolates any response caused by routing through the variable-event backend.

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

## Live reporting

The runner exports `PYTHONUNBUFFERED=1`, launches Python with `-u`, and prints timestamped messages for:

```text
CAMPAIGN START
CASE START
SOLVER PID
HEARTBEAT
CASE COMPLETE
CASE FAILED
ANALYSIS START / COMPLETE
CAMPAIGN COMPLETE / FAILED
```

The default heartbeat interval is 60 seconds and can be changed with

```text
HEARTBEAT_SECONDS=15
```

`tail -F` is recommended so the monitor continues following the output file if it is recreated.

## Running

```bash
OUTROOT=runs/v10_1_7_3_stochastic_avalanche_DBTT_700K_200um_v1 \
CLASS=DBTT \
TEMP_K=700 \
SEEDS="1 2" \
TARGET_EXT_UM=200 \
EVENT_MIN_FACTOR=0.5 \
EVENT_MAX_FACTOR=4.0 \
EVENT_SUBSEGMENT_FRACTION=0.1 \
HEARTBEAT_SECONDS=15 \
bash scripts/run_v10_1_7_3_stochastic_avalanche_pilot.sh
```

The runner is resume-safe. It validates the completed fixed control, deletes stale products from invalid cases, and reruns only the failed or incomplete cases.

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

Each variable-event-backend run also writes

```text
stochastic_avalanche_geometry_events.json
```

with requested and realized event lengths, their error, threshold, desired future subdivision count, actual geometry-commit count, and whether mechanics was re-equilibrated.

## Acceptance checks

The analyzer separates wrapper effects from stochastic event-size effects. The default gates are:

- deterministic wrapper-control response differs from the original fixed response by no more than 2% of the deterministic K range;
- stochastic ensemble mean bias is no more than 5%;
- sampled mean event length lies within 20% of 5 micrometres;
- event-length coefficient of variation is at least 0.30;
- mean 10--90% K band is at least 0.25 MPa sqrt(m);
- mean detrended correlation to the deterministic wrapper control is at most 0.98.

Failure of these gates is informative. A large deterministic-control shift means the event backend itself changed the response. A large ensemble bias means the renewal-reward coupling is not mean-preserving in the evolving system. Persistently high correlation means the geometry/compliance waveform still dominates despite variable event sizes.
