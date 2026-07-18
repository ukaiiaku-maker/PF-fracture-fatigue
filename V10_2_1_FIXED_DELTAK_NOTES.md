# v10.2.1 fixed-DeltaK stochastic fatigue

## Objective

Run each fatigue condition at one prescribed local stress-intensity range and
construct a mechanistic `da/dN` versus `DeltaK` relation without imposing a Paris
law.

## Control mode

The current promoted v10 configuration uses tip-only bulk mechanics and the
scalar waveform fatigue surrogate. Therefore v10.2.1 prescribes the local
waveform directly:

- user input: `DeltaK`, `R`, and frequency;
- `Kmax = DeltaK / (1 - R)` for `0 <= R < 1`;
- every front and branch waveform is constructed with that same target;
- the FEM is held at a small nonzero probe displacement and supplies evolving
  crack geometry, directional J information, and normalized tensor shape;
- the moving MPZ receives the exact prescribed cyclic K waveform after every
  stochastic crack event.

This is not yet a full cyclic-FEM load controller. Once a calibrated bulk cyclic
plasticity law is promoted, displacement amplitude should be adjusted by
feedback so the FEM-computed local DeltaK remains at the same target.

## Stochastic response

The v10.2 defaults remain active:

- exponential integrated-hazard thresholds;
- threshold-correlated, mean-preserving event lengths;
- variable-length sharp-wake geometry commits;
- anisotropic emission;
- validated common active-zone transport.

Zero-event cases are retained as right-censored observations rather than treated
as failed runs.

## Rate definitions

The first event is fatigue initiation and is not used as a propagation rate.
For later events,

```text
da/dN_i = event_advance_i / (cycles_i - cycles_{i-1}),  i >= 2.
```

The seed-level Paris point is the total extension after the first event divided
by the cycles between the first and last events. The analysis reports:

- cycles to first event;
- event-to-event stochastic rates;
- seed-level interval rates;
- right-censored and initiation-only counts;
- median and 16–84% interval across seeds;
- optional log-log Paris exponent when at least three DeltaK levels propagate.

## Run

Single-condition smoke:

```bash
MODE=smoke bash scripts/run_v10_2_1_fixed_deltaK_sweep.sh
```

Initial pilot:

```bash
MODE=pilot bash scripts/run_v10_2_1_fixed_deltaK_sweep.sh
```

The pilot defaults are `DeltaK = 15, 18, 21, 24, 27 MPa*sqrt(m)` and three
stochastic seeds at DBTT, 700 K, 45 degrees, `R=0.1`, and 1000 Hz.
