# v10.1.6 emergent temperature-response matrix

## Purpose

v10.1.6 changes no constitutive physics from v10.1.5.  It tests whether the
promoted Arrhenius cleavage, emission, Peierls, Taylor, and recovery surfaces
produce the required class-dependent temperature response using one common,
temperature-independent source evolution law.

The runtime source capacity, refresh-length scale, back-stress scale, shielding
bound, and blunting law are not functions of temperature.

## Required qualitative response

- ceramic: weak plastic R-curve at low, intermediate, and high temperature;
- weakT: comparable plastic R-curve contribution across temperature;
- DBTT: weak plastic R-curve at low temperature and stronger contribution at
  high temperature.

The DBTT transition must emerge through the Arrhenius kinetics.  The runner
contains no DBTT temperature switch and no temperature-dependent source count,
refresh length, shielding amplitude, or blunting coefficient.

## Geometry-corrected response

Every full calculation has a matched `plasticity_off` calculation with the same
mesh, directional-J mode, crack geometry controls, temperature, and material
barriers.  The analysis first refers each event curve to its own initiation and
then subtracts the matched no-plasticity curve:

```
plastic late R-rise = (late K - K_init)_full
                    - (late K - K_init)_plasticity_off
```

This is more diagnostic than the raw maximum of K(a), which contains a strong
common contribution from the deterministic crack path and directional J.

## Default matrix

- classes: ceramic, weakT, DBTT;
- temperatures: 300, 700, 1100 K;
- modes: full, plasticity_off;
- crack extension: 50 micrometres;
- common campaign scales: 1.0 and 1.0;
- branching and wake shielding: off.

## Outputs

The runner writes:

- `temperature_matrix_manifest.tsv`;
- `temperature_matrix_case_summary.csv`;
- `temperature_matrix_ablation_summary.csv`;
- `temperature_matrix_event_K.json`;
- `temperature_emergence_assessment.json`;
- K-initiation, plastic R-rise, active-population, and emission-back-stress plots.

The numerical thresholds in the assessment JSON are validation criteria only.
They are never passed into the constitutive solver.

## Interpretation

A raw DBTT R-curve increase is insufficient.  A credible transition requires all
of the following:

1. small initiation-referenced full-minus-off R-rise at low temperature;
2. larger full-minus-off R-rise at high temperature;
3. corresponding growth of active/retained population, emission back stress,
   shielding, or blunting diagnostics;
4. weak temperature span for ceramic and weakT;
5. identical runtime evolution scales at every temperature.

If the raw event curves change with temperature while the full-minus-off curves
do not, the apparent transition is geometric or cleavage-barrier driven rather
than a developed plastic-zone transition.
