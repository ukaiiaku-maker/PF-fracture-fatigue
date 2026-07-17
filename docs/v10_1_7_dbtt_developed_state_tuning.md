# v10.1.7 DBTT developed-state tuning

## Purpose

v10.1.6.1 showed that the promoted DBTT barriers generate a strong temperature
dependence in initiation toughness, but essentially no matched plastic R-curve
increment between 300 and 1100 K.  The maximum source population and maximum
back stress were nearly identical at both temperatures, while the population
collapsed during propagation.

v10.1.7 changes no constitutive law.  It adds history diagnostics and performs a
small two-parameter calibration of the existing temperature-independent source
evolution scales.

## Default matrix

- temperatures: 300 and 1100 K;
- back-stress scales: 0.5, 1, and 2;
- refresh-length scales: 0.1, 0.3, and 1;
- crack extension: 50 micrometres;
- one no-plasticity baseline at each temperature;
- nine full-model parameter sets, each run at both temperatures.

A four-set matrix can be requested with

```bash
BACKSTRESS_SCALES="0.5 1" REFRESH_SCALES="0.3 1"
```

## Diagnostics

Every kinetic audit record contains:

- mobile, retained, and total active population;
- retained fraction;
- cumulative emitted and refreshed source capacity;
- cumulative trapping, release, recovery, and escape;
- mobile, retained, and total population residence integrals;
- local emission back stress;
- active shielding;
- source budget remaining and consumed.

The analyzer reports values before and at first advance, means over the final
quarter of crack growth, and final cumulative histories.

## Ranking

For each candidate the full event curve is initiation-referenced and the matched
no-plasticity curve is subtracted.  Ranking favors a positive late plastic
R-curve at 1100 K, penalizes any late plastic R-curve at 300 K, and penalizes
changes in the scale-1 initiation shifts.

A coarse-screen candidate passes when:

- absolute 300 K late plastic R-rise is at most 0.5 MPa sqrt(m);
- 1100 K late plastic R-rise is at least 1 MPa sqrt(m);
- high-minus-low emergence is at least 1 MPa sqrt(m);
- summed initiation-shift deviation from the scale-1 reference is at most
  1 MPa sqrt(m).

These thresholds are analysis-only and never enter the solver.
