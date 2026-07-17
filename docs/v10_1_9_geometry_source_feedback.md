# v10.1.9 bounded geometry/source-capacity feedback

## Purpose

v10.1.8 demonstrated that a spatial source field ahead of the crack tip changed
the magnitude and interpretation of the promoted campaign source budget.  This
revision returns directly to v10.1.7 and adds one bounded post-initiation
geometry feedback without changing the calibrated first-passage problem.

## Source-capacity law

The effective tip radius at the first crack advance is stored as `r_ref`.  No
geometry feedback is allowed before or during that first-passage event.  For
subsequent states,

```
x = max((r_eff - r_ref) / r_ref, 0)
capacity_ratio = 1 + g_site * x / (1 + x)
```

where `g_site` is the only new parameter.  The maximum source capacity is
`(1 + g_site)` times the promoted campaign capacity.  Capacity growth is
irreversible within a crack path, and only newly exposed capacity is added to
the available source pool.  No emission occurs automatically: every new site
remains governed by the existing Arrhenius emission rate and Taylor back stress.

## Unchanged physics

- promoted source capacity at initiation;
- campaign crack-advance refresh law;
- cleavage, emission, Peierls, Taylor, and recovery barrier surfaces;
- separated opening, cleavage, and emission stresses;
- local Taylor back stress;
- active shielding cap and blunting coefficient;
- wake shielding disabled in the validation matrix.

The geometry gain is temperature independent.  Any DBTT contrast must emerge
from the existing Arrhenius kinetics and the amount of post-initiation blunting.

## Default validation matrix

- material class: DBTT;
- temperatures: 300 and 1100 K;
- geometry gains: 0, 1, 4, and 9;
- crack extension: 100 micrometres;
- one no-plasticity baseline per temperature.

The gain-zero case is the exact v10.1.7 control.  Every candidate must preserve
first passage within one percent of that control.  Ranking then requires a weak
300 K plastic R-curve and a substantially larger late 1100 K plastic R-curve.
