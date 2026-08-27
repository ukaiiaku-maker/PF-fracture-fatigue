# Canonical PF fracture run plan V2

This immutable design supersedes the earlier 240-case V1 matrix without
overwriting it. No stochastic trajectory was launched while producing this
plan.

## Primary design

- Unique canonical conditions: **288**.
- Orientation membership: **192**, theta = 0/15/30/45 degrees at rate1x.
- Rate membership: **144**, theta = 0 degrees at rate0p01x/rate1x/rate100x.
- Shared theta0/rate1 membership: **48**, stored once with both flags true.
- Fixed loading increment: dU = 2e-7 m; dt = 840/8.4/0.084 s.
- Common random numbers: identical seed across the three theta0 rates for each
  fixed material class and temperature.
- Scientific plan fingerprint: `f3928476f2564a3eb10ca4737780a38578d9517a860bd77a9321dcd94fd4df99`.

## Preserved and reclassified products

- Theta15/theta30 complete products: **96/96**, verified and frozen for reuse.
- Reusable complete theta45/rate1 products: **0**.
- Complete theta45 extreme-rate supplemental products: **42**.
- Interrupted theta45 extreme-rate directories marked for fail-closed cancellation: **2**.
- Supplemental runs are excluded from the primary rate analysis.

## Mechanics/source coordinates

All four angle-specific families use the horizontal
`forward_100_cleavage_trace`. Theta rotates cubic elasticity and slip/source
coordinates, not the prescribed crack line; the laboratory-x forward cosine
is therefore 1.0 for theta 0/15/30/45. Every family covers at least
1020 micrometres (1000 target plus the established
20 micrometre maximum-event safety margin). Family interpolation is bounded
by its recorded envelope and extrapolation is forbidden.

## Execution gate

This document is a plan/lock record only. Stage A may select only incomplete
theta45/rate1 rows. Stage B may select only theta0 rows. Completed theta15/30
rows cannot be selected by either V2 production stage.
