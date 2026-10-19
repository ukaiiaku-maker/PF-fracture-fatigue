# R10 constrained cohesive-shape and mechanism continuation

## Decision

This executed continuation found two distinct results rather than a strict
all-objective material.

First, adding one physically constrained third scale to the compact cohesive
potential did not remove the strict-slope obstruction under the retained
`Lpz` control.  All 2,048 new complete-cycle analytical vectors were domain
valid, but none passed the 3.4--4.6/two-rate-decade criterion.  The best point
had local slope 13.761: 9.531 from changing accessibility and 4.230 from the
conditional supported rate.  Its optimum was not pinned to a search-box bound.
The subsequent 128-vector state-informed screen and 24-material native block
did find two recurrently advancing prefixes, but neither reached the declared
20-event developed window before its resource stop.

Second, the mechanical-environment audit showed that the declared control is
not the stiffness of the explicit two-micrometre bulk fixture.  Keeping the
material and all EXP_floor physics fixed while identifying that fixture as a
separate environment produced developed HCF growth for archived candidate 027.
The first native adjacent secant is 3.006, close to its complete-cycle
prediction and inside the exploratory 2--8 band, but below the strict 3.4
lower bound.  The completed ladder is reported below; it is an explicit
mechanical-response alternative, not a fitted stiffness or a replacement for
the constitutive control.

There is therefore no strict joint pass and no basis for canonical promotion.
The strongest result is a moderate-slope recurrent-growth positive conditional
on the independently assembled specimen-scale environment.  Under the
original closure, the new potential flexibility and sampled state coupling
remain bounded negatives.

## Source, ownership, and constitutive scope

- Branch: `codex/r10-finite-compliance-inverse-shape`
- Compatible starting commit: `6885410b`
- Execution-source commit: `51be5dcd`
- Qualified interpreter: `arrhenius-sharp-front-v10-codex`
- Result root: `runs/r10_cohesive_shape_continuation`

The additive `three_scale_tail` variant remains an opt-in
`compact_independent` research potential.  One serialized potential supplies
traction, tangent, work, stationary roots, completion, and opening activation.
It is positive, finite-work, single-hump, and C1 at both internal joins.  The
archived compact, smooth-exponential, and bilinear definitions are unchanged.
The algebraic-tail diagnostic is not registered.

EXP_floor emission, Peierls motion, Taylor release, activation entropies and
couplings, signed transport/return, backstress, blunting, wake, contact,
geometry, energy ownership, and high-cycle integration are unchanged.  No
cycle-damage term, artificial delay, source suppression, temperature endpoint
fit, cell-size rate fit, or empirical Paris law was introduced.

## Mechanical environment

For candidate `R10_INVERSE_OPT_027__CM` (complete hash
`b8b75d6335c...`), the declared closure gives:

- `Gamma = 0.6236782096 J/m2`;
- peak traction `61.53702219 GPa`;
- `Lpz = 6.9229816e-11 m`;
- `ke = 3.0358645e21 Pa/m`;
- `ke Gamma / peak^2 = 0.5`.

The bulk-only condensed fixture explicitly excludes the cohesive tangent.  At
half-height `Lpz` it gives `3.2130587e21 Pa/m`, 1.05837 times the declared
control.  At the actual two-micrometre half-height it gives
`1.1121973e17 Pa/m`, only `3.6635e-5` of that control.  Thus the printed `Lpz`
is the constitutive process-zone control length; it is neither the one-
micrometre crack increment, the tip radius, nor the specimen dimension.

Under the retained closure, candidate 027 has birth/fold loads
13.990/158.796 MPa-sqrt(m), minimum complete-cycle slope 6.525, and no strict
connected interval.  Under the explicitly separate two-micrometre fixture it
has birth/fold loads 4.577/154.251 MPa-sqrt(m), minimum slope 2.808, exploratory
2--8 width 3.070 rate decades, and analytical strict width 0.434 rate decades.
No stiffness multiplier was optimized against the desired slope.

## Complete-cycle reference and archived 027 reconciliation

At the four archived developed loads, shrinking-step complete-cycle slopes
are 7.296, 6.761, 6.547, and 6.491.  Their accessibility contributions are
3.268, 2.480, 2.011, and 1.701; conditional-rate contributions are 4.028,
4.281, 4.536, and 4.790.  The decomposition identity closes to floating-point
precision.

The continuum predictions are `4.201e-11`, `5.816e-11`, `7.812e-11`, and
`1.031e-10 m/cycle`.  The archived native 40-event windows are `3.072e-11`,
`5.993e-11`, `7.865e-11`, and `1.136e-10 m/cycle`, with respective 95% event-
count intervals `(2.195,4.183)e-11`, `(4.282,8.161)e-11`,
`(5.619,10.710)e-11`, and `(8.113,15.463)e-11`.  Frozen 8--256-phase checks
show the native midpoint convention converging around the continuum integral.
The residuals are not separable from finite-window scatter and weak evolving-
state effects, so they were not fitted as a constitutive correction.

## New constrained-potential search

The analytical allocation contains 2,048 trials and 2,048 domain-valid
vectors.  Acquisition preserved 128 complete hashes spanning 126 distinct
parent EXP_floor backgrounds and all four retained parent classes.  Every
record carries both the cohesive and complete parent parameter JSON plus
cohesive, plasticity, parent, and complete-material hashes.

The best analytical vector is `R10_CONT3_00664__CM`.  At its selected load its
rate is `4.473e-11 m/cycle`, but slope is 13.761.  None of the rise, drop, knee,
mid-level, terminal-tail, energy, or zero-load-barrier bounds is active at this
point.  The remaining limit is the operating/accessibility geometry of this
admissible shape, not a silent excursion beyond the declared research box.

All 128 exact two-cycle state-informed reductions completed.  They contain no
opening event or unsupported mixed crossover; they are exposure probes, not
runouts.  The 24-material full-state block then produced:

- eight physical runouts to `1e9` accepted cycles;
- seven numerical stops;
- six unresolved-opening-physics cases;
- three resource stops.

Only `R10_CONT3_01111__CM` and `R10_CONT3_01486__CM` advanced recurrently.
Their exact resumptions ended at 14 events / 142,526.98 cycles and 15 events /
238,110.95 cycles, respectively.  Adjacent-amplitude and frozen controls are
preserved, but all remain shorter than the declared developed window.  The
frozen controls used different stochastic seeds, so their timing differences
are not classified as causal plasticity effects.

## Explicit-environment developed ladder

| Kmax (MPa-sqrt(m)) | Streams / window events | Analytical rate | Native pooled rate (m/cycle) | 95% count interval |
|---:|---:|---:|---:|---:|
| 9.25897 | 1 / 40 | 3.026e-11 | 3.631e-11 | 2.594e-11--4.945e-11 |
| 10.10496 | 1 / 40 | 3.966e-11 | 4.723e-11 | 3.374e-11--6.431e-11 |
| 11.02823 | 2 / 80 | 5.286e-11 | 4.994e-11 | 3.960e-11--6.215e-11 |
| 12.03590 | 1 / 40 | 7.194e-11 | 6.346e-11 | 4.534e-11--8.641e-11 |
| 13.13561 | 1 / 40 | 1.003e-10 | 8.803e-11 | 6.289e-11--1.199e-10 |

Every row uses the unchanged candidate-027 complete-material hash, R=0.1,
300 K, 1 kHz, exact event continuation, and the explicit
`1.1121972927517362e17 Pa/m` environment identity sealed into its restart.
Rates use the declared 20--60-event developed window.  The middle amplitude
was repeated with an independent hazard stream; its individual rates are
`4.322e-11` and `5.912e-11 m/cycle`.  Pooling the two exact exposures removes
the apparent negative adjacent slope.  The final native secants are 3.006,
0.638, 2.741, and 3.743.  Only the last is in the strict band, for 0.142 rate
decade.  The largest connected exploratory 2--8 interval is 0.246 decade.
This is reproducible recurrent moderate-rate growth, but neither a broad
strict Paris interval nor a monotone constant-slope power law at the present
window precision.

## Same-material fracture and reduced 2-D feedback

The two recurrent three-scale materials received native seven-temperature
monotonic panels without refitting.

- `R10_CONT3_01486__CM` commits at 38, 31, 33, 30, 28, 25, and
  23 MPa-sqrt(m) from 300 to 1200 K.
- `R10_CONT3_01111__CM` commits at 8, 8, 7, 6, and 6 MPa-sqrt(m) through
  900 K; 1050 and 1200 K are unsupported mixed crossovers, not roots.

These are demonstrated responses of the complete cohesive-plus-EXP_floor
vectors; parent Weak-T/Peak-T/DBTT labels are not transferred as demonstrated
labels.

Bit-exact same-material restart transfer was also checked in the existing
finite-state 2-D feedback operator.  Both full and opening-disabled arms for
01111 and 01486 remain ramp-limit censored through `20 delta0`; closure
magnitudes are 11,265 Pa and 207.9 Pa.  This is a supported negative at the
reduced operator's scope, not a 2-D fatigue-rate validation.

## Numerical qualification and coverage

The final analyzer contains 36 native HCF results: six developed streams, nine
resource stops, eight declared-horizon physical runouts, seven numerical
stops, and six unresolved-opening-physics cases.  There are no unfinished
progress-only trajectories.  Across those results the solver accepted
`8,008,128,818.68` cycles (`8,008,128.82 s` simulated), 444 physical advances,
and `18,093.64` HCF CPU-seconds.  The executed evidence also includes 2,048
analytical trials, 128 state-informed complete vectors, 14 same-material
fracture rows, two reduced 2-D feedback receipts, and 31 hashed restart-state
pairs.

The current full-suite result is **957 passed in 3,074.10 s**.  Focused cohesive-law,
slope-decomposition, evaluator, restart-identity, and explicit-environment
tests also passed.  All accepted live states are inventoried by relative path,
SHA-256, and paired NPZ hash in `campaign_delivery/restart_inventory.csv`.

## Limits and next scientific decision

No new three-scale material has a developed rate, and no current result meets
the strict 3.4--4.6/two-decade requirement.  Resource-limited prefixes are not
rates; one-cycle state probes are not runouts; unresolved mixed crossovers are
not fracture roots; and reduced 2-D ramp censors are not spatial-fatigue
validation.

The executed evidence narrows the obstacle.  Under the `ke Gamma/peak^2=0.5`
control it is not merely the old two-scale tail bounds: the added admissible
shape remains accessibility dominated.  An independently assembled specimen-
scale environment materially changes that conclusion and produces recurrent
moderate-slope HCF growth without changing the material.  The immediate
scientific choice is therefore whether that explicit geometry is the intended
mechanical problem.  If it is, subsequent same-material thermal and spatial
qualification should use that declared environment.  If it is not, further
shape search under the original closure should be treated as conditional on a
constitutive process-zone control that the explicit two-micrometre fixture does
not reproduce.
