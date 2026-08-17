# v9.14 minimal reversible fatigue model

## Parent and scope

This branch is rooted at `52ef0d13c81703f83f4adf4f56553613c47ce9bd`, the completed joint fracture–fatigue Paris-slope study.  The purpose of this branch is to test one missing cyclic state mechanism before changing barrier shapes again: mobile dislocations may reverse during unloading and return to the crack/free surface if they have not become retained/tangled.

This first implementation is deliberately **not** a non-Schmid model and is **not** an R-ratio campaign.  Those are later steps after the reversible state has been qualified.

## Physics retained unchanged

The following remain exactly the v9.13/v9.14 production physics:

- cleavage barrier and cleavage multi-hit hazard;
- dislocation-emission barrier and attempt frequency;
- stochastic unit-exponential first-passage thresholds;
- event-length law;
- Peierls signed mobility law;
- Taylor/forest encounter and retained-state kinetics;
- shielding/backstress definitions except insofar as the state itself changes;
- crack-event translation;
- the complete fatigue waveform and same-cycle continuation after crack events.

No empirical reversible fraction is introduced.

## Minimal added state physics

The qualified stiff MPZ operator already transports mobile dislocations with a signed Peierls velocity.  For Burgers-sign population `q`, its spatial velocity is the existing signed velocity multiplied by the Burgers sign.  If the effective drive reverses during unloading, that operator therefore already sends a still-mobile dislocation back toward the left MPZ boundary.

The new state class makes the boundary fate explicit:

1. **Left-boundary mobile outflow** is classified as return to the crack/free surface and annihilation there.
2. **Right-boundary mobile outflow** is recorded separately as far-field escape.
3. **Retained/tangled content is never removed by the surface-return rule.**
4. Returned mobile line content cancels an equal amount of the near-tip source-slip/blunting ledger, bounded by the available uncancelled slip.
5. The historical cumulative source-slip ledger remains intact as a cumulative activity measure; a separate returned-slip field makes the blunting state depend on the nonnegative net ledger.

Thus reversibility emerges from competition between signed transport and the pre-existing mobile-to-retained encounter kinetics.  There is no prescribed `reversible_fraction(DeltaK)`.

## State quantities

The reversible state records:

- cumulative and interval returned mobile line content;
- cumulative and interval far-field escape;
- cumulative and interval source-slip cancellation;
- cumulative source-slip count;
- returned source-slip count;
- nonnegative net source-slip count;
- return and escape fractions relative to emitted line content.

The returned-slip cancellation field is translated with crack advance just like the baseline accumulated-slip field, preventing old returned content from permanently cancelling slip at a newly advanced tip.

## Qualification path

The first qualification uses **explicit physical cycles** only.  This is intentional: the new state transition must be established before the DMD/projective high-cycle map is allowed to use it.

Use `scripts/run_v914_minimal_reversible_case.py` with `--model baseline` and `--model reversible` to generate matched A/B cases with identical candidate, loading, seed, phase resolution, and crack-growth target.

Recommended first comparison:

- T = 300 K;
- R = 0.1;
- frequency = 1000 Hz;
- seed = 1720;
- canonical Peak-T, DBTT, weak-T, and ceramic-like parameterizations;
- identical load points taken from the completed Paris-slope study.

The first scientific diagnostic is not simply whether `da/dN` changes.  It is whether the fate ledger demonstrates physically resolved return while retained content remains irreversible, and whether the resulting change in cycle-integrated hazard explains any change in local or developed Paris slope.

## Required checks before acceleration

Before extending the fast-cycle/DMD operator, require:

1. zero left-boundary return reproduces the baseline state to numerical precision;
2. reverse signed velocity produces positive surface return and no artificial far-boundary loss;
3. retained content is unchanged by the return rule except through the pre-existing Taylor/release kinetics;
4. the returned-slip field never exceeds the cumulative source-slip field in net effect;
5. checkpoint/restart reproduces reversible ledgers and RNG/event history;
6. same-cycle post-event waveform continuation remains exact;
7. A/B comparisons use identical cleavage physics and stochastic inputs.

Only after these pass should the projective high-cycle map be taught the new reversible state, followed later by R-ratio and non-Schmid/tension–compression-asymmetry studies.
