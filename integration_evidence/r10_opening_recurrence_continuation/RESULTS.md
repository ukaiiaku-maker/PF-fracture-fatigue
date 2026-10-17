# R10 opening renewal and finite-state feedback continuation

## Decision

The completed initial bank remains a zero-Paris-pass result. No trajectory in this continuation establishes recurrent cohesive fatigue, so no local Paris slope or developed rate is reported.

The retained 27231 smooth law has a 300 K athermal first-event bracket of **27.875–28.000 MPa√m**. At 28.770 MPa√m it advances one 1 µm cell and then reaches cyclic shakedown; the bilinear law instead reaches the geometry-owned 50 µm process-zone boundary in the first excursion. Those outcomes are an arrested increment and a target-truncated cascade, respectively—not recurrent growth.

Crossing the same 27231 cohesion with the active 31630 EXP_floor plasticity suppresses athermal advance through 35 MPa√m over the executed 48-cycle exposure. At 30 MPa√m the mixed law produces a three-cell common-coordinate first-passage burst during the first excursion, then no later event through 128 cycles. Phase refinement preserves 3 µm. Cell sizes 0.5, 1, and 2 µm give 2.5, 3, and 4 µm, so the event count is not a mesh-independent material rate and the result remains a bounded finite burst.

The low-barrier mixed crossover now uses an overdamped mean-first-passage calculation on the same finite-compliance cohesive coordinate. Its friction scale is fixed by the existing activation area, initial stiffness, and declared attempt frequency; it is accepted only when harmonic relaxation is at most 10^-3 of the opening phase. Unsupported nonadiabatic cases remain explicit.

The prior 2-D evidence evolved EXP_floor state but did not consume it in equilibrium. The corrected comparison restores the exact same saved active state in both arms. The retained A0-to-current mechanics gives 3.148 GPa of closing stress. Consuming it raises the smooth-interface FE branch-loss bracket from **1833.027 δ0** to **2455.044 δ0**; disabling feedback while holding state and geometry fixed recovers the lower bracket. This is a finite-state mechanical effect, not a Paris result.

## Stop semantics

- Physical non-advance, supported cyclic shakedown, unsupported first passage, finite first-excursion burst, structural-boundary cascade, numerical failure, and resource stop remain distinct.
- The 50 µm retained process-zone length is the structural boundary; the older 60 µm follow-up target is not reused.
- Prior exports did not contain complete active arrays plus RNG state, so they cannot support an exact checkpoint restart. They were deterministically replayed with identical material inputs; this continuation now exports the full active vectors and verifies bit-exact restore.
- No empirical cycle damage, per-cell delay, source suppression, opening-family substitution, or numerical advance cap was added.

## Verification

Focused cohesive/opening tests pass. The earlier full-suite run remains honestly recorded as interrupted after 39 passes; this continuation does not relabel it as a full pass.
