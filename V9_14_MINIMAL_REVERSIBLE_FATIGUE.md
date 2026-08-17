# v9.14 minimal reversible fatigue model

## Parent and scope

This branch is rooted at `52ef0d13c81703f83f4adf4f56553613c47ce9bd`, the completed joint fracture–fatigue Paris-slope study. The purpose is to test one missing cyclic state mechanism before changing barrier shapes again: a newly emitted dislocation can remain mobile, reverse during unloading/compression, return to the crack/free surface, and annihilate there if it has not become retained/tangled.

This implementation is deliberately **not** a non-Schmid model. R-ratio and non-Schmid studies follow only after the reversible state is qualified.

## Physics retained unchanged

The following remain the qualified v9.13/v9.14 production physics:

- cleavage barrier and cleavage multi-hit hazard;
- dislocation-emission barrier and attempt frequency;
- stochastic unit-exponential first-passage thresholds;
- event-length law;
- Peierls barrier/rate law;
- Taylor/forest retained-state kinetics;
- shielding/backstress definitions except insofar as the state itself changes;
- crack-event translation;
- the complete fatigue waveform and same-cycle continuation after crack events.

No empirical reversible fraction is introduced.

## Signed mobile-transport refinement

The initial v1 branch exposed the return boundary but inherited the parent clipping `K_eff=max(K-K_shield,0)` in the Peierls transport channel. The Peak-T `f=1.10`, `R=0.1` smoke test correctly gave baseline parity and essentially zero return, but it also made clear that a later negative-R calculation would not have produced compression-driven reverse transport.

The v2 state therefore separates the transport and opening channels:

- cleavage remains opening-only and continues to use the unchanged parent law;
- new emission remains opening-only and continues to use the unchanged parent law;
- already-mobile line content uses the signed transport intensity
  `K_transport = K_applied - K_shield`, with no zero clipping;
- the existing signed GND stress is added to the transport stress;
- the existing signed Peierls rate determines mobile velocity from that signed stress;
- the encounter rate follows the magnitude of the resulting Peierls travel rate.

Thus positive applied K can still generate reverse transport if shielding/internal stress overcomes the instantaneous external drive, and negative applied K can drive true compression-induced return. No reverse nucleation channel has been introduced.

## Minimal added state physics

1. **Left-boundary mobile outflow** is classified as return to the crack/free surface and annihilation there.
2. **Right-boundary mobile outflow** is recorded separately as far-field escape.
3. **Retained/tangled content is never removed by the surface-return rule.**
4. Returned mobile line content cancels an equal amount of the near-tip source-slip/blunting ledger, bounded by the available uncancelled slip.
5. The historical cumulative source-slip ledger remains intact as a cumulative activity measure; a separate returned-slip field makes blunting depend on the nonnegative net ledger.

Reversibility therefore emerges from competition between signed transport and the pre-existing mobile-to-retained encounter kinetics. There is no prescribed `reversible_fraction(DeltaK)`.

## Diagnostics

The reversible state records:

- cumulative/interval returned mobile line content;
- cumulative/interval far-field escape;
- cumulative/interval source-slip cancellation;
- cumulative source, returned and net slip counts;
- return and escape fractions relative to emitted line content;
- minimum and maximum signed transport K and effective transport stress;
- fraction of transport channel-time in the reverse direction;
- total mobile exposure and reverse-mobile exposure;
- reverse-mobile exposure fraction.

The explicit-cycle history also records the instantaneous signed transport K, transport-stress extrema, emitted-sign velocity extrema, the spatial fraction of emitted-sign transport directed back toward the surface, and the mobile population present while that reverse direction is active.

## Qualification path

The first qualification uses **explicit physical cycles** only. This is intentional: the new state transition must be established before the DMD/projective high-cycle map is allowed to use it.

Use `scripts/run_v914_minimal_reversible_case.py` with `--model baseline` and `--model reversible` to generate matched A/B cases with identical candidate, loading, seed, phase resolution, and crack-growth target.

Initial scope:

- T = 300 K;
- R = 0.1;
- frequency = 1000 Hz;
- seed = 1720;
- canonical Peak-T, DBTT, weak-T, and ceramic-like parameterizations;
- identical load points taken from the completed fatigue study.

The first scientific diagnostic is not simply whether `da/dN` changes. It is whether reverse transport overlaps a nonzero mobile population, produces physically resolved surface return while retained content remains irreversible, and changes the cycle-integrated hazard/state in a way that explains any change in local or developed Paris slope.

## Required checks before acceleration

Before extending the fast-cycle/DMD operator, require:

1. a no-reverse/no-return case reproduces the baseline trajectory to numerical precision;
2. signed negative transport K is preserved rather than clipped;
3. reverse emitted-sign velocity produces positive surface return and no artificial far-boundary loss;
4. retained content is unchanged by the surface-return rule except through pre-existing Taylor/release kinetics;
5. the returned-slip field never exceeds the cumulative source-slip field in net effect;
6. checkpoint/restart reproduces reversible ledgers and RNG/event history;
7. same-cycle post-event waveform continuation remains exact;
8. A/B comparisons use identical cleavage/emission physics and stochastic inputs.

Only after these pass should the projective high-cycle map be taught the new reversible state, followed by R-ratio and non-Schmid/tension-compression-asymmetry studies.
