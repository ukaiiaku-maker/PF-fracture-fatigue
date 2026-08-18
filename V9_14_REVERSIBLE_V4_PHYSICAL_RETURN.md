# v9.14 reversible fatigue v4: physical-return correction

## Why v4 is required

The v3 branch correctly distinguished true cyclic stress reversal from the raw sign of a Burgers-population spatial velocity, but it intentionally retained the v2 boundary-cancellation physics.  A weak-T, `f=1.20`, `R=0.1`, 50-cycle qualification exposed the inconsistency: true reverse-drive fraction and reverse-driven return were both zero, yet raw left-boundary outflow was about 21.6% of emitted line content and the code used that raw flux to cancel about 716 source-slip counts.  This reduced the tip radius and increased cleavage hazard despite the absence of physical reverse loading.

## v4 rule

Raw left-boundary mobile outflow remains a diagnostic transport quantity.  It changes the emission-linked blunting ledger only if all of the following are true:

1. the returning population is the Burgers-sign population emitted by the tensile crack-tip source on that slip system;
2. the effective transport stress at the crack/free-surface boundary is reversed relative to the positive-tension resolved-stress direction for that system; and
3. positive left-boundary outflow is present.

Therefore

`physical return = emitted population AND true reverse drive at x=0 AND left-boundary outflow`.

Only physical return increments `returned_slip_m2` and cancels source-linked blunting.  Raw left-boundary outflow from any other circumstance is diagnostic only.

## Physics unchanged

v4 does not change the cleavage barrier, cleavage multi-hit law, emission barrier, source multiplicity, persistent backstress law, signed Peierls mobility, Taylor/encounter kinetics, event-length law, stochastic thresholds, fatigue waveform, or crack-advance rule.  It changes only which boundary fate is permitted to reverse the emission-linked tip-slip ledger.

## Checkpoint policy

v4 rejects v2/v3 reversible checkpoints because those checkpoints may already contain returned-slip cancellation generated under the older non-fail-closed semantics.  New v4 runs must start from a clean baseline state or a v4 checkpoint.

## Qualification gate

At `R=0.1`, any case with zero true reverse-drive fraction must satisfy exact baseline/reversible parity in tip radius, cumulative cleavage hazard, event history, and extension, regardless of nonzero raw left-boundary outflow.  At negative R, differences are allowed only when the emitted population experiences true reverse drive at the surface and generates physical return.
