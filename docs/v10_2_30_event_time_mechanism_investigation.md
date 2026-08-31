# Gate S0 investigation: existing first-passage event-time mechanism

## Finding

`StochasticHazardDiagnosticTipEngine._integrate_coupled` (`stochastic_hazard_tip.py:140-315`)
is a Strang-split adaptive-substep loop (`while remaining > 0.0`, lines 172-299). For each
substep of size `h` (chosen by `_substep_limit`, line 192), it evaluates a stress
(`sigma_tip(K)` unless `stress_override` is given), converts it to a cleavage rate via
`lambda_cleave` — **unless `lambda_override` is provided, in which case that fixed scalar is
used verbatim at every substep** (lines 187-189, 203-205, 228-230) — and accumulates
`dB = progress_mid * h` into `self.B` until `self.B >= 1.0` (line 282), at which point `fired =
True` and the loop breaks. `consumed` (returned as `dt_consumed`, line 308) is the sum of all
substep `h` values up to that point.

When called from the cyclic-fatigue path (`kinetic_tip_cell.py:cycle_step_waveform`, line 461,
via `_integrate_coupled(..., stress_override=avg_sig, lambda_override=lambda_avg)`), both
`avg_sig` and `lambda_avg` are **single scalars, precomputed once per block** from the
representative phase-resolved rate array (`kinetic_tip_cell.py:454-459`). This confirms
directly: for the cyclic path, `_integrate_coupled` finds the exact crossing time under a
**locally-homogeneous (block-constant) rate assumption** — the adaptive substepping refines
numerical accuracy of the renewal-progress transform (`normalized_progress_rate`), not a
time-varying rate, since the rate itself never changes across substeps within one call.

## Implication for the Gate S4 rebonding-coupled event-time root-finder

Rather than reimplementing a parallel cleavage-action accumulator (as originally sketched), the
root-finder should **reuse this existing loop as its trial evaluator**:

1. Obtain the baseline (uncoupled) `dt_consumed_0` by calling the *existing, unmodified*
   `_integrate_coupled` with the block's ordinary `lambda_avg` — this is simply what RB0
   already computes, untouched.
2. Recompute a rebonding-biased representative rate `lambda_avg'` using the representative-cycle
   Strang trajectory (Gate S3), centered appropriately within the trial interval
   `[0, dt_consumed_0]` (e.g., at `dt_consumed_0 / 2`) rather than the original nominal block.
3. Call `_integrate_coupled` again with `lambda_override = lambda_avg'` (dt still the full
   nominal block, so the internal loop can fire early) to obtain `dt_consumed_1`.
4. If `dt_consumed_1` and the interval used to build `lambda_avg'` (`dt_consumed_0`, or the
   previous iterate) disagree by more than a configured tolerance, recompute `lambda_avg'` for
   the new interval and repeat (bounded iterations, e.g. 10).
5. Converge when the interval assumed for the representative rebonding state and the elapsed
   time that state implies are mutually consistent.

This satisfies the plan's requirement to layer a correction onto the existing computation
rather than replace it: RB0 never calls this iteration at all (it only runs when
`rebonding.enabled`), and every trial evaluation goes through the same unmodified
`_integrate_coupled`/`_substep_limit`/`normalized_progress_rate` machinery RB0 uses, just with
a rebonding-biased `lambda_override` fed in from the outside. Exact RB0 parity is preserved
because `_integrate_coupled` itself is never edited — only the value passed as
`lambda_override` differs, and only when rebonding is enabled.
