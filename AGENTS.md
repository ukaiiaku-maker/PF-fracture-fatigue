# Codex Project Instructions

## Read first

Before modifying code, read `CODEX_HANDOFF.md` completely and inspect the current Git status, branch, and HEAD.

## Scientific objective

Develop and qualify the existing PF/sharp-front Arrhenius-hazard fatigue framework so it can compute event-resolved and developed crack-growth rates, `da/dN` versus fixed local `DeltaK`, for the four existing production parameterizations across low-cycle fatigue, high-cycle fatigue, and very-high-cycle fatigue.

The main production target is approximately 100 micrometres of crack extension per active trajectory. The `1e12`-cycle value is only a maximum censor, not a required endpoint.

## Immutable physics and parameterization constraints

Do not change, refit, rescale, replace, or silently reinterpret the four production parameter rows.

Do not introduce a Paris law, empirical fatigue-growth law, independent fatigue barrier, fitted fracture coefficient, arbitrary toughness floor, athermal fracture criterion, or `Gc0_athermal`.

Cleavage first passage is the only stochastic crack-event trigger. Each event interval uses an independent unit-exponential integrated-hazard threshold. Preserve the existing RNG provenance and common-random-number comparisons.

Preserve the current threshold-correlated, bounded, mean-preserving stochastic event-length proposal unless a narrowly scoped bug fix is required to make the existing formulation execute as intended.

Preserve the post-first-passage energy gate. The continuum `K^2/E'` quantity is diagnostic only and must not veto or rescale first passage. Geometry may advance only through the existing checked sharp-wake transaction after the event-energy balance admits an increment.

Preserve the persistent-site closure: no finite source inventory, source exhaustion, arbitrary refresh, or explicit recovery law. Preserve evolving mobile/retained MPZ state, backstress, shielding, transport, blunting, and moving-frame translation.

Do not change material barriers, entropies, stress scales, material rows, crystal orientation, mesh-independent crack increment, seed mapping, or the intended fixed-`DeltaK` loading merely to obtain a desired trend.

## Numerical-design requirements

Accelerate only the waiting cycles between physical crack events. Exact or independently validated integration must localize first passage.

After every committed geometry event, invalidate any geometry-specific high-cycle representation, rebuild it at the new crack tip, and continue to the next event.

Support efficient progression from LCF through VHCF without changing physical rates. Numerical acceleration must be validated against exact-cycle evolution and must fail closed.

Maintain atomic live checkpoints containing active MPZ state, cumulative ledgers, hazard state, sampled threshold, RNG state, geometry signature, and high-cycle cache.

A watchdog timeout or numerical rejection must leave a diagnostic checkpoint and human-readable plots.

## Required outputs

For every committed event record cumulative cycles, projected extension, path-length extension, event proposal, energy-truncated length, geometry result, threshold, physical hazard action, MPZ state, backstress, shielding, and restart provenance.

Report event-level projected `da_x/dN`, event-level path-length `ds/dN`, tortuosity, moving-window and cumulative developed-growth rates, initiation-cycle distribution, censoring status, stochastic event probability and seed provenance, and visual crack-path/mechanical/MPZ diagnostics.

## Validation practice

Add a regression test for every observed production failure before or with the fix.

Run the narrowest relevant tests first, then the focused v10.2.30 gate. Do not weaken tolerances merely to make tests pass.

Never claim a physical result from a run that did not write a complete event/audit record or a valid live checkpoint.

Keep generated runs under `runs/`; do not add run outputs to Git.

## Git practice

Work only in this Codex copy and on the Codex branch. Keep changes small and reviewable. State explicitly whether each change affects physics, numerical representation, diagnostics, or launch tooling. Prefer separate commits for physics-neutral bug fixes, numerical acceleration, tests, and analysis tooling.
