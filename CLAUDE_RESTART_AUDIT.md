Continue development from HEAD 7815161f32726d35959ae77b27ba19a393c9393d.

Work only in:

/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_codex_v10_2_30

Use only:

arrhenius-sharp-front-v10-codex

The qualification supervisor is committed and its control-plane smoke tests pass. Do not launch the full 12-case matrix yet.

## Blocking objective

Implement physically valid, fail-closed restart after one or more committed crack-growth events.

The current kinetic checkpoint can restore the internal high-cycle and stochastic state, but the outer 2-D driver cannot yet reconstruct the matching committed crack geometry before restoration. Fix this without changing fracture physics, stochastic provenance, event-length physics, energy gating, or geometry-transaction semantics.

First read the Claude restart audit and inspect the current implementation. Briefly report the repair architecture, then implement and test it. Do not stop after planning.

## Required restart contract

A checkpoint may be considered physically restartable only if it can reconstruct all state required to continue the same trajectory, including:

* exact total cycle count;
* current crack-tip coordinates;
* total and projected crack extension;
* committed crack path;
* wake or opened-interface geometry;
* any front-segment or active-interface state;
* geometry generation or transaction index;
* final physical checkpoint length;
* material option and loading state;
* hazard threshold;
* accumulated physical hazard action;
* B;
* event index and event history;
* RNG state;
* energy-gate state required for continuation;
* acceleration-model state where safe and meaningful;
* checkpoint schema version;
* Git and run provenance.

Do not infer committed geometry solely from total scalar crack extension when the path history or wake is required.

## Architecture requirements

Prefer an authoritative versioned run-state checkpoint containing both:

1. outer-driver geometry state; and
2. kinetic/stochastic engine state.

Restore the outer geometry before restoring or advancing stochastic state.

Required initialization order on restart:

```text
parse and validate checkpoint
→ validate option/loading/seed/schema compatibility
→ reconstruct committed outer geometry
→ rebuild dependent fields and tip state
→ set final da_phys/checkpoint geometry
→ restore kinetic and stochastic state
→ verify cross-layer invariants
→ resume cycle evolution
```

Keep restart fail-closed. Never silently start from the initial geometry when a post-event kinetic checkpoint is present.

Use atomic checkpoint replacement so an interrupted write cannot produce a checkpoint that appears valid.

## Cross-layer invariants

Before resuming, assert at minimum:

* restored driver crack tip matches checkpoint tip;
* restored total/projected extension matches event history;
* restored event count matches committed geometry events;
* restored avalanche base checkpoint matches final da_phys;
* threshold and RNG state are restored exactly;
* B and physical hazard action are mutually consistent;
* current event proposal corresponds to the restored threshold and checkpoint base;
* cycle count is monotonic;
* loading option, temperature, ΔK, R, frequency, and seed match the checkpoint;
* no uncommitted geometry transaction is treated as committed;
* checkpoint provenance belongs to the current case.

## Required tests

Add focused tests for:

1. pre-event checkpoint restart;
2. restart immediately after one committed geometry event;
3. restart after multiple committed events;
4. interruption during an uncommitted geometry transaction;
5. truncated or corrupt geometry checkpoint;
6. incompatible option, seed, loading, or checkpoint schema;
7. mismatch between kinetic event history and outer geometry;
8. preservation of threshold and RNG state;
9. preservation of exact cycle accounting;
10. supervisor classification of physically restartable versus nonrestartable cases.

## Real restart-equivalence qualification

Use the existing event-producing weak-T 0.95 configuration and fixed seed.

Create two runs:

### Control

Run continuously through at least:

* two committed crack-growth events;
* one post-event high-cycle restart;
* a later common comparison checkpoint.

### Interrupted/restarted

Run identically but terminate gracefully after the first committed geometry event and after a valid combined checkpoint is confirmed.

Restart using the production supervisor/restart path and continue to the same comparison condition as the control.

At the comparison point, verify equality or explicitly justified numerical tolerance for:

* total cycles;
* event count;
* crack-tip position;
* total and projected crack extension;
* complete committed crack path;
* hazard threshold;
* accumulated physical hazard action;
* B;
* event history;
* RNG state;
* acceleration-mode audit;
* subsequent first-passage or event outcome.

The preferred result is deterministic trajectory equivalence. If any acceleration cache is intentionally discarded on restart, demonstrate that physical state and stochastic provenance remain identical and explain any performance-only divergence.

Also confirm that the restarted run did not duplicate an event, lose cycles, repeat geometry growth, or restart from zero.

## Watchdog review

Reassess the 300-second no-progress timeout using the real weak-T logs.

Ensure a healthy but long:

* exact event localization;
* energy-gate evaluation;
* geometry commit;
* checkpoint write;
* restart reconstruction;
* analyzer pass

cannot be terminated merely because the usual cycle heartbeat pauses.

Add phase-aware heartbeat or an appropriately conservative timeout if needed. Do not fake progress; record the active physical/numerical phase explicitly.

## FAMILY_JSON and launch reproducibility

Replace the placeholder launch requirement:

```text
FAMILY_JSON=/absolute/path/family.json
```

with one reproducible mechanism:

* a committed immutable family manifest;
* a generated manifest with a tested deterministic command; or
* automatic extraction from the authoritative production shelf with exact-ID validation.

The eventual launch command must be directly runnable without an unspecified placeholder path.

## Validation and Git

Run:

* new restart tests;
* supervisor tests;
* event-growth tests;
* exact-burst tests;
* energy-gate tests;
* analyzer tests;
* shell syntax checks;
* the real restart-equivalence qualification.

Commit logical milestones separately, for example:

```text
feat: checkpoint committed outer crack geometry
fix: restore geometry before kinetic fatigue state
test: verify deterministic post-event restart equivalence
```

Do not commit run outputs.

## Stop point

Do not launch the complete 12-case qualification.

Stop after:

* the real post-event restart-equivalence test passes;
* the worktree is clean;
* all fixes are committed;
* the exact 12-case launch command contains no placeholder;
* monitor, stop, and restart commands are verified.

Report:

1. branch and HEAD;
2. commits created;
3. checkpoint schema and saved geometry state;
4. test results;
5. real restart-equivalence results;
6. watchdog decision;
7. exact qualification launch command;
8. exact monitor command;
9. exact stop command;
10. exact restart command;
11. remaining risks.
