# oneD V2 to cyclic-fatigue integration handoff

## Purpose

Carry the completed one-dimensional V2 screening result into the PF/sharp-front fatigue code **without replacing or weakening the v9.14 reversible-dislocation physics**.

This is not a direct copy of the monotonic reduced driver. The fatigue model has cyclic waveform state, signed transport, physical surface return, stochastic first-passage thresholds, cycle acceleration, post-first-passage energy gating, and atomic crack/MPZ transactions that must remain authoritative.

## Source result

- source local branch: `codex/oneD-v2-terminal-predictive-program`
- reported source HEAD: `5045fee`
- producer-code commit recorded by the source manifest: `cddc51605aee93d8ecbaa0dda76c12085e08f9de`
- results-record PR: `ukaiiaku-maker/Arrhenius_FEM_CZM_MPZ#65`

Final shared **screening** rows:

- Peak: `v913_zeroD_sobol_0242980`
- DBTT: `v913_zeroD_sobol_0202500`
- weak-T: `oneD_v2_focused_weak_T_0016`
- ceramic-like: `oneD_v2_focused_ceramic_like_0018`

The exact parameter vectors are in `data/oneD_v2_four_class_screening_registry.csv`.

## Transfer only shared material physics

Transfer the full-precision shared material coordinates:

- cleavage barrier, temperature dependence, activation stress, and shape;
- emission barrier, temperature dependence, activation stress, and shape;
- Peierls and Taylor barrier, entropy, and shape parameters;
- initial source density and Taylor-correlation parameters;
- blunting coefficient and shared source/process-zone geometric references already present in the material row.

Keep this registry versioned and separate from the historical production registry. Do not overwrite existing production options until cyclic integration and regression gates pass.

## Do not transfer monotonic backend reductions

The following oneD V2 quantities are backend reductions, not material constants, and must not replace the fatigue engine's native cyclic algorithms:

- monotonic event-length or crack-translation closures;
- monotonic hazard-progress or reload thresholds;
- monotonic renewal and avalanche-grouping rules;
- straight-path mechanics-map assumptions;
- exact-oracle cache or surrogate settings;
- PF-versus-FEM/CZM lifecycle correction parameters.

The fatigue engine remains owner of cyclic waveform state, signed transport, threshold lifecycle, threshold-correlated event length, the post-first-passage energy gate, sharp-wake geometry, atomic MPZ translation, checkpoint/restart, and high-cycle acceleration.

## Reversibility invariants that must be preserved

### Opening and transport are separate channels

- Cleavage remains opening-only.
- New dislocation emission remains opening-only.
- Already-mobile line content uses the **signed, unclipped** transport drive:

  `K_transport = K_applied - K_shield`

- Signed GND/internal stress is added to the transport stress.
- Signed Peierls mobility sets transport direction and speed.
- Never restore `max(K_transport, 0)` in the mobile transport channel.
- Do not add reverse nucleation as part of this registry handoff.

### Physical-return v4 rule

A left-boundary outflow reverses the emission-linked blunting ledger only when all three conditions hold:

1. the returning population is the Burgers-sign population emitted by the tensile crack-tip source on that system;
2. the effective transport stress at the crack/free-surface boundary is truly reversed relative to the positive-tension direction; and
3. positive left-boundary outflow is present.

`physical return = emitted-sign population AND true reverse drive at x=0 AND left-boundary outflow`

Raw left-boundary outflow without true reverse drive remains diagnostic only.

### Reversible and irreversible ledgers

- Returned mobile line may annihilate at the crack/free surface.
- Far-field escape remains a distinct fate.
- Retained/tangled content is not removed by the return rule.
- Physical return cancels source-linked blunting only up to available uncancelled source slip.
- Preserve cumulative source-slip, cumulative returned-slip, and nonnegative net-slip/blunting ledgers separately.
- Never allow returned slip to exceed cumulative source slip in net effect.

### Cyclic transactionality

Preserve:

- waveform phase and same-cycle continuation after a crack event;
- stochastic cleavage threshold and continuing RNG stream;
- event-length sampling and energy-gate semantics;
- one admitted event distance shared by geometry commit and MPZ translation;
- checkpoint/restart of signed state, cumulative ledgers, thresholds, RNG, waveform phase, geometry signature, and accelerator state;
- separation of reversible active state from monotone cumulative diagnostics.

## Recommended integration architecture

1. `oneD_v2_four_class_screening_registry.csv`
   - material rows only.
2. `FatigueMaterialRowAdapter`
   - exact, fail-closed field mapping from the V2 registry into existing fatigue barrier and MPZ configuration objects.
3. Existing fatigue engine
   - retains cyclic signed transport, reversibility, thresholds, event length, energy gate, crack transaction, and acceleration.
4. Optional V2 sensitivity metadata
   - informs screening priority only; it does not alter state equations.

Do not import the monotonic V2 lifecycle surrogate into fatigue.

## Required field-mapping audit

For every active registry coordinate record:

- V2 field and units;
- target fatigue field/function;
- exact conversion;
- classification as material physics, shared geometry, inactive, or unsupported;
- historical fatigue default;
- full-precision imported value;
- source and registry hashes.

Fail closed on any unmapped active coordinate. Keep backend-reduction fields out of the material adapter.

## Regression ladder

### Gate A: registry and monotonic identity

- Parse all four rows at full precision.
- Peak and DBTT remain bit-identical to their historical rows.
- New weak-T and ceramic rows retain the recorded monotonic bounded-PF anchors.
- Do not overwrite the historical registry.

### Gate B: no-reverse parity

At `R=0.1`, whenever the true reverse-drive fraction is zero:

- baseline and reversible models match in tip radius;
- cumulative cleavage action matches;
- event history and extension match;
- raw left-boundary outflow does not cancel blunting.

### Gate C: true reverse transport and physical return

Use negative R or a source-qualified internally reversed state to prove:

- negative `K_transport` is retained;
- emitted-sign mobile velocity reverses;
- physical return appears only under the three-condition rule;
- retained content is unchanged by surface return;
- returned-slip cancellation is bounded by cumulative source slip.

### Gate D: explicit-cycle versus accelerator parity

Before production DMD/Poincare acceleration:

- explicit and accelerated signed active states agree;
- returned and far-field ledgers agree;
- waveform phase agrees;
- hazard action and threshold crossing agree;
- the accelerator does not positivity-project signed reversible state;
- positivity preservation may remain for cumulative nonnegative ledgers only.

### Gate E: event/energy/geometry transaction

- first passage uses the continuing stochastic threshold;
- event length uses the threshold-correlated law;
- the energy gate truncates the proposed event without altering first passage;
- crack geometry and MPZ translation commit atomically using the same admitted distance;
- same-cycle continuation is exact;
- checkpoint/restart reproduces events and all reversible ledgers.

## Initial cyclic validation matrix

Start bounded, not broad:

- 300 K;
- fixed local `DeltaK`;
- `R=0.1` baseline plus at least one negative-R reversibility case;
- 1000 Hz initially;
- identical seeds across rows and matched load points;
- explicit cycles for reversibility qualification, followed by the qualified high-cycle accelerator.

Suggested cases:

- Peak and DBTT at one active and one near-threshold `DeltaK`;
- weak-T and ceramic-like at one discriminating active `DeltaK`;
- one negative-R case for Peak and weak-T to exercise return physics.

Analyze event-resolved and moving-window `da/dN`, physical avalanches, return/escape fractions, signed state, source/returned/net slip, tip radius, shielding, and hazard action. Do not embed a Paris law into the engine.

## Sensitivity guidance

The completed 972-case monotonic study found 84.8% overall cross-provider onset-sign agreement. Dominant directions agreed for cleavage and emission barriers, cleavage stress/shape, backstress, process-zone length, and blunting. Initial source density was the principal low-magnitude disagreement.

Use this information to prioritize cyclic screening, not to change the cyclic equations or freeze a fatigue calibration.

## Domain and interpretation

The screening rows were established for 300–1200 K and unconditional 0–100 µm monotonic screening. Qualified monotonic rate factors were PF 0.01×–100× and FEM/CZM 0.01×–1×.

The fatigue model has a separate cyclic domain. Monotonic screening does not validate cyclic reversibility, R-ratio dependence, VHCF acceleration, or a Paris window.

These rows are response-class screening parameterizations, not uniquely identified material constants.

## Branch policy

This handoff branch is based on `codex/v9.14-minimal-reversible-fatigue` at `acefa81ccf2ed398d4d207b52439cef9d313c63b`. Integration development should continue on a descendant such as:

`codex/v9.14-oneD-v2-fatigue-integration`

Do not port the rows into an older branch and manually recreate reversibility. Do not merge until all registry, return, transaction, and acceleration gates pass.

## Final acceptance

The integration is complete only when:

1. all four rows load at full precision;
2. no cyclic lifecycle or reversibility law is replaced by monotonic reduction logic;
3. no-reverse parity passes;
4. physical return passes under true reverse drive;
5. explicit-cycle/accelerated reversible-state parity passes;
6. event/energy/geometry transactionality passes;
7. the rows are available through a versioned fatigue registry;
8. provenance is recorded without modifying the historical registry.
