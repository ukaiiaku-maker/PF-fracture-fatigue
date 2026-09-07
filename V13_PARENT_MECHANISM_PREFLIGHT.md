# V13 parent-mechanism prerequisite audit

Disposition: `BLOCKED_PARENT_MECHANISM_CONTRACT_MISMATCH`.

Permanent boundary: `BRANCHING_KINETICS_MODEL_UNCALIBRATED`.

The requested preliminary audit exposes a mismatch between the proposed immutable parent race and the existing production implementation, together with a concrete emission-method restoration defect. No V13 branch overlay has been implemented or qualified. No physical ensemble, FEM solve, branch draw, trajectory continuation, or family rebuild was performed.

## Evidence and scope

Baseline source: `b3d0add6cbb0605adaa3e04006fe987961ad6452`, descended from the qualified wake-remap source and completed V3 atlas execution lineage.

Audit producer: `a213b21`; its full commit and script SHA-256 are recorded in [the machine-readable audit](analysis_outputs/v13_parent_mechanism_preflight/parent_mechanism_audit.json). The producer checks each cited production file against the baseline Git blob before accepting the source evidence.

The audit reads the eight pre-wake-remap source checkpoints and the eight corrected terminal checkpoints identified by the authoritative V3 terminal index. These yield 64 front records and 48 process-engine records. Pre-wake-remap records remain historical defect evidence. This sample does not claim to supply an independent canonical single-front before/after parity replay.

All 16 source files pass their published SHA-256 checks before and after inspection. Accepted runtime, actions, thresholds, ordinals, pending events, lineage RNG, and engine RNG remain unchanged. Complete competition states, complete engine field inventories, cached emission diagnostics, accepted FEM/stress identities, and checkpoint hashes are exported. Raw and effective cleavage evaluators repeat exactly at each frozen engine's cached owner opening; this limited repeatability check is explicitly not a V13 parity pass or a new directional tensor evaluation.

## 1. The proposed joint mechanism race does not exist in this baseline

[UnifiedMPZFrontEngine](arrhenius_fracture/unified_front.py) documents concurrent cleavage and plasticity evolution. The persistent-source implementation integrates backstress-limited emission activations into mobile/signed populations. It has emission rates, but it does not create competing emission first-passage actions, thresholds, event ordinals, or an emission winner in the V12 topology scheduler.

[preview_competitions](arrhenius_fracture/multifront_competition_v12.py) previews directional cleavage clocks. [The accepted-interval driver](arrhenius_fracture/stateful_multifront_production_v12.py) selects the topology proposal from those clocks and separately evolves each physical process owner over the accepted interval. The 64 inspected front records contain zero noncleavage scheduler clocks. The legacy engine hazard RNG and thresholds belong to stochastic cleavage, as documented by [stochastic_hazard_tip.py](arrhenius_fracture/stochastic_hazard_tip.py); their presence is not evidence of emission clocks.

Consequently, a new discrete race with `Lambda_base = sum(lambda_C) + sum(lambda_E)` and mutually exclusive cleavage/emission winners would change the implemented emission lifecycle. It cannot also be certified as exact preservation of the existing process engine. This finding does not imply that dislocations are absent from the intended physical model: continuous emission, shielding, blunting, and transport still couple to cleavage through state evolution.

## 2. Restoring fields does not restore the installed emission method

The persistent-source installer explicitly binds `_persistent_emit` to `mpz._emit`. It declares persistent sites with no source-inventory depletion. [The capture helper](arrhenius_fracture/sharp_front_v11_branching.py) excludes callable attributes from the saved fields. [restore_complete_current_source_engine](arrhenius_fracture/current_source_multifront_hooks_v12.py) allocates the engine and MPZ with `__new__`, then restores serializable fields without rerunning the method installers.

All 48 inspected restored owner states therefore have this combination:

| Property | Restored value |
|---|---|
| Serialized source model | `persistent_areal_sites_backstress_limited_no_inventory` |
| Expected installed emission callable | `persistent_site_source_v10221._persistent_emit` |
| Actual restored emission callable | `unified_mpz.UnifiedMPZState._emit` |

The actual fallback callable consumes `available_sites` and updates the base mobile population. The intended callable uses the persistent-site multiplicity and implicit backstress closure and updates signed populations. These are different production paths despite matching serialized source-model metadata.

A minimal capture/restore regression reproduces the lost binding without a FEM solve, material initialization, random draw, or physical advance. The assertion that the persistent binding survives is marked strict expected-failure while this demonstrated defect remains unresolved. No restore patch was applied during this prerequisite audit. Other dynamically installed methods require review alongside this defect; their quantitative impact has not been measured here.

The earlier atlas checkpoint-versus-portable-field checks establish agreement between stored data representations. They do not establish parity with the fully initialized persistent-source engine. The historical atlas remains immutable, but it must not be used as proof of emission-law preservation for V13.

## 3. The existing correlation selector can delay the proposal endpoint

An executable fixture calls the unmodified production preview with a cleavage rate of 1/s, threshold action 1, and correlation interval 0.25 s. The cleavage clock completes at 1 s. With correlation enabled there is no one-arm proposal at that instant; the proposal becomes eligible at 1.25 s. The runtime and RNG are unchanged by the preview.

Thus raw clock completion and accepted topology-proposal eligibility already differ in V12. Replacing this correlation selection with a mark at the earliest cleavage completion can change the existing endpoint. The V13 contract must identify which single-arm parent endpoint it preserves, and test that endpoint explicitly. A branch-opportunity lifetime must not be silently added to physical time.

## Validation and unfinished work

Focused command:

```text
python -m pytest -q tests/test_v13_parent_mechanism_preflight.py tests/test_directional_competition_transactions_v11.py tests/test_multifront_checkpoint_output_v12.py
```

Result: **15 passed, 1 expected failure**. The expected failure is the newly demonstrated persistent-emission binding loss; it is not a qualification pass. The two new audit files compile, and `git diff --check` passes.

V13-disabled before/after parity and V13-enabled precleavage parity are both **not run**, because no V13 overlay exists. Companion Arrhenius clocks, branch-only barriers, branch RNG, exact pair-to-single fallback integration, frozen-state branch Monte Carlo, and the bounded physical ensemble remain unimplemented. No branch probabilities or campaign results are claimed.

## Recommended decision

Preserve the intended continuous-emission process engine and directional cleavage clock architecture as the V13 parent. First repair the demonstrated restore-binding defect and compare that restored engine with the fully initialized intended production engine. Then introduce the conditional branch mark after the explicitly chosen single-arm parent event, retaining the requested independent randomness, exact pair acceptance, and single-arm fallback.

This requires a scientific clarification of the requested immutable parent, because the pasted specification instead requires a discrete cleavage/emission winner race and exact equality with existing execution. Those requirements cannot all hold for the inspected implementation. It also requires distinguishing the intended canonical engine from the historical trajectories executed through the defective restore path. The existing trajectories should not be rerun automatically.
