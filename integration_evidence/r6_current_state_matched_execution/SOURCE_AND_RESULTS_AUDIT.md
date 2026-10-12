# R5 independent source and results review

## Scope

Reviewed the user's R5 report/transcript, four raw JSON result files, JUnit record, portable review tar, and incremental Git bundle. Extracted the 16 packaged Python source/test files, exact donor archive, and R4 preflight source archive. Verified all 10 entries in R5 `PACKAGE_SHA256.txt`; independently parsed the 810/0/0/0 supplied test record. No native material trajectory, finite-element solve, or repository test suite was rerun.

Executed (1) arithmetic comparisons of the raw results, and (2) a synthetic-input dependency test of the **verbatim R5 DynamicMechanicsProvider** with the actual shipped `BoundedPreconnectionSubgridClosure` and `PhysicalIncrement`. This tests function dependencies, not material physics or a new closure. Also exercised the verbatim first-event callback with a manufactured crossing.

The incremental Git pack requires base objects not all supplied by the source snapshot; independent unpacking reported five unresolved deltas after available R4 objects were seeded. The tar's source was reviewed as supplied; do not describe this as a complete Git ancestry verification. The tar and bundle SHA-256 values are recorded in `review_input_hashes.json`.

## Main conclusions

R5 restores dynamic cavity lifecycle ownership and fixes the overwritten spatial shear signs, but does not yet close the bidirectional current-fracture/plasticity/void-mechanics feedback. A geometry-rich cache identity is not a geometry-dependent mechanical law. The actual donor has an explicit zero far-field subgrid correction and the new wrapper supplies prescribed historical plasticity/site controls rather than the current fracture owner's accepted plasticity. Retain these as explicitly named functional controls; do not equate them with restored current-state feedback.

The outer source-integration test is constructive but is not installed as an error-controlled production update. The remaining replay error has not been causally localized. The fatigue runs vary two numerical inputs but have unequal terminal horizons, a fractional-time reporting error after same-phase continuation, and no output-error acceptance criterion.

## 1. Dynamic provider and its actual consumers

Sources (line numbers refer to the original supplied files):

- `arrhenius_fracture/dynamic_void_execution_r5.py`, lines 103–144: cache key contains crack extension/load/site radius, but `closure.evaluate` receives only fixed opening, clipped historical strain, one site ID and radius. Crack extension does not enter that evaluation.
- Same file, lines 146–200: `dep[0] = history_rate_s * duration_s`; support strain rates, forest/mobile populations, stresses and direct opportunities come from donor interpolation. The current `fracture_adapter_state.engine` is not read to generate these quantities. Applied K scales only `fracture_native_K_scale`; it does not update the site tensors or plastic increments.
- Donor `reduced_fracture_v3/aligned_subgrid_closure.py`, lines 82–108: `G_kinetic=interpolate(self.no_void_G_J_per_m2)` regardless of cavity radius; radius only enables a site tensor. The qualification explicitly says `SUBGRID_ZERO_FAR_FIELD_CORRECTION`.
- The wrapper clamps the strain-history input to the interpolation range (lines 122–124 and 148–150), bypassing the donor's out-of-range rejection. A new cache key for an unsupported state cannot justify this extrapolation-by-clamping.

### Synthetic dependency probe

The actual class and closure were evaluated on positive synthetic interpolation data. At identical historical strain, no cavity, a 42 nm cavity, a 1 µm cavity, and the 42 nm cavity after 5 µm crack advance all give the same G and fracture-drive scale. Changing only the current native activity also leaves every output unchanged. Changing the requested K from 5 to 51.1005 MPa√m changes the fracture drive but leaves site tensors, direct opportunities and plastic increments unchanged. An out-of-range history is silently clipped.

The synthetic values are deliberately not material data and are not proposed as a physical law. These invariances also follow directly from the supplied source, independently of the numerical values used for the probe.

### The supplied active/passive states agree exactly before the pulse

At physical time 2.249990000000002 s, `C_active_dynamic.fracture_pulse_before` and `D_passive_partition_control.terminal` have identical reported physical summaries after excluding their mechanics-enabled flag/receipt. The fracture hashes are both:

`06634369f8924f4ea9cb57d270e6d5841b860fd29fd5b276566bcb1d4c39e460`.

Both have 361 fracture-update intervals, zero fracture events and a 42.1920286 nm stable cavity. The passive arm stops here. Only the active arm receives the 51.1004675 MPa√m, 1 µs pulse followed by a 50 ms continuation. Thus this result does not contain a matched active/passive fracture-event comparison. The pulse-induced event is a valid reported functional event, not evidence of a cavity-induced fracture shift.

A zero correction for sufficiently small subgrid cavities may be an intentional valid approximation. It must remain named and bounded as such. Do not add a positive correction just to obtain a nonzero effect. Use an existing spatially supported finite-effect fixture to test the missing feedback path.

### Current and inherited plastic state remain separate

The current fracture engine contains signed source/mobile/retained/slip state. The donor line fields are advanced from the prescribed strain-rate and density tables. A common dataclass/global scheduler does not by itself make these one physical source history. Cavity-generated loop contributions enter donor fields, but their transfer into the current MPZ is not established by the cache receipt. Preserve independent bulk states where physically intended, with a current mechanical/constitutive map; do not replace them with a second replayed source history and call that current-engine feedback.

The donor near-third-hit fixture also uses vacancy thresholds `(1e100,2e100)` and inherited direct hits/actions. It is a pathway-isolation fixture, not an unbiased vacancy/direct competition experiment. Nonzero vacancy action verifies evaluation, not vacancy-birth reachability. Retain it; separately retain/execute the existing vacancy-active fixture.

## 2. Event callback shortcuts

`dynamic_void_execution_r5.py`, lines 264–300, skips the fracture locator for K <= 12 MPa√m and duration <= 1 s without checking the pending mark/state. For any detected crossing in a maximum horizon <= 1 µs, it returns the end of that horizon rather than the crossing time.

The native defaults and selected reduced builder use `min_substep_s=1e-15`; the hard-coded 1 µs branch is not obtained from that setting. A manufactured threshold at 0.25 µs within a 1 µs requested interval is returned as 1 µs by the verbatim callback. This demonstrates a generic ordering risk, **not an error estimate for the reported saturated unit-mark pulse**.

Use an accepted-state event bound or actual native consumed time/root localization. Retain competing-process state and consumed/unused time exactly. A performance optimization is acceptable when its mathematical/state-domain preconditions are checked and serialized; a constant load/time threshold is not a substitute for them.

## 3. Outer source-coupling refinement

The exact supplied constant-K results are monotone:

| Outer slices | N_em | r_eff (µm) |
|---:|---:|---:|
| 1 | 4360.14697 | 1.7560064 |
| 2 | 3418.29305 | 1.5926982 |
| 8 | 2267.47932 | 1.3931585 |
| 32 | 1966.47242 | 1.3409669 |
| 64 | 1928.12745 | 1.3343183 |
| 128 | 1913.80944 | 1.3318357 |

The final population change is 0.748%, radius change 0.186%, and action changes from 1.52642e-14 to 2.69379e-14. The latter is 43.3% relative but only 1.16737e-14 absolute: both are far from first passage. The coarse population is 127.8% above the finest result. The finite-scale reference is not the exact asymptotic solution.

`state_correspondence_r5.py` adds a fixed maximum outer-step helper and a diagnostic sweep. Its `passed` flag checks event freedom, monotonic population refinement and recovery of old numbers; it does not impose a converged output tolerance. The dynamic `_preview` and mechanical replay still call `advance_signed` directly. The 8.4 s first replay row remains approximately 4360, so the production correspondence path is not using the 128-slice correction.

Implement shared error-controlled accepted updates at the meaningful source/state refresh level, or a fixed cap explicitly validated for each execution domain. Carry source/return/transport, radius, waveform endpoints and mark ownership in the error measure. Do not turn 128 into a universal physical constant, and do not assign huge relative action errors to a numerically negligible action.

## 4. The residual reduced/spatial mismatch is not fully explained yet

The supplied spatial/full-input-replay N_em values after the first 30 µm stage, unloading and reloading are 147.720/65.500, 115.934/24.648, and 102.476/21.251, respectively. These are -55.66%, -78.74% and -79.26% errors. Full sign/factor input changes the reduced K-only result very little in those stages.

`mechanical_correspondence_r5.py` `_run` (lines 28–45) consumes K, duration, drive factors and signed shear. The listed opening, projected tip and path-orientation fields are not installed by that routine; `path_angle_deg` is generated as 0 for every row. The booleans for spatial remap and re-equilibrated source geometry are false. Thus the file correctly flags noncorrespondence but labels its cause without an operator-by-operator decomposition.

These are hypotheses to test, not proof of a unique cause. Use the existing low-level state-record/replay machinery. Locate the first divergent accepted update by comparing pre-source, post-source, post-transport, post-continuous-translation, geometry-admission/remesh and re-equilibrium states under identical consumed inputs and common random owner. Log path tangent/crystal basis and frame transforms rather than applying a guessed rotation. Separate native constitutive replay from the autonomous reduced mechanical approximation. Do not copy spatial populations into the reduced arm or refit them away.

## 5. Fatigue refinement and time reporting

First-event cycles for 12/24/48 phases are 1.00253627085, 1.00091061779 and 0.99620954928. The 24→48 difference is 0.00470107 cycles (0.472%), larger than the previous 12→24 difference. The fixed-24 native action-limit change 0.01→0.005 moves the first event by only 4.52125e-7 cycles. This is useful separate sensitivity evidence but is not an accepted accuracy bound for the phase resolution.

The stop rule is `floor(first_event_cycles)+2`. Consequently the 12/24 phase arms end at 3 cycles while the 48-phase arm ends at 2. Their terminal states are not comparable as a convergence triple. A common 2-cycle observation can be recovered from the raw rows; the 24/48 continuous advance differs by about 0.95% there, while their common checkpoint reward is identical. Keep observation time, continuous micro-advance and completed reward separate.

After an event in a phase bin, `physical_elapsed_cycles` is calculated from only the current call's consumed fraction and ignores previously consumed time in that same bin. The 48-phase row immediately after the first event reports time moving backward. Maximum reported-time discrepancy against cumulative accepted durations is 0.0170429 cycles. This is a reporting-formula defect; raw consumed times are still available. The first-event times above are unaffected because there was no earlier event in their phase.

The final `passed` only requires an event and subsequent reload, not small refinement errors. Report `REFINEMENT_EXECUTED` separately from `ACCURACY_QUALIFIED`. Use a common physical endpoint and a state/event/output tolerance. Keep deterministic versus sampled marks and continuous versus event-conditioned translation separately named; do not change an event strategy to improve apparent convergence.

## 6. Reproducibility limits

The package includes the exact donor Python source but not the observation JSON or the donor `declared_inputs.json` read from machine-local absolute paths. `dynamic_void_execution_r5.py:OBSERVATIONS` and donor `_initial_state` retain these dependencies. The new dynamic execution also does not export a complete restart checkpoint; its JSON is a summary with a fracture-state hash. The reported rollback compares `_summary` hashes, not all owning arrays/RNGs directly. This is useful but narrower than independently reproduced whole-state restart equivalence.

Bundle the exact inputs and full accepted-state restart needed by the dynamic test. Do not fabricate them from the summaries. No remote push is required. A test log documents the author's run; it does not establish that every physical integration requirement is tested.

## 7. Recommended continuation

Keep R5's source-owner integration, context-local geometry queues, signed spatial fix and historical functional controls. Do not start a new seam model or parameter search. Close the existing current-state dataflow, supported mechanics response, common state/time integration and operator correspondence. Reuse accepted R3 connection fixtures where untouched. Require one same-owner active dynamic cavity/fracture trajectory in a supported mechanical domain, with matched passive/no-void controls, accepted advance, downstream history and full restart. No preferred effect sign or magnitude is required; a true zero-effect nanocavity case remains valid, but it cannot by itself qualify an inactive feedback implementation.