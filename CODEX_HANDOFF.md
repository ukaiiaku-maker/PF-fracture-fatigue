# Codex Handoff: PF/Sharp-Front Arrhenius-Hazard Fatigue and Developed da/dN

## 1. Mission

The project objective is to calculate crack-growth rate `da/dN` as a function of fixed local cyclic stress-intensity range `DeltaK` using four existing, fixed material parameterizations in the PF/sharp-front Arrhenius-hazard fracture framework.

The code must span low-cycle fatigue, high-cycle fatigue, and very-high-cycle fatigue. It must not accomplish this by adding a Paris law or a separate fatigue constitutive law. Cyclic crack growth must emerge from the existing state-dependent Arrhenius cleavage first-passage process, persistent-site plasticity, MPZ evolution, stochastic event lengths, and post-first-passage energy-gated geometry transactions.

For active cases, the desired production trajectory is approximately 100 micrometres of crack extension so that initiation can be separated from developed growth and the stability of `da/dN` can be assessed. A total-cycle ceiling of `1e12` is a censor for inactive or extremely slow cases, not the target of every run.

## 2. Codex working baseline

Source repository:

`/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1`

Intended branch at handoff:

`v10.2.30-hazard-energy-gated-fatigue-events`

Observed baseline HEAD at handoff:

`9e884fb0b0845da621d2612bdf1042e481b8df49`

Pull request:

`#51 — Validation v10.2.30 hazard-energy-gated fatigue events`

Conda environment:

`arrhenius-sharp-front-v10`

Editable package:

`arrhenius-sharp-front-mpz==10.2.30`

Before doing any work, verify the actual local branch and HEAD. The repository may have advanced after this handoff.

## 3. Four immutable production parameterizations

Use exactly these current v10.2.30 persistent-site options:

- Peak: `v913_paper_peak01_0242980_persistent_sites`
- DBTT: `v913_paper_dbtt01_0202500_persistent_sites`
- Weak-T: `v913_paper_weakT01_0129902_persistent_sites`
- Ceramic-like: `v913_paper_ceramic01_0077080_persistent_sites`

Do not substitute earlier parameter shelves with similar labels. Do not refit or alter any barrier, entropy, stress scale, reference temperature, site-density parameter, or other row value to improve numerical behavior or force a desired `da/dN` trend.

## 4. Current physical framework

### 4.1 Loading and mechanics

Current qualification scope is:

- temperature: 300 K;
- fixed local `DeltaK`;
- `R=0.1`;
- frequency initially 1000 Hz;
- crystal orientation 30 degrees;
- anisotropic cubic BCC tungsten elasticity;
- one nonbranching crack front;
- `sharp_wake` geometry backend;
- tip-only plasticity in the current fatigue branch;
- FEM supplies the held geometry/tensor probe, while the kinetic engine is the only crack-advance law.

The broader objective is not intrinsically restricted to 1000 Hz or one temperature, but those dimensions must not be expanded until the room-temperature fixed-`DeltaK` event-to-event implementation is qualified.

### 4.2 Cleavage first passage

Cleavage first passage is the only stochastic event trigger.

For every event interval, the code draws an independent unit-exponential integrated-hazard threshold:

`Xi = -log(U), U ~ Uniform(0,1)`.

The physical cleavage hazard action accumulates as `H`. The event occurs when `H >= Xi`. The normalized internal clock is `B = H/Xi`.

The event rule `B=1` is deterministic conditional on the sampled stochastic threshold; the threshold itself is stochastic. After a completed event, the action is reset and the next independent threshold is drawn from the continuing RNG stream.

No noise is added to `K`, barriers, source capacity, shielding, or material properties.

### 4.3 Stochastic event length

The unconstrained event-length proposal is threshold-correlated and mean preserving. The same sampled threshold controls waiting time and the bounded stochastic reward. The current mode is `threshold_scaled` with bounds `[0.5, 4]` relative to the physical checkpoint length and normalization that preserves the mean event length.

The mesh-independent base crack checkpoint is currently intended to be `da_phys = 5 micrometres`.

Do not replace the event distribution with arbitrary random noise or a fixed empirical `da/dN` increment.

### 4.4 Post-first-passage energy gate

First passage creates a stochastic event-length proposal. The proposed event is then truncated by the existing fixed-opening elastic-energy balance at the geometry-event `Kmax`.

The active resistance mapping is based on the existing hazard quantities:

`gamma_rel * m_hits * DeltaG_cleave_eff / b^2`.

The continuum `K^2/E'` comparison is diagnostic only. It cannot suppress, trigger, or rescale cleavage first passage.

Do not add `Gc0_athermal`, a generic fracture-energy floor, or a non-hazard athermal crack criterion.

### 4.5 Persistent-site MPZ state

The persistent-site model has no finite source inventory and no arbitrary refresh or explicit recovery. Persistent statistically independent nucleation sites emit signed line content. Mobile and retained content evolve through transport and storage, producing state-dependent backstress, shielding, and crack-tip blunting.

Crack advance translates the moving MPZ frame and transfers state behind the tip. Geometry and MPZ translation must remain atomic and use the same committed event distance.

## 5. Desired production workflow

For each parameterization and each selected `DeltaK`:

1. Initialize the fixed-`DeltaK` cyclic state with a declared seed.
2. Accelerate the waiting cycles using exact, periodic, stationary-tail, or independently validated projective evolution.
3. Guard every projected interval against crossing the remaining stochastic threshold.
4. Localize first passage with the exact existing stochastic integrator.
5. Form the existing stochastic event-length proposal.
6. Apply the existing post-first-passage energy gate.
7. Commit the checked sharp-wake geometry and MPZ translation atomically.
8. Record the event cycle, projected extension, path extension, event proposal, admitted length, and state.
9. Invalidate the old geometry-specific high-cycle model.
10. Rebuild the local cycle representation at the new tip.
11. Continue until approximately 100 micrometres of projected extension, `1e12` total cycles, or a declared numerical failure.

The scientific output is an ensemble of stochastic crack-growth trajectories, not one deterministic curve.

## 6. da/dN definitions and analysis

For committed event `i`, record cumulative cycle `N_i`, projected crack extension `a_i`, and actual path length `s_i`.

Event-level rates are:

`(da/dN)_i = (a_i-a_(i-1))/(N_i-N_(i-1))`

`(ds/dN)_i = (s_i-s_(i-1))/(N_i-N_(i-1))`

with tortuosity:

`tau_i = (s_i-s_(i-1))/(a_i-a_(i-1))`.

Initial analysis convention:

- first 20 micrometres: initiation and state-development interval;
- subsequent growth: developed-growth interval;
- final 50 micrometres: stability assessment;
- moving windows: approximately 20–25 micrometres.

Report event-level rates, moving-window rates, cumulative developed rates, initiation cycles, censoring, path tortuosity, and stochastic scatter.

Near the active/inactive transition, use multiple hazard seeds with the same seed set at every `DeltaK` for common-random-number comparisons. Begin with roughly 8 seeds per point and expand near the transition as needed.

## 7. Current high-cycle architecture

Important modules include:

- `arrhenius_fracture/persistent_site_high_cycle_engine_v10230.py`
- `arrhenius_fracture/persistent_site_high_cycle_engine_v10230_v5.py`
- `arrhenius_fracture/persistent_site_high_cycle_dmd_v10230_v4.py`
- `arrhenius_fracture/persistent_site_high_cycle_dmd_v10230_v5.py`
- `arrhenius_fracture/persistent_site_high_cycle_checkpoint_v10230.py`
- `arrhenius_fracture/persistent_site_high_cycle_state_v10230.py`
- `arrhenius_fracture/persistent_site_poincare_v10230.py`
- `arrhenius_fracture/persistent_site_periodic_solver_v10230.py`
- `arrhenius_fracture/persistent_site_forward_robust_v10230.py`
- `arrhenius_fracture/stochastic_hazard_tip.py`
- `arrhenius_fracture/stochastic_avalanche_tip.py`
- `arrhenius_fracture/stochastic_avalanche_backend.py`
- `arrhenius_fracture/persistent_site_cyclic_energy_gated_v10230.py`
- `arrhenius_fracture/persistent_site_cyclic_energy_gated_corrected_v10230.py`

The current production alias identifies the v5 event-to-event engine:

`v10.2.30_production_event_to_event_high_cycle_v5_rate_separated_positive_state_dmd`

The v5 design separates active-state DMD projection from cumulative ledger integration, uses positivity-preserving treatment for nonnegative MPZ fields, reuses validated local maps within trust regions, guards against projected first passage, writes atomic live checkpoints, and restarts acceleration after events.

## 8. Current launchers and analyzers

Key launchers:

- `scripts/run_v10_2_30_weakt_high_cycle_1e12.sh`
- `scripts/run_v10_2_30_weakt_0p55_high_cycle_1e12.sh`
- `scripts/run_v10_2_30_300K_four_class_fatigue.sh`
- `scripts/run_v10_2_30_four_class_three_deltaK_energy_gate_qualification.sh`
- `scripts/run_v10_2_30_three_deltaK_energy_gate_qualification.sh`

Key analyzers:

- `scripts/analyze_v10_2_30_high_cycle_live_checkpoint.py`
- `scripts/analyze_v10_2_30_high_cycle_visuals.py`
- `scripts/analyze_v10_2_30_developed_fatigue_growth.py`
- `scripts/analyze_v10_2_30_energy_gated_qualification.py`

Expected diagnostics include:

- `high_cycle_live_checkpoint.json`
- `high_cycle_live_state.npz`
- `high_cycle_live_history.jsonl`
- `kinetic_tip_cell_audit_v101.json`
- `stochastic_avalanche_geometry_events.json`
- `steps_0300K.csv`
- high-cycle timeline and validation plots;
- mechanical-response plots;
- signed MPZ profiles;
- MPZ activity proxy;
- crack-extension-versus-cycles and `da/dN` plots.

## 9. Results obtained so far

### 9.1 Weak-T, fraction 0.55

The qualification reached `1e12` cycles with no first passage and no committed crack event. This is a completed stochastic censoring trajectory.

### 9.2 Weak-T, fraction 0.75

The v5 run reached `1e12` cycles in approximately 532 seconds with no first passage and no crack extension.

For seed `2001726`, the final current-interval values were approximately:

- `H = 0.524952466`
- `Xi = 2.276469549`
- `B = H/Xi = 0.230599380`
- ensemble event probability `1-exp(-H) = 0.4084`

Therefore, this one no-event trajectory is not a deterministic fatigue-limit result. It is a valid high-threshold realization with approximately 59.2% ensemble survival at that accumulated action.

### 9.3 Weak-T, fraction 0.95

The latest run used:

- `DeltaK = 12.0677888 MPa*sqrt(m)`
- `Kmax = 13.4087 MPa*sqrt(m)`
- `R = 0.1`
- `T = 300 K`
- `frequency = 1000 Hz`
- `seed = 2001726`

The high-cycle engine advanced to approximately 37,668 cycles. A projective proposal was rejected by `dmd_event_guard`, correctly indicating that first passage might lie near the proposed interval. The solver then entered the exact transient fallback and failed with:

`RuntimeError: physical checkpoint length changed after stochastic event evolution began: old=2.000000000e-05 m, new=5.000000000e-06 m`

Call path:

- `persistent_site_high_cycle_engine_v10230_v5.py`
- v2 high-cycle state-machine fallback
- `persistent_site_forward_robust_v10230.py`
- exact constant-segment integration
- `persistent_site_cyclic_energy_gated_corrected_v10230.py`
- `persistent_site_cyclic_energy_gated_v10230.py`
- `stochastic_avalanche_tip.py::_synchronize_driver_checkpoint_length`

This is the current highest-priority blocker.

## 10. Interpretation of the current blocker

The stochastic-avalanche engine is constructed before the 2-D driver applies the final mesh-independent `da_phys=5e-6 m`. The engine initially records an inherited default checkpoint of `20e-6 m`.

The synchronization method is intended to adopt the final driver checkpoint before stochastic event evolution begins. In the 0.95 event-guard path, however, some high-cycle/exact-preview evolution changes `B` or `hazard_action_current` before synchronization occurs. When the exact fallback later calls `_synchronize_driver_checkpoint_length`, the method sees that stochastic evolution has started and correctly refuses to change the checkpoint from 20 to 5 micrometres.

Do not bypass the exception by weakening the guard. Fix initialization/order so the final physical checkpoint is synchronized before any preview, DMD training burst, private trial, exact-cycle fallback, or hazard evolution can occur.

The fix must preserve:

- the same 5 micrometre physical checkpoint;
- the same stochastic threshold and RNG stream;
- the same physical hazard action;
- the same event-length distribution;
- exact/private-trial state isolation;
- existing deterministic fixed-mode parity;
- post-event geometry and energy-gate semantics.

Add a regression that reproduces the event-guard-to-exact-fallback path with driver `da_phys` differing from the inherited constructor default.

## 11. Required tests

At minimum retain and run:

- `tests/test_v10_2_30_high_cycle_launcher.py`
- `tests/test_v10_2_30_event_growth_v5.py`
- `tests/test_v10_2_30_high_cycle_affine_dmd.py`
- `tests/test_v10_2_30_high_cycle_engine.py`
- `tests/test_v10_2_30_transactional_engine.py`
- `tests/test_v10_2_29_event_cycle_accounting.py`

Add targeted tests for:

1. physical checkpoint synchronization before any stochastic evolution;
2. private preview/DMD trials preserving the production RNG and threshold;
3. event guard followed by exact localization;
4. one committed event followed by geometry-specific high-cycle cache invalidation;
5. multiple event-to-event restarts;
6. 100 micrometre developed-growth analysis;
7. stochastic seed-ensemble aggregation.

Do not relax physical or DMD validation tolerances merely to make tests pass.

## 12. Acceptance criteria

### Immediate blocker acceptance

The weak-T 0.95 case must pass through the current event guard without checkpoint-length mismatch, localize first passage exactly, preserve threshold provenance, and either:

- commit an energy-admitted crack event and restart at the new geometry; or
- record a physically valid zero-length/nonpropagating first-passage attempt according to the existing law.

### Event-to-event release gate

A real fixed-`DeltaK` case must:

- produce multiple stochastic first passages;
- apply the event-energy gate;
- commit checked geometry transactions;
- restart the high-cycle engine after every event;
- preserve cumulative cycle accounting;
- reach approximately 100 micrometres or a declared censor;
- emit complete event-level and developed `da/dN` diagnostics.

### Four-parameterization campaign gate

For peak, DBTT, weak-T, and ceramic-like rows:

- identify censored and propagated `DeltaK` regimes;
- use common seed sets across `DeltaK`;
- obtain developed-growth intervals where possible;
- report ensemble event probability, initiation distribution, developed `da/dN`, path-length rate, tortuosity, and scatter;
- demonstrate event-length convergence under the existing trial-fraction refinement;
- include at least one positive energy-truncated event across the qualification matrix.

## 13. First task for Codex

1. Read this document and `AGENTS.md`.
2. Inspect the current branch and reproduce or audit the 0.95 failure from the saved log.
3. Trace every place where `eng.f.da`, `avalanche_base_checkpoint_m`, `B`, and `hazard_action_current` are initialized or changed.
4. Design the smallest physics-neutral correction that synchronizes the final driver checkpoint before any stochastic or private-trial evolution.
5. Add a regression for the exact failure path.
6. Run the focused tests.
7. Rerun the weak-T 0.95 qualification to the first event and verify threshold, event length, energy gate, geometry commit, and post-event restart.
8. Do not begin the four-class sweep until the first real event-to-event restart is demonstrated.
