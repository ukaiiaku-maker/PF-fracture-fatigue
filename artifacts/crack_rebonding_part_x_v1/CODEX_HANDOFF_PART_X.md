# Part X handoff — signed-K crack-rebonding compression campaign

Note: the repository's root `CODEX_HANDOFF.md` is a pre-existing, unrelated
project-wide handoff document from an earlier mission (PF/sharp-front
Arrhenius-hazard fatigue, dated to the v10.2.21–v10.2.30 hazard-energy-gated
line). It is left untouched. This file is Part X's dedicated handoff record,
tracked at `artifacts/crack_rebonding_part_x_v1/CODEX_HANDOFF_PART_X.md` per
mission section 14, updated after each major milestone.

- **Branch:** `codex/v10.2.30-crack-rebonding-part-x`
- **Worktree:** `/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_worktrees/v10230-crack-rebonding-part-x`
  (persistent — not `/private/tmp`)
- **Clean/dirty:** clean at last commit
- **Active PIDs / jobs:** none

## Completed stages

- **Initial checks (mission section 1):** base commit
  `a72d46557f5eba45c8c2e0a428574e5b9b624c81` verified, worktree created fresh
  (branch did not previously exist), merge-base confirmed exact, zero
  related workers, both existing verifiers (`verify_static_shield_
  attribution.py`, `verify_developed_confirmation.py`) pass, full
  `crack_rebonding` test selection 196 passed / 0 failed (405s).
- **PX0 (committed `2b72738`):** provenance and completion-contract freeze.
  Real production engine constructed via `a_native_engine_v10230.
  build_a_native_engine()` (genuine `__init__`, not a fixture) to record its
  exact MRO, `n_phase=80`, `mpz_n_bins=80`, `wake_n_bins=160`, and the
  A_NATIVE material-row hash. All 8 named prior studies traced to branch
  SHAs; 7/8 are ancestors of base (already present/hashed in this worktree);
  `causal_pilot_v2` is a separate git line whose substantive work already
  merged into base via a common ancestor — only its tip's
  `regime_equivalence_analysis.json` is unique, pulled read-only without
  merging that branch. That file's finding
  (`REVERSIBLE_PERSISTENT_EQUIVALENT_FOR_SLOPE_SCREEN` at the OLD
  reversible/persistent presets, both seeds, |S_h diff| < 0.0002 decade vs a
  0.005 gate) is carried into `mission_scope.json` as a binding design
  constraint on PX2.3: COMPETING_PERSISTENT must use materially reduced
  rupture relative to COMPETING_REVERSIBLE, not reuse the old presets
  unmodified.

## Source-code reconnaissance done (informs PX1 design)

- `persistent_site_cyclic_v10229.py` (389 lines): `preview_cycle_waveform`
  is the real per-phase-bin cyclic engine path; `cycle_step_waveform` drives
  `_integrate_coupled`. Confirmed injection points for dynamic rebonding and
  static-shield ablation.
- `persistent_site_coupled_hazard_v10229.py` (614 lines): adaptive
  Simpson/bisection outer quadrature over **constant segments**
  (`_commit_constant_segment` / `integrate_state_coupled_waveform`) — the
  "one-cycle mean rate" approximation the static-shield localizer-parity
  audit (a72d465) already identified as structurally distinct from dynamic
  rebonding's exact phase-resolved bisection.
- `crack_rebonding_kinetics_v10230.py` (777 lines): pure math layer.
  `CrackRebondingControls.minimum_load_hold_s` field **already exists**
  (default 0.0) but `validate()` (~line 161) explicitly rejects any nonzero
  value ("hold integration is a documented future interface, not
  implemented") — this is PX1.1's precise target. `propagate()` /
  `build_phase_factors()` / `_partial_product()` implement an exact
  piecewise-constant-generator phase propagator but assume a **uniform**
  `dt_phase` across all bins — PX1.1 needs a heterogeneous-duration
  generalization (sinusoidal bins at `dt_phase = T_base/n_phase` plus one
  hold bin at `dt_hold = minimum_load_hold_s`, evaluated exactly via `expm`,
  not extra rounded phase bins). Also contains `REFERENCE_ACTION_PRESETS`,
  `solve_reference_action_barriers`, `freeze_reference_action_preset` — the
  exact PX2 analytical-regime-design machinery the mission references.
- `crack_rebonding_v10230.py` (1257 lines): engine-integration layer.
  `patch_Q` already implements PASSIVATION_GATED_REBOND's full P↔C↔B
  generator (depassivation/repassivation wired in at ~lines 174–179) — so
  PX1.3 is an **audit/test-coverage** task on existing code, not new
  physics. Contains `solve_coupled_event_time`, `phase_resolved_action`,
  `_periodic_orbit_certificate`, `install_crack_rebonding`,
  `serialize_rebonding_checkpoint` — all call sites that PX1.1's dwell
  segment and PX1.4's localizer-parity generalization must also touch.

## Terminal and pending counts

- Physical trajectories launched: 0 (no physical result directory exists
  yet, per mission section 2's gate — PX2 analytical work must land first).
- PX1–PX7: not started.

## Exact next action

Implement PX1.1 (minimum-load dwell): generalize `propagate`/
`build_phase_factors`/`_partial_product` in `crack_rebonding_kinetics_v10230.py`
to heterogeneous per-bin duration, construct the appended hold generator
(`Kmin`, contact-active only when `Kmin<0`), wire into
`preview_cycle_waveform`'s phase loop, `_commit_constant_segment`'s no-fire
finalize path, `cycle_step_waveform`'s no-fire finalize path, and
`solve_coupled_event_time`, then remove the `validate()` rejection and add
the required test battery (hold=0 bit-identical to a72d465; R>=0 zero
contact-gated formation during hold; R<0 analytically-expected constant-rate
formation action; monodromy vs. fine stepping; event localization inside
hold; block-subdivision invariance; cycle-count/physical-time distinctness;
emission/energy-gate parity; checkpoint-shape parity) before moving to
PX1.2–PX1.4.
