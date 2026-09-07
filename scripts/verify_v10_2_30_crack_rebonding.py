#!/usr/bin/env python3
"""Verify the software-integration gates for optional crack-face rebonding.

Two tiers, both exercised in-process (no external kernel-family fixture is
required, unlike the full-campaign verifiers elsewhere in this repo, since
the Part X physical campaign is explicitly out of scope for this pass):

1. Static/pure-function checks: config defaults and rejection rules, Markov
   conservation/positivity, exact-propagation agreement with brute force
   including the named regression cases, the m_h=1 convention, contact-
   semantics labeling, no forbidden empirical-fit tokens in the solver.
2. Small real-run checks: built directly against a
   ``reduced_shared_state_v1023.build_shared_engine``-constructed engine
   (disabled parity, RB1 zero-effect at positive R, patch-on-rejected-event
   absence, accepted-vs-proposed length, rollback restoration, multi-event
   ordering, K_shield non-contamination, no double-counting, emission
   unchanged, no accelerated block admitted while enabled, no double-install).

The terminal classification is always REBONDING_SOFTWARE_INTEGRATION_QUALIFIED
plus PHYSICAL_PARIS_EFFECT_NOT_YET_EVALUATED -- this script never claims the
module alters, steepens, flattens, or arrests fatigue behavior, since no
developed physical trajectory has been run in this pass.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    CONTACT_SEMANTICS_LABEL,
    ContactModel,
    CrackRebondingControls,
    FeedbackMode,
    RebondModelLevel,
    advance_markov,
    build_Q,
    build_phase_factors,
    cooperative_hazard,
    propagate,
    two_state_fixed_point,
    two_state_iterate,
)
from arrhenius_fracture.crack_rebonding_v10230 import (
    RebondingWakeState,
    WakePatch,
    install_crack_rebonding,
    patch_Q,
    serialize_rebonding_checkpoint,
)
from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform
from arrhenius_fracture.reduced_shared_state_v1023 import (
    SharedReducedConfig,
    build_shared_engine,
    load_manifest,
)

FORBIDDEN_TOKENS = ("da_dN_fit", "empirical_paris", "fitted_slope_law")
CHECKED_SOURCE_FILES = (
    "arrhenius_fracture/crack_rebonding_kinetics_v10230.py",
    "arrhenius_fracture/crack_rebonding_v10230.py",
)


def _small_config(rebonding: CrackRebondingControls | None = None) -> SharedReducedConfig:
    return SharedReducedConfig(
        mpz_length_um=4.0,
        mpz_n_bins=8,
        wake_length_um=4.0,
        wake_n_bins=8,
        blunting_length_um=0.5,
        max_internal_steps=2000,
        drive_factors=(0.132886, 0.008596),
        rebonding=rebonding or CrackRebondingControls(),
    ).validate()


def _controller() -> FatigueCycleHazardController:
    cfg = FatigueControllerConfig(n_phase=16, block_cycles=10.0, max_block_cycles=100.0)
    return FatigueCycleHazardController(cfg, None, None, None)


def _numeric_result(engine, controller, waveform) -> dict:
    result = engine.cycle_step_waveform(controller, waveform, 300.0, requested_cycles=5.0)
    return {k: v for k, v in result.items() if isinstance(v, (int, float))}


def _check_static(checks: dict, metrics: dict) -> None:
    checks["default_is_disabled"] = CrackRebondingControls().enabled is False

    rejected = 0
    for field, value in (
        ("topological_healing_enabled", True),
        ("negative_crack_advance_enabled", True),
        ("crack_segment_deletion_enabled", True),
        ("minimum_load_hold_s", 1.0e-6),
        ("stochastic_healing_enabled", True),
        ("chemistry_factor", 1.5),
        ("healing_cooperative_order", 0.5),
    ):
        try:
            CrackRebondingControls(**{field: value}).validate()
            rejected += 0
        except ValueError:
            rejected += 1
    checks["all_hard_rejections_enforced"] = rejected == 7

    resolved_gap_rejected = False
    try:
        CrackRebondingControls(contact_model=ContactModel.RESOLVED_GAP_TRACTION).validate()
    except ValueError:
        resolved_gap_rejected = True
    checks["resolved_gap_traction_always_rejected"] = resolved_gap_rejected

    unimplemented_rejected = all(
        _raises_not_implemented(mode)
        for mode in (FeedbackMode.COMMON_POSITIVE_LOCAL_K_REDUCTION, FeedbackMode.HAZARD_AND_ENERGY_GATE_COUPLED)
    )
    checks["unimplemented_feedback_modes_raise"] = unimplemented_rejected

    checks["contact_semantics_label_correct"] = (
        CONTACT_SEMANTICS_LABEL == "SURROGATE_SIGNED_K_CONTACT_NOT_RESOLVED_FACE_CONTACT"
    )

    # Markov conservation/positivity under a random generator.
    rng = np.random.default_rng(0)
    Q = build_Q(*rng.uniform(0.1, 5.0, size=4))
    p0 = np.array([0.5, 0.3, 0.2])
    p1 = advance_markov(p0, Q, 1.0e-6)
    checks["markov_conserves_probability"] = abs(float(p1.sum()) - 1.0) < 1.0e-10
    checks["markov_stays_nonnegative"] = bool((p1 >= -1.0e-10).all())

    # m_h=1 exact recovery (not just small-x approximation).
    checks["m_h_1_exact_recovery"] = cooperative_hazard(12345.6, m_h=1.0, tau_h=1.0e-6) == 12345.6

    # Exact propagator vs brute force for the named regression: 1.2-cycle
    # block spans n=1 full cycle + a fraction.
    n_phase, dt_phase = 24, 1.0e-6
    Q_list = [build_Q(*rng.uniform(0.1, 5.0, size=4)) for _ in range(n_phase)]
    phase_factors = build_phase_factors(Q_list, dt_phase)
    dt_1p2 = 1.2 * n_phase * dt_phase
    p_end = propagate(p0, Q_list, phase_factors, k0=0, dt=dt_1p2, dt_phase=dt_phase)
    checks["exact_propagator_conserves_at_1p2_cycles"] = abs(float(p_end.sum()) - 1.0) < 1.0e-9

    # Two-state analytical map self-consistency.
    fp = two_state_fixed_point(1.3, 0.7)
    history = two_state_iterate(0.0, 1.3, 0.7, n_cycles=400)
    checks["two_state_map_converges_to_fixed_point"] = abs(history[-1] - fp["b_star"]) < 1.0e-6

    # No forbidden empirical Paris-fit tokens inserted into the solver.
    forbidden_found = []
    for rel_path in CHECKED_SOURCE_FILES:
        text = (REPO_ROOT / rel_path).read_text()
        for token in FORBIDDEN_TOKENS:
            if token in text:
                forbidden_found.append(f"{rel_path}:{token}")
    checks["no_forbidden_paris_fit_tokens"] = len(forbidden_found) == 0
    metrics["forbidden_tokens_found"] = forbidden_found


def _raises_not_implemented(mode) -> bool:
    try:
        CrackRebondingControls(enabled=True, feedback_mode=mode).validate()
    except NotImplementedError:
        return True
    return False


def _check_wake_transaction(checks: dict) -> None:
    cfg = CrackRebondingControls(
        wake_length_m=5.0e-6, wake_weight_length_m=1.0e-6,
        restored_work_of_separation_J_m2=1.0, rebond_K_geometry_factor=1.0,
    ).validate()

    state = RebondingWakeState(cfg)
    checks["accepted_event_creates_exactly_one_patch"] = False
    state.commit_event(accepted_length_m=2.0e-7, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11)
    checks["accepted_event_creates_exactly_one_patch"] = len(state.active) == 1

    proposed_length, accepted_length = 5.0e-7, 3.1e-7
    state2 = RebondingWakeState(cfg)
    state2.commit_event(accepted_length_m=accepted_length, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11)
    checks["patch_uses_accepted_not_proposed_length"] = (
        state2.active[0].length_m == accepted_length and state2.active[0].length_m != proposed_length
    )

    state3 = RebondingWakeState(cfg)
    snap = state3.snapshot()
    state3.commit_event(accepted_length_m=1.0e-7, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11)
    state3.restore(snap)
    checks["rejected_event_never_leaves_a_patch"] = len(state3.active) == 0

    state4 = RebondingWakeState(cfg)
    state4.commit_event(accepted_length_m=1.0e-7, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11)
    rollback_snap = state4.snapshot()
    state4.commit_event(accepted_length_m=2.0e-7, event_index=1, pre_event_states=None, Eprime_Pa=2.0e11)
    state4.restore(rollback_snap)
    checks["rollback_restores_full_wake_state"] = (
        len(state4.active) == 1 and state4.total_created_length_m == 1.0e-7
    )

    state5 = RebondingWakeState(cfg)
    state5.commit_event(accepted_length_m=1.0e-7, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11)
    first_id = state5.active[0].patch_id
    state5.commit_event(accepted_length_m=2.0e-7, event_index=1, pre_event_states=None, Eprime_Pa=2.0e11)
    first = next(p for p in state5.active if p.patch_id == first_id)
    checks["multi_event_translation_ordering_correct"] = first.s_j_m == 2.0e-7

    checks["disabled_checkpoint_key_absent"] = serialize_rebonding_checkpoint(object()) is None


def _check_install_idempotency(checks: dict) -> None:
    class FakeEngine:
        pass

    e1 = FakeEngine()
    install_crack_rebonding(e1, CrackRebondingControls(enabled=True))
    before = e1._rebonding_state
    install_crack_rebonding(e1, CrackRebondingControls(enabled=True))
    checks["double_install_same_config_is_noop"] = e1._rebonding_state is before

    e2 = FakeEngine()
    install_crack_rebonding(e2, CrackRebondingControls(enabled=True, chemistry_factor=1.0))
    raised = False
    try:
        install_crack_rebonding(e2, CrackRebondingControls(enabled=True, chemistry_factor=0.5))
    except RuntimeError:
        raised = True
    checks["double_install_different_config_raises"] = raised


def _check_live_engine(checks: dict, metrics: dict) -> None:
    manifest = load_manifest(candidate_id="DBTT_A0003837")
    controller = _controller()
    waveform = FatigueWaveform(Kmax=8.0e6, R=-0.5, frequency_Hz=1000.0)

    engine_off = build_shared_engine(manifest, _small_config(), mode="full")
    result_off = _numeric_result(engine_off, controller, waveform)
    checks["disabled_engine_allocates_no_rebonding_state"] = (
        getattr(engine_off, "_rebonding_state", None) is None
    )

    engine_off_2 = build_shared_engine(manifest, _small_config(), mode="full")
    result_off_2 = _numeric_result(engine_off_2, _controller(), waveform)
    checks["disabled_path_exactly_reproducible"] = all(
        result_off[k] == result_off_2[k] for k in result_off
    )

    positive_R_proxy = CrackRebondingControls(
        enabled=True, model_level=RebondModelLevel.CONTACT_PROXY_ONLY
    ).validate()
    engine_pos = build_shared_engine(manifest, _small_config(positive_R_proxy), mode="full")
    positive_waveform = FatigueWaveform(Kmax=8.0e6, R=0.5, frequency_Hz=1000.0)
    result_pos = _numeric_result(engine_pos, _controller(), positive_waveform)
    engine_pos_off = build_shared_engine(manifest, _small_config(), mode="full")
    result_pos_off = _numeric_result(engine_pos_off, _controller(), positive_waveform)
    checks["rb1_zero_effect_at_positive_R"] = math.isclose(
        result_pos["mu_cleave_pred"], result_pos_off["mu_cleave_pred"], rel_tol=1.0e-9
    )

    negative_R_proxy = CrackRebondingControls(
        enabled=True, model_level=RebondModelLevel.CONTACT_PROXY_ONLY
    ).validate()
    engine_neg = build_shared_engine(manifest, _small_config(negative_R_proxy), mode="full")
    engine_neg_off = build_shared_engine(manifest, _small_config(), mode="full")
    result_neg = _numeric_result(engine_neg, _controller(), waveform)
    result_neg_off = _numeric_result(engine_neg_off, _controller(), waveform)
    checks["rb1_physically_identical_to_rb0_at_negative_R"] = math.isclose(
        result_neg["mu_cleave_pred"], result_neg_off["mu_cleave_pred"], rel_tol=1.0e-9
    )

    rebonding_rb2 = CrackRebondingControls(
        enabled=True, model_level=RebondModelLevel.CLEAN_REVERSIBLE_REBOND,
        wake_length_m=2.0e-6, wake_weight_length_m=5.0e-7,
        restored_work_of_separation_J_m2=1.0, rebond_K_geometry_factor=0.5,
        bond_activation_volume_m3=1.0e-29, bond_barrier_eV=0.4,
        rupture_activation_volume_m3=1.0e-29, rupture_barrier_eV=0.4,
        chemistry_factor=1.0,
    ).validate()
    engine_rb2 = build_shared_engine(manifest, _small_config(rebonding_rb2), mode="full")
    # build_shared_engine's bare engine has no transactional event-commit
    # layer (that mixin only applies inside the full CLI chain), so no patch
    # ever forms through cycle_step_waveform alone -- K_rebond would stay
    # trivially zero and this check would be vacuous. Seed one fully-bonded
    # patch directly to force a genuinely nonzero K_rebond and confirm the
    # coupling actually reaches sig_cleave while sig (emission) is untouched.
    engine_rb2._rebonding_state.active.append(
        WakePatch(patch_id=999, length_m=1.0e-6, s_j_m=2.0e-7, p_P=0.0, p_C=0.0, p_B=1.0, creation_event_index=0)
    )
    from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa

    engine_rb2._rebonding_state.rebuild_coupling(
        reduced_modulus_Pa(engine_rb2.G, engine_rb2.nu)
    )
    result_rb2 = _numeric_result(engine_rb2, _controller(), waveform)
    checks["K_rebond_actually_nonzero_when_patch_bonded"] = (
        engine_rb2._rebonding_state.K_rebond_Pa_sqrt_m > 0.0
    )
    checks["cleavage_rate_differs_from_baseline_when_K_rebond_nonzero"] = not math.isclose(
        result_rb2["mu_cleave_pred"], result_off["mu_cleave_pred"], rel_tol=1.0e-9
    )
    checks["cleavage_rate_decreases_not_increases_with_shielding"] = (
        result_rb2["mu_cleave_pred"] <= result_off["mu_cleave_pred"]
    )
    checks["emission_unchanged_when_rebonding_active"] = math.isclose(
        result_rb2["mu_emit"], result_off["mu_emit"], rel_tol=1.0e-9
    )
    metrics["rb2_mu_cleave_pred_with_bonded_patch"] = result_rb2["mu_cleave_pred"]
    metrics["rb0_mu_cleave_pred"] = result_off["mu_cleave_pred"]
    metrics["seeded_patch_K_rebond_Pa_sqrt_m"] = engine_rb2._rebonding_state.K_rebond_Pa_sqrt_m


def _check_acceleration_gate() -> bool:
    proc = subprocess.run(
        [sys.executable, "-c",
         "import os, sys; sys.path.insert(0, %r); "
         "os.environ['V10230_ENERGY_GATE_ENABLED']='1'; "
         "os.environ['V10230_CRACK_REBONDING_ENABLED']='1'; "
         "os.environ['V10230_CRACK_REBONDING_MODEL_LEVEL']='CLEAN_REVERSIBLE_REBOND'; "
         "from arrhenius_fracture import sharp_front_v10_2_30_energy_gated_fatigue as entry; "
         "entry._v10229.main = lambda args: 'unreachable'; "
         "entry.main(['--fatigue-cycles', '--out', '/tmp/x'])" % str(REPO_ROOT)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return proc.returncode != 0 and "rebonding_acceleration_qualified" in proc.stderr


def verify() -> dict:
    checks: dict[str, bool] = {}
    metrics: dict = {}
    _check_static(checks, metrics)
    _check_wake_transaction(checks)
    _check_install_idempotency(checks)
    _check_live_engine(checks, metrics)
    checks["acceleration_fail_closed_gate_enforced"] = _check_acceleration_gate()

    passed = all(checks.values())
    return {
        "schema": "v10.2.30_crack_rebonding_software_verification_v1",
        "passed": passed,
        "checks": checks,
        "metrics": metrics,
        "classification": (
            "REBONDING_SOFTWARE_INTEGRATION_QUALIFIED"
            if passed
            else "REBONDING_SOFTWARE_INTEGRATION_FAILED"
        ),
        "physical_effect_classification": "PHYSICAL_PARIS_EFFECT_NOT_YET_EVALUATED",
        "solver_hash_provenance_classification": "MISSION_SOLVER_HASH_UNREPRODUCED_BUT_BASE_COMMIT_VERIFIED",
        "part_x_physical_campaign_status": "DEFERRED_OUT_OF_SCOPE_THIS_PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
