"""Bounded crack-rebonding causal pilot: config-freezing, calibration, and
causal-gate-logic tests.

Deliberately does NOT re-run the full six-trajectory, 5-event-each campaign
here (that is scripts/run_v10_2_30_crack_rebonding_causal_pilot.py's job,
and its output lives under runs/crack_rebonding_causal_pilot_v1/) -- this
file tests the pure config/calibration/analysis logic fast, plus one small
(1-2 event) real-engine smoke check of the physically-identical invariant
that the campaign itself depends on.
"""
from __future__ import annotations

import pytest

import _crack_rebonding_engine_fixture as fx

from arrhenius_fracture import crack_rebonding_causal_pilot_v10230 as pilot
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import RebondModelLevel
from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa
from arrhenius_fracture.fatigue_v1 import (
    FatigueControllerConfig,
    FatigueCycleHazardController,
    FatigueWaveform,
)
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)


@pytest.fixture(scope="module")
def material_constants():
    Engine.reset_audit()
    bare = fx.build_real_engine(None)
    r_eff = max(bare.r_eff(), 1.0e-9)
    Eprime = reduced_modulus_Pa(bare.G, bare.nu)
    return {"Eprime_Pa": Eprime, "reference_contact_radius_m": r_eff, "G_Pa": bare.G, "nu": bare.nu}


def test_resolve_restored_work_of_separation_hits_the_target_K_rebond_max(material_constants):
    Eprime = material_constants["Eprime_Pa"]
    G_max = pilot.resolve_restored_work_of_separation(Eprime)
    K_rebond_max = pilot.ETA_K * (Eprime * G_max) ** 0.5
    assert K_rebond_max == pytest.approx(pilot.K_REBOND_MAX_TARGET_Pa_sqrt_m, rel=1.0e-9)


def test_calibration_search_satisfies_the_r_zero_tolerance(material_constants):
    calib = pilot.solve_calibrated_rb2_configs(
        Eprime_Pa=material_constants["Eprime_Pa"],
        reference_contact_radius_m=material_constants["reference_contact_radius_m"],
    )
    for name, ratio in calib["r_zero_action_ratios"].items():
        assert ratio <= pilot.R_ZERO_ACTION_REL_TOL, name
    for name, cfg in calib["configs"].items():
        K_rebond_max = cfg.rebond_K_geometry_factor * (
            material_constants["Eprime_Pa"] * cfg.restored_work_of_separation_J_m2
        ) ** 0.5
        assert K_rebond_max == pytest.approx(pilot.K_REBOND_MAX_TARGET_Pa_sqrt_m, rel=1.0e-6), name


def test_rb1_is_the_zero_generator_model_level_with_otherwise_identical_fields(material_constants):
    calib = pilot.solve_calibrated_rb2_configs(
        Eprime_Pa=material_constants["Eprime_Pa"],
        reference_contact_radius_m=material_constants["reference_contact_radius_m"],
    )
    rb2_reversible = calib["configs"]["reversible"]
    rb1 = pilot.rb1_config(rb2_reversible)
    assert rb1.model_level is RebondModelLevel.CONTACT_PROXY_ONLY
    import dataclasses

    assert dataclasses.replace(rb1, model_level=rb2_reversible.model_level) == rb2_reversible


def test_analytical_single_patch_predictions_are_valid_probabilities(material_constants):
    calib = pilot.solve_calibrated_rb2_configs(
        Eprime_Pa=material_constants["Eprime_Pa"],
        reference_contact_radius_m=material_constants["reference_contact_radius_m"],
    )
    rows = pilot.analytical_single_patch_predictions(
        calib["configs"]["persistent"],
        reference_contact_radius_m=material_constants["reference_contact_radius_m"],
    )
    assert len(rows) == len(pilot.ANALYTICAL_KMAX_GRID_Pa_sqrt_m)
    for row in rows:
        assert 0.0 <= row["b_star"] <= 1.0
        assert 0.0 <= row["P_survive"] <= 1.0
        assert row["A_on"] >= 0.0
        assert row["A_off"] >= 0.0


def test_freeze_pilot_configuration_is_deterministic(material_constants):
    frozen_a = pilot.freeze_pilot_configuration(
        Eprime_Pa=material_constants["Eprime_Pa"],
        reference_contact_radius_m=material_constants["reference_contact_radius_m"],
        engine_G_Pa=material_constants["G_Pa"],
        engine_nu=material_constants["nu"],
    )
    frozen_b = pilot.freeze_pilot_configuration(
        Eprime_Pa=material_constants["Eprime_Pa"],
        reference_contact_radius_m=material_constants["reference_contact_radius_m"],
        engine_G_Pa=material_constants["G_Pa"],
        engine_nu=material_constants["nu"],
    )
    assert frozen_a["frozen_configuration_sha256"] == frozen_b["frozen_configuration_sha256"]
    assert set(frozen_a["config_hashes"]) == {"RB0", "RB1", "RB2_reversible", "RB2_persistent"}
    assert frozen_a["config_hashes"]["RB0"] is None
    assert frozen_a["config_hashes"]["RB1"] is not None


def _fake_event(index, waiting_time_s, accepted_length_m, max_pB=0.0, max_K_rebond=0.0, bulk_records=None):
    return {
        "event_index": index,
        "waiting_time_s_this_event": waiting_time_s,
        "accepted_length_m": accepted_length_m,
        "max_pB_post_commit": max_pB,
        "max_K_rebond_post_commit_Pa_sqrt_m": max_K_rebond,
        "bulk_action_records": bulk_records or [],
    }


def _fake_trajectory(name, R, events):
    return {
        "name": name,
        "R": R,
        "n_accepted_events": len(events),
        "events": events,
        "censored": False,
        "censor_reason": None,
        "uncensored": True,
    }


def test_causal_analysis_gate1_passes_on_truly_identical_trajectories():
    events_a = [_fake_event(i, 1.0e-9, 5.0e-6) for i in range(3)]
    events_b = [_fake_event(i, 1.0e-9, 5.0e-6) for i in range(3)]
    p0 = _fake_trajectory("P0", -0.95, events_a)
    p1 = _fake_trajectory("P1", -0.95, events_b)
    gate = pilot._physically_identical(p0, p1)
    assert gate["identical"]
    assert gate["mismatches"] == []


def test_causal_analysis_gate1_fails_on_a_real_length_difference():
    events_a = [_fake_event(0, 1.0e-9, 5.0e-6)]
    events_b = [_fake_event(0, 1.0e-9, 6.0e-6)]
    p0 = _fake_trajectory("P0", -0.95, events_a)
    p1 = _fake_trajectory("P1", -0.95, events_b)
    gate = pilot._physically_identical(p0, p1)
    assert not gate["identical"]
    assert gate["mismatches"][0]["key"] == "accepted_length_m"


def test_causal_analysis_gate1_tolerates_small_waiting_time_noise_but_not_a_causal_scale_delta():
    events_a = [_fake_event(0, 1.0e-6, 5.0e-6)]
    events_noise = [_fake_event(0, 1.0e-6 * 1.005, 5.0e-6)]  # 0.5%: root-finder-scale
    events_real = [_fake_event(0, 2.0e-6, 5.0e-6)]  # 100%: causal-effect-scale
    p0 = _fake_trajectory("P0", -0.95, events_a)
    p_noise = _fake_trajectory("P1", -0.95, events_noise)
    p_real = _fake_trajectory("P2", -0.95, events_real)
    assert pilot._physically_identical(p0, p_noise)["identical"]
    assert not pilot._physically_identical(p0, p_real)["identical"]


def test_full_causal_analysis_flags_nonzero_p5_bonding_as_gate_2_failure():
    results = {
        name: _fake_trajectory(name, -0.95 if name != "P4" and name != "P5" else 0.1, [_fake_event(0, 1.0e-9, 5.0e-6)])
        for name in pilot.TRAJECTORY_NAMES
    }
    # P2/P3 dynamic nonzero (gate 3), P5 has leaked nonzero bonding (gate 2 should fail).
    results["P2"]["events"][0]["max_pB_post_commit"] = 1.0e-5
    results["P2"]["events"][0]["max_K_rebond_post_commit_Pa_sqrt_m"] = 1.0e-3
    results["P3"]["events"][0]["max_pB_post_commit"] = 1.0e-5
    results["P3"]["events"][0]["max_K_rebond_post_commit_Pa_sqrt_m"] = 1.0e-3
    results["P5"]["events"][0]["max_pB_post_commit"] = 1.0e-8
    results["P5"]["events"][0]["max_K_rebond_post_commit_Pa_sqrt_m"] = 1.0e-4

    analysis = pilot.causal_analysis(results)
    assert analysis["gates"]["gate_3_dynamic_nonzero_bonding"]["pass"]
    assert not analysis["gates"]["gate_2_p4_p5_identical_and_p5_zero_bonding"]["pass"]
    assert not analysis["overall_pass"]


def test_full_causal_analysis_flags_direct_energy_gate_coupling_when_lengths_diverge():
    events_p1 = [_fake_event(0, 1.0e-9, 5.0e-6)]
    events_p2 = [_fake_event(0, 1.0e-9, 5.5e-6)]  # length differs beyond tolerance
    results = {name: _fake_trajectory(name, -0.95, [_fake_event(0, 1.0e-9, 5.0e-6)]) for name in pilot.TRAJECTORY_NAMES}
    results["P1"] = _fake_trajectory("P1", -0.95, events_p1)
    results["P2"] = _fake_trajectory("P2", -0.95, events_p2)
    analysis = pilot.causal_analysis(results)
    assert analysis["gates"]["gate_5_K_rebond_not_in_energy_gate_directly"]["direct_coupling_suspected"]
    assert not analysis["overall_pass"]


def test_real_engine_p0_vs_p1_physically_identical_smoke(material_constants):
    """Fast (1-2 event) real-engine confirmation of the invariant the full
    campaign's gate 1 depends on: REBOND_OFF and CONTACT_PROXY_ONLY, with
    the SAME frozen seed and engine-id reset, must be physically identical
    (patch_Q is the exact zero generator for CONTACT_PROXY_ONLY)."""
    Engine.configure_hazard(mode="exponential", seed=pilot.SEED)
    calib = pilot.solve_calibrated_rb2_configs(
        Eprime_Pa=material_constants["Eprime_Pa"],
        reference_contact_radius_m=material_constants["reference_contact_radius_m"],
    )
    rb1 = pilot.rb1_config(calib["configs"]["reversible"])

    def make_controller(n_phase):
        return FatigueCycleHazardController(
            FatigueControllerConfig(n_phase=n_phase, block_cycles=1000.0, max_block_cycles=1.0e6),
            None,
            None,
            None,
        )

    kw = dict(
        build_engine=fx.build_real_engine,
        make_controller=make_controller,
        waveform_cls=FatigueWaveform,
        reset_engine_registry=Engine.reset_audit,
        max_accepted_events=2,
    )
    p0 = pilot.run_trajectory(name="P0", rebonding_cfg=None, R=pilot.R_REF, **kw)
    p1 = pilot.run_trajectory(name="P1", rebonding_cfg=rb1, R=pilot.R_REF, **kw)
    gate = pilot._physically_identical(p0, p1)
    assert gate["identical"], gate
