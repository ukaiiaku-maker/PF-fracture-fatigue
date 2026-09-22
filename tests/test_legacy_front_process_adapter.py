from __future__ import annotations

import pickle

from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.legacy_front_process_adapter import LegacyFrontProcessAdapter
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.voiding_production_v5 import (
    _canonical_front_process_owner, build_production_void_state,
    capture_sharp_front_engine, restore_sharp_front_engine,
    sharp_front_engine_fingerprint,
)


def _fixture():
    bundle = material_bundle("DBTT")
    state, _ = build_production_void_state(bundle=bundle, temperature_K=900.0)
    owner = _canonical_front_process_owner(state, ROOT_BRANCH_ID)
    payload = state.tip_process_state["by_branch"][owner]
    return state, bundle, payload, LegacyFrontProcessAdapter(state.material, bundle)


def test_emission_dominated_interval_has_no_topology_increment():
    state, bundle, payload, adapter = _fixture()
    advanced = adapter.advance_front_process(payload, 1.0e8, 900.0, 2.0e-7)
    assert not advanced.cleavage_event_occurred
    assert advanced.updated_emission_state["N_em"] > 1.0e-4
    assert advanced.updated_mpz_state["accumulated_slip"] > 0.0
    assert advanced.audit["pre_a_adv_m"] == advanced.audit["post_a_adv_m"]


def test_cleavage_dominated_interval_fires_one_legacy_renewal():
    state, bundle, payload, adapter = _fixture()
    engine = restore_sharp_front_engine(state.material, bundle, payload)
    engine.B = 0.999
    near = capture_sharp_front_engine(engine)
    advanced = adapter.advance_front_process(near, 1.0e8, 900.0, 2.0e-9)
    assert advanced.cleavage_event_occurred
    assert advanced.cleavage_event_time_s is not None
    assert advanced.audit["post_n_adv"] == advanced.audit["pre_n_adv"] + 1


def test_multiple_emission_updates_precede_cleavage_and_change_later_rate():
    _, _, payload, adapter = _fixture()
    current = payload
    emitted = []
    for _ in range(3):
        update = adapter.advance_front_process(current, 1.0e8, 900.0, 5.0e-8)
        assert not update.cleavage_event_occurred
        current = update.advanced_front_state
        emitted.append(update.updated_emission_state["N_em"])
    assert emitted[0] < emitted[1] < emitted[2]
    fresh = adapter.advance_front_process(payload, 2.0e7, 300.0, 1.0e-12)
    relaxed = adapter.advance_front_process(current, 2.0e7, 300.0, 1.0e-12)
    fresh_rate = fresh.audit["step_info"]["lambda_c"]
    relaxed_rate = relaxed.audit["step_info"]["lambda_c"]
    assert relaxed.updated_shielding_Pa_sqrt_m > fresh.updated_shielding_Pa_sqrt_m
    assert relaxed_rate < fresh_rate


def test_checkpoint_restore_and_partition_preserve_legacy_owner():
    state, bundle, payload, adapter = _fixture()
    restored = pickle.loads(pickle.dumps(payload, protocol=5))
    direct = adapter.advance_front_process(payload, 1.0e8, 900.0, 1.0e-9)
    restarted = adapter.advance_front_process(restored, 1.0e8, 900.0, 1.0e-9)
    def fingerprint(result):
        engine = restore_sharp_front_engine(state.material, bundle, result.advanced_front_state)
        return sharp_front_engine_fingerprint(engine)
    assert fingerprint(direct) == fingerprint(restarted)
    first = adapter.advance_front_process(payload, 1.0e8, 900.0, 4.0e-10)
    split = adapter.advance_front_process(first.advanced_front_state, 1.0e8, 900.0, 6.0e-10)
    # The canonical nonlinear source update is compared at its frozen numerical
    # tolerance; no analytical replacement is used.
    assert abs(split.updated_emission_state["N_em"]-direct.updated_emission_state["N_em"]) < 1.0e-12
    assert abs(split.audit["post_B"]-direct.audit["post_B"]) < 1.0e-15


def test_topology_owned_mode_evolves_emission_without_second_cleavage():
    _, _, payload, adapter = _fixture()
    update = adapter.advance_front_process(
        payload, 1.0e8, 900.0, 1.0e-9,
        topology_owned_cleavage_action_before=0.0,
        topology_owned_cleavage_action_after=0.001,
    )
    assert not update.cleavage_event_occurred
    assert update.audit["post_n_adv"] == update.audit["pre_n_adv"]
    assert update.updated_emission_state["N_em"] > 0.0
    assert update.audit["post_B"] == 0.001


def test_topology_owned_unified_engine_suppresses_late_renewal_at_high_action():
    _, _, payload, adapter = _fixture()
    update = adapter.advance_front_process(
        payload, 1.0e8, 900.0, 1.0e-6,
        topology_owned_cleavage_action_before=0.8,
        topology_owned_cleavage_action_after=1.2,
    )
    assert not update.cleavage_event_occurred
    assert update.audit["post_n_adv"] == update.audit["pre_n_adv"]
    assert update.audit["step_info"]["dB_step"] == 0.0
    assert update.updated_emission_state["N_em"] > 0.0
    assert update.audit["post_B"] == 1.2
