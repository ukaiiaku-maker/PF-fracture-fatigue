from dataclasses import replace
import copy

import numpy as np
import pytest

from arrhenius_fracture.accepted_update_controller_r6 import _resolved_state
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.current_front_process_adapter_r8 import (
    CurrentFrontProcessAdapter, bind_signed_fem_drive,
)
from arrhenius_fracture.front_constitutive_strategy_r8 import (
    CURRENT_STATE_SCHEMA, CURRENT_STRATEGY, build_current_engine,
)
from arrhenius_fracture.global_physical_scheduler_v1 import advance_global_physical_interval
from arrhenius_fracture.reduced_persistent_reference_r2 import advance_signed
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
)
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.voiding_production_v5 import (
    _bundle_from_state, build_production_void_state, crack_tip_tensor,
    restore_sharp_front_engine,
)


def _current_state():
    state, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), enabled=False,
        front_constitutive_strategy=CURRENT_STRATEGY,
    )
    return state


def _loaded(state):
    displacement = np.asarray(state.displacement).copy()
    displacement[2*np.asarray(state.boundary.top_nodes, int)+1] = 15.0e-6
    displacement[2*np.asarray(state.boundary.bot_nodes, int)+1] = -15.0e-6
    displacement[2*int(state.boundary.left_bot)+1] = -15.0e-6
    return equilibrate_fixed_load_with_production_fem(
        replace(state, displacement=displacement))


def _scheduler_step(state, duration_s):
    tensor, _ = crack_tip_tensor(state, branch_id=ROOT_BRANCH_ID)
    return advance_global_physical_interval(
        state, state, duration_s,
        {"kind": "hold", "temperature_K": 900.0,
         "stress_tensor_Pa": tensor,
         "K_effective_by_front": {ROOT_BRANCH_ID: 51.1004675e6},
         "maximum_events": 4},
    )


def test_current_strategy_is_state_owned_and_disk_restartable(tmp_path):
    state = _current_state()
    owner = state.crack_network.active_tip_ids[0]
    payload = state.tip_process_state["by_branch"][owner]
    assert payload["schema"] == CURRENT_STATE_SCHEMA
    assert payload["constitutive_strategy"] == CURRENT_STRATEGY
    restored_engine = restore_sharp_front_engine(
        state.material, _bundle_from_state(state), payload)
    assert restored_engine._front_constitutive_strategy == CURRENT_STRATEGY

    path = tmp_path / "current-spatial.json"
    write_checkpoint(state, path, compression="gzip")
    restarted = restore_checkpoint(path)
    assert restarted.tip_process_state["by_branch"][owner]["schema"] == CURRENT_STATE_SCHEMA
    assert complete_accepted_state_fingerprint(restarted) == (
        complete_accepted_state_fingerprint(state))


def test_native_current_probe_preserves_continuous_motion_and_signed_fem_drive():
    state = _current_state()
    owner = state.crack_network.active_tip_ids[0]
    advanced = CurrentFrontProcessAdapter(
        state.material, _bundle_from_state(state),
    ).advance_front_process(
        state.tip_process_state["by_branch"][owner],
        {"K_effective_Pa_sqrt_m": 51.1004675e6,
         "stress_tensor_Pa": [[1.0e9, -2.0e8], [-2.0e8, 3.0e9]],
         "tangent_xy": [1.0, 0.0]},
        900.0, 0.25e-6,
    )
    assert advanced.cleavage_event_occurred is False
    assert advanced.audit["continuous_micro_advance_m"] == pytest.approx(1.25e-6)
    assert advanced.audit["artificial_threshold_used"] is False
    assert advanced.audit["signed_drive"]["tau_signed_Pa"][0] > 0.0
    assert advanced.audit["signed_drive"]["tau_signed_Pa"][1] < 0.0


def test_rate_observable_uses_accepted_ledger_increment_not_absent_result_key():
    parent = build_current_engine()
    bind_signed_fem_drive(
        parent, [[1.0e9, -2.0e8], [-2.0e8, 3.0e9]])
    engine = copy.deepcopy(parent)
    result = advance_signed(
        engine, K_Pa_sqrt_m=51.1004675e6,
        temperature_K=900.0, dt_s=0.25e-6)
    assert "gross_source_rate_s" not in result
    observed = _resolved_state(
        engine, result, parent=parent,
        consumed_s=result["kinetic_dt_consumed_s"])["scalars"]
    assert np.isfinite(observed["gross_rate_s"])
    assert observed["gross_rate_s"] > 0.0


def test_native_event_commits_outer_geometry_then_restarts_exactly(tmp_path):
    state = _loaded(_current_state())
    event_state, event_audit = _scheduler_step(state, 1.0e-6)
    selected = [row["selected_event"] for row in event_audit["events"]]
    assert selected == ["DIRECTIONAL_CLEAVAGE"]
    assert len(event_state.crack_network.branch(ROOT_BRANCH_ID).path) == 3
    event_row = event_audit["events"][0]
    assert event_row["front_process_load_policy"] == (
        "NATIVE_CURRENT_CLOCK_WITH_DIRECTIONAL_ADMISSION")
    process = event_row["front_process"][0]
    assert event_row["legacy_front_process"] == ()
    assert process["clock_authority"] == "NATIVE_CURRENT_PERSISTENT_REVERSIBLE"
    assert process["artificial_threshold_used"] is False
    frame = event_row["topology_transaction"]["current_material_frame"]
    assert frame["mpz_translation_owner"] == "NATIVE_CURRENT_EVENT"
    assert frame["outer_geometry_translation_applied"] is False

    path = tmp_path / "post-event-current-spatial.json"
    write_checkpoint(event_state, path, compression="gzip")
    restarted = restore_checkpoint(path)
    uninterrupted, audit_a = _scheduler_step(event_state, 0.25e-6)
    continued, audit_b = _scheduler_step(restarted, 0.25e-6)
    assert [row["selected_event"] for row in audit_a["events"]] == [None]
    assert [row["selected_event"] for row in audit_b["events"]] == [None]
    assert complete_accepted_state_fingerprint(uninterrupted) == (
        complete_accepted_state_fingerprint(continued))
