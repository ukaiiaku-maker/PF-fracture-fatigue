from dataclasses import replace

import pytest

from arrhenius_fracture.closure_lifecycle_evidence import advance_transition, transition_occurred, conservation
from arrhenius_fracture.voiding_production_v5 import build_production_void_state, advance_disabled_v5_stage2, deterministic_trajectory
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
from arrhenius_fracture.sharp_wake_backend_v12 import V11_MODEL_ID,V12_MODEL_ID
from arrhenius_fracture.v12_production_driver import build_loaded_state,execute_event


@pytest.mark.parametrize("partitions",[1,2,4,8,16])
def test_birth_partition_executes_actual_site_and_preserves_accepted_input(partitions):
    before,_=build_production_void_state()
    identity=fingerprint(before)
    after,operations=advance_transition(before,"birth_hit_1",partitions)
    assert transition_occurred("birth_hit_1",before,after)
    assert fingerprint(before)==identity
    assert len(operations)==partitions
    assert any("BIRTH_HIT" in op["events"] for op in operations)
    assert conservation(after,before)["passed"]


def test_disabled_stage2_dispatch_is_v12_not_v11_and_preserves_full_state():
    base=build_loaded_state(V12_MODEL_ID)
    expected,_=execute_event(base,(5.25e-4,0.),transaction_identity="neutral-test")
    observed,_=advance_disabled_v5_stage2(base,(5.25e-4,0.),transaction_identity="neutral-test")
    assert fingerprint(expected)==fingerprint(observed)
    with pytest.raises(ValueError,match="Stage-II V12"):
        advance_disabled_v5_stage2(build_loaded_state(V11_MODEL_ID),(5.25e-4,0.),transaction_identity="bad")


def test_ligament_occurrence_uses_archived_root_event_not_fresh_cavity_clock():
    before,_=deterministic_trajectory(stop_before_ligament=True)
    after,_=advance_transition(before,"ligament",1)
    assert not after.competition.consumed_event_ids
    assert transition_occurred("ligament",before,after)
    assert not transition_occurred("ligament",before,before)


@pytest.mark.parametrize("rate,passed",[(0.,True),(2.4e-13,False)])
def test_frozen_offset_terminal_does_not_treat_positive_unqualified_rate_as_zero(monkeypatch,rate,passed):
    from arrhenius_fracture.closure_lifecycle_evidence import lifecycle_decision
    from arrhenius_fracture import closure_production_evidence
    connected,_=deterministic_trajectory()
    monkeypatch.setattr(closure_production_evidence,"candidate_measurements",lambda state:[
        {"rates_before_resolution_guard":{"effective_rate_s":rate}}])
    row={"dataset":"controlled","case_identity":"positive_offset","terminal_checkpoint":"source",
         "failure":None,"conservation":{"passed":True}}
    result=lifecycle_decision([row],{"source":connected})
    assert result["controlled_histories"][0]["passed"] is passed
