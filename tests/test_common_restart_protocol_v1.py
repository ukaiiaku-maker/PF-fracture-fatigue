import pytest
from arrhenius_fracture.closure_lifecycle_evidence import prepare_common_restart_reload
from arrhenius_fracture.voiding_production_v5 import deterministic_trajectory,ligament_transaction
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint


def test_same_accepted_zero_reload_protocol_resumes_without_repeating_a_load():
    pre,_=deterministic_trajectory(stop_before_ligament=True)
    connected,_=ligament_transaction(pre);initial=fingerprint(connected)
    operations=[];trace=[]
    direct=prepare_common_restart_reload(connected,operations,trace)
    assert fingerprint(connected)==initial
    assert [name for name,_ in trace]==['zero_drive_connected']
    replay_operations=[];resumed=prepare_common_restart_reload(trace[0][1],replay_operations)
    assert fingerprint(resumed)==fingerprint(direct)
    assert operations[1:]==replay_operations
    assert direct.competition==connected.competition and direct.rng_state==connected.rng_state
    assert not direct.crack_network.active_tip_ids
    replay_operations=[]
    assert prepare_common_restart_reload(direct,replay_operations) is direct
    assert replay_operations==[]


def test_common_zero_interval_cannot_force_a_positive_raw_candidate(monkeypatch):
    import arrhenius_fracture.closure_lifecycle_evidence as lifecycle
    pre,_=deterministic_trajectory(stop_before_ligament=True)
    connected,_=ligament_transaction(pre)
    monkeypatch.setattr(lifecycle,'directional_clock_rates',lambda *args:[{'effective_rate_s':1.}])
    with pytest.raises(RuntimeError,match='DORMANT_LOAD_HAS_POSITIVE_SOURCE_DRIVE'):
        prepare_common_restart_reload(connected,[])
