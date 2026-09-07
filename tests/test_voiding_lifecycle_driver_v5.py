from arrhenius_fracture.voiding_production_v5 import build_production_void_state
from arrhenius_fracture.voiding_lifecycle_driver_v5 import advance_production_void_interval,NATURAL_WINDOW_S
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
from arrhenius_fracture.checkpoint_v11 import write_checkpoint,restore_checkpoint


def test_physical_window_exercises_owned_birth_growth_and_real_promotion_without_mutating_input():
    state,_=build_production_void_state(stochastic=True,seed=12000)
    original=fingerprint(state)
    after,operations,result=advance_production_void_interval(state,NATURAL_WINDOW_S)
    assert fingerprint(state)==original
    assert result['failure'] is None and result['elapsed_duration_s']==NATURAL_WINDOW_S
    assert any(op['api']=='promotion_remesh' for op in operations)
    assert after.void_state.cavities and after.void_state.sites[0].hits==2
    assert after.mesh.ne!=state.mesh.ne
    assert not any(op['api']=='downstream_front_transaction' for op in operations)


def test_real_window_midpoint_restart_reproduces_complete_state_and_operations(tmp_path):
    before,_=build_production_void_state(stochastic=True,seed=12000)
    middle,_,result=advance_production_void_interval(before,NATURAL_WINDOW_S/2)
    assert result['failure'] is None
    write_checkpoint(middle,tmp_path/'middle.json')
    restored=restore_checkpoint(tmp_path/'middle.json')
    after,operations,result=advance_production_void_interval(middle,NATURAL_WINDOW_S/2)
    replay,replay_operations,replay_result=advance_production_void_interval(restored,NATURAL_WINDOW_S/2)
    assert fingerprint(after)==fingerprint(replay)
    assert operations==replay_operations and result==replay_result
