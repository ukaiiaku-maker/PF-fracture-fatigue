from arrhenius_fracture.voiding_production_v5 import build_production_void_state
from arrhenius_fracture.voiding_lifecycle_driver_v5 import advance_production_void_interval,NATURAL_WINDOW_S
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
from arrhenius_fracture.checkpoint_v11 import write_checkpoint,restore_checkpoint
import pytest


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


@pytest.mark.parametrize('seed',[12004,12023,12029])
def test_repeated_physical_growth_preserves_frozen_inventory_tolerance(seed):
    from arrhenius_fracture.closure_lifecycle_evidence import conservation
    before,_=build_production_void_state(stochastic=True,seed=seed)
    state=before
    for _ in range(32):
        state,_,result=advance_production_void_interval(state,NATURAL_WINDOW_S/32)
        assert result['failure'] is None
        assert conservation(state,before)['passed']


@pytest.mark.parametrize('partitions',[1,2,4,8,16])
def test_complete_natural_window_retains_exact_accepted_time(partitions):
    from arrhenius_fracture.canonical_kinetic_time_v1 import exact
    state,_=build_production_void_state(stochastic=True,seed=12000)
    for _ in range(2*partitions):
        state,_,result=advance_production_void_interval(state,NATURAL_WINDOW_S/(2*partitions))
        assert result['failure'] is None
    assert state.junction_process_state['canonical_accepted_time_v1'].seconds_exact()==exact(NATURAL_WINDOW_S)
    # This asserts time only. The independently executed full-state partition
    # matrix still detects resolved-remesh path dependence at fine partitions.
