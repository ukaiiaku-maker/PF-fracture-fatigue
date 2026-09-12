import pytest
from arrhenius_fracture.closure_lifecycle_evidence import prepare_common_restart_reload
from arrhenius_fracture.voiding_production_v5 import deterministic_trajectory,ligament_transaction
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint


@pytest.mark.parametrize('version',[1,2])
def test_same_accepted_zero_reload_protocol_resumes_without_repeating_a_load(version):
    pre,_=deterministic_trajectory(stop_before_ligament=True)
    connected,_=ligament_transaction(pre);initial=fingerprint(connected)
    operations=[];trace=[]
    direct=prepare_common_restart_reload(connected,operations,trace,protocol_version=version)
    assert fingerprint(connected)==initial
    assert [name for name,_ in trace]==['zero_drive_connected']
    replay_operations=[];resumed=prepare_common_restart_reload(trace[0][1],replay_operations,protocol_version=version)
    assert fingerprint(resumed)==fingerprint(direct)
    assert operations[1:]==replay_operations
    assert direct.competition==connected.competition and direct.rng_state==connected.rng_state
    assert not direct.crack_network.active_tip_ids
    replay_operations=[]
    assert prepare_common_restart_reload(direct,replay_operations,protocol_version=version) is direct
    assert replay_operations==[]
    assert operations[-1]['opening_m']==(8e-7 if version==1 else 4e-7)
    with pytest.raises(ValueError,match='cannot mix'):
        prepare_common_restart_reload(direct,[],protocol_version=3-version)


def test_common_zero_interval_cannot_force_a_positive_raw_candidate(monkeypatch):
    import arrhenius_fracture.closure_lifecycle_evidence as lifecycle
    pre,_=deterministic_trajectory(stop_before_ligament=True)
    connected,_=ligament_transaction(pre)
    monkeypatch.setattr(lifecycle,'directional_clock_rates',lambda *args:[{'effective_rate_s':1.}])
    with pytest.raises(RuntimeError,match='DORMANT_LOAD_HAS_POSITIVE_SOURCE_DRIVE'):
        prepare_common_restart_reload(connected,[])


def test_failed_resume_preserves_last_accepted_stage(monkeypatch):
    from types import SimpleNamespace
    import arrhenius_fracture.closure_lifecycle_evidence as lifecycle
    def state(hits):
        return SimpleNamespace(void_state=SimpleNamespace(sites=[SimpleNamespace(
            phase=lifecycle.VoidPhase.AVAILABLE_SITE,hits=hits)]))
    initial,accepted=state(0),state(1)
    monkeypatch.setattr(lifecycle,'fingerprint',lambda value:value.void_state.sites[0].hits)
    def transition(value,*args,**kwargs):
        if value is accepted:raise RuntimeError('later gate rejected')
        return accepted,[]
    monkeypatch.setattr(lifecycle,'advance_transition',transition)
    with pytest.raises(RuntimeError,match='later gate rejected') as caught:
        lifecycle.resume_to_guard(initial,[])
    assert caught.value.accepted_lifecycle_state is accepted
    assert initial.void_state.sites[0].hits==0


def test_independent_failed_restart_attempts_retain_both_endpoints(monkeypatch):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/run_voiding_v5_closure_lifecycle.py'
    spec=importlib.util.spec_from_file_location('restart_harness_test',path)
    harness=importlib.util.module_from_spec(spec);spec.loader.exec_module(harness)
    calls=[];accepted=object()
    def fail(value,operations,**kwargs):
        calls.append(value);operations.append({'api':'actual_accepted_stage'})
        error=RuntimeError('guard rejected');error.accepted_lifecycle_state=accepted
        raise error
    monkeypatch.setattr(harness,'resume_to_guard',fail)
    direct,restored=object(),object()
    operations=[];replay_operations=[]
    a,ea=harness.resume_attempt(direct,operations,common_restart_protocol=True)
    b,eb=harness.resume_attempt(restored,replay_operations,common_restart_protocol=True)
    assert calls==[direct,restored] and a is b is accepted
    assert ea==eb and operations==replay_operations


def test_continued_terminal_requires_owned_path_lineage_and_event():
    from types import SimpleNamespace
    from arrhenius_fracture.closure_lifecycle_evidence import continued_front_terminal,VoidPhase
    child=SimpleNamespace(path=((0.,0.),(1.,0.)))
    cavity=SimpleNamespace(phase=VoidPhase.DOWNSTREAM_FRONT_ACTIVE,lineage=('CONTINUED_EVENT',))
    event={'event':'CONTINUED_ACCEPTED_EVENT','source_kind':'sharp_front','source_front_id':'void-front-1'}
    state=SimpleNamespace(void_state=SimpleNamespace(cavities=(cavity,),event_history=(event,)),
        crack_network=SimpleNamespace(active_tip_ids=('void-front-1',),branch=lambda _:child),
        v12_support_state=SimpleNamespace(active_tip_identities=('void-front-1',)))
    assert not continued_front_terminal(state)
    child.path=(*child.path,(2.,0.));assert continued_front_terminal(state)
    event['source_kind']='cavity_surface';assert not continued_front_terminal(state)
