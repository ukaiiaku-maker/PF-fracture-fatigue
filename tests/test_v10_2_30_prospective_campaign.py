import math
import pytest
from scripts.analyze_v10_2_30_prospective_campaign import grid_gate,target_rate
from scripts.verify_v10_2_30_prospective_campaign import validate_history,validate_generation,validate_decision


def test_global_slope_cannot_hide_local_failure():
    Ks=[13.5,15,16.5,18,19.5,21]
    # Alternating rate errors preserve a plausible global exponent but produce
    # adjacent slopes far outside the frozen local tolerances.
    rows=[dict(Kmax=k,target='P40',candidate_id='test',R=.1,seed=1720,developed_qualified=True,
        physical_rate=4.473410023231299e-7*(k/18)**4*10**e,predicted_rate=target_rate('P40',k),
        mean_event_length_m=5e-6,mean_wait_cycles=5e-6/(4.473410023231299e-7*(k/18)**4*10**e))
        for k,e in zip(Ks,[0,.04,-.04,-.04,.04,0])]
    computed=grid_gate(rows,'P40')
    assert abs(computed['m_global']-4)<.35
    assert computed['max_local_slope_error']>.75
    with pytest.raises(ValueError,match='classification'):
        validate_decision({'classification':'PARIS_WINDOW_TRANSFER_VALIDATED'},computed)


def test_censored_rate_cannot_pass_window():
    row=dict(Kmax=18,physical_rate=None,developed_qualified=False)
    assert grid_gate([row],'P40')['classification']=='NUMERICALLY_UNRESOLVED'


def test_reused_path_fails():
    row=dict(attempt_path='same',status='COMPLETE',resume=False)
    with pytest.raises(ValueError,match='reused'):validate_history([row,row])


def test_resume_fails():
    with pytest.raises(ValueError,match='resume'):
        validate_history([dict(attempt_path='a',status='COMPLETE',resume=True)])


def test_preflight_cannot_be_physics():
    with pytest.raises(ValueError,match='preflight'):
        validate_history([dict(attempt_path='a',status='LAUNCH_PREFLIGHT_FAILURE_NO_PHYSICS',physical_initialization=True)])


@pytest.mark.parametrize('K',[13.5,18.,21.])
def test_generation2_old_loads_not_independent(K):
    with pytest.raises(ValueError,match='calibration'):
        validate_generation(dict(candidate_id='P40_TRANSFER_CALIBRATED_GEN2',Kmax=K,R=.1,seed=1720,independent_validation=True))


@pytest.mark.parametrize('R,seed',[(.5,1720),(-.95,1720),(.1,1001723)])
def test_generation2_transfer_conditions_are_independent(R,seed):
    validate_generation(dict(candidate_id='P40_TRANSFER_CALIBRATED_GEN2',Kmax=18,R=R,seed=seed,independent_validation=True))


def test_production_rng_uses_engine_seed_sequence_not_bare_seed():
    import numpy as np
    expected=[.07509316036236147,.4332087756327596,.004735693787404665]
    rng=np.random.default_rng(np.random.SeedSequence([1720,1]))
    assert list(rng.exponential(size=3))==pytest.approx(expected,rel=1e-14)
    assert np.random.default_rng(1720).exponential()!=pytest.approx(expected[0])


def test_renewal_diagnostic_uses_rate_not_normalized_action():
    from scripts.analyze_v10_2_30_prospective_campaign import recorded_renewal_fraction
    rows=[{'lambda_c':'100','B':'1','state_coupled_cleavage_hazard':'1'},
          {'lambda_c':'2000','B':'1','state_coupled_cleavage_hazard':'1'}]
    assert recorded_renewal_fraction(rows,1e-6)==pytest.approx(.002)


def test_self_consistent_manifest_tampering_still_fails(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts import analyze_v10_2_30_prospective_campaign as analysis
    import arrhenius_fracture.prospective_paris_transfer_engine_v10230 as builder
    path=tmp_path/'physical';path.mkdir();(path/'high_cycle_run_manifest.json').touch()
    actual=SimpleNamespace(as_dict=lambda:{'cleavage':'changed'})
    expected=SimpleNamespace(as_dict=lambda:{'cleavage':'frozen'})
    selected=dict(candidate_row_sha256='row',rebonding=False,PT_substitution=False,material_manifest_sha256=analysis.digest(actual.as_dict()))
    documents={'high_cycle_run_manifest.json':dict(git_head='head',prospective_candidate=selected),
        'physical__launch.json':dict(result_path_virgin_at_launch=True,resume=False,launch_time_unix=0,launch_head='head')}
    monkeypatch.setattr(analysis,'read',lambda p:documents[p.name])
    monkeypatch.setattr(analysis.MaterialManifest,'from_csv',lambda p:actual)
    monkeypatch.setattr(builder,'build_transfer_manifest',lambda cid:(expected,selected))
    with pytest.raises(ValueError,match='exact frozen'):
        analysis.validate_identity({'candidate_id':'candidate'},dict(result_path=str(path),launch_head='head'),{'candidates':[dict(candidate_id='candidate',complete_row_sha256='row')]})


def test_complete_event_action_uses_transaction_not_partial_block():
    from scripts.analyze_v10_2_30_prospective_campaign import completed_event_action
    event={'threshold_action':.4332087756327596,'physical_hazard_action_block':.3027119702988727,
           'event_transaction_audit':{'hazard_action_completed':.43320877563275967}}
    assert completed_event_action(event)==pytest.approx(.4332087756327596)
    event['event_transaction_audit']['hazard_action_completed']=.3027119702988727
    with pytest.raises(ValueError,match='localize'):completed_event_action(event)


def test_missing_completed_action_is_not_invented_from_threshold():
    from scripts.analyze_v10_2_30_prospective_campaign import completed_event_action
    with pytest.raises(KeyError):completed_event_action({'threshold_action':1,'event_transaction_audit':{}})


@pytest.mark.parametrize('mutation',['row_change','substitution','classification'])
def test_terminal_selection_cannot_rewrite_frozen_rows_or_gates(mutation):
    import copy
    from arrhenius_fracture.prospective_paris_candidate_engine_v10230 import digest
    from scripts.verify_v10_2_30_prospective_campaign import validate_final_selection
    row={'candidate_id':'frozen','cleavage':'unchanged'}
    decisions={'P25':{'candidate_id':'frozen','classification':'EFFECTIVE_GLOBAL_SLOPE_ONLY'}}
    selected={'retained_candidate_ids':['frozen'],'targets':copy.deepcopy(decisions)}
    eligible={'frozen':{'complete_row_sha256':digest(row)}}
    validate_final_selection(selected,decisions,[row],eligible)
    if mutation=='row_change':row['cleavage']='changed'
    elif mutation=='substitution':row['candidate_id']='ineligible'
    else:selected['targets']['P25']['classification']='PARIS_WINDOW_TRANSFER_VALIDATED'
    with pytest.raises(ValueError):validate_final_selection(selected,decisions,[row],eligible)


def test_terminal_checkpoint_state_is_separate_from_event_state(tmp_path):
    import json
    from scripts.analyze_v10_2_30_prospective_campaign import saved_terminal_checkpoint_state
    (tmp_path/'high_cycle_live_checkpoint.json').write_text(json.dumps({'diagnostics':{
        'tip_radius_m':1e-6,'mobile_count':0.,'retained_count':0.,'sigma_back_Pa':7e8,'active_K_shield_Pa_sqrt_m':0.}}))
    event={'terminal_radius_m':1.1e-6,'mobile_count':580.}
    event.update(saved_terminal_checkpoint_state(tmp_path))
    assert event['terminal_radius_m']==1.1e-6
    assert event['mobile_count']==580.
    assert event['post_geometry_checkpoint_radius_m']==1e-6
    assert event['post_geometry_checkpoint_mobile_count']==0.
