from dataclasses import replace
import pytest

from arrhenius_fracture.voiding_production_v5 import build_production_void_state, deterministic_trajectory
from arrhenius_fracture.voiding_v5 import (
    advance_site,create_subgrid_cavity,update_cavity_growth,promote_cavity,VoidPhase,
    fingerprint as void_fingerprint,
)
from arrhenius_fracture.checkpoint_v11 import write_checkpoint,restore_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint


def fail_at(stage,observed):
    def inject(name,state):
        observed.append((name,state))
        if name==stage: raise RuntimeError('injected:'+stage)
    return inject


@pytest.mark.parametrize('stage',[
    'first_hit_threshold_renewal','second_hit_embryo_transition','stabilization','healing'])
def test_actual_site_internal_failure_never_publishes_trial_or_consumes_input_rng(stage):
    live,_=build_production_void_state(stochastic=True,seed=12000)
    state=live.void_state
    rates={'birth_s':1.,'stabilization_s':0.,'healing_s':0.}
    if stage!='first_hit_threshold_renewal':
        site=state.sites[0]
        state,_=advance_site(state,site.site_id,site.birth.crossing_time(site.candidate_weight),rates=rates)
    if stage in ('stabilization','healing'):
        site=state.sites[0]
        state,_=advance_site(state,site.site_id,site.birth.crossing_time(site.candidate_weight),rates=rates)
        assert state.sites[0].phase==VoidPhase.EMBRYO
        rates[stage+'_s']=1.
        duration=getattr(state.sites[0],stage).crossing_time(1.)
    else:
        duration=state.sites[0].birth.crossing_time(state.sites[0].candidate_weight)
    before=void_fingerprint(state); observed=[]
    with pytest.raises(RuntimeError,match='injected:'+stage):
        advance_site(state,state.sites[0].site_id,duration,rates=rates,failure_injector=fail_at(stage,observed))
    assert observed[-1][0]==stage
    assert void_fingerprint(state)==before
    assert void_fingerprint(observed[-1][1])!=before


@pytest.fixture(scope='module')
def subgrid():
    trace=[];deterministic_trajectory(stop_before_ligament=True,state_trace=trace)
    return dict(trace)


@pytest.mark.parametrize('stage',[
    'initial_inventory_debit','state_owned_growth','inventory_return_under_shrinkage','promotion_criterion'])
def test_actual_owned_inventory_and_promotion_failure_preserves_full_accepted_state(subgrid,stage):
    if stage=='initial_inventory_debit':
        before=subgrid['stabilization'] if 'stabilization' in subgrid else subgrid['multi_hit_2']
        # Obtain the real pre-cavity stable site from the captured stable
        # state by using the production site integrator, not a phase label.
        from arrhenius_fracture.closure_lifecycle_evidence import advance_transition
        before,_=advance_transition(subgrid['multi_hit_2'],'stabilization',1)
        operation=lambda inject:create_subgrid_cavity(before.void_state,'site-1',2.5e-5,failure_injector=inject)
    else:
        before=subgrid['subgrid_growth'] if stage=='promotion_criterion' else subgrid['subgrid_void']
        cavity=before.void_state.cavities[0]
        if stage=='promotion_criterion':
            operation=lambda inject:promote_cavity(before.void_state,cavity.cavity_id,5e-5,failure_injector=inject)
        else:
            operation=lambda inject:update_cavity_growth(before.void_state,cavity.cavity_id,
                rates={'series_limited_growth_s':1.},dt_s=1.,radial_growth_scale_m=1e-8,
                chemical_potential_drive_J=-1e-20 if stage=='inventory_return_under_shrinkage' else 1e-20,
                failure_injector=inject)
    identity=fingerprint(before); observed=[]
    with pytest.raises(RuntimeError,match='injected:'+stage): operation(fail_at(stage,observed))
    assert observed[-1][0]==stage
    assert fingerprint(before)==identity


def test_checkpoint_write_failure_preserves_prior_manifest_payload_and_accepted_state(tmp_path):
    before,_=build_production_void_state()
    target=tmp_path/'restart.json'; manifest=write_checkpoint(before,target)
    payload=target.with_name(manifest['state_file'])
    manifest_bytes=target.read_bytes(); payload_bytes=payload.read_bytes()
    trial=replace(before,checkpoint_generation=before.checkpoint_generation+1)
    observed=[]; identity=fingerprint(trial)
    with pytest.raises(RuntimeError,match='injected:checkpoint_write'):
        write_checkpoint(trial,target,failure_injector=fail_at('checkpoint_write',observed))
    assert observed[-1][0]=='checkpoint_write'
    assert target.read_bytes()==manifest_bytes and payload.read_bytes()==payload_bytes
    assert fingerprint(restore_checkpoint(target))==fingerprint(before)
    assert fingerprint(trial)==identity
