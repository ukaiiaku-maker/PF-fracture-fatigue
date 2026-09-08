"""Source-level callback on exact archived geometry and FEM results, no solve."""
from dataclasses import replace
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest

from arrhenius_fracture.live_topology_runtime_v11 import LiveTopologyRuntime
from arrhenius_fracture.live_topology_kernel_v11 import topology_fingerprint, request_contour_definitions
from arrhenius_fracture.primary_race_production_v13 import evaluate_production_mark
from scripts.v13_frozen_support import initialized_engine
from scripts.run_pf_current_source_multifront_field_atlas_v12 import campaign_environment, sha256
from scripts.v13_value_fingerprint import physical_state_fingerprint as fp

SOURCE=Path('analysis_outputs/v13_clock_and_pair_mechanism')
AUDIT=Path('analysis_outputs/v13_primary_continuation_race')


def fingerprint(request):
    return topology_fingerprint(network=request.crack_network,mesh=request.mesh,damage=request.damage,
        mechanical_configuration_fingerprint=request.mechanical_configuration_fingerprint,
        specimen_geometry=request.specimen_geometry,boundary_condition_identity=request.boundary_condition_identity,
        elastic_constants=request.elastic_constants,cluster_frame=request.cluster_frame,
        mpz_station_coordinates_m=request.mpz_station_coordinates_m,wake_station_coordinates_m=request.wake_station_coordinates_m,
        contour_definitions=request_contour_definitions(request))


@pytest.mark.parametrize('case',['Peak_1000K','weakT_1000K'])
@pytest.mark.parametrize('event',['event00002','event00008','event00015','event00029'])
def test_actual_production_callback_matches_archived_frozen_race(case,event,monkeypatch):
    audit=json.loads((AUDIT/case/event/'race.json').read_text())
    context_path=SOURCE/'later_parents'/case/'events'/event/'parent/event_context.pkl'
    assert sha256(context_path)==audit['source_context_sha256']
    payload=pickle.loads(context_path.read_bytes())
    checkpoint=payload['accepted_single_checkpoint']
    cached=pickle.loads((SOURCE/'later_companions'/case/event/'exact_pair_result.pkl').read_bytes())
    comp=json.loads((SOURCE/'later_companions'/case/event/'companion_qualification.json').read_text())['companions'][0]
    launch=json.loads((SOURCE/'later_parents'/case/'launch.json').read_text())
    for key,value in campaign_environment(Path(launch['family_validation']['family_validation']['family'])).items():
        monkeypatch.setenv(key,value)
    accepted=None
    for manifest_path in (SOURCE/'later_parents'/case/'live_kernel_cache').glob('*/manifest.json'):
        m=json.loads(manifest_path.read_text())
        if m['state_sha256']==audit['mechanics']['accepted_provider_sha256']:
            assert sha256(manifest_path.parent/'provider_state.pkl')==m['state_sha256']
            accepted=pickle.loads((manifest_path.parent/'provider_state.pkl').read_bytes())
            break
    assert accepted is not None
    def evaluate(runtime,request):
        identity=fingerprint(request)
        if request.cluster_frame['mode']=='candidate_marginal_kinetic_drive':
            assert identity==audit['mechanics']['topology_fingerprint']
            energy=audit['mechanics']['trial_energy_J_per_m']
            return runtime,{'base_equilibrium':{'recoverable_potential_energy_J_per_m':energy}}
        assert identity==comp['pair_topology_fingerprint']
        assert fp(request.crack_network)==fp(cached['trial'].state.crack_network)
        state=cached['trial'].state
        return runtime,{'topology_fingerprint':identity,'base_equilibrium':{
            'recoverable_potential_energy_J_per_m':state.stored_energy_J_per_m,'displacement':state.displacement}}
    def accept(runtime,request,live):
        assert fingerprint(request)==live['topology_fingerprint']
        return replace(runtime,routing=replace(runtime.routing,topology_fingerprint=live['topology_fingerprint']))
    monkeypatch.setattr(LiveTopologyRuntime,'evaluate_trial',evaluate)
    monkeypatch.setattr(LiveTopologyRuntime,'accept_trial',accept)
    with initialized_engine(checkpoint.shared_process_state) as engine:
        result=evaluate_production_mark(checkpoint=checkpoint,
            selected=next(t for t in payload['canonical_result'].trials if t.selected),
            solved_pre_event=payload['solved_pre_event_state'],engine=engine,args=SimpleNamespace(**payload['args']),
            cfg=payload['configuration'],context=payload['context'],accepted_live=accepted)
    branch=audit['outcome']=='PAIR_ACCEPTED'
    assert (result.record['outcome']=='PAIR_ACCEPTED')==branch
    if branch:
        assert result.record['T_i_next_s']==audit['T_i_next_s']
        assert result.record['T_j_s']==audit['T_j_s']
        assert fp(result.state)==fp(cached['trial'].state)
        for name in ('competition','rng_state','tip_process_state','event_counters'):
            assert getattr(result.state,name) is getattr(checkpoint.state,name)
    else:
        assert result.state is checkpoint.state
