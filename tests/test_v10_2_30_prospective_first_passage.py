import copy
import numpy as np
import pytest
from scripts.audit_v10_2_30_prospective_first_passage import validate_first_passage_history


def sample():
    rng=np.random.default_rng(np.random.SeedSequence([1720,1]));values=list(rng.exponential(size=4))
    state=dict(hazard_threshold_history=values[:3],hazard_event_index=3,hazard_threshold_action=values[3],rng_state=rng.bit_generator.state)
    geometry=[dict(hazard_event_index=i,threshold_action=values[i]) for i in (0,2)]
    common=dict(event_localized=True,coupled_hazard_event_restart=True)
    kinetic=[dict(common,fired=True),dict(common,fired=False,hazard_energy_gate_attempt_consumed=True,hazard_energy_gate_zero_length_attempt=True,energy_gated_event_advance_m=0),dict(common,fired=True)]
    energy=[dict(inserted=True,committed_event_length_m=1),dict(inserted=False,committed_event_length_m=0,energy_admissible_event_length_m=0,arrest_reason='no_mesh_resolved_admissible_increment'),dict(inserted=True,committed_event_length_m=1)]
    return geometry,kinetic,energy,state


def test_zero_length_first_passage_consumes_its_own_threshold():
    assert validate_first_passage_history(*sample(),1720,1)==1


def test_threshold_gap_without_zero_length_audit_fails():
    g,k,e,s=sample();k[1]['hazard_energy_gate_attempt_consumed']=False
    with pytest.raises(ValueError):validate_first_passage_history(g,k,e,s,1720,1)


def test_geometry_cannot_reuse_nonadvancing_passage_threshold():
    g,k,e,s=sample();g[1]['threshold_action']=s['hazard_threshold_history'][1]
    with pytest.raises(ValueError,match='threshold index'):validate_first_passage_history(g,k,e,s,1720,1)
