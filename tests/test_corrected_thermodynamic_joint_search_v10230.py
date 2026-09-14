import json
import numpy as np
import pytest
from scripts.thermodynamic_joint_barrier_v10230 import Basis,ThermodynamicSurface
from scripts.corrected_thermodynamic_joint_search_v10230 import *

def surf(S,cp=0.,floor=.2,amp=.8):
    return ThermodynamicSurface('emission',1e13,floor,(amp,),(Basis(1.,3e9,2.),),0.,(S,),0.,(cp,))

def test_negative_entropy_means_positive_barrier_temperature_derivative():
    s=surf(-40);stress=np.array([0.,1e9]);fields=sign_explicit_fields(s,stress,300)
    assert fields['activation_entropy_over_kB'][0]==pytest.approx(-40)
    assert fields['barrier_temperature_derivative_over_kB'][0]==pytest.approx(40)
    dt=1e-3;fd=(s.G_eV(0.,300+dt)-s.G_eV(0.,300-dt))/(2*dt)/8.617333262145e-5
    assert fd==pytest.approx(40,rel=1e-8)
    assert json.loads(json.dumps({k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in fields.items()}))['barrier_temperature_derivative_over_kB'][0]==pytest.approx(40)

def test_positive_entropy_means_negative_derivative():assert barrier_temperature_derivative_over_kB(surf(40),0.,300)==pytest.approx(-40)

def test_reference_surface_invariant_to_entropy_and_cp():
    stress=np.linspace(0,10e9,50);assert exact_Tref_equal(surf(-40,-15),surf(50,15),stress)

def test_derivatives_and_maxwell():
    a=finite_difference_audit(surf(-35,7),np.linspace(1e6,20e9,80),750)
    assert a['entropy_max_abs_kB']<2e-6 and a['volume_max_abs_m3']<2e-31 and a['maxwell_max_abs_m3_per_K']<1e-34

def test_serialization_round_trip():
    s=surf(-40,3);r=deserialize_surface(serialize_surface(s));x=np.linspace(0,8e9,20)
    assert np.array_equal(s.G_eV(x,900),r.G_eV(x,900))

def test_no_prefactor_double_counting():
    fields=sign_explicit_fields(surf(-40),0.);assert fields['entropy_representation']==ENTROPY_REPRESENTATION

def test_negative_and_nonfinite_barriers_fail():
    ok,_=thermodynamic_rate_gate(surf(-40,floor=-2,amp=.1),np.linspace(0,30e9,100),[300,1200]);assert not ok

def test_isolated_constitutive_ceiling_contact_uses_occupancy_rule():
    s=ThermodynamicSurface('emission',1e13,.001,(1.,),(Basis(1.,20e9,40.),),0.,(0.,));ok,rows=thermodynamic_rate_gate(s,np.linspace(0,30e9,100),[300]);assert rows[0]['constitutive_ceiling_occupied'];assert rows[0]['fraction_of_path_at_ceiling']<.8

def test_complete_path_saturation_fails():
    ok,_=thermodynamic_rate_gate(surf(0,floor=1e-9,amp=1e-9),np.linspace(0,30e9,100),[300]);assert not ok

def test_paired_mapping_keeps_nonentropy_and_Tref():
    p={'candidate_id':'old','emission_entropy_active_kB':40.,'emission_entropy_infinity_kB':5.,'emission_stratum':'PREFERRED','guard_exponent':4.}
    q=paired_parameters(p,'new');assert q['guard_exponent']==4 and q['emission_entropy_active_kB']==-40 and q['all_nonentropy_coordinates_equal'] and q['Tref_surface_equal']

def test_f0_and_f1a_cannot_promote_system_class():
    for tier in ('F0_INTRINSIC_OPENING_ONLY','F1A_REDUCED_STATE_ONLY'):assert 'NO_SYSTEM_CLASS_PROMOTION' in classify_coupled([1,2],tier,1.,{})

def test_f1b_accessibility_and_missing_anchor_enforced():
    assert classify_coupled([2,1],'F1B_FULL_PRODUCTION_STATE',.79,{'all_class_anchors_valid':True})=='STATE_UNRESOLVED'
    assert classify_coupled([2,1],'F1B_FULL_PRODUCTION_STATE',1.,{'all_class_anchors_valid':False})=='STATE_UNRESOLVED'

def test_class_rules_use_coupled_causal_state():
    causal={'all_class_anchors_valid':True,'opening_precedes_relaxation':True,'emission_admissible':True}
    assert classify_coupled([10,8,6],'F1B_FULL_PRODUCTION_STATE',1.,causal).startswith('CERAMIC')

def test_dbtt_requires_material_state_contribution():
    causal={'all_class_anchors_valid':True,'positive_interval_width_K':900,'plastic_state_increase':True,'expected_rate_shift':True,'state_transition_contribution_MPa_sqrt_m':0.1}
    assert not classify_coupled([10,16],'F1B_FULL_PRODUCTION_STATE',1.,causal).startswith('DBTT')
    causal['state_transition_contribution_MPa_sqrt_m']=1.0
    assert classify_coupled([10,16],'F1B_FULL_PRODUCTION_STATE',1.,causal).startswith('DBTT')

def test_partition_order_and_pair_generation_deterministic():
    rows=[{'candidate_id':str(i),'emission_entropy_active_kB':20+i,'emission_entropy_infinity_kB':i,'emission_stratum':'PREFERRED','x':i} for i in range(8)]
    one={r['candidate_id']:paired_parameters(r,'N'+r['candidate_id']) for r in rows};two={r['candidate_id']:paired_parameters(r,'N'+r['candidate_id']) for r in reversed(rows)};assert one==two
