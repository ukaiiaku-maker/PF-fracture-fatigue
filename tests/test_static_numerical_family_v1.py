from copy import deepcopy
import pytest
from arrhenius_fracture.static_numerical_family_v1 import classify,REGISTRY,GROUPS,RECOVERY_ARCS


def peers():
    measurements={'mesh_quality':.1,'reaction':1.,'compliance':1.,'energy':1.,
        'free_residual_relative':0.,'reaction_balance':0.,'energy_identity':0.,
        'independent_intact_path_certificate':{'intact_cross_graph_path_exists':False,'insufficient_seed_segment_ids':[]},
        'fixed_tip_probe':{'tensor_Pa':[[1.,0.],[0.,1.]]},'cavity_enabled':True,
        'cavity_fields':{'normalized_traction':0.,'edge_owner_valid':True},
        'closed_cavity_boundary_cycle':True,'solid_cavity_polygon_overlap_element_ids':[],
        'support_cavity_polygon_overlap_element_ids':[]}
    row={'measurements':measurements,'recovery':[{'arc_fraction':a,'repeat_exact':True,'reversed_edge_order_exact':True,
        'recovery':{'tensor_Pa':[[1.,0.],[0.,1.]]}} for a in RECOVERY_ARCS]}
    return {str(n):deepcopy(row) for n in (128,256,512)},{'test:{mesh}':{str(n):str(n) for n in (128,256,512)}}


def test_registry_has_96_independent_inputs_and_33_families():
    assert len(REGISTRY)==96 and len(GROUPS)==33
    assert all(set(group)=={'128','256','512'} for group in GROUPS.values())


def test_coarse_quality_failure_retained_but_does_not_replace_two_fine_levels():
    rows,groups=peers();rows['128']['measurements']['mesh_quality']=.01
    result=classify(rows,groups)
    assert result['passed'] and result['families'][0]['coarse_point_quality_retained']==.01
    rows['256']['measurements']['mesh_quality']=.049
    assert not classify(rows,groups)['passed']


@pytest.mark.parametrize('change',('empty','missing','reorder','reversal','repeat'))
def test_missing_or_nondeterministic_recovery_cannot_pass(change):
    rows,groups=peers();arcs=rows['512']['recovery']
    if change=='empty':arcs.clear()
    elif change=='missing':arcs.pop()
    elif change=='reorder':arcs.reverse()
    elif change=='reversal':arcs[0]['reversed_edge_order_exact']=False
    else:arcs[0]['repeat_exact']=False
    assert not classify(rows,groups)['passed']


def test_stability_is_not_final_accuracy_and_failures_cannot_pass():
    rows,groups=peers()
    rows['128']['measurements']['fixed_tip_probe']['tensor_Pa']=[[2.,0.],[0.,2.]]
    rows['256']['measurements']['fixed_tip_probe']['tensor_Pa']=[[1.1,0.],[0.,1.1]]
    result=classify(rows,groups)
    assert result['families'][0]['gates']['fixed_tip_stable_limit']
    assert not result['families'][0]['gates']['fixed_tip_fine_accuracy']
    rows['512']['failure']={'type':'RuntimeError','message':'actual solve rejected'}
    assert not classify(rows,groups)['passed']
