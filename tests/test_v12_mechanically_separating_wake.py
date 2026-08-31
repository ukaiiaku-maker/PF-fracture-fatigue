from dataclasses import dataclass, replace
from types import SimpleNamespace
import numpy as np
import pytest

from arrhenius_fracture.causal_sharp_wake_v11 import CRACK_REPRESENTATION as V11_ID, causal_segment_support
from arrhenius_fracture.crack_network_v11 import CrackBranchState, CrackNetworkState
from arrhenius_fracture.mechanically_separating_sharp_wake_v12 import (
    MODEL_ID, apply_mechanically_separating_graph, certification_arcs, classify_graph_vertices,
    graph_component_count, independent_intact_path_certificate, mechanically_separating_graph_support as _graph_support,
    junction_sector_certificates, selected_support_components, support_record, unique_graph_length,
)

def mechanically_separating_graph_support(*args,**kwargs):
    kwargs.setdefault("allow_offgrid_active_tips_for_screen",True)
    return _graph_support(*args,**kwargs)

def mesh(n=9,perturb=False):
    x=np.linspace(0,1,n); y=np.linspace(-.5,.5,n); X,Y=np.meshgrid(x,y); nodes=np.c_[X.ravel(),Y.ravel()]; elems=[]
    if perturb:
        rng=np.random.default_rng(120031); interior=(np.abs(nodes[:,0])>0)&(np.abs(nodes[:,0]-1)>0)&(np.abs(nodes[:,1]+.5)>0)&(np.abs(nodes[:,1]-.5)>0)
        nodes[interior]+=rng.uniform(-.018,.018,size=(np.count_nonzero(interior),2))
    for j in range(n-1):
        for i in range(n-1):
            a=j*n+i;b=a+1;c=a+n;d=c+1
            elems.extend(((a,b,d),(a,d,c)) if (i+j)%2==0 else ((a,b,c),(b,d,c)))
    elems=np.asarray(elems,int); p=nodes[elems]
    ab=p[:,1]-p[:,0]; ac=p[:,2]-p[:,0]
    area=.5*np.abs(ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0])
    return SimpleNamespace(nodes=nodes,elems=elems,area_e=area,ne=len(elems))

def graded_mesh(far_field_factor):
    n=17; x=np.linspace(0,1,n); inner=np.linspace(-.25,.25,n-2); y=np.r_[-.25-.25*far_field_factor,inner,.25+.25*far_field_factor]
    X,Y=np.meshgrid(x,y); nodes=np.c_[X.ravel(),Y.ravel()]; elems=[]
    for j in range(n-1):
        for i in range(n-1):
            a=j*n+i; b=a+1; c=a+n; d=c+1; elems.extend(((a,b,d),(a,d,c)) if (i+j)%2==0 else ((a,b,c),(b,d,c)))
    elems=np.asarray(elems,int); p=nodes[elems]; ab=p[:,1]-p[:,0]; ac=p[:,2]-p[:,0]
    return SimpleNamespace(nodes=nodes,elems=elems,area_e=.5*np.abs(ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]),ne=len(elems))

@dataclass(frozen=True)
class StateMesh:
    nodes: np.ndarray
    elems: np.ndarray
    area_e: np.ndarray
    ne: int
    element_damage_gp: np.ndarray

@dataclass(frozen=True)
class State:
    mesh: StateMesh
    damage: np.ndarray
    crack_network: object
    junction_process_state: dict

def network(path): return CrackNetworkState.one_tip(path)

def test_v12_has_new_identity_and_does_not_mutate_v11_identity():
    assert MODEL_ID=="sharp_wake_mechanically_separating_v12"
    assert V11_ID=="sharp_wake_causal_v11"

@pytest.mark.parametrize("angle",[0,15,-15,30,-30,45,-45])
@pytest.mark.parametrize("phase",[0.,.037])
def test_offaxis_and_endpoint_phases_never_silently_fall_back(angle,phase):
    m=mesh(); p0=np.array((.2+phase,0.)); length=.5; radians=np.deg2rad(angle); p1=p0+length*np.array((np.cos(radians),np.sin(radians)))
    exact,_=causal_segment_support(m,p0,p1); selected,audit=mechanically_separating_graph_support(m,network((p0,p1)))
    assert audit.certified and set(exact).issubset(set(selected)) and len(selected)>len(exact)
    assert audit.maximum_normal_support_width_m>0 and audit.selected_area_m2>0

def test_partition_and_segment_insertion_order_are_invariant():
    m=mesh(); whole=network(((.125,0.),(.875,0.))); split=network(((.125,0.),(.5,0.),(.875,0.)))
    a,aa=mechanically_separating_graph_support(m,whole); b,bb=mechanically_separating_graph_support(m,split)
    np.testing.assert_array_equal(a,b)
    assert aa.segment_partition_invariant_length_m==pytest.approx(bb.segment_partition_invariant_length_m)
    assert aa.support_fingerprint==bb.support_fingerprint

@pytest.mark.parametrize("fractions",[(1.,),(.5,.5),(.25,.25,.25,.25),(.1,.2,.3,.4),(.07,.13,.31,.49)])
def test_canonical_certificate_is_invariant_to_collinear_history_partition(fractions):
    m=mesh(n=33); start=.125; end=.875; points=[(start,0.)]; position=start
    for fraction in fractions:
        position+=(end-start)*fraction; points.append((position,0.))
    net=network(tuple(points)); ids,a=mechanically_separating_graph_support(m,net)
    reference_net=network(((start,0.),(end,0.))); reference_ids,reference=mechanically_separating_graph_support(m,reference_net)
    np.testing.assert_array_equal(ids,reference_ids)
    assert certification_arcs(net)==certification_arcs(reference_net)
    assert a.support_fingerprint==reference.support_fingerprint
    assert a.certificate_fingerprint==reference.certificate_fingerprint
    assert a.segment_partition_invariant_length_m==pytest.approx(reference.segment_partition_invariant_length_m)
    assert a.selected_area_m2==pytest.approx(reference.selected_area_m2)

@pytest.mark.parametrize("fractions",[(1.,),(.5,.5),(.25,)*4,(.125,)*8,(2/24,4/24,7/24,11/24)])
def test_incrementally_accepted_collinear_histories_reach_same_mechanical_state(fractions):
    m=mesh(n=65); root=.125; initial_tip=.5; final_tip=.875
    accepted=network(((root,0.),(initial_tip,0.))); ids,a=mechanically_separating_graph_support(m,accepted)
    damage=np.zeros(m.ne); damage[ids]=1.; owner=support_record(m,accepted,damage,ids); points=[(root,0.),(initial_tip,0.)]; position=initial_tip
    for fraction in fractions:
        position+=(final_tip-initial_tip)*fraction; points.append((position,0.)); trial=network(tuple(points))
        ids,a=mechanically_separating_graph_support(m,trial,previous_support=owner,accepted_network=accepted,accepted_damage=damage)
        assert a.mechanically_new_element_count>0
        damage=damage.copy(); damage[ids]=1.; owner=support_record(m,trial,damage,ids); accepted=trial
    reference_ids,reference=mechanically_separating_graph_support(m,network(((root,0.),(final_tip,0.))))
    np.testing.assert_array_equal(ids,reference_ids); np.testing.assert_array_equal(damage,np.isin(np.arange(m.ne),reference_ids).astype(float))
    assert a.support_fingerprint==reference.support_fingerprint
    assert a.certificate_fingerprint==reference.certificate_fingerprint
    assert a.segment_partition_invariant_length_m==pytest.approx(reference.segment_partition_invariant_length_m)

def test_active_tip_metrics_are_signed_and_tip_local():
    m=mesh(n=33); _,a=mechanically_separating_graph_support(m,network(((.125,0.),(.875,0.))))
    assert a.h_tip_max_m>0 and a.h_tip_median_m>0 and a.h_tip_tangent_m>0 and a.h_tip_normal_m>0
    assert a.active_tip_signed_footprint_m==pytest.approx(a.active_tip_support_axial_extent_m-.75)
    assert a.active_tip_forward_leakage_m==pytest.approx(max(0.,a.active_tip_signed_footprint_m))
    assert a.active_tip_backward_undershoot_m==pytest.approx(max(0.,-a.active_tip_signed_footprint_m))
    assert a.endpoint_footprint_error_m==pytest.approx(abs(a.active_tip_signed_footprint_m))

def test_growth_is_monotone_and_former_tip_is_closed_as_interior():
    m=mesh(); first=network(((.125,0.),(.5,0.))); second=network(((.125,0.),(.5,0.),(.875,0.)))
    a,_=mechanically_separating_graph_support(m,first); damage=np.zeros(m.ne); damage[a]=1.; owned=support_record(m,first,damage,a)
    b,audit=mechanically_separating_graph_support(m,second,previous_support=owned,accepted_network=first,accepted_damage=damage)
    assert set(a).issubset(set(b)); assert len(audit.newly_selected_element_ids)>0
    assert dict(audit.vertex_classes)["0.5,0"]=="degree_two_interior"

def test_branch_junction_and_active_tips_are_classified_explicitly():
    root=CrackBranchState("b00000000",None,0,0,((.1,0.),(.5,0.)),(0.,),status="arrested")
    up=CrackBranchState("b00000001","b00000000",1,1,((.5,0.),(.8,.2)),(np.arctan2(.2,.3),))
    down=CrackBranchState("b00000002","b00000000",1,1,((.5,0.),(.8,-.2)),(np.arctan2(-.2,.3),))
    net=CrackNetworkState((root,up,down),branching_enabled=True)
    classes=classify_graph_vertices(net)
    assert classes[(.5,0.)]==frozenset(("branch_junction",))
    assert classes[(.8,.2)]==frozenset(("active_tip",)); assert classes[(.8,-.2)]==frozenset(("active_tip",))

def test_empty_graph_fails_closed():
    m=mesh(); net=CrackNetworkState.one_tip(((.2,0.),),initial_orientation_rad=0.)
    with pytest.raises(RuntimeError,match="no edges"):
        mechanically_separating_graph_support(m,net)

def test_independent_path_search_detects_deliberately_defective_exact_only_support():
    m=mesh(n=17); net=network(((.125,0.),(.875,0.)))
    exact,_=causal_segment_support(m,np.array((.125,0.)),np.array((.875,0.)))
    certificate=independent_intact_path_certificate(m,net,exact)
    assert certificate["intact_cross_graph_path_exists"]
    assert certificate["minimum_crossing_path_length"]>0
    assert certificate["bridge_node_ids"] and certificate["bridge_element_ids"]

def test_independent_certificate_rejects_vacuous_missing_side_seeds():
    m=mesh(n=17); centered=network(((.125,0.),(.875,0.)))
    broad=independent_intact_path_certificate(m,centered,np.arange(m.ne))
    assert broad["insufficient_seed_segment_ids"]==("b00000000:arc0",)
    clipped=network(((.125,.48),(.875,.48)))
    exact,_=causal_segment_support(m,np.array((.125,.48)),np.array((.875,.48)))
    boundary=independent_intact_path_certificate(m,clipped,exact)
    assert boundary["insufficient_seed_segment_ids"]

def test_edge_too_short_for_two_sided_certificate_fails_closed():
    m=mesh(n=33); h=np.sqrt(2)/32; short=network(((.125,0.),(.125+.5*h,0.)))
    with pytest.raises(RuntimeError,match="CERTIFICATE_ARC_TOO_SHORT"):
        mechanically_separating_graph_support(m,short)

def test_far_field_grading_does_not_change_crack_local_support_or_metrics():
    net=network(((.125,0.),(.875,0.))); reference=None
    for factor in (1,2,4,8,16):
        ids,a=mechanically_separating_graph_support(graded_mesh(factor),net)
        value=(tuple(ids),a.local_h_max_m,a.local_h_median_m,a.width_over_h,a.forward_leakage_over_h,a.independent_separation_certified)
        if reference is None: reference=value
        else:
            assert value[0]==reference[0]; np.testing.assert_allclose(value[1:5],reference[1:5],rtol=0,atol=1e-14)
            assert value[5] is reference[5]

def test_unique_graph_length_unions_partially_overlapping_collinear_edges():
    a=CrackBranchState("b00000000",None,0,0,((.1,0.),(.6,0.)),(0.,),status="arrested")
    b=CrackBranchState("b00000001","b00000000",1,1,((.4,0.),(.9,0.)),(0.,),status="arrested")
    assert unique_graph_length(CrackNetworkState((a,b),branching_enabled=True))==pytest.approx(.8)

@pytest.mark.parametrize("ratio,accepted",[(.25,False),(.5,False),(1,True),(2,True),(4,True),(8,True)])
def test_event_resolution_matrix_is_history_partition_invariant(ratio,accepted):
    m=mesh(n=33); start=.125; old_tip=.5; first=network(((start,0.),(old_tip,0.)))
    ids,_=_graph_support(m,first); damage=np.zeros(m.ne); damage[ids]=1.; owned=support_record(m,first,damage,ids)
    second=network(((start,0.),(old_tip,0.),(old_tip+ratio/32,0.)))
    if not accepted:
        with pytest.raises(RuntimeError,match="REQUIRES_ACTIVE_TIP_ALIGNMENT_REMESH|CERTIFICATE_ARC_TOO_SHORT|INSUFFICIENT_OPPOSITE_SIDE_SEEDS"):
            _graph_support(m,second,previous_support=owned,accepted_network=first,accepted_damage=damage)
    else:
        _,audit=_graph_support(m,second,previous_support=owned,accepted_network=first,accepted_damage=damage)
        assert audit.mechanically_new_element_count>0 and audit.accepted_damage_fingerprint!=audit.trial_damage_fingerprint

def test_resolved_long_event_and_two_step_sequence_have_same_support_and_stiffness():
    m=mesh(n=33); initial=network(((.125,0.),(.5,0.))); base,_=_graph_support(m,initial); damage=np.zeros(m.ne); damage[base]=1.; owner=support_record(m,initial,damage,base)
    long=network(((.125,0.),(.5,0.),(.75,0.))); long_ids,_=_graph_support(m,long,previous_support=owner,accepted_network=initial,accepted_damage=damage)
    middle=network(((.125,0.),(.5,0.),(.625,0.))); middle_ids,_=_graph_support(m,middle,previous_support=owner,accepted_network=initial,accepted_damage=damage)
    middle_damage=damage.copy(); middle_damage[middle_ids]=1.; middle_owner=support_record(m,middle,middle_damage,middle_ids)
    final=network(((.125,0.),(.5,0.),(.625,0.),(.75,0.))); final_ids,_=_graph_support(m,final,previous_support=middle_owner,accepted_network=middle,accepted_damage=middle_damage)
    np.testing.assert_array_equal(long_ids,final_ids)
    long_damage=damage.copy(); long_damage[long_ids]=1.; final_damage=middle_damage.copy(); final_damage[final_ids]=1.
    np.testing.assert_array_equal(long_damage,final_damage)

def test_offgrid_active_tip_fails_closed_when_production_alignment_is_required():
    m=mesh(n=17); net=network(((.125,0.),(.731,.013)))
    with pytest.raises(RuntimeError,match="REQUIRES_ACTIVE_TIP_ALIGNMENT_REMESH"):
        _graph_support(m,net)

def test_remote_or_whole_mesh_previous_support_fails_locality_gate():
    m=mesh(n=17); net=network(((.125,0.),(.875,0.)))
    damage=np.ones(m.ne); owned=support_record(m,net,damage,np.arange(m.ne))
    with pytest.raises(RuntimeError,match="RETAINED_SUPPORT_NOT_LOCAL"):
        mechanically_separating_graph_support(m,net,previous_support=owned,accepted_network=net,accepted_damage=damage)

def test_support_owner_rejects_stale_mesh_and_damage_provenance():
    m=mesh(n=17); net=network(((.125,0.),(.5,0.))); ids,_=mechanically_separating_graph_support(m,net)
    damage=np.zeros(m.ne); damage[ids]=1.; owned=support_record(m,net,damage,ids)
    moved=mesh(n=17); moved.nodes=moved.nodes.copy(); moved.nodes[0,0]+=.001
    with pytest.raises(RuntimeError,match="STALE_SUPPORT_MESH_GEOMETRY_FINGERPRINT"):
        mechanically_separating_graph_support(moved,net,previous_support=owned,accepted_network=net,accepted_damage=damage)
    corrupted=damage.copy(); corrupted[ids[0]]=0.
    with pytest.raises(RuntimeError,match="STALE_SUPPORT_ACCEPTED_DAMAGE_FINGERPRINT"):
        mechanically_separating_graph_support(m,net,previous_support=owned,accepted_network=net,accepted_damage=corrupted)

def test_subcell_sequential_event_without_new_mechanical_state_fails_closed():
    m=mesh(n=33); h=np.sqrt(2)/32; first=network(((.125,0.),(.125+3*h,0.)))
    ids,_=mechanically_separating_graph_support(m,first); damage=np.zeros(m.ne); damage[ids]=1.; owned=support_record(m,first,damage,ids)
    second=network(((.125,0.),(.125+3*h,0.),(.125+3.25*h,0.)))
    with pytest.raises(RuntimeError,match="NO_MECHANICALLY_NEW_SUPPORT"):
        mechanically_separating_graph_support(m,second,previous_support=owned,accepted_network=first,accepted_damage=damage)

@pytest.mark.parametrize("perturb",[False,True])
def test_near_coalescing_distinct_branches_fail_closed(perturb):
    m=mesh(n=13,perturb=perturb)
    trunk=CrackBranchState("b00000000",None,0,0,((.1,-.1),(.42,-.03),(.62,.08)),(0.,0.),status="arrested")
    upper=CrackBranchState("b00000001","b00000000",1,1,((.62,.08),(.86,.22)),(.5,))
    close=CrackBranchState("b00000002","b00000000",1,1,((.62,.08),(.84,.205)),(.5,))
    net=CrackNetworkState((trunk,upper,close),branching_enabled=True)
    with pytest.raises(RuntimeError,match="PREMATURE_MECHANICAL_COALESCENCE"):
        mechanically_separating_graph_support(m,net)

def two_branch_network(left,right):
    a=CrackBranchState("b00000000",None,0,0,tuple(left),(0.,),status="active")
    # The V11 schema requires one ancestry root but does not require geometric
    # contact between parent and child; physical components are geometry-based.
    b=CrackBranchState("b00000001","b00000000",1,1,tuple(right),(0.,),status="active")
    return CrackNetworkState((a,b),primary_branch_id="b00000000",branching_enabled=True)

def test_support_component_labels_distinguish_edge_and_node_connectivity():
    m=mesh(n=17); selected=np.array((0,1,3))
    edge=selected_support_components(m,selected,shared_nodes=2); node=selected_support_components(m,selected,shared_nodes=1)
    assert len(edge)==len(selected) and len(node)==len(selected)
    assert len(set(node))<=len(set(edge))

def test_separated_physical_cracks_retain_distinct_node_components():
    m=mesh(n=33); net=two_branch_network(((.125,-.2),(.875,-.2)),((.125,.2),(.875,.2)))
    _,audit=mechanically_separating_graph_support(m,net)
    assert graph_component_count(net)==2 and audit.graph_component_count==2
    assert len(set(audit.support_component_ids_node))==2 and not audit.illegal_support_connection
    assert audit.minimum_support_component_separation_m>0

def test_mesh_unresolved_distinct_parallel_cracks_fail_closed_explicitly():
    m=mesh(n=33); net=two_branch_network(((.125,-.01),(.875,-.01)),((.125,.01),(.875,.01)))
    with pytest.raises(RuntimeError,match="DISTINCT_CRACK_COMPONENTS_UNRESOLVED_AT_CURRENT_MESH"):
        mechanically_separating_graph_support(m,net)

def test_geometric_crossing_without_graph_junction_fails_closed_explicitly():
    m=mesh(n=33); net=two_branch_network(((.2,-.2),(.8,.2)),((.2,.2),(.8,-.2)))
    with pytest.raises(RuntimeError,match="DISTINCT_CRACK_COMPONENTS_UNRESOLVED_AT_CURRENT_MESH"):
        mechanically_separating_graph_support(m,net)
    _,audit=mechanically_separating_graph_support(m,net,return_uncertified_audit_for_screen=True)
    assert not audit.certified and audit.illegal_support_connection

def test_kink_and_y_junction_sector_certificates_are_nonvacuous():
    m=mesh(n=65); kink=network(((.125,0.),(.5,0.),(.75,.25))); _,kink_a=mechanically_separating_graph_support(m,kink)
    assert len(kink_a.junction_sector_certificates)==1
    assert kink_a.junction_sector_certificates[0].junction_certificate_status=="ACCEPTED"
    root=CrackBranchState("b00000000",None,0,0,((.125,0.),(.5,0.)),(0.,),status="arrested")
    up=CrackBranchState("b00000001","b00000000",1,1,((.5,0.),(.75,.25)),(.5,))
    down=CrackBranchState("b00000002","b00000000",1,1,((.5,0.),(.75,-.25)),(-.5,))
    _,y_a=mechanically_separating_graph_support(m,CrackNetworkState((root,up,down),branching_enabled=True))
    assert len(y_a.junction_sector_certificates)==1
    certificate=y_a.junction_sector_certificates[0]
    assert certificate.junction_certificate_status=="ACCEPTED" and all(certificate.sector_seed_counts)

def test_junction_sector_verifier_rejects_deliberately_removed_arm_support():
    m=mesh(n=65); kink=network(((.125,0.),(.5,0.),(.75,.25))); selected,_=mechanically_separating_graph_support(m,kink)
    centroids=np.mean(m.nodes[m.elems],axis=1); p0=np.array((.5,0.)); tangent=np.array((1.,1.))/np.sqrt(2); h=np.sqrt(2)/64
    rel=centroids-p0; axial=rel@tangent; normal=np.abs(rel@np.array((-tangent[1],tangent[0])))
    removed=set(map(int,np.flatnonzero((axial>=0)&(axial<=6*h)&(normal<=3*h))))
    damaged=set(map(int,selected))-removed
    certificates=junction_sector_certificates(m,kink,sorted(damaged))
    assert certificates and certificates[0].junction_certificate_status!="ACCEPTED"

def test_duplicate_partial_overlap_and_cross_branch_collinear_continuation_are_explicitly_rejected():
    m=mesh(n=33)
    root=CrackBranchState("b00000000",None,0,0,((.125,0.),(.625,0.)),(0.,),status="arrested")
    overlap=CrackBranchState("b00000001","b00000000",1,1,((.5,0.),(.875,0.)),(0.,))
    with pytest.raises(RuntimeError,match="DUPLICATE_OR_OVERLAPPING_GRAPH_EDGE"):
        mechanically_separating_graph_support(m,CrackNetworkState((root,overlap),branching_enabled=True))
    continuation=CrackBranchState("b00000001","b00000000",1,1,((.625,0.),(.875,0.)),(0.,))
    with pytest.raises(RuntimeError,match="COLLINEAR_PARENT_CHILD_CONTINUATION_REQUIRES_CANONICAL_BRANCH_SEMANTICS"):
        mechanically_separating_graph_support(m,CrackNetworkState((root,continuation),branching_enabled=True))

def test_branch_tuple_order_does_not_change_support_or_fingerprint():
    m=mesh(n=33)
    root=CrackBranchState("b00000000",None,0,0,((.1,0.),(.5,0.)),(0.,),status="arrested")
    up=CrackBranchState("b00000001","b00000000",1,1,((.5,0.),(.8,.2)),(.5,))
    down=CrackBranchState("b00000002","b00000000",1,1,((.5,0.),(.8,-.2)),(-.5,))
    a,aa=mechanically_separating_graph_support(m,CrackNetworkState((root,up,down),branching_enabled=True))
    b,bb=mechanically_separating_graph_support(m,CrackNetworkState((down,root,up),branching_enabled=True))
    np.testing.assert_array_equal(a,b)
    assert aa.support_fingerprint==bb.support_fingerprint and aa.graph_fingerprint==bb.graph_fingerprint

def test_trial_application_is_rollback_exact_and_records_v12_identity():
    raw=mesh(n=17); sm=StateMesh(raw.nodes,raw.elems,raw.area_e,raw.ne,np.zeros(raw.ne)); net=network(((.125,0.),(.75,0.)))
    accepted=State(sm,np.zeros(len(sm.nodes)),net,{"owned":"accepted"})
    before_gp=accepted.mesh.element_damage_gp.copy(); before_damage=accepted.damage.copy(); before_junction=dict(accepted.junction_process_state)
    trial,audit=apply_mechanically_separating_graph(accepted,net)
    np.testing.assert_array_equal(accepted.mesh.element_damage_gp,before_gp); np.testing.assert_array_equal(accepted.damage,before_damage)
    assert accepted.junction_process_state==before_junction
    assert trial.junction_process_state["crack_representation"]==MODEL_ID and audit.certified
    assert np.count_nonzero(trial.mesh.element_damage_gp)>0
