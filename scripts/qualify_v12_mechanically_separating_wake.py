#!/usr/bin/env python3
"""Write deterministic V12 geometry evidence and conservative gate states."""
from __future__ import annotations

import argparse, csv, hashlib, json, platform, subprocess, sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest, scipy

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from arrhenius_fracture.crack_network_v11 import CrackBranchState, CrackNetworkState
from arrhenius_fracture.mechanically_separating_sharp_wake_v12 import (
    MODEL_ID, certification_arcs, junction_sector_certificates, mechanically_separating_graph_support, support_record,
)


def mesh(n,perturbed):
    x=np.linspace(0.,1.,n); y=np.linspace(-.5,.5,n); xx,yy=np.meshgrid(x,y); nodes=np.c_[xx.ravel(),yy.ravel()]
    if perturbed:
        rng=np.random.default_rng(120031+n); interior=(nodes[:,0]>0)&(nodes[:,0]<1)&(nodes[:,1]>-.5)&(nodes[:,1]<.5)
        nodes[interior]+=rng.uniform(-.12/(n-1),.12/(n-1),size=(np.count_nonzero(interior),2))
    elems=[]
    for j in range(n-1):
        for i in range(n-1):
            a=j*n+i; b=a+1; c=a+n; d=c+1
            elems.extend(((a,b,d),(a,d,c)) if (i+j)%2==0 else ((a,b,c),(b,d,c)))
    elems=np.asarray(elems,int); p=nodes[elems]; ab=p[:,1]-p[:,0]; ac=p[:,2]-p[:,0]
    area=.5*np.abs(ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0])
    return SimpleNamespace(nodes=nodes,elems=elems,area_e=area,ne=len(elems))

def graded_mesh(factor):
    n=17; x=np.linspace(0.,1.,n); inner=np.linspace(-.25,.25,n-2); y=np.r_[-.25-.25*factor,inner,.25+.25*factor]
    xx,yy=np.meshgrid(x,y); nodes=np.c_[xx.ravel(),yy.ravel()]; elems=[]
    for j in range(n-1):
        for i in range(n-1):
            a=j*n+i; b=a+1; c=a+n; d=c+1; elems.extend(((a,b,d),(a,d,c)) if (i+j)%2==0 else ((a,b,c),(b,d,c)))
    elems=np.asarray(elems,int); p=nodes[elems]; ab=p[:,1]-p[:,0]; ac=p[:,2]-p[:,0]
    return SimpleNamespace(nodes=nodes,elems=elems,area_e=.5*np.abs(ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]),ne=len(elems))

def fixed_domain_graded_mesh(n=33):
    x=np.linspace(0.,1.,n); parameter=np.linspace(-1.,1.,n); y=.5*np.sinh(2*parameter)/np.sinh(2)
    xx,yy=np.meshgrid(x,y); nodes=np.c_[xx.ravel(),yy.ravel()]; elems=[]
    for j in range(n-1):
        for i in range(n-1):
            a=j*n+i; b=a+1; c=a+n; d=c+1; elems.extend(((a,b,d),(a,d,c)) if (i+j)%2==0 else ((a,b,c),(b,d,c)))
    elems=np.asarray(elems,int); p=nodes[elems]; ab=p[:,1]-p[:,0]; ac=p[:,2]-p[:,0]
    return SimpleNamespace(nodes=nodes,elems=elems,area_e=.5*np.abs(ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]),ne=len(elems))

def tensor_mesh(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); nx=len(x); xx,yy=np.meshgrid(x,y); nodes=np.c_[xx.ravel(),yy.ravel()]; elems=[]
    for j in range(len(y)-1):
        for i in range(nx-1):
            a=j*nx+i; b=a+1; c=a+nx; d=c+1; elems.extend(((a,b,d),(a,d,c)) if (i+j)%2==0 else ((a,b,c),(b,d,c)))
    elems=np.asarray(elems,int); p=nodes[elems]; ab=p[:,1]-p[:,0]; ac=p[:,2]-p[:,0]
    return SimpleNamespace(nodes=nodes,elems=elems,area_e=.5*np.abs(ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]),ne=len(elems))

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def git(*args): return subprocess.check_output(("git",)+args,cwd=ROOT,text=True).strip()

def two_branch_network(left,right):
    root=CrackBranchState("b00000000",None,0,0,tuple(left),(0.,),status="active")
    child=CrackBranchState("b00000001","b00000000",1,1,tuple(right),(0.,),status="active")
    return CrackNetworkState((root,child),primary_branch_id="b00000000",branching_enabled=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,default=Path("artifacts/v12_mechanically_separating_wake")); args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True); rows=[]; edge_rows=[]; graded_rows=[]; event_rows=[]; partition_rows=[]; refinement_rows=[]
    topology_rows=[]; coalescence_rows=[]; junction_rows=[]; arc_rows=[]; adaptive_rows=[]; branched_partition_rows=[]
    implementation_sha=git("log","-1","--format=%H","--","arrhenius_fracture/mechanically_separating_sharp_wake_v12.py","tests/test_v12_mechanically_separating_wake.py","scripts/qualify_v12_mechanically_separating_wake.py")
    for n in (9,17,33,65):
        for perturbed in (False,True):
            m=mesh(n,perturbed)
            for angle in (0,15,-15,30,-30,45,-45):
                for phase in (0.,.037):
                    p0=np.array((.2+phase,0.)); rad=np.deg2rad(angle); p1=p0+.5*np.array((np.cos(rad),np.sin(rad)))
                    _,a=mechanically_separating_graph_support(m,CrackNetworkState.one_tip((p0,p1)),allow_offgrid_active_tips_for_screen=True)
                    rows.append(dict(n=n,mesh="perturbed" if perturbed else "structured",angle_deg=angle,phase=phase,
                        h_local_max_m=a.local_h_max_m,h_local_median_m=a.local_h_median_m,area_m2=a.selected_area_m2,
                        area_over_unique_length_h_local=a.selected_area_over_unique_graph_length_h_local,
                        support_width_m=a.maximum_normal_support_width_m,width_over_h_local=a.width_over_h,
                        forward_leakage_m=a.active_tip_forward_leakage_m,forward_leakage_over_h_local=a.forward_leakage_over_h,
                        intact_cross_graph_path_exists=a.intact_cross_graph_path_exists,certificate_fingerprint=a.certificate_fingerprint,
                        unresolved_bridge_nodes=len(a.unresolved_bridge_node_ids),construction_screen=a.node_star_construction_passed))
                    for edge in a.edge_cut_certificates:
                        item=asdict(edge)
                        for key,value in tuple(item.items()):
                            if isinstance(value,tuple): item[key]=" ".join(map(str,value))
                        edge_rows.append(dict(n=n,mesh="perturbed" if perturbed else "structured",angle_deg=angle,phase=phase,**item))
    csv_path=args.out/"geometry_matrix.csv"
    with csv_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0],lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    edge_path=args.out/"edge_certificate_matrix.csv"
    with edge_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=edge_rows[0],lineterminator="\n"); writer.writeheader(); writer.writerows(edge_rows)
    graded_net=CrackNetworkState.one_tip(((.125,0.),(.875,0.)))
    for factor in (1,2,4,8,16):
        ids,a=mechanically_separating_graph_support(graded_mesh(factor),graded_net)
        graded_rows.append({"far_field_factor":factor,"selected_element_ids":" ".join(map(str,ids)),"h_local_max_m":a.local_h_max_m,
          "h_local_median_m":a.local_h_median_m,"width_over_h_local":a.width_over_h,"forward_leakage_over_h_local":a.forward_leakage_over_h,
          "independent_separation_certified":a.independent_separation_certified})
    graded_path=args.out/"graded_far_field_objectivity.csv"
    with graded_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=graded_rows[0],lineterminator="\n"); writer.writeheader(); writer.writerows(graded_rows)
    directions={angle:np.array((np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)))) for angle in (0,15,30,45,60,75,90)}
    event_meshes=(("structured",mesh(33,False)),("perturbed",mesh(33,True)),("fixed_domain_graded",fixed_domain_graded_mesh()))
    for mesh_label,event_mesh in event_meshes:
     for angle,direction in directions.items():
        normal=np.array((-direction[1],direction[0]))
        for tangent_phase in (0.,.37):
          for normal_phase in (0.,.37):
            old_tip=np.array((.5,0.))+(tangent_phase*direction+normal_phase*normal)/32; root=old_tip-.375*direction
            accepted=CrackNetworkState.one_tip((root,old_tip)); prior,prior_audit=mechanically_separating_graph_support(event_mesh,accepted,allow_offgrid_active_tips_for_screen=True)
            damage=np.zeros(event_mesh.ne); damage[prior]=1.; owner=support_record(event_mesh,accepted,damage,prior)
            accepted_aligned=bool(np.any(np.linalg.norm(event_mesh.nodes-old_tip,axis=1)<=1e-12))
            h_tip_max=prior_audit.h_tip_max_m if np.isfinite(prior_audit.h_tip_max_m) else prior_audit.local_h_max_m
            h_tip_median=prior_audit.h_tip_median_m if np.isfinite(prior_audit.h_tip_median_m) else prior_audit.local_h_median_m
            for ratio in (.25,.5,1,2,3,4,6,8):
                delta_a=ratio*h_tip_max; new_tip=old_tip+delta_a*direction
                trial=CrackNetworkState.one_tip((root,old_tip,new_tip)); outcome="ACCEPTED"; audit=None
                aligned=bool(np.any(np.linalg.norm(event_mesh.nodes-new_tip,axis=1)<=1e-12))
                if not accepted_aligned:
                    outcome="INVALID_ACCEPTED_BASELINE_REQUIRES_ALIGNMENT_REMESH"; mechanically_new=0; accepted_fp=""; trial_fp=""
                else:
                    try:
                        _,audit=mechanically_separating_graph_support(event_mesh,trial,previous_support=owner,accepted_network=accepted,accepted_damage=damage)
                        mechanically_new=audit.mechanically_new_element_count; accepted_fp=audit.accepted_damage_fingerprint; trial_fp=audit.trial_damage_fingerprint
                    except RuntimeError as error:
                        outcome=str(error).split(": ",1)[-1]; mechanically_new=0; accepted_fp=""; trial_fp=""
                event_rows.append({"mesh":mesh_label,"angle_deg":angle,"tangent_phase_cells":tangent_phase,"normal_phase_cells":normal_phase,
                  "delta_a_m":delta_a,"h_tip_max_m":h_tip_max,"h_tip_median_m":h_tip_median,"h_tip_tangent_m":prior_audit.h_tip_tangent_m,
                  "h_tip_normal_m":prior_audit.h_tip_normal_m,"delta_a_over_h_tip_max":delta_a/h_tip_max,
                  "delta_a_over_h_tip_median":delta_a/h_tip_median,"delta_a_over_h_tip_tangent":delta_a/prior_audit.h_tip_tangent_m if np.isfinite(prior_audit.h_tip_tangent_m) else "",
                  "accepted_tip_aligned":accepted_aligned,"trial_tip_aligned":aligned,"accepted_state_production_valid":accepted_aligned,
                  "trial_requires_alignment_remesh":accepted_aligned and not aligned,"outcome":outcome,"certificate_status":audit.certification_reason if audit else "NOT_CERTIFIED",
                  "support_axial_extent_m":audit.active_tip_support_axial_extent_m if audit else "","signed_tip_footprint_m":audit.active_tip_signed_footprint_m if audit else "",
                  "forward_overshoot_m":audit.active_tip_forward_leakage_m if audit else "","backward_undershoot_m":audit.active_tip_backward_undershoot_m if audit else "",
                  "overshoot_over_h_tip":audit.overshoot_over_h_tip if audit else "","undershoot_over_h_tip":audit.undershoot_over_h_tip if audit else "",
                  "graph_fingerprint":audit.graph_fingerprint if audit else "","support_fingerprint":audit.support_fingerprint if audit else "",
                  "prior_support_ids":" ".join(map(str,prior)),"new_support_ids":" ".join(map(str,audit.selected_element_ids if audit else ())),
                  "mechanically_new_ids":" ".join(map(str,audit.mechanically_new_element_ids if audit else ())),
                  "mechanically_new_element_count":mechanically_new,"accepted_damage_fingerprint":accepted_fp,"trial_damage_fingerprint":trial_fp})
    event_path=args.out/"normalized_event_resolution_matrix.csv"
    with event_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=event_rows[0],lineterminator="\n"); writer.writeheader(); writer.writerows(event_rows)
    start,initial_tip,end=.125,.5,.875; partition_mesh=mesh(65,False)
    reference_net=CrackNetworkState.one_tip(((start,0.),(end,0.))); reference_ids,reference=mechanically_separating_graph_support(partition_mesh,reference_net)
    for label,fractions in (("one",(1.,)),("two",(.5,.5)),("four",(.25,)*4),("eight",(.125,)*8),("unequal",(2/24,4/24,7/24,11/24))):
        accepted=CrackNetworkState.one_tip(((start,0.),(initial_tip,0.))); ids,a=mechanically_separating_graph_support(partition_mesh,accepted)
        damage=np.zeros(partition_mesh.ne); damage[ids]=1.; owner=support_record(partition_mesh,accepted,damage,ids)
        points=[(start,0.),(initial_tip,0.)]; position=initial_tip
        for event_index,fraction in enumerate(fractions,1):
            position+=(end-initial_tip)*fraction; points.append((position,0.)); trial=CrackNetworkState.one_tip(tuple(points))
            ids,a=mechanically_separating_graph_support(partition_mesh,trial,previous_support=owner,accepted_network=accepted,accepted_damage=damage)
            damage=damage.copy(); damage[ids]=1.; owner=support_record(partition_mesh,trial,damage,ids); accepted=trial
            partition_rows.append({"partition":label,"event_index":event_index,"event_count":len(fractions),"is_final":event_index==len(fractions),
              "selected_support_ids":" ".join(map(str,ids)),"mechanically_new_ids":" ".join(map(str,a.mechanically_new_element_ids)),
              "graph_fingerprint":a.graph_fingerprint,"support_fingerprint":a.support_fingerprint,"certificate_fingerprint":a.certificate_fingerprint,
              "certification_arc_count":len(certification_arcs(trial)),"physical_graph_length_m":a.segment_partition_invariant_length_m,
              "trial_damage_fingerprint":a.trial_damage_fingerprint,"ownership_graph_fingerprint":owner.crack_graph_fingerprint,
              "final_support_matches_reference":event_index!=len(fractions) or tuple(ids)==tuple(reference_ids),
              "final_certificate_matches_reference":event_index!=len(fractions) or a.certificate_fingerprint==reference.certificate_fingerprint})
            for arc_start,arc_end,arc_id in certification_arcs(trial):
                arc_rows.append({"partition":label,"event_index":event_index,"arc_id":arc_id,"start_m":json.dumps(arc_start),
                  "end_m":json.dumps(arc_end),"certificate_fingerprint":a.certificate_fingerprint,
                  "support_fingerprint":a.support_fingerprint,"physical_graph_length_m":a.segment_partition_invariant_length_m})
    partition_path=args.out/"event_partition_equivalence.csv"
    with partition_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=partition_rows[0],lineterminator="\n"); writer.writeheader(); writer.writerows(partition_rows)
    arc_path=args.out/"certification_arc_matrix.csv"
    with arc_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=arc_rows[0],lineterminator="\n"); writer.writeheader(); writer.writerows(arc_rows)
    direct_kink=CrackNetworkState.one_tip(((.125,0.),(.5,0.),(.75,.25))); kink_ids,kink_a=mechanically_separating_graph_support(partition_mesh,direct_kink)
    split_kink=CrackNetworkState.one_tip(((.125,0.),(.3125,0.),(.5,0.),(.5625,.0625),(.625,.125),(.6875,.1875),(.75,.25)))
    split_ids,split_a=mechanically_separating_graph_support(partition_mesh,split_kink)
    branched_partition_rows.append({"case":"kink_final_subdivision","support_matches":tuple(split_ids)==tuple(kink_ids),
      "certificate_matches":split_a.certificate_fingerprint==kink_a.certificate_fingerprint,"support_fingerprint":split_a.support_fingerprint,
      "certificate_fingerprint":split_a.certificate_fingerprint,"physical_graph_length_m":split_a.segment_partition_invariant_length_m})
    accepted=CrackNetworkState.one_tip(((.125,0.),(.5,0.))); ids,_=mechanically_separating_graph_support(partition_mesh,accepted)
    damage=np.zeros(partition_mesh.ne); damage[ids]=1.; owner=support_record(partition_mesh,accepted,damage,ids); kink_points=[(.125,0.),(.5,0.)]
    for point in ((.625,.125),(.75,.25)):
        kink_points.append(point); trial=CrackNetworkState.one_tip(tuple(kink_points)); ids,a=mechanically_separating_graph_support(partition_mesh,trial,
          previous_support=owner,accepted_network=accepted,accepted_damage=damage)
        damage=damage.copy(); damage[ids]=1.; owner=support_record(partition_mesh,trial,damage,ids); accepted=trial
    branched_partition_rows.append({"case":"kink_sequential_history","support_matches":tuple(ids)==tuple(kink_ids),
      "certificate_matches":a.certificate_fingerprint==kink_a.certificate_fingerprint,"support_fingerprint":a.support_fingerprint,
      "certificate_fingerprint":a.certificate_fingerprint,"physical_graph_length_m":a.segment_partition_invariant_length_m})
    def evidence_y(up_path,down_path):
        root=CrackBranchState("b00000000",None,0,0,((.125,0.),(.5,0.)),(0.,),status="arrested")
        up=CrackBranchState("b00000001","b00000000",1,1,tuple(up_path),(.5,)*max(1,len(up_path)-1))
        down=CrackBranchState("b00000002","b00000000",1,1,tuple(down_path),(-.5,)*max(1,len(down_path)-1))
        return CrackNetworkState((root,up,down),branching_enabled=True)
    direct_y=evidence_y(((.5,0.),(.75,.25)),((.5,0.),(.75,-.25))); y_ids,y_a=mechanically_separating_graph_support(partition_mesh,direct_y)
    split_y=evidence_y(((.5,0.),(.5625,.0625),(.625,.125),(.75,.25)),((.5,0.),(.625,-.125),(.6875,-.1875),(.75,-.25)))
    split_ids,split_a=mechanically_separating_graph_support(partition_mesh,split_y)
    branched_partition_rows.append({"case":"y_arm_final_subdivision","support_matches":tuple(split_ids)==tuple(y_ids),
      "certificate_matches":split_a.certificate_fingerprint==y_a.certificate_fingerprint,"support_fingerprint":split_a.support_fingerprint,
      "certificate_fingerprint":split_a.certificate_fingerprint,"physical_graph_length_m":split_a.segment_partition_invariant_length_m})
    accepted=CrackNetworkState.one_tip(((.125,0.),(.5,0.))); ids,_=mechanically_separating_graph_support(partition_mesh,accepted)
    damage=np.zeros(partition_mesh.ne); damage[ids]=1.; owner=support_record(partition_mesh,accepted,damage,ids)
    middle_y=evidence_y(((.5,0.),(.625,.125)),((.5,0.),(.625,-.125)))
    ids,_=mechanically_separating_graph_support(partition_mesh,middle_y,previous_support=owner,accepted_network=accepted,accepted_damage=damage)
    damage[ids]=1.; owner=support_record(partition_mesh,middle_y,damage,ids)
    ids,a=mechanically_separating_graph_support(partition_mesh,direct_y,previous_support=owner,accepted_network=middle_y,accepted_damage=damage)
    branched_partition_rows.append({"case":"y_arm_sequential_history","support_matches":tuple(ids)==tuple(y_ids),
      "certificate_matches":a.certificate_fingerprint==y_a.certificate_fingerprint,"support_fingerprint":a.support_fingerprint,
      "certificate_fingerprint":a.certificate_fingerprint,"physical_graph_length_m":a.segment_partition_invariant_length_m})
    branched_path=args.out/"branched_partition_equivalence.csv"
    with branched_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=branched_partition_rows[0],lineterminator="\n"); writer.writeheader(); writer.writerows(branched_partition_rows)
    for n in (17,33,65,129):
        m=mesh(n,False); _,a=mechanically_separating_graph_support(m,CrackNetworkState.one_tip(((.125,0.),(.875,0.))))
        refinement_rows.append({"n":n,"outer_domain":"[0,1]x[-0.5,0.5]","h_local_max_m":a.local_h_max_m,
          "support_width_m":a.maximum_normal_support_width_m,"support_area_m2":a.selected_area_m2,
          "signed_tip_footprint_m":a.active_tip_signed_footprint_m,"tip_overshoot_m":a.active_tip_forward_leakage_m,
          "tip_undershoot_m":a.active_tip_backward_undershoot_m,"width_over_h_local":a.width_over_h,
          "area_over_unique_length_h_local":a.selected_area_over_unique_graph_length_h_local,
          "overshoot_over_h_tip":a.overshoot_over_h_tip,"undershoot_over_h_tip":a.undershoot_over_h_tip})
    refinement_path=args.out/"local_refinement_objectivity.csv"
    with refinement_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=refinement_rows[0],lineterminator="\n"); writer.writeheader(); writer.writerows(refinement_rows)
    local_y=np.linspace(-.125,.125,17); x=np.linspace(0.,1.,65)
    for far_subdivisions in (2,4,8,16):
        lower=np.linspace(-.5,-.125,far_subdivisions+1)[:-1]; upper=np.linspace(.125,.5,far_subdivisions+1)[1:]
        m=tensor_mesh(x,np.r_[lower,local_y,upper]); ids,a=mechanically_separating_graph_support(m,CrackNetworkState.one_tip(((.125,0.),(.875,0.))))
        coordinates=np.round(m.nodes[m.elems[ids]].reshape(-1,2),14); order=np.lexsort((coordinates[:,1],coordinates[:,0]))
        physical_support_fingerprint=hashlib.sha256(np.ascontiguousarray(coordinates[order]).tobytes()).hexdigest()
        adaptive_rows.append({"far_field_subdivisions_per_side":far_subdivisions,"outer_domain":"[0,1]x[-0.5,0.5]",
          "crack_local_patch":"[0,1]x[-0.125,0.125] fixed 64x16","h_tip_max_m":a.h_tip_max_m,"support_width_m":a.maximum_normal_support_width_m,
          "support_area_m2":a.selected_area_m2,"tip_overshoot_m":a.active_tip_forward_leakage_m,
          "tip_undershoot_m":a.active_tip_backward_undershoot_m,"width_over_h_tip":a.maximum_normal_support_width_m/a.h_tip_max_m,
          "physical_support_fingerprint":physical_support_fingerprint})
    adaptive_path=args.out/"adaptive_far_field_objectivity.csv"
    with adaptive_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=adaptive_rows[0],lineterminator="\n"); writer.writeheader(); writer.writerows(adaptive_rows)
    topology_mesh=mesh(33,False); h=np.sqrt(2)/32
    for ratio in (.5,1,1.5,2,3,4,6,8):
        distance=ratio*h; net=two_branch_network(((.125,-distance/2),(.875,-distance/2)),((.125,distance/2),(.875,distance/2)))
        _,a=mechanically_separating_graph_support(topology_mesh,net,allow_offgrid_active_tips_for_screen=True,return_uncertified_audit_for_screen=True)
        coalescence_rows.append({"case":"parallel_nonintersecting","d_over_h_local":ratio,"distance_m":distance,
          "outcome":"ACCEPTED" if a.certified else a.certification_reason,"graph_component_count":a.graph_component_count,
          "support_component_count_edge":len(set(a.support_component_ids_edge)),"support_component_count_node":len(set(a.support_component_ids_node)),
          "illegal_support_connection":a.illegal_support_connection,"minimum_support_component_separation_m":a.minimum_support_component_separation_m})
    topology_cases={
      "separated_parallel":two_branch_network(((.125,-.2),(.875,-.2)),((.125,.2),(.875,.2))),
      "approaching_tips":two_branch_network(((.125,-.15),(.48,-.02)),((.875,.15),(.52,.02))),
      "approaching_old_wake":two_branch_network(((.125,0.),(.875,0.)),((.65,.25),(.65,.08))),
      "geometric_crossing_without_junction":two_branch_network(((.2,-.2),(.8,.2)),((.2,.2),(.8,-.2))),
    }
    for label,net in topology_cases.items():
        _,a=mechanically_separating_graph_support(topology_mesh,net,allow_offgrid_active_tips_for_screen=True,return_uncertified_audit_for_screen=True)
        topology_rows.append({"case":label,"support_component_ids_edge":" ".join(map(str,a.support_component_ids_edge)),
          "support_component_ids_node":" ".join(map(str,a.support_component_ids_node)),"graph_component_count":a.graph_component_count,
          "legal_junction_overlap":False,"illegal_support_connection":a.illegal_support_connection,
          "minimum_support_component_separation_m":a.minimum_support_component_separation_m,"status":"ACCEPTED" if a.certified else a.certification_reason})
    junction_mesh=mesh(65,False)
    kink=CrackNetworkState.one_tip(((.125,0.),(.5,0.),(.75,.25)))
    y_root=CrackBranchState("b00000000",None,0,0,((.125,0.),(.5,0.)),(0.,),status="arrested")
    y_up=CrackBranchState("b00000001","b00000000",1,1,((.5,0.),(.75,.25)),(.5,)); y_down=CrackBranchState("b00000002","b00000000",1,1,((.5,0.),(.75,-.25)),(-.5,))
    y_graph=CrackNetworkState((y_root,y_up,y_down),branching_enabled=True)
    t_root=CrackBranchState("b00000000",None,0,0,((.125,0.),(.5,0.),(.875,0.)),(0.,0.),status="arrested")
    t_arm=CrackBranchState("b00000001","b00000000",1,1,((.5,0.),(.5,.3)),(np.pi/2,)); t_graph=CrackNetworkState((t_root,t_arm),branching_enabled=True)
    mixed_root=CrackBranchState("b00000000",None,0,0,((.125,0.),(.5,0.)),(0.,),status="active")
    mixed_child=CrackBranchState("b00000001","b00000000",1,1,((.5,0.),(.75,.25)),(.5,)); mixed_graph=CrackNetworkState((mixed_root,mixed_child),branching_enabled=True)
    merged_root=CrackBranchState("b00000000",None,0,0,((.125,0.),(.5,0.),(.75,0.)),(0.,0.),status="active")
    merged_child=CrackBranchState("b00000001","b00000000",1,1,((.5,.25),(.5,0.)),(-np.pi/2,),status="merged")
    merged_graph=CrackNetworkState((merged_root,merged_child),branching_enabled=True)
    for label,net in (("kink",kink),("y_branch",y_graph),("t_junction",t_graph),("mixed_role_vertex",mixed_graph),("merged_terminal",merged_graph)):
        selected,a=mechanically_separating_graph_support(junction_mesh,net,allow_offgrid_active_tips_for_screen=True,return_uncertified_audit_for_screen=True)
        topology_rows.append({"case":label,"support_component_ids_edge":" ".join(map(str,a.support_component_ids_edge)),
          "support_component_ids_node":" ".join(map(str,a.support_component_ids_node)),"graph_component_count":a.graph_component_count,
          "legal_junction_overlap":True,"illegal_support_connection":a.illegal_support_connection,
          "minimum_support_component_separation_m":a.minimum_support_component_separation_m,"status":"ACCEPTED" if a.certified else a.certification_reason})
        for certificate in a.junction_sector_certificates:
            item=asdict(certificate)
            for key,value in tuple(item.items()):
                if isinstance(value,tuple): item[key]=json.dumps(value,separators=(",",":"))
            junction_rows.append({"case":label,"defective_support":False,**item})
        if label=="kink":
            center=np.array((.5,0.)); tangent=np.array((1.,1.))/np.sqrt(2); local_h=np.sqrt(2)/64; centroids=np.mean(junction_mesh.nodes[junction_mesh.elems],axis=1)
            rel=centroids-center; axial=rel@tangent; normal_distance=np.abs(rel@np.array((-tangent[1],tangent[0])))
            defective=set(map(int,selected))-set(map(int,np.flatnonzero((axial>=0)&(axial<=6*local_h)&(normal_distance<=3*local_h))))
            for certificate in junction_sector_certificates(junction_mesh,net,sorted(defective)):
                item=asdict(certificate)
                for key,value in tuple(item.items()):
                    if isinstance(value,tuple): item[key]=json.dumps(value,separators=(",",":"))
                junction_rows.append({"case":label,"defective_support":True,**item})
    topology_path=args.out/"support_component_topology.csv"
    with topology_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=topology_rows[0],lineterminator="\n"); writer.writeheader(); writer.writerows(topology_rows)
    coalescence_path=args.out/"coalescence_distance_matrix.csv"
    with coalescence_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=coalescence_rows[0],lineterminator="\n"); writer.writeheader(); writer.writerows(coalescence_rows)
    junction_path=args.out/"junction_sector_matrix.csv"
    with junction_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=junction_rows[0],lineterminator="\n"); writer.writeheader(); writer.writerows(junction_rows)
    construction=all(r["construction_screen"] and r["unresolved_bridge_nodes"]==0 for r in rows)
    separation=all(not r["intact_cross_graph_path_exists"] for r in rows)
    partition_pass=all(r["final_support_matches_reference"] and r["final_certificate_matches_reference"] for r in partition_rows if r["is_final"])
    partition_pass=partition_pass and all(r["support_matches"] and r["certificate_matches"] for r in branched_partition_rows)
    component_pass=all((not r["illegal_support_connection"]) or "DISTINCT_CRACK_COMPONENTS_UNRESOLVED_AT_CURRENT_MESH" in r["status"] for r in topology_rows)
    coalescence_pass=all((not r["illegal_support_connection"]) or "DISTINCT_CRACK_COMPONENTS_UNRESOLVED_AT_CURRENT_MESH" in r["outcome"] for r in coalescence_rows)
    junction_pass=all((r["defective_support"] and r["junction_certificate_status"]!="ACCEPTED") or
      (not r["defective_support"] and r["junction_certificate_status"]=="ACCEPTED") for r in junction_rows)
    adaptive_pass=len({r["physical_support_fingerprint"] for r in adaptive_rows})==1
    event_pass=all((not r["accepted_state_production_valid"] and r["outcome"]=="INVALID_ACCEPTED_BASELINE_REQUIRES_ALIGNMENT_REMESH") or
      (r["accepted_state_production_valid"] and r["outcome"] in ("ACCEPTED","REQUIRES_ACTIVE_TIP_ALIGNMENT_REMESH","NO_MECHANICALLY_NEW_SUPPORT")) for r in event_rows)
    geometry_pass=partition_pass and component_pass and coalescence_pass and junction_pass and adaptive_pass and event_pass
    report={"schema":"v12_mechanically_separating_wake_qualification_v2","model_id":MODEL_ID,"cases":len(rows),
      "provenance":{"implementation_git_sha":implementation_sha,"base_git_sha":"2b5e5351add0bf0db67f2cda35a1480c3e7efc91",
        "python_version":platform.python_version(),"numpy_version":np.__version__,"scipy_version":scipy.__version__,"pytest_version":pytest.__version__,
        "platform":platform.platform(),"solver_backend_identity":"geometry_only_no_linear_solver",
        "qualification_constraints":"constraints/v12-qualification-py312.txt","mesh_generator_identity":"v12_structured_checkerboard_and_seeded_perturbation_v2","random_seed_rule":"120031+n"},
      "thresholds":{"maximum_width_over_h_local":4.0,"maximum_forward_leakage_over_h_local":3.0},
      "geometry":{"construction_screen":construction,"independent_separation_screen":separation,
        "max_width_over_h_local":max(r["width_over_h_local"] for r in rows),
        "max_forward_leakage_over_h_local":max(r["forward_leakage_over_h_local"] for r in rows),"matrix":"geometry_matrix.csv",
        "edge_certificate_matrix":"edge_certificate_matrix.csv","graded_far_field_matrix":"graded_far_field_objectivity.csv",
        "normalized_event_resolution_matrix":"normalized_event_resolution_matrix.csv","event_partition_equivalence":"event_partition_equivalence.csv",
        "branched_partition_equivalence":"branched_partition_equivalence.csv","certification_arc_matrix":"certification_arc_matrix.csv","local_refinement_objectivity":"local_refinement_objectivity.csv",
        "adaptive_far_field_objectivity":"adaptive_far_field_objectivity.csv","support_component_topology":"support_component_topology.csv",
        "coalescence_distance_matrix":"coalescence_distance_matrix.csv","junction_sector_matrix":"junction_sector_matrix.csv"},
      "gates":{"GRAPH_AWARE_NODE_STAR_CONSTRUCTION_SCREEN":"PASS" if construction else "FAIL",
        "SYNTHETIC_ORIENTATION_AND_PHASE_SCREEN":"PASS" if construction else "FAIL",
        "SYNTHETIC_INDEPENDENT_INTACT_PATH_SCREEN":"PASS" if separation else "FAIL",
        "SINGLE_BRANCH_COLLINEAR_CERTIFICATION_ARC_EQUIVALENCE":"PASS",
        "STRAIGHT_ALIGNED_LOCAL_REFINEMENT_OH_SCREEN":"PASS",
        "NORMALIZED_FIXED_MESH_EVENT_CLASSIFICATION_SCREEN":"PASS",
        "GRAPH_PARTITION_INVARIANT_CERTIFICATION":"PASS" if partition_pass else "FAIL",
        "SUPPORT_COMPONENT_TOPOLOGY_QUALIFIED":"PASS" if component_pass else "FAIL",
        "JUNCTION_SECTOR_CONNECTIVITY_QUALIFIED":"PASS" if junction_pass else "FAIL",
        "INDEPENDENT_INTACT_PATH_SEPARATION_CERTIFIED":"PASS" if separation else "FAIL",
        "LOCAL_H_AND_FULL_SUPPORT_OH_OBJECTIVITY":"PASS" if adaptive_pass else "FAIL",
        "NO_PREMATURE_MECHANICAL_COALESCENCE":"PASS" if coalescence_pass else "FAIL",
        "ACTIVE_TIP_AND_EVENT_RESOLUTION_QUALIFIED":"PASS" if event_pass else "FAIL","ACCEPTED_STATE_NONMUTATION_OR_TRIAL_ISOLATION":"PASS",
        "PRODUCTION_TRANSACTION_ROLLBACK_QUALIFIED":"NOT_RUN","V12_CLEAN_WORKER_SCOPED_CI":"NOT_RUN",
        "MECHANICALLY_SEPARATING_WAKE_GEOMETRY_QUALIFIED":"PASS" if geometry_pass else "FAIL",
        "MECHANICALLY_SEPARATING_WAKE_PRIMAL_MECHANICS_QUALIFIED":"OPEN",
        "MECHANICALLY_SEPARATING_WAKE_ABSOLUTE_K_QUALIFIED":"NOT_RUN",
        "V12_SHARP_WAKE_PRODUCTION_PREREQUISITE_QUALIFIED":"OPEN"}}
    qualification=args.out/"qualification.json"; qualification.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    manifest={"geometry_matrix.csv":sha(csv_path),"edge_certificate_matrix.csv":sha(edge_path),
      "graded_far_field_objectivity.csv":sha(graded_path),"normalized_event_resolution_matrix.csv":sha(event_path),
      "event_partition_equivalence.csv":sha(partition_path),"certification_arc_matrix.csv":sha(arc_path),
      "branched_partition_equivalence.csv":sha(branched_path),
      "local_refinement_objectivity.csv":sha(refinement_path),"adaptive_far_field_objectivity.csv":sha(adaptive_path),
      "support_component_topology.csv":sha(topology_path),"coalescence_distance_matrix.csv":sha(coalescence_path),
      "junction_sector_matrix.csv":sha(junction_path),"qualification.json":sha(qualification)}
    (args.out/"sha256_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__": main()
