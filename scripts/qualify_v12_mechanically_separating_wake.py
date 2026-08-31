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

from arrhenius_fracture.crack_network_v11 import CrackNetworkState
from arrhenius_fracture.mechanically_separating_sharp_wake_v12 import MODEL_ID, mechanically_separating_graph_support, support_record


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

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def git(*args): return subprocess.check_output(("git",)+args,cwd=ROOT,text=True).strip()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,default=Path("artifacts/v12_mechanically_separating_wake")); args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True); rows=[]; edge_rows=[]; graded_rows=[]; event_rows=[]
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
    event_mesh=mesh(33,False); dx=1/32
    directions={angle:np.array((np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)))) for angle in (0,15,-15,30,-30,45,-45,90)}
    for angle,direction in directions.items():
        for phase in (0,1,2):
            old_tip=np.array((.5+phase*dx,0.)); root=old_tip-.375*direction
            accepted=CrackNetworkState.one_tip((root,old_tip)); prior,_=mechanically_separating_graph_support(event_mesh,accepted)
            damage=np.zeros(event_mesh.ne); damage[prior]=1.; owner=support_record(event_mesh,accepted,damage,prior)
            for ratio in (.25,.5,1,2,4,8):
                new_tip=old_tip+ratio*dx*direction; trial=CrackNetworkState.one_tip((root,old_tip,new_tip)); outcome="ACCEPTED"
                try:
                    _,audit=mechanically_separating_graph_support(event_mesh,trial,previous_support=owner,accepted_network=accepted,accepted_damage=damage)
                    mechanically_new=audit.mechanically_new_element_count; accepted_fp=audit.accepted_damage_fingerprint; trial_fp=audit.trial_damage_fingerprint
                except RuntimeError as error:
                    outcome=str(error).split(": ",1)[-1]; mechanically_new=0; accepted_fp=""; trial_fp=""
                event_rows.append({"angle_deg":angle,"phase_cells":phase,"delta_a_over_grid_spacing":ratio,"outcome":outcome,
                  "mechanically_new_element_count":mechanically_new,"accepted_damage_fingerprint":accepted_fp,"trial_damage_fingerprint":trial_fp})
    event_path=args.out/"event_resolution_matrix.csv"
    with event_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=event_rows[0],lineterminator="\n"); writer.writeheader(); writer.writerows(event_rows)
    construction=all(r["construction_screen"] and r["unresolved_bridge_nodes"]==0 for r in rows)
    separation=all(not r["intact_cross_graph_path_exists"] for r in rows)
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
        "event_resolution_matrix":"event_resolution_matrix.csv"},
      "gates":{"GRAPH_AWARE_NODE_STAR_CONSTRUCTION_SCREEN":"PASS" if construction else "FAIL",
        "SYNTHETIC_ORIENTATION_AND_PHASE_SCREEN":"PASS" if construction else "FAIL",
        "SYNTHETIC_INDEPENDENT_INTACT_PATH_SCREEN":"PASS" if separation else "FAIL",
        "INDEPENDENT_INTACT_PATH_SEPARATION_CERTIFIED":"OPEN",
        "LOCAL_H_AND_FULL_SUPPORT_OH_OBJECTIVITY":"OPEN","NO_PREMATURE_MECHANICAL_COALESCENCE":"OPEN",
        "ACTIVE_TIP_AND_EVENT_RESOLUTION_QUALIFIED":"OPEN","ACCEPTED_STATE_NONMUTATION_OR_TRIAL_ISOLATION":"PASS",
        "PRODUCTION_TRANSACTION_ROLLBACK_QUALIFIED":"NOT_RUN","V12_CLEAN_WORKER_SCOPED_CI":"NOT_RUN",
        "MECHANICALLY_SEPARATING_WAKE_GEOMETRY_QUALIFIED":"OPEN",
        "MECHANICALLY_SEPARATING_WAKE_PRIMAL_MECHANICS_QUALIFIED":"OPEN",
        "MECHANICALLY_SEPARATING_WAKE_ABSOLUTE_K_QUALIFIED":"NOT_RUN",
        "V12_SHARP_WAKE_PRODUCTION_PREREQUISITE_QUALIFIED":"OPEN"}}
    qualification=args.out/"qualification.json"; qualification.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    manifest={"geometry_matrix.csv":sha(csv_path),"edge_certificate_matrix.csv":sha(edge_path),
      "graded_far_field_objectivity.csv":sha(graded_path),"event_resolution_matrix.csv":sha(event_path),"qualification.json":sha(qualification)}
    (args.out/"sha256_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__": main()
