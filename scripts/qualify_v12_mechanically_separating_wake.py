#!/usr/bin/env python3
"""Write deterministic V12 geometry evidence and conservative gate states."""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from arrhenius_fracture.crack_network_v11 import CrackNetworkState
from arrhenius_fracture.mechanically_separating_sharp_wake_v12 import MODEL_ID, mechanically_separating_graph_support


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


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,default=Path("artifacts/v12_mechanically_separating_wake")); args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True); rows=[]
    for n in (9,17,33,65):
        for perturbed in (False,True):
            m=mesh(n,perturbed)
            for angle in (0,15,-15,30,-30,45,-45):
                for phase in (0.,.037):
                    p0=np.array((.2+phase,0.)); rad=np.deg2rad(angle); p1=p0+.5*np.array((np.cos(rad),np.sin(rad)))
                    _,a=mechanically_separating_graph_support(m,CrackNetworkState.one_tip((p0,p1)))
                    rows.append(dict(n=n,mesh="perturbed" if perturbed else "structured",angle_deg=angle,phase=phase,
                        h_m=a.mesh_size_m,area_m2=a.selected_area_m2,area_over_length_h=a.selected_area_m2/(a.segment_partition_invariant_length_m*a.mesh_size_m),
                        width_over_h=a.width_over_h,forward_leakage_over_h=a.forward_leakage_over_h,unresolved_bridge_nodes=len(a.unresolved_bridge_node_ids),certified=a.certified))
    csv_path=args.out/"geometry_matrix.csv"
    with csv_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    all_certified=all(r["certified"] and r["unresolved_bridge_nodes"]==0 for r in rows)
    report={"schema":"v12_mechanically_separating_wake_qualification_v1","model_id":MODEL_ID,"cases":len(rows),
      "geometry":{"all_certified":all_certified,"max_width_over_h":max(r["width_over_h"] for r in rows),
        "max_forward_leakage_over_h":max(r["forward_leakage_over_h"] for r in rows),"matrix":str(csv_path)},
      "gates":{"MECHANICALLY_SEPARATING_WAKE_GEOMETRY_QUALIFIED":"PASS" if all_certified else "FAIL",
        "MECHANICALLY_SEPARATING_WAKE_PRIMAL_MECHANICS_QUALIFIED":"OPEN",
        "MECHANICALLY_SEPARATING_WAKE_ABSOLUTE_K_QUALIFIED":"NOT_RUN",
        "V12_SHARP_WAKE_PRODUCTION_PREREQUISITE_QUALIFIED":"OPEN"}}
    (args.out/"qualification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__": main()
