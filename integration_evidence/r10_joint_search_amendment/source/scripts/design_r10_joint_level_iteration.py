#!/usr/bin/env python3
"""Joint R10 iteration: strict-family level shifts scored for fatigue and fracture.

The update changes only two existing EXP-floor fields.  At 300 K it raises
both the zero-stress barrier and its absolute floor by ``delta_G``; therefore
the complete 300 K surface is shifted by a constant without adding a basis or
rate multiplier.  Other temperatures are evaluated, not claimed invariant.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import gammainc

from arrhenius_fracture.material_manifest import ExpFloorBarrier


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def barrier(row):
    return ExpFloorBarrier(
        G00_eV=float(row["cleave_G00_eV"]),gT_eV_per_K=float(row["cleave_gT_eV_per_K"]),
        sigc0_Pa=1e9*float(row["cleave_sigc0_GPa"]),sT_Pa_per_K=1e9*float(row["cleave_sT_GPa_per_K"]),
        alpha=float(row["cleave_exp_a"]),exponent=float(row["cleave_exp_n"]),
        floor_fraction=float(row["cleave_floor_frac"]),Tref_K=float(row["Tref_K"]),
        attempt_frequency_s=1e12)


def effective_rate(surface,sigma,T,q,tau):
    raw=np.asarray(surface.rate(sigma,T),float)
    return gammainc(max(q,1.),np.minimum(raw*tau,1e12))/tau if q>1+1e-12 else raw


def stress_from_K(K):
    return np.minimum(np.asarray(K,float)*1e6/math.sqrt(2*math.pi*1e-6),30e9)


def action_curve(surface,q,tau,Kmax,T=300.,R=.1,frequency=1000.,nphase=2048):
    phase=(np.arange(nphase)+.5)*2*math.pi/nphase
    K=Kmax[:,None]*(.5*(1+R)+.5*(1-R)*np.cos(phase))[None,:]
    return np.mean(effective_rate(surface,stress_from_K(K),T,q,tau),axis=1)/frequency


def frozen_onset(surface,q,tau,T,Kdot=.005,Kmax=80.,dK=.002):
    K=np.arange(0.,Kmax+dK,dK); rate=effective_rate(surface,stress_from_K(K),T,q,tau)
    action=np.concatenate([[0.],np.cumsum(.5*(rate[:-1]+rate[1:])*dK/Kdot)])
    hit=np.flatnonzero(action>=1.)
    return None if not len(hit) else float(K[hit[0]])


def moderate_band(K,action,lo=3.4,hi=4.6):
    ok=action>0; K=K[ok];action=action[ok]
    slopes=np.diff(np.log(action))/np.diff(np.log(K)); widths=np.diff(np.log10(action))
    best=(0.,None,None)
    for i,m in enumerate(slopes):
        if not lo<=m<=hi:continue
        width=0.
        for j in range(i,len(slopes)):
            if not lo<=slopes[j]<=hi:break
            width+=widths[j]
            if width>best[0]:best=(float(width),float(K[i]),float(K[j+1]))
    return best,float(np.min(slopes)),float(np.max(slopes))


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--frontier",type=Path,required=True)
    p.add_argument("--out-root",type=Path,required=True);a=p.parse_args(argv);a.out_root.mkdir(parents=True,exist_ok=True)
    base=[r for r in csv.DictReader(a.frontier.open()) if r["target"]=="P40"]
    scan=[];selected=[];Kfat=np.geomspace(3.,30.,41)
    for rec in base:
        original=json.loads(rec["complete_parameter_vector"]);s0=barrier(original)
        G300=float(s0.G00_eV+s0.gT_eV_per_K*(300-s0.Tref_K));floor300=s0.floor_fraction*G300
        class_name=rec["candidate_id"].split("_")[1]
        rows=[]
        for delta in np.linspace(0.,.16,81):
            row=dict(original);row["cleave_G00_eV"]=float(s0.G00_eV+delta)
            row["cleave_floor_frac"]=float((floor300+delta)/(G300+delta));s=barrier(row)
            q=float(row["physics__cleavage_hits"]);tau=float(row["physics__cleavage_correlation_time_s"])
            action=action_curve(s,q,tau,Kfat);band_info,smin,smax=moderate_band(.9*Kfat,action)
            band,band_low,band_high=band_info
            onsets={str(T):frozen_onset(s,q,tau,float(T)) for T in (300,450,600,750,900,1050,1200)}
            k300=onsets["300"]
            penalty=(0. if k300 is not None and 5<=k300<=30 else
                     5-k300 if k300 is not None and k300<5 else
                     k300-30 if k300 is not None else 50.)
            item={"base_candidate_id":rec["candidate_id"],"class":class_name,"delta_G_eV":float(delta),
                  "cleave_G00_eV":row["cleave_G00_eV"],"cleave_floor_frac":row["cleave_floor_frac"],
                  "frozen_Kinit_300":k300,"frozen_Kinit_by_T_json":json.dumps(onsets,sort_keys=True),
                  "A0_band_width_decades":band,"A0_band_DeltaK_low":band_low,
                  "A0_band_DeltaK_high":band_high,
                  "A0_local_slope_min":smin,"A0_local_slope_max":smax,
                  "A0_action_min":float(action.min()),"A0_action_max":float(action.max()),
                  "fracture_scale_penalty":float(penalty),"complete_parameter_vector":row}
            rows.append(item);scan.append({k:v for k,v in item.items() if k!="complete_parameter_vector"})
        feasible=[x for x in rows if x["fracture_scale_penalty"]==0 and x["A0_band_width_decades"]>=2]
        pool=feasible or rows
        # Lowest allowed shift that clears the fracture gate protects rate accessibility.
        choice=min(pool,key=lambda x:(x["fracture_scale_penalty"],x["delta_G_eV"]))
        child=dict(choice["complete_parameter_vector"])
        cid=f"R10J_{class_name}_P40_LSHIFT_{int(round(1000*choice['delta_G_eV'])):03d}MEV_V1"
        child["candidate_id"]=cid
        changed={k:{"parent":original[k],"offspring":child[k]} for k in ("cleave_G00_eV","cleave_floor_frac")}
        selected.append({
            "candidate_id":cid,"parent_candidate_id":rec["candidate_id"],"family":rec["family"],"target":"P40",
            "cleavage_hits":child["physics__cleavage_hits"],"cleavage_tau_s":child["physics__cleavage_correlation_time_s"],
            "complete_row_sha256":canonical_hash(child),"changed_fields_json":json.dumps(changed,sort_keys=True),
            "complete_parameter_vector":json.dumps(child,sort_keys=True),"selection_metrics_json":json.dumps({k:v for k,v in choice.items() if k!="complete_parameter_vector"},sort_keys=True),
            "realization":"existing EXP_floor G00 and floor_fraction; exact constant +delta_G surface shift at 300 K only",
        })
    fields=list(scan[0]);
    with (a.out_root/"joint_level_scan.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(scan)
    fields=list(selected[0]);
    with (a.out_root/"joint_level_candidates.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(selected)
    decision={"schema":"r10.joint-level-iteration/1","source":{"path":str(a.frontier.resolve()),"sha256":hashlib.sha256(a.frontier.read_bytes()).hexdigest()},
      "selection_rule":"lowest 300 K EXP-floor level shift with frozen 5-30 scale and >=2-decade moderate A0 band",
      "bounds_provenance":{"floor_fraction_max":{"value":.95,"kind":"constitutive-domain limit","source":"ExpFloorBarrier.floor_max_fraction"},
        "projection_exponent_min":{"value":.25,"kind":"inverse-helper default/design restriction","source":"ExpFloorBounds default and R10 projection call; not a universal physical limit"},
        "inherited_fracture_gate":{"value":"5-30 MPa sqrt(m) at low temperature","kind":"inherited campaign selection gate; not universal impossibility test"}},
      "selected":[{k:v for k,v in x.items() if k!="complete_parameter_vector"} for x in selected],
      "limitations":["frozen ramp is a prospective score, not complete-state qualification","A0 is not used as a hard rejection for state-mediated rows","thermal surfaces outside 300 K are changed and explicitly evaluated"]}
    (a.out_root/"joint_level_decision.json").write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n")
    print(json.dumps(decision,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
