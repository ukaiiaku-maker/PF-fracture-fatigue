#!/usr/bin/env python3
"""Targeted alternative-G* joint search on the accepted R9 material lineage.

The historical banks are read-only.  New rows perturb stress curvature and a
thermodynamic entropy/heat-capacity extension together, while retaining the
same mechanics, emission/plasticity surface, event law, and geometry.
"""
from __future__ import annotations

import argparse,csv,hashlib,json,math
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import gammainc
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.material_manifest import KB_EV_PER_K
from scripts.fatigue_informed_joint_barrier_v3_v10230 import (
    KineticSurface,SmoothBroadBasis,deserialize_surface,serialize_surface,
)
from scripts.thermodynamic_joint_barrier_v10230 import Basis

TEMPS=np.array([300.,450.,600.,750.,900.,1050.,1200.])
LOADS=np.array([4.781762498950186,8.598738351065925,15.462474735549586,27.809,50.])
R=.1;FREQ=1000.;KDOT=.005;R0=1e-6;THRESHOLD=1.;SEED=20261020


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def write_csv(path,rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fields,lineterminator="\n");w.writeheader();w.writerows(rows)


def stress_from_K(K):return np.asarray(K)*1e6/math.sqrt(2*math.pi*R0)


def cooperative(surface,stress,T):
    raw=surface.raw_rate_s(stress,T)
    return gammainc(surface.hits,np.minimum(raw*surface.tau_s,1e12))/surface.tau_s


def fatigue_metrics(surface):
    phase=2*math.pi*(np.arange(256)+.5)/256
    h=.5*(1+R)+.5*(1-R)*np.cos(phase)
    actions=np.array([np.mean(cooperative(surface,stress_from_K(K*h),300.))/FREQ for K in LOADS])
    slopes=np.diff(np.log(actions))/np.diff(np.log(.9*LOADS))
    return actions,slopes,math.log10(actions[-1]/actions[0])


def root(surface,T,points=1400):
    K=np.linspace(0.,80.,points+1);rate=cooperative(surface,stress_from_K(K),T)
    action=cumulative_trapezoid(rate,K,initial=0.)/KDOT
    if action[-1]<THRESHOLD:return math.nan
    j=int(np.searchsorted(action,THRESHOLD))
    return float(np.interp(THRESHOLD,action[j-1:j+1],K[j-1:j+1]))


def topology(roots):
    x=np.asarray(roots,float)
    if not np.all(np.isfinite(x)) or np.any(x<=0):return "CENSORED"
    span=float(x.max()/x.min());d=np.diff(x);imax=int(np.argmax(x))
    if span<=1.25:return "WEAK_T"
    if 0<imax<len(x)-1 and x[imax]>=1.15*max(x[0],x[-1]) and np.all(d[:imax]>0) and np.all(d[imax:]<0):return "PEAK_T"
    if x[-1]/x[0]>=1.5 and np.all(d>=0):return "DBTT"
    if x[-1]/x[0]<=.70 and np.all(d<=0):return "CERAMIC_LIKE"
    return "OTHER"


def make_variant(base,u):
    broad,guard=base.bases[:2]
    new_broad=SmoothBroadBasis(broad.sigma_b_Pa*(.8+.4*u[0]),broad.exponent*(.8+.4*u[1]),broad.smoothing_ratio)
    anchors=stress_from_K(np.array([0.,15.,75.]))
    matrix=np.column_stack([np.ones(3),new_broad.value(anchors),guard.value(anchors)])
    coeff=np.linalg.solve(matrix,np.asarray(base.reference_G_eV(anchors),float))
    entropy_coeff=np.linalg.solve(matrix,np.asarray(base.entropy_kB(anchors,300.),float))
    thermal_basis=Basis(1.,float(base.bases[1].sigma_Pa)*2.2,2.)
    theta_const=-25.+50.*u[2];theta_shape=-30.+60.*u[3]
    cp_const=-80.+160.*u[4];cp_shape=-100.+200.*u[5]
    return KineticSurface(process=base.process,attempt_frequency_s=base.attempt_frequency_s,
      reference_floor_eV=float(coeff[0]),reference_amplitudes_eV=(float(coeff[1]),float(coeff[2]),0.),
      bases=(new_broad,guard,thermal_basis),entropy_infinity_kB=float(entropy_coeff[0]+theta_const),
      entropy_amplitudes_kB=(float(entropy_coeff[1]),float(entropy_coeff[2]),float(theta_shape)),
      heat_capacity_infinity_kB=float(cp_const),heat_capacity_amplitudes_kB=(0.,0.,float(cp_shape)),
      hits=base.hits,tau_s=base.tau_s)


def physical(surface):
    stress=np.r_[0.,np.geomspace(1e6,30e9,192)]
    for T in TEMPS:
        G=np.asarray(surface.G_eV(stress,T),float)
        if not np.all(np.isfinite(G)) or np.min(G)<=1e-5 or np.max(np.diff(G))>2e-6:return False
    return True


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--base-registry",type=Path,required=True)
    p.add_argument("--parent-registry",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--log2",type=int,default=12);a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    base_payload=json.loads(a.base_registry.read_text());base_row=next(r for r in base_payload["rows"] if r["candidate_id"]=="V3_ceramic_V2_P25_007__V3_1_TRUST1")
    base=deserialize_surface(base_row["opening_surface"]);points=qmc.Sobol(6,scramble=True,seed=SEED).random_base2(a.log2)
    records=[];surfaces={}
    for i,u in enumerate(points):
        try:surface=make_variant(base,u)
        except (ValueError,np.linalg.LinAlgError):continue
        if not physical(surface):continue
        actions,slopes,width=fatigue_metrics(surface)
        fatigue_pass=bool(np.all((slopes>=3.4)&(slopes<=4.6)) and width>=2.)
        if not fatigue_pass:continue
        roots=[root(surface,T) for T in TEMPS];role=topology(roots)
        cid=f"R10_ALT_GSTAR_{i:05d}"
        records.append({"candidate_id":cid,"role":role,"fatigue_pass":fatigue_pass,
          "fatigue_rate_width_decades":width,"fatigue_slope_min":float(slopes.min()),"fatigue_slope_max":float(slopes.max()),
          "roots_json":json.dumps(roots,separators=(",",":")),"root_span":None if not np.all(np.isfinite(roots)) else max(roots)/min(roots),
          "coordinates_json":json.dumps(u.tolist(),separators=(",",":")),
          "opening_surface_sha256":canonical(serialize_surface(surface))})
        surfaces[cid]=surface
    selected=[]
    for role in ("WEAK_T","PEAK_T","DBTT","CERAMIC_LIKE"):
        pool=[r for r in records if r["role"]==role]
        if not pool:continue
        bounded=[r for r in pool if min(json.loads(r["roots_json"]))>=2. and max(json.loads(r["roots_json"]))<=40.]
        if role=="CERAMIC_LIKE":
            bounded=[r for r in pool if min(json.loads(r["roots_json"]))>=1. and max(json.loads(r["roots_json"]))<=40.]
        if bounded:pool=bounded
        if role=="PEAK_T":pool.sort(key=lambda r:abs(max(json.loads(r["roots_json"]))/max(json.loads(r["roots_json"])[0],json.loads(r["roots_json"])[-1])-2.))
        elif role=="DBTT":pool.sort(key=lambda r:abs(json.loads(r["roots_json"])[-1]/json.loads(r["roots_json"])[0]-3.))
        elif role=="CERAMIC_LIKE":pool.sort(key=lambda r:abs(json.loads(r["roots_json"])[-1]/json.loads(r["roots_json"])[0]-.5))
        else:pool.sort(key=lambda r:abs(r["root_span"]-1.1))
        selected.append(pool[0])
    rows=[]
    for r in selected:
        surface=surfaces[r["candidate_id"]];raw=serialize_surface(surface)
        row={"candidate_id":r["candidate_id"],"candidate_sha256":"","parent_candidate_id":base_row["parent_candidate_id"],
          "opening_surface":raw,"emission_surface":base_row["emission_surface"],"family":"C_ALTERNATIVE_GSTAR_JOINT",
          "role":r["role"],"reference_300K_barrier_anchor_preserved":True,"mechanics_event_geometry_changed":False}
        row["candidate_sha256"]=canonical(row);rows.append(row);r["candidate_sha256"]=row["candidate_sha256"]
    payload={"schema":"r10.alternative-gstar-joint-registry/1","family":"C_ALTERNATIVE_GSTAR_JOINT",
      "historical_bank_regenerated":False,"source_registry":str(a.base_registry.resolve()),"source_registry_sha256":hashlib.sha256(a.base_registry.read_bytes()).hexdigest(),
      "parent_registry":str(a.parent_registry.resolve()),"allocated_new_vectors":len(points),"qualified_new_vectors":len(records),"rows":rows}
    (a.out/"alternative_gstar_registry.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    write_csv(a.out/"analytical_qualified.csv",records);write_csv(a.out/"selected_roles.csv",selected)
    summary={"schema":payload["schema"],"allocated":len(points),"qualified":len(records),
      "role_counts":{role:sum(r["role"]==role for r in records) for role in ("WEAK_T","PEAK_T","DBTT","CERAMIC_LIKE","OTHER","CENSORED")},
      "selected":[{k:r[k] for k in ("candidate_id","candidate_sha256","role","roots_json","root_span","fatigue_rate_width_decades","fatigue_slope_min","fatigue_slope_max")} for r in selected]}
    (a.out/"analytical_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");print(json.dumps(summary,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
