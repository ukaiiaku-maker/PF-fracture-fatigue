#!/usr/bin/env python3
"""Constrained complete-cycle search for the opt-in three-scale compact tail."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.cohesive_arrhenius import COMPACT_INDEPENDENT,EV_TO_J
from arrhenius_fracture.finite_compliance_inverse_shape_r10 import (
    K_from_imposed,branch_loss,cyclic_rate_curve,metastable_window,shape_coordinates)
from scripts.run_r10_finite_compliance_inverse_shape import _source_material_class,_thermal_class
from scripts.run_r10_global_mechanism_search import canonical_hash
from scripts.run_r10_opening_alternatives import mechanics,thermal_from

SCHEMA="r10.three-scale-complete-cycle-shape-search/1"


def logmap(u,lo,hi):return math.exp(math.log(lo)+float(u)*math.log(hi/lo))


def write_csv(path:Path,rows:list[dict],compressed=False):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    opener=gzip.open if compressed else open
    with opener(path,"wt",newline="") as stream:
        writer=csv.DictWriter(stream,fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)


def atomic(path:Path,payload:object):
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n");tmp.replace(path)


def candidate(index:int,u:np.ndarray,sources:list[dict])->dict:
    source=sources[min(int(u[0]*len(sources)),len(sources)-1)]
    gamma=logmap(u[1],.5,8.);peak=logmap(u[2],8e9,80e9)
    zero_eV=logmap(u[3],.25,12.);area=zero_eV*EV_TO_J/gamma
    entropy=(-.25+u[4])*gamma/900.
    rise=logmap(u[5],.0005,.05)
    drop=rise+(1.-rise)*logmap(u[6],.001,.08)
    knee=drop+(1.-drop)*(.03+.87*u[7])
    middle=logmap(u[8],.03,.6)
    tail=middle*logmap(u[9],.03,.75)
    cohesive={"Gamma_ref_J_m2":gamma,"t_hat_ref_Pa":peak,
        "S_Gamma_J_m2_K":entropy,"T_ref_K":300.,"activation_area_m2":area,
        "attempt_frequency_s":1e12,"delta0_m":gamma/(math.e*peak),
        "zero_traction_barrier_eV_300K":zero_eV,
        "thermal_subfamily":"MINIMAL_COUPLED_COHESIVE_ENTROPY",
        "compact_b":2.,"compact_c":0.,"compact_variant":"three_scale_tail",
        "compact_rise_fraction":rise,"compact_drop_fraction":drop,
        "compact_knee_fraction":knee,"compact_mid_level":middle,
        "compact_tail_level":tail,"compact_fall_fraction":.85,
        "compact_plateau_drop":.05}
    return {"candidate_id":f"R10_CONT3_{index:05d}","source":source,"cohesive":cohesive}


def evaluate(item:dict)->dict:
    c=item["cohesive"];base={"candidate_id":item["candidate_id"],
        "source_material_candidate_id":item["source"]["candidate_id"],
        "source_material_class":_source_material_class(item["source"]),
        "source_plasticity_sha256":item["source"].get("plasticity_physics_sha256") or
            item["source"].get("parent_complete_row_sha256")}
    try:
        thermal=thermal_from(c);thermal.validate_range((300.,600.,900.,1200.))
        _,kenv=mechanics(thermal,300.);window=metastable_window(
            COMPACT_INDEPENDENT,thermal,300.,kenv)
        if window["state"]!="METASTABLE_WINDOW":
            return {**base,"domain_pass":False,"rejection":"NO_METASTABLE_WINDOW"}
        birth=K_from_imposed(float(window["birth_load_Pa"])/kenv,kenv)
        fold=K_from_imposed(float(window["fold_load_Pa"])/kenv,kenv)
        if not (birth>0. and fold>birth*1.02):
            return {**base,"domain_pass":False,"rejection":"INSUFFICIENT_CONNECTED_LOAD_WINDOW"}
        loads=np.geomspace(1.005*birth,.985*fold,25)
        curve=cyclic_rate_curve(loads,kenv,thermal,300.,COMPACT_INDEPENDENT,
            c["activation_area_m2"],c["attempt_frequency_s"],R=.1,
            frequency_Hz=1000.,phases=20)
        supported=[row for row in curve["rows"] if row["local_log_slope"] is not None
            and row["opening_action_per_cycle"]>0.]
        if not supported:return {**base,"domain_pass":False,"rejection":"NO_TWO_SIDED_CYCLE_POINT"}
        target=min(supported,key=lambda row:abs(math.log10(row["opening_action_per_cycle"])+4.22))
        slopes=np.asarray([row["local_log_slope"] for row in supported],float)
        actions=np.asarray([row["opening_action_per_cycle"] for row in supported],float)
        curv=np.asarray([row["local_log_curvature"] for row in supported
            if row["local_log_curvature"] is not None],float)
        folds=[]
        for T in (300.,600.,900.,1200.):
            _,kT=mechanics(thermal,T);loss=branch_loss(COMPACT_INDEPENDENT,thermal,T,kT)
            folds.append(None if loss.get("fold_imposed_m") is None else
                K_from_imposed(float(loss["fold_imposed_m"]),kT))
        if any(value is None for value in folds):
            thermal_class="THERMAL_ACCESS_CENSORED"
        else:thermal_class=_thermal_class([float(value) for value in folds])
        coords=shape_coordinates(COMPACT_INDEPENDENT,thermal,300.)
        slope=float(target["local_log_slope"]);action=float(target["opening_action_per_cycle"])
        strict=float(curve["strict_connected_width_log10rate"])
        moderate=float(curve["moderate_connected_width_log10rate"])
        objective=(abs(slope-4.)+.6*abs(math.log10(action)+4.22)-4.*strict-moderate+
            .03*(float(np.median(np.abs(curv))) if len(curv) else 100.))
        return {**base,"domain_pass":True,"objective":objective,
            "target_K_MPa_sqrt_m":target["Kmax_MPa_sqrt_m"],
            "target_action_per_cycle":action,"target_rate_m_per_cycle":1e-6*action,
            "target_local_slope":slope,
            "target_accessibility_log_slope":target["accessibility_log_slope"],
            "target_conditional_rate_log_slope":target["conditional_rate_log_slope"],
            "target_accessible_phase_fraction":target["accessible_phase_fraction"],
            "strict_connected_width_log10rate":strict,
            "moderate_connected_width_log10rate":moderate,
            "minimum_local_slope":float(np.min(slopes)),
            "maximum_local_slope":float(np.max(slopes)),
            "median_abs_curvature":float(np.median(np.abs(curv))) if len(curv) else None,
            "birth_K_300K":birth,"fold_K_300K":fold,"metastable_load_ratio":window["load_ratio"],
            "fold_K_600K":folds[1],"fold_K_900K":folds[2],"fold_K_1200K":folds[3],
            "cohesive_fold_thermal_prediction":thermal_class,
            "initial_stiffness_Pa_per_m":coords["initial_stiffness_Pa_per_m"],
            "completion_cutoff_m":coords["completion_cutoff_m"],
            "zero_load_energy_eV":c["activation_area_m2"]*c["Gamma_ref_J_m2"]/EV_TO_J,
            "rise_bound_active":c["compact_rise_fraction"]<=.00055,
            "rapid_drop_bound_active":((c["compact_drop_fraction"]-c["compact_rise_fraction"])/
                (1.-c["compact_rise_fraction"]))<=.0011,
            "zero_energy_bound_active":c["zero_traction_barrier_eV_300K"]>=11.9,
            "cohesive_json":json.dumps(c,sort_keys=True,separators=(",",":")),
            "curve_json":json.dumps(curve["rows"],sort_keys=True,separators=(",",":"))}
    except (ValueError,OverflowError,FloatingPointError) as exc:
        return {**base,"domain_pass":False,"rejection":f"{type(exc).__name__}:{exc}"}


def projected(row:dict,source:dict)->dict:
    c=json.loads(row["cohesive_json"]);out=dict(source)
    out.update({"candidate_id":row["candidate_id"]+"__CM",
        "source_material_candidate_id":row["source_material_candidate_id"],
        "selected_shape":COMPACT_INDEPENDENT,"selected_mode":"cohesive_mixed",
        "search_design":"THREE_SCALE_COMPLETE_CYCLE_CONSTRAINED_SEARCH",
        "research_status":"OPT_IN_RESEARCH_CANDIDATE",
        "cohesive_parameter_vector":json.dumps(c,sort_keys=True,separators=(",",":")),
        "predicted_fold_K_300K":row["fold_K_300K"],"predicted_Kaccess_300K":row["fold_K_300K"],
        "predicted_hcf_K_300K":row["target_K_MPa_sqrt_m"],
        "predicted_hcf_cyclic_action_per_cycle":row["target_action_per_cycle"],
        "predicted_hcf_local_slope":row["target_local_slope"],
        "predicted_hcf_accessible_phase_fraction":row["target_accessible_phase_fraction"],
        "predicted_hcf_selection_basis":"COMPLETE_CYCLE_GAUSS_MOVING_BOUNDARY_TWO_SIDED",
        "cohesive_fold_thermal_prediction":row["cohesive_fold_thermal_prediction"],
        "source_material_class":row["source_material_class"],
        "mechanism_prediction":source.get("mechanism_prediction","STATE_MEDIATED_LANE")})
    out["cohesive_parameter_sha256"]=canonical_hash(c)
    out["complete_row_sha256"]=canonical_hash({"source":source.get("complete_row_sha256"),
        "cohesive":c,"shape":COMPACT_INDEPENDENT,"mode":"cohesive_mixed","family":SCHEMA})
    return out


def main(argv=None)->int:
    p=argparse.ArgumentParser();p.add_argument("--source-registry",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--log2-bank",type=int,default=11)
    p.add_argument("--select",type=int,default=128);p.add_argument("--workers",type=int,default=8)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    with a.source_registry.open(newline="") as stream:sources=list(csv.DictReader(stream))
    points=qmc.Sobol(d=10,scramble=True,seed=20261019).random_base2(a.log2_bank)
    items=[candidate(i,u,sources) for i,u in enumerate(points)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(evaluate,items,chunksize=4))
    valid=sorted([row for row in rows if row.get("domain_pass")],key=lambda row:(row["objective"],row["candidate_id"]))
    # Preserve parent class and plasticity diversity before objective fill.
    chosen=[];seen=set()
    for diversity in ("source_material_class","source_plasticity_sha256",None):
        for row in valid:
            if row in chosen:continue
            key=row.get(diversity) if diversity else row["candidate_id"]
            if diversity and key in seen:continue
            chosen.append(row);seen.add(key)
            if len(chosen)>=a.select:break
        if len(chosen)>=a.select:break
        seen=set()
    source_by_id={row["candidate_id"]:row for row in sources}
    registry=[projected(row,source_by_id[row["source_material_candidate_id"]]) for row in chosen]
    write_csv(a.out/"analytical_bank.csv.gz",rows,True)
    write_csv(a.out/"selected_metrics.csv",chosen)
    write_csv(a.out/"state_informed_registry.csv",registry)
    summary={"schema":SCHEMA,"stage":"THREE_SCALE_ANALYTICAL_SEARCH_COMPLETE",
        "allocated_vectors":len(rows),"domain_valid":len(valid),"selected_complete_vectors":len(registry),
        "unique_complete_hashes":len({row["complete_row_sha256"] for row in registry}),
        "unique_parent_plasticity":len({row.get("plasticity_physics_sha256") for row in registry}),
        "strict_passes":sum(float(row["strict_connected_width_log10rate"])>=2. for row in valid),
        "best_candidate":valid[0]["candidate_id"] if valid else None,
        "best_objective":valid[0]["objective"] if valid else None,
        "source_registry":str(a.source_registry),"source_registry_sha256":hashlib.sha256(a.source_registry.read_bytes()).hexdigest(),
        "algebraic_tail_registered":False,"canonical_promotion":False}
    atomic(a.out/"summary.json",summary);print(json.dumps(summary,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
