#!/usr/bin/env python3
"""Mechanical-environment and analytical/native-reference audit for R10.

This is an executed diagnostic, not a replacement mechanics model.  It keeps
the declared process-zone closure as the control, measures bulk-only Schur
responses for explicit FE fixtures, and resolves the complete-cycle opening
action with both production quadrature and the native midpoint convention.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.stats import chi2

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.fem import assemble_mechanics,plane_strain_D
from arrhenius_fracture.finite_compliance_inverse_shape_r10 import cyclic_rate,cyclic_rate_curve
from arrhenius_fracture.mesh import rebuild_tri_mesh
from scripts.run_r10_opening_alternatives import mechanics,thermal_from


def atomic(path:Path,payload:object)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n")
    tmp.replace(path)


def sha256(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registry_row(path:Path,candidate:str)->dict:
    with path.open(newline="") as stream:
        return next(row for row in csv.DictReader(stream) if row["candidate_id"]==candidate)


def bulk_schur_stiffness(width_m:float,half_height_m:float)->dict:
    """Assembled bulk derivative dt/d(Delta-delta), excluding cohesive tangent."""
    W=float(width_m);H=float(half_height_m);jump=1e-9
    nodes=np.array([[0,0],[W,0],[0,H],[W,H],
                    [0,0],[W,0],[0,-H],[W,-H]],float)
    elems=np.array([[0,1,2],[1,3,2],[4,6,5],[5,6,7]],int)
    mesh=rebuild_tri_mesh(nodes,elems);mat=ElasticProperties();D=plane_strain_D(mat)
    u=np.zeros(mesh.ndof)
    prescribed={2*2+1:jump/2,2*3+1:jump/2,2*6+1:-jump/2,2*7+1:-jump/2,
                2*0+1:0.,2*1+1:0.,2*4+1:0.,2*5+1:0.,0:0.,2*4:0.}
    fixed=np.asarray(sorted(prescribed),int);u[fixed]=[prescribed[int(i)] for i in fixed]
    free=np.setdiff1d(np.arange(mesh.ndof),fixed)
    K,R,*_=assemble_mechanics(mesh,u,np.zeros((3,mesh.ne)),np.zeros(mesh.ne),
        np.zeros(mesh.nn),D,mat,cohesive_network=None)
    if len(free):
        u[free]+=spsolve(K[free][:,free],-R[free]);u[fixed]=[prescribed[int(i)] for i in fixed]
    K,R,*_=assemble_mechanics(mesh,u,np.zeros((3,mesh.ne)),np.zeros(mesh.ne),
        np.zeros(mesh.nn),D,mat,cohesive_network=None)
    top_reaction=float(R[2*2+1]+R[2*3+1])
    traction=abs(top_reaction)/W
    return {"width_m":W,"half_height_m":H,"bulk_only":True,
        "cohesive_tangent_included":False,"imposed_total_jump_m":jump,
        "interface_jump_m":0.,"traction_Pa":traction,
        "environment_stiffness_Pa_per_m":traction/jump,
        "free_residual_N_per_m":float(np.linalg.norm(R[free]))}


def poisson_rate_interval(events:int,exposure_cycles:float,patch_m:float,alpha:float=.05):
    n=int(events);ex=float(exposure_cycles)
    low=.5*chi2.ppf(alpha/2,2*n)/ex if n else 0.
    high=.5*chi2.ppf(1-alpha/2,2*(n+1))/ex
    return [patch_m*low,patch_m*high]


def main(argv=None)->int:
    p=argparse.ArgumentParser();p.add_argument("--registry",type=Path,required=True)
    p.add_argument("--candidate",default="R10_INVERSE_OPT_027__CM")
    p.add_argument("--campaign-root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    row=registry_row(a.registry,a.candidate);cohesive=json.loads(row["cohesive_parameter_vector"])
    thermal=thermal_from(cohesive);Lpz,control_k=mechanics(thermal,300.)
    gamma=thermal.gamma(300.);peak=thermal.t_hat(300.)
    fixtures=[bulk_schur_stiffness(1e-6,H) for H in (Lpz,2e-6)]
    for fixture in fixtures:
        k=fixture["environment_stiffness_Pa_per_m"]
        fixture["dimensionless_k_Gamma_over_peak2"]=k*gamma/peak**2
        fixture["ratio_to_declared_control"]=k/control_k
    environment={"schema":"r10.mechanical-environment-audit/1",
        "candidate_id":a.candidate,"complete_row_sha256":row["complete_row_sha256"],
        "declared_control":{"Eprime_Pa":400e9/(1-.22**2),"Gamma_J_m2":gamma,
            "peak_traction_Pa":peak,"Lpz_m":Lpz,
            "environment_stiffness_Pa_per_m":control_k,
            "dimensionless_k_Gamma_over_peak2":control_k*gamma/peak**2,
            "interpretation":"CONSTITUTIVE_PROCESS_ZONE_HALF_HEIGHT_CONTROL_NOT_NUMERICAL_CELL_LENGTH"},
        "distinct_lengths":{"crack_increment_m":1e-6,"tip_radius_m":1e-6,
            "fixture_half_height_m":2e-6,"declared_Lpz_m":Lpz},
        "assembled_bulk_only_fixtures":fixtures,
        "conclusion":("DECLARED_CLOSURE_IS_REPRODUCED_ONLY_BY_A_FIXTURE_WITH_HALF_HEIGHT_LPZ; "
            "THE_MICROMETRE_FIXTURE_HAS_A DISTINCT, GEOMETRY_CONTROLLED SCHUR STIFFNESS")}
    atomic(a.out/"mechanical_environment_audit.json",environment)

    shape=row["selected_shape"];area=float(cohesive["activation_area_m2"])
    attempt=float(cohesive["attempt_frequency_s"])
    case_paths=sorted(a.campaign_root.glob(
        "hcf_bounded_inverse_027_amplitude*/R10_INVERSE_OPT_027__CM/m_*/result.json"))
    native=[]
    for path in case_paths:
        result=json.loads(path.read_text());window=result.get("developed_window")
        if not window:continue
        event_count=round((float(window["end_extension_m"])-float(window["start_extension_m"]))/
                          float(result["patch_m"]))
        duration=float(window["duration_cycles"])
        native.append({"path":str(path),"result_sha256":sha256(path),
            "Kmax_MPa_sqrt_m":float(result["Kmax_MPa_sqrt_m"]),
            "hazard_seed":result["hazard_seed"],"phase_count":result["phase_count"],
            "developed_rate_m_per_cycle":result["developed_rate_m_per_cycle"],
            "developed_window_cycles":duration,"developed_window_events":event_count,
            "poisson_95_rate_interval_m_per_cycle":poisson_rate_interval(
                event_count,duration,float(result["patch_m"]))})
    loads=sorted({item["Kmax_MPa_sqrt_m"] for item in native})
    analytical=[]
    for K in loads:
        record={"Kmax_MPa_sqrt_m":K,"gauss":{},"native_midpoint":{}}
        for n in (8,16,32,64,128,256):
            for integration,key in (("finite_interval_gauss","gauss"),("midpoint","native_midpoint")):
                point=cyclic_rate(K,control_k,thermal,300.,shape,area,attempt,
                    R=.1,frequency_Hz=1000.,phases=n,integration=integration)
                record[key][str(n)]={k:point.get(k) for k in (
                    "opening_action_per_cycle","ensemble_rate_m_per_cycle",
                    "accessible_phase_fraction","accessible_phase_count","phase_count")}
        derivatives={}
        for eps in (.02,.01,.005,.002):
            curve=cyclic_rate_curve([K*(1-eps),K,K*(1+eps)],control_k,thermal,
                300.,shape,area,attempt,R=.1,frequency_Hz=1000.,phases=64)
            mid=curve["rows"][1]
            derivatives[str(eps)]={k:mid.get(k) for k in (
                "local_log_slope","accessibility_log_slope",
                "conditional_rate_log_slope","slope_decomposition_error")}
        record["shrinking_amplitude_derivatives"]=derivatives;analytical.append(record)
    discrepancy={"schema":"r10.analytical-native-reference-audit/1",
        "candidate_id":a.candidate,"complete_row_sha256":row["complete_row_sha256"],
        "continuum_cycle_definition":"FINITE_INTERVAL_GAUSS_SPLIT_AT_BIRTH_AND_FOLD",
        "native_discrete_definition":"UNIFORM_PHASE_MIDPOINT",
        "native_developed_windows":native,"frozen_reference_by_phase_resolution":analytical,
        "interpretation":("MIDPOINT RESOLUTION IS A NUMERICAL CONVENTION CHECK; POISSON INTERVALS "
            "BOUND FINITE STOCHASTIC WINDOWS; RESIDUAL DIFFERENCE MAY INCLUDE EVOLVING STATE")}
    atomic(a.out/"analytical_native_reference_audit.json",discrepancy)
    atomic(a.out/"summary.json",{"schema":"r10.continuation-reference-summary/1",
        "environment_audit":"mechanical_environment_audit.json",
        "reference_audit":"analytical_native_reference_audit.json",
        "native_windows":len(native),"unique_loads":len(loads)})
    print(json.dumps(json.loads((a.out/"summary.json").read_text()),indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
