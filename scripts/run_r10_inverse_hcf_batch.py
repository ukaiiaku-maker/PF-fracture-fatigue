#!/usr/bin/env python3
"""Parallel restart-safe HCF execution for selected inverse-shape materials."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_r10_coupled_opening_search import reference_load
from scripts.run_r10_czm_hcf import run_case
from arrhenius_fracture.finite_compliance_inverse_shape_r10 import cyclic_rate
from scripts.run_r10_opening_alternatives import mechanics,thermal_from


def atomic(path:Path,payload:dict)->None:
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n")
    temporary.replace(path)


def task_run(task:dict)->dict:
    return run_case(Path(task["registry"]),task["candidate_id"],task["factor"],
        task["horizon"],Path(task["out"]),task["seed"],task["phases"],
        T=task["temperature_K"],R=task["R"],frequency=task["frequency_Hz"],
        patch_m=1e-6,path_limit_m=task["path_limit_um"]*1e-6,
        max_wall_s=task["max_wall_s"],plastic_feedback=task["plastic_feedback"])


def resolved_phase_count(row:dict,base:int,minimum_samples:int,maximum:int)->int:
    fraction=float(row.get("predicted_hcf_accessible_phase_fraction") or 0.)
    if minimum_samples <= 0 or fraction <= 0.:
        return int(base)
    needed=math.ceil(int(minimum_samples)/fraction)
    return min(int(maximum),max(int(base),needed))


def accessible_fraction_for_load(row:dict,Kmax:float,temperature_K:float,
                                 R:float,frequency_Hz:float)->float:
    """Resolve each protocol load, rather than reusing the base-load width."""
    cohesive=json.loads(row["cohesive_parameter_vector"])
    thermal=thermal_from(cohesive);_,kenv=mechanics(thermal,temperature_K)
    point=cyclic_rate(Kmax,kenv,thermal,temperature_K,row["selected_shape"],
        float(cohesive["activation_area_m2"]),
        float(cohesive["attempt_frequency_s"]),R=R,
        frequency_Hz=frequency_Hz,phases=8)
    return float(point.get("accessible_phase_fraction") or 0.)


def main(argv=None)->int:
    parser=argparse.ArgumentParser();parser.add_argument("--registry",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True);parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--candidate-ids",default="",
        help="optional comma-separated subset; omitted means every registry row")
    parser.add_argument("--amplitude-multipliers",default="1.0")
    parser.add_argument("--horizon",type=float,default=1e8);parser.add_argument("--max-wall-s",type=float,default=900.)
    parser.add_argument("--phases",type=int,default=16);parser.add_argument("--temperature-K",type=float,default=300.)
    parser.add_argument("--minimum-accessible-phase-samples",type=int,default=0,
        help="raise the uniform phase count from registry accessibility support")
    parser.add_argument("--maximum-auto-phases",type=int,default=4096)
    parser.add_argument("--R",type=float,default=.1);parser.add_argument("--frequency-Hz",type=float,default=1000.)
    parser.add_argument("--path-limit-um",type=float,default=120.);parser.add_argument("--seed",type=int,default=20261018)
    parser.add_argument("--frozen-initial-plastic-state",action="store_true")
    args=parser.parse_args(argv);args.out.mkdir(parents=True,exist_ok=True)
    with args.registry.open(newline="") as stream:rows=list(csv.DictReader(stream))
    wanted={value for value in args.candidate_ids.split(",") if value}
    if wanted:
        rows=[row for row in rows if row["candidate_id"] in wanted]
        missing=wanted-{row["candidate_id"] for row in rows}
        if missing:raise ValueError(f"candidate IDs absent from registry: {sorted(missing)}")
    multipliers=[float(value) for value in args.amplitude_multipliers.split(",")]
    tasks=[]
    for index,row in enumerate(rows):
        reference,_=reference_load(row,args.temperature_K,args.phases)
        predicted=float(row.get("predicted_hcf_K_300K") or row["predicted_Kaccess_300K"])
        for j,multiplier in enumerate(multipliers):
            desired=predicted*multiplier;label=f"m_{multiplier:.5f}".replace(".","p")
            phase_row=dict(row)
            fraction=accessible_fraction_for_load(row,desired,args.temperature_K,
                                                  args.R,args.frequency_Hz)
            phase_row["predicted_hcf_accessible_phase_fraction"]=fraction
            task_phases=resolved_phase_count(phase_row,args.phases,
                args.minimum_accessible_phase_samples,args.maximum_auto_phases)
            tasks.append({"registry":str(args.registry),"candidate_id":row["candidate_id"],
                "factor":desired/reference,"desired_Kmax":desired,"horizon":args.horizon,
                "out":str(args.out/row["candidate_id"]/label),"seed":args.seed+1000*index+j,
                "phases":task_phases,"reference_phases":args.phases,
                "minimum_accessible_phase_samples":args.minimum_accessible_phase_samples,
                "temperature_K":args.temperature_K,"R":args.R,
                "frequency_Hz":args.frequency_Hz,"path_limit_um":args.path_limit_um,
                "phase_selection_accessible_fraction":fraction,
                "max_wall_s":args.max_wall_s,
                "plastic_feedback":not args.frozen_initial_plastic_state})
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(task_run,task):task for task in tasks}
        for future in as_completed(futures):
            task=futures[future]
            try:result=future.result();error=None
            except Exception as exc:result={};error=f"{type(exc).__name__}: {exc}"
            summary={"candidate_id":task["candidate_id"],"desired_Kmax_MPa_sqrt_m":task["desired_Kmax"],
                "temperature_K":task["temperature_K"],"R":task["R"],"frequency_Hz":task["frequency_Hz"],
                "reference_phase_count":task["reference_phases"],
                "phase_count":task["phases"],
                "minimum_accessible_phase_samples":task["minimum_accessible_phase_samples"],
                "phase_selection_accessible_fraction":task["phase_selection_accessible_fraction"],
                "accepted_cycles":result.get("accepted_cycles"),"accepted_seconds":result.get("accepted_seconds"),
                "physical_event_count":result.get("physical_event_count"),"advance_m":result.get("advance_m"),
                "developed_rate_m_per_cycle":result.get("developed_rate_m_per_cycle"),
                "disposition":result.get("disposition"),"restart_checkpoint":result.get("restart_checkpoint"),
                "error":error};results.append(summary);print(json.dumps(summary),flush=True)
    atomic(args.out/"batch_summary.json",{"schema":"r10.inverse-shape-hcf-batch/1",
        "registry":str(args.registry),"tasks":len(tasks),"results":results})
    return 0


if __name__=="__main__":raise SystemExit(main())
