#!/usr/bin/env python3
"""Idempotent same-material fracture and developed-fatigue native campaign."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.paris_slope_r9 import run_trajectory
from arrhenius_fracture.reduced_persistent_reference_r2 import runtime_receipt
from arrhenius_fracture.retained_material_cohesive_r10 import (
    build_cohesive_current_engine,load_cohesive_binding,
)
from scripts.run_r10_cohesive_arrhenius_search import opening_action
from scripts.run_r10_current_fracture_panel import run as run_fracture


def atomic(path,payload):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n");temp.replace(path)


def select_loads(cohesive,temperature=300.):
    grid=[2.+38.*i/759 for i in range(760)]
    rows=[(K,5e-6*opening_action(cohesive,K,temperature)) for K in grid]
    targets=(1e-10,1e-9,1e-8,1e-7,1e-6)
    used=[]
    for target in targets:
        index=min(range(len(rows)),key=lambda i:abs(math.log10(max(rows[i][1],1e-300))-math.log10(target)))
        if index not in used:used.append(index)
    if len(used)<5:
        used=sorted(set(used+list(range(0,len(rows),max(len(rows)//5,1)))))[:5]
    return [rows[i][0] for i in sorted(used)]


def window(result,lo=20e-6,hi=60e-6):
    events=result["events"]
    left=next((e for e in events if e["extension_m"]>=lo),None)
    right=next((e for e in events if e["extension_m"]>=hi),None)
    if left is None or right is None or right["cycles"]<=left["cycles"]:return None
    return {"start_extension_m":left["extension_m"],"end_extension_m":right["extension_m"],
            "start_cycles":left["cycles"],"end_cycles":right["cycles"],
            "rate_m_per_cycle":(right["extension_m"]-left["extension_m"])/(right["cycles"]-left["cycles"]),
            "events_inclusive":right["event_index"]-left["event_index"]+1}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--registry",type=Path,required=True)
    p.add_argument("--candidate-id",required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--mode",choices=("fracture","fatigue","all"),default="all")
    p.add_argument("--loads");p.add_argument("--temperature-K",type=float,default=300.)
    p.add_argument("--seed",type=int,default=10162026);p.add_argument("--maximum-cycles",type=float,default=1e12)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    binding=load_cohesive_binding(a.registry,a.candidate_id)
    fingerprint=runtime_receipt(build_cohesive_current_engine(
        binding,kernel_family_path=R4_KERNEL_FAMILY),stage="cohesive_native_preflight")
    atomic(a.out/"consumed_runtime_fingerprint.json",fingerprint)
    with a.registry.open(newline="") as stream: record=next(r for r in csv.DictReader(stream) if r["candidate_id"]==a.candidate_id)
    cohesive=json.loads(record["cohesive_parameter_vector"])
    if a.mode in ("fracture","all"):
        root=a.out/"fracture";root.mkdir(exist_ok=True)
        cases=[]
        for T in (300.,450.,600.,750.,900.,1050.,1200.):
            path=root/f"T_{int(T):04d}K.json"
            if path.exists():result=json.loads(path.read_text())
            else:
                result=run_fracture(binding,T,.005,80.,.05,50.,2_000_000);atomic(path,result)
            cases.append({k:result[k] for k in ("temperature_K","status","K_init_MPa_sqrt_m",
                "K_final_event_MPa_sqrt_m","final_extension_um","event_count")})
            print(json.dumps({"stage":"fracture",**cases[-1]},sort_keys=True),flush=True)
        atomic(root/"panel_summary.json",{"candidate_id":a.candidate_id,"cases":cases,
            "same_complete_material_sha256":binding.provenance["complete_row_sha256"]})
    if a.mode in ("fatigue","all"):
        loads=[float(x) for x in a.loads.split(",")] if a.loads else select_loads(cohesive,a.temperature_K)
        summaries=[]
        for i,K in enumerate(loads):
            case=a.out/"fatigue"/f"Kmax_{K:.6f}";case.mkdir(parents=True,exist_ok=True)
            path=case/"result.json"
            if path.exists():result=json.loads(path.read_text())
            else:
                result=run_trajectory(binding,kernel_family_path=R4_KERNEL_FAMILY,
                    Kmax_MPa_sqrt_m=K,seed=a.seed,checkpoint_directory=case/"restart",
                    target_events=14,target_extension_m=70e-6,maximum_events=60,
                    maximum_cycles=a.maximum_cycles,phase_count=64,
                    temperature_K=a.temperature_K,R=.1,frequency_Hz=1000.)
                result["binding_provenance"]=dict(binding.provenance);atomic(path,result)
            if "consumed_runtime_fingerprint" not in result:
                result["consumed_runtime_fingerprint"]=fingerprint
                result["binding_provenance"]=dict(binding.provenance);atomic(path,result)
            row={"Kmax_MPa_sqrt_m":K,"DeltaK_MPa_sqrt_m":.9*K,"status":result["status"],
                 "events":len(result["events"]),"cycles":result["final_cycles"],
                 "extension_m":result["final_extension_m"],"developed_window":window(result)}
            summaries.append(row);print(json.dumps({"stage":"fatigue",**row},sort_keys=True),flush=True)
        atomic(a.out/"fatigue"/"campaign_summary.json",{"candidate_id":a.candidate_id,
            "temperature_K":a.temperature_K,"R":.1,"frequency_Hz":1000.,"loads":summaries,
            "same_complete_material_sha256":binding.provenance["complete_row_sha256"]})
    return 0

if __name__=="__main__":raise SystemExit(main())
