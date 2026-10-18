#!/usr/bin/env python3
"""Incremental, resumable current full-state 1-D acquisition block."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.run_r10_coupled_opening_search import reference_load
from scripts.run_r10_opening_recurrence import run_case


def atomic(path: Path, payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n")
    tmp.replace(path)


def execute(task):
    row=task["row"]
    result=run_case(Path(task["registry"]),row["candidate_id"],row["selected_shape"],
        row["selected_mode"],task["Kmax"],task["temperature_K"],task["cycles"],
        task["plastic_phases"],task["opening_phases"],task["patch_m"],task["seed"],
        task["path_limit_m"],task["transport_substeps"])
    result.update({"source_material_candidate_id":row["source_material_candidate_id"],
        "search_design":row["search_design"],"reference_factor":task["factor"],
        "reference_Kmax_MPa_sqrt_m":task["Kref"],"acquisition_rank":task["rank"]})
    return task,result


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--registry",type=Path,required=True)
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--cycles",type=int,default=12)
    parser.add_argument("--temperature-K",type=float,default=300.)
    parser.add_argument("--plastic-phases",type=int,default=8)
    parser.add_argument("--opening-phases",type=int,default=16)
    parser.add_argument("--transport-substeps",type=int,default=1)
    parser.add_argument("--patch-um",type=float,default=1.)
    parser.add_argument("--path-limit-um",type=float,default=120.)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--seed",type=int,default=202610183000)
    args=parser.parse_args(argv)
    with args.registry.open(newline="") as stream: registry={r["candidate_id"]:r for r in csv.DictReader(stream)}
    with args.selection.open(newline="") as stream: selection=list(csv.DictReader(stream))
    tasks=[];existing=[]
    for index,item in enumerate(selection):
        row=registry[item["candidate_id"]]
        factor=float(item["reference_factor"]);Kref,_=reference_load(row,args.temperature_K,args.opening_phases)
        case=args.out/row["candidate_id"]/f"T_{args.temperature_K:g}__f_{factor:.6f}"/"result.json"
        if case.is_file():
            existing.append(case);continue
        tasks.append({"registry":str(args.registry),"row":row,"factor":factor,"Kref":Kref,
            "Kmax":factor*Kref,"temperature_K":args.temperature_K,"cycles":args.cycles,
            "plastic_phases":args.plastic_phases,"opening_phases":args.opening_phases,
            "patch_m":args.patch_um*1e-6,"path_limit_m":args.path_limit_um*1e-6,
            "transport_substeps":args.transport_substeps,"seed":args.seed+index,
            "rank":int(item["acquisition_rank"]),"case":str(case)})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(execute,t) for t in tasks]
        for future in as_completed(futures):
            task,result=future.result();path=Path(task["case"]);atomic(path,result)
            print(json.dumps({"candidate_id":result["candidate_id"],
                "source_material_candidate_id":result["source_material_candidate_id"],
                "cycles":len(result["cycle_history"]),"events":len(result["events"]),
                "stop_category":result["stop_category"],
                "unsupported":result["unsupported_crossover_encountered"]}),flush=True)
    paths=sorted(args.out.glob("*/*/result.json"));rows=[]
    for path in paths:
        result=json.loads(path.read_text());history=result["cycle_history"]
        rows.append({"candidate_id":result["candidate_id"],
            "source_material_candidate_id":result["source_material_candidate_id"],
            "search_design":result["search_design"],"shape":result["shape"],"mode":result["mode"],
            "reference_factor":result["reference_factor"],"Kmax_MPa_sqrt_m":result["Kmax_MPa_sqrt_m"],
            "DeltaK_MPa_sqrt_m":result["DeltaK_MPa_sqrt_m"],"temperature_K":result["temperature_K"],
            "cycles":len(history),"events":len(result["events"]),
            "extension_m":history[-1]["extension_m"] if history else 0.,
            "stop_category":result["stop_category"],
            "unsupported_crossover":result["unsupported_crossover_encountered"],
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"result_path":str(path)})
    fields=list(rows[0]) if rows else []
    with (args.out/"native_case_summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)
    atomic(args.out/"execution_summary.json",{"requested":len(selection),"completed":len(rows),
        "newly_dispatched":len(tasks),"exact_reused":len(existing),"cycles_per_case":args.cycles,
        "physical_vectors":len({r["source_material_candidate_id"] for r in rows})})


if __name__=="__main__":raise SystemExit(main())
