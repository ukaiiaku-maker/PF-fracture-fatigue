"""Conditional source, geometry, event and working-weight sensitivity table."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import AstAssumptions
from .campaign import _append_jsonl,_load_existing
from .data import AstData
from .native import historical_start,load_native
from .objective import WorkingObjective


def _variant_task(spec):
    source_dir,data_dir,scenario,x,base_assumptions,name,kind,value=spec
    start=perf_counter()
    source=Path(source_dir);mods,receipt=load_native(source)
    candidate,physics,_=historical_start(source,mods[2],mods[3].CommonPhysics)
    assumptions=AstAssumptions(**base_assumptions)
    if kind=="geometry":assumptions=replace(assumptions,**value)
    elif kind=="candidate":candidate=replace(candidate,**value)
    elif kind=="physics":physics=replace(physics,**value)
    elif kind!="baseline":raise ValueError(kind)
    data=AstData.load(Path(data_dir))
    objective=WorkingObjective(data,scenario,assumptions,mods[0],candidate,physics)
    residual,detail=objective.evaluate(x,detail=True)
    return {"variant":name,"kind":kind,"override":value,"cost":float(.5*np.sum(residual**2)),
            "runtime_s":perf_counter()-start,"forward_case_calls":objective.case_calls,
            "detail":detail,"interpretation":"fixed fitted vector; conditional model sensitivity"}


def _weighted_cost(details,family,factor):
    by_condition={}
    for row in details:by_condition.setdefault(row["condition_id"],[]).append(row)
    total=0.0
    for rows in by_condition.values():
        weights=[factor if row["family"]==family else 1.0 for row in rows]
        denominator=sum(weights)
        total+=sum(row["block_cost"]*len(rows)*w/denominator
                   for row,w in zip(rows,weights))
    return total


def analyze(source:Path,run_dir:Path,scenario:str,vector_path:Path,*,workers=8):
    cfg=json.loads((run_dir/f"config_{scenario.lower()}.json").read_text())
    x=json.loads(vector_path.read_text())["x"]
    source_args=(str(source.resolve()),str((run_dir/"input/Ast2018_joint_fit_data").resolve()),
                 scenario,x,cfg["ast_assumptions"])
    variants=[
        ("baseline","baseline",{}),
        ("L_11um","geometry",{"L_m":11e-6}),
        ("L_14um","geometry",{"L_m":14e-6}),
        ("a0_2um","geometry",{"a0_m":2e-6}),
        ("a0_2p5um","geometry",{"a0_m":2.5e-6}),
        ("radius_50nm","geometry",{"r0_m":50e-9}),
        ("radius_200nm","geometry",{"r0_m":200e-9}),
        ("MPZ_1um","geometry",{"mpz_length_m":1e-6}),
        ("MPZ_3um","geometry",{"mpz_length_m":3e-6}),
        ("source_zone_0p25um","geometry",{"source_zone_length_m":.25e-6}),
        ("source_zone_1um","geometry",{"source_zone_length_m":1e-6}),
        ("A2_speed_120nm_per_s","geometry",{"A2_reference_speed_m_per_s":120e-9}),
        ("event_advance_10nm","geometry",{"event_advance_m":10e-9}),
    ]
    mods,_=load_native(source)
    candidate,physics,_=historical_start(source,mods[2],mods[3].CommonPhysics)
    variants.extend([
        ("source_density_half","candidate",{"rho_source0_m2":candidate.rho_source0_m2*.5}),
        ("source_density_double","candidate",{"rho_source0_m2":candidate.rho_source0_m2*2}),
        ("cleavage_prefactor_half","physics",{"cleavage_nu0_s":physics.cleavage_nu0_s*.5}),
        ("cleavage_prefactor_double","physics",{"cleavage_nu0_s":physics.cleavage_nu0_s*2}),
        ("emission_prefactor_half","physics",{"emission_nu0_s":physics.emission_nu0_s*.5}),
        ("emission_prefactor_double","physics",{"emission_nu0_s":physics.emission_nu0_s*2}),
    ])
    tasks=[source_args+(name,kind,value) for name,kind,value in variants]
    basis={"scenario":scenario,"vector":x,"assumptions":cfg["ast_assumptions"],
           "variants":[{"name":name,"kind":kind,"override":value}
                       for name,kind,value in variants]}
    basis_path=run_dir/f"sensitivity_basis_{scenario.lower()}.json"
    if basis_path.exists() and json.loads(basis_path.read_text())!=basis:
        raise RuntimeError("fixed-vector sensitivity checkpoint basis changed")
    basis_path.write_text(json.dumps(basis,indent=2,allow_nan=False)+"\n")
    checkpoint=run_dir/f"sensitivity_calls_{scenario.lower()}.jsonl"
    completed=_load_existing(checkpoint)
    names={name for name,_,_ in variants}
    if set(completed)-names:
        raise RuntimeError("unknown fixed-vector sensitivity checkpoint key")
    results=list(completed.values())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(_variant_task,spec):spec[5] for spec in tasks
                 if spec[5] not in completed}
        for future in as_completed(futures):
            row=future.result();row["key"]=row["variant"]
            _append_jsonl(checkpoint,row)
            results.append(row)
            print(f"sensitivity {len(results)}/{len(tasks)} {row['variant']} "
                  f"cost={row['cost']:.5g}",flush=True)
    baseline=next(row for row in results if row["variant"]=="baseline")
    for family in ("onset","J_curve","nominal_K_trace","relative_stiffness"):
        for factor in (.5,2.0):
            results.append({"variant":f"weight_{family}_{factor:g}x",
                "kind":"working_weight","override":{"family":family,"factor":factor},
                "cost":_weighted_cost(baseline["detail"],family,factor),
                "runtime_s":0.0,"forward_case_calls":0,
                "interpretation":"fixed-vector block reweighting; no refit"})
    results.sort(key=lambda r:r["variant"])
    for row in results:
        row["cost_delta_from_baseline"]=row["cost"]-baseline["cost"]
    output=run_dir/f"sensitivity_{scenario.lower()}.json"
    output.write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,default=Path("runs/ast2018_joint_fit_v1"))
    parser.add_argument("--scenario",choices=("TABLE_PRIMARY","FIGURE5_ALTERNATIVE"),
                        default="TABLE_PRIMARY")
    parser.add_argument("--vector",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    analyze(args.native_source,args.run_dir,args.scenario,args.vector,workers=args.workers)


if __name__=="__main__":main()
