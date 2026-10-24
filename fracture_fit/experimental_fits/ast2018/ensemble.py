"""Common-seed stochastic onset and first-event checks for the final vector."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import AstAssumptions
from .campaign import _append_jsonl,_load_existing
from .data import AstData
from .mechanics import continuous_onset_K,observe_like_ast
from .native import historical_start,load_native
from .objective import material_from_normalized
from .reduced import predict_reduced
from .spatial import predict_spatial


_CTX=None


def _init(source_dir,data_dir,assumptions_dict,x,level):
    global _CTX
    source=Path(source_dir);mods,receipt=load_native(source)
    base,physics,_=historical_start(source,mods[2],mods[3].CommonPhysics)
    data=AstData.load(Path(data_dir))
    _CTX=(material_from_normalized(base,x),physics,data,
          AstAssumptions(**assumptions_dict),mods[0] if level=="reduced" else mods[1],
          predict_reduced if level=="reduced" else predict_spatial,level)


def _one(spec):
    condition_id,seed,K_reference=spec
    candidate,base,data,assumptions,module,predictor,level=_CTX
    condition=data.condition(condition_id);specimen=assumptions.specimen(condition)
    physics=assumptions.physics(base,specimen)
    t=perf_counter()
    result=predictor(module,candidate,physics,specimen,
        float(condition["temperature_K"]),
        float(condition["initial_elastic_Kdot_MPa_sqrt_m_per_s"]),
        assumptions,event_seed=seed,deterministic_threshold=False,
        required_extension_m=.22e-6)
    try:obs=observe_like_ast(result.history,specimen)
    except Exception:obs=[]
    first_K=(result.events[0].get("K_pre_MPa_sqrt_m") if result.events else None)
    if first_K is None:first_event_by=None if result.status.startswith("numerical") else False
    else:first_event_by=first_K<=K_reference
    onset=continuous_onset_K(obs,specimen) if obs else float("nan")
    return {"condition_id":condition_id,"seed":seed,"level":level,
        "status":result.status,"runtime_s":perf_counter()-t,
        "event_count":len(result.events),"first_event_K_MPa_sqrt_m":first_K,
        "first_event_by_observed_reference_K":first_event_by,
        "observed_reference_K_MPa_sqrt_m":K_reference,
        "onset_K_Jcontinuous_MPa_sqrt_m":float(onset) if np.isfinite(onset) else None,
        "threshold_actions":[r["threshold_action"] for r in result.events],
        "numerical_failure":result.numerical_failure}


def run(source:Path,data_dir:Path,run_dir:Path,scenario:str,vector:Path,
        *,level="reduced",n_seeds=8,workers=8,condition_ids=None):
    cfg=json.loads((run_dir/f"config_{scenario.lower()}.json").read_text())
    data=AstData.load(data_dir)
    x=json.loads(vector.read_text())["x"]
    reference={}
    for row in data.toughness_scenario(scenario):
        if row["observable"]=="Kq_Jcontinuous_onset":
            reference[row["condition_id"]]=float(row["value"])
    for row in data.toughness_scenario(scenario):
        reference.setdefault(row["condition_id"],float(row["value"]))
    selected=(condition_ids if condition_ids is not None else
              [r["condition_id"] for r in data.conditions])
    seeds=[2018001+i for i in range(n_seeds)]
    tasks=[(cid,seed,reference.get(cid,20.0)) for cid in selected for seed in seeds]
    output=run_dir/f"ensemble_{scenario.lower()}_{level}"
    output.mkdir(parents=True,exist_ok=True)
    basis={"scenario":scenario,"level":level,"vector":x,
           "seeds":seeds,"condition_ids":selected,
           "source_commit":cfg["source_receipt"]["commit"],
           "adapter_sha256":cfg["adapter_sha256"]}
    basis_path=output/"checkpoint_basis.json"
    if basis_path.exists() and json.loads(basis_path.read_text())!=basis:
        raise RuntimeError("stochastic ensemble checkpoint basis changed")
    basis_path.write_text(json.dumps(basis,indent=2)+"\n")
    checkpoint=output/"completed_runs.jsonl"
    completed=_load_existing(checkpoint)
    rows=list(completed.values())
    pending=[task for task in tasks if f"{task[0]}:{task[1]}" not in completed]
    with ProcessPoolExecutor(max_workers=workers,initializer=_init,
        initargs=(str(source.resolve()),str(data_dir.resolve()),cfg["ast_assumptions"],x,level)) as pool:
        futures={pool.submit(_one,task):task for task in pending}
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            row["key"]=f"{row['condition_id']}:{row['seed']}"
            _append_jsonl(checkpoint,row)
            print(f"ensemble {level} {len(rows)}/{len(tasks)} "
                  f"{row['condition_id']} seed={row['seed']} {row['status']}",flush=True)
    rows.sort(key=lambda r:(r["condition_id"],r["seed"]))
    (output/"runs.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")
    by_condition=defaultdict(list)
    for row in rows:by_condition[row["condition_id"]].append(row)
    summary=[]
    for cid,items in sorted(by_condition.items()):
        onsets=np.array([r["onset_K_Jcontinuous_MPa_sqrt_m"] for r in items
                         if r["onset_K_Jcontinuous_MPa_sqrt_m"] is not None])
        definite=[r["first_event_by_observed_reference_K"] for r in items
                  if r["first_event_by_observed_reference_K"] is not None]
        summary.append({"condition_id":cid,"level":level,"seeds":seeds,
            "n_seeds":len(items),"n_onset_supported":len(onsets),
            "n_numerical_failure":sum(r["status"].startswith("numerical") for r in items),
            "n_physical_ligament_failure":sum(r["status"]=="physical_ligament_failure" for r in items),
            "mean_conditional_onset_K_MPa_sqrt_m":float(np.mean(onsets)) if len(onsets) else None,
            "onset_K_quantiles_10_50_90":np.quantile(onsets,[.1,.5,.9]).tolist() if len(onsets) else None,
            "first_event_probability_at_observed_reference_K":(
                sum(definite)/len(definite) if definite else None),
            "first_event_probability_denominator":len(definite),
            "observed_reference_K_MPa_sqrt_m":items[0]["observed_reference_K_MPa_sqrt_m"],
            "interpretation":"common-seed descriptive ensemble; not binomial fit to unknown experimental subgroups"})
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,default=Path("runs/ast2018_joint_fit_v1"))
    parser.add_argument("--scenario",choices=("TABLE_PRIMARY","FIGURE5_ALTERNATIVE"),
                        default="TABLE_PRIMARY")
    parser.add_argument("--vector",type=Path,required=True)
    parser.add_argument("--level",choices=("reduced","spatial"),default="reduced")
    parser.add_argument("--seeds",type=int,default=8)
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    run(args.native_source,args.run_dir/"input/Ast2018_joint_fit_data",args.run_dir,
        args.scenario,args.vector,level=args.level,n_seeds=args.seeds,workers=args.workers)


if __name__=="__main__":main()
