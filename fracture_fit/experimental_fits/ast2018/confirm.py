"""Direct paired 0-D/1-D confirmation at an unchanged fitted material vector."""
from __future__ import annotations

import argparse
import csv
import json
import os
import math
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import AstAssumptions
from .data import AstData
from .mechanics import basic_J_at_failure,continuous_onset_K,lefm_Kq_at_failure,observe_like_ast
from .native import historical_start,load_native
from .objective import WorkingObjective,material_from_normalized,robust_quadrature_residual
from .reduced import predict_reduced
from .spatial import predict_spatial


_CTX=None


def _json_clean(value):
    if isinstance(value,dict):return {k:_json_clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_clean(v) for v in value]
    if isinstance(value,(float,np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value,np.integer):return int(value)
    return value


def _init(source_dir,data_dir,scenario,assumptions_dict,x,level,output_dir,
          validation_endpoint_horizon):
    global _CTX
    source=Path(source_dir);mods,receipt=load_native(source)
    base,physics,_=historical_start(source,mods[2],mods[3].CommonPhysics)
    candidate=material_from_normalized(base,x)
    data=AstData.load(Path(data_dir));assumptions=AstAssumptions(**assumptions_dict)
    module=mods[0] if level=="reduced" else mods[1]
    predictor=predict_reduced if level=="reduced" else predict_spatial
    objective=WorkingObjective(data,scenario,assumptions,module,base,physics,
                               predictor=predictor)
    _CTX=(candidate,physics,data,assumptions,module,predictor,objective,
          Path(output_dir),receipt,level,validation_endpoint_horizon)


def _write_csv(path,items):
    if not items:return
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(items[0]))
        writer.writeheader();writer.writerows(items)


def _one_case(condition_id):
    candidate,base_physics,data,assumptions,module,predictor,objective,out,receipt,level,validation_endpoint_horizon=_CTX
    condition=data.condition(condition_id);specimen=assumptions.specimen(condition)
    physics=assumptions.physics(base_physics,specimen)
    target,required_u=objective.case_targets(condition_id)
    endpoint_values=[float(r["K_preferred_MPa_sqrt_m"]) for r in data.no_cleavage
                     if validation_endpoint_horizon and r["condition_id"]==condition_id]
    endpoint_K=max(endpoint_values) if endpoint_values else None
    endpoint_u=(endpoint_K*1e6/specimen.geometry_factor(specimen.a0_m)*
                (specimen.initial_compliance_m_per_N+
                 specimen.instrument_compliance_m_per_N)) if endpoint_K is not None else 0.0
    required_u=max(required_u,endpoint_u)
    t=perf_counter()
    result=predictor(module,candidate,physics,specimen,
        float(condition["temperature_K"]),
        float(condition["initial_elastic_Kdot_MPa_sqrt_m_per_s"]),
        assumptions,deterministic_threshold=True,
        required_u_m=required_u,required_extension_m=target)
    elapsed=perf_counter()-t
    try:observed=observe_like_ast(result.history,specimen)
    except Exception as exc:
        observed=[]
        result.numerical_failure=f"observer: {type(exc).__name__}: {exc}"
        result.status="numerical_failure"
    folder=out/level/condition_id;folder.mkdir(parents=True,exist_ok=True)
    _write_csv(folder/"mechanical_history.csv",[asdict(row) for row in result.history])
    _write_csv(folder/"ast_observer.csv",[asdict(row) for row in observed])
    (folder/"events.json").write_text(json.dumps(_json_clean(result.events),indent=2,
                                             allow_nan=False)+"\n")
    detail=[]
    for family in objective.families.get(condition_id,()):
        residual=objective._family_residual(family,result,observed,specimen)
        scaled=robust_quadrature_residual(residual,np.array(family.weights),
                                         1/max(len(objective.families[condition_id]),1))
        detail.append({"family":family.name,"curve_id":family.curve_id,
                       "cost":float(.5*np.sum(scaled**2)),
                       "rms_standardized":float(np.sqrt(np.mean(residual**2)))})
    summary={"condition_id":condition_id,"level":level,"status":result.status,
        "numerical_failure":result.numerical_failure,"runtime_s":elapsed,
        "load_steps":result.load_steps,"event_count":len(result.events),
        "max_hazard_step":result.maximum_hazard_step,
        "achieved_extension_m":result.history[-1].a_m-specimen.a0_m,
        "final_u_m":result.history[-1].imposed_u_m,
        "validation_endpoint_nominal_K_MPa_sqrt_m":endpoint_K,
        "validation_endpoint_initial_elastic_u_m":endpoint_u if endpoint_K is not None else None,
        "source_commit":receipt["commit"],"candidate_id":candidate.candidate_id,
        "deterministic_threshold_action":1.0,
        "translation_mode":"hazard_coupled" if level=="spatial" else "source_zone_tip_weighted",
        "translation_action_exponent":.95 if level=="spatial" else None,
        "onset_K_Jcontinuous_MPa_sqrt_m":continuous_onset_K(observed,specimen),
        "LEFM_Kq_MPa_sqrt_m":lefm_Kq_at_failure(observed,specimen),
        "basic_J_Kq_a0_MPa_sqrt_m":basic_J_at_failure(observed,specimen),
        "basic_J_Kq_current_a_MPa_sqrt_m":basic_J_at_failure(observed,specimen,use_current_crack=True),
        "residual_blocks":detail,
        "specimen":asdict(specimen),"physics_G_Pa":physics.G_Pa}
    temp=folder/"summary.tmp.json";final=folder/"summary.json"
    temp.write_text(json.dumps(_json_clean(summary),indent=2,allow_nan=False)+"\n")
    os.replace(temp,final)
    return summary


def confirm_all(source:Path,data_dir:Path,run_dir:Path,scenario:str,x:list[float],
                assumptions:AstAssumptions,*,levels=("reduced","spatial"),workers=8,
                condition_ids:list[str]|None=None,stage:str="final",
                validation_endpoint_horizon:bool=False):
    if workers<1 or workers>8:raise ValueError("workers must be 1..8")
    data=AstData.load(data_dir)
    output=run_dir/(f"confirmation_{scenario.lower()}" if stage=="final" else
                    f"confirmation_{scenario.lower()}_{stage}")
    output.mkdir(parents=True,exist_ok=True)
    basis={"x":x,"scenario":scenario,
        "assumptions":asdict(assumptions),"levels":list(levels),
        "validation_endpoint_horizon":validation_endpoint_horizon}
    basis_path=output/"vector.json"
    if basis_path.exists() and json.loads(basis_path.read_text())!=basis:
        raise RuntimeError("paired confirmation vector or convention changed on resume")
    basis_path.write_text(json.dumps(basis,indent=2)+"\n")
    selected=(condition_ids if condition_ids is not None else
              [r["condition_id"] for r in data.conditions])
    if len(set(selected))!=len(selected) or any(cid not in {r["condition_id"] for r in data.conditions}
                                          for cid in selected):
        raise ValueError("invalid confirmation condition subset")
    summaries=[]
    for level in levels:
        pending=[]
        for cid in selected:
            folder=output/level/cid
            required=("summary.json","mechanical_history.csv","ast_observer.csv",
                      "events.json")
            if all((folder/name).exists() and (folder/name).stat().st_size>0
                   for name in required):
                cached=json.loads((folder/"summary.json").read_text())
                if cached["status"].startswith("numerical") or cached.get("numerical_failure"):
                    pending.append(cid)
                else:summaries.append(cached)
            else:pending.append(cid)
        with ProcessPoolExecutor(max_workers=workers,initializer=_init,
            initargs=(str(source.resolve()),str(data_dir.resolve()),scenario,
                      asdict(assumptions),x,level,str(output.resolve()),
                      validation_endpoint_horizon)) as pool:
            futures={pool.submit(_one_case,cid):cid for cid in pending}
            for future in as_completed(futures):
                row=future.result();summaries.append(row)
                print(f"{level} {len([r for r in summaries if r['level']==level])}/{len(selected)} "
                      f"{row['condition_id']} {row['status']} {row['runtime_s']:.2f}s",flush=True)
    (output/"summary.json").write_text(json.dumps(_json_clean(summaries),indent=2,
                                           allow_nan=False)+"\n")
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,default=Path("runs/ast2018_joint_fit"))
    parser.add_argument("--scenario",choices=("TABLE_PRIMARY","FIGURE5_ALTERNATIVE"),
                        default="TABLE_PRIMARY")
    parser.add_argument("--vector",type=Path,required=True,
                        help="JSON file with normalized x array and provenance")
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--level",choices=("both","reduced","spatial"),default="both")
    parser.add_argument("--subset",choices=("all","priority11"),default="all")
    args=parser.parse_args()
    cfg=json.loads((args.run_dir/f"config_{args.scenario.lower()}.json").read_text())
    vector=json.loads(args.vector.read_text())["x"]
    levels=("reduced","spatial") if args.level=="both" else (args.level,)
    priority=["Tp21C_rate0p2","Tp100C_rate0p2","Tp250C_rate0p2",
              "Tp21C_rate0p5","Tp100C_rate0p5","Tp250C_rate0p5",
              "Tp100C_rate1p6","Tp500C_rate1p6","Tm90C_rate0p2",
              "Tp0C_rate0p05","Tp75C_rate1p6"]
    confirm_all(args.native_source,args.run_dir/"input/Ast2018_joint_fit_data",
        args.run_dir,args.scenario,vector,AstAssumptions(**cfg["ast_assumptions"]),
        levels=levels,workers=args.workers,
        condition_ids=priority if args.subset=="priority11" else None,
        stage="priority11" if args.subset=="priority11" else "final")


if __name__=="__main__":main()
