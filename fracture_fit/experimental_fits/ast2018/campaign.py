"""Resumable, pinned-source Ast analytical search (never invokes 2-D code)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
from pathlib import Path
from time import perf_counter,time
from zipfile import ZipFile

import numpy as np
from scipy.stats import qmc

from .config import AstAssumptions
from .data import AstData
from .native import PIN,historical_start,load_native
from .objective import PARAMETERS,WorkingObjective,material_from_normalized,normalized_start,physically_admissible


_WORKER=None


def _init_worker(source_dir:str,data_dir:str,scenario:str,assumptions:dict,curve_points:int):
    global _WORKER
    source=Path(source_dir)
    mods,receipt=load_native(source)
    candidate,physics,provenance=historical_start(source,mods[2],mods[3].CommonPhysics)
    data=AstData.load(Path(data_dir))
    _WORKER=WorkingObjective(data,scenario,AstAssumptions(**assumptions),mods[0],
                             candidate,physics,curve_points=curve_points)


def _score(x:list[float])->dict:
    global _WORKER
    t=perf_counter()
    before=_WORKER.case_calls
    try:
        residual,detail=_WORKER.evaluate(np.array(x),detail=True)
        cost=float(.5*np.sum(residual**2))
        return {"x":x,"cost":cost,"rms":float(np.sqrt(np.mean(residual**2))),
                "runtime_s":perf_counter()-t,"case_calls":_WORKER.case_calls-before,
                "details":detail,"error":None}
    except Exception as exc:
        return {"x":x,"cost":1e300,"runtime_s":perf_counter()-t,
                "case_calls":_WORKER.case_calls-before,"details":[],
                "error":f"{type(exc).__name__}: {exc}"}


def _key(x)->str:
    return hashlib.sha256(json.dumps([float(v).hex() for v in x]).encode()).hexdigest()


def _append_jsonl(path:Path,row:dict)->None:
    with path.open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(row,allow_nan=False)+"\n")
        stream.flush();os.fsync(stream.fileno())


def _load_existing(path:Path)->dict[str,dict]:
    if not path.exists():return {}
    out={}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row=json.loads(line);out[row["key"]]=row
    return out


def _prepare_data(archive:Path,destination:Path)->Path:
    root=destination/"Ast2018_joint_fit_data"
    if (root/"data/table1_conditions.csv").exists():return root
    destination.mkdir(parents=True,exist_ok=True)
    with ZipFile(archive) as bundle:
        for item in bundle.infolist():
            if item.filename.startswith("Ast2018_joint_fit_data/") and not item.is_dir():
                bundle.extract(item,destination)
    return root


def analytical_search(source:Path,archive:Path,run_dir:Path,*,scenario:str,
                      starts:int=128,workers:int=8,seed:int=2018,
                      nominal_K_step:float=.05,curve_points:int=25):
    if workers<1 or workers>8:raise ValueError("workers must be 1..8")
    run_dir.mkdir(parents=True,exist_ok=True)
    data_dir=_prepare_data(archive,run_dir/"input")
    mods,receipt=load_native(source)
    base,physics,history=historical_start(source,mods[2],mods[3].CommonPhysics)
    assumptions=replace(AstAssumptions(),nominal_K_step_MPa_sqrt_m=nominal_K_step)
    adapter_files=[Path(__file__).parent/name for name in (
        "config.py","data.py","mechanics.py","native.py","reduced.py",
        "objective.py","campaign.py")]
    adapter_sha=hashlib.sha256(b"".join(path.name.encode()+b"\0"+path.read_bytes()
                                        for path in adapter_files)).hexdigest()
    config={"source_receipt":receipt,"historical_start":history,
            "ast_assumptions":asdict(assumptions),"scenario":scenario,
            "optimizer_initialization_seed":seed,"starts":starts,"workers":workers,
            "curve_points":curve_points,"two_dimensional_solver_enabled":False,
            "event_statistic":"unit-action deterministic representative trajectory",
            "parameter_bounds":[{"key":key,"lower":lo,"upper":hi} for key,lo,hi
                                in PARAMETERS],
            "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
            "adapter_sha256":adapter_sha,
            "working_objective":"soft_l1_summed_by_condition_and_family"}
    config_path=run_dir/f"config_{scenario.lower()}.json"
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise RuntimeError("resume configuration changed; choose a new run directory")
    config_path.write_text(json.dumps(config,indent=2,allow_nan=False)+"\n")
    output=run_dir/f"search_{scenario.lower()}.jsonl"
    existing=_load_existing(output)
    sampler=qmc.LatinHypercube(d=7,seed=seed)
    generated=[]
    baseline=np.clip(normalized_start(base),0,1).tolist()
    if physically_admissible(base):generated.append(baseline)
    for batch in range(100):
        if len(generated)>=starts:break
        for sample in sampler.random(max(starts,32)):
            x=sample.tolist()
            if physically_admissible(material_from_normalized(base,x)):
                generated.append(x)
                if len(generated)>=starts:break
    if len(generated)<starts:raise RuntimeError("admissible LHS initialization exhausted")
    pending=[x for x in generated if _key(x) not in existing]
    started=time()
    with ProcessPoolExecutor(max_workers=workers,initializer=_init_worker,
            initargs=(str(source.resolve()),str(data_dir.resolve()),scenario,
                      asdict(assumptions),curve_points)) as pool:
        future_map={pool.submit(_score,x):x for x in pending}
        for future in as_completed(future_map):
            x=future_map[future]
            row=future.result();row.update({"key":_key(x),"scenario":scenario,
                                            "solver":"pinned_v913_coupled_scalar_0d",
                                            "completed_unix_s":time()})
            _append_jsonl(output,row)
            existing[row["key"]]=row
            print(f"{scenario} {len(existing)}/{len(generated)} cost={row['cost']:.6g} "
                  f"seconds={row['runtime_s']:.2f} error={row['error']}",flush=True)
    valid=[r for r in existing.values() if r["error"] is None]
    best=min(valid,key=lambda r:r["cost"]) if valid else None
    summary={"scenario":scenario,"n_requested":starts,"n_valid":len(valid),
             "wall_time_s":time()-started,"best":best,"search_file":str(output)}
    (run_dir/f"search_summary_{scenario.lower()}.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False)+"\n")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source",type=Path,required=True)
    parser.add_argument("--archive",type=Path,default=Path("Ast2018_tungsten_joint_fit_data.zip"))
    parser.add_argument("--run-dir",type=Path,default=Path("runs/ast2018_joint_fit"))
    parser.add_argument("--scenario",choices=("TABLE_PRIMARY","FIGURE5_ALTERNATIVE"),
                        default="TABLE_PRIMARY")
    parser.add_argument("--starts",type=int,default=128)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--seed",type=int,default=2018)
    parser.add_argument("--K-step",type=float,default=.05)
    parser.add_argument("--curve-points",type=int,default=25)
    args=parser.parse_args()
    analytical_search(args.native_source,args.archive,args.run_dir,
                      scenario=args.scenario,starts=args.starts,
                      workers=args.workers,seed=args.seed,
                      nominal_K_step=args.K_step,curve_points=args.curve_points)


if __name__=="__main__":main()
