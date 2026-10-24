"""Conditional local 1-D refinement after unchanged-parameter confirmation."""
from __future__ import annotations

import argparse
import json
import os
import signal
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from time import perf_counter,time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .campaign import _append_jsonl,_key,_load_existing
from .config import AstAssumptions
from .data import AstData
from .native import historical_start,load_native
from .objective import WorkingObjective,material_from_normalized
from .spatial import predict_spatial


PRIORITY11=("Tp21C_rate0p2","Tp100C_rate0p2","Tp250C_rate0p2",
            "Tp21C_rate0p5","Tp100C_rate0p5","Tp250C_rate0p5",
            "Tp100C_rate1p6","Tp500C_rate1p6","Tm90C_rate0p2",
            "Tp0C_rate0p05","Tp75C_rate1p6")
_CTX=None


def _init(source_dir,data_dir,scenario,assumptions):
    global _CTX
    source=Path(source_dir);mods,receipt=load_native(source)
    base,physics,_=historical_start(source,mods[2],mods[3].CommonPhysics)
    data=AstData.load(Path(data_dir))
    objective=WorkingObjective(data,scenario,AstAssumptions(**assumptions),
                               mods[1],base,physics,predictor=predict_spatial)
    objective.condition_ids=PRIORITY11
    objective.n_residuals=sum(len(f.values) for cid in PRIORITY11
                              for f in objective.families.get(cid,()))
    _CTX=objective


def _timeout(_signum,_frame):
    raise TimeoutError("per-candidate spatial local objective runtime limit")


def _runtime_plot(run_dir,scenario,rows):
    fig,ax=plt.subplots(figsize=(7,4))
    valid=[r for r in rows if r.get("error") is None]
    rejected=[r for r in rows if r.get("error") is not None]
    if valid:ax.scatter([r["runtime_s"] for r in valid],[r["cost"] for r in valid],
                        s=13,color="#1f77b4",label="completed direct 1-D objective")
    if rejected:ax.scatter([r["runtime_s"] for r in rejected],
                           [max((r["cost"] for r in valid),default=1.0)]*len(rejected),
                           marker="x",color="#d62728",label="numerical rejection")
    ax.set_xlabel("Priority-11 candidate runtime (s)")
    ax.set_ylabel("Working cost; rejections at top")
    ax.legend(fontsize=8)
    fig.tight_layout()
    root=run_dir/f"spatial_local_runtime_audit_{scenario.lower()}"
    fig.savefig(root.with_suffix(".png"),dpi=180)
    fig.savefig(root.with_suffix(".svg"))
    plt.close(fig)


def _score(x,timeout_s=3600.0):
    t=perf_counter()
    before=_CTX.case_calls
    progress=[]
    _CTX.progress_callback=lambda cid,status,count:progress.append(
        {"condition_id":cid,"status":status,"case_index":count})
    old_handler=signal.signal(signal.SIGALRM,_timeout)
    signal.setitimer(signal.ITIMER_REAL,timeout_s)
    try:
        residual,detail=_CTX.evaluate(np.array(x),detail=True)
        return {"x":x,"cost":float(.5*np.sum(residual**2)),
                "details":detail,"runtime_s":perf_counter()-t,
                "case_calls":_CTX.case_calls-before,"error":None,
                "watchdog_timeout":False,"completed_condition_progress":progress}
    except Exception as exc:
        return {"x":x,"cost":1e300,"details":[],"runtime_s":perf_counter()-t,
                "case_calls":_CTX.case_calls-before,
                "error":f"{type(exc).__name__}: {exc}",
                "watchdog_timeout":isinstance(exc,TimeoutError),
                "completed_condition_progress":progress}
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_handler)
        _CTX.progress_callback=None


def local_refine(source:Path,run_dir:Path,scenario:str,start_vector:Path,
                 *,indices=(0,3,6),step=.04,rounds=3,workers=8,
                 timeout_s=3600.0):
    cfg=json.loads((run_dir/f"config_{scenario.lower()}.json").read_text())
    data_dir=run_dir/"input/Ast2018_joint_fit_data"
    x=json.loads(start_vector.read_text())["x"]
    output=run_dir/f"spatial_local_{scenario.lower()}.jsonl"
    state_path=run_dir/f"spatial_local_state_{scenario.lower()}.json"
    state=(json.loads(state_path.read_text()) if state_path.exists() else
           {"round":0,"x":x,"cost":1e300,"step":step,
            "watchdog_limit_s":timeout_s,
            "indices":list(indices),"source_analytical_vector":str(start_vector),
            "condition_ids":list(PRIORITY11)})
    if state.get("watchdog_limit_s")!=timeout_s:
        raise RuntimeError("spatial local watchdog convention changed")
    evaluated=_load_existing(output)
    with ProcessPoolExecutor(max_workers=workers,initializer=_init,
        initargs=(str(source.resolve()),str(data_dir.resolve()),scenario,
                  cfg["ast_assumptions"])) as pool:
        center_key=_key(state["x"])
        if center_key not in evaluated:
            row=pool.submit(_score,state["x"],timeout_s).result()
            row.update({"key":center_key,"stage":"unchanged_material_1d_confirmation",
                        "completed_unix_s":time()})
            _append_jsonl(output,row);evaluated[center_key]=row
            _runtime_plot(run_dir,scenario,list(evaluated.values()))
        state["cost"]=evaluated[center_key]["cost"]
        for current_round in range(state["round"],rounds):
            proposals={}
            for j in indices:
                for sign in (-1,1):
                    trial=np.array(state["x"])
                    trial[j]=np.clip(trial[j]+sign*state["step"],0,1)
                    proposals[_key(trial)]=trial.tolist()
            pending={key:v for key,v in proposals.items() if key not in evaluated}
            futures={pool.submit(_score,v,timeout_s):key for key,v in pending.items()}
            for future in as_completed(futures):
                key=futures[future];row=future.result()
                row.update({"key":key,"stage":"local_direct_1d",
                            "round":current_round,"completed_unix_s":time()})
                _append_jsonl(output,row);evaluated[key]=row
                _runtime_plot(run_dir,scenario,list(evaluated.values()))
                print(f"1-D local round {current_round+1} "
                      f"{len([k for k in proposals if k in evaluated])}/{len(proposals)} "
                      f"cost={row['cost']:.5g} error={row['error']}",flush=True)
            candidates=[evaluated[key] for key in proposals if key in evaluated
                        and evaluated[key].get("error") is None]
            if candidates:
                best=min(candidates,key=lambda r:r["cost"])
                if best["cost"]+1e-9<state["cost"]:
                    state["x"]=best["x"];state["cost"]=best["cost"]
                else:state["step"]*=.5
            state["round"]=current_round+1
            temp=state_path.with_suffix(".tmp")
            temp.write_text(json.dumps(state,indent=2,allow_nan=False)+"\n")
            os.replace(temp,state_path)
    best=run_dir/f"best_1d_refined_{scenario.lower()}.json"
    best.write_text(json.dumps({"x":state["x"],"cost_priority11":state["cost"],
        "scenario":scenario,"stage":"local_1d_refinement",
        "source_analytical_vector":str(start_vector),
        "condition_ids":list(PRIORITY11),"rounds":state["round"]},
        indent=2,allow_nan=False)+"\n")
    return best


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,default=Path("runs/ast2018_joint_fit_v1"))
    parser.add_argument("--scenario",choices=("TABLE_PRIMARY","FIGURE5_ALTERNATIVE"),
                        default="TABLE_PRIMARY")
    parser.add_argument("--vector",type=Path,required=True)
    parser.add_argument("--indices",type=int,nargs="+",default=(0,3,6))
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--timeout-s",type=float,default=3600.0)
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    local_refine(args.native_source,args.run_dir,args.scenario,args.vector,
                 indices=tuple(args.indices),rounds=args.rounds,workers=args.workers,
                 timeout_s=args.timeout_s)


if __name__=="__main__":main()
