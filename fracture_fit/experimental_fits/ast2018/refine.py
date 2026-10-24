"""Resumable bounded direct refinement of diverse analytical search finalists."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from time import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .campaign import _append_jsonl,_init_worker,_key,_load_existing
from .search_bounded import _score_bounded


def _runtime_plot(run_dir,scenario,rows):
    fig,ax=plt.subplots(figsize=(7,4))
    valid=[r for r in rows if r.get("error") is None]
    rejected=[r for r in rows if r.get("error") is not None]
    if valid:ax.scatter([r["runtime_s"] for r in valid],[r["cost"] for r in valid],
                        s=13,color="#1f77b4",label="completed objective")
    if rejected:ax.scatter([r["runtime_s"] for r in rejected],
                           [max((r["cost"] for r in valid),default=1.0)]*len(rejected),
                           marker="x",color="#d62728",label="numerical rejection")
    ax.set_xlabel("Candidate runtime (s)")
    ax.set_ylabel("Working cost; rejections at top")
    ax.legend(fontsize=8)
    fig.tight_layout()
    root=run_dir/f"refine_runtime_audit_{scenario.lower()}"
    fig.savefig(root.with_suffix(".png"),dpi=180)
    fig.savefig(root.with_suffix(".svg"))
    plt.close(fig)


def diverse_starts(search_rows:list[dict],n:int)->list[dict]:
    eligible=sorted((r for r in search_rows if r.get("error") is None),
                    key=lambda r:r["cost"])
    selected=[]
    for row in eligible:
        x=np.array(row["x"])
        if not selected or min(np.linalg.norm(x-np.array(r["x"])) for r in selected)>=.12:
            selected.append(row)
        if len(selected)>=n:break
    return selected


def _read_search(path:Path)->list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def refine(source:Path,run_dir:Path,scenario:str,*,workers=8,n_starts=8,
           rounds=3,initial_step=.08,warm_vectors:list[list[float]]|None=None,
           timeout_s=900.0):
    if workers<1 or workers>8:raise ValueError("workers must be 1..8")
    cfg=json.loads((run_dir/f"config_{scenario.lower()}.json").read_text())
    search=_read_search(run_dir/f"search_{scenario.lower()}.jsonl")
    starts=diverse_starts(search,n_starts)
    if warm_vectors:
        for i,x in enumerate(warm_vectors):
            starts.insert(i,{"x":x,"cost":1e300,"error":None})
        starts=starts[:n_starts]
    state_path=run_dir/f"refine_state_{scenario.lower()}.json"
    output=run_dir/f"refine_{scenario.lower()}.jsonl"
    evaluated=_load_existing(output)
    evaluated.update({_key(r["x"]):r for r in search})
    if state_path.exists():
        state=json.loads(state_path.read_text())
    else:
        if not starts:raise RuntimeError("no valid analytical finalists")
        state={"scenario":scenario,"round":0,"watchdog_limit_s":timeout_s,
               "starts":[{"x":r["x"],"cost":r["cost"],"step":initial_step}
                         for r in starts],"source":"direct_full_objective_pattern_search"}
    if state.get("watchdog_limit_s")!=timeout_s:
        raise RuntimeError("refinement watchdog convention changed on resume")
    data_dir=run_dir/"input/Ast2018_joint_fit_data"
    with ProcessPoolExecutor(max_workers=workers,initializer=_init_worker,
        initargs=(str(source.resolve()),str(data_dir.resolve()),scenario,
                  cfg["ast_assumptions"],cfg["curve_points"])) as pool:
        warm_pending={_key(start["x"]):start["x"] for start in state["starts"]
                      if start["cost"]>=1e299 and _key(start["x"]) not in evaluated}
        for future in as_completed({pool.submit(_score_bounded,x,timeout_s):key
                                    for key,x in warm_pending.items()}):
            row=future.result();key=_key(row["x"])
            row.update({"key":key,"scenario":scenario,"stage":"warm_center",
                        "completed_unix_s":time()})
            _append_jsonl(output,row);evaluated[key]=row
            _runtime_plot(run_dir,scenario,[r for r in evaluated.values()
                                            if r.get("stage") in
                                            ("warm_center","bounded_pattern_refinement")])
        for start in state["starts"]:
            if start["cost"]>=1e299 and _key(start["x"]) in evaluated:
                start["cost"]=evaluated[_key(start["x"])]["cost"]
        for current_round in range(state["round"],rounds):
            proposals={}
            by_start={}
            for index,start in enumerate(state["starts"]):
                x=np.array(start["x"],dtype=float);step=float(start["step"])
                keys=[]
                for coordinate in range(len(x)):
                    for direction in (-1,1):
                        trial=x.copy();trial[coordinate]=np.clip(trial[coordinate]+direction*step,0,1)
                        key=_key(trial)
                        if key==_key(x):continue
                        proposals[key]=trial.tolist();keys.append(key)
                by_start[index]=keys
            pending={key:x for key,x in proposals.items() if key not in evaluated}
            futures={pool.submit(_score_bounded,x,timeout_s):key
                     for key,x in pending.items()}
            for future in as_completed(futures):
                key=futures[future];row=future.result()
                row.update({"key":key,"scenario":scenario,
                            "stage":"bounded_pattern_refinement",
                            "round":current_round,"completed_unix_s":time()})
                _append_jsonl(output,row);evaluated[key]=row
                _runtime_plot(run_dir,scenario,[r for r in evaluated.values()
                                                if r.get("stage") in
                                                ("warm_center","bounded_pattern_refinement")])
                print(f"{scenario} refinement round={current_round+1} "
                      f"{len(pending)-len(futures)+1}/{len(pending)} "
                      f"cost={row['cost']:.6g} error={row['error']}",flush=True)
                del futures[future]
            for index,start in enumerate(state["starts"]):
                choices=[evaluated[key] for key in by_start[index] if key in evaluated
                         and evaluated[key].get("error") is None]
                if choices:
                    best=min(choices,key=lambda row:row["cost"])
                    if best["cost"]+1e-9<start["cost"]:
                        start["x"]=best["x"];start["cost"]=best["cost"]
                        start["step"]=min(start["step"]*1.1,.2)
                    else:start["step"]*=.5
            state["round"]=current_round+1
            temp=state_path.with_suffix(".tmp")
            temp.write_text(json.dumps(state,indent=2,allow_nan=False)+"\n")
            os.replace(temp,state_path)
            print(f"{scenario} refinement completed round {current_round+1}; "
                  f"best={min(s['cost'] for s in state['starts']):.6g}",flush=True)
    best=min(state["starts"],key=lambda s:s["cost"])
    (run_dir/f"best_{scenario.lower()}.json").write_text(
        json.dumps({"x":best["x"],"cost":best["cost"],"scenario":scenario,
                    "stage":"analytical_pattern_refinement",
                    "rounds":state["round"],"search_config":cfg},
                   indent=2,allow_nan=False)+"\n")
    return best


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,default=Path("runs/ast2018_joint_fit_v1"))
    parser.add_argument("--scenario",choices=("TABLE_PRIMARY","FIGURE5_ALTERNATIVE"),
                        default="TABLE_PRIMARY")
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--starts",type=int,default=8)
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--timeout-s",type=float,default=900.0)
    args=parser.parse_args()
    refine(args.native_source,args.run_dir,args.scenario,workers=args.workers,
           n_starts=args.starts,rounds=args.rounds,timeout_s=args.timeout_s)


if __name__=="__main__":main()
