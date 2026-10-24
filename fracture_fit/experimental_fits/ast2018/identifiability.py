"""Working-scale sensitivities and conditional local barrier profiles."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

from . import campaign
from .objective import PARAMETERS


def _residual_task(x):
    t=perf_counter()
    before=campaign._WORKER.case_calls
    vector,detail=campaign._WORKER.evaluate(np.array(x),detail=True)
    return {"x":x,"residual":vector.tolist(),"detail":detail,
            "runtime_s":perf_counter()-t,
            "case_calls":campaign._WORKER.case_calls-before}


def _profile_task(spec):
    """One fixed-coordinate profile point with bounded coordinate reoptimization."""
    index,value,x0,step,passes=spec
    start=perf_counter()
    before=campaign._WORKER.case_calls
    x=np.array(x0,dtype=float);x[index]=value
    best=float(.5*np.sum(campaign._WORKER.evaluate(x)**2))
    calls=1
    for _ in range(passes):
        proposals=[]
        for j in range(len(x)):
            if j==index:continue
            for sign in (-1,1):
                trial=x.copy();trial[j]=np.clip(trial[j]+sign*step,0,1)
                proposals.append(trial)
        scores=[]
        for trial in proposals:
            cost=float(.5*np.sum(campaign._WORKER.evaluate(trial)**2));calls+=1
            scores.append(cost)
        choice=int(np.argmin(scores))
        if scores[choice]<best:
            best=scores[choice];x=proposals[choice]
        else:step*=.5
    return {"parameter_key":PARAMETERS[index][0],"fixed_normalized":value,
            "cost":best,"optimized_x":x.tolist(),"forward_objective_calls":calls,
            "forward_case_calls":campaign._WORKER.case_calls-before,
            "runtime_s":perf_counter()-start,
            "profile_method":"bounded_coordinate_reoptimization",
            "passes":passes,"final_step":step}


def analyze(source:Path,run_dir:Path,scenario:str,vector_path:Path,
            *,workers=8,step=.01,profile_indices=(0,1,3,6),
            profile_offsets=(-.1,0,.1),profile_passes=2):
    cfg=json.loads((run_dir/f"config_{scenario.lower()}.json").read_text())
    x=np.array(json.loads(vector_path.read_text())["x"],dtype=float)
    data_dir=run_dir/"input/Ast2018_joint_fit_data"
    initargs=(str(source.resolve()),str(data_dir.resolve()),scenario,
              cfg["ast_assumptions"],cfg["curve_points"])
    plusminus=[]
    for i in range(len(x)):
        xp=x.copy();xm=x.copy()
        xp[i]=min(xp[i]+step,1);xm[i]=max(xm[i]-step,0)
        plusminus.extend([xp.tolist(),xm.tolist()])
    outdir=run_dir/f"identifiability_{scenario.lower()}";outdir.mkdir(parents=True,exist_ok=True)
    basis={"scenario":scenario,"vector":x.tolist(),"normalized_step":step,
           "profile_indices":list(profile_indices),
           "profile_offsets":list(profile_offsets),
           "profile_passes":profile_passes,
           "adapter_sha256":cfg["adapter_sha256"]}
    basis_path=outdir/"checkpoint_basis.json"
    if basis_path.exists() and json.loads(basis_path.read_text())!=basis:
        raise RuntimeError("identifiability checkpoint basis changed")
    basis_path.write_text(json.dumps(basis,indent=2)+"\n")
    sensitivity_checkpoint=outdir/"sensitivity_calls.jsonl"
    sensitivity_done=campaign._load_existing(sensitivity_checkpoint)
    outputs=[sensitivity_done.get(str(i)) for i in range(len(plusminus))]
    with ProcessPoolExecutor(max_workers=workers,initializer=campaign._init_worker,
                             initargs=initargs) as pool:
        tasks={pool.submit(_residual_task,v):i for i,v in enumerate(plusminus)
               if outputs[i] is None}
        for future in as_completed(tasks):
            i=tasks[future];outputs[i]=future.result()
            outputs[i]["key"]=str(i)
            campaign._append_jsonl(sensitivity_checkpoint,outputs[i])
            print(f"sensitivity {sum(o is not None for o in outputs)}/{len(outputs)}",flush=True)
    J=np.column_stack([(np.array(outputs[2*i]["residual"])-
                        np.array(outputs[2*i+1]["residual"])) /
                       (plusminus[2*i][i]-plusminus[2*i+1][i])
                       for i in range(len(x))])
    U,S,Vt=np.linalg.svd(J,full_matrices=False)
    diag={"scenario":scenario,"vector":x.tolist(),
          "parameter_keys":[p[0] for p in PARAMETERS],
          "normalized_step":step,"singular_values":S.tolist(),
          "right_singular_vectors":Vt.tolist(),
          "condition_number":float(S[0]/S[-1]) if S[-1]>0 else None,
          "smallest_direction":Vt[-1].tolist(),
          "interpretation":"working-scale local sensitivity; not covariance or confidence interval",
          "forward_objective_calls":len(outputs),
          "case_calls":sum(o["case_calls"] for o in outputs),
          "sum_process_runtime_s":sum(o["runtime_s"] for o in outputs)}
    (outdir/"sensitivity.json").write_text(json.dumps(diag,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(outdir/"scaled_jacobian.npz",jacobian=J,
                        singular_values=S,right_singular_vectors=Vt)
    specs=[]
    for index in profile_indices:
        for offset in profile_offsets:
            value=float(np.clip(x[index]+offset,0,1))
            specs.append((index,value,x.tolist(),.05,profile_passes))
    profile_checkpoint=outdir/"profile_calls.jsonl"
    profile_done=campaign._load_existing(profile_checkpoint)
    profiles=list(profile_done.values())
    with ProcessPoolExecutor(max_workers=workers,initializer=campaign._init_worker,
                             initargs=initargs) as pool:
        futures={pool.submit(_profile_task,spec):f"{spec[0]}:{spec[1]:.17g}"
                 for spec in specs if f"{spec[0]}:{spec[1]:.17g}" not in profile_done}
        for future in as_completed(futures):
            row=future.result();profiles.append(row)
            row["key"]=futures[future]
            campaign._append_jsonl(profile_checkpoint,row)
            print(f"profile {len(profiles)}/{len(specs)} "
                  f"{row['parameter_key']}={row['fixed_normalized']:.3f} "
                  f"cost={row['cost']:.5g}",flush=True)
    profiles.sort(key=lambda r:(r["parameter_key"],r["fixed_normalized"]))
    (outdir/"profiles.json").write_text(json.dumps(profiles,indent=2,allow_nan=False)+"\n")
    return diag,profiles


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
