"""Resume the frozen Ast search with a per-candidate numerical watchdog.

The fitted objective and source adapter remain byte-for-byte unchanged. A
timed-out vector is checkpointed as a numerical rejection, never as a physical
prediction. The original campaign receipt and completed scores are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict,replace
from pathlib import Path
from time import perf_counter, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from . import campaign
from .config import AstAssumptions
from .native import historical_start, load_native
from .objective import PARAMETERS,material_from_normalized, normalized_start, physically_admissible


class _CandidateTimeout(BaseException):
    """Escape native broad ``except Exception`` recovery paths."""


class _CandidateNumericalReject(BaseException):
    """Stop scoring a vector once a required forward case fails numerically."""


def _timeout(_signum, _frame):
    raise _CandidateTimeout("per-candidate analytical objective runtime limit")


def _score_bounded(x, limit_s):
    started=perf_counter()
    before=campaign._WORKER.case_calls
    progress=[]
    def record_progress(cid,status,count):
        progress.append({"condition_id":cid,"status":status,"case_index":count})
        if status.startswith("numerical"):
            raise _CandidateNumericalReject(f"{cid}: {status}")
    campaign._WORKER.progress_callback=record_progress
    old_handler=signal.signal(signal.SIGALRM,_timeout)
    signal.setitimer(signal.ITIMER_REAL,limit_s)
    try:
        row=campaign._score(x)
    except _CandidateTimeout as exc:
        row={"x":x,"cost":1e300,"runtime_s":perf_counter()-started,
             "case_calls":campaign._WORKER.case_calls-before,
             "details":[],"error":f"TimeoutError: {exc}"}
    except _CandidateNumericalReject as exc:
        row={"x":x,"cost":1e300,"runtime_s":perf_counter()-started,
             "case_calls":campaign._WORKER.case_calls-before,
             "details":[],"error":f"NumericalReject: {exc}"}
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_handler)
        campaign._WORKER.progress_callback=None
    row["watchdog_limit_s"]=limit_s
    row["watchdog_timeout"]=bool(row.get("error") and "TimeoutError" in row["error"])
    row["early_numerical_rejection"]=bool(row.get("error") and
                                          "NumericalReject" in row["error"])
    row["completed_condition_progress"]=progress
    return row


def _audit_plot(run_dir,rows,scenario):
    fig,ax=plt.subplots(figsize=(7,4))
    elapsed=np.array([r.get("runtime_s",0) for r in rows],dtype=float)
    cost=np.array([r.get("cost",1e300) for r in rows],dtype=float)
    timed=np.array([bool(r.get("watchdog_timeout")) for r in rows])
    rejected=np.array([bool(r.get("early_numerical_rejection")) for r in rows])
    valid=np.isfinite(cost)&(cost<1e299)&~timed
    ax.scatter(elapsed[valid],cost[valid],s=12,color="#1f77b4",label="completed objective")
    if np.any(timed):
        ax.scatter(elapsed[timed],np.full(np.count_nonzero(timed),
                   max(1.0,float(np.max(cost[valid])) if np.any(valid) else 1.0)),
                   marker="x",color="#d62728",label="watchdog rejection")
    if np.any(rejected):
        ax.scatter(elapsed[rejected],np.full(np.count_nonzero(rejected),
                   max(1.0,float(np.max(cost[valid])) if np.any(valid) else 1.0)),
                   marker="+",color="#d95f02",label="numerical rejection")
    ax.set_xlabel("Candidate runtime (s)")
    ax.set_ylabel("Working cost; timeout markers at top")
    ax.legend(fontsize=8)
    fig.tight_layout()
    root=run_dir/f"search_runtime_audit_{scenario.lower()}"
    fig.savefig(root.with_suffix(".png"),dpi=180)
    fig.savefig(root.with_suffix(".svg"))
    plt.close(fig)


def resume(source:Path, archive:Path, run_dir:Path, *, scenario:str,
           timeout_s:float=900.0,workers:int=8,starts:int=128,
           seed:int=2018,nominal_K_step:float=.05,curve_points:int=25):
    if timeout_s<=0 or workers<1 or workers>8:
        raise ValueError("invalid search watchdog or worker count")
    run_dir.mkdir(parents=True,exist_ok=True)
    data_dir=campaign._prepare_data(archive,run_dir/"input")
    config_path=run_dir/f"config_{scenario.lower()}.json"
    if not config_path.exists():
        mods,receipt=load_native(source)
        base,physics,history=historical_start(source,mods[2],mods[3].CommonPhysics)
        assumptions=replace(AstAssumptions(),nominal_K_step_MPa_sqrt_m=nominal_K_step)
        adapter_files=[Path(campaign.__file__).parent/name for name in
                       ("config.py","data.py","mechanics.py","native.py",
                        "reduced.py","objective.py","campaign.py")]
        adapter_sha=hashlib.sha256(b"".join(p.name.encode()+b"\0"+p.read_bytes()
                                           for p in adapter_files)).hexdigest()
        config={"source_receipt":receipt,"historical_start":history,
                "ast_assumptions":asdict(assumptions),"scenario":scenario,
                "optimizer_initialization_seed":seed,"starts":starts,"workers":workers,
                "curve_points":curve_points,"two_dimensional_solver_enabled":False,
                "event_statistic":"unit-action deterministic representative trajectory",
                "parameter_bounds":[{"key":key,"lower":lo,"upper":hi}
                                    for key,lo,hi in PARAMETERS],
                "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
                "adapter_sha256":adapter_sha,
                "working_objective":"soft_l1_summed_by_condition_and_family"}
        config_path.write_text(json.dumps(config,indent=2,allow_nan=False)+"\n")
    cfg=json.loads(config_path.read_text())
    if (cfg["scenario"]!=scenario or cfg["workers"]!=workers or
        cfg["starts"]!=starts or cfg["optimizer_initialization_seed"]!=seed or
        cfg["curve_points"]!=curve_points or
        cfg["ast_assumptions"]["nominal_K_step_MPa_sqrt_m"]!=nominal_K_step):
        raise RuntimeError("frozen search configuration changed")
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=cfg["archive_sha256"]:
        raise RuntimeError("supplied archive differs from frozen search input")
    _,receipt=load_native(source)
    if receipt["commit"]!=cfg["source_receipt"]["commit"] or \
       receipt["sha256"]!=cfg["source_receipt"]["sha256"]:
        raise RuntimeError("pinned native source changed")
    adapter_files=[Path(campaign.__file__).parent/name for name in
                   ("config.py","data.py","mechanics.py","native.py",
                    "reduced.py","objective.py","campaign.py")]
    adapter_sha=hashlib.sha256(b"".join(p.name.encode()+b"\0"+p.read_bytes()
                                       for p in adapter_files)).hexdigest()
    if adapter_sha!=cfg["adapter_sha256"]:
        raise RuntimeError("frozen analytical adapter changed")
    mods,_=load_native(source)
    base,_,_=historical_start(source,mods[2],mods[3].CommonPhysics)
    generated=[]
    if physically_admissible(base):
        generated.append(np.clip(normalized_start(base),0,1).tolist())
    sampler=qmc.LatinHypercube(d=7,seed=cfg["optimizer_initialization_seed"])
    for _ in range(100):
        if len(generated)>=cfg["starts"]:break
        for sample in sampler.random(max(cfg["starts"],32)):
            x=sample.tolist()
            if physically_admissible(material_from_normalized(base,x)):
                generated.append(x)
                if len(generated)>=cfg["starts"]:break
    if len(generated)!=cfg["starts"]:
        raise RuntimeError("cannot reconstruct frozen search candidates")
    output=run_dir/f"search_{scenario.lower()}.jsonl"
    existing=campaign._load_existing(output)
    recovery=run_dir/f"search_recovery_{scenario.lower()}.json"
    provenance={"original_adapter_sha256":adapter_sha,
                "watchdog_limit_s":timeout_s,"completed_before_resume":len(existing),
                "search_candidates":cfg["starts"],
                "interpretation":"timeout is a numerical rejection; objective and physics unchanged"}
    if recovery.exists():
        prior=json.loads(recovery.read_text())
        for key in ("original_adapter_sha256","watchdog_limit_s",
                    "search_candidates","interpretation"):
            if prior[key]!=provenance[key]:
                raise RuntimeError("search recovery convention changed")
    else:
        temporary=recovery.with_suffix(".tmp")
        temporary.write_text(json.dumps(provenance,indent=2)+"\n")
        os.replace(temporary,recovery)
    pending=[x for x in generated if campaign._key(x) not in existing]
    started=time()
    with ProcessPoolExecutor(max_workers=workers,initializer=campaign._init_worker,
        initargs=(str(source.resolve()),str(data_dir.resolve()),scenario,
                  cfg["ast_assumptions"],cfg["curve_points"])) as pool:
        futures={pool.submit(_score_bounded,x,timeout_s):x for x in pending}
        for future in as_completed(futures):
            x=futures[future];row=future.result()
            row.update({"key":campaign._key(x),"scenario":scenario,
                        "solver":"pinned_v913_coupled_scalar_0d",
                        "stage":"watchdog_resumed_space_filling_search",
                        "completed_unix_s":time()})
            campaign._append_jsonl(output,row)
            existing[row["key"]]=row
            _audit_plot(run_dir,list(existing.values()),scenario)
            print(f"{scenario} {len(existing)}/{len(generated)} "
                  f"cost={row['cost']:.6g} seconds={row['runtime_s']:.1f} "
                  f"error={row['error']}",flush=True)
    valid=[r for r in existing.values() if r["error"] is None]
    best=min(valid,key=lambda r:r["cost"]) if valid else None
    summary={"scenario":scenario,"n_requested":cfg["starts"],
             "n_valid":len(valid),
             "n_watchdog_rejections":sum(r.get("watchdog_timeout",False)
                                         for r in existing.values()),
             "wall_time_after_resume_s":time()-started,"best":best,
             "search_file":str(output),"recovery_receipt":str(recovery)}
    summary_path=run_dir/f"search_summary_{scenario.lower()}.json"
    temporary=summary_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    os.replace(temporary,summary_path)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source",type=Path,required=True)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--scenario",choices=("TABLE_PRIMARY","FIGURE5_ALTERNATIVE"),
                        default="TABLE_PRIMARY")
    parser.add_argument("--timeout-s",type=float,default=900.0)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--starts",type=int,default=128)
    parser.add_argument("--seed",type=int,default=2018)
    args=parser.parse_args()
    resume(args.native_source,args.archive,args.run_dir,scenario=args.scenario,
           timeout_s=args.timeout_s,workers=args.workers,
           starts=args.starts,seed=args.seed)


if __name__=="__main__":main()
