#!/usr/bin/env python3
"""Fresh second-seed controller for the frozen physical transfer audit."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/physical_slope_transfer_v1"
SOURCE=ROOT/"runs/inverse_fatigue_barrier_design_v1"
BRANCH="codex/v10.2.30-physical-slope-transfer"
PYTHON="/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python"
FAMILY=Path("/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_2_28_kernel_cache/4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/family.json")


def now():return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
def write_json(path,payload):path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")


def preflight(head: str) -> dict:
    if git("branch","--show-current")!=BRANCH or git("rev-parse","HEAD")!=head:
        raise SystemExit("branch/HEAD preflight mismatch")
    if git("status","--porcelain"):
        raise SystemExit("physical seed check requires a clean worktree")
    freeze=json.loads((OUT/"transfer_freeze.json").read_text())
    if freeze["git_head"]!=head or freeze["new_physics_runs_before_freeze"]!=0:
        raise SystemExit("transfer operator was not frozen at this HEAD")
    for name,digest in freeze["analysis_artifact_hashes"].items():
        if sha(OUT/name)!=digest:raise SystemExit(f"frozen transfer artifact changed: {name}")
    if not FAMILY.is_file():raise SystemExit("qualified kernel family is missing")
    return freeze


def initialize(head: str) -> None:
    preflight(head)
    rows=[]
    for K in (12.0,18.0,24.3):
        path=OUT/"physical"/"second_seed"/"INV_OPENING_M4_V1"/f"K_{K:g}_R_0.1_seed_1001723"
        rows.append({"job_id":f"M4_SECOND_SEED_K{K:g}","stage":"SECOND_SEED","option_key":"INV_OPENING_M4_V1",
                     "Kmax_MPa_sqrt_m":K,"DeltaK_MPa_sqrt_m":.9*K,"R":.1,"seed":1001723,
                     "n_bins":80,"target_extension_um":100.,"cycles_max":1e12,"fresh":True,"resume":False,
                     "solver_head":head,"result_path":str(path.resolve()),"status":"PENDING","exit_code":"","wall_seconds":""})
    pd.DataFrame(rows).to_csv(OUT/"physical_transfer_job_registry.csv",index=False)
    write_json(OUT/"physical_transfer_controller_state.json",{"schema":"v10.2.30_physical_transfer_controller_v1",
      "phase":"SECOND_SEED_PENDING","expected_branch":BRANCH,"expected_head":head,"active_worker_count":0,
      "job_count":3,"created_utc":now()})
    print(json.dumps({"result":"PASS","jobs":3,"head":head}))


def run_one(job: dict, head: str) -> dict:
    job=dict(job);path=Path(job["result_path"])
    if path.exists():job["status"]="INVALID_DUPLICATE_OUTPUT_PATH";return job
    env=os.environ.copy();env.update({
      "PYTHON_BIN":PYTHON,"CONDA_ENV":"arrhenius-sharp-front-v10-codex","CONDA_DEFAULT_ENV":"arrhenius-sharp-front-v10-codex",
      "EXPECTED_BRANCH":BRANCH,"EXPECTED_HEAD":head,"FAMILY_JSON":str(FAMILY),
      "V10230_ENTRY_MODULE":"arrhenius_fracture.sharp_front_v10_2_30_candidate_fixed_deltaK",
      "V10230_CANDIDATE_REGISTRY":str((SOURCE/"inverse_design_candidate_registry.csv").resolve()),
      "V10230_CANDIDATE_SELECTION":str((SOURCE/"inverse_design_candidate_selection.json").resolve()),
      "V10230_HIGH_CYCLE_EXPLICIT_ONLY":"1","V10230_SAVE_ACTIVE_STATE_SNAPSHOT":"1",
      "PARAMETER_OPTION":"INV_OPENING_M4_V1","TARGET_DELTAK":str(job["DeltaK_MPa_sqrt_m"]),"R_RATIO":"0.1",
      "TARGET_EXT_UM":"100","CYCLES_MAX":"1000000000000","HAZARD_SEED":"1001723","MAX_WALL_SECONDS":"43200",
      "TARGET_FRACTION":"physical_slope_transfer_second_seed","RUN_LABEL":job["job_id"],"OUTROOT":str(path),
    });env.pop("V10230_RESTART_CHECKPOINT_DIR",None)
    start=time.time();result=subprocess.run(["bash","scripts/run_v10_2_30_weakt_high_cycle_1e12.sh"],cwd=ROOT,env=env)
    job["exit_code"]=result.returncode;job["wall_seconds"]=time.time()-start
    summary=path/"developed_fatigue_growth_summary.json"
    if result.returncode==0 and summary.is_file():
        data=json.loads(summary.read_text());job["status"]="COMPLETE" if data.get("target_reached") else "INVALID_NONTERMINAL"
    elif (path/"high_cycle_live_checkpoint.json").is_file():job["status"]="NUMERICAL_NONTERMINAL"
    else:job["status"]="NUMERICAL_FAILURE"
    return job


def run_seed(head: str, workers: int) -> None:
    freeze=preflight(head);path=OUT/"physical_transfer_job_registry.csv"
    rows=pd.read_csv(path,keep_default_na=False).to_dict("records")
    pending=[row for row in rows if row["stage"]=="SECOND_SEED" and row["status"]=="PENDING"]
    if len(pending)!=3:raise SystemExit(f"expected three pending second-seed jobs; found {len(pending)}")
    if freeze["physics_launch_utc"] is None:
        freeze["physics_launch_utc"]=now();write_json(OUT/"transfer_freeze.json",freeze)
    state=json.loads((OUT/"physical_transfer_controller_state.json").read_text())
    state.update({"phase":"SECOND_SEED_RUNNING","active_worker_count":min(workers,3),"last_update_utc":now()});write_json(OUT/"physical_transfer_controller_state.json",state)
    byid={row["job_id"]:row for row in rows}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_one,row,head):row for row in pending}
        for future in as_completed(futures):
            result=future.result();byid[result["job_id"]]=result
            pd.DataFrame(list(byid.values())).to_csv(path,index=False)
            print(json.dumps({"job":result["job_id"],"status":result["status"],"seconds":result["wall_seconds"]}),flush=True)
    state.update({"phase":"SECOND_SEED_TERMINAL","active_worker_count":0,"last_update_utc":now(),
                  "status_counts":pd.Series([row["status"] for row in byid.values()]).value_counts().to_dict()})
    write_json(OUT/"physical_transfer_controller_state.json",state)
    if any(row["status"]!="COMPLETE" for row in byid.values()):raise SystemExit("second-seed check did not complete")


def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=["seed-init","seed-run"]);p.add_argument("--expected-head",required=True);p.add_argument("--workers",type=int,default=3);a=p.parse_args()
    initialize(a.expected_head) if a.command=="seed-init" else run_seed(a.expected_head,a.workers)
    return 0


if __name__=="__main__":raise SystemExit(main())
