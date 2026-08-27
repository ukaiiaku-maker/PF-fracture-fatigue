#!/usr/bin/env python3
"""Conditional PT overlay anchors required by the fixed-load endpoint domain."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arrhenius_fracture.two_scale_virtual_ct_v10230 import LogPchipRateSurface

PYTHON = "/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python"
BRANCH = "codex/v10.2.30-two-scale-virtual-CT"
ROOT = Path("runs/A_native_two_scale_virtual_CT_v1")
SOURCE = Path("runs/A_native_PT03_PT08_R_nominal_deltaK_v1")
FAMILY = "/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_2_28_kernel_cache/4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/family.json"
PT03 = "A_PT_03_oneD_v2_dbtt_TP_4895f9e5b44deea5"
PT08 = "A_PT_08_oneD_v2_dbtt_TP_f2817e7998cb7be6"
RS = (-0.95, 0.1, 0.5)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");os.replace(temporary,path)


def atomic_csv(path: Path, rows) -> None:
    temporary=path.with_suffix(path.suffix+".tmp");pd.DataFrame(rows).to_csv(temporary,index=False);os.replace(temporary,path)


def git(*arguments: str) -> str:
    return subprocess.check_output(["git",*arguments],text=True).strip()


def alive(pid) -> bool:
    try:os.kill(int(pid),0);return True
    except (ProcessLookupError,PermissionError,TypeError,ValueError):return False


def option_rows(option: str) -> list[dict]:
    frame=pd.read_csv(SOURCE/"A_PT03_PT08_R_developed_points.csv")
    frame=frame[(frame.option==option)&frame.stage.str.contains("PRIMARY")&(frame.seed==1720)]
    return frame.sort_values(["R","Kmax_MPa_sqrt_m"]).to_dict("records")


def initialize(root: Path) -> list[dict]:
    native=pd.read_csv(root/"two_scale_anchor_job_registry.csv")
    if len(native)!=9 or not (native.status=="PHYSICAL_TARGET_REACHED").all():
        raise RuntimeError("conditional PT anchors cannot precede nine terminal native anchors")
    folder=root/"PT_overlay_anchors";folder.mkdir(exist_ok=False)
    for name in ("job_contracts","worker_terminals","controller_logs","results"):(folder/name).mkdir()
    predictions=[];frozen_ns=time.time_ns()
    conditions=[("FIRST_CONDITIONAL",PT03,.5,21.0),("FIRST_CONDITIONAL",PT08,-.95,21.0)]
    conditions += [("ENDPOINT_DOMAIN_RESOLUTION",option,R,24.3) for option in (PT03,PT08) for R in RS]
    for stage,option,R,K in conditions:
        surface=LogPchipRateSurface(option_rows(option),option=option,version="v0_overlay")
        prediction=surface.prospective_anchor_prediction(K,R)
        prediction.update({"stage":stage,"option":option,"prediction_frozen_unix_ns":frozen_ns,
          "trigger":"FIXED_LOAD_ENDPOINT_DOMAIN_AMBIGUITY" if stage=="ENDPOINT_DOMAIN_RESOLUTION" else "LARGEST_PT_EFFECT_INTERVAL_CHECK",
          "result_read_before_prediction":False,"result_path_existed_when_predicted":False})
        predictions.append(prediction)
    atomic_csv(folder/"PT_conditional_anchor_predictions_v0.csv",predictions)
    prediction_sha=sha256(folder/"PT_conditional_anchor_predictions_v0.csv")
    jobs=[]
    for stage,option,R,K in conditions:
        label="PT03" if option==PT03 else "PT08";job_id=f"{stage.lower()}__{label}__R{R:g}__Kmax{K:g}__seed1720"
        jobs.append({"job_id":job_id,"stage":stage,"option":option,"R":R,"Kmax_MPa_sqrt_m":K,
          "deltaK_driver_MPa_sqrt_m":(1-R)*K,"seed":1720,"n_bins":80,"target_extension_um":100,"cycles_max":1_000_000,
          "result_path":str((folder/"results"/job_id).resolve()),"status":"PENDING","attempt":0,"pid":None,
          "resumed":False,"reused":False,"acceleration_mode":"explicit_only","prediction_file_sha256":prediction_sha,
          "prediction_frozen_unix_ns":frozen_ns,"contract_path":"","terminal_path":"","log_path":""})
    atomic_csv(folder/"PT_anchor_job_registry.csv",jobs)
    atomic_json(folder/"PT_anchor_trigger.json",{"schema":"PT_anchor_trigger_v1",
      "first_conditional_reason":"largest predeclared PT-effect regimes are checked first",
      "endpoint_resolution_reason":"fixed-load Kmax reaches 24.255719738394 while frozen PT domains end at 24; fail-closed integration requires physical endpoint rows",
      "automatic_full_PT_panel":False,"conditions":conditions,"prediction_sha256":prediction_sha})
    return jobs


def contract(job: dict, folder: Path, head: str) -> Path:
    output=Path(job["result_path"])
    if output.exists():raise RuntimeError(f"fresh PT result path exists: {output}")
    attempt=int(job["attempt"])+1;log=(folder/"controller_logs"/f"{job['job_id']}__attempt{attempt}.log").resolve();terminal=(folder/"worker_terminals"/f"{job['job_id']}__attempt{attempt}.json").resolve();path=folder/"job_contracts"/f"{job['job_id']}__attempt{attempt}.json"
    environment={"PYTHON_BIN":PYTHON,"CONDA_ENV":"arrhenius-sharp-front-v10-codex","CONDA_DEFAULT_ENV":"arrhenius-sharp-front-v10-codex",
      "EXPECTED_BRANCH":BRANCH,"EXPECTED_HEAD":head,"FAMILY_JSON":FAMILY,"V10230_ENTRY_MODULE":"arrhenius_fracture.sharp_front_v10_2_30_candidate_fixed_deltaK",
      "V10230_CANDIDATE_REGISTRY":str((SOURCE/"A_PT03_PT08_R_registry.csv").resolve()),"V10230_CANDIDATE_SELECTION":str((SOURCE/"A_PT03_PT08_R_selection.json").resolve()),
      "PARAMETER_OPTION":job["option"],"TARGET_DELTAK":f"{job['deltaK_driver_MPa_sqrt_m']:.17g}","R_RATIO":f"{job['R']:.17g}","TARGET_FRACTION":job["stage"],
      "RUN_LABEL":job["job_id"],"TARGET_EXT_UM":"100","CYCLES_MAX":"1000000","HAZARD_SEED":"1720","MAX_WALL_SECONDS":"43200",
      "OUTROOT":job["result_path"],"V10230_HIGH_CYCLE_EXPLICIT_ONLY":"1"}
    payload={"schema":"two_scale_PT_anchor_contract_v1","job_id":job["job_id"],"attempt":attempt,"created_unix_ns":time.time_ns(),
      "result_path":job["result_path"],"log_path":str(log),"terminal_path":str(terminal),"launch_head":head,
      "qualified_solver_head":"94871be15702e7fb85116b92af62c1226c61be42","production_solver_sha256":"c15a957161e8cdb43bf944243d8a18a64839dbfe0518aa2d1c5f0c93773b817b",
      "prediction_file_sha256":job["prediction_file_sha256"],"prediction_frozen_unix_ns":int(job["prediction_frozen_unix_ns"]),
      "fresh_virgin_start":True,"resume":False,"environment":environment}
    atomic_json(path,payload);job.update({"attempt":attempt,"contract_path":str(path.resolve()),"terminal_path":str(terminal),"log_path":str(log)});return path


def valid(job: dict) -> bool:
    path=Path(job["result_path"])/"developed_fatigue_growth_summary.json"
    if not path.is_file():return False
    data=json.loads(path.read_text());return bool(data.get("target_reached")) and bool(data.get("stable_growth_provisional")) and int(data.get("event_count",0))>=10 and int(data.get("restart_count") or 0)==0


def reconcile(folder: Path,jobs:list[dict])->None:
    changed=False
    for job in jobs:
        if job["status"] not in {"RUNNING","INVALID_OR_NONTERMINAL"} or (job["status"]=="RUNNING" and alive(job.get("pid"))):continue
        output=Path(job["result_path"]);physical=any((output/name).is_file() for name in ("kinetic_tip_cell_audit_v101.json","high_cycle_live_checkpoint.json"))
        if physical:raise RuntimeError(f"interrupted conditional PT physics cannot resume: {job['job_id']}")
        if output.exists():
            quarantine=folder/"quarantine"/f"{job['job_id']}__attempt{int(job['attempt'])}__prephysics";quarantine.parent.mkdir(exist_ok=True);os.replace(output,quarantine)
        job.update({"status":"PENDING","pid":None,"exit_code":None,"wall_seconds":None});changed=True
    if changed:atomic_csv(folder/"PT_anchor_job_registry.csv",jobs)


def run_stage(folder:Path,jobs:list[dict],stage:str,head:str,workers:int)->None:
    registry=folder/"PT_anchor_job_registry.csv"
    while True:
        for job in [x for x in jobs if x["stage"]==stage and x["status"]=="RUNNING"]:
            terminal=Path(job["terminal_path"])
            if terminal.is_file() or not alive(job["pid"]):
                data=json.loads(terminal.read_text()) if terminal.is_file() else {};job["exit_code"]=data.get("exit_code");job["wall_seconds"]=data.get("wall_seconds");job["pid"]=None;job["status"]="PHYSICAL_TARGET_REACHED" if valid(job) else "INVALID_OR_NONTERMINAL";atomic_csv(registry,jobs)
                if job["status"]!="PHYSICAL_TARGET_REACHED":raise RuntimeError(f"conditional PT anchor failed: {job['job_id']}")
        active=[x for x in jobs if x["stage"]==stage and x["status"]=="RUNNING"];pending=[x for x in jobs if x["stage"]==stage and x["status"]=="PENDING"]
        while pending and len(active)<workers:
            job=pending.pop(0);path=contract(job,folder,head);process=subprocess.Popen([PYTHON,"scripts/run_v10_2_30_two_scale_anchor_worker.py","--contract",str(path)],start_new_session=True);job["pid"]=process.pid;job["status"]="RUNNING";active.append(job);atomic_csv(registry,jobs)
        if not active and not pending:return
        time.sleep(5)


def validate_first(folder:Path,jobs:list[dict])->None:
    predictions=pd.read_csv(folder/"PT_conditional_anchor_predictions_v0.csv");rows=[]
    for job in [x for x in jobs if x["stage"]=="FIRST_CONDITIONAL"]:
        summary=json.loads((Path(job["result_path"])/"developed_fatigue_growth_summary.json").read_text());rate=float(summary["developed_interval"]["da_dN"]);prediction=predictions[(predictions.option==job["option"])&np.isclose(predictions.R,job["R"])&np.isclose(predictions.Kmax_MPa_sqrt_m,job["Kmax_MPa_sqrt_m"])].iloc[0];error=math.log10(float(prediction.predicted_da_dN)/rate);rows.append({"job_id":job["job_id"],"option":job["option"],"R":job["R"],"Kmax_MPa_sqrt_m":job["Kmax_MPa_sqrt_m"],"predicted_da_dN":prediction.predicted_da_dN,"physical_da_dN":rate,"epsilon_log_decade":error,"abs_epsilon_log_decade":abs(error),"passes_0p05_decade":abs(error)<=.05})
    atomic_csv(folder/"PT_first_conditional_validation.csv",rows)
    if not all(x["passes_0p05_decade"] for x in rows):raise RuntimeError("PT first conditional interpolation requires midpoint refinement")


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=ROOT);parser.add_argument("--workers",type=int,default=3);args=parser.parse_args()
    if not 1<=args.workers<=3:raise SystemExit("workers must be 1..3")
    if git("branch","--show-current")!=BRANCH or git("status","--short"):raise SystemExit("conditional PT controller requires clean worktree")
    root=args.root.resolve();folder=root/"PT_overlay_anchors";jobs=pd.read_csv(folder/"PT_anchor_job_registry.csv").to_dict("records") if (folder/"PT_anchor_job_registry.csv").is_file() else initialize(root);reconcile(folder,jobs);head=git("rev-parse","HEAD")
    atomic_json(folder/"PT_controller_state.json",{"phase":"FIRST_CONDITIONAL","pid":os.getpid(),"head":head});run_stage(folder,jobs,"FIRST_CONDITIONAL",head,args.workers);validate_first(folder,jobs)
    atomic_json(folder/"PT_controller_state.json",{"phase":"ENDPOINT_DOMAIN_RESOLUTION","pid":os.getpid(),"head":head});run_stage(folder,jobs,"ENDPOINT_DOMAIN_RESOLUTION",head,args.workers)
    atomic_json(folder/"PT_controller_state.json",{"phase":"COMPLETE","result":"PASS","pid":os.getpid(),"head":head});return 0


if __name__=="__main__":raise SystemExit(main())
