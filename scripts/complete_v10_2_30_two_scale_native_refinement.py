#!/usr/bin/env python3
"""Fresh-only low-K refinement required by failed v0 13.5 anchor predictions."""
from __future__ import annotations
import argparse, hashlib, json, os, math, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from arrhenius_fracture.two_scale_virtual_ct_v10230 import LogPchipRateSurface

PYTHON="/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python";BRANCH="codex/v10.2.30-two-scale-virtual-CT"
ROOT=Path("runs/A_native_two_scale_virtual_CT_v1");SOURCE=Path("runs/A_native_PT03_PT08_R_nominal_deltaK_v1")
FAMILY="/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_2_28_kernel_cache/4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/family.json"

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic_json(path,value):
 temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");os.replace(temporary,path)
def atomic_csv(path,rows):
 temporary=path.with_suffix(path.suffix+".tmp");pd.DataFrame(rows).to_csv(temporary,index=False);os.replace(temporary,path)
def git(*args):return subprocess.check_output(["git",*args],text=True).strip()
def alive(pid):
 try:os.kill(int(pid),0);return True
 except (ProcessLookupError,PermissionError,TypeError,ValueError):return False

def initialize(root):
 validation=pd.read_csv(root/"A_NATIVE_anchor_validation.csv")
 failures=validation[(validation.classification=="INTERPOLATION_FAILURE")&np.isclose(validation.Kmax_MPa_sqrt_m,13.5)]
 if len(failures)!=3:raise RuntimeError("expected the three documented low-K interpolation failures")
 folder=root/"A_NATIVE_low_K_refinement";folder.mkdir(exist_ok=False)
 for name in ("results","job_contracts","worker_terminals","controller_logs"):(folder/name).mkdir()
 source=pd.read_csv(SOURCE/"A_PT03_PT08_R_developed_points.csv");source=source[(source.option=="A_NATIVE")&source.stage.str.contains("PRIMARY")&(source.seed==1720)]
 anchors=pd.read_csv(root/"A_NATIVE_anchor_results.csv");surface=LogPchipRateSurface(source.to_dict("records")+anchors[["R","Kmax_MPa_sqrt_m","developed_da_dN"]].to_dict("records"),option="A_NATIVE",version="v1_refinement_predictor")
 frozen=time.time_ns();predictions=[];jobs=[]
 for R in (-.95,.1,.5):
  p=surface.prospective_anchor_prediction(12.75,R);p.update({"option":"A_NATIVE","surface_version":"v1_refinement_predictor","prediction_frozen_unix_ns":frozen,"trigger":"V0_K13p5_INTERPOLATION_FAILURE","result_read_before_prediction":False,"result_path_existed_when_predicted":False});predictions.append(p)
 atomic_csv(folder/"A_NATIVE_refinement_predictions_v1.csv",predictions);prediction_sha=sha(folder/"A_NATIVE_refinement_predictions_v1.csv")
 for R in (-.95,.1,.5):
  jid=f"low_K_refinement__A_NATIVE__R{R:g}__Kmax12.75__seed1720";jobs.append({"job_id":jid,"option":"A_NATIVE","R":R,"Kmax_MPa_sqrt_m":12.75,"deltaK_driver_MPa_sqrt_m":12.75*(1-R),"seed":1720,"n_bins":80,"result_path":str((folder/"results"/jid).resolve()),"status":"PENDING","attempt":0,"pid":None,"resumed":False,"reused":False,"prediction_file_sha256":prediction_sha,"prediction_frozen_unix_ns":frozen,"contract_path":"","terminal_path":"","log_path":""})
 atomic_csv(folder/"A_NATIVE_refinement_job_registry.csv",jobs);atomic_json(folder/"A_NATIVE_refinement_trigger.json",{"classification":"INTERPOLATION_FAILURE","failed_Kmax_MPa_sqrt_m":13.5,"additional_midpoint_Kmax_MPa_sqrt_m":12.75,"offending_original_interval":[12,15],"prediction_sha256":prediction_sha});return jobs

def contract(job,folder,head):
 if Path(job["result_path"]).exists():raise RuntimeError("fresh refinement path exists")
 attempt=int(job["attempt"])+1;log=(folder/"controller_logs"/f"{job['job_id']}__attempt{attempt}.log").resolve();terminal=(folder/"worker_terminals"/f"{job['job_id']}__attempt{attempt}.json").resolve();path=folder/"job_contracts"/f"{job['job_id']}__attempt{attempt}.json"
 env={"PYTHON_BIN":PYTHON,"CONDA_ENV":"arrhenius-sharp-front-v10-codex","CONDA_DEFAULT_ENV":"arrhenius-sharp-front-v10-codex","EXPECTED_BRANCH":BRANCH,"EXPECTED_HEAD":head,"FAMILY_JSON":FAMILY,"V10230_ENTRY_MODULE":"arrhenius_fracture.sharp_front_v10_2_30_candidate_fixed_deltaK","V10230_CANDIDATE_REGISTRY":str((SOURCE/"A_PT03_PT08_R_registry.csv").resolve()),"V10230_CANDIDATE_SELECTION":str((SOURCE/"A_PT03_PT08_R_selection.json").resolve()),"PARAMETER_OPTION":"A_NATIVE","TARGET_DELTAK":f"{job['deltaK_driver_MPa_sqrt_m']:.17g}","R_RATIO":f"{job['R']:.17g}","TARGET_FRACTION":"LOW_K_REFINEMENT","RUN_LABEL":job["job_id"],"TARGET_EXT_UM":"100","CYCLES_MAX":"1000000","HAZARD_SEED":"1720","MAX_WALL_SECONDS":"43200","OUTROOT":job["result_path"],"V10230_HIGH_CYCLE_EXPLICIT_ONLY":"1"}
 payload={"schema":"two_scale_native_refinement_contract_v1","job_id":job["job_id"],"attempt":attempt,"created_unix_ns":time.time_ns(),"result_path":job["result_path"],"log_path":str(log),"terminal_path":str(terminal),"launch_head":head,"qualified_solver_head":"94871be15702e7fb85116b92af62c1226c61be42","production_solver_sha256":"c15a957161e8cdb43bf944243d8a18a64839dbfe0518aa2d1c5f0c93773b817b","prediction_file_sha256":job["prediction_file_sha256"],"prediction_frozen_unix_ns":int(job["prediction_frozen_unix_ns"]),"fresh_virgin_start":True,"resume":False,"environment":env};atomic_json(path,payload);job.update({"attempt":attempt,"contract_path":str(path.resolve()),"terminal_path":str(terminal),"log_path":str(log)});return path

def valid(job):
 path=Path(job["result_path"])/"developed_fatigue_growth_summary.json"
 if not path.is_file():return False
 d=json.loads(path.read_text());return bool(d.get("target_reached")) and bool(d.get("stable_growth_provisional")) and int(d.get("event_count",0))>=10 and int(d.get("restart_count") or 0)==0

def reconcile(folder,jobs):
 for job in jobs:
  if job["status"] not in {"RUNNING","INVALID_OR_NONTERMINAL"} or (job["status"]=="RUNNING" and alive(job.get("pid"))):continue
  out=Path(job["result_path"]);physical=any((out/name).is_file() for name in ("kinetic_tip_cell_audit_v101.json","high_cycle_live_checkpoint.json"))
  if physical:raise RuntimeError("interrupted physical refinement cannot resume")
  if out.exists():q=folder/"quarantine"/f"{job['job_id']}__attempt{int(job['attempt'])}__prephysics";q.parent.mkdir(exist_ok=True);os.replace(out,q)
  job.update({"status":"PENDING","pid":None,"exit_code":None,"wall_seconds":None})
 atomic_csv(folder/"A_NATIVE_refinement_job_registry.csv",jobs)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=ROOT);ap.add_argument("--workers",type=int,default=3);a=ap.parse_args()
 if not 1<=a.workers<=3:raise SystemExit("workers must be 1..3")
 if git("branch","--show-current")!=BRANCH or git("status","--short"):raise SystemExit("refinement controller requires clean worktree")
 root=a.root.resolve();folder=root/"A_NATIVE_low_K_refinement";jobs=pd.read_csv(folder/"A_NATIVE_refinement_job_registry.csv").to_dict("records") if (folder/"A_NATIVE_refinement_job_registry.csv").is_file() else initialize(root);reconcile(folder,jobs);head=git("rev-parse","HEAD");registry=folder/"A_NATIVE_refinement_job_registry.csv";atomic_json(folder/"A_NATIVE_refinement_controller_state.json",{"phase":"RUNNING","pid":os.getpid(),"head":head})
 while True:
  for job in [x for x in jobs if x["status"]=="RUNNING"]:
   terminal=Path(job["terminal_path"])
   if terminal.is_file() or not alive(job["pid"]):
    d=json.loads(terminal.read_text()) if terminal.is_file() else {};job["exit_code"]=d.get("exit_code");job["wall_seconds"]=d.get("wall_seconds");job["pid"]=None;job["status"]="PHYSICAL_TARGET_REACHED" if valid(job) else "INVALID_OR_NONTERMINAL";atomic_csv(registry,jobs)
    if job["status"]!="PHYSICAL_TARGET_REACHED":raise RuntimeError(f"refinement failed: {job['job_id']}")
  active=[x for x in jobs if x["status"]=="RUNNING"];pending=[x for x in jobs if x["status"]=="PENDING"]
  while pending and len(active)<a.workers:
   job=pending.pop(0);path=contract(job,folder,head);p=subprocess.Popen([PYTHON,"scripts/run_v10_2_30_two_scale_anchor_worker.py","--contract",str(path)],start_new_session=True);job["pid"]=p.pid;job["status"]="RUNNING";active.append(job);atomic_csv(registry,jobs)
  if not active and not pending:break
  time.sleep(5)
 atomic_json(folder/"A_NATIVE_refinement_controller_state.json",{"phase":"COMPLETE","result":"PASS","pid":os.getpid(),"head":head});return 0
if __name__=="__main__":raise SystemExit(main())
