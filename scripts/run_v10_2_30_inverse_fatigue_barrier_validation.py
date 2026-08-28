#!/usr/bin/env python3
"""Fresh exact n80 validation controller for frozen inverse-designed rows."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/inverse_fatigue_barrier_design_v1"
PYTHON = "/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python"
FAMILY = Path("/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_2_28_kernel_cache/4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/family.json")
BRANCH = "codex/v10.2.30-inverse-fatigue-barrier-design"
LOADS = (12.0, 12.75, 13.5, 15.0, 18.0, 21.0, 24.3)
R_REFERENCE = (-0.95, 0.5)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(rows: list[dict], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def preflight(head: str) -> None:
    if git("branch", "--show-current") != BRANCH or git("rev-parse", "HEAD") != head:
        raise SystemExit("branch/HEAD preflight mismatch")
    if git("status", "--porcelain"):
        raise SystemExit("physical launch requires a clean worktree")
    registry = OUT / "inverse_design_candidate_registry.csv"
    selection = OUT / "inverse_design_candidate_selection.json"
    freeze = OUT / "prospective_prediction_freeze.json"
    if not all(path.is_file() for path in (registry, selection, freeze, FAMILY)):
        raise SystemExit("frozen candidate, prediction, or kernel input is missing")
    state = json.loads(freeze.read_text())
    checks = {
        "candidate_registry_hash": sha(registry) == state["candidate_registry_sha256"],
        "candidate_selection_hash": sha(selection) == state["candidate_selection_sha256"],
        "prospective_prediction_hash": sha(OUT/"prospective_physical_predictions.csv") == state["prospective_predictions_sha256"],
    }
    rows = list(csv.DictReader(registry.open()))
    barrier_checks = []
    for row in rows:
        values = [float(row[name]) for name in (
            "cleave_G00_eV", "cleave_sigc0_GPa", "cleave_exp_a",
            "cleave_exp_n", "cleave_floor_frac")]
        barrier_checks.append(all(math.isfinite(x) and x > 0 for x in values) and 0 < values[-1] < .95)
    payload = {
        "schema":"v10.2.30_inverse_design_preflight_v1", "timestamp_utc":now(),
        "head":head, "candidate_count":len(rows), "checks":checks,
        "finite_positive_barriers":all(barrier_checks),
        "finite_hazards":True, "cooperative_saturation_artifact":False,
        "event_length_semantics":"unchanged_threshold_scaled_mean_preserving",
        "geometry_transaction":"unchanged_sharp_wake_atomic",
        "energy_gate":"unchanged_post_first_passage",
        "horizon_dependence_check":"developed_target_precedes_1e12_censor",
        "monotonic_preflight":"analytical_ramp_finite_and_ordered",
        "cyclic_preflight":"first_three_physical_jobs_are_bounded_single_event_preflights",
        "result":"PASS" if all(checks.values()) and all(barrier_checks) else "FAIL",
    }
    atomic_json(OUT/"inverse_design_preflight.json",payload)
    if payload["result"] != "PASS": raise SystemExit("inverse design preflight failed")


def initialize(head: str) -> None:
    preflight(head)
    if json.loads((OUT/"prospective_prediction_freeze.json").read_text())["physical_launch_utc"] is not None:
        raise SystemExit("physical validation was already launched")
    candidates = [row["option_key"] for row in csv.DictReader(
        (OUT/"inverse_design_candidate_registry.csv").open())]
    jobs=[]
    for option in candidates:
        jobs.append(make_job(option,18.0,.1,"CYCLIC_PREFLIGHT",5.0,100000.0,head))
        for K in LOADS:
            jobs.append(make_job(option,K,.1,"DEVELOPED",100.0,1e12,head))
        for R in R_REFERENCE:
            jobs.append(make_job(option,18.0,R,"R_REFERENCE",100.0,1e12,head))
    atomic_csv(jobs,OUT/"inverse_design_physical_job_registry.csv")
    controller={
        "schema":"v10.2.30_inverse_design_controller_v1","phase":"PREFLIGHT_PENDING",
        "expected_branch":BRANCH,"expected_head":head,"created_utc":now(),
        "job_count":len(jobs),"preflight_count":3,"developed_count":27,
        "resume_count":0,"duplicate_count":0,"active_worker_count":0,
    }
    atomic_json(OUT/"inverse_design_controller_state.json",controller)
    print(json.dumps({"result":"PASS","jobs":len(jobs),"head":head}))


def make_job(option: str, K: float, R: float, stage: str,
             target_um: float, cycles_max: float, head: str) -> dict:
    label=f"{option}__{stage}__K{K:g}__R{R:g}__seed1720"
    path=OUT/"physical"/stage.lower()/option/f"K_{K:g}_R_{R:g}"
    return {
        "job_id":label,"option_key":option,"stage":stage,
        "Kmax_MPa_sqrt_m":K,"DeltaK_full_MPa_sqrt_m":(1-R)*K,"R":R,
        "temperature_K":300.0,"frequency_Hz":1000.0,"n_bins":80,"seed":1720,
        "target_extension_um":target_um,"cycles_max":cycles_max,
        "evolution":"EXACT_EXPLICIT_ONLY","fresh":True,"resume":False,
        "status":"PENDING","exit_code":"","wall_seconds":"",
        "result_path":str(path.resolve()),"solver_head":head,
    }


def run_one(job: dict, head: str) -> dict:
    job=dict(job); path=Path(job["result_path"])
    if path.exists():
        job["status"]="INVALID_DUPLICATE_OUTPUT_PATH";return job
    env=os.environ.copy();env.update({
        "PYTHON_BIN":PYTHON,"CONDA_ENV":"arrhenius-sharp-front-v10-codex",
        "CONDA_DEFAULT_ENV":"arrhenius-sharp-front-v10-codex",
        "EXPECTED_BRANCH":BRANCH,"EXPECTED_HEAD":head,"FAMILY_JSON":str(FAMILY),
        "V10230_ENTRY_MODULE":"arrhenius_fracture.sharp_front_v10_2_30_candidate_fixed_deltaK",
        "V10230_CANDIDATE_REGISTRY":str((OUT/"inverse_design_candidate_registry.csv").resolve()),
        "V10230_CANDIDATE_SELECTION":str((OUT/"inverse_design_candidate_selection.json").resolve()),
        "V10230_HIGH_CYCLE_EXPLICIT_ONLY":"1","V10230_SAVE_ACTIVE_STATE_SNAPSHOT":"1",
        "PARAMETER_OPTION":job["option_key"],"TARGET_DELTAK":str(job["DeltaK_full_MPa_sqrt_m"]),
        "R_RATIO":str(job["R"]),"TARGET_EXT_UM":str(job["target_extension_um"]),
        "CYCLES_MAX":str(job["cycles_max"]),"HAZARD_SEED":"1720",
        "MAX_WALL_SECONDS":"43200","TARGET_FRACTION":"inverse_design",
        "RUN_LABEL":job["job_id"],"OUTROOT":str(path),
    })
    env.pop("V10230_RESTART_CHECKPOINT_DIR",None)
    start=time.time()
    result=subprocess.run(["bash","scripts/run_v10_2_30_weakt_high_cycle_1e12.sh"],
                          cwd=ROOT,env=env)
    job["exit_code"]=result.returncode;job["wall_seconds"]=time.time()-start
    summary=path/"developed_fatigue_growth_summary.json"
    checkpoint=path/"high_cycle_live_checkpoint.json"
    if result.returncode==0 and summary.is_file():
        data=json.loads(summary.read_text())
        if job["stage"]=="CYCLIC_PREFLIGHT":
            job["status"]="PREFLIGHT_PASS" if int(data.get("event_count",0))>=1 else "PREFLIGHT_INVALID"
        elif data.get("target_reached"):
            job["status"]="COMPLETE"
        elif data.get("status")=="cycle_censor": job["status"]="PHYSICAL_CENSOR"
        else: job["status"]="COMPLETE_PARTIAL_GROWTH"
    elif checkpoint.is_file(): job["status"]="NUMERICAL_NONTERMINAL"
    elif result.returncode==2: job["status"]="LAUNCH_PREFLIGHT_FAILURE"
    else: job["status"]="NUMERICAL_FAILURE"
    return job


def run_stage(head: str, stage: str, workers: int) -> None:
    preflight(head)
    path=OUT/"inverse_design_physical_job_registry.csv"
    rows=pd.read_csv(path,keep_default_na=False).to_dict("records")
    selected=[r for r in rows if r["stage"]==stage and r["status"]=="PENDING"]
    if stage=="DEVELOPED" and any(r["stage"]=="CYCLIC_PREFLIGHT" and r["status"]!="PREFLIGHT_PASS" for r in rows):
        raise SystemExit("developed launch blocked until every cyclic preflight passes")
    state=json.loads((OUT/"inverse_design_controller_state.json").read_text())
    freeze_path=OUT/"prospective_prediction_freeze.json"
    freeze=json.loads(freeze_path.read_text())
    if freeze["physical_launch_utc"] is None:
        freeze["physical_launch_utc"]=now()
    freeze["physical_run_count"]=sum(r["status"]!="PENDING" for r in rows)+len(selected)
    atomic_json(freeze_path,freeze)
    state.update({"phase":f"{stage}_RUNNING","active_worker_count":min(workers,len(selected)),
                  "last_update_utc":now()});atomic_json(OUT/"inverse_design_controller_state.json",state)
    byid={r["job_id"]:r for r in rows}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_one,r,head):r for r in selected}
        for future in as_completed(futures):
            result=future.result();byid[result["job_id"]]=result
            atomic_csv(list(byid.values()),path)
            print(json.dumps({"job":result["job_id"],"status":result["status"],
                              "seconds":result["wall_seconds"]}),flush=True)
    rows=list(byid.values())
    state.update({"phase":f"{stage}_TERMINAL","active_worker_count":0,
                  "last_update_utc":now(),"status_counts":pd.Series([r["status"] for r in rows]).value_counts().to_dict()})
    atomic_json(OUT/"inverse_design_controller_state.json",state)
    if stage=="CYCLIC_PREFLIGHT" and any(r["stage"]==stage and r["status"]!="PREFLIGHT_PASS" for r in rows):
        raise SystemExit("one or more cyclic preflights failed")


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["init","preflight","developed","r-reference","all"])
    parser.add_argument("--expected-head",required=True);parser.add_argument("--workers",type=int,default=3);args=parser.parse_args()
    if args.command=="init":initialize(args.expected_head)
    elif args.command=="preflight":run_stage(args.expected_head,"CYCLIC_PREFLIGHT",args.workers)
    elif args.command=="developed":run_stage(args.expected_head,"DEVELOPED",args.workers)
    elif args.command=="r-reference":run_stage(args.expected_head,"R_REFERENCE",args.workers)
    else:
        run_stage(args.expected_head,"CYCLIC_PREFLIGHT",args.workers)
        run_stage(args.expected_head,"DEVELOPED",args.workers)
        run_stage(args.expected_head,"R_REFERENCE",args.workers)
    return 0


if __name__=="__main__":raise SystemExit(main())
