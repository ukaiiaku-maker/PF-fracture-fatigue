#!/usr/bin/env python3
"""Fail-closed completion verifier for the physical slope-transfer mission."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/physical_slope_transfer_v1"
BRANCH="codex/v10.2.30-physical-slope-transfer"
REQUIRED=[
    "archive_resolution_audit.csv","archived_event_stage_history.parquet",
    "counterfactual_replay_status.csv","cross_target_transfer_fit.csv",
    "equation_and_state_lineage.md","physical_transfer_operator.json",
    "r_eff_and_shielding_audit.csv","second_seed_prospective_predictions.csv",
    "stagewise_slope_transmission.csv","transfer_freeze.json",
    "second_seed_physical_results.csv","second_seed_transfer_validation.csv",
    "second_seed_transfer_decision.json","corrected_candidate_registry.csv",
    "corrected_candidate_selection.json","corrected_candidate_diff_audit.csv",
    "corrected_candidate_projection.json","corrected_candidate_prospective_predictions.csv",
    "corrected_candidate_R_predictions.csv","corrected_candidate_freeze.json",
    "corrected_candidate_job_registry.csv","corrected_candidate_controller_state.json",
    "corrected_candidate_physical_points.csv","corrected_candidate_local_slopes.csv",
    "corrected_candidate_prediction_comparison.csv","physical_slope_transfer_final_decision.json",
    "physical_slope_transfer_final_decision.md","physical_slope_transfer_analysis_manifest.json",
]
FIGURES=[
    "A_PHYS_BY_K_AND_M.png","M4_STAGEWISE_SLOPE_TRANSMISSION.png",
    "RADIUS_DERIVATIVE_VS_STRESS_TRANSMISSION.png","M4_HELDOUT_TRANSFER_TEST.png",
    "EVENT_CONDITIONED_RADIUS_BY_K.png","ACTUAL_CLEAVAGE_SHIELDING_FRACTION.png",
    "M4_CONTROLLED_COUNTERFACTUAL_SLOPES.png","REDUCED_REPLAY_TRANSMISSION.png",
    "SECOND_SEED_TRANSFER_VALIDATION.png","CORRECTED_INVERSE_SLOPE_PROFILE.png",
    "CORRECTED_PROSPECTIVE_LOCAL_SLOPES.png","CORRECTED_PROSPECTIVE_DADN.png",
    "CORRECTED_MONOTONIC_BARRIER_CHANGE.png","CORRECTED_PROSPECTIVE_R_DEPENDENCE.png",
    "PHYSICAL_SLOPE_TRANSFER_ALL_DATA.png","CORRECTED_PREDICTED_VS_PHYSICAL_DADN.png",
    "CORRECTED_PREDICTED_VS_PHYSICAL_SLOPES.png","CORRECTED_PHYSICAL_CENSOR_MARKERS.png",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def main() -> int:
    require(subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()==BRANCH,"wrong branch")
    for name in REQUIRED:
        path=OUT/name;require(path.is_file() and path.stat().st_size>0,f"missing {name}")
    transfer=json.loads((OUT/"transfer_freeze.json").read_text())
    for name,digest in transfer["analysis_artifact_hashes"].items():
        require(sha(OUT/name)==digest,f"frozen transfer artifact changed: {name}")
    operator=json.loads((OUT/"physical_transfer_operator.json").read_text())
    require(operator["universal_acceptance"] and not operator["physics_modified"],"transfer decision invalid")
    require(not operator["exact_T1_T2_T3_replay_available"],"missing phase history was silently reconstructed")
    events=pd.read_parquet(OUT/"archived_event_stage_history.parquet")
    require(len(events)==378 and len(events[["option_key","Kmax_MPa_sqrt_m"]].drop_duplicates())==21,"archive reconstruction incomplete")
    require(events.geometry_commit_inserted.all() and events.first_passage_action_closes_to_threshold.all(),"event audit incomplete")
    seed=json.loads((OUT/"second_seed_transfer_decision.json").read_text())
    seed_points=pd.read_csv(OUT/"second_seed_physical_results.csv")
    seed_validation=pd.read_csv(OUT/"second_seed_transfer_validation.csv")
    require(seed["result"]=="PASS" and not seed["operator_refit_performed"],"second-seed decision failed")
    require(len(seed_points)==3 and seed_points.target_reached.all(),"second-seed trajectories incomplete")
    require(not seed_points.resume_environment_present.any(),"second-seed resume admitted")
    require(seed_validation.acceptance_pass.all() and not seed_validation.operator_refit_performed.any(),"second-seed transfer bound failed")
    freeze=json.loads((OUT/"corrected_candidate_freeze.json").read_text())
    require(freeze["physics_launch_utc"] is not None and freeze["new_corrected_physics_runs_before_freeze"]==0,"corrected freeze ordering invalid")
    require(not freeze["operator_refit_after_second_seed"],"post-seed refit admitted")
    for name,digest in freeze["artifact_hashes"].items():
        require(sha(OUT/name)==digest,f"corrected frozen artifact changed: {name}")
    selection=json.loads((OUT/"corrected_candidate_selection.json").read_text())
    require(sha(OUT/"corrected_candidate_registry.csv")==selection["installed_registry_sha256"],"candidate registry hash mismatch")
    audit=pd.read_csv(OUT/"corrected_candidate_diff_audit.csv")
    require(not audit.unexpected_change.any(),"undeclared candidate field change")
    jobs=pd.read_csv(OUT/"corrected_candidate_job_registry.csv",keep_default_na=False)
    require(len(jobs)==7 and jobs.status.eq("COMPLETE").all(),"corrected jobs are not complete")
    require(jobs.fresh.all() and not jobs.resume.any() and jobs.seed.eq(1720).all(),"restart/resume or seed mismatch")
    require(jobs.solver_head.eq(freeze["git_head"]).all(),"corrected solver HEAD differs from freeze")
    state=json.loads((OUT/"corrected_candidate_controller_state.json").read_text())
    require(state["phase"]=="CORRECTED_TERMINAL" and state["active_worker_count"]==0,"corrected controller not terminal")
    physical=pd.read_csv(OUT/"corrected_candidate_physical_points.csv",keep_default_na=False)
    require(len(physical)==7 and physical.target_reached.all(),"corrected physical points incomplete")
    require(physical.censor_or_failure_reason.eq("").all(),"censored corrected result admitted")
    require(not physical.restart_environment_present.any(),"corrected restart environment admitted")
    decision=json.loads((OUT/"physical_slope_transfer_final_decision.json").read_text())
    report=(OUT/"physical_slope_transfer_final_decision.md").read_text()
    require(decision["result"]=="PASS" and decision["primary_classification"] in report,"decision mismatch")
    require(decision["all_terminal_uncensored"] and decision["all_fresh_without_resume"],"nonproduction result admitted")
    require(all(f"{number}. **" in report for number in range(1,11)),"completion questions missing")
    for name in FIGURES:
        path=OUT/"figures"/name;require(path.is_file() and path.stat().st_size>10000,f"missing figure {name}")
    process=subprocess.check_output(["ps","-ax","-o","command="],text=True)
    active=[line for line in process.splitlines() if (
        "run_v10_2_30_physical_slope_transfer.py" in line or
        "physical_slope_transfer_corrected_candidate" in line
    ) and "verify_v10_2_30_physical_slope_transfer.py" not in line]
    require(not active,"active physical slope-transfer worker remains")
    status=subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    payload={
        "schema":"v10.2.30_physical_slope_transfer_verification_v1","result":"PASS",
        "branch":BRANCH,"head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "worktree_clean":not bool(status),"active_worker_count":0,"controller_terminal":True,
        "original_trajectory_count":21,"second_seed_trajectory_count":3,"corrected_trajectory_count":7,
        "event_count":378,"classification":decision["primary_classification"],
        "target_slope_success":decision["target_physical_slope_success"],
        "transfer_success":decision["transfer_success"],"operator_refit_after_second_seed":False,
        "physics_modified":False,
    }
    (OUT/"physical_slope_transfer_verification.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    require(payload["worktree_clean"],"worktree is not clean")
    print(json.dumps(payload,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
