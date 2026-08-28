#!/usr/bin/env python3
"""Fail-closed verifier for the inverse fatigue-barrier design study."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/inverse_fatigue_barrier_design_v1"
REQUIRED=[
 "inverse_design_source_manifest.json","inverse_design_source_hashes.json","inverse_design_equation_lineage.md",
 "target_design_schema.json","target_design_configurations.json","target_design_manifest.json",
 "exact_inverse_barriers.parquet","exact_inverse_barrier_equations.md","rare_event_inverse_validation.csv",
 "cooperative_gamma_inverse_validation.csv","exp_floor_projection_candidates.csv","exp_floor_projection_errors.csv",
 "exp_floor_parameter_manifolds.parquet","B1_event_conditioned_predictions.csv","B1_archived_validation.csv",
 "slope_contribution_decomposition.csv","inverse_design_candidate_registry.csv",
 "inverse_design_candidate_diff_audit.csv","inverse_design_candidate_hashes.json",
 "prospective_physical_predictions.csv","inverse_design_physical_points.csv","inverse_design_local_slopes.csv",
 "inverse_design_validation_summary.csv","inverse_design_phase_space.parquet","inverse_design_pareto_candidates.csv",
 "inverse_design_final_decision.md","inverse_design_final_decision.json",
]
FIGURES=[
 "01_TARGET_SLOPE_TO_EXACT_BARRIER.png","02_EXACT_GAMMA_INVERSE_VS_RARE_EVENT_LIMIT.png",
 "03_EXP_FLOOR_SLOPE_CAPACITY_MAP.png","04_TARGET_BARRIER_VS_EXP_FLOOR_PROJECTION.png",
 "05_TARGET_AND_ANALYTICAL_DADN_CURVES.png","06_OPENING_COOPERATIVE_STATE_SLOPE_DECOMPOSITION.png",
 "07_A0_A1_B1_LOW_K_COMPARISON.png","08_TARGET_VS_PHYSICAL_LOCAL_SLOPES.png",
 "09_R_DEPENDENCE_OF_INVERSE_DESIGNS.png","10_MONOTONIC_VS_FATIGUE_EFFECT_OF_DESIGNED_BARRIERS.png",
 "11_PHASE_SPACE_RESPONSE_TOPOLOGY_MAP.png","12_PARETO_CANDIDATE_SUMMARY.png",
]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise SystemExit(f"FAIL: {message}")


def main()->int:
    for name in REQUIRED:require((OUT/name).is_file() and (OUT/name).stat().st_size>0,f"missing {name}")
    hashes=json.loads((OUT/"inverse_design_source_hashes.json").read_text())["sha256"]
    for name,digest in hashes.items():require(sha(ROOT/name)==digest,f"source hash changed: {name}")
    source=json.loads((OUT/"inverse_design_source_manifest.json").read_text())
    require(source["created_before_target_generation"] and source["active_worker_count_at_freeze"]==0,"source freeze invalid")
    target=json.loads((OUT/"target_design_manifest.json").read_text())
    require(target["archived_rate_access"]["PURE_SYNTHETIC"]==0,"synthetic target accessed archive")
    require(target["archived_rate_access"]["REFERENCE_SCALE_ANCHORED"]==1,"anchor access count is not one")
    freeze=json.loads((OUT/"prospective_prediction_freeze.json").read_text())
    require(freeze["target_frozen_utc"]<=freeze["candidate_frozen_utc"]<=freeze["prospective_predictions_frozen_utc"]<=freeze["physical_launch_utc"],"freeze ordering invalid")
    require(sha(OUT/"inverse_design_candidate_registry.csv")==freeze["candidate_registry_sha256"],"registry changed after freeze")
    require(sha(OUT/"prospective_physical_predictions.csv")==freeze["prospective_predictions_sha256"],"prediction changed after freeze")
    audit=pd.read_csv(OUT/"inverse_design_candidate_diff_audit.csv")
    require(bool(audit.changed_fields_equal_declared.all()) and bool(audit.noncleavage_physics_unchanged.all()),"candidate diff audit failed")
    exact=pd.read_parquet(OUT/"exact_inverse_barriers.parquet")
    require(bool(exact.saturation_compatible.all()),"inadmissible cooperative inverse")
    gamma=pd.read_csv(OUT/"cooperative_gamma_inverse_validation.csv")
    require(float(gamma.relative_error.max())<=1e-8,"gamma inverse roundtrip failed")
    atlas=pd.read_parquet(OUT/"inverse_design_phase_space.parquet")
    require(len(atlas)>=100000 and bool(atlas.production_compatible.any()),"phase-space atlas incomplete")
    physical=pd.read_csv(OUT/"inverse_design_physical_points.csv",keep_default_na=False)
    require(len(physical)==30,"physical run count is not 30")
    require(not bool((physical.status=="PENDING").any()),"physical jobs remain pending")
    require(int(pd.to_numeric(physical.restart_count).sum())==0 and not bool(physical.resume_used.astype(str).str.lower().isin(["true","1"]).any()),"restart/resume admitted")
    require(len(set(physical.result_path))==len(physical),"duplicate physical result path")
    state=json.loads((OUT/"inverse_design_controller_state.json").read_text())
    require(state["active_worker_count"]==0 and state["phase"].endswith("TERMINAL"),"controller not terminal")
    process=subprocess.check_output(["ps","-ax","-o","command="],text=True)
    active=[line for line in process.splitlines() if any(token in line for token in (
        "run_v10_2_30_inverse_fatigue_barrier_validation.py","inverse_design_physical"
    )) and "verify_v10" not in line]
    require(not active,"active inverse-design worker remains")
    for name in FIGURES:require((OUT/"figures"/name).is_file() and (OUT/"figures"/name).stat().st_size>10000,f"missing figure {name}")
    decision=json.loads((OUT/"inverse_design_final_decision.json").read_text())
    markdown=(OUT/"inverse_design_final_decision.md").read_text()
    require(decision["primary_classification"] in markdown,"Markdown/JSON decision mismatch")
    require("g_A2 = g_A1" in markdown,"A2 rate-equality statement missing")
    status=subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    payload={
      "schema":"v10.2.30_inverse_design_verification_v1","result":"PASS",
      "branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
      "head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "solver_sha256":decision["solver_sha256"],"target_count":4,"candidate_count":3,
      "projection_count":len(pd.read_csv(OUT/"exp_floor_projection_candidates.csv")),
      "atlas_count":len(atlas),"physical_run_count":len(physical),
      "censor_count":int(pd.to_numeric(physical.physical_censor).sum()),
      "active_worker_count":0,"controller_terminal":True,"worktree_clean":not bool(status),
      "classification":decision["primary_classification"],
    }
    (OUT/"inverse_design_verification.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    require(payload["worktree_clean"],"worktree is not clean")
    print(json.dumps(payload,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
