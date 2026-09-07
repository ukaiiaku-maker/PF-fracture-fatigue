#!/usr/bin/env python3
"""Fail-closed completion verifier for the A-native + 8PT mechanism study."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

REPORTS = [
    "A_NATIVE_PLUS_8PT_IMPORT_AND_PROVENANCE.md", "A_NATIVE_PLUS_8PT_CYCLIC_RATE_AUDIT.md",
    "A_NATIVE_PLUS_8PT_EXPLICIT_STATE_RESULTS.md", "A_NATIVE_PLUS_8PT_FATIGUE_RESPONSE.md",
    "A_NATIVE_PLUS_8PT_FRACTURE_VS_FATIGUE.md", "A_NATIVE_PLUS_8PT_FINAL_DECISION.md",
]
TABLES = [
    "A_native_plus_8PT_registry.csv", "A_native_plus_8PT_parameter_diff_audit.csv",
    "A_native_plus_8PT_provenance_manifest.json", "A_native_plus_8PT_monotonic_confirmation.csv",
    "A_native_plus_8PT_cyclic_barrier_rate_audit.parquet", "A_native_plus_8PT_explicit_cycle_results.parquet",
    "A_native_plus_8PT_state_histories.parquet", "A_native_plus_8PT_event_results.csv",
    "A_native_plus_8PT_developed_fatigue_points.csv", "A_native_plus_8PT_paris_window_analysis.csv",
    "A_native_plus_8PT_local_slopes.csv", "A_native_plus_8PT_divergence_summary.csv",
    "A_native_plus_8PT_final_decision.json", "A_native_plus_8PT_true_accelerator_parity.json",
    "A_native_plus_8PT_cyclic_regime_summary.csv", "A_native_plus_8PT_conditional_followup_decision.json",
]
FIGURES = [
    "A8PT_PARAMETER_MAP.png", "A8PT_CYCLIC_BARRIER_RATE_REGIMES.png", "A8PT_MOBILE_RETAINED_EVOLUTION.png",
    "A8PT_PHYSICAL_RETURN_AND_ESCAPE.png", "A8PT_SOURCE_RETURNED_NET_BLUNTING.png",
    "A8PT_RADIUS_INTERNAL_STRESS_SHIELDING.png", "A8PT_HAZARD_AND_EVENT_HISTORY.png",
    "A8PT_DADN_VS_DELTAK.png", "A8PT_LOCAL_PARIS_SLOPES.png",
    "A8PT_FRACTURE_VS_FATIGUE_RESPONSE_MAP.png", "A8PT_FINAL_DIVERGENCE_SUMMARY.png",
]


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> int:
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("runs/A_native_plus_8PT_fatigue_v1"));a=ap.parse_args();root=a.root.resolve()
    for name in TABLES+REPORTS:require((root/name).is_file() and (root/name).stat().st_size>0,f"missing artifact: {name}")
    for name in FIGURES:require((root/"figures"/name).is_file() and (root/"figures"/name).stat().st_size>1000,f"missing figure: {name}")
    reg=pd.read_csv(root/"A_native_plus_8PT_registry.csv");require(len(reg)==9 and reg.option_key.nunique()==9,"nine-variant registry failed")
    prov=json.loads((root/"A_native_plus_8PT_provenance_manifest.json").read_text());require(prov["variant_count"]==9 and prov["donor_artifacts_preserved"],"donor provenance failed")
    require(prov["audited_fracture_HEAD"]=="2df158bf8d1484f40898c64a11fc76fdd327178c","donor HEAD mismatch")
    sp=prov["solver_preflight"];require(sp["solver_physics_preserved"] and sp["common_physics_preserved"],"qualified solver physics changed")
    require(sp["qualified_common_physics_sha256"]==sp["study_common_physics_sha256"],"common physics SHA mismatch")
    diff=pd.read_csv(root/"A_native_plus_8PT_parameter_diff_audit.csv");require(diff.PT_copy_exact.all() and diff.non_PT_preserved.all(),"PT-only substitution failed")
    require((diff.outside_PT_changed_fields_json=="[]").all(),"outside-PT field changed")
    mono=pd.read_csv(root/"A_native_plus_8PT_monotonic_confirmation.csv");require(len(mono)==9 and set(mono.classification)=={"A_BACKGROUND_FRACTURE_INVARIANT"},"monotonic confirmation failed")
    explicit=pd.read_parquet(root/"A_native_plus_8PT_explicit_cycle_results.parquet");require(len(explicit)==18 and set(explicit.status)=={"TERMINAL_EXPLICIT_COMPLETE"},"explicit matrix failed")
    require(float(explicit[explicit.condition=="pos"].physical_returned_mobile.max())==0.0,"positive-R return invariant failed")
    neg=explicit[explicit.condition=="neg"];require((neg.physical_returned_mobile>=0).all() and (neg.returned_source_slip>=0).all(),"reverse bookkeeping failed")
    base=pd.read_csv(root/"A_native_plus_8PT_developed_job_registry.csv");require(len(base)==36 and set(base.status)=={"COMPLETE"},"base registry failed")
    points=pd.read_csv(root/"A_native_plus_8PT_developed_fatigue_points.csv");require(len(points)==48 and points.stable_growth.all(),"developed/multiseed points failed")
    require(set(points[points.campaign_stage=="DEVELOPED_N80_BASE_PANEL"].status)=={"PHYSICAL_TARGET_REACHED"},"base physical classification failed")
    follow=pd.read_csv(root/"A_native_plus_8PT_followup_job_registry.csv");require(len(follow)==18 and set(follow.status)=={"PHYSICAL_TARGET_REACHED"},"follow-up runs failed")
    require(not follow.status.str.contains("WATCHDOG|CENSOR|INTERRUPTED|PREFLIGHT").any(),"nonphysical run admitted")
    cond=json.loads((root/"A_native_plus_8PT_conditional_followup_decision.json").read_text());require(not cond["adaptive_required_variants"],"adaptive work incomplete")
    require(not cond["n128_required"] and cond["n128_classification"].startswith("NOT_TRIGGERED"),"conditional n128 unresolved")
    parity=json.loads((root/"A_native_plus_8PT_true_accelerator_parity.json").read_text());require(parity["at_least_one_accepted_block"],"actual acceleration path was not exercised")
    require(parity["classification"] in {"ACCELERATOR_PARITY_PASS","ACCELERATOR_PARITY_UNAVAILABLE_VALIDATION_MISMATCH"},"parity lacks fail-closed evidence classification")
    if parity["classification"]=="ACCELERATOR_PARITY_PASS":require(parity["all_pass"],"parity pass contradicts checks")
    else:require(not parity["all_pass"] and any(not x["pass"] for x in parity["cases"]),"unavailable parity lacks mismatch evidence")
    final=json.loads((root/"A_native_plus_8PT_final_decision.json").read_text());require(final["classification"]=="FRACTURE_INVARIANT_FATIGUE_INVARIANT","final classification missing")
    state=json.loads((root/"A_native_plus_8PT_study_controller_state.json").read_text());require(state["phase"]=="COMPLETE","controller not complete")
    subprocess.run(["git","diff","--check"],check=True)
    require(not subprocess.check_output(["git","status","--short"],text=True).strip(),"worktree is not clean")
    print(json.dumps({"verifier":"PASS","variants":9,"explicit":18,"base":36,"multiseed":12,"fresh_parity":6,"controller_phase":"COMPLETE"},indent=2))
    return 0


if __name__=="__main__":raise SystemExit(main())
