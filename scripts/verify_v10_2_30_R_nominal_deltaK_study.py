#!/usr/bin/env python3
"""Fail-closed verifier for the R-ratio and nominal-C(T) completion contract."""
from __future__ import annotations
import argparse,json,math,subprocess
from pathlib import Path
import pandas as pd

REQUIRED=("A_PT03_PT08_R_registry.csv","A_PT03_PT08_R_provenance_manifest.json","A_PT03_PT08_R_parameter_invariance_audit.csv",
"deltaK_semantics_audit.md","deltaK_semantics_audit.json","deltaK_code_path.csv","virtual_CT_geometry.json","virtual_CT_geometry_validation.csv",
"A_PT03_PT08_R_job_registry.csv","A_PT03_PT08_R_explicit_preflight_results.parquet","A_PT03_PT08_R_developed_points.csv","A_PT03_PT08_R_second_seed_points.csv",
"A_PT03_PT08_R_state_histories.parquet","A_PT03_PT08_R_event_results.csv","A_PT03_PT08_R_paris_analysis.csv","A_PT03_PT08_R_local_slopes.csv",
"A_PT03_PT08_R_PT_native_rate_ratios.csv","A_PT03_PT08_nominal_deltaK_points.csv","A_PT03_PT08_constant_load_CT_windows.csv",
"A_PT03_PT08_local_to_nominal_K_transfer.csv","A_PT03_PT08_geometry_sensitivity.csv","A_PT03_PT08_R_final_decision.md","A_PT03_PT08_R_final_decision.json")
FIGURES=("DADN_VS_DRIVER_DELTAK_BY_R","DADN_VS_NOMINAL_FULL_DELTAK_BY_R","DADN_VS_NOMINAL_TENSILE_DELTAK_BY_R","DADN_VS_NOMINAL_KMAX_BY_R",
"PT_TO_NATIVE_RATE_RATIO_VS_R_AND_KMAX","LOCAL_PARIS_SLOPES_BY_R","LOCAL_TO_NOMINAL_K_TRANSFER","CONSTANT_LOAD_CT_APPARENT_CURVES","MOBILE_RETAINED_VS_R",
"PHYSICAL_RETURN_AND_SOURCE_CANCELLATION_VS_R","RADIUS_INTERNAL_STRESS_SHIELDING_VS_R","NET_BLUNTING_HAZARD_EVENT_HISTORY_VS_R","FINAL_R_RATIO_MECHANISM_SUMMARY")
def check(condition,msg,errors):
 if not condition:errors.append(msg)
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);ap.add_argument("--allow-dirty-controller",action="store_true");a=ap.parse_args();root=a.root.resolve();errors=[]
 for n in REQUIRED:check((root/n).is_file() and (root/n).stat().st_size>0,f"missing artifact {n}",errors)
 for n in FIGURES:check((root/"figures"/f"{n}.png").is_file(),f"missing figure {n}",errors)
 if errors: print(json.dumps({"verifier":"FAIL","errors":errors},indent=2));return 1
 jobs=pd.read_csv(root/"A_PT03_PT08_R_job_registry.csv");pre=pd.read_parquet(root/"A_PT03_PT08_R_explicit_preflight_results.parquet");pts=pd.read_csv(root/"A_PT03_PT08_R_developed_points.csv");seed=pd.read_csv(root/"A_PT03_PT08_R_second_seed_points.csv");fits=pd.read_csv(root/"A_PT03_PT08_R_paris_analysis.csv");tr=pd.read_csv(root/"A_PT03_PT08_local_to_nominal_K_transfer.csv");windows=pd.read_csv(root/"A_PT03_PT08_constant_load_CT_windows.csv")
 prov=json.loads((root/"A_PT03_PT08_R_provenance_manifest.json").read_text());sem=json.loads((root/"deltaK_semantics_audit.json").read_text());decision=json.loads((root/"A_PT03_PT08_R_final_decision.json").read_text());md=(root/"A_PT03_PT08_R_final_decision.md").read_text()
 check(len(jobs)==60,"job registry must contain 60 rows (45 new, 15 reuse)",errors);check(jobs.job_id.nunique()==len(jobs),"duplicate job identity",errors)
 check((~jobs.resumed.astype(bool)).all(),"resumed trajectory admitted",errors);check((jobs[jobs.reused==False].acceleration_mode=="explicit_only").all(),"unqualified acceleration admitted",errors)
 check(set(jobs.status).issubset({"PHYSICAL_TARGET_REACHED","REUSED_PHYSICAL_TARGET_REACHED"}),"nonphysical terminal admitted",errors)
 check(len(pre)==6 and (~pre.admitted_as_fatigue_result).all(),"preflight count/admission failure",errors)
 check(len(pts)==36 and pts.target_reached.all() and pts.stable_growth.all(),"primary developed matrix incomplete",errors)
 check(len(seed)==9 and seed.target_reached.all() and seed.stable_growth.all(),"paired-seed matrix incomplete",errors)
 check(len(jobs[jobs.stage=="CONSTANT_LOAD_CT"])==9 and len(windows)>0,"constant-load C(T) matrix incomplete",errors)
 check(prov["all_non_PT_identical"] and prov["all_cleavage_identical"] and prov["all_emission_identical"] and not prov["material_or_common_physics_changed"],"physics/provenance invariance failed",errors)
 check(sem["classification"]=="LOCAL_TIP_EFFECTIVE_K" and not sem["fit_from_crack_growth"],"deltaK semantic/transfer gate failed",errors)
 check((~tr.tip_radius_used.astype(bool)).all() and (~tr.fit_from_growth.astype(bool)).all(),"nominal K improperly uses tip radius or growth fit",errors)
 check(not decision["closure_corrected_deltaK_reported"],"unvalidated closure-corrected DeltaK reported",errors)
 check(all((r.admissible_points>=3) or pd.isna(r.m) for r in fits.itertuples()),"global exponent reported with fewer than three points",errors)
 check(f"`{decision['primary_classification']}`" in md,"Markdown/JSON decision mismatch",errors)
 diff=subprocess.run(["git","diff","--check"],capture_output=True,text=True);check(diff.returncode==0,"git diff --check failed",errors)
 status=subprocess.check_output(["git","status","--short"],text=True).strip();check(a.allow_dirty_controller or not status,"worktree not clean",errors)
 payload={"schema":"A_PT03_PT08_R_verification_v1","verifier":"PASS" if not errors else "FAIL","errors":errors,"job_count":len(jobs),"new_physical_count":int((~jobs.reused.astype(bool)).sum()),
  "reuse_count":int(jobs.reused.astype(bool).sum()),"primary_count":len(pts),"second_seed_count":len(seed),"constant_load_count":len(jobs[jobs.stage=="CONSTANT_LOAD_CT"]),"preflight_count":len(pre),
  "censor_count":int(jobs.status.astype(str).str.contains("CENSOR").sum()),"active_worker_count":int((jobs.status=="RUNNING").sum()),"worktree_clean":not bool(status)}
 (root/"A_PT03_PT08_R_verification.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print(json.dumps(payload,indent=2))
 return 1 if errors else 0
if __name__=="__main__":raise SystemExit(main())

