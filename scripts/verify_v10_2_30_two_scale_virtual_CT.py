#!/usr/bin/env python3
"""Authoritative fail-closed verifier for the two-scale virtual C(T) study."""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE=Path("runs/A_native_PT03_PT08_R_nominal_deltaK_v1")
SOURCE_FILES=("A_PT03_PT08_R_developed_points.csv","A_PT03_PT08_R_second_seed_points.csv","A_PT03_PT08_constant_load_CT_windows.csv","A_PT03_PT08_local_to_nominal_K_transfer.csv","A_PT03_PT08_R_event_results.csv","A_PT03_PT08_R_final_decision.json","deltaK_semantics_audit.json")
ARTIFACTS=("two_scale_source_manifest.json","two_scale_source_rows.csv","two_scale_source_hashes.json","A_NATIVE_rate_surface_v0.json","A_NATIVE_rate_surface_v0.parquet","A_NATIVE_rate_surface_v0_validation.csv","A_NATIVE_anchor_predictions_v0.csv","A_NATIVE_anchor_results.csv","A_NATIVE_anchor_validation.csv","A_NATIVE_refinement_results.csv","A_NATIVE_refinement_validation.csv","A_NATIVE_rate_surface_v1.json","A_NATIVE_rate_surface_v1.parquet","A_NATIVE_surface_version_comparison.csv","PT03_rate_surface.json","PT08_rate_surface.json","PT_overlay_surface_ratios.csv","PT_conditional_anchor_results.csv","PT_conditional_anchor_validation.csv","constant_load_dynamic_validation.csv","constant_load_dynamic_validation.json","state_history_residual_diagnostics.csv","virtual_CT_protocols.json","virtual_CT_curves.parquet","virtual_CT_life_summary.csv","virtual_CT_load_histories.csv","virtual_CT_local_slopes.csv","virtual_CT_PT_life_ratios.csv","virtual_CT_seed_sensitivity.csv","two_scale_final_decision.md","two_scale_final_decision.json","two_scale_analysis_state.json","two_scale_controller_state.json")
FIGURES=("LOCAL_RATE_DATA_AND_PCHIP_SURFACES","LOCAL_EFFECTIVE_SLOPE_VS_KMAX","LEAVE_ONE_OUT_AND_NEW_ANCHOR_VALIDATION","DYNAMIC_CONSTANT_LOAD_PREDICTED_VS_PHYSICAL","VIRTUAL_CT_K_AND_LOAD_VS_A_OVER_W","VIRTUAL_CT_A_OVER_W_VS_CYCLES","VIRTUAL_CT_DADN_VS_KMAX_BY_R","VIRTUAL_CT_DADN_VS_FULL_DELTAK_BY_R","VIRTUAL_CT_DADN_VS_TENSILE_DELTAK_BY_R","FIXED_LOAD_VS_LOAD_SHEDDING_PATHS","PT03_PT08_RATE_RATIO_ALONG_CT_PATH","PT03_PT08_CUMULATIVE_LIFE_DIFFERENCE","W10_VS_W25_SPECIMEN_SCALE_EFFECT","TWO_SCALE_FINAL_MECHANISM_SUMMARY")

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(condition,message):
 if not condition:raise RuntimeError(message)
def json_read(path):return json.loads(path.read_text())

def verify_jobs(registry_path,expected):
 jobs=pd.read_csv(registry_path);require(len(jobs)==expected,f"wrong job count in {registry_path}");require((jobs.status=="PHYSICAL_TARGET_REACHED").all(),"nonterminal physical anchor");require(not jobs.resumed.astype(bool).any(),"resumed physical trajectory admitted");require(not jobs.reused.astype(bool).any(),"reused physical anchor admitted")
 for job in jobs.itertuples():
  contract=json_read(Path(job.contract_path));require(contract["fresh_virgin_start"] and not contract["resume"],"nonfresh contract admitted");require(contract["production_solver_sha256"]=="c15a957161e8cdb43bf944243d8a18a64839dbfe0518aa2d1c5f0c93773b817b","solver fingerprint mismatch");summary_path=Path(job.result_path)/"developed_fatigue_growth_summary.json";require(summary_path.is_file(),"missing complete anchor audit");summary=json_read(summary_path);require(summary.get("target_reached") and summary.get("stable_growth_provisional"),"nonstationary or nonphysical local datum enters surface");require(int(summary.get("restart_count") or 0)==0,"resumed physical trajectory admitted");require(summary_path.stat().st_mtime_ns>int(contract["prediction_frozen_unix_ns"]),"anchor prediction generated after result was read")
 return jobs

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);ap.add_argument("--allow-dirty",action="store_true");a=ap.parse_args();root=a.root.resolve()
 for name in ARTIFACTS:require((root/name).is_file(),f"missing required artifact {name}")
 for name in FIGURES:require((root/"figures"/f"{name}.png").is_file(),f"missing required figure {name}")
 hashes=json_read(root/"two_scale_source_hashes.json");require(hashes=={name:sha(SOURCE/name) for name in SOURCE_FILES},"source row changes after source manifest frozen")
 manifest=json_read(root/"two_scale_source_manifest.json");require(manifest["held_out_constant_load_rows_used_for_fit"]==0,"constant-load trajectory fitted rather than held out");require(manifest["production_solver_sha256"]=="c15a957161e8cdb43bf944243d8a18a64839dbfe0518aa2d1c5f0c93773b817b","source solver hash mismatch")
 mandatory=verify_jobs(root/"two_scale_anchor_job_registry.csv",9);refinement=verify_jobs(root/"A_NATIVE_low_K_refinement"/"A_NATIVE_refinement_job_registry.csv",3);PT=verify_jobs(root/"PT_overlay_anchors"/"PT_anchor_job_registry.csv",26)
 validation=pd.read_csv(root/"A_NATIVE_anchor_validation.csv");require(len(validation)==9,"wrong mandatory validation count");require(validation.admissible.all(),"inadmissible mandatory anchor");failures=validation[validation.classification=="INTERPOLATION_FAILURE"];require(len(failures)==3 and np.allclose(failures.Kmax_MPa_sqrt_m,13.5),"unexpected anchor failure pattern")
 refined=pd.read_csv(root/"A_NATIVE_refinement_validation.csv");require(len(refined)==3 and refined.admissible.all(),"anchor exceeds error gate without refinement");PT_validation=pd.read_csv(root/"PT_conditional_anchor_validation.csv");require(len(PT_validation)==26 and PT_validation.admissible.all(),"conditional PT interpolation unresolved")
 v0=json_read(root/"A_NATIVE_rate_surface_v0.json");v1=json_read(root/"A_NATIVE_rate_surface_v1.json");require(v0["K_domain_MPa_sqrt_m"]==[12.0,24.0] and not v0["extrapolation"],"v0 interpolation extrapolates");require(v1["K_domain_MPa_sqrt_m"]==[12.0,24.3] and not v1["extrapolation"],"v1 domain invalid");require(not v1["global_Paris_law"] and v1["held_out_constant_load_fit_rows"]==0,"global Paris law or held-out fit entered surface")
 for name in ("PT03_rate_surface.json","PT08_rate_surface.json"):
  surface=json_read(root/name);require(surface["K_domain_MPa_sqrt_m"]==[12.0,24.3] and not surface["extrapolation"],"PT overlay domain is not physically anchored");require(surface["role"]=="DIAGNOSTIC_MECHANISTIC_OVERLAY","PT overlay provenance/role invalid")
 dynamic=json_read(root/"constant_load_dynamic_validation.json");require(dynamic["held_out_trajectory_count"]==9 and dynamic["constant_load_rows_used_for_fit"]==0,"held-out constant-load validation incomplete or fitted")
 curves=pd.read_parquet(root/"virtual_CT_curves.parquet");summary=pd.read_csv(root/"virtual_CT_life_summary.csv");require(len(summary)==42,"virtual protocol matrix incomplete");require(len(curves)==42*401,"virtual curve row matrix incomplete");require(not any("eff" in c.lower() for c in curves.columns),"closure-corrected DeltaK_eff reported");require(not any("radius" in c.lower() for c in curves.columns),"tip radius entered nominal K");require((summary.K_domain_margin_low>=-1e-10).all() and (summary.K_domain_margin_high>=-1e-10).all(),"virtual path leaves validated local domain");require((summary.integration_convergence<1e-5).all(),"numerical integration convergence failed")
 for _,group in curves.groupby(["geometry","option","R","protocol"]):require(np.all(np.diff(group.sort_values("a_over_W").cumulative_cycles)>=0),"nonmonotonic N(a) or a(N)")
 controls=summary[(summary.option=="A_NATIVE")&(summary.protocol=="CONSTANT_KMAX")]
 for row in controls.itertuples():require(math.isclose(row.total_cycles,row.total_extension_m/row.minimum_da_dN,rel_tol=1e-8),"constant-K analytic life mismatch")
 for (R,protocol),group in summary[summary.option=="A_NATIVE"].groupby(["R","protocol"]):
  values=group.set_index("geometry").total_cycles;require(math.isclose(values["W25_B6.25"]/values["W10_B2.5"],2.5,rel_tol=1e-8),"specimen scale effect inconsistent")
 PT_life=pd.read_csv(root/"virtual_CT_PT_life_ratios.csv");require(len(PT_life)==24,"PT fixed/shedding matrix incomplete")
 seed=pd.read_csv(root/"virtual_CT_seed_sensitivity.csv");require(len(seed)==18 and not seed.probabilistic_confidence_interval.astype(bool).any(),"seed sensitivity misclassified as statistical CI")
 decision=json_read(root/"two_scale_final_decision.json");markdown=(root/"two_scale_final_decision.md").read_text();require(decision["primary_classification"] in markdown,"Markdown/JSON decision mismatch");require(not decision["closure_corrected_deltaK_eff_basis"] and not decision["physics_changed"],"unsupported closure correction or physics change claimed");require(decision["new_physical_anchor_count"]==38,"new physical anchor count mismatch")
 controller=json_read(root/"two_scale_controller_state.json");require(controller["phase"]=="COMPLETE" and controller["result"]=="PASS","controller is not terminal")
 process=subprocess.check_output(["ps","-ax","-o","command="],text=True);active=[line for line in process.splitlines() if any(token in line for token in ("two_scale_anchor_worker.py","complete_v10_2_30_two_scale")) and "verify_v10" not in line];require(not active,"active two-scale worker remains")
 status=subprocess.check_output(["git","status","--short"],text=True).strip();require(a.allow_dirty or not status,"worktree is not clean")
 payload={"schema":"two_scale_verification_v1","result":"PASS","source_artifact_count":len(SOURCE_FILES),"mandatory_anchor_count":len(mandatory),"refinement_anchor_count":len(refinement),"conditional_PT_anchor_count":len(PT),"new_physical_anchor_count":38,"reuse_count":0,"censor_count":0,"resume_count":0,"virtual_summary_rows":len(summary),"virtual_curve_rows":len(curves),"figure_count":len(FIGURES),"active_worker_count":len(active),"worktree_clean":not bool(status),"branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),"head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"production_solver_sha256":manifest["production_solver_sha256"]}
 temporary=root/"two_scale_verification.json.tmp";temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");temporary.replace(root/"two_scale_verification.json");print(json.dumps(payload,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
