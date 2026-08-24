#!/usr/bin/env python3
"""Audit fracture/fatigue joinability and import qualified legacy fracture data."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq

ATLAS_COORDS=["cleave_G00_eV","cleave_sigc0_GPa","cleave_exp_a","cleave_exp_n","cleave_floor_frac","emit_G00_eV","emit_sigc0_GPa","emit_exp_a","emit_exp_n","emit_floor_frac","peierls_H0_eV","peierls_activation_entropy_kB","peierls_exp_a","peierls_exp_n","taylor_H0_eV","taylor_activation_entropy_kB","taylor_exp_a","taylor_exp_n","rho_source0_m2","taylor_corr_rho_c_m2","taylor_corr_scale","c_blunt"]
THERMAL=["Tref_K","cleave_gT_eV_per_K","cleave_sT_GPa_per_K","emit_gT_eV_per_K","emit_sT_GPa_per_K"]
FIXED=["peierls_nu0_s","taylor_nu0_s"]
CANONICAL={"v913_zeroD_sobol_0242980":"Peak-T","v913_zeroD_sobol_0202500":"DBTT","v913_zeroD_sobol_0129902":"weak-T","v913_zeroD_sobol_0077080":"ceramic-like"}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def row_sha(row,cols):return hashlib.sha256("|".join(f"{c}={float(row[c]).hex()}" for c in cols).encode()).hexdigest()
def git(repo,*args):return subprocess.check_output(["git","-C",str(repo),*args],text=True).strip()

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--atlas",type=Path,required=True);ap.add_argument("--fracture-repo",type=Path,required=True);a=ap.parse_args();out=a.atlas
 screen=a.fracture_repo/"runs/v9_13_zeroD_promoted_1d_384_50um_v2/one_d_screen"
 weak_screen=a.fracture_repo/"runs/v9_13_weakT_ceramic_search_5T_100um_v1/one_d_100um"
 source_pool=a.fracture_repo/"runs/v9_13_zero_d_large_persistent_search_v1/zero_d_ranked_candidates_corrected.csv"
 feature_frames=[]
 for source in (screen,weak_screen):
  q=pd.read_csv(source/"candidate_pool_features.csv");q.columns=[c.removeprefix("x_raw__") for c in q.columns];q["source_registry"]=str((source/"candidate_pool_features.csv").resolve());feature_frames.append(q)
 features=pd.concat(feature_frames,ignore_index=True).drop_duplicates("candidate_id",keep="last")
 cases=pd.concat([pd.read_csv(source/"case_results_checkpoint.csv").assign(source_file=str((source/"case_results_checkpoint.csv").resolve())) for source in (screen,weak_screen)],ignore_index=True)
 events=pd.concat([pd.read_csv(source/"R_curve_events.csv").assign(source_file=str((source/"R_curve_events.csv").resolve())) for source in (screen,weak_screen)],ignore_index=True)
 allcols=THERMAL+ATLAS_COORDS+FIXED
 missing=[c for c in allcols if c not in features];
 if missing:raise RuntimeError(f"fracture features missing: {missing}")
 features["fracture_parameter_sha256"]=[row_sha(r,allcols) for _,r in features.iterrows()]
 features=features.sort_values("fracture_parameter_sha256").reset_index(drop=True)
 features.insert(0,"fracture_atlas_id",[f"V914_FRACTURE_LEGACY_{i+1:06d}" for i in range(len(features))])
 features["source_candidate_id"]=features.candidate_id;features["source_repository_head"]=git(a.fracture_repo,"rev-parse","HEAD")
 features["canonical_anchor_label"]=features.candidate_id.map(CANONICAL)
 features["exact_fatigue_atlas_candidate_id"]=pd.NA
 pq.write_table(pa.Table.from_pandas(features,preserve_index=False),out/"fracture_legacy_candidate_registry.parquet",compression="zstd");features.to_csv(out/"fracture_legacy_candidate_registry.csv",index=False)

 mapping=[]
 for c in ATLAS_COORDS:mapping.append({"fracture_parameter":c,"mapping_class":"identical_atlas_coordinate","atlas_coordinate":c,"fatigue_atlas_value_policy":"varied","join_implication":"direct coordinate identity"})
 for c in THERMAL:mapping.append({"fracture_parameter":c,"mapping_class":"fracture_only_coordinate","atlas_coordinate":"","fatigue_atlas_value_policy":"not represented; queue uses a fixed 300 K reduction","join_implication":"required for temperature-response identity"})
 for c in FIXED:mapping.append({"fracture_parameter":c,"mapping_class":"fixed_common_physics_value","atlas_coordinate":"","fatigue_atlas_value_policy":"fixed in production launch registry","join_implication":"joinable only while fixed values agree"})
 pd.DataFrame(mapping).to_csv(out/"fracture_fatigue_parameter_mapping.csv",index=False)

 byid=features.set_index("candidate_id");anchors=[]
 for cid,label in CANONICAL.items():
  r=byid.loc[cid];q=cases[cases.candidate_id.eq(cid)];e=events[events.candidate_id.eq(cid)]
  anchors.append({"fracture_atlas_id":r.fracture_atlas_id,"source_candidate_id":cid,"anchor_label":label,"fracture_parameter_sha256":r.fracture_parameter_sha256,"exact_fatigue_atlas_candidate_id":"","temperature_count":int(q.temperature_K.nunique()),"terminal_case_count":int(q.status.isin(["completed","target_reached","growth_target_reached"]).sum()),"R_curve_event_count":len(e),"K_first_min_MPa_sqrt_m":q.K_first_MPa_sqrt_m.min(),"K_first_max_MPa_sqrt_m":q.K_first_MPa_sqrt_m.max(),"K_50um_min_MPa_sqrt_m":q.K_50um_MPa_sqrt_m.min(),"K_50um_max_MPa_sqrt_m":q.K_50um_MPa_sqrt_m.max(),"join_status":"LEGACY_FRACTURE_ANCHOR_NOT_IDENTICAL_TO_22D_FATIGUE_ATLAS"})
 anchors=pd.DataFrame(anchors);anchors.to_csv(out/"fracture_anchor_records.csv",index=False);(out/"fracture_anchor_records.json").write_text(json.dumps(anchors.to_dict("records"),indent=2)+"\n")

 idmap=features.set_index("candidate_id").fracture_atlas_id.to_dict();obs=[]
 caseprops=["K_first_MPa_sqrt_m","K_10um_MPa_sqrt_m","K_25um_MPa_sqrt_m","K_50um_MPa_sqrt_m","achieved_projected_extension_um","max_backstress_GPa","min_front_width_um","max_tip_radius_um","max_source_multiplicity","n_events"]
 for _,r in cases.iterrows():
  prov=json.dumps({"source":r.source_file,"sha256":digest(r.source_file),"fracture_repo_head":git(a.fracture_repo,"rev-parse","HEAD")},sort_keys=True)
  condition=json.dumps({"temperature_K":float(r.temperature_K),"seed":int(r.seed),"checkpoint_um":float(r.checkpoint_um)},sort_keys=True)
  for prop in caseprops:
   value=pd.to_numeric(r[prop],errors="coerce")
   if np.isfinite(value):obs.append({"candidate_id":idmap[r.candidate_id],"physics_model":"v9.13_persistent_site_1D_pure_fracture","fidelity":"1D_50um_temperature_screen","condition":condition,"property_name":prop,"value":float(value),"uncertainty":np.nan,"status":str(r.status),"provenance":prov})
 observations=pd.DataFrame(obs)
 opath=out/"candidate_property_observations.parquet"
 if opath.exists() and pq.ParquetFile(opath).metadata.num_rows:
  old=pq.read_table(opath).to_pandas();observations=pd.concat([old,observations],ignore_index=True).drop_duplicates(["candidate_id","physics_model","fidelity","condition","property_name","provenance"],keep="last")
 pq.write_table(pa.Table.from_pandas(observations,preserve_index=False),opath,compression="zstd")
 inventory=pd.DataFrame([{"source":"v9.13 ranked zero-D fracture population","unique_candidates":pd.read_csv(source_pool,usecols=["candidate_id"]).candidate_id.nunique(),"qualified_property_level":"analytical/exact zero-D temperature response","path":str(source_pool),"sha256":digest(source_pool)},{"source":"combined v9.13 promoted 1-D fracture screens","unique_candidates":features.candidate_id.nunique(),"qualified_property_level":"5-10-temperature 1-D initiation and short R-curve","path":json.dumps([str(screen/"case_results_checkpoint.csv"),str(weak_screen/"case_results_checkpoint.csv")]),"sha256":hashlib.sha256((digest(screen/"case_results_checkpoint.csv")+digest(weak_screen/"case_results_checkpoint.csv")).encode()).hexdigest()},{"source":"combined v9.13 1-D R-curve event ledgers","unique_candidates":events.candidate_id.nunique(),"qualified_property_level":"event-resolved through 50-100um","path":json.dumps([str(screen/"R_curve_events.csv"),str(weak_screen/"R_curve_events.csv")]),"sha256":hashlib.sha256((digest(screen/"R_curve_events.csv")+digest(weak_screen/"R_curve_events.csv")).encode()).hexdigest()}]);inventory.to_csv(out/"fracture_existing_results_inventory.csv",index=False)
 report={"schema":"v914_fracture_fatigue_join_audit_v1","joinability":"LARGER_UNIFIED_PARAMETER_VECTOR_REQUIRED_FOR_TEMPERATURE_RESPONSE","directly_shared_coordinate_count":len(ATLAS_COORDS),"fracture_only_temperature_coordinate_count":len(THERMAL),"fixed_shared_value_count":len(FIXED),"legacy_fracture_candidates_imported":len(features),"fracture_observations_imported":len(observations),"canonical_anchors_imported":len(anchors),"large_new_fracture_campaign_launched":False,"reason":"The 22-D fatigue atlas omits Tref and thermal barrier/stress derivatives that vary in the historical fracture population. A fatigue candidate therefore does not uniquely define Kinit(T), DBTT, Peak-T, or ceramic-like behavior."};(out/"fracture_joinability_audit.json").write_text(json.dumps(report,indent=2)+"\n")
 (out/"fracture_joinability_report.md").write_text(f"""# Fracture/fatigue atlas joinability audit

## Decision

**A larger unified parameter vector is required for temperature-dependent fracture claims.** The two models share {len(ATLAS_COORDS)} constitutive coordinates exactly, but the immutable fatigue atlas does not carry the {len(THERMAL)} independently varied historical fracture temperature coordinates (`Tref_K`, cleavage/emission `gT`, and cleavage/emission `sT`). Attempt frequencies are fixed launch/common-physics values and are joinable only under that frozen convention.

Consequently, no legacy Peak-T, DBTT, weak-T, or ceramic-like row has been relabeled as an identical 22-D fatigue-atlas candidate. The anchors remain provenance-complete legacy fracture identities. This prevents an unsupported combined-property claim.

## Recovered evidence

- Ranked zero-D fracture pool: {inventory.iloc[0].unique_candidates:,} candidates.
- Qualified 10-temperature, 50 µm 1-D screen: {len(features):,} candidates.
- Archived event-resolved R-curve ledger: {len(events):,} events across {events.candidate_id.nunique():,} candidates.
- Canonical anchors recovered: Peak-T, DBTT, weak-T, and ceramic-like.

No new fracture mechanics calculation was launched. Existing qualified properties were imported into the common observation table under stable `V914_FRACTURE_LEGACY_*` identities.
""")
 print(json.dumps(report,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
