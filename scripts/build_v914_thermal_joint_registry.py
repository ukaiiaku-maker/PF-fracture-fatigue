#!/usr/bin/env python3
"""Build immutable thermal profiles and a compositional joint-atlas sample."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
from scipy.stats import qmc,spearmanr

BASE=["cleave_G00_eV","cleave_sigc0_GPa","cleave_exp_a","cleave_exp_n","cleave_floor_frac","emit_G00_eV","emit_sigc0_GPa","emit_exp_a","emit_exp_n","emit_floor_frac","peierls_H0_eV","peierls_activation_entropy_kB","peierls_exp_a","peierls_exp_n","taylor_H0_eV","taylor_activation_entropy_kB","taylor_exp_a","taylor_exp_n","rho_source0_m2","taylor_corr_rho_c_m2","taylor_corr_scale","c_blunt"]
THERMAL=["Tref_K","cleave_gT_eV_per_K","cleave_sT_GPa_per_K","emit_gT_eV_per_K","emit_sT_GPa_per_K"]
VARIABLE=["cleave_G00_eV","cleave_gT_eV_per_K","cleave_sigc0_GPa","cleave_sT_GPa_per_K","cleave_exp_a","cleave_exp_n","cleave_floor_frac","emit_G00_eV","emit_gT_eV_per_K","emit_sigc0_GPa","emit_sT_GPa_per_K","emit_exp_a","emit_exp_n","emit_floor_frac","peierls_H0_eV","peierls_activation_entropy_kB","peierls_exp_a","peierls_exp_n","taylor_H0_eV","taylor_activation_entropy_kB","taylor_exp_a","taylor_exp_n","rho_source0_m2","taylor_corr_rho_c_m2","taylor_corr_scale","c_blunt"]
CANON={"v913_zeroD_sobol_0242980":"Peak-T","v913_zeroD_sobol_0202500":"DBTT","v913_zeroD_sobol_0129902":"weak-T","v913_zeroD_sobol_0077080":"ceramic-like"}

def sha(vals):return hashlib.sha256("|".join(float(v).hex() for v in vals).encode()).hexdigest()
def git(repo,*args):return subprocess.check_output(["git","-C",str(repo),*args],text=True).strip()
def support_label(legacy_id,historical_source_ids_json):
    return "HISTORICAL_THERMAL_SUPPORT" if legacy_id and legacy_id in json.loads(historical_source_ids_json) else "COMPOSITIONAL_EXTRAPOLATION"
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--atlas",type=Path,required=True);ap.add_argument("--fracture-repo",type=Path,required=True);ap.add_argument("--joint-power",type=int,default=20);ap.add_argument("--creation-commit",required=True);a=ap.parse_args();out=a.atlas
 run=a.fracture_repo/"runs/v9_13_zero_d_large_persistent_search_v1";source=run/"zero_d_ranked_candidates_corrected.csv";contract=json.loads((run/"run_contract.json").read_text())["contract"]
 src=pd.read_csv(source);src=src.sort_values("sample_index").reset_index(drop=True)
 # Recover exact latent Sobol coordinates from the original 28-D sequence.
 sampler=qmc.Sobol(d=2+len(VARIABLE),scramble=True,seed=int(contract["search"]["seed"]));u=sampler.random_base2(18);idx=src.sample_index.to_numpy(int);latent=u[idx]
 rows=src[["candidate_id","sample_index","sample_mode","anchor_candidate_id",*THERMAL]].copy()
 for name in THERMAL:
  if name=="Tref_K":rows[f"u__{name}"]=np.nan
  else:rows[f"u__{name}"]=latent[:,2+VARIABLE.index(name)]
 rows["historical_thermal_sha256"]=[sha(r[THERMAL]) for _,r in rows.iterrows()]
 # Exact duplicates are collapsed, retaining all historical source IDs.
 grouped=[]
 for _,g in rows.groupby("historical_thermal_sha256",sort=True):
  r=g.iloc[0].copy();r["historical_source_candidate_ids_json"]=json.dumps(g.candidate_id.tolist());r["historical_profile_multiplicity"]=len(g);r["canonical_anchor_labels_json"]=json.dumps(sorted({CANON[x] for x in g.candidate_id if x in CANON}));grouped.append(r)
 profiles=pd.DataFrame(grouped).sort_values("historical_thermal_sha256").reset_index(drop=True)
 profiles["parent_domain_version"]="v913_historical_4096_thermal_profile_v1"
 profiles["source_registry"]=str(source.resolve())
 # The weak-T and ceramic anchors came from the later five-temperature search,
 # not the 4,096 retained DBTT/Peak population. Append all distinct qualified
 # legacy-screen profiles so every canonical anchor remains representable.
 legacy_path=out/"fracture_legacy_candidate_registry.parquet"
 supplemental_count=0
 if legacy_path.exists():
  legacy=pq.read_table(legacy_path,columns=["source_candidate_id","source_registry","canonical_anchor_label",*THERMAL]).to_pandas();legacy["historical_thermal_sha256"]=[sha(r[THERMAL]) for _,r in legacy.iterrows()];known=set(profiles.historical_thermal_sha256);extra=[]
  for digest_value,g in legacy[~legacy.historical_thermal_sha256.isin(known)].groupby("historical_thermal_sha256",sort=True):
   r=g.iloc[0];extra.append({"candidate_id":r.source_candidate_id,"sample_index":np.nan,"sample_mode":"qualified_legacy_screen","anchor_candidate_id":"","Tref_K":r.Tref_K,"cleave_gT_eV_per_K":r.cleave_gT_eV_per_K,"cleave_sT_GPa_per_K":r.cleave_sT_GPa_per_K,"emit_gT_eV_per_K":r.emit_gT_eV_per_K,"emit_sT_GPa_per_K":r.emit_sT_GPa_per_K,**{f"u__{x}":np.nan for x in THERMAL},"historical_thermal_sha256":digest_value,"historical_source_candidate_ids_json":json.dumps(g.source_candidate_id.tolist()),"historical_profile_multiplicity":len(g),"canonical_anchor_labels_json":json.dumps(sorted(set(g.canonical_anchor_label.dropna()))),"parent_domain_version":"v913_qualified_legacy_screen_thermal_profile_v1","source_registry":json.dumps(sorted(set(g.source_registry.astype(str))))})
  if extra:profiles=pd.concat([profiles,pd.DataFrame(extra)],ignore_index=True,sort=False);supplemental_count=len(extra)
 profiles.insert(0,"thermal_profile_id",[f"V914_THERMAL_{i+1:06d}" for i in range(len(profiles))]);profiles["creation_commit"]=a.creation_commit;profiles["source_registry"]=profiles.source_registry.fillna(str(source.resolve()));profiles["source_repository_head"]=git(a.fracture_repo,"rev-parse","HEAD")
 pq.write_table(pa.Table.from_pandas(profiles,preserve_index=False),out/"thermal_profile_registry.parquet",compression="zstd");profiles.to_csv(out/"thermal_profile_registry.csv",index=False)
 bounds={name:{"fixed":481.33} if name=="Tref_K" else contract["policy"]["search_dimensions"][name] for name in THERMAL};(out/"historical_thermal_bounds.json").write_text(json.dumps({"schema":"v914_historical_thermal_bounds_v1","bounds":bounds,"seed":contract["search"]["seed"],"sobol_dimensions":28,"latent_control_dimensions":["local_mode","anchor_index"],"local_anchor_fraction":contract["policy"]["local_anchor_fraction"]},indent=2)+"\n")
 corr=src[THERMAL+BASE].corr(method="spearman");corr.to_csv(out/"historical_thermal_shared_spearman.csv")
 max_cross=float(np.nanmax(np.abs(corr.loc[THERMAL,BASE].to_numpy())))
 (out/"historical_thermal_parameter_audit.md").write_text(f"""# Historical thermal-parameter audit

The source generator used a scrambled 28-D Sobol sequence (seed `{contract['search']['seed']}`): two control dimensions followed by 26 active variable dimensions. `Tref_K=481.33 K` was fixed; only the other four thermal quantities varied.

The four thermal Sobol columns and the 22 shared columns are separate latent coordinates. They were **not historically factorized**, however: 85% of rows used one shared anchor-local mode and one shared anchor identity for every parameter. The retained 4,096 population was also selected using temperature-response objectives. The largest absolute thermal/shared Spearman correlation in that selected population is {max_cross:.5g}.

Joint pairing may therefore explore beyond historical correlations, but such rows must be labeled `COMPOSITIONAL_EXTRAPOLATION` and pass physical surface checks. Historical-profile pairings retain `HISTORICAL_THERMAL_SUPPORT`.

## Room-temperature preservation

The fatigue base stores its constitutive intercept at 300 K. When materializing a child for the historical fracture implementation, which uses `G(T)=G00+gT*(T-Tref)` and the analogous stress scale, intercepts must be re-anchored:

`G00_at_Tref = G_base_300K + gT*(Tref-300 K)`

`sigc0_at_Tref = sigc_base_300K + sT*(Tref-300 K)`.

Without this transformation, attaching a profile would silently change the base candidate's 300 K state.
""")
 schema={"schema":"v914_hierarchical_27D_joint_candidate_v1","identity":"base_candidate_id + thermal_profile_id + common_physics_id","base_coordinates":BASE,"thermal_coordinates":THERMAL,"materialization":{"reference_temperature_K":300.0,"cleave_G00_at_profile_Tref":"base.cleave_G00_eV + thermal.cleave_gT_eV_per_K*(thermal.Tref_K-300)","cleave_sigc0_at_profile_Tref":"base.cleave_sigc0_GPa + thermal.cleave_sT_GPa_per_K*(thermal.Tref_K-300)","emit_G00_at_profile_Tref":"base.emit_G00_eV + thermal.emit_gT_eV_per_K*(thermal.Tref_K-300)","emit_sigc0_at_profile_Tref":"base.emit_sigc0_GPa + thermal.emit_sT_GPa_per_K*(thermal.Tref_K-300)"},"fixed_attempt_frequencies":{"peierls_nu0_s":1e12,"taylor_nu0_s":1e11}};(out/"thermal_profile_schema.json").write_text(json.dumps(schema,indent=2)+"\n")
 provenance={"schema":"v914_thermal_profile_provenance_v1","source_registry":str(source.resolve()),"source_run_contract":str((run/"run_contract.json").resolve()),"source_repository_head":git(a.fracture_repo,"rev-parse","HEAD"),"historical_4096_candidates":len(src),"supplemental_qualified_legacy_profiles":supplemental_count,"distinct_profiles":len(profiles),"creation_commit":a.creation_commit,"physics_calculations_rerun":False};(out/"thermal_profile_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
 # Deterministic, non-Cartesian 2-D Sobol pairing over base and profile indices.
 n=2**a.joint_power;j=qmc.Sobol(d=2,scramble=True,seed=271914).random_base2(a.joint_power);base_n=pq.ParquetFile(out/"master_candidate_registry.parquet").metadata.num_rows;base_idx=np.minimum((j[:,0]*base_n).astype(np.int64),base_n-1);thermal_idx=np.minimum((j[:,1]*len(profiles)).astype(np.int64),len(profiles)-1)
 base=pq.read_table(out/"master_candidate_registry.parquet",columns=["candidate_id","candidate_sha256","legacy_candidate_id",*BASE]).take(pa.array(base_idx)).to_pandas();therm=profiles.iloc[thermal_idx].reset_index(drop=True)
 # A row is HISTORICAL_THERMAL_SUPPORT only if this exact base candidate (by
 # its legacy generation identity, if any) was one of the original historical
 # sources this thermal profile was extracted from; every other pairing is a
 # compositional extrapolation beyond observed co-occurrence (see
 # historical_thermal_parameter_audit.md).
 support=[support_label(legacy,src) for legacy,src in zip(base.legacy_candidate_id,therm.historical_source_candidate_ids_json)]
 joint=pd.DataFrame({"joint_candidate_id":[f"V914_JOINT_{i+1:09d}" for i in range(n)],"base_candidate_id":base.candidate_id,"thermal_profile_id":therm.thermal_profile_id,"common_physics_id":"v913_zeroD_transfer_common_physics_bd356ce6","joint_sequence_index":np.arange(n,dtype=np.int64),"pairing_u_base":j[:,0],"pairing_u_thermal":j[:,1],"pairing_strategy":"SOBOL_PRODUCT_NONCARTESIAN","support_label":support})
 for c in BASE:joint[c]=base[c].to_numpy()
 for c in THERMAL:joint[c]=therm[c].to_numpy()
 joint["base_candidate_sha256"]=base.candidate_sha256.to_numpy();joint["thermal_profile_sha256"]=therm.historical_thermal_sha256.to_numpy();joint["joint_sha256"]=[hashlib.sha256(f"{b}|{t}|v913_zeroD_transfer_common_physics_bd356ce6".encode()).hexdigest() for b,t in zip(joint.base_candidate_sha256,joint.thermal_profile_sha256)]
 if joint[["base_candidate_id","thermal_profile_id","common_physics_id"]].duplicated().any():raise RuntimeError("duplicate compositional joint identity")
 pq.write_table(pa.Table.from_pandas(joint,preserve_index=False),out/"joint_candidate_registry.parquet",compression="zstd")
 print(json.dumps({"historical_profiles":len(profiles),"joint_candidates":len(joint),"maximum_selected_cross_spearman":max_cross,"room_temperature_reanchoring_required":True},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
