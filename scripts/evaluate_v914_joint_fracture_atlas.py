#!/usr/bin/env python3
"""Evaluate the compositional joint atlas with the historical zero-D proxy."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.build_v914_thermal_joint_registry import BASE,THERMAL

TEMPS=np.array([700,800,900,950,1000,1050,1100,1200,1300,1400],float)
def predict(model,x):
 z=(x-np.asarray(model["feature_mean"]))/np.asarray(model["feature_scale"]);return np.c_[np.ones(len(z)),z]@np.asarray(model["coefficients"])
def materialize(q):
 r=q.copy();dt=r.Tref_K-300.
 for p in ("cleave","emit"):
  r[f"{p}_G00_eV"]=r[f"{p}_G00_eV"]+r[f"{p}_gT_eV_per_K"]*dt
  r[f"{p}_sigc0_GPa"]=r[f"{p}_sigc0_GPa"]+r[f"{p}_sT_GPa_per_K"]*dt
 r["peierls_nu0_s"]=1e12;r["taylor_nu0_s"]=1e11;r["sample_index"]=r.joint_sequence_index;r["candidate_id"]=r.joint_candidate_id;r["sample_mode"]="joint_composition";r["anchor_candidate_id"]="";return r
def valid_surface(q):
 ok=np.ones(len(q),bool)
 for p in ("cleave","emit"):
  G=q[f"{p}_G00_eV"].to_numpy()[:,None]+q[f"{p}_gT_eV_per_K"].to_numpy()[:,None]*(TEMPS[None,:]-q.Tref_K.to_numpy()[:,None]);S=q[f"{p}_sigc0_GPa"].to_numpy()[:,None]+q[f"{p}_sT_GPa_per_K"].to_numpy()[:,None]*(TEMPS[None,:]-q.Tref_K.to_numpy()[:,None]);ok&=np.isfinite(G).all(1)&np.isfinite(S).all(1)&(G>0).all(1)&(S>0).all(1)
 return ok
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--atlas",type=Path,required=True);ap.add_argument("--fracture-repo",type=Path,required=True);ap.add_argument("--chunk-size",type=int,default=8192);a=ap.parse_args();sys.path.insert(0,str(a.fracture_repo))
 from arrhenius_fracture.zero_d_search_v913 import _proxy_response_batch
 from arrhenius_fracture.emergent_gnd_rcurve_v913 import RCurveLoadingMap
 from scripts.run_mpz_v9_13_persistent_top5 import load_physics
 physics,_=load_physics(a.fracture_repo/"mpz_v9_13_v10222_transfer_common_physics.json");loading=RCurveLoadingMap.from_dict(json.loads((a.fracture_repo/"runs/v9_13_long_map_exponential_110um_v2/v10_2_22_long_rcurve_loading_map_exponential_110um.json").read_text()));loading.validate();transfer=json.loads((a.atlas/"fracture_property_transfer_model.json").read_text());joint=pq.ParquetFile(a.atlas/"joint_candidate_registry.parquet");point_path=a.atlas/"joint_fracture_analytical_points.parquet";prop_path=a.atlas/"joint_candidate_property_atlas.parquet";point_w=prop_w=None
 # Measured canonical initiation curves define continuous anchor distances.
 legacy=pq.read_table(a.atlas/"fracture_legacy_candidate_registry.parquet",columns=["fracture_atlas_id","canonical_anchor_label"]).to_pandas();obs=pq.read_table(a.atlas/"candidate_property_observations.parquet").to_pandas();canon={}
 for _,r in legacy.dropna(subset=["canonical_anchor_label"]).iterrows():
  g=obs[(obs.candidate_id==r.fracture_atlas_id)&(obs.property_name=="K_first_MPa_sqrt_m")];vals={json.loads(c)["temperature_K"]:v for c,v in zip(g.condition,g.value)};canon[r.canonical_anchor_label]=np.array([vals.get(float(T),np.nan) for T in TEMPS])
 done=0
 for batch in joint.iter_batches(batch_size=a.chunk_size):
  q=batch.to_pandas();mat=materialize(q);valid=valid_surface(mat);proxy=np.full((len(q),len(TEMPS)),np.nan)
  if np.any(valid):
   evaluated=_proxy_response_batch(mat.loc[valid].reset_index(drop=True),TEMPS,physics=physics,loading_map=loading,target_rate_s=1e-3,history_events=4.,target_extension_m=50e-6);proxy[valid]=np.column_stack([evaluated[f"proxy_K_T{int(T)}"].to_numpy(float) for T in TEMPS])
  base_x=q[BASE+THERMAL].to_numpy(float);pred={p:np.full_like(proxy,np.nan) for p in ["K_first_MPa_sqrt_m","K_10um_MPa_sqrt_m","K_25um_MPa_sqrt_m","K_50um_MPa_sqrt_m"]}
  for j,T in enumerate(TEMPS):
   x=np.c_[proxy[:,j],base_x]
   for p in pred:pred[p][valid,j]=predict(transfer["models"][f"{p}@{int(T)}K"],x[valid])
  points=pd.DataFrame({"joint_candidate_id":np.repeat(q.joint_candidate_id.to_numpy(),len(TEMPS)),"base_candidate_id":np.repeat(q.base_candidate_id.to_numpy(),len(TEMPS)),"thermal_profile_id":np.repeat(q.thermal_profile_id.to_numpy(),len(TEMPS)),"temperature_K":np.tile(TEMPS,len(q)),"zeroD_proxy_K_MPa_sqrt_m":proxy.ravel(),"predicted_1D_Kinit_MPa_sqrt_m":pred["K_first_MPa_sqrt_m"].ravel(),"predicted_1D_K10um_MPa_sqrt_m":pred["K_10um_MPa_sqrt_m"].ravel(),"predicted_1D_K25um_MPa_sqrt_m":pred["K_25um_MPa_sqrt_m"].ravel(),"predicted_1D_K50um_MPa_sqrt_m":pred["K_50um_MPa_sqrt_m"].ravel(),"physical_surface_valid":np.repeat(valid,len(TEMPS))})
  K=pred["K_first_MPa_sqrt_m"];imax=np.argmax(np.where(np.isfinite(K),K,-np.inf),axis=1);low=(K[:,2]-K[:,0])/(TEMPS[2]-TEMPS[0]);high=(K[:,-1]-K[:,-3])/(TEMPS[-1]-TEMPS[-3]);curv=np.full(len(K),np.nan);kmax=np.full(len(K),np.nan);span=np.full(len(K),np.nan);rises={p:np.full(len(K),np.nan) for p in ("10um","25um","50um")}
  if np.any(valid):
   curv[valid]=np.mean(np.abs(np.diff(K[valid],n=2,axis=1)),axis=1);kmax[valid]=np.max(K[valid],axis=1);span[valid]=np.max(K[valid],axis=1)-np.min(K[valid],axis=1)
   for tag,p in [("10um","K_10um_MPa_sqrt_m"),("25um","K_25um_MPa_sqrt_m"),("50um","K_50um_MPa_sqrt_m")]:rises[tag][valid]=np.median(pred[p][valid]-K[valid],axis=1)
  props=pd.DataFrame({"joint_candidate_id":q.joint_candidate_id,"base_candidate_id":q.base_candidate_id,"thermal_profile_id":q.thermal_profile_id,"common_physics_id":q.common_physics_id,"joint_sha256":q.joint_sha256,"physical_surface_valid":valid,"predicted_Kinit_lowT":K[:,0],"predicted_Kinit_intermediateT":K[:,4],"predicted_Kinit_highT":K[:,-1],"predicted_Kinit_max":kmax,"predicted_Kinit_peak_temperature_K":np.where(valid,TEMPS[imax],np.nan),"predicted_lowT_slope":low,"predicted_highT_slope":high,"predicted_transition_magnitude":span,"predicted_temperature_curvature":curv,"predicted_short_Rcurve_rise_10um":rises["10um"],"predicted_short_Rcurve_rise_25um":rises["25um"],"predicted_short_Rcurve_rise_50um":rises["50um"]})
  distance_names={"Peak-T":"PeakT","DBTT":"DBTT","weak-T":"weakT","ceramic-like":"ceramic"}
  for label,anchor in canon.items():
   mask=np.isfinite(anchor);props[f"distance_to_{distance_names[label]}"]=np.sqrt(np.mean((K[:,mask]-anchor[None,mask])**2,axis=1))
  pt=pa.Table.from_pandas(points,preserve_index=False);pr=pa.Table.from_pandas(props,preserve_index=False)
  if point_w is None:point_w=pq.ParquetWriter(point_path,pt.schema,compression="zstd");prop_w=pq.ParquetWriter(prop_path,pr.schema,compression="zstd")
  point_w.write_table(pt);prop_w.write_table(pr);done+=len(q);print(f"JOINT_ANALYTICAL {done}/{joint.metadata.num_rows}",flush=True)
 point_w.close();prop_w.close();print(json.dumps({"joint_candidates":done,"point_rows":done*len(TEMPS),"property_models":list(pred),"canonical_distances":sorted(canon)},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
