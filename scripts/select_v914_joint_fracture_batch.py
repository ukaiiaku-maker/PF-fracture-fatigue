#!/usr/bin/env python3
"""Select the first 50 diverse joint candidates for short fracture confirmation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--atlas",type=Path,required=True);a=ap.parse_args();props=pq.read_table(a.atlas/"joint_candidate_property_atlas.parquet").to_pandas();fat=pq.read_table(a.atlas/"fatigue_candidate_properties.parquet",columns=["candidate_id","best_broad_window_m","best_window_C","maximum_valid_window_width"]).to_pandas().rename(columns={"candidate_id":"base_candidate_id"});q=props.merge(fat,on="base_candidate_id",how="left",validate="many_to_one");q=q[q.physical_surface_valid&np.isfinite(q.predicted_Kinit_max)].copy();q["fatigue_evidence"]="ANALYTICAL_PROVISIONAL";real=q[q.best_broad_window_m.between(2,6)].copy();selected=[];used_base=set()
 for role,col in [("REALISTIC_FATIGUE_PEAKT","distance_to_PeakT"),("REALISTIC_FATIGUE_DBTT","distance_to_DBTT"),("REALISTIC_FATIGUE_WEAKT","distance_to_weakT"),("REALISTIC_FATIGUE_CERAMIC","distance_to_ceramic")]:
  pool=real.sort_values([col,"maximum_valid_window_width"],ascending=[True,False])
  count=0
  for i,r in pool.iterrows():
   if r.base_candidate_id in used_base:continue
   selected.append((i,role));used_base.add(r.base_candidate_id);count+=1
   if count==10:break
 distances=q[["distance_to_PeakT","distance_to_DBTT","distance_to_weakT","distance_to_ceramic"]].to_numpy(float);q["intermediate_balance"]=np.std(distances,axis=1)/(np.mean(distances,axis=1)+1e-12);pool=q.sort_values(["intermediate_balance","predicted_temperature_curvature"],ascending=[True,False]);count=0
 for i,r in pool.iterrows():
  if r.base_candidate_id in used_base:continue
  selected.append((i,"INTERMEDIATE_MAXIMIN"));used_base.add(r.base_candidate_id);count+=1
  if count==10:break
 out=q.loc[[i for i,_ in selected]].copy();out.insert(1,"selection_role",[r for _,r in selected]);out.insert(2,"selection_rank",np.arange(1,len(out)+1));out["simulation_status"]="SELECTED_NOT_LAUNCHED";out.to_csv(a.atlas/"joint_fracture_batch01_50.csv",index=False);(a.atlas/"joint_fracture_batch01_manifest.json").write_text(json.dumps({"schema":"v914_joint_fracture_batch01_v1","candidate_count":len(out),"role_counts":out.selection_role.value_counts().to_dict(),"fatigue_evidence":"ANALYTICAL_PROVISIONAL until n64 observations become available","protocol":"qualified historical 1D short fracture/R-curve screen","target_extension_um":"100-250 staged; no 1mm launch","launched":False},indent=2)+"\n");print(out.selection_role.value_counts().to_string());return 0
if __name__=="__main__":raise SystemExit(main())
