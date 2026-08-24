#!/usr/bin/env python3
"""Train transparent property-specific proxy-to-1D fracture transfer models."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.build_v914_thermal_joint_registry import BASE,THERMAL

TEMPS=[700,800,900,950,1000,1050,1100,1200,1300,1400]
PROPS=["K_first_MPa_sqrt_m","K_10um_MPa_sqrt_m","K_25um_MPa_sqrt_m","K_50um_MPa_sqrt_m"]

def base300(frame):
 q=frame.copy();dt=300-q.Tref_K
 for p in ("cleave","emit"):
  q[f"{p}_G00_eV"]=q[f"{p}_G00_eV"]+q[f"{p}_gT_eV_per_K"]*dt
  q[f"{p}_sigc0_GPa"]=q[f"{p}_sigc0_GPa"]+q[f"{p}_sT_GPa_per_K"]*dt
 return q

def ridge_fit(x,y,lam):
 return np.linalg.solve(x.T@x+np.eye(x.shape[1])*lam,x.T@y)

def train_one(x,y,groups):
 finite=np.isfinite(x).all(1)&np.isfinite(y);x=x[finite];y=y[finite];groups=groups[finite];mu=x.mean(0);scale=x.std(0);scale[scale<1e-12]=1;z=(x-mu)/scale;design=np.c_[np.ones(len(z)),z];lams=[1e-4,1e-3,1e-2,.1,1,10,100];best=None
 for lam in lams:
  pred=np.full(len(y),np.nan)
  for fold in range(5):
   test=(groups%5)==fold;coef=ridge_fit(design[~test],y[~test],lam);pred[test]=design[test]@coef
  rmse=float(np.sqrt(np.mean((pred-y)**2)))
  if best is None or rmse<best[0]:best=(rmse,lam,pred)
 coef=ridge_fit(design,y,best[1]);ss=np.sum((y-y.mean())**2);return {"n":len(y),"lambda":best[1],"feature_mean":mu.tolist(),"feature_scale":scale.tolist(),"coefficients":coef.tolist(),"cv_RMSE":best[0],"cv_MAE":float(np.mean(abs(best[2]-y))),"cv_R2":float(1-np.sum((best[2]-y)**2)/ss) if ss>0 else np.nan}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--fracture-repo",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args();root=a.fracture_repo/"runs";zpath=root/"v9_13_zero_d_large_persistent_search_v1/zero_d_ranked_candidates_corrected.csv";z=base300(pd.read_csv(zpath));screens=[root/"v9_13_zeroD_promoted_1d_384_50um_v2/one_d_screen/case_results_checkpoint.csv",root/"v9_13_weakT_ceramic_search_5T_100um_v1/one_d_100um/case_results_checkpoint.csv"];cases=pd.concat([pd.read_csv(x) for x in screens],ignore_index=True);cases=cases[cases.candidate_id.isin(z.candidate_id)].copy();z=z.set_index("candidate_id");features=BASE+THERMAL;models={};metrics=[]
 for T in TEMPS:
  proxy=f"proxy_K_T{T}";q=cases[cases.temperature_K.eq(T)].merge(z.reset_index()[["candidate_id",proxy,*features]],on="candidate_id",validate="many_to_one");groups=q.candidate_id.str.extract(r"(\d+)$")[0].astype(int).to_numpy()
  x=np.c_[q[proxy].to_numpy(float),q[features].to_numpy(float)]
  for prop in PROPS:
   model=train_one(x,pd.to_numeric(q[prop],errors="coerce").to_numpy(float),groups);key=f"{prop}@{T}K";models[key]=model;metrics.append({"model":key,**{k:model[k] for k in ["n","lambda","cv_RMSE","cv_MAE","cv_R2"]}})
 payload={"schema":"v914_proxy_to_1D_fracture_transfer_v1","features":["zeroD_proxy_K",*features],"temperatures_K":TEMPS,"properties":PROPS,"models":models,"training_candidate_count":cases.candidate_id.nunique(),"training_rows":len(cases),"base_parameterization":"300K_REANCHORED","model_family":"standardized_ridge_per_property_per_temperature","opaque_joint_score":False};a.out.mkdir(parents=True,exist_ok=True);(a.out/"fracture_property_transfer_model.json").write_text(json.dumps(payload,indent=2)+"\n");pd.DataFrame(metrics).to_csv(a.out/"fracture_property_transfer_cv.csv",index=False);print(json.dumps({"training_candidates":cases.candidate_id.nunique(),"models":len(models),"median_cv_R2":float(pd.DataFrame(metrics).cv_R2.median()),"median_cv_RMSE":float(pd.DataFrame(metrics).cv_RMSE.median())},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
