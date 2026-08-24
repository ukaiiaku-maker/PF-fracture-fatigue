#!/usr/bin/env python3
"""Incrementally ingest terminal atlas n64 trajectories and update descriptors."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq

TERMINAL={"growth_target_reached","cycle_horizon_reached","work_budget_exhausted"}

def classify(d,rate):
 if int(d.get("restart_count",-1))!=0 or d.get("status") not in TERMINAL:return "numerical invalid"
 e=int(d.get("event_count",0));cy=float(d.get("completed_physical_cycles",0));mx=float(d.get("maximum_physical_cycles",1))
 if e==0:return "inert/censored"
 if e==1:return "threshold-dominated"
 if d.get("status")=="growth_target_reached" and cy<=10:return "rapid/unstable"
 if e>=3 and np.isfinite(rate) and rate>0:return "sustained low-rate growth"
 if cy>=.9*mx:return "burst/arrest"
 return "threshold-dominated"

def trajectory(path):
 d=json.loads(path.read_text());ident=d.get("identity_contract",{});events=d.get("events",[])
 initiation=float(events[0]["cumulative_cycles"]) if events else np.nan
 intervals=[float(events[i]["cumulative_cycles"])-float(events[i-1]["cumulative_cycles"]) for i in range(1,len(events))]
 rates=[float(events[i]["committed_advance_m"])/intervals[i-1] for i in range(1,len(events)) if intervals[i-1]>0]
 if len(events)>=2:
  dc=float(events[-1]["cumulative_cycles"])-float(events[0]["cumulative_cycles"]);da=float(events[-1]["cumulative_extension_m"])-float(events[0]["cumulative_extension_m"]);developed=da/dc if dc>0 else np.nan
 else:developed=np.nan
 loading=ident.get("loading",{})
 return {"candidate_id":ident.get("candidate_id",path.parents[1].name),"fidelity":"n64","DeltaK_MPa_sqrt_m":float(loading.get("deltaK_MPa_sqrt_m",np.nan)),"R":float(loading.get("R",np.nan)),"frequency_Hz":float(loading.get("frequency_Hz",np.nan)),"temperature_K":float(loading.get("temperature_K",np.nan)),"seed":int(ident.get("seed",-1)),"status":d.get("status"),"response_class":classify(d,developed),"restart_count":int(d.get("restart_count",-1)),"event_count":len(events),"initiation_cycles":initiation,"post_initiation_event_intervals_json":json.dumps(intervals),"post_initiation_event_rates_json":json.dumps(rates),"developed_da_dN_m_per_cycle":developed,"completed_physical_cycles":float(d.get("completed_physical_cycles",np.nan)),"cumulative_extension_m":float(d.get("cumulative_extension_m",np.nan)),"run_directory":str(path.parent.resolve()),"producer_commit":ident.get("repository_head"),"physics_sha256":ident.get("physics_sha256"),"candidate_sha256":ident.get("candidate_sha256"),"result_json":str(path.resolve())}

def fits(rows):
 summaries=[]
 for cid,g in rows.groupby("candidate_id"):
  q=g[(g.restart_count==0)&g.developed_da_dN_m_per_cycle.gt(0)&np.isfinite(g.developed_da_dN_m_per_cycle)].sort_values("DeltaK_MPa_sqrt_m")
  complete_loads=g.DeltaK_MPa_sqrt_m.nunique();rec={"candidate_id":cid,"initial_descriptor_terminal_load_count":int(complete_loads),"provisional_m64":np.nan,"provisional_C64":np.nan,"provisional_R2":np.nan,"measured_window_span_fraction":np.nan,"candidate_response_class":"descriptor incomplete"}
  if complete_loads>=3:
   classes=set(g.response_class)
   if len(q)>=2:
    x=np.log10(q.DeltaK_MPa_sqrt_m);y=np.log10(q.developed_da_dN_m_per_cycle);m,b=np.polyfit(x,y,1);fit=b+m*x;ss=np.sum((y-y.mean())**2);r2=1-np.sum((y-fit)**2)/ss if ss>0 else 1.;rec.update({"provisional_m64":m,"provisional_C64":10**b,"provisional_R2":r2,"measured_window_span_fraction":q.DeltaK_MPa_sqrt_m.max()/q.DeltaK_MPa_sqrt_m.min()-1,"candidate_response_class":"realistic Paris" if 2<=m<=6 else ("sustained low-rate growth" if m<=8 else "steep Paris")})
   elif classes=={"inert/censored"}:rec["candidate_response_class"]="inert/censored"
   elif "rapid/unstable" in classes:rec["candidate_response_class"]="rapid/unstable"
   elif "numerical invalid" in classes:rec["candidate_response_class"]="numerical invalid"
   else:rec["candidate_response_class"]="threshold-dominated/burst-arrest"
  summaries.append(rec)
 return pd.DataFrame(summaries)

def atomic_parquet(frame,path):
 tmp=path.with_suffix(path.suffix+".tmp");pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),tmp,compression="zstd");tmp.replace(path)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--campaign-root",type=Path,required=True);ap.add_argument("--atlas",type=Path,required=True);a=ap.parse_args()
 paths=sorted(a.campaign_root.glob("V914_ATLAS_*/DeltaK*_seed*/result.json"));rows=pd.DataFrame([trajectory(p) for p in paths])
 if rows.empty:print(json.dumps({"terminal_trajectories":0,"complete_candidates":0}));return 0
 summary=fits(rows);rows=rows.merge(summary,on="candidate_id",how="left",validate="many_to_one");atomic_parquet(rows,a.atlas/"n64_candidate_results.parquet")
 obs=[];props=["initiation_cycles","developed_da_dN_m_per_cycle","event_count","completed_physical_cycles","cumulative_extension_m"]
 for _,r in rows.iterrows():
  cond=json.dumps({"DeltaK_MPa_sqrt_m":r.DeltaK_MPa_sqrt_m,"R":r.R,"frequency_Hz":r.frequency_Hz,"temperature_K":r.temperature_K,"seed":int(r.seed)},sort_keys=True);prov=json.dumps({"result_json":r.result_json,"producer_commit":r.producer_commit,"physics_sha256":r.physics_sha256},sort_keys=True)
  for p in props:
   v=pd.to_numeric(r[p],errors="coerce")
   if np.isfinite(v):obs.append({"candidate_id":r.candidate_id,"physics_model":"v9.14_v7_CAL_TRANS_REV","fidelity":"n64","condition":cond,"property_name":p,"value":float(v),"uncertainty":np.nan,"status":r.response_class,"provenance":prov})
 for _,r in summary[summary.initial_descriptor_terminal_load_count.ge(3)].iterrows():
  for p in ["provisional_m64","provisional_C64","provisional_R2","measured_window_span_fraction"]:
   v=pd.to_numeric(r[p],errors="coerce")
   if np.isfinite(v):obs.append({"candidate_id":r.candidate_id,"physics_model":"v9.14_v7_CAL_TRANS_REV","fidelity":"n64_descriptor","condition":json.dumps({"R":.1,"frequency_Hz":1000.,"temperature_K":300.,"seed":1720},sort_keys=True),"property_name":p,"value":float(v),"uncertainty":np.nan,"status":r.candidate_response_class,"provenance":json.dumps({"source":"n64_candidate_results.parquet"},sort_keys=True)})
 new=pd.DataFrame(obs);opath=a.atlas/"candidate_property_observations.parquet";old=pq.read_table(opath).to_pandas() if opath.exists() else pd.DataFrame(columns=new.columns);combined=pd.concat([old,new],ignore_index=True).drop_duplicates(["candidate_id","physics_model","fidelity","condition","property_name","provenance"],keep="last");atomic_parquet(combined,opath)
 complete=summary[summary.initial_descriptor_terminal_load_count.ge(3)].copy();complete.to_csv(a.atlas/"n64_completed_candidate_descriptors.csv",index=False)
 adaptive=complete[complete.provisional_m64.le(8)&np.isfinite(complete.provisional_m64)].copy();adaptive["recommendation"]="ADD_1_TO_3_ADAPTIVE_LOADS";adaptive.to_csv(a.atlas/"n64_adaptive_load_recommendations.csv",index=False)
 promotion=complete[complete.provisional_m64.between(2,6)&np.isfinite(complete.provisional_m64)].copy();promotion["recommendation"]="IMMEDIATE_N128_4_TO_6_LOADS_40_TO_50UM";promotion.to_csv(a.atlas/"n128_promotion_queue.csv",index=False)
 print(json.dumps({"terminal_trajectories":len(rows),"complete_candidates":len(complete),"sustained_candidates":int(complete.candidate_response_class.isin(["realistic Paris","sustained low-rate growth","steep Paris"]).sum()),"m64_le_8":len(adaptive),"m64_2_6":len(promotion),"best_m64":float(complete.provisional_m64.min()) if complete.provisional_m64.notna().any() else None,"observation_rows":len(combined)},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
