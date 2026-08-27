#!/usr/bin/env python3
"""Final scientific analysis for the A/PT03/PT08 R and nominal-C(T) study."""
from __future__ import annotations
import argparse, json, math, subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arrhenius_fracture.virtual_ct_v10230 import PRIMARY_CT,SENSITIVITY_CT,ct_geometry_factor,ct_load_from_k_pa_sqrt_m

NATIVE="A_NATIVE"; PT03="A_PT_03_oneD_v2_dbtt_TP_4895f9e5b44deea5"; PT08="A_PT_08_oneD_v2_dbtt_TP_f2817e7998cb7be6"
LABEL={NATIVE:"A_NATIVE",PT03:"PT03",PT08:"PT08"}; GATE_LOG=.05;GATE_M=.25
def read(path:Path):return json.loads(path.read_text())
def atomic_json(path:Path,x):
 t=path.with_suffix(path.suffix+".tmp");t.write_text(json.dumps(x,indent=2,sort_keys=True,default=float)+"\n");t.replace(path)
def ledger_sum(row:dict,needle:str)->float:
 return sum(float(v or 0) for k,v in (row.get("coupled_hazard_ledger_delta") or {}).items() if needle in k)
def fit_rows(points:pd.DataFrame):
 fits=[];slopes=[]
 for (o,R),g in points.groupby(["option","R"]):
  g=g[g.stable_growth & g.developed_da_dN.notna() & (g.developed_da_dN>0)].sort_values("deltaK_driver_MPa_sqrt_m")
  for (_,a),(_,b) in zip(g.iloc[:-1].iterrows(),g.iloc[1:].iterrows()):
   m=math.log(b.developed_da_dN/a.developed_da_dN)/math.log(b.deltaK_driver_MPa_sqrt_m/a.deltaK_driver_MPa_sqrt_m)
   slopes.append({"option":o,"R":R,"deltaK_low":a.deltaK_driver_MPa_sqrt_m,"deltaK_high":b.deltaK_driver_MPa_sqrt_m,"local_m":m})
  if len(g)>=3:
   x=np.log(g.deltaK_driver_MPa_sqrt_m.to_numpy(float));y=np.log(g.developed_da_dN.to_numpy(float));m,b=np.polyfit(x,y,1);pred=m*x+b
   fits.append({"option":o,"R":R,"admissible_points":len(g),"m":m,"C":math.exp(b),"R2":1-np.sum((y-pred)**2)/max(np.sum((y-y.mean())**2),1e-300),
    "rate_span":math.exp(y.max()-y.min()),"curvature":max([x["local_m"] for x in slopes if x["option"]==o and x["R"]==R],default=np.nan)-min([x["local_m"] for x in slopes if x["option"]==o and x["R"]==R],default=np.nan)})
  else:fits.append({"option":o,"R":R,"admissible_points":len(g),"m":None,"C":None,"R2":None,"rate_span":None,"curvature":None})
 return pd.DataFrame(fits),pd.DataFrame(slopes)

def collect(root:Path):
 jobs=pd.read_csv(root/"A_PT03_PT08_R_job_registry.csv");prov=read(root/"A_PT03_PT08_R_provenance_manifest.json");vh={v["composite_candidate_id"]:v for v in prov["variants"]}
 points=[];second=[];events=[];states=[];transfers=[];ctwindows=[]
 for j in jobs.itertuples():
  if j.stage=="EXPLICIT_PREFLIGHT":continue
  out=Path(j.result_path);summary=read(out/"developed_fatigue_growth_summary.json");dev=summary.get("developed_interval") or {}
  row={"job_id":j.job_id,"stage":j.stage,"option":j.option,"R":float(j.R),"Kmax_MPa_sqrt_m":float(j.kmax),"Kmin_MPa_sqrt_m":float(j.R*j.kmax),
   "deltaK_driver_MPa_sqrt_m":float(j.deltaK_driver_MPa_sqrt_m),"deltaK_nominal_full_MPa_sqrt_m":float(j.deltaK_driver_MPa_sqrt_m),
   "deltaK_nominal_tensile_MPa_sqrt_m":float(j.kmax-max(j.R*j.kmax,0)),"Kmax_nominal_MPa_sqrt_m":float(j.kmax),
   "developed_da_dN":dev.get("da_dN"),"event_count":summary.get("event_count"),"cycles":summary.get("cycles_consumed"),
   "final_extension_um":summary.get("final_projected_extension_um"),"target_reached":summary.get("target_reached"),"stable_growth":summary.get("stable_growth_provisional"),
   "stationarity_ratio":summary.get("late_to_early_rate_ratio"),"terminal_classification":j.status,"result_path":j.result_path,
   "seed":int(j.seed),"n_bins":80,"temperature_K":300,"frequency_Hz":1000,"branch":prov["branch"],"head":prov["launch_head"],
   "production_solver_hash":prov["production_solver_hash"],"common_physics_hash":prov["common_physics_hash"],"composite_hash":vh[j.option]["complete_composite_material_hash"],
   "acceleration_mode":j.acceleration_mode,"reused":bool(j.reused),"resumed":False,"stationarity_classification":"STABLE" if summary.get("stable_growth_provisional") else "UNSTABLE"}
  audit=read(out/"kinetic_tip_cell_audit_v101.json");recs=audit.get("records",[])
  if recs:
   last=recs[-1]; snap=(last.get("coupled_hazard_active_state_snapshot") or {}).get("diagnostics",{})
   row.update({"mobile_count":last.get("state_mobile_count"),"retained_count":last.get("state_retained_count"),"mobile_to_retained":float(last.get("state_mobile_count",0))/max(float(last.get("state_retained_count",0)),1e-300),
    "tip_radius_m":last.get("persistent_tip_radius_m"),"internal_stress_Pa":last.get("persistent_sigma_back_Pa"),"shielding_Pa_sqrt_m":last.get("state_active_K_shield_signed_Pa_sqrt_m"),
    "physical_return":sum(ledger_sum(x,"cumulative_physical_returned_mobile") for x in recs),"returned_source_slip":sum(ledger_sum(x,"cumulative_cancelled_source_slip") for x in recs),
    "gross_source_activity":sum(ledger_sum(x,"cumulative_gross_source_activity") for x in recs),"gross_return_activity":sum(ledger_sum(x,"cumulative_gross_return_activity") for x in recs),
    "far_field_escape":sum(ledger_sum(x,"cumulative_escaped_mobile") for x in recs),"wake_transfer":sum(ledger_sum(x,"cumulative_source_slip_wake_transfer") for x in recs),
    "net_source_linked_blunting":float(last.get("state_micro_advance_total_m",0)),"cleavage_action":sum(float(x.get("physical_hazard_action_block",0) or 0) for x in recs),
    "emission_action":sum(float(x.get("persistent_aggregate_emission_hazard_s",0) or 0) for x in recs)})
   for i,x in enumerate(recs):
    states.append({**{k:row[k] for k in ["job_id","stage","option","R","Kmax_MPa_sqrt_m","Kmin_MPa_sqrt_m","deltaK_driver_MPa_sqrt_m","seed","result_path","branch","head","production_solver_hash","common_physics_hash","composite_hash","acceleration_mode","stationarity_classification"]},
      "record_index":i,"cycles_consumed":x.get("cycles_consumed"),"mobile_count":x.get("state_mobile_count"),"retained_count":x.get("state_retained_count"),"tip_radius_m":x.get("persistent_tip_radius_m"),
      "internal_stress_Pa":x.get("persistent_sigma_back_Pa"),"shielding_Pa_sqrt_m":x.get("state_active_K_shield_signed_Pa_sqrt_m"),"hazard_action":x.get("physical_hazard_action_block"),
      "emission_activity":x.get("persistent_aggregate_emission_hazard_s"),"physical_return":ledger_sum(x,"cumulative_physical_returned_mobile"),"returned_source_slip":ledger_sum(x,"cumulative_cancelled_source_slip"),
      "gross_source_activity":ledger_sum(x,"cumulative_gross_source_activity"),"gross_return_activity":ledger_sum(x,"cumulative_gross_return_activity"),"far_field_escape":ledger_sum(x,"cumulative_escaped_mobile"),"wake_transfer":ledger_sum(x,"cumulative_source_slip_wake_transfer")})
  for e in summary.get("event_measurements",[]):events.append(row|e)
  if j.stage in ("PRIMARY","PRIMARY_REUSE"):points.append(row)
  if j.stage in ("SECOND_SEED","SECOND_SEED_REUSE"):second.append(row)
  # Mode-A load-adjusted history or Mode-B fixed-load history at each event endpoint.
  constant=j.stage=="CONSTANT_LOAD_CT";pmax=ct_load_from_k_pa_sqrt_m(j.kmax*1e6,PRIMARY_CT,PRIMARY_CT.initial_crack_m)
  for e in summary.get("event_measurements",[]):
   ext=float(e["projected_extension_post_m"]);crack=PRIMARY_CT.initial_crack_m+ext
   if constant:kmax=j.kmax*ct_geometry_factor(crack/PRIMARY_CT.width_m)/ct_geometry_factor(.5);load=pmax
   else:kmax=j.kmax;load=ct_load_from_k_pa_sqrt_m(kmax*1e6,PRIMARY_CT,crack)
   transfers.append({"job_id":j.job_id,"stage":j.stage,"option":j.option,"R":j.R,"seed":j.seed,"projected_extension_m":ext,"a_macro_m":crack,"a_over_W":crack/PRIMARY_CT.width_m,
    "K_nominal_macro_MPa_sqrt_m":kmax,"K_driver_supplied_to_local_solver_MPa_sqrt_m":kmax,"J_equivalence_scale":1.0,"Pmax_N":load,"Pmin_N":j.R*load,
    "deltaK_nominal_full_MPa_sqrt_m":(1-j.R)*kmax,"deltaK_nominal_tensile_MPa_sqrt_m":kmax-max(j.R*kmax,0),"tip_radius_used":False,"fit_from_growth":False,"result_path":j.result_path})
  if constant:
   for w in summary.get("moving_windows",[]):
    ext=float(w["window_mid_m"]);kmax=j.kmax*ct_geometry_factor((PRIMARY_CT.initial_crack_m+ext)/PRIMARY_CT.width_m)/ct_geometry_factor(.5)
    ctwindows.append({"job_id":j.job_id,"option":j.option,"R":j.R,"seed":j.seed,"window_start_m":w["window_start_m"],"window_stop_m":w["window_stop_m"],"window_mid_m":ext,"da_dN":w.get("da_dN"),
      "Kmax_nominal_mid_MPa_sqrt_m":kmax,"deltaK_nominal_full_mid_MPa_sqrt_m":kmax*(1-j.R),"deltaK_nominal_tensile_mid_MPa_sqrt_m":kmax-max(j.R*kmax,0),"W_m":PRIMARY_CT.width_m,"B_m":PRIMARY_CT.thickness_m,"result_path":j.result_path})
 return pd.DataFrame(points),pd.DataFrame(second),pd.DataFrame(events),pd.DataFrame(states),pd.DataFrame(transfers),pd.DataFrame(ctwindows)

def figures(root,p,ratios,slopes,states,transfer,windows):
 out=root/"figures";out.mkdir(exist_ok=True)
 def save(name,title,x,y,group="option",data=None,log=True):
  d=p if data is None else data;fig,ax=plt.subplots(figsize=(9,5.5))
  for key,g in d.groupby(group):ax.plot(g[x],g[y],"o-",label=LABEL.get(key,str(key)))
  if log:ax.set_xscale("log");ax.set_yscale("log")
  ax.set_xlabel(x);ax.set_ylabel(y);ax.set_title(title);ax.grid(alpha=.25);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/f"{name}.png",dpi=180);plt.close(fig)
 save("DADN_VS_DRIVER_DELTAK_BY_R","Developed growth vs local driver range","deltaK_driver_MPa_sqrt_m","developed_da_dN",group=["option","R"])
 save("DADN_VS_NOMINAL_FULL_DELTAK_BY_R","Developed growth vs nominal full range","deltaK_nominal_full_MPa_sqrt_m","developed_da_dN",group=["option","R"])
 save("DADN_VS_NOMINAL_TENSILE_DELTAK_BY_R","Developed growth vs nominal tensile range","deltaK_nominal_tensile_MPa_sqrt_m","developed_da_dN",group=["option","R"])
 save("DADN_VS_NOMINAL_KMAX_BY_R","Developed growth vs nominal Kmax","Kmax_nominal_MPa_sqrt_m","developed_da_dN",group=["option","R"])
 save("PT_TO_NATIVE_RATE_RATIO_VS_R_AND_KMAX","PT/native log rate ratio","Kmax_MPa_sqrt_m","S_PT_log10",group=["option","R"],data=ratios,log=False)
 save("LOCAL_PARIS_SLOPES_BY_R","Adjacent-point local slopes","deltaK_high","local_m",group=["option","R"],data=slopes,log=False)
 save("LOCAL_TO_NOMINAL_K_TRANSFER","Energy-equivalent local/nominal transfer","K_nominal_macro_MPa_sqrt_m","K_driver_supplied_to_local_solver_MPa_sqrt_m",group="stage",data=transfer,log=False)
 save("CONSTANT_LOAD_CT_APPARENT_CURVES","Constant-load C(T) window rates","deltaK_nominal_full_mid_MPa_sqrt_m","da_dN",group=["option","R"],data=windows)
 final=states.sort_values("record_index").groupby("job_id").tail(1)
 save("MOBILE_RETAINED_VS_R","Terminal mobile population vs R","R","mobile_count",group="option",data=final,log=False)
 save("PHYSICAL_RETURN_AND_SOURCE_CANCELLATION_VS_R","Physical return vs R","R","physical_return",group="option",data=final,log=False)
 save("RADIUS_INTERNAL_STRESS_SHIELDING_VS_R","Terminal internal stress vs R","R","internal_stress_Pa",group="option",data=final,log=False)
 save("NET_BLUNTING_HAZARD_EVENT_HISTORY_VS_R","Cleavage hazard action vs record","record_index","hazard_action",group=["option","R"],data=states,log=False)
 fig,ax=plt.subplots(figsize=(9,5.5));q=ratios.groupby(["option","R"]).S_PT_log10.apply(lambda x:max(abs(x))).unstack();q.plot.bar(ax=ax);ax.axhline(.05,color="k",ls="--");ax.set_ylabel("max |log10 PT/native|");ax.set_title("Final R-ratio mechanism summary");fig.tight_layout();fig.savefig(out/"FINAL_R_RATIO_MECHANISM_SUMMARY.png",dpi=180);plt.close(fig)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
 p,s,e,st,tr,w=collect(root);p.to_csv(root/"A_PT03_PT08_R_developed_points.csv",index=False);s.to_csv(root/"A_PT03_PT08_R_second_seed_points.csv",index=False);e.to_csv(root/"A_PT03_PT08_R_event_results.csv",index=False);st.to_parquet(root/"A_PT03_PT08_R_state_histories.parquet",index=False)
 fits,slopes=fit_rows(p);fits.to_csv(root/"A_PT03_PT08_R_paris_analysis.csv",index=False);slopes.to_csv(root/"A_PT03_PT08_R_local_slopes.csv",index=False)
 native=p[p.option==NATIVE].set_index(["R","Kmax_MPa_sqrt_m"]);rat=[]
 for x in p[p.option!=NATIVE].itertuples():
  n=native.loc[(x.R,x.Kmax_MPa_sqrt_m)];S=math.log10(x.developed_da_dN/n.developed_da_dN);base=math.log10(p[(p.option==x.option)&(p.R==.1)&(p.Kmax_MPa_sqrt_m==x.Kmax_MPa_sqrt_m)].developed_da_dN.iloc[0]/native.loc[(.1,x.Kmax_MPa_sqrt_m)].developed_da_dN)
  rat.append({"option":x.option,"R":x.R,"Kmax_MPa_sqrt_m":x.Kmax_MPa_sqrt_m,"S_PT_log10":S,"delta_R_S_PT":S-base,"rate_ratio":10**S})
 ratios=pd.DataFrame(rat);ratios.to_csv(root/"A_PT03_PT08_R_PT_native_rate_ratios.csv",index=False)
 p.to_csv(root/"A_PT03_PT08_nominal_deltaK_points.csv",index=False);w.to_csv(root/"A_PT03_PT08_constant_load_CT_windows.csv",index=False);tr.to_csv(root/"A_PT03_PT08_local_to_nominal_K_transfer.csv",index=False)
 gs=[]
 for geom,name in [(PRIMARY_CT,"W10mm"),(SENSITIVITY_CT,"W25mm")]:
  for ext in (0,50e-6,102.66778366452745e-6):
   ratio=ct_geometry_factor((geom.initial_crack_m+ext)/geom.width_m)/ct_geometry_factor(.5)
   gs.append({"geometry":name,"W_m":geom.width_m,"B_m":geom.thickness_m,"projected_extension_m":ext,"K_over_initial_K":ratio,"deltaK_over_initial_deltaK":ratio,"static_transfer_scale":1.0,"local_slope_change_static":0.0})
 pd.DataFrame(gs).to_csv(root/"A_PT03_PT08_geometry_sensitivity.csv",index=False)
 maxlog=float(max(abs(ratios.S_PT_log10)));maxdr=float(max(abs(ratios.delta_R_S_PT)));native_m=fits[fits.option==NATIVE].set_index("R").m;maxdm=float(max(abs(x.m-native_m.loc[x.R]) for x in fits[fits.option!=NATIVE].itertuples()))
 # Paired seed spread at Kmax=18, compared with same primary point.
 spread=0.0
 for o in [PT03,PT08]:
  for R in (.5,.1,-.95):
   q=s[(s.option==o)&(s.R==R)].developed_da_dN.iloc[0];qn=s[(s.option==NATIVE)&(s.R==R)].developed_da_dN.iloc[0];ss=math.log10(q/qn)
   pp=ratios[(ratios.option==o)&(ratios.R==R)&(ratios.Kmax_MPa_sqrt_m==18)].S_PT_log10.iloc[0];spread=max(spread,abs(ss-pp))
 significant=maxlog>=GATE_LOG or maxdm>=GATE_M
 classification="PT_R_INVARIANT" if not significant else "TRUE_LOCAL_KINETIC_DIVERGENCE"
 geometry_change_10=float(ct_geometry_factor((PRIMARY_CT.initial_crack_m+102.66778366452745e-6)/PRIMARY_CT.width_m)/ct_geometry_factor(.5)-1)
 geometry_change_25=float(ct_geometry_factor((SENSITIVITY_CT.initial_crack_m+102.66778366452745e-6)/SENSITIVITY_CT.width_m)/ct_geometry_factor(.5)-1)
 decision={"schema":"A_PT03_PT08_R_final_decision_v1","primary_classification":classification,"mechanism_qualifiers":[],"significance_gates":{"log10_rate":GATE_LOG,"global_m":GATE_M},
  "maximum_abs_log10_PT_native":maxlog,"maximum_abs_delta_R_S_PT":maxdr,"maximum_abs_global_m_shift":maxdm,"paired_seed_log_effect_change":spread,"effect_survives_seed":significant and maxlog>spread,
  "deltaK_semantics":"LOCAL_TIP_EFFECTIVE_K","nominal_transfer":"unit_scale_from_J_equivalence","closure_corrected_deltaK_reported":False,
  "constant_load_K_increase_100um_W10_fraction":geometry_change_10,"constant_load_K_increase_100um_W25_fraction":geometry_change_25,
  "n128_merited":significant and maxlog>spread,"broader_PT_bank_merited":significant and maxlog>spread,
  "answers":{
   "1_R_changes_PT_dadN":"yes" if significant else "no measurable effect above the predeclared gates",
   "2_positive_or_negative":"unresolved/no material sensitivity" if not significant else "see matched-R ratio table",
   "3_controlling_pathway":"none resolved" if not significant else "local event kinetics; inspect mechanism qualifiers",
   "4_paired_seed":"no material effect to challenge" if not significant else ("survives" if maxlog>spread else "does not survive"),
   "5_deltaK_semantics":"LOCAL_TIP_EFFECTIVE_K, not the FEM outer-boundary probe",
   "6_nominal_curve_difference":"unit J-equivalent mapping for fixed-K mode; horizontal values coincide",
   "7_static_vs_evolving":"static transfer leaves slopes unchanged; constant-load geometry changes K by the recorded fractions",
   "8_negative_R_axes":"full range is 1.95*Kmax; tensile-only range is Kmax; DeltaK_eff omitted without validated opening",
   "9_load_control":"quantified in constant-load window table; W=10 mm K change fraction is %.6g"%geometry_change_10,
   "10_followup":"n=128/broader bank warranted" if significant and maxlog>spread else "no n=128 or broader PT-bank study warranted"}}
 atomic_json(root/"A_PT03_PT08_R_final_decision.json",decision)
 md="# A/PT03/PT08 R-ratio and nominal-DeltaK decision\n\n**Primary classification: `%s`.**\n\n"%classification+"\n".join(f"{i}. {v}" for i,v in enumerate(decision["answers"].values(),1))+"\n"
 (root/"A_PT03_PT08_R_final_decision.md").write_text(md)
 figures(root,p,ratios,slopes,st,tr,w)
 print(json.dumps({"classification":classification,"primary_points":len(p),"second_seed":len(s),"constant_windows":len(w),"max_abs_log10":maxlog},indent=2))
if __name__=="__main__":main()
