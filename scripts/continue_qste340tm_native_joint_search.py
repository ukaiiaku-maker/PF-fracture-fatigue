#!/usr/bin/env python3
"""Substantive activity-stratified QSTE340TM joint inverse continuation."""
from __future__ import annotations
import argparse,csv,json,math,time
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution,least_squares

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.calibrate_qste340tm_stateful_exp_floor import build_records
from scripts.refit_qste340tm_faithful_reduction import (
    DATA,LOWER as OLD_LOWER,UPPER as OLD_UPPER,opening_prediction,pack,
    plastic_trajectory,starting_vectors,unpack,metrics,
)
from scripts.qualify_qste340tm_native_transfer import OUT,compare_case

PRIOR=DATA/"model_conditional/stateful_exp_floor_capacity_policy/faithful_retained_reduction/fit_faithful_reduction.json"
LOWER=OLD_LOWER.copy();UPPER=OLD_UPPER.copy()
# Original accepted material manifests span zero to 6.95e-3 /s.  The 1e-2 /s
# upper envelope is a documented constitutive-domain bound, not a fitted cap.
LOWER[8]=math.log(1e-10);UPPER[8]=math.log(1e-2)


def block_arrays(records):
    rv=np.asarray([float(r["R"]) for r in records]);obs=np.log10([float(r["observed_dadn_m_per_cycle"]) for r in records]);weights=np.empty(len(records))
    for r in sorted(set(rv)):weights[rv==r]=1/math.sqrt(np.sum(rv==r))
    return rv,obs,weights


def objective_residual(mode,records,trajectory,vector,rv,observed,weights):
    pred,raw,_=opening_prediction(vector,trajectory);error=np.log10(pred)-observed
    if mode=="absolute":residual=error*weights
    else:
        centered=np.empty_like(error)
        for r in sorted(set(rv)):
            mask=rv==r;centered[mask]=error[mask]-np.mean(error[mask])
        residual=np.r_[.5*error*weights,centered*weights]
    return np.r_[residual,max(0.,.25-float(np.min(raw)))*10.]


def fit_opening(mode,records,trajectory,plastic,rv,observed,weights,seed=None,nfev=16):
    positive=trajectory["stress"][trajectory["stress"]>0.];quantiles=np.quantile(positive,[.2,.55,.85])
    base=np.zeros(11);base[5:]=plastic
    starts=[]
    if seed is not None:starts.append(np.asarray(seed,float))
    for sigma,exponent in zip(quantiles,(.3,1.,2.)):
        starts.append(np.array([17.,20.,math.log(max(sigma/1e9,.05)),math.log(exponent),.5]))
    starts.append(np.array([17.,0.,math.log(max(np.median(quantiles)/1e9,.05)),0.,0.]))
    def residual(op):
        vector=base.copy();vector[:5]=op
        return objective_residual(mode,records,trajectory,vector,rv,observed,weights)
    candidates=[]
    for start in starts:
        try:
            sol=least_squares(residual,np.clip(start,LOWER[:5],UPPER[:5]),bounds=(LOWER[:5],UPPER[:5]),max_nfev=nfev,x_scale="jac",ftol=2e-9,xtol=2e-9,gtol=2e-9)
            z=residual(sol.x);candidates.append((float(z@z),sol))
        except (ValueError,FloatingPointError,np.linalg.LinAlgError,OverflowError):pass
    if not candidates:raise FloatingPointError("no conditional opening solution")
    return min(candidates,key=lambda z:z[0])


def candidate_summary(mode,records,vector,trajectory,prediction,objective):
    met=metrics(records,prediction,trajectory);rv=np.asarray([r["R"] for r in records]);obs=np.log10([r["observed_dadn_m_per_cycle"] for r in records]);err=np.log10(prediction)-obs
    centered=[]
    for r in sorted(set(rv)):
        mask=rv==r;centered.extend((err[mask]-np.mean(err[mask])).tolist())
    retained_span=float(np.max(trajectory["retained"])-np.min(trajectory["retained"]))
    mobile_span=float(np.max(trajectory["mobile"])-np.min(trajectory["mobile"]))
    return {"mode":mode,"objective":objective,"parameters_vector":vector.tolist(),"parameters":unpack(vector),
        "absolute_rmse_log10":met["overall_rmse_log10_rate"],"shape_rmse_log10":float(np.sqrt(np.mean(np.asarray(centered)**2))),
        "metrics":met,"activity":{"retained_span":retained_span,"mobile_span":mobile_span,
            "max_retained":float(np.max(trajectory["retained"])),"max_mobile":float(np.max(trajectory["mobile"])),
            "max_emission_per_cycle":float(np.max(trajectory["emission"])),"max_escape_per_cycle":float(np.max(trajectory["escaped"])),
            "max_recovery_per_cycle":float(np.max(trajectory["recovered"]))},
        "validity":{"max_density_m2":float(np.max(trajectory["density_m2"])),"recovery_at_bound":bool(abs(vector[8]-LOWER[8])<1e-5 or abs(vector[8]-UPPER[8])<1e-5)}}


class Profile:
    def __init__(self,mode,records,substeps,waveform):
        self.mode=mode;self.records=records;self.substeps=substeps;self.waveform=waveform;self.rv,self.observed,self.weights=block_arrays(records);self.cache={};self.summaries=[];self.base=np.asarray(json.load(open(PRIOR))["parameters_vector"],float)
    def detailed(self,plastic,seed=None):
        key=np.asarray(plastic,float).tobytes()
        if key in self.cache:return self.cache[key]
        vector=self.base.copy();vector[5:]=plastic
        try:
            trajectory=plastic_trajectory(vector,self.records,interval_substeps=self.substeps,waveform_steps=self.waveform)
            opening_objective,opening=fit_opening(self.mode,self.records,trajectory,plastic,self.rv,self.observed,self.weights,seed)
            vector[:5]=opening.x;prediction,raw,_=opening_prediction(vector,trajectory)
            density=float(np.max(trajectory["density_m2"]));validity=max(0.,math.log10(max(density,1.))-18.)**2*10.
            objective=opening_objective+validity;summary=candidate_summary(self.mode,self.records,vector,trajectory,prediction,objective)
            result=(objective,vector,trajectory,prediction,summary)
            self.summaries.append(summary)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError,OverflowError):result=(1e12,None,None,None,None)
        self.cache[key]=result;return result
    def __call__(self,plastic):return self.detailed(plastic)[0]


def stratified_population(records,size=24):
    rng=np.random.default_rng(20261022);population=rng.uniform(LOWER[5:],UPPER[5:],size=(size,6))
    seeds=[]
    for _,x in starting_vectors(records):
        for recovery in (1e-9,3.0677e-4,6.9475e-3):
            z=np.clip(x[5:].copy(),LOWER[5:],UPPER[5:]);z[3]=math.log(recovery);seeds.append(z)
    prior=np.asarray(json.load(open(PRIOR))["parameters_vector"],float)[5:];
    for emission,transport,sigc,recovery,mobile,retained in ((.8,.8,1.,1e-4,5.,20.),(1.2,2.5,2.,1e-6,20.,200.),(2.,1.,5.,3e-4,100.,100.),(1.,3.5,.5,7e-3,1.,500.)):
        z=prior.copy();z[:]=[emission,math.log(sigc),transport,math.log(recovery),math.log1p(mobile),math.log1p(retained)];seeds.append(z)
    for i,z in enumerate(seeds[:size]):population[i]=z
    return population


def local_pattern(profile,center,rounds=2):
    best=profile.detailed(center);x=np.asarray(center,float).copy();span=UPPER[5:]-LOWER[5:]
    for round_index in range(rounds):
        scale=.08/(round_index+1);trials=[]
        for j in range(len(x)):
            for sign in (-1.,1.):
                trial=x.copy();trial[j]=np.clip(trial[j]+sign*scale*span[j],LOWER[5+j],UPPER[5+j]);trials.append(trial)
        with ThreadPool(4) as pool:values=pool.map(profile,trials)
        index=int(np.argmin(values))
        if values[index]<best[0]:x=trials[index];best=profile.detailed(x)
    return best


def run_mode(mode,records,maxiter,substeps,waveform):
    print(f"[{mode}] profiled plastic acquisition",flush=True);profile=Profile(mode,records,substeps,waveform);started=time.time();init=stratified_population(records)
    with ThreadPool(4) as pool:
        de=differential_evolution(profile,list(zip(LOWER[5:],UPPER[5:])),init=init,seed=20261022+(mode=="shape"),maxiter=maxiter,polish=False,workers=pool.map,updating="deferred",tol=1e-5)
    ranked=sorted((profile.detailed(x) for x in de.population),key=lambda z:z[0]);active=[z for z in ranked if z[4] and (z[4]["activity"]["retained_span"]>1e-3 or z[4]["activity"]["mobile_span"]>1e-3)]
    refinements=[]
    print(f"[{mode}] local active-basin refinement",flush=True)
    for item in active[:2]:refinements.append(local_pattern(profile,np.asarray(item[1][5:]),1))
    best=min([profile.detailed(de.x),*refinements],key=lambda z:z[0])
    return {"best":best,"profile":profile,"optimizer":{"success":bool(de.success),"message":str(de.message),"nfev":int(de.nfev),"iterations":int(de.nit),"elapsed_s":time.time()-started,"local_active_basins":len(refinements)}}


def pareto(summaries):
    unique={tuple(np.round(x["parameters_vector"],10)):x for x in summaries}.values();front=[]
    for item in unique:
        if not any((other["absolute_rmse_log10"]<=item["absolute_rmse_log10"] and other["shape_rmse_log10"]<=item["shape_rmse_log10"] and (other["absolute_rmse_log10"]<item["absolute_rmse_log10"] or other["shape_rmse_log10"]<item["shape_rmse_log10"])) for other in unique):front.append(item)
    return sorted(front,key=lambda x:x["absolute_rmse_log10"])


def fixed_kmax(records,predictions):
    rows=[]
    for target in (45.,48.,50.):
        for r in (.1,.3,.5,.7):
            indices=[i for i,row in enumerate(records) if row["R"]==r];kmax=np.asarray([records[i]["DeltaK_MPa_sqrt_m"]/(1-r) for i in indices]);j=indices[int(np.argmin(abs(kmax-target)))]
            rows.append({"target_Kmax_MPa_sqrt_m":target,"R":r,"source_series_id":records[j]["series_id"],"source_order":records[j]["source_order"],"reported_DeltaK_MPa_sqrt_m":records[j]["DeltaK_MPa_sqrt_m"],"derived_Kmax_MPa_sqrt_m":records[j]["DeltaK_MPa_sqrt_m"]/(1-r),"observed_dadn_m_per_cycle":records[j]["observed_dadn_m_per_cycle"],**{name:float(value[j]) for name,value in predictions.items()}})
    return rows


def make_plot(records,cases,path,axis):
    rv=np.asarray([r["R"] for r in records]);dk=np.asarray([r["DeltaK_MPa_sqrt_m"] for r in records]);obs=np.asarray([r["observed_dadn_m_per_cycle"] for r in records]);fig,axes=plt.subplots(2,2,figsize=(11,8),sharey=True)
    for ax,r in zip(axes.flat,(.1,.3,.5,.7)):
        m=rv==r;x=dk[m] if axis=="DeltaK" else dk[m]/(1-r);order=np.argsort(x);ax.scatter(x,obs[m],s=10,c="#333",alpha=.45,label="measured")
        for name,pred,color,style in cases:ax.plot(x[order],pred[m][order],c=color,ls=style,lw=1.8,label=name)
        ax.set(xscale="log",yscale="log",title=f"R={r:.1f}",xlabel=(r"$\Delta K$ (MPa$\sqrt{m}$)" if axis=="DeltaK" else r"$K_{max}$ (MPa$\sqrt{m}$)"));ax.grid(True,which="both",alpha=.2)
    axes[0,0].set_ylabel("da/dN (m/cycle)");axes[1,0].set_ylabel("da/dN (m/cycle)");h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=4,loc="upper center",bbox_to_anchor=(.5,.98),frameon=False);fig.tight_layout(rect=(0,0,1,.92));fig.savefig(path,dpi=240,bbox_inches="tight");plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--iterations",type=int,default=6);ap.add_argument("--substeps",type=int,default=2);ap.add_argument("--waveform",type=int,default=64);args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);records=build_records(DATA)
    results={mode:run_mode(mode,records,args.iterations,args.substeps,args.waveform) for mode in ("absolute","shape")}
    finalists={};predictions={}
    for mode,result in results.items():
        key="absolute_rmse_log10" if mode=="absolute" else "shape_rmse_log10"
        chosen=min(result["profile"].summaries,key=lambda x:x[key]);vector=np.asarray(chosen["parameters_vector"]);traj=plastic_trajectory(vector,records,interval_substeps=args.substeps,waveform_steps=args.waveform);pred=opening_prediction(vector,traj)[0];summary=candidate_summary(mode,records,vector,traj,pred,chosen["objective"])
        frozen=plastic_trajectory(vector,records,interval_substeps=args.substeps,waveform_steps=args.waveform,evolve=False);frozen_pred=opening_prediction(vector,frozen)[0]
        convergence={"history":{},"waveform":{}}
        for n in (1,2,4,8):
            ct=plastic_trajectory(vector,records,interval_substeps=n,waveform_steps=args.waveform);cp=opening_prediction(vector,ct)[0];convergence["history"][str(n)]=metrics(records,cp,ct)
        for n in (32,64,128,256):
            ct=plastic_trajectory(vector,records,interval_substeps=args.substeps,waveform_steps=n);cp=opening_prediction(vector,ct)[0];convergence["waveform"][str(n)]=metrics(records,cp,ct)
        finalists[mode]={"summary":summary,"optimizer":result["optimizer"],"frozen_metrics":metrics(records,frozen_pred,frozen),"convergence":convergence,"prediction":pred.tolist(),"frozen_prediction":frozen_pred.tolist(),"state":{k:v.tolist() for k,v in traj.items() if k!="stress"}}
        predictions[mode+"_prediction"]=pred
    all_summaries=[x for result in results.values() for x in result["profile"].summaries];front=pareto(all_summaries);contrasts=fixed_kmax(records,predictions)
    # Qualify both selected active finalists with the actual native owner.
    native_active=[]
    ppmod=__import__('scripts.refit_qste340tm_faithful_reduction',fromlist=['plastic_parameters'])
    for mode in ("absolute","shape"):
        av=np.asarray(finalists[mode]["summary"]["parameters_vector"]);aplast=ppmod.plastic_parameters(av);aopen=unpack(av)
        for r in (.1,.3,.5,.7):
            block=[x for x in records if x["R"]==r];row=block[len(block)//2];state=np.array([aplast.initial_mobile_count,aplast.initial_retained_count,aplast.initial_source_activity]);native_active.append(compare_case(f"selected_{mode}_mid",row["DeltaK_MPa_sqrt_m"],r,state,aplast,aopen,512))
    payload={"schema":"qste340tm-native-transfer-joint-search/1","parent_commit":"3c840232","temperature_K":298.15,"frequency_Hz_provisional":10.,"DeltaK_convention":"reported DeltaK; Kmax derived exactly as DeltaK/(1-R)","recovery_domain":{"lower_per_s":1e-10,"upper_per_s":1e-2,"basis":"accepted original material manifests span 0 to 6.9475e-3 /s"},"finalists":finalists,"pareto_front":front,"fixed_Kmax_contrasts":contrasts,"active_native_qualification":native_active,"search":{"total_profiled_plastic_evaluations":sum(len(r["profile"].summaries) for r in results.values()),"modes":{m:r["optimizer"] for m,r in results.items()},"conditional_opening_derivatives_only":True},"limitations":["aggregate replay lacks moving-frame accumulated-slip/wake/source-refresh correspondence","strong-escape aggregate correspondence failed and such cases remain unresolved","fixed-Kmax rows summarize existing observations and are not extra likelihood data"]}
    (OUT/"joint_search.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    fields=list(records[0])+["absolute_prediction","absolute_frozen","shape_prediction","shape_frozen"]
    with (OUT/"joint_predictions.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields,lineterminator="\n");w.writeheader()
        for i,row in enumerate(records):w.writerow(dict(row)|{"absolute_prediction":finalists["absolute"]["prediction"][i],"absolute_frozen":finalists["absolute"]["frozen_prediction"][i],"shape_prediction":finalists["shape"]["prediction"][i],"shape_frozen":finalists["shape"]["frozen_prediction"][i]})
    with (OUT/"fixed_Kmax_contrasts.csv").open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=list(contrasts[0]),lineterminator="\n");w.writeheader();w.writerows(contrasts)
    cases=[("best absolute",np.asarray(finalists["absolute"]["prediction"]),"#0072B2","-"),("best shape",np.asarray(finalists["shape"]["prediction"]),"#D55E00","--"),("absolute frozen",np.asarray(finalists["absolute"]["frozen_prediction"]),"#56B4E9",":" )]
    make_plot(records,cases,OUT/"joint_fit_DeltaK.png","DeltaK");make_plot(records,cases,OUT/"joint_fit_Kmax.png","Kmax")
    print(json.dumps({"evaluations":payload["search"]["total_profiled_plastic_evaluations"],"absolute":finalists["absolute"]["summary"]["absolute_rmse_log10"],"shape":finalists["shape"]["summary"]["shape_rmse_log10"],"output":str(OUT)},indent=2));return 0

if __name__=="__main__":raise SystemExit(main())
