"""Deterministic analytical-only thermodynamic joint fatigue/fracture search."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import csv, hashlib, json, math, sys, warnings

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
from scipy.stats import qmc
from scipy.special import gammainc

from arrhenius_fracture.material_manifest import ExpFloorBarrier,KB_EV_PER_K
from arrhenius_fracture.inverse_fatigue_barrier_design_v10230 import waveform_fraction,ideal_mode_I_drive_factors
from scripts.thermodynamic_joint_barrier_v10230 import (
    EV_J,TR_K,Basis,ThermodynamicSurface,candidate_surface_from_parameters,
    cooperative_rate,derive_guard_sigma,exp_floor_reference,weighted_quantile,
)
from scripts.analyze_forward_temperature_v10230 import source_rows
from scripts.analyze_row_renewal_forward_v10230 import Controls,TransientBlunting

ART=ROOT/'artifacts/thermodynamic_joint_barrier_search'
RUN=ROOT/'runs/thermodynamic_joint_barrier_search_v1'
PARENTS=('P25_TRANSFER_V1_RANK1','P40_TRANSFER_CALIBRATED_GEN2')
KGRID=np.array([12,12.75,13.5,15,16.5,18,19.5,21,24.3],float)
TEMPS=np.arange(300.,1200.1,25.)
COARSE=np.array([300.,450.,600.,750.,900.,1050.,1200.])
RATES=np.array([.0005,.005,.05])
THRESHOLDS={'LN2':math.log(2),'UNIT_ACTION':1.,'SEED_1720':.07509316036236147}
HITS=3.2732414351776242;TAU=6.992153587194454e-7
RADIUS=1e-6;CAP=30e9;EVENT_LENGTH=5e-6;FREQUENCY=1000.


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def cid(parent,index,stage='L'):return f"{parent.split('_')[0]}_TJBS_{stage}_{index:06d}"


def write_csv(path,rows):
    rows=list(rows)
    if not rows:
        path.write_text('status\nEMPTY\n');return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def write_partitioned(df,path,partition='part-0000.parquet',limit=50000):
    path.mkdir(parents=True,exist_ok=True)
    for old in path.glob('part-*.parquet'):old.unlink()
    for i,start in enumerate(range(0,len(df),limit)):
        df.iloc[start:start+limit].to_parquet(path/f'part-{i:04d}.parquet',index=False,compression='zstd')
    (path/'_schema.json').write_text(json.dumps({c:str(t) for c,t in df.dtypes.items()},indent=2)+'\n')


def physical_controls():
    data=pd.read_csv(ROOT/'artifacts/prospective_paris_candidates/physical_developed_rates.csv')
    rows={}
    for parent in PARENTS:
        q=data[(data.candidate_id==parent)&(data.R==.1)&(data.seed==1720)&data.Kmax.isin(KGRID)].sort_values('Kmax')
        if list(q.Kmax)!=list(KGRID) or not q.developed_qualified.all():raise ValueError('physical parent grid incomplete')
        rates=q.physical_rate.to_numpy(float);logs=np.log(rates)
        local=np.diff(logs)/np.diff(np.log(KGRID));global_slope=float(np.polyfit(np.log(KGRID),logs,1)[0])
        rows[parent]=dict(rates=rates,local=local,global_slope=global_slope,frame=q)
    return rows


def action_weights(surface,T,K,R=.1,n=512,radius=RADIUS):
    h=waveform_fraction(R,n);stress=np.minimum(K*1e6*h/np.sqrt(2*np.pi*radius),CAP)
    effective=cooperative_rate(surface.raw_rate_s(stress,T),HITS,TAU)
    return stress,effective


def active_stresses(rows):
    physical=pd.read_csv(ROOT/'artifacts/prospective_paris_candidates/physical_state_summary.csv')
    result={}
    factors=ideal_mode_I_drive_factors(30.,.5)
    for parent in PARENTS:
        row,m,_=rows[parent]
        s12,w12=action_weights(SimpleNamespace(raw_rate_s=m.cleavage.rate),300.,12.)
        s18,w18=action_weights(SimpleNamespace(raw_rate_s=m.cleavage.rate),300.,18.)
        state=physical[(physical.candidate_id==parent)&(physical.R==.1)&(physical.seed==1720)&(physical.Kmax==18)].iloc[0]
        h=waveform_fraction(.1,512);opening=np.minimum(18e6*h/np.sqrt(2*np.pi*float(state.terminal_radius_m)),CAP)
        drive=np.maximum(factors[:,None]*opening[None,:]-float(state.sigma_back),0.)
        ew=m.emission.rate(drive,300.)
        result[parent]=dict(cleavage_low_Pa=weighted_quantile(s12,w12,.05),cleavage_active_Pa=weighted_quantile(s18,w18,.5),cleavage_p05_Pa=weighted_quantile(s18,w18,.05),cleavage_p95_Pa=weighted_quantile(s18,w18,.95),emission_active_Pa=weighted_quantile(drive.ravel(),ew.ravel(),.5),emission_p05_Pa=weighted_quantile(drive.ravel(),ew.ravel(),.05),emission_p95_Pa=weighted_quantile(drive.ravel(),ew.ravel(),.95),state_source='physical_state_summary.csv Kmax=18 R=0.1 seed=1720')
    return result


def base_action(manifest,K,T=300.,R=.1,n=512):
    h=waveform_fraction(R,n);stress=np.minimum(K*1e6*h/np.sqrt(2*np.pi*RADIUS),CAP)
    return float(np.mean(cooperative_rate(manifest.cleavage.rate(stress,T),HITS,TAU))/FREQUENCY)


def surface_action(surface,K,T=300.,R=.1,n=512):
    h=waveform_fraction(R,n);stress=np.minimum(K*1e6*h/np.sqrt(2*np.pi*RADIUS),CAP)
    return float(np.mean(cooperative_rate(surface.raw_rate_s(stress,T),HITS,TAU))/FREQUENCY)


def fatigue_metrics(surface,parent_manifest,physical):
    base=np.array([base_action(parent_manifest,k,n=512) for k in KGRID])
    candidate=np.array([surface_action(surface,k,n=512) for k in KGRID])
    predicted=physical['rates']*candidate/base
    errors=np.log10(predicted/physical['rates'])
    local=np.diff(np.log(predicted))/np.diff(np.log(KGRID));global_slope=float(np.polyfit(np.log(KGRID),np.log(predicted),1)[0])
    base_ceiling=max(float(np.max(gammainc(HITS,parent_manifest.cleavage.rate(np.minimum(k*1e6*waveform_fraction(.1,512)/np.sqrt(2*np.pi*RADIUS),CAP),300.)*TAU))) for k in KGRID)
    cand_ceiling=max(float(np.max(gammainc(HITS,surface.raw_rate_s(np.minimum(k*1e6*waveform_fraction(.1,512)/np.sqrt(2*np.pi*RADIUS),CAP),300.)*TAU))) for k in KGRID)
    return dict(rms_log10_rate_error=float(np.sqrt(np.mean(errors**2))),max_log10_rate_error=float(np.max(abs(errors))),global_slope=float(global_slope),global_slope_change=float(global_slope-physical['global_slope']),max_adjacent_local_slope_change=float(np.max(abs(local-physical['local']))),minimum_rate=float(predicted.min()),maximum_rate=float(predicted.max()),monotonic_positive=bool(np.all(np.diff(predicted)>0)&np.all(predicted>0)),new_renewal_ceiling=bool(cand_ceiling>=.99 and base_ceiling<.99),rates=predicted,local_slopes=local)


def single_exp_feasibility(rows,physical):
    results=[];h=waveform_fraction(.1,512);stress=KGRID[:,None]*1e6*h[None,:]/np.sqrt(2*np.pi*RADIUS)
    for parent in PARENTS:
        row,m,_=rows[parent]
        base=np.mean(cooperative_rate(m.cleavage.rate(np.minimum(stress,CAP),300.),HITS,TAU),axis=1)
        for target in (1.,1.25,1.5,1.75,2.):
            best=None
            for start in ([1.5,.3,2,.2],[2.5,1,4,.4],[1,.1,8,.5],[6,3,1,.2]):
                def evaluate(x):
                    b=ExpFloorBarrier(target,0,x[0]*1e9,0,x[1],x[2],x[3],m.cleavage.floor_min_eV,m.cleavage.floor_max_fraction,300,m.cleavage.attempt_frequency_s)
                    action=np.mean(cooperative_rate(b.rate(np.minimum(stress,CAP),300.),HITS,TAU),axis=1)
                    pred=physical[parent]['rates']*action/base
                    local=np.diff(np.log(pred))/np.diff(np.log(KGRID))
                    return b,pred,np.log10(pred/physical[parent]['rates']),local
                def residual(x):
                    _,_,err,local=evaluate(x);return np.r_[err/0.05,(local-physical[parent]['local'])/.30]
                fit=least_squares(residual,start,bounds=([.2,.02,.25,.001],[8,5,12,.8]),max_nfev=2500,xtol=1e-11,ftol=1e-11,gtol=1e-11)
                b,pred,err,local=evaluate(fit.x);global_slope=float(np.polyfit(np.log(KGRID),np.log(pred),1)[0])
                metrics=dict(rms=float(np.sqrt(np.mean(err**2))), maximum=float(np.max(abs(err))),
                    global_change=float(global_slope-physical[parent]['global_slope']),
                    local_change=float(np.max(abs(local-physical[parent]['local']))))
                score=max(metrics['rms']/.05,metrics['maximum']/.1,abs(metrics['global_change'])/.15,metrics['local_change']/.3)
                item=(score,fit,b,metrics)
                if best is None or item[0]<best[0]:best=item
            score,fit,b,metrics=best
            results.append(dict(parent_id=parent,cleavage_zero_target_eV=target,status='PASS' if score<=1 else 'SINGLE_EXP_CANNOT_SEPARATE_LOW_STRESS_SCALE_AND_FATIGUE_WINDOW',gate_score=score,sigc_GPa=b.sigc0_Pa/1e9,alpha=b.alpha,exponent=b.exponent,floor_fraction=b.floor_fraction,nfev=fit.nfev,at_bound=bool(np.min(np.r_[fit.x-[.2,.02,.25,.001],[8,5,12,.8]-fit.x])<1e-7),**metrics))
    return results


def map_sobol(parent,n,seed,stage='S'):
    u=qmc.Sobol(9,scramble=True,seed=seed).random_base2(int(math.log2(n)));out=[]
    for i,x in enumerate(u):
        preferred=x[7]>=.25
        if preferred:
            z=(x[7]-.25)/.75
            se=28+14*(z/.6) if z<.6 else 20+40*((z-.6)/.4)
        else:se=20*x[7]/.25
        out.append(dict(candidate_id=cid(parent,i,stage),parent_id=parent,family='GUARD_LINEAR_ENTROPY',stage=stage,cleavage_zero_target_eV=1+x[0],guard_log10_q_low=-8+5*x[1],guard_exponent=1.5+6.5*x[2],cleavage_entropy_zero_kB=-50+100*x[3],cleavage_entropy_active_kB=-40+80*x[4],cleavage_entropy_infinity_kB=-20+40*x[5],emission_entropy_active_kB=se,emission_entropy_infinity_kB=-5+20*x[8],emission_stratum='PREFERRED_20_60_DENSE_30_40' if preferred else 'LOW_ENTROPY_MECHANISM_CONTROL',cleavage_heat_capacity_base_kB=0.,cleavage_heat_capacity_guard_kB=0.,emission_heat_capacity_base_kB=0.,complexity=2))
    return out


def structured(parent,n=1536):
    target=[1,1.25,1.5,1.75,2];uq=[-3,-4,-5,-6,-7,-8];ng=[2,4,6,8]
    sc0=[-40,-20,-10,0,10,20,40];sca=[-40,-20,-10,0,10,20,40];sci=[-20,-10,0,10,20]
    sea=[0,10,20,30,35,40,50,60];sei=[-5,0,5,10,15]
    sob=qmc.Sobol(9,scramble=True,seed=260000+(25 if parent.startswith('P25') else 40)).random_base2(11)
    out=[]
    for i,x in enumerate(sob[:n]):
        pick=lambda a,j:a[min(int(x[j]*len(a)),len(a)-1)]
        out.append(dict(candidate_id=cid(parent,i,'T'),parent_id=parent,family='GUARD_LINEAR_ENTROPY',stage='STRUCTURED',cleavage_zero_target_eV=pick(target,0),guard_log10_q_low=pick(uq,1),guard_exponent=pick(ng,2),cleavage_entropy_zero_kB=pick(sc0,3),cleavage_entropy_active_kB=pick(sca,4),cleavage_entropy_infinity_kB=pick(sci,5),emission_entropy_active_kB=pick(sea,6),emission_entropy_infinity_kB=pick(sei,7),emission_stratum='PREFERRED_20_60_DENSE_30_40' if pick(sea,6)>=20 else 'LOW_ENTROPY_MECHANISM_CONTROL',cleavage_heat_capacity_base_kB=0.,cleavage_heat_capacity_guard_kB=0.,emission_heat_capacity_base_kB=0.,complexity=2))
    return out


def thermodynamic_check(params,manifest,active):
    try:c,e=candidate_surface_from_parameters(manifest,params,active['cleavage_low_Pa'],active['cleavage_active_Pa'],active['emission_active_Pa'])
    except (ValueError,np.linalg.LinAlgError) as exc:return None,dict(thermodynamic_pass=False,rejection='COEFFICIENT_CONSTRUCTION:'+str(exc))
    ez=float(e.entropy_kB(0.,300.))
    stress=np.unique(np.r_[0.,c.bases[-1].sigma_Pa,active['cleavage_p05_Pa'],active['cleavage_active_Pa'],active['cleavage_p95_Pa'],active['emission_p05_Pa'],active['emission_active_Pa'],active['emission_p95_Pa'],15e9,30e9])
    cmin=math.inf;eminc=math.inf;max_dg=-math.inf;min_v=math.inf;maxwell=0.;finite=True
    for T in COARSE:
        for surf in (c,e):
            G=surf.G_eV(stress,T);V=surf.activation_volume_m3(stress,T)
            finite &= bool(np.all(np.isfinite(G))&np.all(np.isfinite(V)))
            if surf is c:
                cmin=min(cmin,float(G.min()));max_dg=max(max_dg,float(surf.dG_dsigma_eV_per_Pa(stress,T).max()));min_v=min(min_v,float(V.min()))
            else:eminc=min(eminc,float(G.min()))
            dt=.02;ds=2e4
            ss=stress[stress>ds]
            dvdT=(surf.activation_volume_m3(ss,T+dt)-surf.activation_volume_m3(ss,T-dt))/(2*dt)
            dS=(surf.entropy_kB(ss+ds,T)-surf.entropy_kB(ss-ds,T))/(2*ds)*KB_EV_PER_K*EV_J
            maxwell=max(maxwell,float(np.max(abs(dvdT-dS))))
    cleavage_ok=finite and cmin>0 and max_dg<=1e-16 and min_v>=-1e-34 and maxwell<=1e-34
    emission_status='EMISSION_BARRIER_COLLAPSED_OR_NUMERICALLY_SATURATED' if eminc<=0 else 'EMISSION_HIGH_RATE_PHYSICALLY_ACTIVE' if np.max(e.raw_rate_s(stress,1200.))*TAU<100 else 'EMISSION_BARRIER_COLLAPSED_OR_NUMERICALLY_SATURATED'
    zero_ok=-20<=ez<=100
    return (c,e),dict(thermodynamic_pass=bool(cleavage_ok and zero_ok),rejection='' if cleavage_ok and zero_ok else 'CLEAVAGE_THERMODYNAMIC_ADMISSIBILITY' if not cleavage_ok else 'EMISSION_ZERO_ENTROPY_BOUND',cleavage_minimum_G_eV=cmin,emission_minimum_G_eV=eminc,cleavage_max_dG_dsigma_eV_per_Pa=max_dg,cleavage_minimum_V_m3=min_v,maxwell_max_residual_m3_per_K=maxwell,emission_zero_entropy_kB=ez,emission_status=emission_status)


def root_curve(surface,temperatures,rate=.005,xi=math.log(2),nK=320):
    K=np.r_[0.,np.geomspace(1e-7,80,nK)];stress=np.minimum(K[:,None]*1e6/np.sqrt(2*np.pi*RADIUS),CAP)
    rows=[]
    for T in temperatures:
        lam=cooperative_rate(surface.raw_rate_s(stress[:,0],T),HITS,TAU)
        action=cumulative_trapezoid(lam,K,initial=0.)/rate
        if action[-1]<xi:
            rows.append(dict(T=float(T),K_FP=np.nan,status='RAMP_CENSORED',zero_fraction=np.nan,AK=np.nan,ceiling_fraction=np.nan,stress_cap_fraction=np.nan));continue
        j=int(np.searchsorted(action,xi));root=float(np.interp(xi,action[j-1:j+1],K[j-1:j+1]));sig=np.minimum(root*1e6/np.sqrt(2*np.pi*RADIUS),CAP);end=float(cooperative_rate(surface.raw_rate_s(sig,T),HITS,TAU));zero=float(cooperative_rate(surface.raw_rate_s(0.,T),HITS,TAU));xs=(np.arange(256)+.5)/256;ss=np.minimum(root*xs*1e6/np.sqrt(2*np.pi*RADIUS),CAP);frac=gammainc(HITS,surface.raw_rate_s(ss,T)*TAU)
        rows.append(dict(T=float(T),K_FP=root,status='FIRST_PASSAGE',zero_fraction=zero*root/(rate*xi),AK=root*end/(rate*xi),ceiling_fraction=float(np.mean(frac>=.99)),stress_cap_fraction=float(np.mean(ss>=CAP))))
    return rows


def accessibility(row,primary=False):
    if row['status']!='FIRST_PASSAGE':return 'RAMP_CENSORED'
    if primary and not 1<=row['K_FP']<=50:return 'ABSOLUTE_FRACTURE_SCALE_FAILURE'
    if row['zero_fraction'] >= (1e-3 if primary else .10):return 'ZERO_LOAD_FIRST_PASSAGE_DOMINATED'
    if row['ceiling_fraction']>=.95:return 'CLEAVAGE_RENEWAL_CEILING_DOMINATED'
    if row['stress_cap_fraction']>=.95:return 'STRESS_CAP_DOMINATED'
    if row['AK'] <= (0 if primary else 1.5):return 'INSUFFICIENT_LOAD_SENSITIVITY'
    return 'FRACTURE_ACCESSIBLE'


def classify(curve,state=False):
    good=[r for r in curve if accessibility(r)== 'FRACTURE_ACCESSIBLE']
    if len(good)<math.ceil(.8*len(curve)):return 'TEMPERATURE_TOPOLOGY_NOT_INTERPRETABLE'
    K=np.array([r['K_FP'] for r in good]);T=np.array([r['T'] for r in good]);ratio=K[-1]/K[0]
    d=np.gradient(K,T)
    if np.all(d<=max(1e-9,1e-6*np.max(K))) and ratio<=.70:return 'ANALYTICAL_JOINT_CANDIDATE_CERAMIC_LIKE'
    if K.max()/K.min()<=1.25:return 'ANALYTICAL_JOINT_CANDIDATE_WEAK_T'
    imax=int(np.argmax(K))
    if 0<imax<len(K)-1 and K[imax]>=1.15*max(K[0],K[-1]) and np.all(np.diff(K[:imax+1])>0) and np.all(np.diff(K[imax:])<0):return 'ANALYTICAL_JOINT_CANDIDATE_PEAK_T_F1_PROVISIONAL' if state else 'ACCESSIBLE_UNCLASSIFIED'
    return 'ACCESSIBLE_UNCLASSIFIED'


def coarse_screen(candidates,rows,active,physical):
    admissions=[];coarse=[];fatigue=[];survivors=[]
    for i,p in enumerate(candidates):
        row,m,_=rows[p['parent_id']];surfs,thermo=thermodynamic_check(p,m,active[p['parent_id']]);record={**p,**thermo}
        if surfs is None or not thermo['thermodynamic_pass']:
            record.update(fatigue_pass=False,coarse_f0_pass=False);admissions.append(record);continue
        c,e=surfs;fm=fatigue_metrics(c,m,physical[p['parent_id']]);guard_amp=p['cleavage_zero_target_eV']-float(m.cleavage.values_eV(0.,300.));guard_max=abs(guard_amp)*10**p['guard_log10_q_low']
        fpass=fm['rms_log10_rate_error']<=.05 and fm['max_log10_rate_error']<=.1 and abs(fm['global_slope_change'])<=.15 and fm['max_adjacent_local_slope_change']<=.30 and not fm['new_renewal_ceiling'] and fm['monotonic_positive']
        fatigue.append(dict(candidate_id=p['candidate_id'],parent_id=p['parent_id'],guard_max_fatigue_window_eV=guard_max,fatigue_pass=fpass,**{k:v for k,v in fm.items() if k not in ('rates','local_slopes')},predicted_rates_json=json.dumps(fm['rates'].tolist()),local_slopes_json=json.dumps(fm['local_slopes'].tolist())))
        if not fpass:
            record.update(fatigue_pass=False,coarse_f0_pass=False,rejection='FATIGUE_PRESERVATION');admissions.append(record);continue
        curve=root_curve(c,COARSE);primary=accessibility(curve[0],True);cls=classify(curve)
        access_fraction=sum(accessibility(r)=='FRACTURE_ACCESSIBLE' for r in curve)/len(curve)
        cpass=primary=='FRACTURE_ACCESSIBLE' and access_fraction>=.8
        coarse.append(dict(candidate_id=p['candidate_id'],parent_id=p['parent_id'],family=p['family'],primary_accessibility=primary,accessible_fraction=access_fraction,coarse_topology=cls,KFP_300K=curve[0]['K_FP'],KFP_1200K=curve[-1]['K_FP'],KFP_ratio=curve[-1]['K_FP']/curve[0]['K_FP'] if np.isfinite(curve[-1]['K_FP']) else np.nan,curve_json=json.dumps(curve)))
        reason='' if cpass else ('TEMPERATURE_ACCESSIBILITY' if primary=='FRACTURE_ACCESSIBLE' else primary)
        record.update(fatigue_pass=True,coarse_f0_pass=cpass,rejection=reason);admissions.append(record)
        if cpass:survivors.append(p)
        if i and i%10000==0:print('screened',i,flush=True)
    return pd.DataFrame(admissions),pd.DataFrame(fatigue),pd.DataFrame(coarse),survivors


def diverse_selection(survivors,coarse,limit=768):
    if len(survivors)<=limit:return survivors
    frame=pd.DataFrame(survivors).merge(coarse[['candidate_id','KFP_ratio','KFP_300K','coarse_topology']],on='candidate_id')
    chosen=[]
    features=['cleavage_zero_target_eV','guard_log10_q_low','guard_exponent','cleavage_entropy_zero_kB','cleavage_entropy_active_kB','cleavage_entropy_infinity_kB','emission_entropy_active_kB','emission_entropy_infinity_kB','KFP_ratio','KFP_300K']
    for parent in PARENTS:
        q=frame[frame.parent_id==parent].copy();X=q[features].to_numpy(float);X=(X-X.min(0))/np.maximum(X.max(0)-X.min(0),1e-12)
        first=int(np.argmin(q.KFP_ratio.to_numpy()));indices=[first];distance=np.linalg.norm(X-X[first],axis=1)
        for _ in range(min(limit//2,len(q))-1):
            j=int(np.argmax(distance));indices.append(j);distance=np.minimum(distance,np.linalg.norm(X-X[j],axis=1))
        chosen.extend(q.iloc[indices].drop(columns=['KFP_ratio','KFP_300K','coarse_topology']).to_dict('records'))
    return chosen


def full_f0(candidates,rows,active):
    records=[];summary=[]
    for i,p in enumerate(candidates):
        _,m,_=rows[p['parent_id']];clean={k:v for k,v in p.items() if not (isinstance(v,float) and np.isnan(v))};c,_=candidate_surface_from_parameters(m,clean,active[p['parent_id']]['cleavage_low_Pa'],active[p['parent_id']]['cleavage_active_Pa'],active[p['parent_id']]['emission_active_Pa'])
        primary_curve=None
        for label,xi in THRESHOLDS.items():
            for rate in RATES:
                curve=root_curve(c,TEMPS,rate,xi,480)
                for r in curve:records.append(dict(candidate_id=p['candidate_id'],parent_id=p['parent_id'],threshold=label,Kdot=rate,accessibility=accessibility(r),**r))
                if label=='LN2' and rate==.005:primary_curve=curve
        cls=classify(primary_curve);good=sum(accessibility(r)=='FRACTURE_ACCESSIBLE' for r in primary_curve)
        summary.append(dict(candidate_id=p['candidate_id'],parent_id=p['parent_id'],F0_topology=cls,F0_accessible_points=good,KFP_300K=primary_curve[0]['K_FP'],KFP_1200K=primary_curve[-1]['K_FP'],KFP_ratio=primary_curve[-1]['K_FP']/primary_curve[0]['K_FP'],zero_fraction_300K=primary_curve[0]['zero_fraction'],AK_300K=primary_curve[0]['AK']))
        if i and i%100==0:print('full F0',i,flush=True)
    return pd.DataFrame(records),pd.DataFrame(summary)


def pareto_and_f1(candidates,summary,admissions,fatigue,rows,active):
    thermo=admissions[['candidate_id','emission_status','emission_minimum_G_eV','maxwell_max_residual_m3_per_K']]
    base=pd.DataFrame(candidates).merge(summary,on=['candidate_id','parent_id']).merge(fatigue,on=['candidate_id','parent_id']).merge(thermo,on='candidate_id')
    base=base[base.F0_accessible_points>=30].copy()
    base['topology_strength']=np.where(base.F0_topology.str.contains('CERAMIC'),1-base.KFP_ratio,np.where(base.F0_topology.str.contains('WEAK'),-abs(np.log(base.KFP_ratio)),0))
    objectives=base[['rms_log10_rate_error','max_adjacent_local_slope_change','zero_fraction_300K']].to_numpy(float)
    objectives=np.c_[objectives,-base.topology_strength.to_numpy(float),base.complexity.to_numpy(float)]
    nondom=np.ones(len(base),bool)
    for i in range(len(base)):
        if np.any(np.all(objectives<=objectives[i]+1e-12,axis=1)&np.any(objectives<objectives[i]-1e-12,axis=1)):nondom[i]=False
    base['epsilon_nondominated']=nondom
    pool=base.sort_values(['epsilon_nondominated','rms_log10_rate_error','candidate_id'],ascending=[False,True,True]).head(24)
    f1rows=[];f1summary=[]
    for _,p in pool.iterrows():
        _,m,_=rows[p.parent_id];clean={k:v for k,v in p.items() if not (isinstance(v,float) and np.isnan(v))};c,e=candidate_surface_from_parameters(m,clean,active[p.parent_id]['cleavage_low_Pa'],active[p.parent_id]['cleavage_active_Pa'],active[p.parent_id]['emission_active_Pa']);proxy=replace(m,cleavage=c,emission=e);controls=replace(Controls(),hits=HITS,tau=TAU)
        curve=[]
        for T in COARSE:
            f0=root_curve(c,[T])[0]
            try:
                model=TransientBlunting(proxy,rows[p.parent_id][0],T,.005,controls);K,v,end,nfev=model.first_passage(math.log(2),f0['K_FP'])
                if K is None:raise ValueError('RAMP_CENSORED')
                action=float(v(K)[2]);radius=float(model.state(K,v(K)[:2])[0]);rec=dict(T=float(T),K_FP=float(K),status='FIRST_PASSAGE',zero_fraction=f0['zero_fraction'],AK=float(K*cooperative_rate(c.raw_rate_s(min(K*1e6/np.sqrt(2*np.pi*radius),CAP),T),HITS,TAU)/(.005*math.log(2))),ceiling_fraction=f0['ceiling_fraction'],stress_cap_fraction=f0['stress_cap_fraction'])
                curve.append(rec);f1rows.append(dict(candidate_id=p.candidate_id,parent_id=p.parent_id,temperature_K=T,Kdot=.005,threshold='LN2',F1_status='FIRST_PASSAGE',K_FP=K,r_eff_m=radius,action=action,nfev=nfev,accessibility=accessibility(rec)))
            except Exception as exc:
                f1rows.append(dict(candidate_id=p.candidate_id,parent_id=p.parent_id,temperature_K=T,Kdot=.005,threshold='LN2',F1_status='STATE_CLOSURE_UNAVAILABLE',reason=str(exc),accessibility='STATE_CLOSURE_UNAVAILABLE'))
        available=[r for r in curve if accessibility(r)=='FRACTURE_ACCESSIBLE'];status='F0_ACCESSIBLE_STATE_UNRESOLVED' if len(available)<6 else 'F1_CONDITIONAL_AVAILABLE';cls=classify(curve,True) if len(curve)==len(COARSE) else 'TEMPERATURE_TOPOLOGY_NOT_INTERPRETABLE'
        f1summary.append(dict(candidate_id=p.candidate_id,F1_status=status,F1_topology=cls,F1_available_points=len(available),reason=''))
    return base,pd.DataFrame(f1rows),pd.DataFrame(f1summary)


def main():
    RUN.mkdir(parents=True,exist_ok=True);ART.mkdir(exist_ok=True)
    protocol=json.loads((ART/'search_protocol_freeze.json').read_text())
    if protocol['parent_head']!='0399e12e8f015071b3d3d8c7ad8cb03b435d6aaf':raise ValueError('protocol parent changed')
    rows=source_rows();physical=physical_controls();active=active_stresses(rows)
    (ART/'active_stress_coordinates.json').write_text(json.dumps(active,indent=2)+'\n')
    single=single_exp_feasibility(rows,physical);write_csv(ART/'single_exp_feasibility.csv',single)
    structured_rows=structured(PARENTS[0])+structured(PARENTS[1]);write_csv(ART/'structured_seed_design.csv',structured_rows)
    sobol=map_sobol(PARENTS[0],65536,260911)+map_sobol(PARENTS[1],65536,260912)
    sobdf=pd.DataFrame(sobol);sobdf.to_parquet(ART/'sobol_candidate_bank.parquet',index=False,compression='zstd')
    all_candidates=structured_rows+sobol
    admissions,fatigue,coarse,survivors=coarse_screen(all_candidates,rows,active,physical)
    write_csv(ART/'candidate_admissibility.csv',admissions.to_dict('records'));write_csv(ART/'fatigue_preservation_metrics.csv',fatigue.to_dict('records'));coarse.to_parquet(ART/'f0_coarse_screen.parquet',index=False,compression='zstd')
    selected=diverse_selection(survivors,coarse,768);full,summary=full_f0(selected,rows,active);write_partitioned(full,ART/'f0_full_temperature_screen.parquet');summary.to_csv(ART/'fracture_accessibility_metrics.csv',index=False)
    base,f1,f1summary=pareto_and_f1(selected,summary,admissions,fatigue,rows,active);write_partitioned(f1,ART/'f1_state_screen.parquet');f1summary.to_csv(ART/'f1_state_summary.csv',index=False);base.to_parquet(RUN/'pareto_pool.parquet',index=False)
    counts=dict(structured=len(structured_rows),sobol=len(sobol),thermodynamic_pass=int(admissions.thermodynamic_pass.sum()),fatigue_pass=int(admissions.fatigue_pass.sum()),coarse_F0_pass=int(admissions.coarse_f0_pass.sum()),full_F0_candidates=len(selected),F1_candidates=len(f1summary),F1_conditions=len(f1),single_EXP_pass=sum(r['status']=='PASS' for r in single))
    (RUN/'stage_counts.json').write_text(json.dumps(counts,indent=2)+'\n');print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
