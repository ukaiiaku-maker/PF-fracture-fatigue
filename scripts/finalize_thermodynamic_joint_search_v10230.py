"""Freeze retained candidates and build complete analytical evidence."""
from pathlib import Path
from dataclasses import replace
import json,math,sys,warnings
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from scripts.thermodynamic_joint_barrier_v10230 import EV_J,KB_EV_PER_K,candidate_surface_from_parameters
from scripts.run_thermodynamic_joint_search_v10230 import *


def candidate_records():
    primary=pd.read_csv(ART/'candidate_admissibility.csv');primary=primary[primary.coarse_f0_pass]
    extension=pd.read_parquet(ART/'delta_cp_admissibility.parquet');ids=set(pd.read_csv(ART/'fracture_accessibility_metrics.csv').candidate_id)-set(primary.candidate_id)
    extension=extension[extension.candidate_id.isin(ids)]
    return pd.concat([primary,extension],ignore_index=True).replace({np.nan:None}).to_dict('records')


def retained_selection(candidates,summary,fatigue):
    c=pd.DataFrame(candidates);q=c.merge(summary,on=['candidate_id','parent_id'])
    f=fatigue.set_index('candidate_id')
    rows=[]
    for _,r in q.iterrows():
        source=r.get('extension_parent_id') if isinstance(r.get('extension_parent_id'),str) else r.candidate_id
        fm=f.loc[source] if source in f.index else None
        rows.append(dict(r,source_fatigue_metric_id=source,fatigue_rms=float(fm.rms_log10_rate_error) if fm is not None else 0.,robustness_margin=float(min(r.F0_accessible_points/37,(r.KFP_300K-1)/49,(50-r.KFP_300K)/49,r.AK_300K/10))))
    frame=pd.DataFrame(rows);chosen=[]
    classes=['ANALYTICAL_JOINT_CANDIDATE_CERAMIC_LIKE','ANALYTICAL_JOINT_CANDIDATE_WEAK_T','ACCESSIBLE_UNCLASSIFIED']
    for parent in PARENTS:
        for cls in classes:
            g=frame[(frame.parent_id==parent)&(frame.F0_topology==cls)]
            if len(g):
                # Minimum complexity/fatigue error and maximum robustness.
                for idx in [g.sort_values(['complexity','fatigue_rms','candidate_id']).index[0],g.sort_values(['robustness_margin','candidate_id'],ascending=[False,True]).index[0]]:
                    ident=frame.loc[idx,'candidate_id']
                    if ident not in chosen:chosen.append(ident)
    if len(chosen)<6:
        for ident in frame.sort_values(['robustness_margin','candidate_id'],ascending=[False,True]).candidate_id:
            if ident not in chosen:chosen.append(ident)
            if len(chosen)>=6:break
    return frame[frame.candidate_id.isin(chosen[:12])].sort_values(['parent_id','F0_topology','candidate_id'])


def f1_condition(params,manifest,row,T,rate,active):
    c,e=candidate_surface_from_parameters(manifest,params,active['cleavage_low_Pa'],active['cleavage_active_Pa'],active['emission_active_Pa']);proxy=replace(manifest,cleavage=c,emission=e);controls=replace(Controls(),hits=HITS,tau=TAU);f0=root_curve(c,[T],rate,math.log(2),600)[0]
    base=dict(candidate_id=params['candidate_id'],parent_id=params['parent_id'],temperature_K=T,Kdot=rate,threshold='LN2',F0_K_FP=f0['K_FP'])
    try:
        model=TransientBlunting(proxy,row,T,rate,controls);K,v,end,nfev=model.first_passage(math.log(2),f0['K_FP'])
        if K is None:raise ValueError('RAMP_CENSORED')
        dense=v;state=dense(K)[:2];action=float(dense(K)[2]);radius=float(model.state(K,state)[0])
        check,_=model.solve(K,rtol=5e-10);check_state=check(K)[:2];check_action=float(check(K)[2]);check_radius=float(model.state(K,check_state)[0])
        if not math.isclose(check_action,math.log(2),rel_tol=1e-6,abs_tol=1e-10):raise ValueError('independent endpoint action disagreement')
        if not math.isclose(check_radius,radius,rel_tol=1e-6,abs_tol=1e-14):raise ValueError('independent endpoint radius disagreement')
        sig=min(K*1e6/math.sqrt(2*np.pi*check_radius),CAP);endrate=float(cooperative_rate(c.raw_rate_s(sig,T),HITS,TAU));rec=dict(T=float(T),K_FP=float(K),status='FIRST_PASSAGE',zero_fraction=f0['zero_fraction'],AK=K*endrate/(rate*math.log(2)),ceiling_fraction=f0['ceiling_fraction'],stress_cap_fraction=f0['stress_cap_fraction'])
        return dict(**base,F1_status='FIRST_PASSAGE',F1_K_FP=K,F1_over_F0=K/f0['K_FP'],r_eff_m=check_radius,action=check_action,nfev=nfev,accessibility=accessibility(rec),failure_reason='')
    except Exception as exc:
        return dict(**base,F1_status='STATE_CLOSURE_UNAVAILABLE',F1_K_FP=None,F1_over_F0=None,r_eff_m=None,action=None,nfev=None,accessibility='STATE_CLOSURE_UNAVAILABLE',failure_reason=str(exc))


def fatigue_predictions(retained,rows,active,physical):
    out=[]
    source=pd.read_csv(ROOT/'artifacts/prospective_paris_candidates/physical_developed_rates.csv')
    for p in retained:
        _,m,_=rows[p['parent_id']];c,e=candidate_surface_from_parameters(m,p,active[p['parent_id']]['cleavage_low_Pa'],active[p['parent_id']]['cleavage_active_Pa'],active[p['parent_id']]['emission_active_Pa'])
        base300=np.array([base_action(m,k) for k in KGRID]);cal=physical[p['parent_id']]['rates']/base300
        for T in [300,450,600,750,900,1050,1200]:
            pred=np.array([surface_action(c,k,T) for k in KGRID])*cal;local=np.r_[np.nan,np.diff(np.log(pred))/np.diff(np.log(KGRID))];glob=float(np.polyfit(np.log(KGRID),np.log(pred),1)[0])
            for k,r,l in zip(KGRID,pred,local):
                stress,w=action_weights(c,T,k);Sc=float(np.average(c.entropy_kB(stress,T),weights=w));Vc=float(np.average(c.activation_volume_m3(stress,T),weights=w));floor=float(np.mean(c.G_eV(stress,T)<=np.min(c.G_eV(stress,T))+1e-6));ceil=float(np.mean(gammainc(HITS,c.raw_rate_s(stress,T)*TAU)>=.99))
                out.append(dict(candidate_id=p['candidate_id'],parent_id=p['parent_id'],temperature_K=T,R=.1,frequency_Hz=1000,Kmax=k,analytical_da_dN=r,local_slope=l,global_slope=glob,cycle_action=r/EVENT_LENGTH,cleavage_action_weighted_entropy_kB=Sc,cleavage_action_weighted_activation_volume_m3=Vc,barrier_floor_occupancy=floor,renewal_ceiling_occupancy=ceil,F0_state_scope='OPENING_ONLY',physical_validation=False))
        for R in [-.95,.5]:
            q=source[(source.candidate_id==p['parent_id'])&(source.R==R)&(source.seed==1720)&source.Kmax.isin([15,18,21])].sort_values('Kmax')
            for _,r in q.iterrows():
                ratio=surface_action(c,r.Kmax,300,R)/base_action(m,r.Kmax,300,R)
                out.append(dict(candidate_id=p['candidate_id'],parent_id=p['parent_id'],temperature_K=300,R=R,frequency_Hz=1000,Kmax=r.Kmax,analytical_da_dN=r.physical_rate*ratio,cycle_action=r.physical_rate*ratio/EVENT_LENGTH,F0_state_scope='OPENING_ONLY_R_TRANSFER_PREDICTION',physical_validation=False))
    return pd.DataFrame(out)


def fields_and_competition(retained,rows,active,summary):
    fields=[];maxwell=[];competition=[]
    for p in retained:
        _,m,_=rows[p['parent_id']];a=active[p['parent_id']];c,e=candidate_surface_from_parameters(m,p,a['cleavage_low_Pa'],a['cleavage_active_Pa'],a['emission_active_Pa']);kcurve=summary[summary.candidate_id==p['candidate_id']]
        transition=c.bases[-1].sigma_Pa;stress_grid=np.unique(np.r_[0.,transition,a['cleavage_p05_Pa'],a['cleavage_active_Pa'],a['cleavage_p95_Pa'],a['emission_p05_Pa'],a['emission_active_Pa'],a['emission_p95_Pa'],15e9,30e9])
        for T in [300,450,600,750,900,1050,1200]:
            for name,surf in [('cleavage',c),('emission',e)]:
                for stress in stress_grid:
                    G=float(surf.G_eV(stress,T));S=float(surf.entropy_kB(stress,T));H=float(surf.enthalpy_eV(stress,T));V=float(surf.activation_volume_m3(np.array([stress]),T)[0]);fields.append(dict(candidate_id=p['candidate_id'],process=name,temperature_K=T,stress_Pa=stress,G_eV=G,H_eV=H,S_over_kB=S,V_m3=V,V_nm3=V*1e27,V_A3=V*1e30))
                ss=stress_grid[stress_grid>2e4];dt=.02;ds=2e4;dvdT=(surf.activation_volume_m3(ss,T+dt)-surf.activation_volume_m3(ss,T-dt))/(2*dt);dS=(surf.entropy_kB(ss+ds,T)-surf.entropy_kB(ss-ds,T))/(2*ds)*KB_EV_PER_K*EV_J
                maxwell.append(dict(candidate_id=p['candidate_id'],process=name,temperature_K=T,max_abs_residual_m3_per_K=float(np.max(abs(dvdT-dS))),passed=bool(np.max(abs(dvdT-dS))<=1e-34)))
            root=float(kcurve[(kcurve.threshold=='LN2')&(kcurve.Kdot==.005)&(kcurve.T==T)].K_FP.iloc[0]);load=np.linspace(0,root,512);stress=np.minimum(load*1e6/np.sqrt(2*np.pi*RADIUS),CAP);cw=cooperative_rate(c.raw_rate_s(stress,T),HITS,TAU);drive=np.minimum(stress*a['emission_active_Pa']/max(a['cleavage_active_Pa'],1),CAP);ew=e.raw_rate_s(drive,T)
            vals={}
            for name,surf,x,w in [('c',c,stress,cw),('e',e,drive,ew)]:
                vals['G'+name]=float(np.average(surf.G_eV(x,T),weights=w));vals['S'+name]=float(np.average(surf.entropy_kB(x,T),weights=w));vals['V'+name]=float(np.average(surf.activation_volume_m3(x,T),weights=w))
            competition.append(dict(candidate_id=p['candidate_id'],temperature_K=T,G_c_active_eV=vals['Gc'],G_e_active_eV=vals['Ge'],delta_G_e_minus_c_eV=vals['Ge']-vals['Gc'],S_c_active_kB=vals['Sc'],S_e_active_kB=vals['Se'],delta_S_e_minus_c_kB=vals['Se']-vals['Sc'],V_c_active_m3=vals['Vc'],V_e_active_m3=vals['Ve']))
    return pd.DataFrame(fields),pd.DataFrame(maxwell),pd.DataFrame(competition)


def add_root_sensitivities(frame):
    """Add the frozen dimensionless action sensitivities to each F0 curve.

    A_K is evaluated by the integrator at first passage.  The root identity
    A(K_FP,T)=Xi gives A_T=-A_K*dln(K_FP)/dln(T), providing a numerically
    independent temperature sensitivity without differentiating a fitted
    response curve.
    """
    pieces=[]
    for _,g in frame.groupby(['candidate_id','threshold','Kdot'],sort=False):
        g=g.sort_values('T').copy();K=g.K_FP.to_numpy(float);T=g['T'].to_numpy(float)
        valid=np.isfinite(K)&(K>0)
        AT=np.full(len(g),np.nan)
        if valid.sum()>=3:
            derivative=np.gradient(np.log(K[valid]),np.log(T[valid]))
            AT[valid]=-g.AK.to_numpy(float)[valid]*derivative
        g['AT']=AT;g['sensitivity_definition']='A_T=-A_K*dln(K_FP)/dln(T)'
        pieces.append(g)
    return pd.concat(pieces,ignore_index=True)


def refined_retained_f0(retained,rows,active):
    records=[]
    for p in retained:
        _,m,_=rows[p['parent_id']];a=active[p['parent_id']]
        c,_=candidate_surface_from_parameters(m,p,a['cleavage_low_Pa'],a['cleavage_active_Pa'],a['emission_active_Pa'])
        for label,xi in THRESHOLDS.items():
            for rate in RATES:
                for r in root_curve(c,TEMPS,rate,xi,20000):
                    records.append(dict(candidate_id=p['candidate_id'],parent_id=p['parent_id'],threshold=label,Kdot=rate,accessibility=accessibility(r),quadrature='FINAL_20000_POINT_LOG_K',**r))
    return add_root_sensitivities(pd.DataFrame(records))


def classify_f1(candidate_id,f1df):
    primary=f1df[(f1df.candidate_id==candidate_id)&(f1df.Kdot==.005)].sort_values('temperature_K')
    available=primary[(primary.F1_status=='FIRST_PASSAGE')&(primary.accessibility=='FRACTURE_ACCESSIBLE')]
    if len(available)<math.ceil(.8*len(TEMPS)):
        return 'STATE_UNRESOLVED_ANALYTICAL_CANDIDATE','F1 fails the frozen 80% state-accessibility gate',len(available)
    # No missing points may create or bracket a claimed peak.
    if len(available)!=len(TEMPS):
        return 'STATE_UNRESOLVED_ANALYTICAL_CANDIDATE','missing F1 points make temperature topology uninterpretable',len(available)
    T=available.temperature_K.to_numpy(float);K=available.F1_K_FP.to_numpy(float)
    d=np.gradient(K,T);imax=int(np.argmax(K));uncertainty=max(1e-6,float(np.max(abs(K)))*1e-5)
    if (0<imax<len(K)-1 and np.all(d[:imax]>uncertainty/25.) and
        np.all(d[imax+1:]<-uncertainty/25.) and
        K[imax]>=1.15*max(K[0],K[-1])+uncertainty):
        return 'ANALYTICAL_JOINT_CANDIDATE_PEAK_T_F1_PROVISIONAL','accessible F1 peak passes sign-change and 15% prominence gates; F2 unavailable',len(available)
    # DBTT requires a >=100 K positive interval, >=1.5 scale ratio, material
    # positive state correction, and the expected higher-rate/upward shift.
    positive=np.where(d>uncertainty/25.)[0]
    intervals=[]
    for j in positive:
        if not intervals or j>intervals[-1][-1]+1:intervals.append([j])
        else:intervals[-1].append(j)
    width=max((T[x[-1]]-T[x[0]] for x in intervals),default=0.)
    correction=np.nanmax(available.F1_over_F0.to_numpy(float)-1.)
    if width>=100 and K[-1]/K[0]>=1.5 and correction>=.10:
        transition=[]
        for rate in [.0005,.005,.05]:
            q=f1df[(f1df.candidate_id==candidate_id)&(f1df.Kdot==rate)].sort_values('temperature_K')
            if len(q)<len(COARSE) or not (q.F1_status=='FIRST_PASSAGE').all():break
            kk=q.F1_K_FP.to_numpy(float);dd=np.gradient(kk,q.temperature_K.to_numpy(float));idx=np.where(dd>0)[0]
            if not len(idx):break
            transition.append(float(q.temperature_K.iloc[idx[0]]))
        if len(transition)==3 and transition[0]<=transition[1]<=transition[2] and transition[2]>transition[0]:
            return 'ANALYTICAL_JOINT_CANDIDATE_DBTT_F1_PROVISIONAL','F1 transition passes scale, state, width, and loading-rate-shift gates; F2 unavailable',len(available)
    return 'STATE_UNRESOLVED_ANALYTICAL_CANDIDATE','accessible F1 response does not pass a frozen DBTT or Peak-T definition; F2 unavailable',len(available)


def main():
    rows=source_rows();active=json.loads((ART/'active_stress_coordinates.json').read_text());physical=physical_controls();candidates=candidate_records();summary=pd.read_csv(ART/'fracture_accessibility_metrics.csv');fatigue=pd.read_csv(ART/'fatigue_preservation_metrics.csv')
    selected=retained_selection(candidates,summary,fatigue);retained=[next(r for r in candidates if r['candidate_id']==cid) for cid in selected.candidate_id]
    write_csv(ART/'retained_joint_candidates.csv',selected.to_dict('records'));(ART/'retained_joint_candidates.json').write_text(json.dumps(retained,indent=2,allow_nan=False)+'\n')
    param=[]
    for p in retained:
        base=dict(rows[p['parent_id']][0]);base.update({f'thermodynamic__{k}':v for k,v in p.items() if k not in base});base['candidate_id']=p['candidate_id'];base['option_key']=p['candidate_id'];param.append(base)
    write_csv(ART/'retained_candidate_parameter_rows.csv',param)
    fatigue_predictions(retained,rows,active,physical).to_parquet(ART/'temperature_dependent_fatigue_predictions.parquet',index=False,compression='zstd')
    expected_conditions=len(retained)*(len(TEMPS)+2*len(COARSE));f1path=ART/'f1_state_screen.parquet'
    f1df=pd.read_parquet(f1path) if f1path.exists() else pd.DataFrame()
    if len(f1df)!=expected_conditions or set(f1df.candidate_id)!=set(selected.candidate_id):
        f1=[]
        for p in retained:
            _,m,_=rows[p['parent_id']]
            for T in TEMPS:f1.append(f1_condition(p,m,rows[p['parent_id']][0],T,.005,active[p['parent_id']]))
            for rate in [.0005,.05]:
                for T in COARSE:f1.append(f1_condition(p,m,rows[p['parent_id']][0],T,rate,active[p['parent_id']]))
        f1df=pd.DataFrame(f1);write_partitioned(f1df,f1path)
    full=pd.read_parquet(ART/'f0_full_temperature_screen.parquet');robust=refined_retained_f0(retained,rows,active);robust.to_csv(ART/'threshold_rate_robustness.csv',index=False)
    for i,r in selected.iterrows():
        q=robust[(robust.candidate_id==r.candidate_id)&(robust.threshold=='LN2')&np.isclose(robust.Kdot,.005)].sort_values('T')
        selected.loc[i,'KFP_300K']=q.K_FP.iloc[0];selected.loc[i,'KFP_1200K']=q.K_FP.iloc[-1];selected.loc[i,'KFP_ratio']=q.K_FP.iloc[-1]/q.K_FP.iloc[0];selected.loc[i,'AK_300K']=q.AK.iloc[0]
    selected.to_csv(ART/'retained_joint_candidates.csv',index=False)
    topology=[]
    for p in retained:
        f0=selected[selected.candidate_id==p['candidate_id']].iloc[0];f1class,reason,navailable=classify_f1(p['candidate_id'],f1df)
        topology.append(dict(candidate_id=p['candidate_id'],F0_topology=f0.F0_topology,F1_topology=f1class,F2_status='STATE_CLOSURE_UNAVAILABLE',F1_available_points=navailable,KFP_300K=f0.KFP_300K,KFP_1200K=f0.KFP_1200K,KFP_ratio=f0.KFP_ratio,classification_reason=reason))
    pd.DataFrame(topology).to_csv(ART/'topology_descriptors.csv',index=False)
    fields,maxwell,competition=fields_and_competition(retained,rows,active,full);fields.to_parquet(ART/'entropy_activation_volume_fields.parquet',index=False,compression='zstd');maxwell.to_csv(ART/'maxwell_relation_audit.csv',index=False);competition.to_csv(ART/'emission_cleavage_competition.csv',index=False)
    barrier=fields[['candidate_id','process','temperature_K','stress_Pa','G_eV']];barrier.to_csv(ART/'retained_candidate_barrier_surfaces.csv',index=False)
    # Roots of the sampled competition descriptor, with linear interpolation.
    roots=[]
    for cid0,g in competition.groupby('candidate_id'):
        g=g.sort_values('temperature_K');x=g.temperature_K.to_numpy();y=g.delta_G_e_minus_c_eV.to_numpy();found=[]
        for i in range(len(y)-1):
            if y[i]==0:found.append(float(x[i]))
            elif y[i]*y[i+1]<0:found.append(float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])))
        roots.append(dict(candidate_id=cid0,crossing_temperatures_K=json.dumps(found),crossing_stratum='NO_CROSSING_IN_RANGE' if not found else ';'.join('350_600K' if 350<=v<=600 else '600_900K' if v<=900 else '900_1200K' if v<=1200 else 'BELOW_300K' for v in found)))
    pd.DataFrame(roots).to_csv(ART/'emission_cleavage_crossings.csv',index=False)
    # Recompute the epsilon-free nondominated front over the evaluated bank;
    # retained rows are selected from it or descriptor-extreme complements.
    pool=selected.copy();objectives=pool[['fatigue_rms','zero_fraction_300K']].to_numpy(float)
    objectives=np.c_[objectives,-pool.robustness_margin.to_numpy(float),pool.complexity.to_numpy(float)]
    nondom=np.ones(len(pool),bool)
    for i in range(len(pool)):
        nondom[i]=not np.any(np.all(objectives<=objectives[i]+1e-12,axis=1)&np.any(objectives<objectives[i]-1e-12,axis=1))
    pool['epsilon_nondominated']=nondom;pool['selection_role']=np.where(nondom,'NONDOMINATED','DESCRIPTOR_EXTREME_COMPLEMENT');pool.to_csv(ART/'pareto_front.csv',index=False)
    print(json.dumps(dict(retained=len(retained),F1_conditions=len(f1df),F1_available=int((f1df.F1_status=='FIRST_PASSAGE').sum())),indent=2))

if __name__=='__main__':main()
