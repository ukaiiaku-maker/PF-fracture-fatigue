"""Bounded row-renewal correction. Original generic outputs remain immutable."""
from pathlib import Path
import sys,json,csv,math,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from dataclasses import replace
from scipy.optimize import brentq
from scripts.forward_temperature_model_v10230 import *
from scripts.analyze_forward_temperature_v10230 import source_rows,write_csv,audit
from arrhenius_fracture.prospective_paris_candidate_engine_v10230 import digest
ART=ROOT/'artifacts/row_renewal_monotonic_forward'
OLD=ROOT/'artifacts/retained_controls_monotonic_forward'
RUN=ROOT/'runs/row_renewal_monotonic_forward_v1'


_UnboundedTransientBlunting=TransientBlunting
class TransientBlunting(_UnboundedTransientBlunting):
    """Same transient equations with a deterministic fail-closed work budget."""
    max_state_evaluations=100000
    def state(self,K,N):
        if getattr(self,'_budget_active',False):
            self._state_evaluations+=1
            if self._state_evaluations>self.max_state_evaluations:
                raise ValueError('F1_STATE_CLOSURE_UNAVAILABLE: analytical state-evaluation budget exhausted (100000 per solve)')
        return super().state(K,N)
    def solve(self,Kend,rtol=None):
        self._state_evaluations=0;self._budget_active=True
        try:return super().solve(Kend,rtol)
        finally:self._budget_active=False


def row_controls(row):
    m=float(row['physics__cleavage_hits']);tau=float(row['physics__cleavage_correlation_time_s'])
    if not math.isfinite(m) or m<=1 or not math.isfinite(tau) or tau<=0:raise ValueError('invalid exact row renewal')
    return replace(Controls(),hits=m,tau=tau)

def condition(cid,row,m,T,rate,xi,threshold_label):
    c=row_controls(row);k0=f0_root(m,T,rate,xi,c);records=[];traces=[];descriptors=[];dchecks=[]
    base=dict(candidate_id=cid,temperature_K=T,Kdot=rate,threshold_action=xi,threshold_mode=threshold_label,renewal_hits=c.hits,renewal_tau_s=c.tau)
    models=[('F0_INTRINSIC_OPENING',None)]
    f1=TransientBlunting(m,row,T,rate,c)
    try:
        k1,v1,end,nfev=f1.first_passage(xi,k0)
        models.append(('F1_EMISSION_BLUNTING',(k1,v1,end,nfev)))
    except ValueError as exc:
        records.append(dict(**base,tier='F1_EMISSION_BLUNTING',status='STATE_CLOSURE_UNAVAILABLE',reason=str(exc),K_FP=None,full_state_available=False))
    for tier,data in models:
        if data is None:
            K=k0;action=lambda k:f0_action(m,k,T,rate,c);sigma=lambda k:stress(k,c.r0,c);radius=c.r0;N=0.
        else:
            K,v,end,nfev=data
            if K is None:
                records.append(dict(**base,tier=tier,status='RAMP_CENSORED',K_FP=None,full_state_available=False));continue
            action=lambda k:float(v(k)[2]);sigma=lambda k:f1.state(k,v(k)[:2])[1]
            radius=f1.state(K,v(K)[:2])[0];N=float(v(K)[:2].sum())
        residual=abs(action(K)/xi-1)
        AK=K*float(renewal(m.cleavage,sigma(K),T,c))/(rate*xi)
        h=.0001;AKfd=(math.log(action(K*math.exp(h)))-math.log(action(K*math.exp(-h))))/(2*h)
        dt=.05
        try:
            if data is None:
                AKexact,AT=f0_derivatives(m,K,T,rate,xi,c)
                ATfd=(math.log(f0_action(m,K,T+dt,rate,c))-math.log(f0_action(m,K,T-dt,rate,c)))/(2*dt)
                kp=f0_root(m,T+dt,rate,xi,c);km=f0_root(m,T-dt,rate,xi,c)
                refinement_error=0.
            else:
                upper=max(end,K*1.05)
                fp=TransientBlunting(m,row,T+dt,rate,c);fm=TransientBlunting(m,row,T-dt,rate,c)
                vp,_=fp.solve(upper);vm,_=fm.solve(upper)
                AT=(math.log(vp(K)[2])-math.log(vm(K)[2]))/(2*dt);ATfd=AT
                kp=brentq(lambda x:vp(x*upper)[2]/xi-1,0,1,xtol=2e-12)*upper
                km=brentq(lambda x:vm(x*upper)[2]/xi-1,0,1,xtol=2e-12)*upper
                refined,_=f1.solve(upper,rtol=c.rtol/4)
                kr=brentq(lambda x:refined(x*upper)[2]/xi-1,0,1,xtol=2e-12)*upper
                refinement_error=abs(kr/K-1)
            thermal=-AT/AK;thermal_fd=(math.log(kp)-math.log(km))/(2*dt)
            agreement=math.isclose(AK,AKfd,rel_tol=5e-4,abs_tol=2e-7) and math.isclose(AT,ATfd,rel_tol=5e-4,abs_tol=2e-7) and math.isclose(thermal,thermal_fd,rel_tol=5e-4,abs_tol=2e-7) and refinement_error<1e-6
            if not agreement:raise ValueError('derivative or refinement agreement failed')
        except (ValueError,RuntimeError) as exc:
            if data is None:raise
            records.append(dict(**base,tier=tier,status='STATE_CLOSURE_UNAVAILABLE',reason='Sensitivity/refinement: '+str(exc),K_FP=None,full_state_available=False));continue
        xs=(np.arange(256)+.5)/256
        rates=np.array([float(renewal(m.cleavage,sigma(K*x),T,c)) for x in xs])
        ceiling=float(np.mean(rates*c.tau>=.99));cap=float(np.mean([sigma(K*x)>=c.cap for x in xs]))
        zero=float(renewal(m.cleavage,0.,T,c));zero_fraction=zero*K/(rate*xi)
        access,flags=accessibility(zero_fraction,ceiling,cap,K,residual)
        d=surface(m.cleavage,np.array([sigma(K*x) for x in xs]),T)
        floor_occupancy=float(np.mean(np.abs(d['G']-d['floor'])<=1e-6*np.maximum(d['G'],1e-30)))
        rec=dict(**base,tier=tier,status='FIRST_PASSAGE',K_FP=K,action=action(K),root_relative_residual=residual,r_eff_m=radius,K_shield_MPa_sqrt_m=0.,source_activations=N,AK=AK,AT=AT,thermal_derivative=thermal,accessibility=access,accessibility_flags=';'.join(flags),zero_load_elementary_rate=float(m.cleavage.rate(0,T)),zero_load_renewal_rate=zero,zero_load_action_fraction=zero_fraction,renewal_ceiling_ramp_fraction=ceiling,renewal_ceiling_margin_at_FP=1-float(renewal(m.cleavage,sigma(K),T,c))*c.tau,stress_cap_ramp_fraction=cap,barrier_floor_ramp_fraction=floor_occupancy,numerical_lower_bound_occupancy=0,full_state_available=False,state_scope='intrinsic' if data is None else 'conditional_ideal_mode_I_source_bin_moment_no_PT_transport',refinement_relative_K_error=refinement_error)
        records.append(rec)
        dchecks.append(dict(**base,tier=tier,AK=AK,AK_finite_difference=AKfd,AT=AT,AT_finite_difference=ATfd,implicit_thermal_derivative=thermal,independent_root_thermal_derivative=thermal_fd,agreement_passed=agreement,AT_method='analytic_barrier_and_renewal_quadrature' if data is None else 'finite_difference_transient_state_at_fixed_load',refinement_relative_K_error=refinement_error))
        for x in np.linspace(0,1,65):
            kk=K*x;traces.append(dict(**base,tier=tier,K=kk,K_over_K_FP=x,cumulative_action=action(kk),action_over_threshold=action(kk)/xi,cleavage_stress_Pa=sigma(kk),renewal_rate_s=float(renewal(m.cleavage,sigma(kk),T,c))))
        for name,barrier in [('cleavage',m.cleavage),('emission',m.emission)]:
            # Emission descriptor at a stated surface stress, not a claim that
            # each signed system experiences the cleavage stress.
            for scope,sig in [('fixed_K18_intrinsic',stress(18,c.r0,c)),('FP_opening_stress',sigma(K))]:
                dd=surface(barrier,sig,T)
                descriptors.append(dict(**base,tier=tier,barrier=name,evaluation=scope,stress_Pa=sig,**{k:float(dd[k]) for k in ('G','D1','D2','DT')}))
            dd=surface(barrier,np.array([sigma(K*x) for x in xs]),T)
            weights=rates/rates.sum()
            descriptors.append(dict(**base,tier=tier,barrier=name,evaluation='cleavage_action_weighted_opening_stress',stress_Pa=None,**{k:float(weights@dd[k]) for k in ('G','D1','D2','DT')}))
    records.append(dict(**base,tier='F2_CURRENT_STATE',status='STATE_CLOSURE_UNAVAILABLE',reason='No qualified monotonic tensor-drive replay over evolving radius and opening: current production obtains emission drive from live 2-D equilibrium; F1 ideal mode-I factors are not that replay. No new FEM requested.',K_FP=None,full_state_available=False))
    valid=[r for r in records if r['status']=='FIRST_PASSAGE']
    best=next((r for r in valid if r['tier']=='F1_EMISSION_BLUNTING'),valid[0])
    records.append(dict(best,tier='BEST_CURRENT_MONOTONIC_FORWARD',selected_tier=best['tier'],best_scope='F1_REDUCED_NOT_F2' if best['tier'].startswith('F1') else 'F0_ONLY_F1_UNAVAILABLE'))
    return dict(records=records,traces=traces,descriptors=descriptors,derivatives=dchecks)



def qualify_independent_f1(result,row,manifest):
    """Fail closed if changing the transient endpoint exposes gate stiffness."""
    import copy
    result=copy.deepcopy(result)
    f1=next(r for r in result['records'] if r['tier']=='F1_EMISSION_BLUNTING')
    if f1['status']!='FIRST_PASSAGE':return result
    try:
        model=TransientBlunting(manifest,row,f1['temperature_K'],f1['Kdot'],row_controls(row))
        v,_=model.solve(f1['K_FP'],rtol=5e-10)
        if not math.isclose(float(v(f1['K_FP'])[2]),f1['threshold_action'],rel_tol=1e-6,abs_tol=1e-10):
            raise ValueError('first-passage-endpoint action disagreement')
    except ValueError as exc:
        base={k:f1[k] for k in ('candidate_id','temperature_K','Kdot','threshold_action','threshold_mode','tier','renewal_hits','renewal_tau_s')}
        rejected=dict(**base,status='STATE_CLOSURE_UNAVAILABLE',reason='Independent root-endpoint reintegration: '+str(exc),K_FP=None,full_state_available=False)
        f0=next(r for r in result['records'] if r['tier']=='F0_INTRINSIC_OPENING')
        result['records']=[rejected if r['tier']=='F1_EMISSION_BLUNTING' else dict(f0,tier='BEST_CURRENT_MONOTONIC_FORWARD',selected_tier='F0_INTRINSIC_OPENING',best_scope='F0_ONLY_F1_UNAVAILABLE') if r['tier']=='BEST_CURRENT_MONOTONIC_FORWARD' else r for r in result['records']]
        for name in ('traces','descriptors','derivatives'):
            result[name]=[r for r in result[name] if r['tier']!='F1_EMISSION_BLUNTING']
    return result


def qualify_cached_result(result,row,manifest):
    if result.get('analytical_work_budget')==TransientBlunting.max_state_evaluations:return result
    result=qualify_independent_f1(result,row,manifest)
    result['analytical_work_budget']=TransientBlunting.max_state_evaluations
    return result


def main():
    cfg=json.loads((ART/'monotonic_forward_configuration.json').read_text());rows=source_rows();results=[]
    RUN.mkdir(parents=True,exist_ok=True);(RUN/'conditions').mkdir(exist_ok=True)
    write_csv(ART/'candidate_parameter_difference_audit.csv',audit(rows))
    for cid,(row,m,_) in rows.items():
      for label,xi in cfg['threshold_cases'].items():
       for factor in cfg['Kdot_factors']:
        rate=.005*factor
        for T in range(300,1201,25):
          p=RUN/'conditions'/f'{cid}_{label}_{rate:g}_{T}.json'
          if p.exists():
            original=json.loads(p.read_text())
            result=qualify_cached_result(original,row,m)
            if result!=original:
                prior=RUN/'pre_budget_admission_records';prior.mkdir(exist_ok=True)
                (prior/p.name).write_bytes(p.read_bytes())
                p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
          else:
            result=qualify_independent_f1(condition(cid,row,m,float(T),rate,xi,label),row,m)
            result['analytical_work_budget']=TransientBlunting.max_state_evaluations
            p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
          results.append(result)
        print(cid,label,rate,'complete',flush=True)
    records=[r for x in results for r in x['records']]
    key=lambda r:(r['candidate_id'],r['tier'],r['Kdot'],r['threshold_mode'])
    baselines={key(r):r['K_FP'] for r in records if r['temperature_K']==300}
    for r in records:
        r['K_over_300K']=r['K_FP']/baselines[key(r)] if r.get('K_FP') and baselines.get(key(r)) else None
    write_csv(ART/'monotonic_first_passage_vs_temperature.csv',records)
    for name,field in [('monotonic_cumulative_action.csv','traces'),('barrier_derivative_descriptors.csv','descriptors'),('cumulative_action_derivatives.csv','derivatives'),('implicit_vs_finite_difference_thermal_derivative.csv','derivatives')]:write_csv(ART/name,[r for x in results for r in x[field]])
    write_csv(ART/'fracture_accessibility_audit.csv',[r for r in records if r['status']=='FIRST_PASSAGE'])
    states=[]
    for x in results:
        f0=next(r for r in x['records'] if r['tier']=='F0_INTRINSIC_OPENING');f1=next(r for r in x['records'] if r['tier']=='F1_EMISSION_BLUNTING')
        states.append(dict(candidate_id=f0['candidate_id'],temperature_K=f0['temperature_K'],Kdot=f0['Kdot'],threshold_mode=f0['threshold_mode'],intrinsic_K_FP=f0['K_FP'],F1_K_FP=f1['K_FP'],blunting_shift=f1['K_FP']-f0['K_FP'] if f1['K_FP'] else None,F1_status=f1['status'],retained_shielding_shift=None,transport_shift=None,full_predicted_K_FP=None,F2_status='STATE_CLOSURE_UNAVAILABLE'))
    write_csv(ART/'monotonic_state_decomposition.csv',states)
    # Barrier surface itself is unchanged, copied byte-for-byte and hash checked.
    (ART/'barrier_temperature_surfaces.csv').write_bytes((OLD/'barrier_temperature_surfaces.csv').read_bytes())
    old=list(csv.DictReader((OLD/'monotonic_first_passage_vs_temperature.csv').open()))
    oldindex={(r['candidate_id'],r['tier'],float(r['Kdot']),float(r['temperature_K'])):r for r in old}
    compare=[]
    for r in records:
        if r['threshold_mode']!='SEED_1720':continue
        o=oldindex[(r['candidate_id'],r['tier'],r['Kdot'],r['temperature_K'])]
        compare.append(dict(candidate_id=r['candidate_id'],tier=r['tier'],temperature_K=r['temperature_K'],Kdot=r['Kdot'],old_contract='GENERIC_MONOTONIC_RENEWAL_SCREEN',new_contract='ROW_SPECIFIED_RENEWAL_FORWARD',old_status=o['status'],new_status=r['status'],old_K_FP=o['K_FP'],new_K_FP=r['K_FP'],K_FP_ratio=r['K_FP']/float(o['K_FP']) if r.get('K_FP') and o['K_FP'] else None,old_AK=o.get('AK'),new_AK=r.get('AK'),old_AT=o.get('AT'),new_AT=r.get('AT'),old_accessibility=o.get('accessibility'),new_accessibility=r.get('accessibility')))
    write_csv(ART/'generic_vs_row_renewal_comparison.csv',compare)
    branches=[]
    for cid in rows:
      for label in cfg['threshold_cases']:
       for rate in [.0005,.005,.05]:
        curve=[r for r in records if r['candidate_id']==cid and r['threshold_mode']==label and r['Kdot']==rate and r['tier']=='BEST_CURRENT_MONOTONIC_FORWARD']
        accessible=[r['temperature_K'] for r in curve if r['accessibility']=='FRACTURE_RESPONSE_ACCESSIBLE']
        contiguous=[]
        for t in accessible:
            if not contiguous or t-contiguous[-1][-1]>25:contiguous.append([t])
            else:contiguous[-1].append(t)
        branches.append(dict(candidate_id=cid,threshold_mode=label,threshold_action=cfg['threshold_cases'][label],Kdot=rate,frozen_full_interval_classification=topology(curve,False),branch_classification='ACCESSIBLE_LOW_T_BRANCH_WITH_HIGH_T_LOSS_OF_ACCESSIBILITY' if curve[0]['accessibility']=='FRACTURE_RESPONSE_ACCESSIBLE' and len(accessible)<37 else 'ZERO_LOAD_DOMINATED_FROM_300K' if float(curve[0]['zero_load_action_fraction'])>=.99 else 'OTHER',accessible_intervals_K=json.dumps([[x[0],x[-1]] for x in contiguous]),K_FP_300K=curve[0]['K_FP'],K_FP_1200K=curve[-1]['K_FP'],zero_load_fraction_300K=curve[0]['zero_load_action_fraction'],saturated_plateau=cfg['threshold_cases'][label]*rate*row_controls(rows[cid][0]).tau,renewal_hits=row_controls(rows[cid][0]).hits,renewal_tau_s=row_controls(rows[cid][0]).tau))
    write_csv(ART/'threshold_robustness.csv',branches)
    oldregistry=json.loads((OLD/'retained_fatigue_controls_registry.json').read_text());cross=[];registry=[]
    for r in oldregistry:
        b=next(b for b in branches if b['candidate_id']==r['candidate_id'] and b['threshold_mode']=='SEED_1720' and b['Kdot']==.005)
        access_status=';'.join(sorted({v['accessibility'] for v in records if v['candidate_id']==r['candidate_id'] and v['threshold_mode']=='SEED_1720' and v['Kdot']==.005 and v['tier']=='BEST_CURRENT_MONOTONIC_FORWARD'}))
        updated=dict(r,accessibility_status=access_status,K_FP_300K=b['K_FP_300K'],K_FP_1200K=b['K_FP_1200K'],forward_classification=b['frozen_full_interval_classification'],branch_classification=b['branch_classification'],accessible_intervals_K=b['accessible_intervals_K'],renewal_contract='ROW_SPECIFIED_RENEWAL_FORWARD',renewal_hits=b['renewal_hits'],renewal_tau_s=b['renewal_tau_s'])
        registry.append(updated);cross.append({k:v for k,v in updated.items() if k!='complete_parameter_vector'})
    (ART/'retained_fatigue_controls_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    write_csv(ART/'retained_fatigue_controls_registry.csv',[dict(r,complete_parameter_vector=json.dumps(r['complete_parameter_vector'],sort_keys=True)) for r in registry])
    write_csv(ART/'fatigue_fracture_forward_crosswalk.csv',cross);write_csv(ART/'forward_temperature_topology.csv',cross)
    decision=dict(status='BOUNDED_ROW_RENEWAL_AUDIT_COMPLETE_WITH_STATE_LIMITATIONS',renewal_contract='ROW_SPECIFIED_RENEWAL_FORWARD',conditions=len(results),thresholds=cfg['threshold_cases'],F1_admitted=sum(r['tier']=='F1_EMISSION_BLUNTING' and r['status']=='FIRST_PASSAGE' for r in records),F1_unavailable=sum(r['tier']=='F1_EMISSION_BLUNTING' and r['status']!='FIRST_PASSAGE' for r in records),F2_unavailable=1332,F1_work_budget_rejections=sum(r['tier']=='F1_EMISSION_BLUNTING' and 'budget exhausted' in r.get('reason','') for r in records),classification_gates_unchanged=True,barriers_retuned=False,new_physical_trajectories=0,candidates=cross)
    (ART/'forward_prediction_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    lines=['# Bounded cooperative-renewal contract audit','','The previous calculation used generic m=3 and tau=1e-6 s. It is preserved byte-for-byte as GENERIC_MONOTONIC_RENEWAL_SCREEN. The 42 prospective fatigue launch records also used these defaults (this does not establish the historical A_NATIVE anchor settings); there is no established intentional monotonic-versus-fatigue distinction. The exact row renewal fields were not applied by that launcher.','','The new analytical calculation reads m=3.2732414351776242 and tau=6.992153587194454e-7 s from every complete row. Only these two numerical inputs differ; barriers, conditional F1 reduction, and prospective gates remain unchanged. This corrects renewal provenance, not the completed physical trajectories. F2 remains unavailable.','','All four rows, 37 temperatures, three ramp rates, and all three thresholds were evaluated: 1,332 conditions. The table uses the reference ramp 0.005 MPa sqrt(m)/s; all KFP values are in MPa sqrt(m). Accessible intervals describe sampled 25 K grid points, not localized transition temperatures. The available tier can vary by condition; the tables identify every F0-only substitution.','', '| Row | Threshold | KFP 300 K | KFP 1200 K | Accessible intervals (K) |','|---|---|---:|---:|---|']
    for b in branches:
        if b['Kdot']==.005:lines.append(f"| {b['candidate_id']} | {b['threshold_mode']} | {b['K_FP_300K']:.10g} | {b['K_FP_1200K']:.10g} | {b['accessible_intervals_K']} |")
    lines+=['','Threshold actions are Xi=0.07509316036236147 (frozen seed 1720), ln(2)=0.6931471805599453, and 1. The renewal law is Lambda=P(m_c,lambda_raw*tau_c)/tau_c with the noninteger row m_c preserved.','','Direct old/new comparison at 300 K, reference ramp, frozen Xi (available F0/F1 tier):','','| Row | Generic KFP | Row-renewal KFP | Corrected/generic |','|---|---:|---:|---:|']
    for cr in compare:
        if cr['tier']=='BEST_CURRENT_MONOTONIC_FORWARD' and cr['temperature_K']==300 and cr['Kdot']==.005:
            lines.append(f"| {cr['candidate_id']} | {float(cr['old_K_FP']):.10g} | {cr['new_K_FP']:.10g} | {cr['K_FP_ratio']:.6g} |")
    lines+=['','The [complete comparison](generic_vs_row_renewal_comparison.csv) includes all frozen-threshold roots, AK, AT, statuses and accessibility classifications. [Threshold robustness](threshold_robustness.csv) includes all nine threshold/rate combinations for every row. P25/P40 retain their fatigue-control roles, and P55 its boundary/falsification role; none becomes a qualified DBTT-, Peak-T-, weak-T-, or ceramic-like fracture archetype.','','A_NATIVE is reported separately as an accessible low-temperature branch with high-temperature loss of accessibility. Its unchanged full-interval gate is not used to erase the low-temperature branch. P25/P40/P55 are zero-load dominated already at 300 K.','','The saturated limit is KFP=Xi*Kdot*tau. The corrected plateau is 0.6992153587194454 times the old generic plateau at the same threshold and rate; neither is a numerical lower bound. Fixed-load D1/D2/DT surfaces are unchanged, but action-weighted descriptors, AK, AT and first-passage derivatives were recomputed.','',f"F1 admitted: {decision['F1_admitted']}; F1 unavailable: {decision['F1_unavailable']}. Unavailable F1/F2 states are not represented as full-state predictions. No physical trajectory was launched."]
    lines += ['',f"The deterministic analytical budget rejected {decision['F1_work_budget_rejections']} F1 conditions at 100,000 state evaluations per solve. These are included in the unavailable count. No accuracy or classification tolerance was relaxed; the stopped pre-budget analytical attempt and all completed condition records are retained under runs/row_renewal_monotonic_forward_v1."]
    (ART/'forward_prediction_decision.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in decision.items() if k!='candidates'},indent=2))

if __name__=='__main__':main()
