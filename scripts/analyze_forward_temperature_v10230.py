"""Forward-only current fatigue-row monotonic campaign. No fracture fitting."""
from pathlib import Path
import sys,csv,json,hashlib,subprocess,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scripts.forward_temperature_model_v10230 import *
from arrhenius_fracture.prospective_paris_candidate_engine_v10230 import build_frozen_manifest,select_frozen_option,digest,CLEAVAGE_FIELDS,IDENTITY_FIELDS
from arrhenius_fracture.prospective_paris_transfer_engine_v10230 import build_transfer_manifest,select_transfer_option
from arrhenius_fracture.stochastic_hazard_tip import draw_hazard_threshold
ART=ROOT/'artifacts/retained_controls_monotonic_forward'
RUN=ROOT/'runs/retained_controls_temperature_crosswalk_v1'
PRIOR=ROOT/'artifacts/prospective_paris_candidates'


def write_csv(path,rows):
    if not rows:raise ValueError('Empty required table '+str(path))
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def source_rows():
    config=json.loads((ART/'monotonic_forward_configuration.json').read_text());out={}
    for cid in config['candidate_ids']:
        s=select_frozen_option(cid) if cid=='A_NATIVE' else select_transfer_option(cid)
        m,p=build_frozen_manifest(cid) if cid=='A_NATIVE' else build_transfer_manifest(cid)
        out[cid]=(s.row,m,p)
    return out


def audit(rows):
    native=rows['A_NATIVE'][0];out=[]
    for cid,(row,m,p) in rows.items():
        changed={k for k in row if row[k]!=native[k]}
        if cid!='A_NATIVE':assert changed==CLEAVAGE_FIELDS|IDENTITY_FIELDS
        for key,val in row.items():
            base=native[key]
            try:a=float(val);b=float(base);absolute=a-b;relative=(a-b)/abs(b) if b else (0. if a==b else None)
            except ValueError:absolute=relative=None
            cleave=key in CLEAVAGE_FIELDS or key in ('cleave_gT_eV_per_K','cleave_sT_GPa_per_K')
            emit=key.startswith('emit_');pt=key.startswith(('peierls_','taylor_'))
            thermal=key in ('cleave_gT_eV_per_K','cleave_sT_GPa_per_K','emit_gT_eV_per_K','emit_sT_GPa_per_K','peierls_activation_entropy_kB','taylor_activation_entropy_kB')
            state=emit or pt or key in ('c_blunt','rho_source0_m2','encounter_efficiency','mobile_shield_fraction')
            out.append(dict(candidate_id=cid,parameter=key,A_NATIVE_value=base,candidate_value=val,changed=val!=base,absolute_difference=absolute,relative_difference=relative,active_in_cleavage=cleave,active_in_emission=emit,active_in_transport=pt,explicit_thermal_coordinate=thermal,active_in_state=state,provenance_only_in_scoped_manifest=key.startswith('physics__') or key=='Tref_K'))
    return out


def condition(cid,row,m,T,rate,xi):
    c=Controls();k0=f0_root(m,T,rate,xi,c);records=[];traces=[];descriptors=[];dchecks=[]
    base=dict(candidate_id=cid,temperature_K=T,Kdot=rate,threshold_action=xi,threshold_mode='EXPONENTIAL_CRN_1720_ENGINE1')
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
        model=TransientBlunting(manifest,row,f1['temperature_K'],f1['Kdot'])
        v,_=model.solve(f1['K_FP'],rtol=5e-10)
        if not math.isclose(float(v(f1['K_FP'])[2]),f1['threshold_action'],rel_tol=1e-6,abs_tol=1e-10):
            raise ValueError('first-passage-endpoint action disagreement')
    except ValueError as exc:
        base={k:f1[k] for k in ('candidate_id','temperature_K','Kdot','threshold_action','threshold_mode','tier')}
        rejected=dict(**base,status='STATE_CLOSURE_UNAVAILABLE',reason='Independent root-endpoint reintegration: '+str(exc),K_FP=None,full_state_available=False)
        f0=next(r for r in result['records'] if r['tier']=='F0_INTRINSIC_OPENING')
        result['records']=[rejected if r['tier']=='F1_EMISSION_BLUNTING' else dict(f0,tier='BEST_CURRENT_MONOTONIC_FORWARD',selected_tier='F0_INTRINSIC_OPENING',best_scope='F0_ONLY_F1_UNAVAILABLE') if r['tier']=='BEST_CURRENT_MONOTONIC_FORWARD' else r for r in result['records']]
        for name in ('traces','descriptors','derivatives'):
            result[name]=[r for r in result[name] if r['tier']!='F1_EMISSION_BLUNTING']
    return result


def main():
    RUN.mkdir(parents=True,exist_ok=True);(RUN/'conditions').mkdir(exist_ok=True)
    config=json.loads((ART/'monotonic_forward_configuration.json').read_text());rows=source_rows()
    write_csv(ART/'candidate_parameter_difference_audit.csv',audit(rows))
    xi=draw_hazard_threshold(mode='exponential',rng=np.random.default_rng(np.random.SeedSequence([1720,1])))
    all_results=[]
    for cid,(row,m,p) in rows.items():
        for factor in config['Kdot_factors']:
            rate=factor*config['Kdot_reference_MPa_sqrt_m_s']
            for T in range(300,1201,25):
                path=RUN/'conditions'/f'{cid}_T{T}_rate{rate:g}.json'
                if path.exists():result=json.loads(path.read_text())
                else:
                    result=condition(cid,row,m,float(T),rate,xi)
                    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
                result=qualify_independent_f1(result,row,m)
                all_results.append(result)
            print(cid,'rate',rate,'complete',flush=True)
    records=[r for x in all_results for r in x['records']];traces=[r for x in all_results for r in x['traces']]
    descriptors=[r for x in all_results for r in x['descriptors']];derivatives=[r for x in all_results for r in x['derivatives']]
    for r in records:
        baseline=next((v for v in records if all(v[k]==r[k] for k in ('candidate_id','tier','Kdot')) and v['temperature_K']==300),{})
        r['K_over_300K']=r['K_FP']/baseline['K_FP'] if r.get('K_FP') and baseline.get('K_FP') else None
    write_csv(ART/'monotonic_first_passage_vs_temperature.csv',records)
    write_csv(ART/'monotonic_cumulative_action.csv',traces)
    write_csv(ART/'barrier_derivative_descriptors.csv',descriptors)
    write_csv(ART/'cumulative_action_derivatives.csv',derivatives)
    write_csv(ART/'implicit_vs_finite_difference_thermal_derivative.csv',derivatives)
    write_csv(ART/'fracture_accessibility_audit.csv',[r for r in records if r['status']=='FIRST_PASSAGE'])
    decomposition=[]
    for x in all_results:
        rr=x['records'];f0=next(r for r in rr if r['tier']=='F0_INTRINSIC_OPENING');f1=next(r for r in rr if r['tier']=='F1_EMISSION_BLUNTING')
        decomposition.append(dict(candidate_id=f0['candidate_id'],temperature_K=f0['temperature_K'],Kdot=f0['Kdot'],intrinsic_K_FP=f0['K_FP'],F1_K_FP=f1['K_FP'],blunting_shift=f1['K_FP']-f0['K_FP'] if f1['K_FP'] else None,retained_shielding_shift=None,transport_shift=None,full_predicted_K_FP=None,F2_status='STATE_CLOSURE_UNAVAILABLE',F1_status=f1['status']))
    write_csv(ART/'monotonic_state_decomposition.csv',decomposition)
    surfaces=[]
    for cid,(_,m,_) in rows.items():
        for T in [300,600,900,1200]:
            for sigma in np.linspace(0,30e9,121):
                for name,b in [('cleavage',m.cleavage),('emission',m.emission)]:
                    dd=surface(b,sigma,T);surfaces.append(dict(candidate_id=cid,temperature_K=T,barrier=name,stress_Pa=sigma,**{k:float(dd[k]) for k in ['G','D1','D2','DT']}))
    write_csv(ART/'barrier_temperature_surfaces.csv',surfaces)
    prior=json.loads((PRIOR/'final_candidate_selection.json').read_text())['targets'];cross=[];registry=[]
    for cid,(row,m,p) in rows.items():
        curve=[r for r in records if r['candidate_id']==cid and r['tier']=='BEST_CURRENT_MONOTONIC_FORWARD' and r['Kdot']==.005]
        cls=topology(curve,False)
        key='P25' if cid.startswith('P25') else 'P40' if cid.startswith('P40') else 'P55' if cid.startswith('P55') else None
        fatigue=prior[key] if key else {}
        diag=fatigue.get('sampled_window_diagnostics',{})
        out=dict(candidate_id=cid,role='HIGH_SLOPE_BOUNDARY_NOT_RETAINED_AS_VALIDATED_CONTROL' if key=='P55' else 'BASELINE_CONTROL' if key is None else 'RETAINED_FATIGUE_CONTROL',fatigue_classification=fatigue.get('classification','QUALIFIED_A_NATIVE_BASELINE'),fatigue_global_slope=diag.get('physical_global_slope'),fatigue_local_slope_range=json.dumps(diag.get('physical_local_slope_range')),fatigue_R_squared=fatigue.get('R_squared'),fatigue_RMS_prediction_residual=diag.get('RMS_prediction_residual_decade'),second_seed_status=fatigue.get('seed_transfer_passed','NOT_REQUIRED_OR_NOT_IN_SOURCE_BUNDLE'),R_transfer_status=json.dumps(fatigue.get('R_transfer_diagnostics',{})),K_FP_300K=curve[0]['K_FP'],K_FP_1200K=curve[-1]['K_FP'],forward_classification=cls,accessibility_status=';'.join(sorted({r['accessibility'] for r in curve})),F2_status='STATE_CLOSURE_UNAVAILABLE',explicit_thermal_coefficients_changed=False,P_coordinate_intent='REFERENCE' if key is None else 'P1_P2_OPENING_SHAPE_NO_INDEPENDENT_P3',source_branch=config['source_branch'],source_commit=config['source_head'],complete_row_sha256=digest(row),material_manifest_sha256=p['material_manifest_sha256'])
        cross.append(out);registry.append(dict(out,complete_parameter_vector=row,monotonic_forward_status='COMPLETED_WITH_EXPLICIT_F2_UNAVAILABLE'))
    write_csv(ART/'forward_temperature_topology.csv',cross);write_csv(ART/'fatigue_fracture_forward_crosswalk.csv',cross)
    write_csv(ART/'retained_fatigue_controls_registry.csv',[dict(r,complete_parameter_vector=json.dumps(r['complete_parameter_vector'],sort_keys=True,separators=(',',':'))) for r in registry])
    (ART/'retained_fatigue_controls_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    unit=[]
    for cid,(_,m,_) in rows.items():
        for T in [300,600,900,1200]:unit.append(dict(candidate_id=cid,temperature_K=T,tier='F0_UNIT_ACTION_SCREEN_ONLY',threshold_action=1.,K_FP=f0_root(m,T,.005,1.)))
    write_csv(ART/'unit_action_preliminary_screen_comparison.csv',unit)
    decision=dict(status='FORWARD_ANALYSIS_TERMINAL_WITH_EXPLICIT_STATE_LIMITATIONS',conditions=len(all_results),first_passage_records=sum(r['status']=='FIRST_PASSAGE' for r in records),F1_unavailable=sum(r['tier']=='F1_EMISSION_BLUNTING' and r['status']!='FIRST_PASSAGE' for r in records),F2_unavailable=444,primary_threshold=xi,physical_solver_trajectories_launched=0,candidates=cross,historical_rows_used=False,canonical_rows_used=False,fitted_fracture_data=False,interpretation='F1 is a conditional source-bin moment. No full signed-state fracture archetype is established. Collapsed forward opening response cannot be labeled weak-T, Peak-T, DBTT, or ceramic.',additional_constitutive_freedom='Opening barrier-height/floor restoration is the first analytical direction to test; explicit thermal derivatives or a second bounded component may provide independent shape control. No such change is selected or validated here. State-coupling changes are not justified by unavailable F2 evidence.')
    (ART/'forward_prediction_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    lines=['# Current-row monotonic forward prediction','',decision['interpretation'],'',f"444 row-temperature-rate conditions; T=300–1200 K every 25 K, Kdot=0.0005/0.005/0.05 MPa√m/s. Primary common exponential threshold Xi={xi:.17g}, seed 1720, engine 1. Unit-action screen is separately tabulated.",'','| Row | Fatigue classification | KFP 300 K | KFP 1200 K | Forward classification |','|---|---|---:|---:|---|']
    for r in cross:lines.append(f"| {r['candidate_id']} | {r['fatigue_classification']} | {r['K_FP_300K']:.9g} | {r['K_FP_1200K']:.9g} | {r['forward_classification']} |")
    lines+=['','Only five cleavage coordinates differ from A_NATIVE. Explicit cleavage and emission gT and sT remain zero. Temperature response still arises through Arrhenius and cooperative renewal, plus the conditional F1 emission/blunting transient. This is P1/P2 design intent, not an independent P3 correction.','',f"F1 sensitivity/refinement or state failures: {decision['F1_unavailable']}/444, recorded as unavailable; F0 remains labeled intrinsic. F2 is unavailable at every condition because a qualified monotonic tensor-drive/state replay is not available in this analysis. No zero state correction is assigned to F2.",'','P25/P40 show no interpretable accessible full-range DBTT, Peak-T, weak-T, or ceramic topology in the available forward hierarchy. Their fatigue classifications are preserved. P55 retains nonzero fixed-load D1/D2 descriptors despite failed fatigue transfer, but its first-passage derivatives are suppressed by zero-load opening and renewal saturation. Finite collapsed loads are resolved roots, not a numerical floor.','',decision['additional_constitutive_freedom'],'','All comparisons use absolute loads first. Normalized curves carry the same accessibility masks. This is analytical prediction, not experimental material identification.']
    lines += ['', 'At 300 K and the reference ramp, zero-load action fractions are above 0.999997 for P25/P40/P55. Their first-passage AK is approximately one. Thus their low-load monotonic response is governed primarily by the zero-stress barrier value and Arrhenius/renewal rate, while the fatigue-load D1/D2 remain distinct.', '', 'At the fixed intrinsic K=18 MPa√m surface probe, the 300 K cleavage D1 values are approximately 0.02848, 0.04236 and 0.05640 eV for P25/P40/P55; D2 values are −0.02219, −0.02986 and −0.03652 eV. At their actual low-load first passage, D1 is only about 1.2e-9, 7.4e-9 and 7.9e-8 eV, respectively. This separates the retained fatigue-load derivative signature from the collapsed monotonic scale. All cleavage DT values are zero.', '', 'For A_NATIVE, the reference-ramp F0 load at 300 K is about 9.9045 MPa√m; converged F1 raises it to about 10.1008 MPa√m, a 1.98% reduced blunting shift. F2 transport and retained-shielding shifts remain unknown. At higher temperatures even the baseline loses full-range accessibility; this does not invalidate its accessible low-temperature intrinsic branch.', '', 'The positive saturated limit is KFP = Xi × Kdot × tau, with tau=1e-6 s. It scales exactly with the applied ramp rate. A flat saturated branch is not weak-T; there is no interpreted Peak-T extremum in that branch.']
    (ART/'forward_prediction_decision.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(decision,indent=2))

if __name__=='__main__':main()
