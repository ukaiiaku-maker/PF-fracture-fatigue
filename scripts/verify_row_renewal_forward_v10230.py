"""Strict independent audit of current-row analytical monotonic outputs."""
from pathlib import Path
import sys,csv,json,hashlib,subprocess,math,re,ast
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.integrate import quad
from scipy.special import gammainc
from scripts.analyze_forward_temperature_v10230 import source_rows
from scripts.analyze_row_renewal_forward_v10230 import row_controls,TransientBlunting
from scripts.forward_temperature_model_v10230 import topology,accessibility
from arrhenius_fracture.prospective_paris_candidate_engine_v10230 import digest
ART=ROOT/'artifacts/row_renewal_monotonic_forward';BASE='b29d1b54c6d69c9db969a9d95cd56a7cf429d022'
REQUIRED=['retained_fatigue_controls_registry.csv','retained_fatigue_controls_registry.json','candidate_parameter_difference_audit.csv','monotonic_forward_configuration.json','monotonic_cumulative_action.csv','monotonic_first_passage_vs_temperature.csv','monotonic_state_decomposition.csv','barrier_temperature_surfaces.csv','barrier_derivative_descriptors.csv','cumulative_action_derivatives.csv','implicit_vs_finite_difference_thermal_derivative.csv','fracture_accessibility_audit.csv','forward_temperature_topology.csv','fatigue_fracture_forward_crosswalk.csv','forward_prediction_decision.md','forward_prediction_decision.json','verification.json','file_hashes.json','figure_data_manifest.json','source_equation_audit.json','renewal_contract_audit.json','executed_candidate_campaign_renewal.csv','generic_vs_row_renewal_comparison.csv','threshold_robustness.csv','numerical_budget_audit.json']

def read(name):return list(csv.DictReader((ART/name).open()))
def close(a,b,rtol=5e-4,atol=2e-7):
    if not math.isclose(float(a),float(b),rel_tol=rtol,abs_tol=atol):raise ValueError(f'numerical disagreement {a} vs {b}')
def validate_registry(registry,source):
    if len(registry)!=4 or {r['candidate_id'] for r in registry}!=set(source):raise ValueError('registry identity changed')
    for r in registry:
        row=source[r['candidate_id']][0]
        if r['complete_parameter_vector']!=row or r['complete_row_sha256']!=digest(row):raise ValueError('candidate parameter changed')

def validate_classification(claim,curve):
    actual=topology(curve,False)
    if claim!=actual:raise ValueError('classification contradicts accessibility or full-state limitation')

def validate_series(series,rows):
    pairs={(r[series['x_column']],r[series['y_column']]) for r in rows if r.get(series['x_column']) and r.get(series['y_column'])}
    numeric={(float(a),float(b)) for a,b in pairs}
    if len(series['x'])!=len(series['y']) or any((float(x),float(y)) not in numeric for x,y in zip(series['x'],series['y'])):raise ValueError('figure coordinates not in source table')

def verify():
    for name in REQUIRED:
        if not (ART/name).is_file():raise ValueError('missing '+name)
    cfg=json.loads((ART/'monotonic_forward_configuration.json').read_text())
    frozen=subprocess.check_output(['git','show','e5668b34:artifacts/row_renewal_monotonic_forward/monotonic_forward_configuration.json'],cwd=ROOT)
    if (ART/'monotonic_forward_configuration.json').read_bytes()!=frozen:raise ValueError('prospective numerical/classification protocol changed')
    if subprocess.check_output(['git','diff',BASE,'--','arrhenius_fracture','artifacts/prospective_paris_candidates'],cwd=ROOT):raise ValueError('production source or frozen fatigue artifacts changed')
    oldroot=ROOT/'artifacts/retained_controls_monotonic_forward'
    contract=json.loads((ART/'renewal_contract_audit.json').read_text())
    if contract['old_result_classification']!='GENERIC_MONOTONIC_RENEWAL_SCREEN':raise ValueError('generic screen not labeled')
    for name,h in contract['old_hashes'].items():
        if hashlib.sha256((oldroot/name).read_bytes()).hexdigest()!=h:raise ValueError('original screen not preserved')
    oldcfg=json.loads((oldroot/'monotonic_forward_configuration.json').read_text())
    for field in ('accessibility_rules','topology_rules','derivative_relative_tolerance','derivative_absolute_tolerance'):
        if cfg[field]!=oldcfg[field]:raise ValueError('prospective gate changed')
    if TransientBlunting.max_state_evaluations!=100000:raise ValueError('analytical work budget changed')
    budget=subprocess.check_output(['git','show','313ed166:artifacts/row_renewal_monotonic_forward/numerical_budget_audit.json'],cwd=ROOT)
    if (ART/'numerical_budget_audit.json').read_bytes()!=budget:raise ValueError('work-budget audit changed')
    source=source_rows();registry=json.loads((ART/'retained_fatigue_controls_registry.json').read_text());validate_registry(registry,source)
    sourceaudit=json.loads((ART/'source_equation_audit.json').read_text())
    for name,h in sourceaudit['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('source provenance changed')
    for name in ['scripts/analyze_row_renewal_forward_v10230.py','scripts/forward_temperature_model_v10230.py']:
        tree=ast.parse((ROOT/name).read_text())
        forbidden={'least_squares','curve_fit','minimize','read_parquet','Popen','spawn','system'}
        if any(isinstance(n,ast.Call) and (getattr(n.func,'id','') in forbidden or getattr(n.func,'attr','') in forbidden) for n in ast.walk(tree)):raise ValueError('unapproved fitting or execution call')
    records=read('monotonic_first_passage_vs_temperature.csv');index={(r['candidate_id'],float(r['temperature_K']),float(r['Kdot']),r['tier'],r['threshold_mode']):r for r in records}
    if len(records)!=5328 or len(index)!=5328:raise ValueError('missing or duplicate condition/tier')
    for cid in source:
        for T in range(300,1201,25):
            for rate in [.0005,.005,.05]:
                for tier in ['F0_INTRINSIC_OPENING','F1_EMISSION_BLUNTING','F2_CURRENT_STATE','BEST_CURRENT_MONOTONIC_FORWARD']:
                    for label in cfg['threshold_cases']:
                        if (cid,float(T),rate,tier,label) not in index:raise ValueError('grid incomplete')
    from arrhenius_fracture.stochastic_hazard_tip import draw_hazard_threshold
    xi=draw_hazard_threshold(mode='exponential',rng=np.random.default_rng(np.random.SeedSequence([1720,1])))
    close(cfg['threshold_cases']['SEED_1720'],xi,1e-14,0)
    close(cfg['threshold_cases']['MEDIAN_LN2'],math.log(2),1e-14,0)
    close(cfg['threshold_cases']['UNIT_ACTION'],1.,1e-14,0)
    verified_f0=0;verified_f1=0
    action_rows=read('monotonic_cumulative_action.csv')
    action_groups={}
    for ar in action_rows:
        key=(ar['candidate_id'],float(ar['temperature_K']),float(ar['Kdot']),ar['tier'],ar['threshold_mode'])
        action_groups.setdefault(key,[]).append(ar)
    gx,gw=np.polynomial.legendre.leggauss(96);gx=(gx+1)/2;gw=gw/2
    for r in records:
        xi=cfg['threshold_cases'][r['threshold_mode']]
        close(r['threshold_action'],xi,1e-14,0)
        c=row_controls(source[r['candidate_id']][0]);mc=c.hits;tc=c.tau
        close(r['renewal_hits'],mc,1e-14,0);close(r['renewal_tau_s'],tc,1e-14,0)
        if r['tier']=='F2_CURRENT_STATE':
            if r['status']!='STATE_CLOSURE_UNAVAILABLE' or r['K_FP']:raise ValueError('unqualified full-state substitution')
        if r['status']!='FIRST_PASSAGE':
            if r['K_FP']:raise ValueError('failure converted to finite prediction')
            continue
        cid=r['candidate_id'];m=source[cid][1];T=float(r['temperature_K']);rate=float(r['Kdot']);K=float(r['K_FP'])
        if not math.isfinite(K) or K<=0:raise ValueError('invalid root')
        if r['tier']=='F0_INTRINSIC_OPENING':
            # Direct production surface and independent load-interval quadrature.
            def lam(k):return float(gammainc(mc,float(m.cleavage.rate(min(k*1e6/math.sqrt(2*math.pi*c.r0),30e9),T))*tc)/tc)
            action=quad(lam,0,K,epsabs=1e-14,epsrel=5e-11,limit=250)[0]/rate
            close(action,xi,1e-7,1e-12);close(K*lam(K)/(rate*action),r['AK'],1e-7,1e-10);verified_f0+=1
            sigma_at=lambda kk: min(kk*1e6/math.sqrt(2*math.pi*c.r0),30e9)
            curve=action_groups[(cid,T,rate,r['tier'],r['threshold_mode'])]
            kk=np.array([float(a['K']) for a in curve])
            stress_grid=np.minimum(kk[:,None]*gx[None,:]*1e6/math.sqrt(2*math.pi*c.r0),30e9)
            exact_curve=kk/rate*((gammainc(mc,m.cleavage.rate(stress_grid,T)*tc)/tc)@gw)
            dt=.05
            def temperature_action(temp):
                return K/rate*float((gammainc(mc,m.cleavage.rate(np.minimum(K*gx*1e6/math.sqrt(2*math.pi*c.r0),30e9),temp)*tc)/tc)@gw)
            independent_AT=(math.log(temperature_action(T+dt))-math.log(temperature_action(T-dt)))/(2*dt)
            close(r['AT'],independent_AT)
        if r['tier']=='F1_EMISSION_BLUNTING':
            # Fresh independent stricter transient solve, not a cached root.
            f=TransientBlunting(m,source[cid][0],T,rate,c);v,_=f.solve(K,rtol=5e-10)
            close(v(K)[2],xi,1e-6,1e-10);close(f.state(K,v(K)[:2])[0],r['r_eff_m'],1e-6,1e-14);verified_f1+=1
            sigma_at=lambda kk:f.state(kk,v(kk)[:2])[1]
            curve=action_groups[(cid,T,rate,r['tier'],r['threshold_mode'])]
            exact_curve=np.array([float(v(float(a['K']))[2]) for a in curve])
        if r['tier'] in ('F0_INTRINSIC_OPENING','F1_EMISSION_BLUNTING'):
            if len(curve)!=65:raise ValueError('cumulative action curve incomplete')
            for ar,expected in zip(curve,exact_curve):
                close(ar['cumulative_action'],expected,1e-6,xi*1e-7)
                close(ar['action_over_threshold'],float(ar['cumulative_action'])/xi,1e-12,1e-14)
            zero=float(gammainc(mc,float(m.cleavage.rate(0.,T))*tc)/tc)
            close(r['zero_load_action_fraction'],zero*K/(rate*xi),1e-12,1e-14)
            stresses=np.array([sigma_at(K*x) for x in (np.arange(256)+.5)/256])
            renewal_fraction=gammainc(mc,m.cleavage.rate(stresses,T)*tc)
            close(r['renewal_ceiling_ramp_fraction'],np.mean(renewal_fraction>=.99),1e-12,1e-14)
            close(r['stress_cap_ramp_fraction'],np.mean(stresses>=30e9),1e-12,1e-14)
        baseline=index[(cid,300.,rate,r['tier'],r['threshold_mode'])]
        if baseline['K_FP']:close(r['K_over_300K'],K/float(baseline['K_FP']),1e-12,1e-14)
        actual,_=accessibility(float(r['zero_load_action_fraction']),float(r['renewal_ceiling_ramp_fraction']),float(r['stress_cap_ramp_fraction']),K,float(r['root_relative_residual']))
        if actual!=r['accessibility']:raise ValueError('accessibility changed')
        if r['tier']=='BEST_CURRENT_MONOTONIC_FORWARD':
            selected=index[(cid,T,rate,r['selected_tier'],r['threshold_mode'])]
            if selected['status']!='FIRST_PASSAGE' or selected['K_FP']!=r['K_FP'] or r['full_state_available']!='False':raise ValueError('F0/F1 mislabeled full-state')
    deriv=read('implicit_vs_finite_difference_thermal_derivative.csv')
    if len(deriv)!=verified_f0+verified_f1:raise ValueError('derivative table incomplete')
    for r in deriv:
        if r['agreement_passed']!='True':raise ValueError('derivative gate failed')
        close(r['AK'],r['AK_finite_difference']);close(r['AT'],r['AT_finite_difference']);close(r['implicit_thermal_derivative'],r['independent_root_thermal_derivative'])
    for r in read('barrier_temperature_surfaces.csv'):
        b=getattr(source[r['candidate_id']][1],r['barrier']);sigma=float(r['stress_Pa']);T=float(r['temperature_K']);h=1e-4
        close(r['G'],b.values_eV(sigma,T),1e-13,1e-14)
        d1=-(b.values_eV(sigma*math.exp(h),T)-b.values_eV(sigma*math.exp(-h),T))/(2*h)
        d2=-(b.values_eV(sigma*math.exp(h),T)-2*b.values_eV(sigma,T)+b.values_eV(sigma*math.exp(-h),T))/h**2
        close(r['D1'],d1,2e-5,1e-7);close(r['D2'],d2,2e-4,1e-6)
    for r in read('forward_temperature_topology.csv'):
        curve=[dict(v,K_FP=float(v['K_FP']),thermal_derivative=float(v['thermal_derivative'])) for v in records if v['candidate_id']==r['candidate_id'] and v['tier']=='BEST_CURRENT_MONOTONIC_FORWARD' and float(v['Kdot'])==.005 and v['threshold_mode']=='SEED_1720']
        validate_classification(r['forward_classification'],curve)
    robustness=read('threshold_robustness.csv')
    if len(robustness)!=36:raise ValueError('threshold/rate summary incomplete')
    for b in robustness:
        curve=[r for r in records if r['candidate_id']==b['candidate_id'] and r['threshold_mode']==b['threshold_mode'] and r['Kdot']==b['Kdot'] and r['tier']=='BEST_CURRENT_MONOTONIC_FORWARD']
        good=[float(r['temperature_K']) for r in curve if r['accessibility']=='FRACTURE_RESPONSE_ACCESSIBLE']
        intervals=[]
        for t in good:
            if not intervals or t-intervals[-1][1]>25:intervals.append([t,t])
            else:intervals[-1][1]=t
        if json.loads(b['accessible_intervals_K'])!=intervals:raise ValueError('accessible branch omitted or invented')
        close(b['saturated_plateau'],float(b['threshold_action'])*float(b['Kdot'])*row_controls(source[b['candidate_id']][0]).tau,1e-12,0)
    if not all(b['branch_classification']=='ACCESSIBLE_LOW_T_BRANCH_WITH_HIGH_T_LOSS_OF_ACCESSIBILITY' for b in robustness if b['candidate_id']=='A_NATIVE'):raise ValueError('A_NATIVE branch distinction lost')
    comparison=read('generic_vs_row_renewal_comparison.csv')
    oldrecords=list(csv.DictReader((oldroot/'monotonic_first_passage_vs_temperature.csv').open()))
    oldindex={(r['candidate_id'],float(r['temperature_K']),float(r['Kdot']),r['tier']):r for r in oldrecords}
    if len(comparison)!=1776:raise ValueError('old-new comparison incomplete')
    for cr in comparison:
        key=(cr['candidate_id'],float(cr['temperature_K']),float(cr['Kdot']),cr['tier'])
        before=oldindex[key];after=index[(*key,'SEED_1720')]
        if cr['old_K_FP']!=before['K_FP'] or cr['new_K_FP']!=after['K_FP']:raise ValueError('comparison substituted root')
        for field in ('status','AK','AT','accessibility'):
            if cr['old_'+field]!=before.get(field,'') or cr['new_'+field]!=after.get(field,''):raise ValueError('comparison substituted '+field)
        if after['K_FP'] and before['K_FP']:close(cr['K_FP_ratio'],float(after['K_FP'])/float(before['K_FP']),1e-12,0)
    physical=read('executed_candidate_campaign_renewal.csv')
    if len(physical)!=42:raise ValueError('executed contract audit incomplete')
    for pr in physical:
        path=ROOT/pr['result_path']/'run_args.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=pr['run_args_sha256']:raise ValueError('physical launch metadata changed')
        args=json.loads(path.read_text())
        close(pr['executed_multihit_m'],args['multihit_m'],1e-14,0)
        close(pr['executed_multihit_tau'],args['multihit_tau'],1e-14,0)
    figures=json.loads((ART/'figure_data_manifest.json').read_text())
    if len(figures)!=12:raise ValueError('missing figure')
    for p in figures:
        if not (ART/'figures'/(p['name']+'.png')).is_file():raise ValueError('missing rendered figure')
        for name,h in p['source_hashes'].items():
            if hashlib.sha256((ART/name).read_bytes()).hexdigest()!=h:raise ValueError('figure source stale')
        for s in p['series']:validate_series(s,read(s['table']))
    newrun=ROOT/'runs/row_renewal_monotonic_forward_v1'
    cached=list((newrun/'conditions').glob('*.json'))
    if len(cached)!=1332:raise ValueError('analytical condition cache incomplete')
    if any(json.loads(p.read_text()).get('analytical_work_budget')!=100000 for p in cached):raise ValueError('cache lacks current budget admission')
    for pattern in ['**/run_args.json','**/steps_*.csv','**/stochastic_avalanche_geometry_events.json','**/high_cycle_run_manifest.json']:
        if list(newrun.glob(pattern)):raise ValueError('new physical trajectory in analysis root')
    hashes=json.loads((ART/'file_hashes.json').read_text())
    if not set(REQUIRED)-{'file_hashes.json'}<=set(hashes):raise ValueError('required hash missing')
    actual_files={str(p.relative_to(ART)) for p in ART.rglob('*') if p.is_file() and p.name not in ('file_hashes.json','.DS_Store')}
    if set(hashes)!=actual_files:raise ValueError('artifact hash coverage incomplete')
    for name,h in hashes.items():
        if hashlib.sha256((ART/name).read_bytes()).hexdigest()!=h:raise ValueError('artifact hash mismatch '+name)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('worktree not clean')
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    pattern=re.compile(r'^\s*\d+\s+\S*python\S*\s+-m\s+arrhenius_fracture[.]sharp_front_v10_2_30_')
    if any(pattern.search(s) for s in subprocess.check_output(['ps','-axo','pid,command'],text=True).splitlines()):raise ValueError('physical workers active')
    return dict(passed=True,conditions=1332,tier_records=len(records),F0_roots_independently_verified=verified_f0,F1_roots_reintegrated=verified_f1,derivative_records=len(deriv),figures=12,new_physical_trajectories=0,production_source_and_fatigue_freeze_unchanged=True)

if __name__=='__main__':
    try:print(json.dumps(verify(),indent=2))
    except Exception as exc:print(json.dumps(dict(passed=False,error=str(exc)),indent=2));raise SystemExit(1)
