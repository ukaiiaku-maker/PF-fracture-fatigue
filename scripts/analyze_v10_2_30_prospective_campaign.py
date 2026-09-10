"""Independent campaign rates, gates, and source-qualified result harvesting."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from arrhenius_fracture.material_manifest import MaterialManifest
from arrhenius_fracture.prospective_paris_candidate_engine_v10230 import digest
ART=ROOT/'artifacts/prospective_paris_candidates'
RUN=ROOT/'runs/prospective_paris_transfer_v1'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def truth(v):return v is True or v=='True'


def validate_identity(job,result,freeze):
    p=Path(result['result_path']);audit=read(p/'high_cycle_run_manifest.json')
    launch=read(p.parent/(p.name+'__launch.json'))
    if not launch['result_path_virgin_at_launch'] or launch['resume']:
        raise ValueError('reused or resumed trajectory')
    if launch['launch_time_unix'] > (p/'high_cycle_run_manifest.json').stat().st_birthtime:
        raise ValueError('result predates launch')
    if audit['git_head']!=result['launch_head'] or result['launch_head']!=launch['launch_head']:
        raise ValueError('producer HEAD mismatch')
    row=next(c for c in freeze['candidates'] if c['candidate_id']==job['candidate_id'])
    selected=audit['prospective_candidate']
    if selected['candidate_row_sha256']!=row['complete_row_sha256']:
        raise ValueError('candidate source row mismatch')
    if selected['rebonding'] or selected['PT_substitution']:
        raise ValueError('forbidden physical substitution')
    actual=MaterialManifest.from_csv(p/'selected_material_manifest_v10_2_22.csv')
    if digest(actual.as_dict())!=selected['material_manifest_sha256']:
        raise ValueError('physical material manifest mismatch')
    from arrhenius_fracture.prospective_paris_transfer_engine_v10230 import build_transfer_manifest
    expected_manifest,expected_audit=build_transfer_manifest(job['candidate_id'])
    if selected!=expected_audit or actual.as_dict()!=expected_manifest.as_dict():
        raise ValueError('run manifest differs from exact frozen candidate construction')
    args=read(p/'run_args.json');control=read(p/'v10_2_30_fixed_deltaK_control.json')
    expected={'R':job['R'],'frequency_Hz':1000.,'mpz_n_bins':80,'cycles_max':1e12,
              'da_phys':5e-6,'crack_backend':'sharp_wake','target_crack_extension_um':100.,'temperatures':[300.]}
    if any(args[k]!=v for k,v in expected.items()):raise ValueError('physical job conditions mismatch')
    if audit['hazard_seed']!=job['seed']:raise ValueError('wrong seed')
    if not math.isclose(control['target_Kmax_MPa_sqrt_m'],job['Kmax'],rel_tol=1e-12):
        raise ValueError('wrong Kmax')
    if not math.isclose(control['target_deltaK_MPa_sqrt_m'],(1-job['R'])*job['Kmax'],rel_tol=1e-12):
        raise ValueError('wrong full nominal DeltaK')
    return p,audit,args,actual


def recorded_renewal_fraction(step_rows,tau_s):
    """Rate times renewal time; normalized action B is not a rate."""
    return max(float(row['lambda_c']) for row in step_rows)*float(tau_s)


def completed_event_action(event):
    # The block/summary increment can omit a committed locator prefix.
    # The checked transaction preserves the engine's complete interval H.
    action=float(event['event_transaction_audit']['hazard_action_completed'])
    threshold=float(event['threshold_action'])
    if not math.isfinite(action) or not math.isclose(action,threshold,rel_tol=1e-10,abs_tol=1e-14):
        raise ValueError('completed event action does not localize the sampled threshold')
    return action


def interval(events):
    da=sum(e['projected_advance_m'] for e in events);dn=sum(e['cycles_between_events'] for e in events)
    return dict(da_m=da,dN=dn,event_count=len(events),rate=da/dn if da>0 and dn>0 else None)


def harvest(job,result,freeze):
    base=dict(job,**{k:result[k] for k in ('result_path','launch_head','exit_code','wall_seconds','status','attempt','launch_time_unix','completion_time_unix','freeze_sha256','result_path_virgin_at_launch','resume')})
    base.update(physical_rate=None,developed_qualified=False,applied_full_DeltaK=(1-job['R'])*job['Kmax'])
    if result['exit_code']!=0:return base
    p,audit,args,material=validate_identity(job,result,freeze)
    summary=read(p/'developed_fatigue_growth_summary.json')
    base.update(target_reached=summary['target_reached'],total_events=summary['event_count'],cycles=summary['cycles_consumed'],final_extension_um=summary['final_projected_extension_um'],
                candidate_row_sha256=audit['prospective_candidate']['candidate_row_sha256'],material_manifest_sha256=audit['prospective_candidate']['material_manifest_sha256'])
    if not summary['target_reached']:
        base['status']='PHYSICAL_CENSOR' if summary['cycles_consumed']>=1e12*(1-1e-12) else 'NUMERICALLY_UNRESOLVED'
        return base
    events=summary['event_measurements'];selected=[e for e in events if e['projected_extension_pre_m']>=20e-6]
    stat=interval(selected);final=events[-1]['projected_extension_post_m'];start=max(final-50e-6,20e-6);mid=(start+final)/2
    early=interval([e for e in selected if e['projected_extension_post_m']>start and e['projected_extension_pre_m']<mid]);late=interval([e for e in selected if e['projected_extension_post_m']>mid])
    ratio=late['rate']/early['rate'] if late['rate'] and early['rate'] else None
    qualified=stat['event_count']>=10 and stat['da_m']>=50e-6 and ratio is not None and .5<=ratio<=2
    kinetic=read(p/'kinetic_tip_cell_audit_v101.json')['records'];energy=read(p/'hazard_energy_gated_events_v10_2_30.json')
    if not energy or any(e['paris_law_used'] or e['athermal_Gc_used'] or e['independent_toughness_floor_used'] for e in energy):
        raise ValueError('invalid energy transaction')
    step=list(csv.DictReader((p/'steps_0300K.csv').open()))
    floor=material.cleavage.G00_eV*material.cleavage.floor_fraction
    barrier=min(e['hazard_barrier_J']/1.602176634e-19 for e in energy)
    stress=max(float(e['sigma_cleave_eff_Pa']) for e in step)
    renewal=recorded_renewal_fraction(step,args['multihit_tau'])
    last=kinetic[-1]
    base.update(physical_rate=stat['rate'] if qualified else None,unqualified_interval_rate=stat['rate'],developed_qualified=qualified,
        developed_event_count=stat['event_count'],developed_extension_m=stat['da_m'],developed_cycles=stat['dN'],late_early_ratio=ratio,
        mean_event_length_m=stat['da_m']/stat['event_count'] if stat['event_count'] else None,mean_wait_cycles=stat['dN']/stat['event_count'] if stat['event_count'] else None,
        terminal_radius_m=last['persistent_tip_radius_m'],mobile_count=last['state_mobile_count'],retained_count=last['state_retained_count'],K_shield=last['state_active_K_shield_signed_Pa_sqrt_m'],
        sigma_back=last['persistent_sigma_back_Pa'],floor_eV=floor,minimum_event_barrier_eV=barrier,
        max_recorded_stress_Pa=stress,stress_cap_active=stress>=args['sigma_cap_GPa']*1e9,barrier_floor_active=barrier<=floor*(1+1e-10),renewal_ceiling_active=renewal>=.1,maximum_recorded_renewal_fraction=renewal)
    base['initiation_cycles']=events[0]['cycles_post'] if events else None
    developed_path=sum(e['path_advance_m'] for e in selected)
    base['developed_path_rate']=developed_path/stat['dN'] if qualified else None
    base['developed_tortuosity']=developed_path/stat['da_m'] if stat['da_m'] else None
    base['threshold_distribution']='unit_exponential'
    base['seed_mapping']='SeedSequence([seed, engine_id]); NumPy exponential(1)'
    if qualified:base['prediction_residual_decade']=math.log10(stat['rate']/job['predicted_rate'])
    return base


def target_rate(target,K):
    table=[r for r in csv.DictReader((ART/'target_rate_profiles.csv').open()) if r['profile_id']==target+'_STANDARD_WINDOW']
    table.sort(key=lambda r:float(r['Kmax_MPa_sqrt_m']))
    return float(np.exp(np.interp(math.log(K),[math.log(float(r['Kmax_MPa_sqrt_m'])) for r in table],[math.log(float(r['g_target_m_per_cycle'])) for r in table])))


def slopes(rows):
    ordered=sorted(rows,key=lambda r:r['Kmax']);out=[]
    for a,b in zip(ordered,ordered[1:]):
        if not a['physical_rate'] or not b['physical_rate']:continue
        width=math.log(b['Kmax']/a['Kmax']);physical=math.log(b['physical_rate']/a['physical_rate'])/width
        target=math.log(target_rate(b['target'],b['Kmax'])/target_rate(a['target'],a['Kmax']))/width
        predicted=math.log(b['predicted_rate']/a['predicted_rate'])/width
        size=math.log(b['mean_event_length_m']/a['mean_event_length_m'])/width
        waiting=-math.log(b['mean_wait_cycles']/a['mean_wait_cycles'])/width
        if not math.isclose(physical,size+waiting,abs_tol=1e-10):raise ValueError('rate decomposition mismatch')
        out.append(dict(candidate_id=a['candidate_id'],Klo=a['Kmax'],Khi=b['Kmax'],R=a['R'],seed=a['seed'],physical_slope=physical,target_slope=target,predicted_slope=predicted,event_size_contribution=size,waiting_contribution=waiting))
    return out


def grid_gate(rows,target):
    window=sorted([r for r in rows if 13.5<=r['Kmax']<=21],key=lambda r:r['Kmax'])
    if len(window)!=6 or any(not r['developed_qualified'] for r in window):
        return dict(classification='NUMERICALLY_UNRESOLVED',window_qualified=False)
    x=np.log([r['Kmax'] for r in window]);y=np.log([r['physical_rate'] for r in window]);fit=np.polyfit(x,y,1);r2=1-float(np.sum((y-np.polyval(fit,x))**2)/np.sum((y-y.mean())**2))
    local=slopes(window);errors=np.array([r['physical_slope']-r['target_slope'] for r in local]);rates=np.array([math.log10(r['physical_rate']/target_rate(target,r['Kmax'])) for r in window])
    m={'P25':2.5,'P40':4.,'P55':5.5}[target]
    pathology=any(r.get(k,False) for r in window for k in ('stress_cap_active','barrier_floor_active','renewal_ceiling_active'))
    global_ok=abs(fit[0]-m)<=.35 and not pathology
    all_ok=global_ok and r2>=.995 and np.sqrt(np.mean(errors**2))<=.50 and max(abs(errors))<=.75 and np.sqrt(np.mean(rates**2))<=.10 and max(abs(rates))<=.20
    return dict(classification='PARIS_WINDOW_TRANSFER_VALIDATED' if all_ok else 'EFFECTIVE_GLOBAL_SLOPE_ONLY' if global_ok else 'TARGET_NOT_TRANSFERRED_WITH_SINGLE_EXP_FLOOR',
                window_qualified=True,m_global=float(fit[0]),R_squared=r2,RMS_local_slope_error=float(np.sqrt(np.mean(errors**2))),max_local_slope_error=float(max(abs(errors))),RMS_rate_error=float(np.sqrt(np.mean(rates**2))),max_rate_error=float(max(abs(rates))))


def main():
    freeze=read(ART/'transfer_candidate_freeze_v1.json');results=[]
    for job in freeze['jobs']:
        paths=list((RUN/job['candidate_id']).glob(job['job_key']+'__attempt*__result.json'))
        if not paths:continue
        if len(paths)!=1:raise ValueError('multiple attempts require explicit audited selection')
        results.append(harvest(job,read(paths[0]),freeze))
    out=ROOT/'runs/prospective_paris_transfer_v1/analysis_work'
    (out/'harvested_results.json').write_text(json.dumps(results,indent=2)+'\n')
    decisions=[]
    for candidate in freeze['candidates']:
        pilot=[r for r in results if r['candidate_id']==candidate['candidate_id'] and r['stage']=='PILOT'];local=slopes(pilot)
        qualified=len(pilot)==3 and all(r['developed_qualified'] for r in pilot)
        passed=qualified and max(abs(r['prediction_residual_decade']) for r in pilot)<=.20 and len(local)==2 and max(abs(r['physical_slope']-r['predicted_slope']) for r in local)<=1 and all(r['physical_slope']>0 for r in local) and not any(r.get(k,False) for r in pilot for k in ('stress_cap_active','barrier_floor_active','renewal_ceiling_active')) and all(abs(r['event_size_contribution'])<abs(r['waiting_contribution']) for r in local)
        decisions.append(dict(candidate_id=candidate['candidate_id'],completed_pilots=len(pilot),qualified=qualified,pilot_prediction_gate_passed=passed,local_slopes=local))
    (out/'pilot_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n');print(json.dumps(decisions,indent=2))


if __name__=='__main__':main()
