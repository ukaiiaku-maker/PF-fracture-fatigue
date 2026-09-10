"""Strict terminal verification; missing science never becomes a passing gate."""
from pathlib import Path
import csv
import hashlib
import json
import math
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_v10_2_30_prospective_campaign import harvest,grid_gate,slopes,read
from scripts.verify_v10_2_30_prospective_launch import verify as verify_original
ART=ROOT/'artifacts/prospective_paris_candidates'
RUN=ROOT/'runs/prospective_paris_transfer_v1'
REQUIRED=['p40_launch_pathway_amendment.json','p40_launch_pathway_verification.json',
 'physical_attempt_registry.csv','physical_job_registry_final.csv','p40_pilot_physical_results.csv',
 'p40_pilot_prediction_comparison.csv','p40_pilot_local_slopes.csv','p40_transfer_update.json',
 'final_candidate_parameter_rows.csv','final_candidate_selection.json','physical_developed_rates.csv',
 'physical_local_slopes.csv','physical_state_summary.csv','analytical_vs_physical_rates.csv',
 'analytical_vs_physical_slopes.csv','slope_error_decomposition.csv','second_seed_comparison.csv',
 'R_transfer_comparison.csv','monotonic_side_effect_check.csv','prospective_paris_candidate_decision.json',
 'prospective_paris_candidate_decision.md','PROSPECTIVE_PARIS_CANDIDATE_HANDOFF.md','verification.json','file_hashes.json']


def validate_decision(claim,computed):
    if claim['classification']!=computed['classification']:
        raise ValueError('final classification does not follow independent local/global gates')


def validate_history(attempts):
    paths=[r['attempt_path'] for r in attempts]
    if len(paths)!=len(set(paths)):raise ValueError('physical path reused')
    for r in attempts:
        if str(r.get('resume','')).lower()=='true':raise ValueError('resume forbidden')
        if 'NO_PHYSICS' in r['status'] and str(r.get('physical_initialization','')).lower()=='true':
            raise ValueError('failed preflight admitted as physics')


def validate_generation(job):
    if job['candidate_id']=='P40_TRANSFER_CALIBRATED_GEN2' and job['R']==.1 and job['seed']==1720 and job['Kmax'] in (13.5,18.,21.) and job['independent_validation']:
        raise ValueError('generation-2 calibration load called independent validation')


def physical_source_identity(head):
    # Analysis commits may advance the producer HEAD. The qualified physical
    # source snapshot must remain byte-for-byte identical for pilot reuse.
    paths=['arrhenius_fracture','scripts/run_v10_2_30_weakt_high_cycle_1e12.sh',
           'scripts/run_v10_2_30_weakt_0p55_high_cycle_1e12.sh',
           'scripts/run_v10_2_30_prospective_transfer_jobs.py']
    return subprocess.check_output(['git','ls-tree','-r',head,'--',*paths],cwd=ROOT)


def independently_check_event_ledger(path,job,measured_rate):
    import numpy as np
    steps=list(csv.DictReader((path/'steps_0300K.csv').open()))
    geometry=read(path/'stochastic_avalanche_geometry_events.json')
    kinetic=read(path/'kinetic_tip_cell_audit_v101.json')['records']
    cumulative=0.;previous=0.;developed_da=0.;developed_dn=0.;n=0
    for row in steps:
        cycles=float(row['fatigue_cycles'])
        if not math.isfinite(cycles) or cycles<0:raise ValueError('invalid cycle ledger')
        cumulative+=cycles
        advance=float(row['da_block_m'])
        if float(row['n_fire'])<=0 or advance<=0:continue
        event=geometry[n]
        projected=event['x1']-event['x0']
        if not math.isclose(advance,projected,rel_tol=1e-9,abs_tol=1e-14):raise ValueError('geometry/event advance mismatch')
        if float(row['crack_extension_m'])-advance>=20e-6:
            developed_da+=advance;developed_dn+=cumulative-previous
        previous=cumulative;n+=1
    if n!=len(geometry):raise ValueError('incomplete event ledger')
    if measured_rate is not None and not math.isclose(developed_da/developed_dn,measured_rate,rel_tol=1e-12):
        raise ValueError('raw cycle/event rate disagrees with summary')
    engine_ids={int(r['engine_id']) for r in kinetic}
    if len(engine_ids)!=1:raise ValueError('unexpected engine/seed mapping')
    rng=np.random.default_rng(np.random.SeedSequence([job['seed'],next(iter(engine_ids))]))
    for event in geometry:
        expected=max(float(rng.exponential(1.0)),1e-12)
        if not math.isclose(event['threshold_action'],expected,rel_tol=1e-12):raise ValueError('threshold RNG provenance changed')
        if event['committed_event_length_m']>event['stochastic_proposed_event_length_m']*(1+1e-8):raise ValueError('event exceeds proposal')
    if any(not r['event_localized'] or not r['coupled_hazard_event_restart'] for r in kinetic if r['fired']):
        raise ValueError('first passage or event restart not qualified')


def validate_seed_transfer(primary,seed2):
    import numpy as np
    primary=sorted([r for r in primary if r['Kmax'] in (15,18,21)],key=lambda r:r['Kmax'])
    seed2=sorted(seed2,key=lambda r:r['Kmax'])
    if len(primary)!=3 or len(seed2)!=3:raise ValueError('missing seed transfer points')
    if any(not r['developed_qualified'] for r in primary+seed2):raise ValueError('new seed stationarity failure')
    def slope(rows):return float(np.polyfit(np.log([r['Kmax'] for r in rows]),np.log([r['physical_rate'] for r in rows]),1)[0])
    m1,m2=slope(primary),slope(seed2)
    if abs(m1-m2)>.4:raise ValueError('seed slope transfer failed')
    if any(a['physical_slope']*b['physical_slope']<=0 for a,b in zip(slopes(primary),slopes(seed2))):raise ValueError('seed local slope sign changed')
    residual=float(np.median([abs(r['prediction_residual_decade']) for r in seed2]))
    if residual>.15:raise ValueError('seed rate residual exceeded')
    return m1,m2


def verify():
    missing=[name for name in REQUIRED if not (ART/name).is_file()]
    if missing:raise ValueError('missing campaign completion artifacts: '+', '.join(missing))
    verify_original()
    transfer=read(ART/'p40_transfer_update.json')
    if transfer['transfer_update_number']!=1:raise ValueError('more than one transfer update')
    frozen_transfer=subprocess.check_output(['git','show','46aa535b:artifacts/prospective_paris_candidates/p40_transfer_update.json'],cwd=ROOT)
    if (ART/'p40_transfer_update.json').read_bytes()!=frozen_transfer:raise ValueError('transfer update changed after freeze')
    freeze=read(ART/'transfer_candidate_freeze_v1.json')
    original=subprocess.check_output(['git','show','f2d69226:artifacts/prospective_paris_candidates/transfer_candidate_freeze_v1.json'],cwd=ROOT)
    if (ART/'transfer_candidate_freeze_v1.json').read_bytes()!=original:raise ValueError('prediction or candidate freeze changed after physical launch')
    attempts=list(csv.DictReader((ART/'physical_attempt_registry.csv').open()));validate_history(attempts)
    if sum('NO_PHYSICS' in r['status'] for r in attempts)!=6:raise ValueError('lost preflight attempt accounting')
    decisions=read(ART/'prospective_paris_candidate_decision.json')['targets']
    registry=list(csv.DictReader((ART/'physical_job_registry_final.csv').open()))
    results=[]
    for job in freeze['jobs']:
        validate_generation(job)
        matching=[r for r in registry if r['candidate_id']==job['candidate_id'] and r['job_key']==job['job_key']]
        if len(matching)!=1:raise ValueError('missing or duplicated job registry key')
        rec=matching[0]
        if rec['status']=='NOT_REQUIRED_AFTER_FAILED_GATE':
            if job['stage']=='PILOT' or decisions[job['target']]['classification'] in ('PARIS_WINDOW_TRANSFER_VALIDATED','EFFECTIVE_GLOBAL_SLOPE_ONLY'):
                raise ValueError('required physical job skipped')
            continue
        path=Path(rec['result_path']);result=read(path.parent/(path.name+'__result.json'))
        if result['freeze_sha256']!=hashlib.sha256(original).hexdigest():raise ValueError('wrong launch freeze hash')
        if physical_source_identity(result['launch_head'])!=physical_source_identity('f2d69226'):
            raise ValueError('physical source differs from qualified cohort source commit')
        record=harvest(job,result,freeze);results.append(record)
        if record['physical_rate'] is not None:independently_check_event_ledger(path,job,record['physical_rate'])
        declared=rec.get('physical_rate','')
        if record['physical_rate'] is None and declared not in ('','nan','NaN','None'):
            raise ValueError('censor or numerical exclusion converted to a finite rate')
        if record['physical_rate'] is not None and not math.isclose(float(declared),record['physical_rate'],rel_tol=1e-12):
            raise ValueError('rate calculation mismatch')
    seed_ranks=[]
    for target,claim in decisions.items():
        primary=[r for r in results if r['target']==target and r['R']==.1 and r['seed']==1720]
        computed=grid_gate(primary,target)
        if len(primary)==9:
            if computed['classification'] in ('PARIS_WINDOW_TRANSFER_VALIDATED','EFFECTIVE_GLOBAL_SLOPE_ONLY'):
                seed2=[r for r in results if r['target']==target and r['seed']==1001723]
                Rtransfer=[r for r in results if r['target']==target and r['R'] in (-.95,.5)]
                if len(seed2)!=3 or len(Rtransfer)!=6:raise ValueError('missing seed/R transfer')
                try:
                    m1,m2=validate_seed_transfer(primary,seed2)
                    seed_ok=True
                except ValueError:
                    seed_ok=False
                if not seed_ok or any(not r['developed_qualified'] for r in Rtransfer):
                    computed['classification']='TARGET_NOT_TRANSFERRED_WITH_SINGLE_EXP_FLOOR'
                else:seed_ranks.append((target,m1,m2))
            validate_decision(claim,computed)
        elif claim['classification'] in ('PARIS_WINDOW_TRANSFER_VALIDATED','EFFECTIVE_GLOBAL_SLOPE_ONLY'):
            raise ValueError('validated target missing complete grid')
        elif target=='P40':
            raise ValueError('generation-2 P40 full grid is required')
        elif len(primary)!=3:
            raise ValueError('bounded failed target lacks complete pilot')
        else:
            qualified=all(r['developed_qualified'] for r in primary)
            local=slopes(primary)
            pilot_passed=(qualified and len(local)==2
                and max(abs(r['prediction_residual_decade']) for r in primary)<=.20
                and max(abs(r['physical_slope']-r['predicted_slope']) for r in local)<=1
                and all(r['physical_slope']>0 and abs(r['event_size_contribution'])<abs(r['waiting_contribution']) for r in local)
                and not any(r.get(k,False) for r in primary for k in ('stress_cap_active','barrier_floor_active','renewal_ceiling_active')))
            if pilot_passed:raise ValueError('passing pilot requires the complete grid')
            expected='TARGET_NOT_TRANSFERRED_WITH_SINGLE_EXP_FLOOR' if qualified else 'NUMERICALLY_UNRESOLVED'
            if claim['classification']!=expected:raise ValueError('failed-pilot classification mismatch')
    if [r[0] for r in sorted(seed_ranks,key=lambda r:r[1])] != [r[0] for r in sorted(seed_ranks,key=lambda r:r[2])]:
        raise ValueError('candidate slope ranking changed under second seed')
    mono=list(csv.DictReader((ART/'monotonic_side_effect_check.csv').open()))
    for candidate in ['A_NATIVE']+[v['candidate_id'] for v in decisions.values() if v['classification'] in ('PARIS_WINDOW_TRANSFER_VALIDATED','EFFECTIVE_GLOBAL_SLOPE_ONLY')]:
        subset=[r for r in mono if r['candidate_id']==candidate]
        if sorted(float(r['temperature_K']) for r in subset)!=[300,600,900,1200]:raise ValueError('missing monotonic temperature check')
        for r in subset:
            if r['classification']=='FIRST_PASSAGE' and (not math.isfinite(float(r['K_first_MPa_sqrt_m'])) or abs(float(r['hazard_action'])-1)>1e-7):raise ValueError('invalid monotonic first passage')
    figures=['target_prediction_physical_rates.png','target_prediction_physical_local_slopes.png','barrier_profiles.png','prediction_residuals.png','state_stress_transmission.png','seed_transfer.png','R_transfer.png','P25_P40_P55_comparison.png','monotonic_side_effects.png']
    for name in figures:
        if not (ART/'figures'/name).is_file():raise ValueError('missing figure: '+name)
    hashes=read(ART/'file_hashes.json')
    for name,expected in hashes.items():
        if hashlib.sha256((ART/name).read_bytes()).hexdigest()!=expected:raise ValueError('artifact hash mismatch: '+name)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('worktree not clean')
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    return dict(passed=True,terminal_jobs=len(results),attempts=len(attempts),original_freeze_preserved=True,transfer_updates=1)


if __name__=='__main__':
    try:print(json.dumps(verify(),indent=2))
    except (ValueError,KeyError,FileNotFoundError,AssertionError) as e:
        print(json.dumps(dict(passed=False,error=str(e)),indent=2));raise SystemExit(1)
