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
from scripts.analyze_v10_2_30_prospective_campaign import harvest,grid_gate,slopes,read,completed_event_action
from scripts.verify_v10_2_30_prospective_launch import verify as verify_original
ART=ROOT/'artifacts/prospective_paris_candidates'
RUN=ROOT/'runs/prospective_paris_transfer_v1'
REQUIRED=['p40_launch_pathway_amendment.json','p40_launch_pathway_verification.json',
 'physical_attempt_registry.csv','physical_event_ledger.csv','physical_job_registry_final.csv','p40_pilot_physical_results.csv',
 'p40_pilot_prediction_comparison.csv','p40_pilot_local_slopes.csv','p40_transfer_update.json',
 'final_candidate_parameter_rows.csv','final_candidate_selection.json','physical_developed_rates.csv',
 'physical_local_slopes.csv','physical_state_summary.csv','analytical_vs_physical_rates.csv',
 'analytical_vs_physical_slopes.csv','slope_error_decomposition.csv','second_seed_comparison.csv',
 'R_transfer_comparison.csv','monotonic_side_effect_check.csv','prospective_paris_candidate_decision.json',
 'prospective_paris_candidate_decision.md','PROSPECTIVE_PARIS_CANDIDATE_HANDOFF.md','verification.json','file_hashes.json']


def validate_decision(claim,computed):
    if claim['classification']!=computed['classification']:
        raise ValueError('final classification does not follow independent local/global gates')


def validate_final_selection(selection,decisions,selected_rows,eligible):
    from arrhenius_fracture.prospective_paris_candidate_engine_v10230 import digest
    retained={d['candidate_id'] for d in decisions.values() if d['classification'] in ('PARIS_WINDOW_TRANSFER_VALIDATED','EFFECTIVE_GLOBAL_SLOPE_ONLY')}
    if len(selection['retained_candidate_ids'])!=len(retained) or set(selection['retained_candidate_ids'])!=retained or selection['targets']!=decisions:
        raise ValueError('final selection disagrees with terminal classifications')
    if len(selected_rows)!=len(retained) or {r['candidate_id'] for r in selected_rows}!=retained:
        raise ValueError('retained parameter rows substituted or omitted')
    for row in selected_rows:
        if row['candidate_id'] not in eligible or digest(row)!=eligible[row['candidate_id']]['complete_row_sha256']:
            raise ValueError('retained complete parameter row changed')


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
    checkpoint=read(path/'run_state_checkpoint.json')
    generation=path/'run_state_generations'/checkpoint['generation']
    if set(checkpoint['files'])!={'kinetic.json','outer.json','state.npz'}:raise ValueError('incomplete atomic checkpoint manifest')
    for name,expected in checkpoint['files'].items():
        if hashlib.sha256((generation/name).read_bytes()).hexdigest()!=expected:raise ValueError('atomic checkpoint checksum mismatch')
    for name in ('high_cycle_live_checkpoint.json','final_mechanical_response.png','final_mpz_state_profiles.png','event_da_dN_vs_extension.png','window_da_dN_vs_extension.png','crack_extension_vs_cycles.png'):
        if not (path/name).is_file():raise ValueError('missing physical diagnostic: '+name)
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
        completed_event_action(event)
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
    for name in ('p40_launch_pathway_amendment.json','p40_launch_pathway_verification.json'):
        original_amendment=subprocess.check_output(['git','show','ef390c94:artifacts/prospective_paris_candidates/'+name],cwd=ROOT)
        if (ART/name).read_bytes()!=original_amendment:raise ValueError('prelaunch pathway amendment changed')
    from scripts.analyze_v10_2_30_prospective_pilots import analyze as analyze_original
    original_jobs=read(ART/'prediction_freeze_manifest.json')['physical_jobs']
    original_rows=list(csv.DictReader((ART/'p40_pilot_physical_results.csv').open()))
    if len(original_rows)!=3:raise ValueError('original pilot result accounting incomplete')
    for job,declared in zip(original_jobs,original_rows):
        measured=analyze_original(job)
        if not measured['developed_qualified'] or not math.isclose(measured['physical_rate'],float(declared['physical_rate']),rel_tol=1e-12):raise ValueError('original pilot rate or qualification mismatch')
        independently_check_event_ledger(Path(measured['result_path']),{'seed':job['seed']},measured['physical_rate'])
    transfer=read(ART/'p40_transfer_update.json')
    if transfer['transfer_update_number']!=1:raise ValueError('more than one transfer update')
    frozen_transfer=subprocess.check_output(['git','show','46aa535b:artifacts/prospective_paris_candidates/p40_transfer_update.json'],cwd=ROOT)
    if (ART/'p40_transfer_update.json').read_bytes()!=frozen_transfer:raise ValueError('transfer update changed after freeze')
    freeze=read(ART/'transfer_candidate_freeze_v1.json')
    original=subprocess.check_output(['git','show','f2d69226:artifacts/prospective_paris_candidates/transfer_candidate_freeze_v1.json'],cwd=ROOT)
    if (ART/'transfer_candidate_freeze_v1.json').read_bytes()!=original:raise ValueError('prediction or candidate freeze changed after physical launch')
    attempts=list(csv.DictReader((ART/'physical_attempt_registry.csv').open()));validate_history(attempts)
    if sum('NO_PHYSICS' in r['status'] for r in attempts)!=6:raise ValueError('lost preflight attempt accounting')
    import io
    preserved=list(csv.DictReader(io.StringIO(subprocess.check_output(['git','show','46aa535b:artifacts/prospective_paris_candidates/physical_attempt_registry.csv'],cwd=ROOT,text=True))))
    indexed={r['attempt_path']:r for r in attempts}
    for prior in preserved:
        current=indexed.get(prior['attempt_path'],{})
        if any(current.get(k)!=v for k,v in prior.items()):raise ValueError('preserved attempt history changed')
    for attempt in attempts:
        if 'NO_PHYSICS' in attempt['status']:
            path=Path(attempt['attempt_path'])
            if not path.is_dir():raise ValueError('preflight evidence directory missing')
            if any((path/name).exists() for name in ('kinetic_audit.json','kinetic_tip_cell_audit_v101.json','high_cycle_live_checkpoint.json','stochastic_avalanche_geometry_events.json','developed_fatigue_growth_summary.json')):
                raise ValueError('preflight no-physics claim contradicted by physical artifacts')
    decision_payload=read(ART/'prospective_paris_candidate_decision.json')
    if decision_payload['original_P40_classification']!=transfer['original_pilot_classification']:raise ValueError('original P40 classification changed')
    decisions=decision_payload['targets']
    registry=list(csv.DictReader((ART/'physical_job_registry_final.csv').open()))
    if len(registry)!=len(freeze['jobs']):raise ValueError('extra or missing planned job registry rows')
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
    if len(attempts)!=len(results)+9:raise ValueError('physical attempt accounting incomplete')
    validate_final_selection(read(ART/'final_candidate_selection.json'),decisions,
        list(csv.DictReader((ART/'final_candidate_parameter_rows.csv').open())),
        {c['candidate_id']:c for c in freeze['candidates']})
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
    rates=list(csv.DictReader((ART/'physical_developed_rates.csv').open()))
    bykey={(r['candidate_id'],r['job_key']):r for r in rates}
    if len(bykey)!=len(results) or len(rates)!=len(results):raise ValueError('rate table missing or duplicated results')
    for r in results:
        declared=bykey[(r['candidate_id'],r['job_key'])]['physical_rate']
        if r['physical_rate'] is None:
            if declared not in ('','None','nan','NaN'):raise ValueError('unqualified rate published as finite')
        elif not math.isclose(float(declared),r['physical_rate'],rel_tol=1e-12):raise ValueError('published rate table mismatch')
    expected_slopes=[]
    for cid in {r['candidate_id'] for r in results}:
        for ratio,seed in ((.1,1720),(.1,1001723),(-.95,1720),(.5,1720)):
            expected_slopes.extend(slopes([r for r in results if r['candidate_id']==cid and r['R']==ratio and r['seed']==seed]))
    def slope_key(r):return r['candidate_id'],float(r['R']),int(r['seed']),float(r['Klo']),float(r['Khi'])
    for table in ('physical_local_slopes.csv','analytical_vs_physical_slopes.csv','slope_error_decomposition.csv'):
        published=list(csv.DictReader((ART/table).open()));indexed={slope_key(r):r for r in published}
        if len(indexed)!=len(expected_slopes) or len(published)!=len(expected_slopes):raise ValueError('missing or duplicate local slope')
        for expected in expected_slopes:
            actual=indexed[slope_key(expected)]
            for field in ('physical_slope','predicted_slope','target_slope','event_size_contribution','waiting_contribution'):
                if not math.isclose(float(actual[field]),expected[field],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('published local slope calculation mismatch')
    published_events=list(csv.DictReader((ART/'physical_event_ledger.csv').open()))
    expected_events={}
    for run in results+original_rows:
        path=Path(run['result_path'])
        for event_index,event in enumerate(read(path/'stochastic_avalanche_geometry_events.json'),1):
            expected_events[(str(path),event_index)]=(completed_event_action(event),event['threshold_action'])
    if len(published_events)!=len(expected_events):raise ValueError('published event count mismatch')
    seen=set()
    for event in published_events:
        key=(event['result_path'],int(event['event_index']))
        if key in seen or key not in expected_events:raise ValueError('duplicate or substituted event')
        seen.add(key);action,threshold=expected_events[key]
        if not math.isclose(float(event['physical_hazard_action']),action,rel_tol=1e-12):raise ValueError('published complete event action mismatch')
        if not math.isclose(float(event['threshold_action']),threshold,rel_tol=1e-12):raise ValueError('published event threshold mismatch')
        if not math.isclose(float(event['stochastic_event_probability']),-math.expm1(-action),rel_tol=1e-12):raise ValueError('event probability mismatch')
    hashes=read(ART/'file_hashes.json')
    if not set(REQUIRED)-{'file_hashes.json'} <= set(hashes):raise ValueError('required artifact omitted from hash manifest')
    for name,expected in hashes.items():
        if hashlib.sha256((ART/name).read_bytes()).hexdigest()!=expected:raise ValueError('artifact hash mismatch: '+name)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('worktree not clean')
    process_lines=subprocess.check_output(['ps','-axo','command'],text=True).splitlines()
    if any(' -m arrhenius_fracture.sharp_front_v10_2_30_' in line for line in process_lines):raise ValueError('physical workers remain active')
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    return dict(passed=True,terminal_jobs=len(results),physical_trajectories_total=len(results)+3,events=len(published_events),attempts=len(attempts),original_freeze_preserved=True,transfer_updates=1)


if __name__=='__main__':
    try:print(json.dumps(verify(),indent=2))
    except (ValueError,KeyError,FileNotFoundError,AssertionError) as e:
        print(json.dumps(dict(passed=False,error=str(e)),indent=2));raise SystemExit(1)
