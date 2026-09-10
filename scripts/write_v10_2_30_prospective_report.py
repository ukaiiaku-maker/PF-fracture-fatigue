"""Assemble terminal campaign evidence; refuses incomplete required stages."""
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_v10_2_30_prospective_campaign import read,grid_gate,slopes,target_rate,completed_event_action
from scripts.verify_v10_2_30_prospective_campaign import validate_seed_transfer,physical_source_identity
ART=ROOT/'artifacts/prospective_paris_candidates';WORK=ROOT/'runs/prospective_paris_transfer_v1/analysis_work'


def write_csv(path,rows,fields=None):
    fields=fields or sorted(set().union(*(r.keys() for r in rows)))
    if not fields:raise ValueError('empty table requires explicit schema')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    rows=read(WORK/'harvested_results.json');freeze=read(ART/'transfer_candidate_freeze_v1.json');pilots=read(WORK/'pilot_decisions.json')
    decisions={};retained=[];seed_comparison=[];R_comparison=[]
    for candidate in freeze['candidates']:
        cid=candidate['candidate_id'];target=candidate['target'];pilot=next(r for r in pilots if r['candidate_id']==cid)
        if pilot['completed_pilots']!=3:raise ValueError('pilot incomplete: '+cid)
        primary=[r for r in rows if r['candidate_id']==cid and r['R']==.1 and r['seed']==1720]
        if pilot['pilot_prediction_gate_passed'] or target=='P40':
            if len(primary)!=9:raise ValueError('required grid incomplete: '+cid)
            d=grid_gate(primary,target)
        else:
            d=dict(classification='TARGET_NOT_TRANSFERRED_WITH_SINGLE_EXP_FLOOR' if pilot['qualified'] else 'NUMERICALLY_UNRESOLVED',window_qualified=False)
        d.update(candidate_id=cid,target=target,family=candidate['family'],generation=candidate['generation'],candidate_row_sha256=candidate['complete_row_sha256'],pilot=pilot)
        if d['classification'] in ('PARIS_WINDOW_TRANSFER_VALIDATED','EFFECTIVE_GLOBAL_SLOPE_ONLY'):
            seed2=[r for r in rows if r['candidate_id']==cid and r['stage']=='SEED2'];rt=[r for r in rows if r['candidate_id']==cid and r['stage']=='R_TRANSFER']
            if len(seed2)!=3 or len(rt)!=6:raise ValueError('required seed/R transfer incomplete: '+cid)
            try:
                m1,m2=validate_seed_transfer(primary,seed2);d['seed_transfer_passed']=True
            except ValueError as error:
                d['seed_transfer_passed']=False;d['seed_transfer_failure']=str(error)
                m1=m2=None
            seed_comparison.append(dict(candidate_id=cid,m_seed1=m1,m_seed2=m2,passed=d['seed_transfer_passed'],median_abs_prediction_residual_decade=float(np.median([abs(r['prediction_residual_decade']) for r in seed2])) if all('prediction_residual_decade' in r for r in seed2) else None))
            for r in rt:R_comparison.append(dict(candidate_id=cid,Kmax=r['Kmax'],R=r['R'],applied_full_DeltaK=(1-r['R'])*r['Kmax'],seed=r['seed'],physical_rate=r['physical_rate'],predicted_rate=r['predicted_rate'],developed_qualified=r['developed_qualified']))
            d['R_transfer_diagnostics']={}
            for ratio in (-.95,.5):
                group=sorted([r for r in rt if r['R']==ratio],key=lambda r:r['Kmax'])
                good=len(group)==3 and all(r['developed_qualified'] for r in group)
                d['R_transfer_diagnostics'][str(ratio)]={
                    'developed_qualified':good,
                    'global_slope':float(np.polyfit(np.log([r['Kmax'] for r in group]),np.log([r['physical_rate'] for r in group]),1)[0]) if good else None,
                    'median_abs_prediction_residual_decade':float(np.median([abs(r['prediction_residual_decade']) for r in group])) if good else None,
                    'scope':'fixed-row transfer diagnostic; no additional fitted R parameter or invented R acceptance tolerance'}
            if d['seed_transfer_passed'] and all(r['developed_qualified'] for r in rt):retained.append(cid)
            else:
                d['pre_transfer_classification']=d['classification'];d['classification']='TARGET_NOT_TRANSFERRED_WITH_SINGLE_EXP_FLOOR';d['transfer_failure']=True
        diagnostic_window=sorted([r for r in primary if 13.5<=r['Kmax']<=21],key=lambda r:r['Kmax'])
        if diagnostic_window and all(r['developed_qualified'] for r in diagnostic_window):
            x=np.log([r['Kmax'] for r in diagnostic_window]);local=slopes(diagnostic_window)
            d['sampled_window_diagnostics']={
                'point_count':len(diagnostic_window),
                'scope':'complete six-point primary window' if len(diagnostic_window)==6 else 'three-point pilot diagnostic only; not a qualified full window',
                'target_global_slope':float(np.polyfit(x,np.log([target_rate(target,r['Kmax']) for r in diagnostic_window]),1)[0]),
                'predicted_global_slope':float(np.polyfit(x,np.log([r['predicted_rate'] for r in diagnostic_window]),1)[0]),
                'physical_global_slope':float(np.polyfit(x,np.log([r['physical_rate'] for r in diagnostic_window]),1)[0]),
                'target_local_slope_range':[min(r['target_slope'] for r in local),max(r['target_slope'] for r in local)],
                'physical_local_slope_range':[min(r['physical_slope'] for r in local),max(r['physical_slope'] for r in local)],
                'RMS_prediction_residual_decade':float(np.sqrt(np.mean([r['prediction_residual_decade']**2 for r in diagnostic_window]))),
                'max_abs_prediction_residual_decade':max(abs(r['prediction_residual_decade']) for r in diagnostic_window)}
        d['retained']=cid in retained;decisions[target]=d
    registry=[]
    for job in freeze['jobs']:
        matches=[r for r in rows if r['candidate_id']==job['candidate_id'] and r['job_key']==job['job_key']]
        if len(matches)>1:raise ValueError('duplicate job')
        if matches:registry.append(matches[0])
        else:
            if decisions[job['target']]['retained'] or job['stage']=='PILOT':raise ValueError('required job missing')
            registry.append(dict(job,status='NOT_REQUIRED_AFTER_FAILED_GATE',physical_rate=None,result_path=''))
    # Only call this script after workers terminate and the report is ready to commit.
    import subprocess
    active=subprocess.check_output(['ps','-axo','command'],text=True)
    if ' -m arrhenius_fracture.sharp_front_v10_2_30_' in active:raise ValueError('physical workers still active')
    source_commit='f2d692263518b64a9bbef5619fea9fe359dee037'
    source_hash=hashlib.sha256(physical_source_identity(source_commit)).hexdigest()
    launch_heads=sorted({r['launch_head'] for r in rows})
    equivalence={head:hashlib.sha256(physical_source_identity(head)).hexdigest() for head in launch_heads}
    if any(value!=source_hash for value in equivalence.values()):raise ValueError('pilot reuse source differs from qualified cohort')
    (ART/'physical_source_equivalence.json').write_text(json.dumps(dict(qualified_physical_source_commit=source_commit,physical_source_tree_sha256=source_hash,
        producer_heads=equivalence,scope='entire arrhenius_fracture tree plus both physical shell launchers and the frozen transfer controller',
        note='Producer HEADs may include later analysis-only commits. Reused pilots and grid trajectories must have the exact same qualified physical source snapshot, row hash, manifest hash, and physical contract.'),indent=2)+'\n')
    for r in registry:r['qualified_physical_source_commit']=source_commit
    write_csv(ART/'physical_job_registry_final.csv',registry)
    write_csv(ART/'physical_developed_rates.csv',rows)
    event_rows=[]
    original_runs=list(csv.DictReader((ART/'p40_pilot_physical_results.csv').open()))
    for run in rows+original_runs:
        path=Path(run['result_path']);summary=read(path/'developed_fatigue_growth_summary.json');geometry=read(path/'stochastic_avalanche_geometry_events.json')
        if len(summary['event_measurements'])!=len(geometry):raise ValueError('event ledger incomplete')
        for event,transaction in zip(summary['event_measurements'],geometry):
            action=completed_event_action(transaction)
            record=dict(event,candidate_id=run['candidate_id'],result_path=str(path),producer_head=run.get('launch_head',run.get('producer_head')),
                legacy_summary_physical_hazard_action=event['physical_hazard_action'],physical_hazard_action=action,
                physical_hazard_action_source='stochastic_avalanche_geometry_events.json:event_transaction_audit.hazard_action_completed',
                stochastic_event_probability=-math.expm1(-action),threshold_distribution='unit_exponential',seed_mapping='SeedSequence([seed, engine_id]); NumPy exponential(1)',
                source_row_sha256=run.get('candidate_row_sha256',run.get('source_row_sha256')),material_manifest_sha256=run['material_manifest_sha256'])
            for field in ('x0','x1','y0','y1'):record[field]=transaction.get(field)
            event_rows.append(record)
    write_csv(ART/'physical_event_ledger.csv',event_rows)
    locals=[]
    for cid in [c['candidate_id'] for c in freeze['candidates']]:
        for R,seed in ((.1,1720),(.1,1001723),(-.95,1720),(.5,1720)):
            locals.extend(slopes([r for r in rows if r['candidate_id']==cid and r['R']==R and r['seed']==seed]))
    write_csv(ART/'physical_local_slopes.csv',locals,fields=list(locals[0]) if locals else ['candidate_id','Klo','Khi','physical_slope'])
    write_csv(ART/'analytical_vs_physical_slopes.csv',locals,fields=list(locals[0]) if locals else ['candidate_id','Klo','Khi','physical_slope'])
    write_csv(ART/'slope_error_decomposition.csv',locals,fields=list(locals[0]) if locals else ['candidate_id','event_size_contribution','waiting_contribution'])
    write_csv(ART/'analytical_vs_physical_rates.csv',rows)
    write_csv(ART/'physical_state_summary.csv',[{k:r.get(k) for k in ('candidate_id','Kmax','R','seed','terminal_radius_m','mobile_count','retained_count','K_shield','sigma_back','barrier_floor_active','stress_cap_active','maximum_recorded_renewal_fraction')} for r in rows])
    write_csv(ART/'second_seed_comparison.csv',seed_comparison,fields=['candidate_id','m_seed1','m_seed2','passed','median_abs_prediction_residual_decade'])
    write_csv(ART/'R_transfer_comparison.csv',R_comparison,fields=['candidate_id','Kmax','R','applied_full_DeltaK','seed','physical_rate','predicted_rate','developed_qualified'])
    shutil.copyfile(WORK/'monotonic_side_effect_check_all_frozen.csv',ART/'monotonic_side_effect_check.csv')
    material=list(csv.DictReader((ART/'transfer_candidate_registry_v1.csv').open()))
    write_csv(ART/'final_candidate_parameter_rows.csv',[r for r in material if r['candidate_id'] in retained],fields=list(material[0]))
    selection=dict(retained_candidate_ids=retained,targets=decisions,all_tested_rows_registry='transfer_candidate_registry_v1.csv',selection_scope='bounded prospective fatigue-response controls, not material archetypes')
    (ART/'final_candidate_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    payload=dict(targets=decisions,terminal=True,physical_jobs_executed=len(rows)+3,zero_physics_preflight_failures=6,transfer_updates=1,original_P40_classification='P40_EFFECTIVE_RATE_TRANSFER_ONLY',
        gstar=4.473410023231299e-7,physics_scope='new rows change only five cleavage coordinates; canonical rows, Peierls, Taylor, source closure, first passage, and energy transaction unchanged',
        monotonic_side_effect='No-feedback monotonic screening finds substantial low-load thermal first passage for new rows and high-temperature renewal saturation. No material-archetype promotion.',
        development_convention='Only events wholly beyond 20 micrometres; boundary-overlap estimates preserved separately for original P40',
        limitations=['Monotonic values are analysis-only no-plastic screening, not full state-resolved monotonic fracture validation.','Generation-2 original pilot loads remain calibration loads, not independent validation.','Recorded pathology diagnostics do not reconstruct unsaved phasewise histories.','Legacy block and developed-summary hazard increments omit committed locator-prefix contributions. The campaign event ledger uses the complete action saved by the checked event transaction; raw files and physical evolution are unchanged.'])
    payload['attempt_accounting']=dict(physical_trajectories=len(rows)+3,preflight_no_physics=6,interrupted_trajectories=0,resumed_trajectories=0,physical_censors=sum(r['status']=='PHYSICAL_CENSOR' for r in rows),numerically_unresolved=sum(r['status']=='NUMERICALLY_UNRESOLVED' for r in rows),developed_unqualified=sum(not r['developed_qualified'] for r in rows))
    payload['bounded_candidate_search']='One prospectively selected single-EXP row per target, plus the explicitly allowed P40 generation-2 correction. All were analytically eligible, so no dual-barrier family was admitted. Failed physical transfer is a result for the tested row, not a proof that every row in the family must fail.'
    (ART/'prospective_paris_candidate_decision.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Prospective Paris-slope candidate campaign','',
        'The original P40 row transferred sensitivity-predicted rates but missed the target local-slope profile. Its row and original predictions remain unchanged. One analysis-only transfer update was frozen before deriving the three new single-EXP candidates.','',
        '| Target | Exact candidate | Classification | Global slope |','|---|---|---|---|']
    for t,d in decisions.items():lines.append(f"| {t} | {d['candidate_id']} | {d['classification']} | {d.get('m_global','not qualified')} |")
    lines+=['','Only the separately recorded retained rows are admitted as fatigue-response controls. The monotonic side check reveals substantial side effects and does not establish material archetypes.','',
        'All raw trajectories remain under runs/. Each launch was fresh, committed, hash-qualified, and capped by the original 1e12-cycle censor. No trajectory was resumed.','',
        'The first 20 µm are excluded using whole events. Local slopes, interval rates, event-size/waiting decomposition, seed transfer, and full nominal ΔK are available in the accompanying tables. No closure-corrected ΔK_eff is reported.']
    lines+=['','Window diagnostics (P55 is a three-point pilot only):','',
        '| Target | Target fit | Predicted fit | Physical fit | Physical local range | RMS prediction residual (decade) |',
        '|---|---:|---:|---:|---|---:|']
    for target,d in decisions.items():
        m=d.get('sampled_window_diagnostics')
        if m:
            lo,hi=m['physical_local_slope_range']
            lines.append(f"| {target} | {m['target_global_slope']:.6f} | {m['predicted_global_slope']:.6f} | {m['physical_global_slope']:.6f} | {lo:.6f}–{hi:.6f} | {m['RMS_prediction_residual_decade']:.6f} |")
    lines+=['','Second-seed comparisons use the same Kmax = 15, 18, 21 subset for both seeds.','',
        '| Candidate | Seed 1720 slope | Seed 1001723 slope | Median absolute prediction residual | Passed |',
        '|---|---:|---:|---:|---|']
    for r in seed_comparison:lines.append(f"| {r['candidate_id']} | {r['m_seed1']} | {r['m_seed2']} | {r['median_abs_prediction_residual_decade']} | {r['passed']} |")
    lines+=['','R-transfer results hold every material field fixed. Full applied DeltaK is (1−R)Kmax.','',
        '| Candidate | R | Kmax | Full DeltaK | Physical da/dN (m/cycle) |',
        '|---|---:|---:|---:|---:|']
    for r in R_comparison:lines.append(f"| {r['candidate_id']} | {r['R']} | {r['Kmax']} | {r['applied_full_DeltaK']} | {r['physical_rate']} |")
    lines+=['','Monotonic first-passage K values below are reduced no-feedback screening values, not state-resolved fracture toughness.','',
        '| Row | 300 K | 600 K | 900 K | 1200 K |','|---|---:|---:|---:|---:|']
    mono=list(csv.DictReader((WORK/'monotonic_side_effect_check_all_frozen.csv').open()))
    for cid in dict.fromkeys(r['candidate_id'] for r in mono):
        values=[float(r['K_first_MPa_sqrt_m']) for r in mono if r['candidate_id']==cid]
        lines.append('| '+cid+' | '+' | '.join(f'{v:.6g}' for v in values)+' |')
    lines+=['','Exact retained rows are in final_candidate_parameter_rows.csv; all tested frozen rows remain in transfer_candidate_registry_v1.csv.',
        'Complete event actions come from checked event transactions. The legacy block and summary increments are preserved but are not substituted for whole-event hazard action.',
        'The qualified physical source snapshot is f2d692263518b64a9bbef5619fea9fe359dee037. Later producer commits change analysis only; exact source-tree equivalence is recorded in physical_source_equivalence.json.',
        'Terminal verification command: `python scripts/verify_v10_2_30_prospective_campaign.py` using the qualified environment.']
    (ART/'prospective_paris_candidate_decision.md').write_text('\n'.join(lines)+'\n')
    (ART/'PROSPECTIVE_PARIS_CANDIDATE_HANDOFF.md').write_text('\n'.join(lines)+'\n\nAuthoritative physical run roots: runs/prospective_paris_p40_pilot_v1 and runs/prospective_paris_transfer_v1.\n')
    shutil.copytree(WORK/'figures',ART/'figures',dirs_exist_ok=True)
    old=list(csv.DictReader((ART/'physical_attempt_registry.csv').open()))
    existing={r['attempt_path'] for r in old}
    for r in rows:
        if r['result_path'] not in existing:
            old.append(dict(r,attempt_path=r['result_path'],attempt_number=r['attempt'],parameter_option=r['candidate_id'],Kmax_MPa_sqrt_m=r['Kmax'],deltaK_MPa_sqrt_m=r['applied_full_DeltaK'],cycles_max=1e12,target_ext_um=100,fresh=True,max_wall_seconds=43200,resume=False,physical_initialization=True))
    write_csv(ART/'physical_attempt_registry.csv',old)
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
