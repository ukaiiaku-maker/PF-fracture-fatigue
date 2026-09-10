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
from scripts.analyze_v10_2_30_prospective_campaign import read,grid_gate,slopes,target_rate
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
            if d['seed_transfer_passed'] and all(r['developed_qualified'] for r in rt):retained.append(cid)
            else:
                d['pre_transfer_classification']=d['classification'];d['classification']='TARGET_NOT_TRANSFERRED_WITH_SINGLE_EXP_FLOOR';d['transfer_failure']=True
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
        limitations=['Monotonic values are analysis-only no-plastic screening, not full state-resolved monotonic fracture validation.','Generation-2 original pilot loads remain calibration loads, not independent validation.','Recorded pathology diagnostics do not reconstruct unsaved phasewise histories.'])
    (ART/'prospective_paris_candidate_decision.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Prospective Paris-slope candidate campaign','',
        'The original P40 row transferred sensitivity-predicted rates but missed the target local-slope profile. Its row and original predictions remain unchanged. One analysis-only transfer update was frozen before deriving the three new single-EXP candidates.','',
        '| Target | Exact candidate | Classification | Global slope |','|---|---|---|---|']
    for t,d in decisions.items():lines.append(f"| {t} | {d['candidate_id']} | {d['classification']} | {d.get('m_global','not qualified')} |")
    lines+=['','Only the separately recorded retained rows are admitted as fatigue-response controls. The monotonic side check reveals substantial side effects and does not establish material archetypes.','',
        'All raw trajectories remain under runs/. Each launch was fresh, committed, hash-qualified, and capped by the original 1e12-cycle censor. No trajectory was resumed.','',
        'The first 20 µm are excluded using whole events. Local slopes, interval rates, event-size/waiting decomposition, seed transfer, and full nominal ΔK are available in the accompanying tables. No closure-corrected ΔK_eff is reported.']
    (ART/'prospective_paris_candidate_decision.md').write_text('\n'.join(lines)+'\n')
    (ART/'PROSPECTIVE_PARIS_CANDIDATE_HANDOFF.md').write_text('\n'.join(lines)+'\n\nAuthoritative physical run roots: runs/prospective_paris_p40_pilot_v1 and runs/prospective_paris_transfer_v1.\n')
    shutil.copytree(WORK/'figures',ART/'figures',dirs_exist_ok=True)
    old=list(csv.DictReader((ART/'physical_attempt_registry.csv').open()))
    existing={r['attempt_path'] for r in old}
    for r in rows:
        if r['result_path'] not in existing:
            old.append(dict(r,attempt_path=r['result_path'],resume=False,physical_initialization=True))
    write_csv(ART/'physical_attempt_registry.csv',old)
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
