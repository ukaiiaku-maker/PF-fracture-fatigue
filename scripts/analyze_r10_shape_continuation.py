#!/usr/bin/env python3
"""Assemble the executed 19 October cohesive-shape continuation evidence."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+'\n');tmp.replace(path)


def write_csv(path,rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def hcf_rows(root):
    rows=[]
    for path in sorted(root.rglob('result.json')):
        if 'hcf_' not in str(path):continue
        try:d=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):continue
        if d.get('schema')!='r10.czm-owned-hcf-trajectory/1':continue
        rows.append({'candidate_id':d.get('candidate_id'),'complete_row_sha256':d.get('complete_row_sha256'),
            'Kmax_MPa_sqrt_m':d.get('Kmax_MPa_sqrt_m'),'DeltaK_MPa_sqrt_m':d.get('DeltaK_MPa_sqrt_m'),
            'temperature_K':d.get('temperature_K'),'R':d.get('R'),'frequency_Hz':d.get('frequency_Hz'),
            'phase_count':d.get('phase_count'),'hazard_seed':d.get('hazard_seed'),
            'plastic_feedback_reference':d.get('plastic_feedback_reference'),
            'environment_model':d.get('environment_model','DECLARED_LPZ_CONTROL'),
            'environment_stiffness_Pa_per_m':d.get('environment_stiffness_Pa_per_m'),
            'accepted_cycles':d.get('accepted_cycles'),'accepted_seconds':d.get('accepted_seconds'),
            'physical_event_count':d.get('physical_event_count'),'advance_m':d.get('advance_m'),
            'first_event_cycle':d.get('first_event_cycle'),
            'developed_rate_m_per_cycle':d.get('developed_rate_m_per_cycle'),
            'developed_window_cycles':(d.get('developed_window') or {}).get('duration_cycles'),
            'disposition':d.get('disposition'),'numeric_failure':d.get('numeric_failure'),
            'unsupported_validities':json.dumps(d.get('unsupported_validities') or []),
            'restart_checkpoint':d.get('restart_checkpoint'),'result_path':str(path.relative_to(root)),
            'result_sha256':sha(path)})
    return rows


def progress_only(root,result_paths):
    rows=[]
    result_dirs={path.parent for path in result_paths}
    for path in sorted(root.rglob('progress.json')):
        if 'hcf_' not in str(path) or path.parent in result_dirs:continue
        d=json.loads(path.read_text());rows.append({'candidate_id':d.get('candidate_id'),
            'complete_row_sha256':d.get('complete_row_sha256'),'accepted_cycles':d.get('accepted_cycles'),
            'accepted_seconds':(float(d.get('accepted_cycles') or 0.)/float(d.get('frequency_Hz') or 1000.)),
            'physical_event_count':len(d.get('events') or []),
            'advance_m':(d.get('events') or [{}])[-1].get('extension_m',0.),
            'disposition':'ACCEPTED_PROGRESS_NO_FINAL_RESULT','progress_path':str(path.relative_to(root)),
            'progress_sha256':sha(path)})
    return rows


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source-head',required=True)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    analytical=json.loads((a.root/'three_scale_search_2048/summary.json').read_text())
    state=list(csv.DictReader((a.root/'state_informed_128/reduced_case_dispositions.csv').open()))
    result_paths=[p for p in a.root.rglob('result.json') if 'hcf_' in str(p)]
    hcf=hcf_rows(a.root);unfinished=progress_only(a.root,result_paths)
    write_csv(a.out/'hcf_results.csv',hcf);write_csv(a.out/'accepted_progress_without_result.csv',unfinished)
    fracture=list(csv.DictReader((a.root/'fracture_eventful_pair/ramp_dispositions.csv').open()))
    write_csv(a.out/'fracture_results.csv',fracture)
    restarts=[]
    for checkpoint in sorted(a.root.rglob('high_cycle_live_checkpoint.json')):
        state_path=checkpoint.with_name('high_cycle_live_state.npz')
        restarts.append({'checkpoint_path':str(checkpoint.relative_to(a.root)),
            'checkpoint_sha256':sha(checkpoint),'state_path':str(state_path.relative_to(a.root)),
            'state_sha256':sha(state_path) if state_path.is_file() else None})
    write_csv(a.out/'restart_inventory.csv',restarts)
    two_d=[]
    for path in sorted(a.root.glob('two_d_feedback_*/feedback_qualification.json')):
        d=json.loads(path.read_text());two_d.append({'candidate_id':d['candidate_id'],
            'complete_row_sha256':d['same_complete_material_sha256'],
            'state_restore_bit_exact':d['state_restore_bit_exact'],
            'closure_magnitude_Pa':d['closure_magnitude_Pa'],
            'full_trigger':d['cases'][0]['trigger'],'disabled_trigger':d['cases'][1]['trigger'],
            'receipt_path':str(path.relative_to(a.root)),'receipt_sha256':sha(path)})
    write_csv(a.out/'two_d_results.csv',two_d)
    categories={}
    for row in hcf:categories[row['disposition']]=categories.get(row['disposition'],0)+1
    developed=[row for row in hcf if row.get('developed_rate_m_per_cycle') not in (None,'')]
    summary={'schema':'r10.constrained-cohesive-shape-continuation-analysis/1',
        'source_head':a.source_head,'analytical_allocated':analytical['allocated_vectors'],
        'analytical_domain_valid':analytical['domain_valid'],'analytical_strict_passes':analytical['strict_passes'],
        'selected_complete_vectors':analytical['selected_complete_vectors'],
        'state_informed_cases':len(state),'state_informed_events':sum(int(r['event_cycle_count']) for r in state),
        'hcf_final_results':len(hcf),'hcf_unfinished_progress':len(unfinished),
        'hcf_categories':categories,'hcf_total_accepted_cycles':sum(float(r.get('accepted_cycles') or 0.) for r in hcf),
        'hcf_total_events':sum(int(r.get('physical_event_count') or 0) for r in hcf),
        'developed_rate_results':len(developed),'fracture_rows':len(fracture),
        'two_d_rows':len(two_d),'restart_states':len(restarts),
        'reference_environment_audit_sha256':sha(a.root/'reference_audit/mechanical_environment_audit.json'),
        'reference_native_audit_sha256':sha(a.root/'reference_audit/analytical_native_reference_audit.json')}
    atomic(a.out/'campaign_summary.json',summary);print(json.dumps(summary,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
