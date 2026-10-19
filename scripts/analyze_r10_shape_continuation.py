#!/usr/bin/env python3
"""Assemble the executed 19 October cohesive-shape continuation evidence."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
from scipy.stats import chi2


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


def poisson_rate_interval(event_count,exposure_cycles,reward_m=1e-6,alpha=.05):
    if not event_count or not exposure_cycles:return (None,None)
    low=.5*chi2.ppf(alpha/2,2*event_count)/exposure_cycles*reward_m
    high=.5*chi2.ppf(1-alpha/2,2*(event_count+1))/exposure_cycles*reward_m
    return low,high


def hcf_rows(root):
    rows=[]
    for path in sorted(root.rglob('result.json')):
        if 'hcf_' not in str(path):continue
        try:d=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):continue
        if d.get('schema')!='r10.czm-owned-hcf-trajectory/1':continue
        window=d.get('developed_window') or {};patch=float(d.get('patch_m') or 1e-6)
        window_events=round((float(window.get('end_extension_m') or 0.)-float(window.get('start_extension_m') or 0.))/patch) if window else None
        rows.append({'candidate_id':d.get('candidate_id'),'complete_row_sha256':d.get('complete_row_sha256'),
            'Kmax_MPa_sqrt_m':d.get('Kmax_MPa_sqrt_m'),'DeltaK_MPa_sqrt_m':d.get('DeltaK_MPa_sqrt_m'),
            'temperature_K':d.get('temperature_K'),'R':d.get('R'),'frequency_Hz':d.get('frequency_Hz'),
            'phase_count':d.get('phase_count'),'hazard_seed':d.get('hazard_seed'),
            'plastic_feedback_reference':d.get('plastic_feedback_reference'),
            'environment_model':d.get('environment_model','DECLARED_LPZ_CONTROL'),
            'environment_stiffness_Pa_per_m':d.get('environment_stiffness_Pa_per_m'),
            'accepted_cycles':d.get('accepted_cycles'),'accepted_seconds':d.get('accepted_seconds'),
            'cpu_seconds':d.get('cpu_seconds'),'wall_seconds':d.get('wall_seconds'),
            'physical_event_count':d.get('physical_event_count'),'advance_m':d.get('advance_m'),
            'first_event_cycle':d.get('first_event_cycle'),
            'developed_rate_m_per_cycle':d.get('developed_rate_m_per_cycle'),
            'developed_window_cycles':(d.get('developed_window') or {}).get('duration_cycles'),
            'developed_window_events':window_events,
            'disposition':d.get('disposition'),'numeric_failure':d.get('numeric_failure'),
            'unsupported_validities':json.dumps(d.get('unsupported_validities') or []),
            'restart_checkpoint':d.get('restart_checkpoint'),'result_path':str(path.relative_to(root)),
            'result_sha256':sha(path)})
    return rows


def progress_only(root,result_paths):
    rows=[]
    result_cycles={path.parent:float(json.loads(path.read_text()).get('accepted_cycles') or 0.) for path in result_paths}
    for path in sorted(root.rglob('progress.json')):
        if 'hcf_' not in str(path):continue
        d=json.loads(path.read_text())
        if path.parent in result_cycles and float(d.get('accepted_cycles') or 0.)<=result_cycles[path.parent]+1e-12:continue
        rows.append({'candidate_id':d.get('candidate_id'),
            'complete_row_sha256':d.get('complete_row_sha256'),'accepted_cycles':d.get('accepted_cycles'),
            'accepted_seconds':(float(d.get('accepted_cycles') or 0.)/float(d.get('frequency_Hz') or 1000.)),
            'cpu_seconds':d.get('cpu_seconds'),
            'physical_event_count':len(d.get('events') or []),
            'advance_m':(d.get('events') or [{}])[-1].get('extension_m',0.),
            'environment_model':d.get('environment_model','DECLARED_LPZ_CONTROL'),
            'environment_stiffness_Pa_per_m':d.get('environment_stiffness_Pa_per_m'),
            'disposition':'ACCEPTED_PROGRESS_NO_FINAL_RESULT','progress_path':str(path.relative_to(root)),
            'progress_sha256':sha(path)})
    return rows


def developed_ladder(rows,mechanical_audit):
    """Compare explicit-environment native windows with the matching analytical curve."""
    native=sorted((r for r in rows if r.get('environment_model')=='EXPLICIT_MECHANICAL_RESPONSE_ALTERNATIVE'
        and r.get('developed_rate_m_per_cycle') not in (None,'')),key=lambda r:float(r['Kmax_MPa_sqrt_m']))
    alternatives=mechanical_audit['explicit_alternative_closure_comparison']
    analytical=next(r for r in alternatives if r['closure']=='ASSEMBLED_2UM_HALF_HEIGHT_FIXTURE')['rows']
    stream_rows=[]
    for row in native:
        k=float(row['Kmax_MPa_sqrt_m']);rate=float(row['developed_rate_m_per_cycle'])
        prediction=min(analytical,key=lambda r:abs(float(r['Kmax_MPa_sqrt_m'])-k))
        window_events=int(row['developed_window_events'])
        low,high=poisson_rate_interval(window_events,float(row['developed_window_cycles']))
        stream_rows.append({'candidate_id':row['candidate_id'],'complete_row_sha256':row['complete_row_sha256'],
            'Kmax_MPa_sqrt_m':k,'native_developed_rate_m_per_cycle':rate,
            'analytical_rate_m_per_cycle':prediction['ensemble_rate_m_per_cycle'],
            'native_over_analytical':rate/float(prediction['ensemble_rate_m_per_cycle']),
            'poisson_95_rate_low_m_per_cycle':low,'poisson_95_rate_high_m_per_cycle':high,
            'analytical_local_log_slope':prediction['local_log_slope'],
            'developed_window_cycles':row['developed_window_cycles'],'developed_window_events':window_events,
            'hazard_seed':row['hazard_seed'],
            'accepted_cycles':row['accepted_cycles'],'physical_event_count':row['physical_event_count'],
            'result_path':row['result_path'],'result_sha256':row['result_sha256']})
    grouped=[]
    for row in sorted(stream_rows,key=lambda r:r['Kmax_MPa_sqrt_m']):
        if grouped and abs(row['Kmax_MPa_sqrt_m']/grouped[-1][0]['Kmax_MPa_sqrt_m']-1.)<=1e-4:
            grouped[-1].append(row)
        else:grouped.append([row])
    point_rows=[]
    for group in grouped:
        first=group[0];events=sum(int(r['developed_window_events']) for r in group)
        exposure=sum(float(r['developed_window_cycles']) for r in group)
        rate=events*1e-6/exposure;low,high=poisson_rate_interval(events,exposure)
        point_rows.append({'candidate_id':first['candidate_id'],'complete_row_sha256':first['complete_row_sha256'],
            'Kmax_MPa_sqrt_m':sum(r['Kmax_MPa_sqrt_m'] for r in group)/len(group),'replicate_count':len(group),
            'pooled_developed_rate_m_per_cycle':rate,'analytical_rate_m_per_cycle':first['analytical_rate_m_per_cycle'],
            'pooled_native_over_analytical':rate/float(first['analytical_rate_m_per_cycle']),
            'pooled_poisson_95_rate_low_m_per_cycle':low,'pooled_poisson_95_rate_high_m_per_cycle':high,
            'pooled_developed_window_cycles':exposure,'pooled_developed_window_events':events,
            'analytical_local_log_slope':first['analytical_local_log_slope'],
            'hazard_seeds':json.dumps([r['hazard_seed'] for r in group]),
            'result_paths':json.dumps([r['result_path'] for r in group]),
            'result_sha256s':json.dumps([r['result_sha256'] for r in group])})
    point_rows.sort(key=lambda r:r['Kmax_MPa_sqrt_m'])
    interval_rows=[]
    for lo,hi in zip(point_rows,point_rows[1:]):
        slope=math.log(hi['pooled_developed_rate_m_per_cycle']/lo['pooled_developed_rate_m_per_cycle'])/math.log(hi['Kmax_MPa_sqrt_m']/lo['Kmax_MPa_sqrt_m'])
        interval_rows.append({'K_low_MPa_sqrt_m':lo['Kmax_MPa_sqrt_m'],'K_high_MPa_sqrt_m':hi['Kmax_MPa_sqrt_m'],
            'rate_low_m_per_cycle':lo['pooled_developed_rate_m_per_cycle'],'rate_high_m_per_cycle':hi['pooled_developed_rate_m_per_cycle'],
            'native_secant_log_slope':slope,'strict_3p4_to_4p6':3.4<=slope<=4.6,'moderate_2_to_8':2.<=slope<=8.,
            'interval_width_log10rate':math.log10(hi['pooled_developed_rate_m_per_cycle']/lo['pooled_developed_rate_m_per_cycle'])})
    def connected_width(key):
        best=run=0.
        for interval in interval_rows:
            run=run+interval['interval_width_log10rate'] if interval[key] else 0.
            best=max(best,run)
        return best
    return point_rows,interval_rows,stream_rows,{'points':len(point_rows),'streams':len(stream_rows),'intervals':len(interval_rows),
        'strict_connected_width_log10rate':connected_width('strict_3p4_to_4p6'),
        'moderate_connected_width_log10rate':connected_width('moderate_2_to_8')}


def reference_curve_rows(reference):
    windows=reference['native_developed_windows'];rows=[]
    for item in reference['frozen_reference_by_phase_resolution']:
        k=float(item['Kmax_MPa_sqrt_m']);native=min(windows,key=lambda r:abs(float(r['Kmax_MPa_sqrt_m'])-k))
        derivative=item['shrinking_amplitude_derivatives']['0.002']
        row={'Kmax_MPa_sqrt_m':k,'continuum_gauss_rate_m_per_cycle':item['gauss']['32']['ensemble_rate_m_per_cycle'],
            'native_developed_rate_m_per_cycle':native['developed_rate_m_per_cycle'],
            'native_poisson_95_low_m_per_cycle':native['poisson_95_rate_interval_m_per_cycle'][0],
            'native_poisson_95_high_m_per_cycle':native['poisson_95_rate_interval_m_per_cycle'][1],
            'local_log_slope_eps_0p002':derivative['local_log_slope'],
            'accessibility_log_slope_eps_0p002':derivative['accessibility_log_slope'],
            'conditional_rate_log_slope_eps_0p002':derivative['conditional_rate_log_slope']}
        for phase in ('8','16','32','64','128','256'):
            row[f'native_midpoint_{phase}_rate_m_per_cycle']=item['native_midpoint'][phase]['ensemble_rate_m_per_cycle']
        rows.append(row)
    return rows


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source-head',required=True)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    analytical=json.loads((a.root/'three_scale_search_2048/summary.json').read_text())
    mechanical_audit=json.loads((a.root/'reference_audit/mechanical_environment_audit.json').read_text())
    reference_audit=json.loads((a.root/'reference_audit/analytical_native_reference_audit.json').read_text())
    state=list(csv.DictReader((a.root/'state_informed_128/reduced_case_dispositions.csv').open()))
    result_paths=[p for p in a.root.rglob('result.json') if 'hcf_' in str(p)]
    hcf=hcf_rows(a.root);unfinished=progress_only(a.root,result_paths)
    write_csv(a.out/'hcf_results.csv',hcf);write_csv(a.out/'accepted_progress_without_result.csv',unfinished)
    ladder_points,ladder_intervals,ladder_streams,ladder_summary=developed_ladder(hcf,mechanical_audit)
    write_csv(a.out/'explicit_environment_developed_points.csv',ladder_points)
    write_csv(a.out/'explicit_environment_developed_streams.csv',ladder_streams)
    write_csv(a.out/'explicit_environment_native_slope_intervals.csv',ladder_intervals)
    write_csv(a.out/'archived_027_analytical_native_comparison.csv',reference_curve_rows(reference_audit))
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
        'hcf_total_accepted_seconds':sum(float(r.get('accepted_seconds') or 0.) for r in hcf),
        'hcf_total_cpu_seconds':sum(float(r.get('cpu_seconds') or 0.) for r in hcf),
        'hcf_total_events':sum(int(r.get('physical_event_count') or 0) for r in hcf),
        'developed_rate_results':len(developed),'fracture_rows':len(fracture),
        'explicit_environment_developed_ladder':ladder_summary,
        'two_d_rows':len(two_d),'restart_states':len(restarts),
        'reference_environment_audit_sha256':sha(a.root/'reference_audit/mechanical_environment_audit.json'),
        'reference_native_audit_sha256':sha(a.root/'reference_audit/analytical_native_reference_audit.json')}
    atomic(a.out/'campaign_summary.json',summary);print(json.dumps(summary,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
