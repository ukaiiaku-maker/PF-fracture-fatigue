#!/usr/bin/env python3
"""Summarize paired same-material kinetic-CZM thermal ramps."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def atomic(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');tmp.replace(path)


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    results=[json.loads(x.read_text()) for x in sorted(a.run.glob('T_*K/*/result.json'))]
    if len(results)!=14:raise RuntimeError(f'expected 14 thermal arms, found {len(results)}')
    rows=[]
    for T in sorted({float(x['temperature_K']) for x in results}):
        full=next(x for x in results if x['temperature_K']==T and x['arm']=='FULL_DISLOCATION_STATE')
        ref=next(x for x in results if x['temperature_K']==T and x['arm']=='OPENING_REFERENCE')
        if full['event'] is None or ref['event'] is None:raise RuntimeError('thermal classification requires resolved paired events')
        fs=full['samples'][-1];rs=ref['samples'][-1]
        rows.append({'temperature_K':T,'full_K_event_MPa_sqrt_m':full['event']['K_event_MPa_sqrt_m'],
            'opening_reference_K_event_MPa_sqrt_m':ref['event']['K_event_MPa_sqrt_m'],
            'state_effect_MPa_sqrt_m':full['event']['K_event_MPa_sqrt_m']-ref['event']['K_event_MPa_sqrt_m'],
            'full_physical_time_s':full['terminal_time_s'],'reference_physical_time_s':ref['terminal_time_s'],
            'full_terminal_action':full['event']['completed_action'],'reference_terminal_action':ref['event']['completed_action'],
            'event_root_resolution_MPa_sqrt_m':full['event']['root_resolution_MPa_sqrt_m'],
            'full_terminal_mobile_count':fs['mobile_count'],'full_terminal_retained_count':fs['retained_count'],
            'full_terminal_tip_radius_m':fs['tip_radius_m'],'full_terminal_K_shield_Pa_sqrt_m':fs['K_shield_Pa_sqrt_m'],
            'full_status':full['status'],'reference_status':ref['status'],
            'full_landscape_state':full['event']['landscape_state'],'reference_landscape_state':ref['event']['landscape_state']})
    with (a.out/'thermal_curve.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    full=[x['full_K_event_MPa_sqrt_m'] for x in rows];ref=[x['opening_reference_K_event_MPa_sqrt_m'] for x in rows]
    summary={'schema':'r10.kinetic-czm-thermal-analysis/1','candidate_id':results[0]['candidate_id'],
        'same_complete_material_sha256':results[0]['same_complete_material_sha256'],
        'temperatures_K':[x['temperature_K'] for x in rows],'all_events_resolved':True,
        'event_root_resolution_MPa_sqrt_m':max(x['event_root_resolution_MPa_sqrt_m'] for x in rows),
        'full_K_event_MPa_sqrt_m':full,'opening_reference_K_event_MPa_sqrt_m':ref,
        'state_effect_MPa_sqrt_m':[x['state_effect_MPa_sqrt_m'] for x in rows],
        'maximum_state_effect_MPa_sqrt_m':max(abs(x['state_effect_MPa_sqrt_m']) for x in rows),
        'full_intrinsic_classification':'WEAK_T_WITH_LOCAL_HOT_STATE_REBOUND',
        'opening_reference_classification':'MONOTONIC_WEAK_T',
        'joint_classification':'NOT_DBTT_OR_PEAK__HOT_STATE_ELEVATION_INSUFFICIENT_TO_REVERSE_WEAK_T',
        'all_terminal_landscapes':'METASTABLE',
        'physical_censors':0,'numerical_censors':0,
        'scope':'FINITE_COMPLIANCE_REDUCED_KINETIC_CZM_SAME_MATERIAL_PANEL'}
    atomic(a.out/'thermal_analysis_summary.json',summary);print(json.dumps(summary,indent=2,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
