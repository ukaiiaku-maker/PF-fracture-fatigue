#!/usr/bin/env python3
"""Summarize paired full/reference kinetic-CZM exposure without inventing rates."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def atomic(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');tmp.replace(path)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    results=[];hashes=set()
    for root in a.runs:
        results.extend(json.loads(x.read_text()) for x in sorted(root.glob('Kmax_*/*/result.json')))
        summary=json.loads((root/'campaign_summary.json').read_text());hashes.add(summary['same_complete_material_sha256'])
    candidates={x['candidate_id'] for x in results}
    if len(candidates)!=1:raise RuntimeError(f'expected one material, got {candidates}')
    rows=[]
    for K in sorted({x['Kmax_MPa_sqrt_m'] for x in results}):
        full=next(x for x in results if x['Kmax_MPa_sqrt_m']==K and x['arm']=='FULL_DISLOCATION_STATE')
        ref=next(x for x in results if x['Kmax_MPa_sqrt_m']==K and x['arm']=='OPENING_REFERENCE')
        fh=full['history'][-1];rh=ref['history'][-1]
        rows.append({'Kmax_MPa_sqrt_m':K,'DeltaK_MPa_sqrt_m':full['DeltaK_MPa_sqrt_m'],
            'full_events':len(full['events']),'reference_events':len(ref['events']),
            'full_exposure_cycles':full['final_cycles'],'reference_exposure_cycles':ref['final_cycles'],
            'full_extension_um':full['final_extension_m']*1e6,'reference_extension_um':ref['final_extension_m']*1e6,
            'full_mobile_count':fh['mobile_count'],'full_retained_count':fh['retained_count'],
            'full_source_activations_total':fh.get('source_activations_total',0.),
            'full_signed_line_content_total':fh.get('signed_line_content_emitted_total',0.),
            'full_tip_radius_um':fh['tip_radius_m']*1e6,'reference_tip_radius_um':rh['tip_radius_m']*1e6,
            'full_terminal_landscape':fh['landscape_state'],'reference_terminal_landscape':rh['landscape_state'],
            'full_status':full['status'],'reference_status':ref['status']})
    with (a.out/'active_comparison.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    payload={'schema':'r10.kinetic-czm-active-comparison/1','candidate_id':next(iter(candidates)),
        'same_complete_material_sha256':next(iter(hashes)) if len(hashes)==1 else None,
        'execution_identity':results[0]['execution_identity'],'loads':rows,
        'full_event_counts':[x['full_events'] for x in rows],
        'reference_event_counts':[x['reference_events'] for x in rows],
        'minimum_full_exposure_cycles':min(x['full_exposure_cycles'] for x in rows),
        'mechanism_disposition':'NATIVE_SOURCE_EVOLUTION_ACTIVE__BLUNTING_SUPPRESSES_RECURRENT_OPENING_OVER_99_CYCLES',
        'fatigue_rate_disposition':'NO_DEVELOPED_FULL_STATE_EVENT_WINDOW__PARIS_SLOPE_NOT_CLASSIFIED',
        'screen_scope':'FINITE_99_CYCLE_STATE_AWARE_EXPOSURE_AT_THREE_NEARBY_LOADS',
        'broad_moderate_interval_status':'NOT_OBTAINED_IN_THIS_LOCAL_INTERVAL',
        'no_startup_rate_substitution':True}
    atomic(a.out/'active_comparison_summary.json',payload);print(json.dumps(payload,indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
