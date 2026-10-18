#!/usr/bin/env python3
"""Package the transaction-qualified R10 cohesive continuation."""
from __future__ import annotations
import csv,gzip,hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/r10_opening_transaction_continuation'
OUT=ROOT/'integration_evidence/r10_opening_transaction_continuation'

CASES={
 'smooth_corrected':RUN/'corrected_pair_smooth/result.json',
 'smooth_refined':RUN/'corrected_pair_smooth_refined/result.json',
 'bilinear_120um':RUN/'corrected_pair_bilinear_120um/result.json',
 'high_barrier_control':RUN/'accepted_high_barrier_30745_K20/result.json',
 'cross_active_31630':RUN/'search_v2/R10E_COH27231_PL31630_bilinear_a_K28p77/result.json',
 'cross_shield_11267':RUN/'search_v2/R10E_COH27231_PL11267_bilinear_a_K28p77/result.json',
 'cross_opening_15732':RUN/'search_v3/R10E_COH27231_PL15732_bilinear_a_K28p77_120um/result.json',
 'cross_opening_19672':RUN/'search_v3/R10E_COH27231_PL19672_bilinear_a_K28p77_120um/result.json',
 'own_11267_low':RUN/'mechanism_own/own11267_low/result.json',
 'own_11267_high':RUN/'mechanism_own/own11267_high/result.json',
 'own_15732_low':RUN/'mechanism_own/own15732_low/result.json',
 'own_19672_low':RUN/'mechanism_own/own19672_low/result.json',
}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 missing=[str(p) for p in CASES.values() if not p.exists()]
 if missing:raise RuntimeError('unfinished controlling cases: '+', '.join(missing))
 OUT.mkdir(parents=True,exist_ok=True)
 histories={name:json.loads(path.read_text()) for name,path in CASES.items()}
 rows=[]
 for name,d in histories.items():
  cycles=d['cycle_history'];events=d['events']
  support=d.get('process_zone_support_m',d.get('process_zone_structural_boundary_m'))
  support_exceeded=bool(cycles and support is not None and
                        cycles[-1]['tip_radius_m']>support)
  reported_stop=('REDUCED_MECHANICS_SUPPORT_CENSOR' if support_exceeded
                 else d['stop_category'])
  rows.append({'case':name,'candidate_id':d['candidate_id'],'shape':d['shape'],
   'raw_schema':d.get('schema'),'consumer_semantics':'r10.opening-renewal-recurrence/2',
   'mode':d['mode'],'Kmax_MPa_sqrt_m':d['Kmax_MPa_sqrt_m'],
   'physical_exposure_s':cycles[-1]['physical_time_s'] if cycles else 0.,
   'cycles':len(cycles),'events':len(events),
   'extension_m':cycles[-1]['extension_m'] if cycles else 0.,
   'first_accepted_event_time_s':events[0].get('accepted_event_time_s') if events else None,
   'last_accepted_event_time_s':events[-1].get('accepted_event_time_s') if events else None,
   'stop_category':reported_stop,'native_stop_category':d['stop_category'],
   'unsupported_crossover':d['unsupported_crossover_encountered'],
   'process_zone_support_m':support,'reduced_mechanics_support_exceeded':support_exceeded,
   'cohesive_path_observation_limit_m':d.get('cohesive_path_observation_limit_m'),
   'path_support_kind':d.get('path_support_kind','HISTORICAL_LOCAL_SUPPORT'),
   'final_mobile_count':cycles[-1]['mobile_count'] if cycles else None,
   'final_K_shield_Pa_sqrt_m':cycles[-1]['K_shield_Pa_sqrt_m'] if cycles else None,
   'final_tip_radius_m':cycles[-1]['tip_radius_m'] if cycles else None})
 with (OUT/'case_dispositions.csv').open('w',newline='') as stream:
  w=csv.DictWriter(stream,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 with gzip.open(OUT/'controlling_histories.json.gz','wt') as stream:
  json.dump(histories,stream,sort_keys=True,separators=(',',':'))
 shutil.copy2(RUN/'reconciliation/transaction_reconciliation.json',OUT/'transaction_reconciliation.json')
 shutil.copy2(RUN/'mechanism_balanced_registry.csv',OUT/'mechanism_balanced_registry.csv')
 rec=json.loads((OUT/'transaction_reconciliation.json').read_text())
 by={r['case']:r for r in rows}
 fingerprints={}
 with (OUT/'mechanism_balanced_registry.csv').open(newline='') as stream:
  for r in csv.DictReader(stream):fingerprints.setdefault(r['plasticity_physics_sha256'],[]).append(r['candidate_id'])
 summary={'schema':'r10.opening-transaction-continuation/1','parent_commit':'0991faff',
  'prior_evidence_preserved':True,'zero_paris_pass_preserved':True,
  'reporting_corrections':[
   'Executed raw files retain the /1 schema token; the transaction semantics are /2. This label correction does not alter trajectory data.',
   'Endpoints with r_eff greater than L_pz are reported as reduced-mechanics support censors.'
  ],
  'finite_state_2d_replayed':False,
  'finite_state_2d_disposition':'PRESERVED_AT_DECLARED_SCALAR_REDUCED_MECHANICS_SCOPE',
  'transaction':rec,'plasticity_physics_equivalence_groups':fingerprints,
  'cases':rows,
  'decisions':{
   'archived_active30_events':'DISCARDED_AS_SADDLE_ARRIVALS_WITH_FULL_PHASE_ACTION; NOT COMMITTED_RUPTURE',
   'smooth_27231':'BRANCH_TRANSIT_WITHOUT_COMPLETION; ZERO_COMMITTED_ADVANCE',
   'bilinear_27231':'MECHANICAL_SAME_TIME_CASCADE_TO_120UM_OBSERVATION_ENDPOINT',
   'high_barrier_30745':'FINITE_16_CYCLE_NO_EVENT_EXPOSURE',
   'fixed_cohesion_mechanism_comparison':(
    'PL31630_AND_PL11267_NONADVANCE_FOR_64_CYCLES; '
    'PL15732_AND_PL19672_SAME_TIME_CASCADE_TO_120UM'),
   'same_material_thermal_comparison':(
    'WEAKT_15732_AND_PEAK_19672_SUPPORTED_SHAKEDOWN; '
    'WEAKT_11267_REDUCED_MECHANICS_SUPPORT_CENSOR'),
   'developed_recurrent_growth_found':False,'paris_slope_passes':0}}
 (OUT/'results_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 md=f"""# R10 opening transaction and mechanism-balanced continuation

The zero-Paris-pass conclusion remains unchanged. The archived `active30_m` events reproduce the frozen saddle-arrival clock, but their published action is the discarded full-phase trial action. The commitment-density reference uses reflecting contact and the existing cohesive completion cutoff; it does not support accepting those saddle arrivals as committed rupture.

The smooth 27231 event was branch transit at incomplete separation, not a completed material-cell transaction. The corrected trajectory therefore has zero committed advance. The bilinear law satisfies mechanical completion and continues as a same-time cascade through the separate 120 µm cohesive-path observation endpoint while its 50 µm MPZ moves with the front; it is still target-truncated, not fatigue growth.

The mechanism batch is deduplicated by consumed plasticity physics. `PL30745` and `PL31630` remain provenance aliases for plasticity acquisition and are not counted as distinct mechanisms. The batch compares distinct DBTT/Weak-T/Peak and opening/plasticity backgrounds. No empirical cycle damage, per-cell delay, new opening family, or source suppression was introduced.

The high-barrier 30745 control completed 16 cycles without an accepted event. No finite recurrent rate or Paris slope is reported.

At fixed 27231 cohesion and load, the distinct 31630 and 11267 plasticity backgrounds suppress bilinear completion for the full 64-cycle, 0.064 s exposure while remaining inside reduced-mechanics support. The distinct 15732 and 19672 backgrounds leave plasticity inactive and the bilinear response traverses the 120 µm observation path in the first mechanical instant. This is the new mechanism-controlled comparison: plastic state can change the result from an unbounded mechanical cascade to finite non-advance, but no delayed reactivation was observed.

With each material's own cohesion, Weak-T 15732 and Peak 19672 reach supported 32-cycle shakedown. Weak-T 11267 develops an effective radius of 114–117 µm, beyond the 50 µm reduced-mechanics support, so both of its endpoints are model-scope censors rather than qualified arrest. The corrected smooth 27231 result is unchanged by the 64/1 versus 128/2 opening/transport refinement: both have zero committed events and a 1 µm radius; differences in tiny retained/shielding values are not promoted as mechanics.

See `case_dispositions.csv`, `transaction_reconciliation.json`, and `results_summary.json` for the controlling receipts.
"""
 (OUT/'RESULTS.md').write_text(md)
 (OUT/'README.md').write_text('# Contents\n\nThis directory contains the corrected transaction receipt, mechanism-deduplicated registry, controlling native histories, case dispositions, and decision summary.\n')
 hashes={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS.json'}
 (OUT/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'out':str(OUT),'cases':len(rows),'files':len(hashes)},indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
