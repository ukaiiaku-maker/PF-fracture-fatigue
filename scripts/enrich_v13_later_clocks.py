"""Add residual-clock diagnostics to cached later results; no mechanics replay."""
import argparse
import json
import math
from pathlib import Path
import pickle

from arrhenius_fracture.production_step_loop_v11 import _logmean_rate
from scripts.run_pf_current_source_multifront_field_atlas_v12 import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    path=args.root/'later_companions/summary.json'
    summary=json.loads(path.read_text())
    for result in summary['cases']:
        if not result.get('companions'):
            continue
        record=result['clean_parent_record']
        payload=pickle.loads((args.root/'later_parents'/result['material_case']/'events'/result['case']/'parent/event_context.pkl').read_bytes())
        before=payload['pre_cleavage_checkpoint']; context=payload['context']
        rates={r.candidate_id:r for r in payload['canonical_result'].rates}
        o=result['companions'][0]
        h=next(h for h in before.state.competition.hazard_states if h.candidate_id==o['candidate_id'])
        dt=record['raw_first_passage_s']-context.physical_time_s
        if not 0<=dt<=context.duration_s+math.ulp(context.physical_time_s):
            result['later_nonwinner_residual']={'status':'crossing_outside_captured_interval_not_inferred'}
            continue
        mean=_logmean_rate(h.previous_rate_per_s,rates[h.candidate_id].lambda_per_s)
        H=h.action+mean*dt
        residual=max(0.,h.current_threshold_action-H)
        if h.pending_events:
            residual=0.
        rate=o.get('effective_parent_law_rate_diagnostic_per_s')
        result['later_nonwinner_residual']={'candidate_id':h.candidate_id,'H_at_primary_crossing':H,
            'eta_next':h.current_threshold_action,'event_ordinal':h.completed_event_count+1,
            'residual_action':residual,'threshold_seed':h.threshold_seed,
            'pending_completed_event_ids':[p.event_id for p in h.pending_events],
            'action_roundoff_bound_from_absolute_time':mean*2*math.ulp(context.physical_time_s),
            'post_primary_frozen_residual_s':residual/rate if rate else None,
            'tau_c_s':result['sensitivity_surface'][0]['tau_c_s'] if result.get('sensitivity_surface') else None,
            'scope':'inherited next-clock residual; cumulative H/eta is not used as a late-event near-completion criterion'}
        print(result['material_case'],result['case'],result['later_nonwinner_residual']['post_primary_frozen_residual_s'])
    summary['later_clock_enrichment_performed_without_FEM_or_mark_replay']=True
    atomic_json(path,summary)


if __name__=='__main__':
    main()
