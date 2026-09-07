#!/usr/bin/env python3
"""Sparse frozen checks only after both branch-disabled continuations terminate."""
import argparse
import json
from pathlib import Path
import subprocess
import traceback
import copy
import math
import pickle

from scripts.run_v13_later_clean_parents import CASES
from scripts.qualify_v13_physical_companions import evaluate_parent
from scripts.run_pf_current_source_multifront_field_atlas_v12 import atomic_json
from scripts.qualify_v13_process_restore import safe_json
from scripts.audit_v13_inherited_clocks import balance
from scripts.v13_frozen_support import initialized_engine


def later_clock_driving_factors(event):
    payload=pickle.loads((event/'parent/event_context.pkl').read_bytes())
    checkpoint=payload['pre_cleavage_checkpoint']
    rates=payload['canonical_result'].rates
    T=float(payload['args']['temperatures'][0])
    rows=[]
    with initialized_engine(checkpoint.shared_process_state) as engine:
        for rate in rates:
            preview=copy.deepcopy(engine)
            effective,raw,barrier=preview.lambda_cleave(preview.sigma_tip(rate.K_directional_Pa_sqrt_m/math.sqrt(rate.gamma_rel)),T)
            if not math.isclose(effective,rate.lambda_per_s,rel_tol=1e-12,abs_tol=1e-300):
                raise RuntimeError('later saved pre-primary rate was not reproduced')
            h=next(h for h in checkpoint.state.competition.hazard_states if h.candidate_id==rate.candidate_id)
            rows.append({'candidate_id':rate.candidate_id,'raw_per_s':raw,'effective_per_s':effective,
                'raw_barrier_J':barrier,'H_at_last_accepted_state':h.action,'next_threshold_eta':h.current_threshold_action,
                'next_event_ordinal':h.completed_event_count+1,'threshold_seed':h.threshold_seed,
                'pending_completed_events':[{'event_id':p.event_id,'time_s':p.completion_time_s,'threshold_action':p.action_after} for p in h.pending_events]})
    return {'candidates':rows,'raw_rate_balance_pre':balance(*(r['raw_per_s'] for r in rows)),
        'effective_rate_balance_pre':balance(*(r['effective_per_s'] for r in rows)),
        'scope':'pre-primary frozen driving factors; no assigned cooperative pair kinetic scales'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    parents=args.root/'later_parents'
    if any(not (parents/case/'terminal.json').exists() for case in CASES):
        raise RuntimeError('continuations still active; no extra heavy worker launched')
    plan=json.loads(args.plan.read_text())
    rows=[]
    for case in CASES:
        for event in sorted((parents/case/'events').glob('event*')):
            try:
                result=evaluate_parent(event.name,event.parent,args.root/'later_companions'/case,plan,
                    later_launch_path=parents/case/'launch.json')
                result['inherited_clock_and_pair_driving_factors']=later_clock_driving_factors(event)
            except Exception as exc:
                result={'case':event.name,'status':'FROZEN_COMPANION_STOP','reason':str(exc),
                    'type':type(exc).__name__,'traceback':traceback.format_exc(),'automatic_retry':False}
                atomic_json(args.root/'later_companions'/case/event.name/'stop.json',result)
            result['material_case']=case
            rows.append(result)
            print(case,event.name,result['status'],flush=True)
    atomic_json(args.root/'later_companions/summary.json',safe_json({'cases':rows,
        'producer_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'branching_enabled_in_parent':False,'short_ensemble_launched':False}))


if __name__=='__main__':
    main()
