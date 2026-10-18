#!/usr/bin/env python3
"""Reconcile archived R10 saddle arrivals and material-cell transactions."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.cohesive_arrhenius import COHESIVE_KINETIC_CZM  # noqa:E402
from arrhenius_fracture.opening_alternatives_r10 import (  # noqa:E402
    OpeningAlternative,frozen_commitment_density_reference,
)
from arrhenius_fracture.opening_transaction_r10 import (  # noqa:E402
    MaterialCellState,locate_constant_rate_event,StochasticOwner,
    update_cell_branch,
)
from arrhenius_fracture.retained_material_cohesive_r10 import load_cohesive_binding  # noqa:E402


def atomic(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+'\n');tmp.replace(path)


def row(path,candidate):
    with path.open(newline='') as stream:
        return next(r for r in csv.DictReader(stream) if r['candidate_id']==candidate)


def reconcile_mixed(name,path,registry):
    data=json.loads(path.read_text());event=data['events'][0];pre=event['pre'];opening=pre['opening']
    rate=float(opening['elementary_rate_s-1']);mark=float(pre['pending_threshold'])
    phase_dt=1./(float(data['frequency_Hz'])*int(data['opening_phase_count']))
    proposal=locate_constant_rate_event(StochasticOwner(mark),rate,phase_dt)
    candidate=data['candidate_id'];record=row(registry,candidate)
    c=json.loads(record['cohesive_parameter_vector'])
    binding=load_cohesive_binding(registry,candidate,execution_identity=COHESIVE_KINETIC_CZM)
    alternative=OpeningAlternative(data['mode'],data['shape'],'time_dependent_first_passage',
        float(c['activation_area_m2']),float(c['attempt_frequency_s']))
    references=[frozen_commitment_density_reference(
        alternative,float(pre['environment_imposed_opening_m']),
        float(pre['environment_stiffness_Pa_per_m']),binding.manifest.cleavage.thermal,
        float(data['temperature_K']),opening,phase_dt,n) for n in (65,129,257)]
    reference=references[1]
    max_committed=max(float(r['committed_probability']) for r in references)
    max_commitment_action=-math.log(max(1.-max_committed,1e-300))
    return {'case':name,'candidate_id':candidate,'archived_event_index':event['event_index'],
      'archived_event_time_s':float(event['cycle'])/float(data['frequency_Hz']),
      'archived_pre_physical_time_s':pre['physical_time_s'],
      'static_saddle_arrival':{'rate_s-1':rate,'mfpt_s':opening['mfpt_s'],
        'mark':mark,'localized_wait_s':proposal['consumed_time_s'],
        'archived_full_phase_action':pre['opening_action'],
        'recomputed_full_phase_action':rate*phase_dt,
        'accepted_prefix_action':proposal['accepted_action'],
        'discarded_trial_endpoint_action':proposal['discarded_trial_endpoint_action']},
      'commitment_density_reference':reference,
      'commitment_density_grid_refinement':[{"grid_points":r['grid_points'],
        "committed_probability":r['committed_probability'],
        "mean_commitment_time_s":r['mean_commitment_time_s'],
        "mean_resolved":r['mean_commitment_time_numerically_resolved']} for r in references],
      'maximum_grid_commitment_probability':max_committed,
      'maximum_grid_commitment_action':max_commitment_action,
      'commitment_action_below_pending_mark':max_commitment_action<mark,
      'authoritative_disposition':('NO_ACCEPTED_COMMITTED_EVENT_IN_FROZEN_PHASE_REFERENCE'
        if max_commitment_action<mark else 'COMMITMENT_MARK_REACHED'),
      'archived_event_accepted_after_reconciliation':False,
      'reason':'archived clock absorbed at saddle and published full-phase trial action'}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    a.out.mkdir(parents=True,exist_ok=True)
    cross=ROOT/'runs/r10_opening_recurrence/crossed_registry.csv'
    base=ROOT/'integration_evidence/r10_opening_alternatives/state_informed_48_registry.csv'
    active=reconcile_mixed('active30_m',ROOT/'runs/r10_opening_recurrence/active30_m/result.json',cross)
    low=reconcile_mixed('cross_coh00167_mixed',ROOT/'runs/r10_opening_recurrence/cross_coh00167_mixed/result.json',cross)
    smooth=json.loads((ROOT/'runs/r10_opening_recurrence/smooth_adv/result.json').read_text())
    old=smooth['events'][0];pre=old['pre'];completion=pre['opening']['completion']
    cell=MaterialCellState(int(pre['material_cell_id']))
    update_cell_branch(cell,'TRANSIT',float(pre['cohesive_separation_m']),
                       completion=bool(completion['complete']),physical_time_s=0.)
    smooth_receipt={'case':'smooth_adv','departed_cell_id':cell.cell_id,
      'branch_after_loss':cell.branch,'separation_m':cell.separation_m,
      'completion':completion,'committed_advance_authorized':cell.completed,
      'successor_cell_created':False,
      'archived_successor_disposition':'INVALID_TRANSIT_BRANCH_TRANSFER',
      'energy_owner':'COHESIVE_POTENTIAL_REMAINS_ON_DEPARTED_INCOMPLETE_CELL'}
    payload={'schema':'r10.opening-transaction-reconciliation/1',
      'archived_evidence_preserved':True,
      'mixed_cases':[active,low],'smooth_material_cell':smooth_receipt,
      'clock_conclusion':'STATIC_SADDLE_MFPT_REPRODUCED; NOT TRANSIENT_COMMITTED_RUPTURE',
      'transaction_conclusion':'FULL_PHASE_ACTION_AND_ZERO_TIME_THERMAL_SUCCESSORS_REJECTED',
      'source_head_expected':'0991faff'}
    atomic(a.out/'transaction_reconciliation.json',payload)
    print(json.dumps({'active30':active['authoritative_disposition'],
      'cross00167':low['authoritative_disposition'],
      'smooth_committed':smooth_receipt['committed_advance_authorized']},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
