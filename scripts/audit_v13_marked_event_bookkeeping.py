"""Read-only source/archive audit; never evaluates mechanics or advances clocks."""
from collections import Counter
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from scripts.run_v13_primary_race_short_ensemble import OUT, CASES, case_folder
from scripts.run_pf_current_source_multifront_field_atlas_v12 import atomic_json, sha256
from scripts.v13_value_fingerprint import physical_state_fingerprint as fp

DEST=OUT/'marked_event_readonly_audit'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def length(branch):
    return sum(math.dist(a,b) for a,b in zip(branch.path,branch.path[1:]))


def close(a,b):
    return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)


def audit_pair(case):
    original=OUT/'short_ensemble'/case
    folder=case_folder(case)
    birth=original/'v13_branch'
    single=restore_branch_checkpoint(birth/'canonical_single.json')
    pair=restore_branch_checkpoint(birth/'accepted_pair.json')
    final=restore_branch_checkpoint(folder/'checkpoint/latest.json')
    marks=rows(folder/'v13_primary_race.jsonl')
    accepted=[r for r in marks if r['outcome']=='PAIR_ACCEPTED']
    assert len(accepted)==1
    mark=accepted[0];pid=mark['primary_candidate_id'];cid=mark['companion_candidate_id']
    hazards={h.candidate_id:h for h in single.state.competition.hazard_states}
    expected_companion=f"{cid}#event:{hazards[cid].completed_event_count+1:016d}"
    initial_ids=pair.state.competition.consumed_event_ids
    checks={name:fp(getattr(single.state,name))==fp(getattr(pair.state,name))
            for name in ('competition','rng_state','tip_process_state','event_counters')}
    checks.update(process_exact=fp(single.shared_process_state)==fp(pair.shared_process_state),
        physical_time_exact=single.physical_time_s==pair.physical_time_s,
        primary_consumed_once=initial_ids.count(mark['parent_event_id'])==1,
        companion_not_consumed_at_birth=expected_companion not in initial_ids)
    original_trials=rows(original/'branch_action_trials.jsonl')
    parent=next(t for t in original_trials if t['accepted'] and mark['parent_event_id'] in t['pending_event_ids'])
    pre_release=single.state.energy_ledgers['topology_release_J_per_m']-parent['released_energy_J_per_m']
    pre_cost=single.state.energy_ledgers['hazard_dissipation_J_per_m']-parent['total_dissipative_cost_J_per_m']
    pair_release=pair.state.energy_ledgers['topology_release_J_per_m']-pre_release
    pair_cost=pair.state.energy_ledgers['hazard_dissipation_J_per_m']-pre_cost
    added_release=single.state.stored_energy_J_per_m-pair.state.stored_energy_J_per_m
    checks['one_pair_release_increment']=close(pair_release,parent['released_energy_J_per_m']+added_release)
    checks['pair_energy_margin_matches_mark']=close(pair_release-pair_cost,mark['pair_margin_J_per_m'])
    checks['positive_extra_companion_cost']=pair_cost>parent['total_dissipative_cost_J_per_m']
    daughters=[b for b in pair.state.crack_network.branches if b.generation==1]
    checks['two_initial_5um_arms']=len(daughters)==2 and all(close(length(b),5e-6) for b in daughters)
    checks['pair_adds_only_companion_length_to_single']=close(
        pair.state.crack_network.total_physical_crack_length_m-single.state.crack_network.total_physical_crack_length_m,5e-6)
    checks['pair_total_initial_increment_10um']=close(sum(length(b) for b in daughters),10e-6)
    post=[t for t in rows(folder/'branch_action_trials.jsonl') if t['accepted'] and t['step']>mark['step']]
    mapping=[]
    rates=rows(folder/'directional_rates.jsonl')
    for daughter in daughters:
        candidate=daughter.local_state['candidate_id'];h=hazards[candidate]
        events=[t for t in post if candidate in t['candidate_ids']]
        first=events[0] if events else None
        expected=f'{candidate}#event:{h.completed_event_count+1:016d}'
        first_rate=next((r for r in rates if r['step']>mark['step'] and r['candidate_id']==candidate),None)
        end=final.state.crack_network.branch(daughter.branch_id)
        actual=sum(t['realized_arm_lengths_m'][t['candidate_ids'].index(candidate)] for t in events)
        checks['shared_clock_inventory_'+daughter.branch_id]=fp(pair.front_competitions[daughter.branch_id])==fp(pair.state.competition)
        checks['geometry_count_'+daughter.branch_id]=close(length(end)-length(daughter),actual)
        checks['first_post_birth_event_'+daughter.branch_id]=(first is not None and expected in first['pending_event_ids'])
        checks['threshold_consumed_once_'+daughter.branch_id]=sum(expected in t['pending_event_ids'] for t in post)==1
        checks['retained_threshold_visible_'+daughter.branch_id]=(first_rate is not None and
            first_rate['current_threshold_H_star']==h.current_threshold_action and
            first_rate['directional_event_ordinal']==h.completed_event_count+1 and
            first_rate['tip_id']==daughter.branch_id)
        mapping.append({'role':'primary' if candidate==pid else 'companion','daughter_id':daughter.branch_id,
            'candidate_id':candidate,'clock_immediately_before_mark':asdict(h),
            'threshold_rng_identity':f'v11-directional-threshold|{h.threshold_seed}|{candidate}|{h.completed_event_count+1}',
            'daughter_clock_after_mark':asdict(next(v for v in pair.front_competitions[daughter.branch_id].hazard_states if v.candidate_id==candidate)),
            'first_post_birth_observation':first_rate,'first_post_birth_accepted_event':first,
            'initial_length_m':length(daughter),'final_length_m':length(end),'accepted_growth_sum_m':actual})
    post_ids=[eid for t in post for eid in t['pending_event_ids']]
    checks['final_release_ledger_no_duplicate_birth']=close(final.state.energy_ledgers['topology_release_J_per_m'],
        pair.state.energy_ledgers['topology_release_J_per_m']+sum(t['released_energy_J_per_m'] for t in post))
    checks['final_cost_ledger_no_duplicate_birth']=close(final.state.energy_ledgers['hazard_dissipation_J_per_m'],
        pair.state.energy_ledgers['hazard_dissipation_J_per_m']+sum(t['total_dissipative_cost_J_per_m'] for t in post))
    checks['all_post_ids_unique']=len(post_ids)==len(set(post_ids))
    checks['final_consumptions_exact']=Counter(final.state.competition.consumed_event_ids)==Counter(list(initial_ids)+post_ids)
    checks['primary_birth_one_canonical_transaction']=sum(t['accepted'] and mark['parent_event_id'] in t['pending_event_ids'] for t in original_trials)==1
    if not all(checks.values()):
        raise RuntimeError('bookkeeping proof failed: '+repr({k:v for k,v in checks.items() if not v}))
    return {'case':case,'outcome':'INTENDED_MARKED_EVENT_SEMANTICS','checks':checks,'daughters':mapping,
        'birth_mark':mark,'single_checkpoint_sha256':sha256(birth/'canonical_single.json.state.pkl'),
        'pair_checkpoint_sha256':sha256(birth/'accepted_pair.json.state.pkl'),
        'pair_release_J_per_m':pair_release,'pair_cost_J_per_m':pair_cost,
        'primary_cost_J_per_m':parent['total_dissipative_cost_J_per_m'],
        'companion_initial_cost_J_per_m':pair_cost-parent['total_dissipative_cost_J_per_m'],
        'additional_release_beyond_single_J_per_m':added_release,
        'initial_pair_owner':'one shared cluster with candidate-to-daughter scoping; not duplicate independent clock inventories',
        'interpretation':'parent event creates both initial arms; retained companion clock predicts the mark and later consumes its own event once for growth from 5 to 10 um; no companion event is claimed at birth'}


def opportunities(case):
    original=OUT/'short_ensemble'/case
    marked=rows(case_folder(case)/'v13_primary_race.jsonl')
    directional=rows(original/'directional_rates.jsonl')
    result=[]
    for number,mark in enumerate(marked,1):
        row=dict(mark,case=case,seed=int(case.rsplit('seed',1)[1]),accepted_event_number=number,
            primary_forward_reach_um=mark['parent_forward_extension_um'])
        p=mark.get('primary');c=mark.get('companion')
        ti=mark.get('T_i_next_s');tj=mark.get('T_j_s')
        row.update(T_primary_next_s=ti,T_companion_s=tj,
            T_companion_over_tau_c=tj/mark['tau_c_s'] if tj is not None else None,
            T_companion_over_T_primary_next=tj/ti if tj is not None and ti is not None and ti>0 else None,
            primary_time_status='ARCHIVED_FROZEN_PROJECTION' if ti is not None else 'NOT_EVALUATED_BY_EXECUTION_SHORT_CIRCUIT',
            pair_admissible=True if mark['outcome']=='PAIR_ACCEPTED' else None,
            pair_admissibility_status='ACCEPTED_PAIR_RECORDED' if mark['outcome']=='PAIR_ACCEPTED' else 'FULL_ADMISSIBILITY_NOT_RECORDED',
            pair_energy_margin_J_per_m=mark.get('pair_margin_J_per_m'),
            primary_raw_rate_per_s=mark.get('primary_raw_rate_per_s'),
            companion_raw_rate_per_s=mark.get('companion_raw_rate_per_s'),
            primary_effective_rate_per_s=p['effective_rate_per_s'] if p else None,
            companion_effective_rate_per_s=c['effective_rate_per_s'] if c else None,
            primary_drive_source=('valid_local_J' if mark.get('primary_local_J_valid') else 'same_plane_marginal_energy') if p else None,
            companion_drive_source='single_to_pair_marginal_energy' if c else None)
        clock_rows=[]
        for d in directional:
            if d['step']!=mark['step']:continue
            cid=d['candidate_id'];ordinal=d['directional_event_ordinal']
            clock_rows.append({'candidate_id':cid,'action':d['accumulated_integrated_hazard_H'],
                'cumulative_threshold':d['current_threshold_H_star'],'ordinal':ordinal,
                'threshold_rng_identity':f"v11-directional-threshold|{row['seed']}|{cid}|{ordinal}",
                'scope':'accepted_endpoint_clock; accompanying source rate belongs to pre-topology interval, not substituted as post-primary rate'})
        row['both_candidate_clocks']=clock_rows
        result.append(row)
    return result


def main():
    DEST.mkdir(exist_ok=True)
    protected=[]
    for case in CASES:
        for root in {case_folder(case),OUT/'short_ensemble'/case}:
            for name in ('terminal.json','exception.json','launch.json','v13_primary_race.jsonl',
                'branch_action_trials.jsonl','directional_rates.jsonl','checkpoint/latest.json','checkpoint/latest.json.state.pkl',
                'v13_branch/accepted_pair.json','v13_branch/accepted_pair.json.state.pkl','v13_branch/canonical_single.json','v13_branch/canonical_single.json.state.pkl'):
                p=root/name
                if p.exists():protected.append({'path':str(p),'sha256':sha256(p)})
    atomic_json(DEST/'input_hashes.json',{'files':protected})
    audits=[audit_pair(c) for c in ('Peak_300K_seed3621','Peak_1000K_seed3621')]
    atomic_json(DEST/'marked_event_bookkeeping.json',{'cases':audits,'mechanics_or_clock_replays':0,
        'outcome':'INTENDED_MARKED_EVENT_SEMANTICS','scope':'two completed Peak seed-3621 branches',
        'semantics_change_warranted':False})
    cases=[];events=[]
    for case in CASES:
        p=case_folder(case)/'terminal.json'
        terminal=json.loads(p.read_text()) if p.exists() else {'status':'NOT_LAUNCHED','reason':'queue_paused_on_state_owner_error'}
        marks=opportunities(case);events+=marks
        branch=next((m for m in marks if m['outcome']=='PAIR_ACCEPTED'),None)
        cases.append(dict(terminal,case=case,seed=int(case.rsplit('seed',1)[1]),
            first_branch_event_number=branch['accepted_event_number'] if branch else None,
            single_opportunities_before_first_branch=branch['accepted_event_number']-1 if branch else len(marks),
            first_branch_primary_forward_um=branch['primary_forward_reach_um'] if branch else None))
    contrasts=[]
    for seed in (3621,3622,3623,3624):
        paired=[c for c in cases if c['seed']==seed]
        for a,b in itertools.combinations(paired,2):
            if a['first_branch_event_number'] is not None and b['first_branch_event_number'] is not None:
                contrasts.append({'seed':seed,'a':a['case'],'b':b['case'],
                    'delta_first_branch_event_number_b_minus_a':b['first_branch_event_number']-a['first_branch_event_number'],
                    'delta_first_branch_primary_forward_um_b_minus_a':b['first_branch_primary_forward_um']-a['first_branch_primary_forward_um'],
                    'scope':'observed first births; postbranch stop status separately retained'})
    atomic_json(DEST/'case_table.json',{'rows':cases,'expected':16,'scientific_terminal_count':sum(c['status']=='TERMINATED' for c in cases),
        'final_classification':None,'classification_withheld_reason':'state-owner software error pauses queue; two unlaunched cases are not physical nonbranching results'})
    atomic_json(DEST/'prebranch_opportunity_table.json',{'rows':events,'missing_primary_time_rows':sum(e['T_primary_next_s'] is None for e in events),
        'missing_pair_margin_rows':sum(e.get('pair_margin_J_per_m') is None for e in events),
        'scope':'archive export only; absent frozen mechanics/primary times are explicit nulls, never inferred from pre-event rates'})
    atomic_json(DEST/'paired_seed_contrasts.json',{'rows':contrasts,'calibrated_probability_claimed':False})
    assert all(sha256(Path(r['path']))==r['sha256'] for r in protected)
    atomic_json(DEST/'verification.json',{'input_artifacts_unchanged':len(protected),'all_peak_bookkeeping_checks_passed':True,
        'mechanics_solves':0,'clock_updates':0,'RNG_draws':0,'production_source_changed':False})
    print('Audit passed; cases',len(cases),'opportunities',len(events))


if __name__=='__main__':main()
