"""Read-only transition gate; diagnostic predictions are not observed branches."""
import argparse
import json
import math
from pathlib import Path

from scripts.run_v13_transition_screen import OUT,PLAN,BOUNDARY,verify_frozen
from scripts.run_pf_current_source_multifront_field_atlas_v12 import atomic_json,sha256


def summarize(plan_path=PLAN):
    plan=json.loads(plan_path.read_text());cases=[];events=[];inputs={}
    for case in plan['screening_cases']:
        folder=OUT/'parents'/case
        terminal=folder/'terminal.json'
        if not terminal.exists():raise RuntimeError('screen incomplete: '+case)
        t=json.loads(terminal.read_text())
        if t['status']=='SOFTWARE_OR_UNCLASSIFIED_STOP':raise RuntimeError('unclassified screen stop: '+case)
        paths=sorted((folder/'opportunities').glob('*/race.json'))
        r=[json.loads(p.read_text()) for p in paths]
        events.extend(dict(e,case=case,inside_common_75um_interval=e['parent_forward_extension_um']<=75.) for e in r)
        for p in paths+[terminal,folder/'launch.json']:
            inputs[str(p)]=sha256(p)
        positive=next((e for e in r if e['predicted_branch'] and e['parent_forward_extension_um']<=75.),None)
        last=json.loads((folder/'checkpoint/latest.json').read_text())
        extension=last['projected_extension_m']*1e6
        complete_negative=positive is None and extension>=75.-1e-9
        cases.append(dict(case=case,condition='_'.join(case.split('_')[:2]),
            group='_'.join(case.split('_')[2:4]),seed=int(case.rsplit('seed',1)[1]),
            predicted_first_branch=positive is not None,
            right_censored_at_75_um=complete_negative,
            early_gate=positive is None and not complete_negative,
            screening_endpoint_um=extension,
            first_predicted_branch_extension_um=positive['parent_forward_extension_um'] if positive else None,
            first_predicted_branch_event_number=positive['accepted_event_count'] if positive else None,
            terminal=t,opportunities=len(r),observed_committed_branches=0,
            scope='BRANCH_DISABLED_COUNTERFACTUAL_FIRST_BRANCH_SCREEN'))
    conditions=[]
    for condition in sorted({c['condition'] for c in cases}):
        group=[c for c in cases if c['condition']==condition]
        rows=[e for e in events if e['case'].startswith(condition+'_') and e['inside_common_75um_interval']]
        chi=[e['chi_B'] for e in rows if e.get('chi_B') is not None]
        nonsat=[e for e in rows if e.get('primary_effective_lambda_tau_c') is not None and
                e.get('companion_effective_lambda_tau_c') is not None and
                min(e['primary_effective_lambda_tau_c'],e['companion_effective_lambda_tau_c'])<.99]
        positives=sum(c['predicted_first_branch'] for c in group)
        negatives=sum(c['right_censored_at_75_um'] for c in group)
        pair_positive=all(e.get('exact_pair_admissible') is True and e.get('pair_margin_J_per_m',-1)>0
                          for e in rows if e['predicted_branch'])
        margins=[dict(case=e['case'],event=e['accepted_event_count'],margin_J_per_m=e.get('pair_margin_J_per_m'),
            margin_over_release=e['pair_margin_J_per_m']/e['pair_energy_release_J_per_m'] if e.get('pair_energy_release_J_per_m',0)>0 else None,
            margin_over_cost=e['pair_margin_J_per_m']/e['pair_dissipative_cost_J_per_m'] if e.get('pair_dissipative_cost_J_per_m',0)>0 else None)
            for e in rows if e.get('exact_pair_admissible') is True]
        gate=len(group)==4 and positives>0 and negatives>0 and any(x<1 for x in chi) and any(x>1 for x in chi) and bool(nonsat) and pair_positive
        conditions.append(dict(condition=condition,predicted_positive_parents=positives,
            unbranched_75um_screen_controls=negatives,early_gates=4-positives-negatives,
            finite_chi_range=[min(chi),max(chi)] if chi else None,
            nonsaturated_opportunity_count=len(nonsat),pair_mechanics_for_predictions_admissible=pair_positive,
            pair_boundary_distances=margins,useful_transition_gate=gate))
    useful=[c for c in conditions if c['useful_transition_gate']]
    output=OUT/('orientation_screen_results' if plan_path==PLAN else plan_path.stem+'_results')
    output.mkdir()
    atomic_json(output/'case_table.json',dict(rows=cases))
    atomic_json(output/'opportunity_table.json',dict(rows=events,
        original_ensemble_nulls_untouched=True,
        missing_physical_observations='Explicit unavailable statuses; never replaced with invented rates or tensors.'))
    atomic_json(output/'condition_gate.json',dict(conditions=conditions,useful_conditions=[c['condition'] for c in useful],
        followup_ensemble_authorized_by_gate=bool(useful),branch_incidence_measured=False,boundary=BOUNDARY))
    atomic_json(output/'provenance.json',dict(files=inputs,plan_sha256=sha256(plan_path),
        immutable_accepted_ensemble_files=verify_frozen()))
    lines=['# V13 transition-regime screen','',f'`{BOUNDARY}`','',
        'This is a branch-disabled counterfactual screen. A positive row is the first exact frozen state predicting an admissible mark, not an actually committed branch. No observed incidence probability is estimated.','',
        '| Parent | First predicted event | Extension (µm) | 75 µm censored control | Terminal |',
        '|---|---:|---:|---|---|']
    for c in cases:
        lines.append(f"| {c['case']} | {c['first_predicted_branch_event_number']} | {c['screening_endpoint_um']:.6g} | {c['right_censored_at_75_um']} | {c['terminal']['reason']} |")
    lines+=['','## Gate','',
        'Useful conditions: '+(', '.join(c['condition'] for c in useful) if useful else 'none in this screen')+'.','',
        'The gate requires a positive prediction and a completed 75 µm negative parent across the four groups, chi on both sides of one, a nonsaturated channel, and positive exact pair margin for positive predictions. Early legitimate gates are neither 75 µm negatives nor evidence of a transition.','',
        'The fixed 5 µm physical event is not shortened to hit a reporting boundary. An accepted endpoint may overshoot 75 µm; its frozen diagnostic is preserved but excluded from the common-window transition gate. The common censor is 75 µm in accepted maximum-forward-reach coordinates, not branch-junction position.','',
        'A positive pair margin is necessary. Its distance from the boundary is reported as margin/release and margin/cost rather than silently treating a large positive margin as a near-boundary observation.','',
        'Seed 3621 is paired across the four material–temperature groups at a fixed orientation. Candidate identities depend on orientation; the same numerical seed does not imply identical threshold streams across different orientations. This screen locates candidate conditions and is not an isolated causal estimate of orientation effects.','']
    (output/'TRANSITION_SCREEN_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(conditions,indent=2))
    return conditions


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,default=PLAN)
    summarize(p.parse_args().plan)
