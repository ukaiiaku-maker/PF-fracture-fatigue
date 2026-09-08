"""Final observed incidence and censored first-branch outcomes; no model fitting."""
from collections import Counter
from copy import deepcopy
import itertools
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from scripts.run_v13_transition_screen import OUT,BOUNDARY,GROUPS,verify_frozen
from scripts.run_v13_transition_ensemble import DEST,ENSEMBLE_PLAN
from scripts.run_pf_current_source_multifront_field_atlas_v12 import atomic_json,sha256
from scripts.v13_value_fingerprint import physical_state_fingerprint as fp

FINAL=OUT/'final_transition_record'


def process_parity(a,b):
    """Keep raw hashes; compare physical values excluding one observer counter.

    Complete screen diagnostics perform additional isolated mechanics calls.
    OBSERVER.mechanics_serial counts those calls, not accepted physical states.
    Within-run serial/freshness checks remain untouched; this comparison only
    excludes the absolute counter across two separately executed processes.
    No other field is excluded and neither input is modified.
    """
    values=[];serials=[]
    for original in (a,b):
        value=deepcopy(original)
        drive=value['engine_fields']['_anisotropic_drive']
        serials.append(drive.pop('mechanics_serial'))
        values.append(value)
    return dict(raw_hashes=[fp(a),fp(b)],raw_equal=fp(a)==fp(b),
        excluded_observer_field='engine_fields/_anisotropic_drive/mechanics_serial',
        observer_serials=serials,physical_values_equal=fp(values[0])==fp(values[1]))


def km(rows):
    """Descriptive product-limit curve, with ties handled event-before-censor."""
    risk=len(rows);survival=1.;result=[dict(extension_um=0.,at_risk=risk,events=0,censored=0,survival=1.)]
    for x in sorted({r['survival_extension_um'] for r in rows}):
        events=sum(r['branch_observed'] and r['survival_extension_um']==x for r in rows)
        censored=sum(not r['branch_observed'] and r['survival_extension_um']==x for r in rows)
        survival*=1-events/risk
        result.append(dict(extension_um=x,at_risk=risk,events=events,censored=censored,survival=survival))
        risk-=events+censored
    assert risk==0
    return result


def has_observed_mixed_incidence(rows):
    return any(r['branch_observed'] for r in rows) and any(r['unbranched_75um_complete'] for r in rows)


def main():
    plan=json.loads(ENSEMBLE_PLAN.read_text());cases=[];networks={};inputs={};marks=[]
    producer=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    bundle=OUT/'transition_source_increment.bundle'
    bundle_heads=subprocess.check_output(['git','bundle','list-heads',str(bundle)],text=True)
    assert producer in {line.split()[0] for line in bundle_heads.splitlines()}
    inputs[str(bundle)]=sha256(bundle)
    for case in plan['cases']:
        folder=DEST/case
        t=json.loads((folder/'terminal.json').read_text())
        launch=json.loads((folder/'launch.json').read_text())
        assert t['status'] in ('TERMINATED','EXISTING_GATE_STOP'), 'unfinished/unclassified case: '+case
        cp=restore_branch_checkpoint(folder/'checkpoint/latest.json')
        records=[json.loads(s) for s in (folder/'v13_primary_race.jsonl').read_text().splitlines()] if (folder/'v13_primary_race.jsonl').exists() else []
        births=[(i+1,r) for i,r in enumerate(records) if r['outcome']=='PAIR_ACCEPTED']
        assert len(births)<=1
        event,birth=births[0] if births else (None,None)
        assert all(b.generation<=1 for b in cp.state.crack_network.branches), 'second branch is outside scope'
        assert any(b.generation>0 for b in cp.state.crack_network.branches)==(birth is not None), 'mark/checkpoint topology mismatch'
        assert len(births)==t['branch_count']
        if birth:assert birth['parent_forward_extension_um']<=75., 'branch committed beyond common window'
        censored=t['reason']=='75_um_without_committed_branch'
        assert not (birth is not None and censored)
        theta,rate,material,temp,seed=case.split('_');group=material+'_'+temp
        actual=cp.projected_extension_m*1e6
        if censored:assert actual>=75., 'incomplete trajectory cannot be a completed 75-um negative'
        x=birth['parent_forward_extension_um'] if birth else 75. if censored else min(actual,75.)
        cases.append(dict(case=case,group=group,seed=int(seed[4:]),terminal=t,branch_observed=birth is not None,
            execution_source_commit=launch['source_commit'],family_sha256=launch['family_sha256'],
            material_row_sha256=launch['material_row_sha256'],
            first_branch_event_number=event,first_branch_primary_reach_um=birth['parent_forward_extension_um'] if birth else None,
            first_branch_junction_forward_um=(birth['branch_junction_xy_m'][0]-.0005)*1e6 if birth else None,
            branch_time_s=birth['accepted_time_s'] if birth else None,branch_opening_m=birth['accepted_opening_m'] if birth else None,
            accepted_terminal_reach_um=actual,unbranched_75um_complete=censored,
            early_gate_censor=birth is None and not censored,survival_extension_um=x,
            first_branch_clock_record=birth,accepted_steps=cp.state.event_counters['accepted_steps'],
            final_accepted_cleavage_count=len(cp.state.competition.consumed_event_ids)))
        networks[case]=cp.state.crack_network.to_dict()
        marks.extend(dict(r,case=case,accepted_cleavage_opportunity=i+1) for i,r in enumerate(records))
        for name in ('terminal.json','launch.json','v13_primary_race.jsonl','checkpoint/latest.json','checkpoint/latest.json.state.pkl',
                     'branch_action_trials.jsonl','directional_rates.jsonl','final_accepted_fields.npz'):
            p=folder/name
            if p.exists():inputs[str(p)]=sha256(p)
    assert len(cases)==32
    source_paths=('arrhenius_fracture/primary_race_production_v13.py',
        'arrhenius_fracture/sharp_front_v11_branching.py',
        'arrhenius_fracture/inherited_primary_race_v13.py',
        'scripts/run_v13_transition_ensemble.py')
    execution_sources={}
    for commit in sorted({r['execution_source_commit'] for r in cases}):
        execution_sources[commit]={}
        for path in source_paths:
            archived=subprocess.check_output(['git','show',commit+':'+path])
            assert archived==Path(path).read_bytes(), 'physical execution source changed: '+path
            execution_sources[commit][path]=hashlib.sha256(archived).hexdigest()
    summaries=[];curves={}
    for group in GROUPS:
        rows=[r for r in cases if r['group']==group]
        branches=sum(r['branch_observed'] for r in rows)
        negative=sum(r['unbranched_75um_complete'] for r in rows)
        early=sum(r['early_gate_censor'] for r in rows)
        assert len(rows)==8 and branches+negative+early==8
        curves[group]=km(rows)
        events=[r['first_branch_event_number'] for r in rows if r['branch_observed']]
        reaches=[r['first_branch_primary_reach_um'] for r in rows if r['branch_observed']]
        summaries.append(dict(group=group,seeds=8,observed_branches=branches,
            completed_75um_nonbranching=negative,early_gate_censored=early,
            branch_fraction_observed=branches/8,branch_fraction_bounds_if_early_gates=[branches/8,(branches+early)/8],
            observed_first_branch_event_range=[min(events),max(events)] if events else None,
            observed_first_branch_reach_range_um=[min(reaches),max(reaches)] if reaches else None,
            inferential_probability_calibration=False))
    paired=[]
    for a,b in itertools.combinations(GROUPS,2):
        cells=Counter()
        for seed in plan['seeds']:
            x=next(r for r in cases if r['group']==a and r['seed']==seed)
            y=next(r for r in cases if r['group']==b and r['seed']==seed)
            label='unknown_due_to_early_censor' if x['early_gate_censor'] or y['early_gate_censor'] else (
                'both_branch' if x['branch_observed'] and y['branch_observed'] else
                'a_only' if x['branch_observed'] else 'b_only' if y['branch_observed'] else 'neither')
            cells[label]+=1
        paired.append(dict(a=a,b=b,paired_counts=dict(cells)))
    # First seed is a prospectively matched screen/production source control.
    prefix=[]
    for group in GROUPS:
        case=f"{plan['condition']}_{group}_seed3621"
        row=next(r for r in cases if r['case']==case)
        screening=OUT/'parents'/case
        races=[json.loads(p.read_text()) for p in sorted((screening/'opportunities').glob('*/race.json'))]
        expected=next((r for r in races if r['predicted_branch'] and r['parent_forward_extension_um']<=75.),None)
        assert row['branch_observed']==(expected is not None)
        if expected:
            actual=row['first_branch_clock_record']
            checks={k:actual[k]==expected[k] for k in ('parent_event_id','parent_forward_extension_um',
                'accepted_time_s','accepted_opening_m','parent_competition_sha256','parent_engine_rng_sha256')}
            a=restore_branch_checkpoint(screening/'checkpoint/latest.json')
            b=restore_branch_checkpoint(DEST/case/'v13_branch/canonical_single.json')
        else:
            a=restore_branch_checkpoint(screening/'checkpoint/latest.json')
            b=restore_branch_checkpoint(DEST/case/'checkpoint/latest.json')
            checks={k:fp(getattr(a.state,k))==fp(getattr(b.state,k)) for k in ('crack_network','competition','rng_state','displacement','damage','ep_gp','rho_gp')}
        process=process_parity(a.shared_process_state,b.shared_process_state)
        checks['shared_process_physical_values']=process['physical_values_equal']
        for key in ('crack_network','competition','rng_state','displacement','damage','ep_gp','rho_gp'):
            checks[key]=fp(getattr(a.state,key))==fp(getattr(b.state,key))
        assert all(checks.values()), (case,checks)
        prefix.append(dict(case=case,checks=checks,process_fingerprint_comparison=process))
    count=sum(c['branch_observed'] for c in cases)
    additional=[c for c in cases if c['seed']!=3621]
    mixed=has_observed_mixed_incidence(cases)
    decision='MIXED_FIRST_BRANCH_INCIDENCE_DEMONSTRATED_IN_TESTED_MODEL' if mixed else 'FOLLOWUP_DID_NOT_DEMONSTRATE_MIXED_INCIDENCE'
    FINAL.mkdir()
    atomic_json(FINAL/'summary.json',dict(status='COMPLETE',condition=plan['condition'],cases=32,
        observed_branches=count,decision=decision,group_summaries=summaries,boundary=BOUNDARY,
        additional_seed_cases=len(additional),additional_seed_observed_branches=sum(c['branch_observed'] for c in additional),
        additional_seed_completed_nonbranching=sum(c['unbranched_75um_complete'] for c in additional),
        material_parameters_changed=False,calibrated_probability=False,recursive_spacing_validated=False,
        archive_theta40_unchanged_files=verify_frozen(),report_producer_code_commit=producer))
    for name,payload in (('case_table.json',dict(rows=cases)),('survival_curves.json',dict(groups=curves,
        interpretation='Descriptive product-limit curves; early numerical/model gates may be informative censoring. No confidence intervals or calibrated probability claims.')),
        ('paired_seed_contrasts.json',dict(rows=paired)),('first_seed_screen_production_parity.json',dict(rows=prefix)),
        ('final_topologies.json',networks),('native_mark_opportunities.json',dict(rows=marks,
            unavailable_fields='Production early-exit fields remain unevaluated; the separate orientation screen has full diagnostics for admissible opportunities.')),
        ('provenance.json',dict(files=inputs,plan_sha256=sha256(ENSEMBLE_PLAN),
            execution_source_files=execution_sources))):
        atomic_json(FINAL/name,payload)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    colors=dict(zip(GROUPS,('tab:blue','tab:orange','tab:green','tab:red')))
    for index,(g,curve) in enumerate(curves.items()):
        offset=(index-1.5)*.12
        axes[0].step([r['extension_um'] for r in curve],[r['survival'] for r in curve],where='post',label=g,color=colors[g])
        c=[r for r in cases if r['group']==g]
        axes[1].scatter([r['seed']+offset for r in c if r['branch_observed']],
            [r['survival_extension_um'] for r in c if r['branch_observed']],label=g,color=colors[g])
        axes[1].scatter([r['seed']+offset for r in c if not r['branch_observed']],
            [r['survival_extension_um'] for r in c if not r['branch_observed']],marker='>',facecolors='none',edgecolors=colors[g])
    axes[0].set(xlim=(0,75),ylim=(0,1.05),xlabel='Maximum forward reach (µm)',ylabel='Fraction without first branch (product-limit)')
    axes[1].set(xlabel='Seed (groups offset for visibility)',xticks=plan['seeds'],ylabel='First branch / censor reach (µm)',ylim=(0,80))
    axes[0].legend(fontsize=8);axes[1].set_title('Open triangles: censored, not branch events')
    fig.suptitle('V13 θ=15° canonical rate — uncalibrated bounded model')
    fig.savefig(FINAL/'incidence_and_first_branch_survival.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(4,8,figsize=(20,10),sharex=True,sharey=True,constrained_layout=True)
    for c in cases:
        ax=axes[GROUPS.index(c['group']),c['seed']-3621]
        for b in networks[c['case']]['branches']:
            ax.plot([(x-.0005)*1e6 for x,y in b['path_m']],[y*1e6 for x,y in b['path_m']],
                color='black' if b['generation']==0 else 'tab:blue',lw=1.2)
        ax.set_title(c['group']+' / '+str(c['seed'])+'\n'+('branch' if c['branch_observed'] else 'censored'),fontsize=7)
        ax.set(xlim=(-2,100),ylim=(-45,30));ax.set_aspect('equal')
    fig.supxlabel('Forward coordinate from initial tip (µm)')
    fig.supylabel('Transverse coordinate (µm)')
    fig.suptitle('Final accepted topologies — black: parent; blue: daughters\nBRANCHING_KINETICS_MODEL_UNCALIBRATED')
    fig.savefig(FINAL/'final_32_case_topologies.png',dpi=160);plt.close(fig)
    lines=['# V13 transition-regime screen and eight-seed followup','',f'**{BOUNDARY}**','',
        f'Observed result: **{decision}**. {count}/32 cases formed a first branch within the prespecified common observation interval. These are outcomes of the tested uncalibrated model, not experimentally calibrated material probabilities.','',
        '## Screening and scope','',
        'Eight fresh branch-disabled parents screened 15° and 30° at the canonical loading rate; twelve qualified θ=40° frozen controls were reused, with no θ=40° trajectory rerun. At 15°, Peak at both temperatures predicted branches near 10.95 µm while both Weak-T controls remained negative within 75 µm. Weak-T/1000 K first predicted a branch at the overshooting 77.27 µm endpoint, explicitly outside the common interval. Weak-T/300 K remained negative even there. All four 30° parents predicted branches.','',
        'The 15° screen had chi on both sides of one and 26 nonsaturated opportunities. Exact pairs were admissible but far from energy veto (margin/release approximately 0.984–1.000): this selects a kinetic transition, not an energy-boundary transition. Loading-rate variation and new branch-model parameters were unnecessary and were not used.','',
        '## Observed incidence by group','',
        '| Group | Branches / 8 | Completed 75 µm negatives | Early-gate censored | Observed first-branch event range |','|---|---:|---:|---:|---|']
    for g in summaries:lines.append(f"| {g['group']} | {g['observed_branches']} | {g['completed_75um_nonbranching']} | {g['early_gate_censored']} | {g['observed_first_branch_event_range']} |")
    lines+=['','## Interpretation','',
        'The eight common-random-number seeds are paired across groups at fixed orientation. Identical numeric seeds do not establish identical clock streams across orientations because candidate identifiers change; no cross-orientation paired-stream claim is made. The finite-sample incidence contrasts and censored first-branch curves describe these simulations only. Early gates are separately identified and are not counted as completed 75 µm negatives. Their censoring may be informative. No significance test, calibrated probability, or independent-trial binomial interval is assigned.','',
        'Seed 3621 was used to select the condition and serves as a screen/production replication control, not independent holdout evidence. Seeds 3622–3628 supply seven additional paired realizations. The eight-seed table is descriptive and does not remove condition-selection bias.','',
        f"Excluding the selection seed, the additional 28 cases contain {sum(c['branch_observed'] for c in additional)} observed branches, {sum(c['unbranched_75um_complete'] for c in additional)} completed 75 µm nonbranching observations, and {sum(c['early_gate_censor'] for c in additional)} early-gate censors.",'',
        'Branch position remains distinct from primary maximum reach at acceptance. The survival coordinate is the latter. The last discrete 5 µm event can overshoot the 75 µm reporting threshold; an unbranched run is censored at 75 µm without shortening or changing that event. No mark beyond the cutoff is committed. After an earlier first branch, the existing 20 µm daughter-growth stop or an existing legitimate gate applies.','',
        'The first-seed prospective screen/production controls match in physical process values, clocks, endpoint and geometry checks. Raw process hashes differ because complete screen diagnostics increment the mechanics-call observer serial. Both raw hashes and serials are retained; only engine_fields/_anisotropic_drive/mechanics_serial is excluded from the separate physical-value comparison. No tensor, kinetic state or RNG field is excluded. There is no new memory term, precursor lifetime, barrier, correlation time, material row, emission law, or marked-event bookkeeping change. Branch angles remain crystallographic; the morphology mark is still zero-global-time, and no recursive branch spacing or resolved two-embryo time history is validated.','',
        'The accepted θ=40° ensemble and its original nulls remain immutable. Its correct conclusion remains seed-dependent first-branch onset with saturated bounded incidence; the present condition is a separate domain, not a reinterpretation of that ensemble.','',
        'Excluding the absolute mechanics-call serial from a cross-run value comparison does not disable any within-run serial/freshness check. Those checks remain unchanged in production.','',
        '## Provenance and verification','',
        'The case table separates execution-source commits from the report-producer commit in summary.json. Raw terminal, checkpoint, mechanics, mark-ledger and final-field hashes are in provenance.json. The compact archive contains selected reports, tables and an incremental source Git bundle requiring accepted base f09ec8468d4d95df73385742611f3de44455f865. Full native trajectories, accepted checkpoints and portable final fields remain in the adjacent transition_ensemble directory on the Data drive. Screening source, qualified-family manifests and the immutable historical-ensemble seal are preserved in this branch.','',
        'Twenty preflight/default-off production fixture checks and ten reporting/launch-gate checks passed. The four seed-3621 prospective screen/production controls passed physical-value parity; observer-counter differences are explicitly retained. These checks qualify this bounded comparison, not the branch model against experiment.','',
        '![Observed first-branch outcomes and censoring](incidence_and_first_branch_survival.png)','',
        '![Final accepted topologies](final_32_case_topologies.png)','']
    (FINAL/'V13_TRANSITION_REGIME_FINAL_REPORT.md').write_text('\n'.join(lines))
    assert all(sha256(Path(p))==h for p,h in inputs.items())
    atomic_json(FINAL/'verification.json',dict(status='PASS',case_count=32,first_seed_parity=True,
        input_files_verified=len(inputs),frozen_ensemble_files_verified=verify_frozen()))
    entries={f'final/{p.name}':p for p in FINAL.iterdir() if p.is_file()}
    for p in (OUT/'orientation_screen_results').iterdir():
        if p.is_file():entries['screen/'+p.name]=p
    for name in ('ACCEPTED_ENSEMBLE_AND_PROVENANCE_ADDENDUM.md','accepted_ensemble_freeze.json',
                 'reusable_state_inventory.json','screen_plan.json','transition_ensemble_plan.json',
                 'preflight_tests.xml','reporting_and_gate_tests.xml','transition_source_increment.bundle'):
        entries['provenance/'+name]=OUT/name
    with zipfile.ZipFile(OUT/'V13_TRANSITION_REVIEW.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,p in sorted(entries.items()):archive.write(p,name)
        archive.writestr('SHA256_MANIFEST.json',json.dumps({k:sha256(p) for k,p in entries.items()},indent=2,sort_keys=True)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
