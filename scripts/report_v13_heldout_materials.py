"""Eight-group descriptive comparison from immutable native records; no solves."""
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from scripts.run_v13_heldout_materials import OUT,DEST,PLAN,ACCEPTED,PHYSICS,RECORD,verify_source,verify_accepted
from scripts.run_pf_current_source_multifront_field_atlas_v12 import atomic_json,sha256
from scripts.report_v13_transition_ensemble import km

FINAL=OUT/'final_four_material_record'
GROUPS=('Peak_300K','Peak_1000K','weakT_300K','weakT_1000K',
    'DBTT_300K','DBTT_1000K','ceramic_300K','ceramic_1000K')


def distance_summary(rows,horizon=75.):
    curve=km(rows)
    supported=curve[-1]['extension_um']>=horizon or curve[-1]['survival']==0.
    area=0.;last=0.;survival=1.
    for point in curve[1:]:
        x=min(point['extension_um'],horizon)
        area+=(x-last)*survival;last=x
        if point['extension_um']>horizon:break
        survival=point['survival']
    area+=max(0.,horizon-last)*survival
    median=next((r['extension_um'] for r in curve if r['survival']<=.5 and r['extension_um']<=horizon),None)
    return dict(product_limit=curve,median_first_branch_reach_um=median,
        median_definition='first reach with product-limit survival <= 0.5; not a median conditional on branching',
        restricted_mean_branch_free_reach_um=area if supported else None,
        restricted_mean_supported_to_75um=supported,
        support_end_um=curve[-1]['extension_um'],horizon_um=horizon,
        informative_early_censoring_possible=any(r['early_gate_censor'] for r in rows))


def mark_metrics(record):
    tau=record['tau_c_s'];p=record.get('primary');c=record.get('companion')
    si=p['effective_rate_per_s']*tau if p else None
    sj=c['effective_rate_per_s']*tau if c else None
    pending=c.get('pending_event_id') is not None if c else None
    if pending:assert record['T_j_s']==0.
    ti=record.get('T_i_next_s');tj=record.get('T_j_s');chi=None;chi_status='NOT_EVALUATED'
    if ti is not None and tj is not None:
        window=min(ti,tau)
        if window>0 and math.isfinite(window):
            if math.isfinite(tj):chi=tj/window;chi_status='FINITE'
            else:chi_status='INFINITE_COMPANION_WAIT'
        else:chi_status='NONPOSITIVE_OR_INVALID_PRIMARY_WINDOW'
    return dict(primary_effective_lambda_tau_c=si,companion_effective_lambda_tau_c=sj,
        primary_raw_lambda_tau_c=record['primary_raw_rate_per_s']*tau if 'primary_raw_rate_per_s' in record else None,
        companion_raw_lambda_tau_c=record['companion_raw_rate_per_s']*tau if 'companion_raw_rate_per_s' in record else None,
        primary_near_saturation=si>=.99 if si is not None else None,
        companion_near_saturation=sj>=.99 if sj is not None else None,
        completed_pending_companion=pending,exact_pair_margin_J_per_m=record.get('pair_margin_J_per_m'),
        chi_B=chi,chi_B_status=chi_status)


def audit_accepted():
    rows=json.loads((ACCEPTED/'final_transition_record/case_table.json').read_text())['rows']
    expected={'Peak_300K':(10.95,13.25),'Peak_1000K':(9.66,12.12),
        'weakT_300K':(62.79,58.03),'weakT_1000K':(43.47,48.49)}
    groups={}
    for g,(median,mean) in expected.items():
        d=distance_summary([r for r in rows if r['group']==g]);groups[g]=d
        assert round(d['median_first_branch_reach_um'],2)==median
        assert round(d['restricted_mean_branch_free_reach_um'],2)==mean
    births=[mark_metrics(r['first_branch_clock_record']) for r in rows if r['branch_observed']]
    assert len(births)==26 and sum(r['primary_near_saturation'] for r in births)==26
    assert sum(r['completed_pending_companion'] for r in births)==5
    return dict(status='PASS',groups=groups,accepted_births=26,primary_saturated_births=26,
        pending_companion_births=5,source_record_commit=RECORD,source_case_table_sha256=sha256(ACCEPTED/'final_transition_record/case_table.json'))


def read_new_case(case):
    folder=DEST/case;t=json.loads((folder/'terminal.json').read_text());launch=json.loads((folder/'launch.json').read_text())
    assert t['status'] in ('TERMINATED','EXISTING_GATE_STOP'), 'unfinished/unclassified '+case
    manifest=folder/'checkpoint/latest.json'
    assert sha256(manifest)==t['last_checkpoint_manifest_sha256'], 'terminal checkpoint changed '+case
    assert json.loads(manifest.read_text())['state_sha256']==t['last_checkpoint_state_sha256']
    assert sha256(folder/'final_accepted_fields.npz')==t['final_fields_sha256'], 'terminal fields changed '+case
    cp=restore_branch_checkpoint(folder/'checkpoint/latest.json')
    ledger=folder/'v13_primary_race.jsonl'
    records=list(map(json.loads,ledger.read_text().splitlines())) if ledger.exists() else []
    births=[(i+1,r) for i,r in enumerate(records) if r['outcome']=='PAIR_ACCEPTED']
    assert len(births)<=1 and len(births)==t['branch_count']
    event,birth=births[0] if births else (None,None)
    assert any(b.generation>0 for b in cp.state.crack_network.branches)==bool(birth)
    assert all(b.generation<=1 for b in cp.state.crack_network.branches)
    censored=t['reason']=='75_um_without_committed_branch';reach=cp.projected_extension_m*1e6
    assert not (censored and birth)
    if birth:assert birth['parent_forward_extension_um']<=75.
    if censored:assert reach>=75.
    theta,rate,material,temp,seed=case.split('_')
    row=dict(case=case,group=material+'_'+temp,seed=int(seed[4:]),terminal=t,branch_observed=bool(birth),
        first_branch_event_number=event,first_branch_primary_reach_um=birth['parent_forward_extension_um'] if birth else None,
        first_branch_junction_forward_um=(birth['branch_junction_xy_m'][0]-.0005)*1e6 if birth else None,
        branch_time_s=birth['accepted_time_s'] if birth else None,branch_opening_m=birth['accepted_opening_m'] if birth else None,
        accepted_terminal_reach_um=reach,unbranched_75um_complete=censored,early_gate_censor=not birth and not censored,
        survival_extension_um=birth['parent_forward_extension_um'] if birth else 75. if censored else min(reach,75.),
        first_branch_clock_record=birth,execution_source_commit=launch['source_commit'],frozen_physics_commit=PHYSICS,
        family_sha256=launch['family_sha256'],material_row_sha256=launch['material_row_sha256'],
        accepted_steps=cp.state.event_counters['accepted_steps'],held_out_material_class=True)
    return row,cp.state.crack_network.to_dict(),records


def suite_result():
    result=json.loads((OUT/'full_suite_result.json').read_text())
    tree=ET.parse(OUT/'full_suite.xml')
    nodes=list(tree.getroot().iter('testcase'))
    failures=[]
    inventory=[]
    for case in nodes:
        status='error' if case.find('error') is not None else 'failed' if case.find('failure') is not None else 'skipped' if case.find('skipped') is not None else 'passed'
        inventory.append(dict(test=case.attrib,status=status,details=[dict(kind=c.tag,attributes=c.attrib,text=c.text) for c in case]))
        for tag in ('failure','error'):
            item=case.find(tag)
            if item is not None:failures.append(dict(test=case.attrib,kind=tag,message=item.get('message'),detail=item.text))
    assert result['returncode']==0 or failures or result.get('collection_or_environment_failure')
    return dict(result,recorded_testcases=len(nodes),failures=failures,test_inventory=inventory,
        status_counts=dict(Counter(r['status'] for r in inventory)),
        skipped=sum(c.find('skipped') is not None for c in nodes),
        merge_gate='NOT_CLEARED_PENDING_FINAL_PROVENANCE_REVIEW' if result['returncode']==0 else 'NOT_CLEARED_FULL_SUITE_FAILURES')


def main():
    p=verify_source();freeze=verify_accepted();audit=audit_accepted();suite=suite_result()
    producer=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    bundle=OUT/'heldout_source_increment.bundle'
    heads=subprocess.check_output(['git','bundle','list-heads',str(bundle)],text=True)
    assert producer in {line.split()[0] for line in heads.splitlines()}
    q=json.loads((DEST/'queue_status.json').read_text());assert not q['active'] and not q['pending'] and not q['paused']
    old=ACCEPTED/'final_transition_record'
    cases=json.loads((old/'case_table.json').read_text())['rows']
    for r in cases:r['held_out_material_class']=False;r['frozen_physics_commit']=PHYSICS
    networks=json.loads((old/'final_topologies.json').read_text());marks=json.loads((old/'native_mark_opportunities.json').read_text())['rows']
    for c in p['cases']:
        row,network,records=read_new_case(c);cases.append(row);networks[c]=network
        marks.extend(dict(r,case=c,accepted_cleavage_opportunity=i+1) for i,r in enumerate(records))
    assert len(cases)==64 and len({r['case'] for r in cases})==64
    assert sum(r['branch_observed'] for r in cases)==57
    assert sum(r['unbranched_75um_complete'] for r in cases)==7
    assert not any(r['early_gate_censor'] for r in cases)
    groups=[];curves={};birth_rows=[]
    for g in GROUPS:
        rows=[r for r in cases if r['group']==g];assert len(rows)==8
        stats=distance_summary(rows);curves[g]=stats['product_limit']
        births=[dict(case=r['case'],group=g,seed=r['seed'],**mark_metrics(r['first_branch_clock_record'])) for r in rows if r['branch_observed']]
        birth_rows.extend(births)
        events=[r['first_branch_event_number'] for r in rows if r['branch_observed']]
        reaches=[r['first_branch_primary_reach_um'] for r in rows if r['branch_observed']]
        groups.append(dict(group=g,held_out_material_class=rows[0]['held_out_material_class'],cases=8,
            branches=len(births),completed_nonbranching=sum(r['unbranched_75um_complete'] for r in rows),
            early_gate_censored=sum(r['early_gate_censor'] for r in rows),
            primary_saturated_births=sum(b['primary_near_saturation'] for b in births),
            companion_saturated_births=sum(b['companion_near_saturation'] for b in births),
            pending_companion_births=sum(b['completed_pending_companion'] for b in births),
            first_branch_event_range=[min(events),max(events)] if events else None,
            first_branch_reach_range_um=[min(reaches),max(reaches)] if reaches else None,**stats))
    paired=[]
    for a,b in itertools.combinations(GROUPS,2):
        counts=Counter();pairs=[]
        for seed in p['seeds']:
            x=next(r for r in cases if r['group']==a and r['seed']==seed);y=next(r for r in cases if r['group']==b and r['seed']==seed)
            label='unknown_early_censor' if x['early_gate_censor'] or y['early_gate_censor'] else 'both' if x['branch_observed'] and y['branch_observed'] else 'a_only' if x['branch_observed'] else 'b_only' if y['branch_observed'] else 'neither'
            counts[label]+=1;pairs.append(dict(seed=seed,outcome=label,a_reach=x['survival_extension_um'],b_reach=y['survival_extension_um']))
        paired.append(dict(a=a,b=b,counts=dict(counts),pairs=pairs))
    inputs={str(f):sha256(f) for f in sorted(DEST.rglob('*')) if f.is_file()}
    for f in (PLAN,bundle,OUT/'full_suite.xml',OUT/'full_suite.log',OUT/'full_suite_result.json',old/'case_table.json',old/'final_topologies.json',old/'native_mark_opportunities.json'):
        inputs[str(f)]=sha256(f)
    FINAL.mkdir()
    clock_distributions=[]
    for g in GROUPS:
        ids={c['case'] for c in cases if c['group']==g}
        for subset in ('all_native_opportunities','accepted_births'):
            selected=[r for r in marks if r['case'] in ids and (subset=='all_native_opportunities' or r['outcome']=='PAIR_ACCEPTED')]
            metrics=[mark_metrics(r) for r in selected]
            clock_distributions.append(dict(group=g,subset=subset,count=len(selected),
                chi_B_definition='T_j / min(T_i_next, tau_c), from archived native waits; no clock evolution',
                status_counts=dict(Counter(m['chi_B_status'] for m in metrics)),
                finite_chi_B_values=sorted(m['chi_B'] for m in metrics if m['chi_B'] is not None),
                exact_pair_margins_J_per_m=[m['exact_pair_margin_J_per_m'] for m in metrics if m['exact_pair_margin_J_per_m'] is not None]))
    summary=dict(status='COMPLETE',groups=groups,boundary=p['boundary'],new_cases=32,reused_cases=32,
        simulation_program_stopped=True,source_physics_changed=False,heldout_results_used_to_retune=False,
        accepted_transition_record_commit=RECORD,frozen_physics_commit=PHYSICS,report_producer_code_commit=producer,
        launch_commits=sorted({r['execution_source_commit'] for r in cases if r['held_out_material_class']}),
        experimental_calibration=False,predictive_branching_validated=False,recursive_spacing_tested=False,
        merge_gate=suite['merge_gate'])
    for name,value in (('summary.json',summary),('case_table.json',dict(rows=cases)),('paired_seed_contrasts.json',dict(rows=paired)),
        ('chi_B_and_pair_margin_distributions.json',dict(rows=clock_distributions)),
        ('birth_saturation_and_pair_margins.json',dict(rows=birth_rows,near_saturation_definition='effective lambda*tau_c >= 0.99; descriptive only')),
        ('native_opportunities.json',dict(rows=[dict(r,metrics=mark_metrics(r)) for r in marks],
            unavailable='Native early-exit fields remain unevaluated; no rates/tensors/energy margins imputed or replayed.')),
        ('accepted_assessment_verification.json',audit),('final_topologies.json',networks),('full_suite_review.json',suite),
        ('provenance.json',dict(files=inputs,accepted_freeze=freeze,heldout_plan_sha256=sha256(PLAN)))):
        atomic_json(FINAL/name,value)
    figures(cases,groups,curves,networks)
    distribution_figures(cases,clock_distributions)
    report(summary,cases,suite)
    assert all(sha256(Path(path))==h for path,h in inputs.items())
    atomic_json(FINAL/'verification.json',dict(status='PASS',inputs_verified=len(inputs),accepted_freeze=verify_accepted(),cases=64))
    entries={'final/'+f.name:f for f in FINAL.iterdir() if f.is_file()}
    for name in ('heldout_plan.json','PREREGISTERED_HELDOUT_BLOCK.md','launch_gate_tests.xml','full_suite.xml','full_suite.log','full_suite_result.json','full_suite_claim.json','heldout_source_increment.bundle','HELDOUT_QUEUE_PAUSE.json','STORAGE_PAUSE_HANDOFF.md','publication_preflight.json','SIMULATIONS_COMPLETE_STORAGE_HOLD.json'):
        entries['provenance/'+name]=OUT/name
    entries['provenance/continuation_claim.json']=DEST/'continuation_claim.json'
    with zipfile.ZipFile(OUT/'V13_FOUR_MATERIAL_REVIEW.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,path in entries.items():z.write(path,name)
        z.writestr('SHA256_MANIFEST.json',json.dumps({name:sha256(path) for name,path in entries.items()},indent=2,sort_keys=True)+'\n')
    print(json.dumps([dict(group=g['group'],branches=g['branches'],censored=g['completed_nonbranching'],early=g['early_gate_censored'],rmst=g['restricted_mean_branch_free_reach_um']) for g in groups],indent=2))


def figures(cases,groups,curves,networks):
    colors={'Peak':'tab:blue','weakT':'tab:green','DBTT':'tab:orange','ceramic':'tab:red'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,temp in zip(axes,(300,1000)):
        for material,color in colors.items():
            g=f'{material}_{temp}K';curve=curves[g]
            x=[p['extension_um'] for p in curve];y=[p['survival'] for p in curve]
            if y[-1]==0. and x[-1]<75.:x.append(75.);y.append(0.)
            ax.step(x,y,where='post',label=material+(' (held out)' if material in ('DBTT','ceramic') else ''),color=color)
        ax.set(title=f'{temp} K',xlim=(0,75),ylim=(0,1.04),xlabel='Maximum forward reach (µm)',ylabel='Fraction without first branch')
        ax.legend(fontsize=8)
    fig.suptitle('V13 θ=15°, rate=1x — descriptive product-limit curves; uncalibrated model')
    fig.savefig(FINAL/'eight_group_first_branch_survival.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ti,temp in enumerate((300,1000)):
        selected=[next(g for g in groups if g['group']==f'{m}_{temp}K') for m in colors]
        x=[i+(ti-.5)*.35 for i in range(4)]
        axes[0].bar(x,[g['branches']/8 for g in selected],width=.35,label=f'{temp} K')
        means=[g['restricted_mean_branch_free_reach_um'] for g in selected]
        axes[1].bar([v for v,y in zip(x,means) if y is not None],[y for y in means if y is not None],width=.35,label=f'{temp} K')
        for v,g in zip(x,selected):
            axes[0].text(v,g['branches']/8+.015,f"{g['branches']}/8",ha='center',fontsize=8)
    for ax in axes:
        ax.set_xticks(range(4),['Peak','Weak-T','DBTT\nheld out','Ceramic-like\nheld out']);ax.legend()
    axes[0].set(ylabel='Observed branch fraction',ylim=(0,1.13))
    axes[1].set(ylabel='Restricted mean branch-free reach (µm)',ylim=(0,80))
    fig.suptitle('Fixed 75 µm observation window — model outcomes, not calibrated probabilities')
    fig.savefig(FINAL/'incidence_and_restricted_mean.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(8,8,figsize=(20,20),sharex=True,sharey=True,constrained_layout=True)
    xmax=max(100.,5*math.ceil(max(c['accepted_terminal_reach_um'] for c in cases)/5))
    for c in cases:
        ax=axes[GROUPS.index(c['group']),c['seed']-3621]
        for b in networks[c['case']]['branches']:
            ax.plot([(x-.0005)*1e6 for x,y in b['path_m']],[y*1e6 for x,y in b['path_m']],color='black' if b['generation']==0 else 'tab:blue',lw=1.2)
        status='branch' if c['branch_observed'] else 'early gate' if c['early_gate_censor'] else '75 µm censor'
        ax.set(title=c['group']+' / '+str(c['seed'])+'\n'+status,xlim=(-2,xmax),ylim=(-45,32));ax.title.set_fontsize(7);ax.set_aspect('equal')
    fig.supxlabel('Forward coordinate from initial tip (µm)');fig.supylabel('Transverse coordinate (µm)')
    fig.suptitle('64 final accepted topologies — black: parent; blue: daughters\nBRANCHING_KINETICS_MODEL_UNCALIBRATED')
    fig.savefig(FINAL/'all_64_final_topologies.png',dpi=150);plt.close(fig)


def distribution_figures(cases,clocks):
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),constrained_layout=True)
    for gi,g in enumerate(GROUPS):
        rows=[r for r in cases if r['group']==g]
        for r in rows:
            y=gi+(r['seed']-3624.5)*.07
            if r['branch_observed']:
                axes[0].plot(r['first_branch_primary_reach_um'],y,'o',ms=4,color='tab:blue')
                axes[1].plot(r['first_branch_event_number'],y,'o',ms=4,color='tab:blue')
            else:axes[0].plot(75.,y,'>',ms=6,color='tab:orange')
    for ax in axes:ax.set_yticks(range(8),GROUPS);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    axes[0].set(xlabel='First-branch reach (µm); triangles: completed 75 µm negatives',xlim=(0,79))
    axes[1].set(xlabel='First-branch event number (observed branches only)')
    fig.suptitle('Native event/reach distributions — seed offsets are display-only; uncalibrated model')
    fig.savefig(FINAL/'first_branch_distributions.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,temp in zip(axes,(300,1000)):
        for material in ('Peak','weakT','DBTT','ceramic'):
            r=next(r for r in clocks if r['group']==f'{material}_{temp}K' and r['subset']=='all_native_opportunities')
            vals=r['finite_chi_B_values']
            if vals:ax.step(vals,[(i+1)/len(vals) for i in range(len(vals))],where='post',label=f'{material} (n={len(vals)})')
        ax.axvline(1.,color='grey',ls='--',lw=1);ax.set_xscale('symlog',linthresh=.1)
        ax.set(title=f'{temp} K',xlabel='χ_B = T_j / min(T_i,next, τ_c)',ylabel='Empirical CDF among finite, evaluated opportunities',ylim=(0,1.04));ax.legend(fontsize=8)
    fig.suptitle('Native clock-race distributions — zero/pending values retained; unavailable waits not imputed')
    fig.savefig(FINAL/'chi_B_distributions.png',dpi=170);plt.close(fig)


def report(summary,cases,suite):
    def number(v):return 'not reached / unsupported' if v is None else f'{v:.2f}'
    groups=summary['groups'];held=[r for r in cases if r['held_out_material_class']]
    lines=['# Final V13 four-material first-branch comparison','',
        '**BRANCHING_KINETICS_MODEL_UNCALIBRATED**','',
        f"All 32 held-out cases terminated under the frozen rules: {sum(r['branch_observed'] for r in held)} observed branches, {sum(r['unbranched_75um_complete'] for r in held)} completed nonbranching observations, and {sum(r['early_gate_censor'] for r in held)} early-gate censors. The 32 accepted Peak/weak-T trajectories were reused without modification. The simulation program is stopped; no further screening, tuning or trajectories are authorized by this block.",'',
        '## Eight-group descriptive results','',
        '| Group | Branches / 8 | Completed 75 µm negatives | Early gates | Median first-branch reach (µm) | Restricted mean to 75 µm (µm) | First-branch event range |',
        '|---|---:|---:|---:|---:|---:|---|']
    for g in groups:lines.append(f"| {g['group']} | {g['branches']} | {g['completed_nonbranching']} | {g['early_gate_censored']} | {number(g['median_first_branch_reach_um'])} | {number(g['restricted_mean_branch_free_reach_um'])} | {g['first_branch_event_range']} |")
    lines+=['','DBTT and ceramic-like were held out from selection of this condition. Their results were not used to change orientation, loading rate, seeds, parameters or the V13 rule. They are not experimentally validated held-out data. Peak/weak-T seed 3621 remains the condition-selection seed; the other seven seeds are additional paired realizations, not a correction for all selection bias.','',
        'Peak (both temperatures), DBTT/300 K and ceramic-like (both temperatures) show high incidence: 8/8 each. Weak-T has the clearest mixed-incidence transition (4/8 at 300 K, 6/8 at 1000 K). DBTT/1000 K is delayed and slightly reduced (7/8; restricted mean branch-free reach 40.77 µm versus 13.09 µm at 300 K). Temperature effects are parameterization-specific: the descriptive direction differs between weak-T and DBTT; ceramic-like has equal incidence and equal restricted means at these two temperatures. These small-sample counts are model outcomes, not calibrated physical probabilities.','',
        'Larger restricted mean branch-free reach denotes later or absent branching within 75 µm. The median is the first reach where the product-limit curve is at or below 0.5, not the median conditional on an observed branch. No confidence interval or statistical material-probability calibration is assigned. Early model/numerical censors can be informative and are never counted as completed nonbranching observations. A restricted mean to 75 µm is left unavailable if observation support ends earlier with positive branch-free survival.','',
        'The censor is the prespecified 75 µm maximum-forward-reach threshold. Native 5 µm events are not shortened; the last accepted unbranched endpoint can exceed that threshold. Branches already accepted within the window continue through the existing 20 µm daughter-growth stop, or a legitimate gate. First-branch primary reach, junction coordinate, event number, time and opening are retained separately in the case table.','',
        '## Saturation, pending companions and energy margins','',
        '| Group | Births | Primary near asymptote | Companion near asymptote | Completed pending companion |',
        '|---|---:|---:|---:|---:|']
    for g in groups:lines.append(f"| {g['group']} | {g['branches']} | {g['primary_saturated_births']} | {g['companion_saturated_births']} | {g['pending_companion_births']} |")
    lines+=['','Near saturation means effective λτc ≥ 0.99, a descriptive convention retained from the accepted audit, not a modified physical threshold. Raw and effective λτc, completed-pending status and the exact native pair-energy margin in J/m are exported for every birth. All native opportunities are also retained. Early-exit fields that were not evaluated remain unavailable; no rates, tensors, release energies or margins were fabricated or recomputed to fill gaps.','',
        'The χ_B distributions use T_j / min(T_i,next, τ_c) directly from the archived native waits, separately for all opportunities and accepted births. Inherited-pending companions retain χ_B=0 when the primary race window is positive. Infinite and unevaluated waits have separate status counts and are not silently assigned finite values. The plotted empirical distributions condition on finite evaluated values; all statuses and exact pair-energy margins are exported in chi_B_and_pair_margin_distributions.json.','',
        'The accepted 15° screening energy margins were far from veto; this remains evidence for a kinetic transition in the selection groups. Absolute held-out pair margins are reported directly and are not silently converted to normalized energy margins without archived release/cost data. A pending completion survives later rate changes: its current small rate does not erase its already-completed event.','',
        '## Interpretation boundaries','',
        'The accepted mixed first-branch incidence and model-internal material sensitivity remain established for the tested window. The tables extend the same frozen rule to the two unused parameterizations; they do not establish experimental probability laws. Common seeds are paired across groups at fixed orientation; no independent-exchangeable interpretation of all 64 cases is used. Full paired-seed outcome tables are provided, including early-censor unknowns.','',
        'Temperature contrasts are descriptive, not statistically established. Candidate angles remain crystallographic, the morphology mark remains zero-global-time, and these runs do not test recursive branch spacing or calibrated microbranch survival/arrest. Continuous emission, cleavage hazards, tau_c, material rows, pair-energy rules and marked-event bookkeeping were not changed.','',
        '## Verification and merge disposition','',
        f"Full repository suite: return code {suite['returncode']}; {suite['recorded_testcases']} recorded test cases; {len(suite['failures'])} failures/errors; {suite['skipped']} skipped. Merge gate: **{suite['merge_gate']}**. The full suite was run once after all new trajectories terminated. Full failure details, if any, are retained; no source-physics fixes or simulation reruns were made in response.",'',
        'The assessment’s four restricted means and medians reproduce to the quoted two decimals. Its 26/26 primary-saturated births and five pending-companion births also reproduce. The accepted transition and theta40 file seals are verified before and after analysis. Raw process fingerprint differences between the earlier screen and production remain documented as run-local mechanics-call serial differences; no within-run freshness check was removed.','',
        f"Accepted transition record: `{RECORD}`. Frozen physics source: `{PHYSICS}`. New launch commits: `{', '.join(summary['launch_commits'])}`. Report-producer commit: `{summary['report_producer_code_commit']}`. These provenance roles are separate.",'',
        'Raw held-out results and final portable fields are in `../ensemble/<case>/`. The compact archive contains reports, tables, figures and test records; native checkpoint/file hashes remain available in provenance.json. No automatic merge is performed by the reporting script.','',
        '![First-branch curves](eight_group_first_branch_survival.png)','',
        '![Incidence and restricted mean](incidence_and_restricted_mean.png)','',
        '![First-branch event and reach distributions](first_branch_distributions.png)','',
        '![Native clock-race distributions](chi_B_distributions.png)','',
        '![Final topologies](all_64_final_topologies.png)','']
    (FINAL/'V13_FOUR_MATERIAL_FIRST_BRANCH_REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
