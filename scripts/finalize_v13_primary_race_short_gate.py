"""Final descriptive short-gate report and compact review; no simulation calls."""
from collections import defaultdict
import argparse
import json
import math
from pathlib import Path
import zipfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from scripts.run_v13_primary_race_short_ensemble import OUT, CASES, case_folder
from scripts.report_v13_primary_continuation_race import main as report
from scripts.run_pf_current_source_multifront_field_atlas_v12 import atomic_json,sha256


def main():
    report()
    summary=json.loads((OUT/'short_ensemble_summary.json').read_text())
    if not summary['complete']:
        raise RuntimeError('short gate incomplete; no final review archive generated')
    if any(r['status']=='SOFTWARE_OR_UNCLASSIFIED_STOP' for r in summary['records']):
        raise RuntimeError('unclassified software stop; not a completed physical gate')
    groups=defaultdict(list)
    fig,axes=plt.subplots(4,4,figsize=(12,10),sharex=True,sharey=True,constrained_layout=True)
    order=['Peak_300K','Peak_1000K','weakT_300K','weakT_1000K']
    records=[]
    for r in summary['records']:
        group,seed=r['case'].rsplit('_seed',1)
        groups[group].append(r)
        folder=case_folder(r['case'])
        checkpoint=restore_branch_checkpoint(folder/'checkpoint/latest.json')
        state=checkpoint.state
        initial_tip_x=max(b.tip[0] for b in state.crack_network.branches)-checkpoint.projected_extension_m
        if r['first_branch']:
            r['first_branch_junction_forward_um']=(r['first_branch']['branch_junction_xy_m'][0]-initial_tip_x)*1e6
        ax=axes[order.index(group),int(seed)-3621]
        for branch in state.crack_network.branches:
            path=branch.path
            points=[((x-initial_tip_x)*1e6,y*1e6) for x,y in path]
            ax.plot([x for x,y in points],[y for x,y in points],lw=1.2,
                color='black' if branch.generation==0 else 'tab:blue')
        ax.set_title(f'{group}, seed {seed}\n{r["reason"]}',fontsize=7)
        ax.set_xlim(-5,130);ax.set_ylim(-75,75);ax.set_aspect('equal')
        if int(seed)==3621:ax.set_ylabel('Transverse coordinate (µm)')
        if group==order[-1]:ax.set_xlabel('Forward extension (µm)')
        records.append({'case':r['case'],'terminal_sha256':sha256(folder/'terminal.json'),
            'launch_sha256':sha256(folder/'launch.json'),
            'race_ledger_sha256':sha256(folder/'v13_primary_race.jsonl') if (folder/'v13_primary_race.jsonl').exists() else None,
            'checkpoint_state_sha256':r.get('last_checkpoint_state_sha256'),
            'first_branch_primary_forward_um':r['first_branch']['parent_forward_extension_um'] if r['first_branch'] else None,
            'first_branch_junction_forward_um':r.get('first_branch_junction_forward_um'),
            'first_branch_time_s':r['first_branch']['accepted_time_s'] if r['first_branch'] else None,
            'first_branch_opening_m':r['first_branch']['accepted_opening_m'] if r['first_branch'] else None,
            'terminal_forward_um':r.get('forward_extension_um'),
            'spacing_semantics':'initial tip to first bifurcation only; recurrent branch spacing not sampled'})
    fig.suptitle('V13 short physical gate — model-native, branching kinetics uncalibrated')
    fig.savefig(OUT/'short_gate_topologies.png',dpi=180);plt.close(fig)
    incidence=[]
    for group in order:
        rows=groups[group]
        births=[r['first_branch_junction_forward_um'] for r in rows if r['first_branch']]
        incidence.append({'case':group,'seeds':4,'observed_branches':len(births),
            'first_branch_junction_forward_um':births,'no_branch_terminal_reasons':[r['reason'] for r in rows if not r['first_branch']],
            'calibrated_probability':None,'recurrent_spacing_estimated':False})
    atomic_json(OUT/'short_gate_incidence_and_spacing.json',{'groups':incidence,'records':records,
        'boundary':'BRANCHING_KINETICS_MODEL_UNCALIBRATED'})
    path=OUT/'V13_PRIMARY_CONTINUATION_RACE.md'
    lines=['','## Completed descriptive incidence','',
        '| Case | Observed first branches / four seeds | First branch-junction distances (µm) |','|---|---:|---|']
    for item in incidence:
        lines.append(f"| {item['case']} | {item['observed_branches']}/4 | {', '.join(f'{x:.5f}' for x in item['first_branch_junction_forward_um']) or 'none'} |")
    lines+=['','Stops before a branch are retained with their exact reasons, not silently interpreted as full-window nonbranching. These counts are descriptive outcomes of the bounded model, not calibrated material probabilities. Only first-bifurcation distance is observed; this experiment cannot estimate recurrent branch spacing.','']
    path.write_text(path.read_text()+'\n'.join(lines))
    selected=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.md','.png','.jsonl')
        and ('short_ensemble' not in p.parts or p.name in ('terminal.json','launch.json','v13_primary_race.jsonl'))
        and p.name not in ('review_manifest.json','queue_status.json')]
    atomic_json(OUT/'review_manifest.json',{'files':[{'path':str(p.relative_to(OUT)),'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(selected)],
        'large_fields_and_checkpoints_retained_on_Data_not_embedded':True})
    with zipfile.ZipFile(OUT/'V13_PRIMARY_RACE_SHORT_GATE_REVIEW.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in selected+[OUT/'review_manifest.json']:
            archive.write(p,p.relative_to(OUT))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    if args.watch:
        pause_name='remaining_queue_pause.json' if (OUT/'short_ensemble/remaining_queue_claim.json').exists() else 'recovery_queue_pause.json' if (OUT/'snapshot_output_recovery_registry.json').exists() else 'queue_pause.json'
        while not all((case_folder(case)/'terminal.json').exists() for case in CASES):
            if (OUT/'short_ensemble'/pause_name).exists():
                raise RuntimeError('queue paused for software/unclassified stop; no final result invented')
            time.sleep(30)
    main()
