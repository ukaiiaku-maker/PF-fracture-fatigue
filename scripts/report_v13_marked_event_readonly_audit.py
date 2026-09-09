"""Interim report: explicit queue pause, archive gaps, and accepted geometries."""
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from scripts.audit_v13_marked_event_bookkeeping import DEST
from scripts.run_v13_primary_race_short_ensemble import OUT, case_folder
from scripts.run_pf_current_source_multifront_field_atlas_v12 import atomic_json,sha256


def main():
    cases=json.loads((DEST/'case_table.json').read_text())['rows']
    events=json.loads((DEST/'prebranch_opportunity_table.json').read_text())
    audit=json.loads((DEST/'marked_event_bookkeeping.json').read_text())
    stopped=case_folder('Peak_300K_seed3624')
    cp=restore_branch_checkpoint(stopped/'checkpoint/latest.json')
    owner_error={'case':'Peak_300K_seed3624','accepted_step':cp.state.event_counters['accepted_steps'],
        'accepted_checkpoint_sha256':sha256(stopped/'checkpoint/latest.json.state.pkl'),
        'exception_sha256':sha256(stopped/'exception.json'),'reason':'one topology transaction spans multiple pre-event tips',
        'source':'arrhenius_fracture/tip_directional_observation_v11.py:selected_event_owner',
        'source_sha256':sha256(Path('arrhenius_fracture/tip_directional_observation_v11.py')),
        'transaction_details_not_archived':True,
        'source_diagnosis':'selected multi-candidate proposal maps to distinct pre-event daughter tips; selected_event_owner requires one tip for a multi-candidate transaction',
        'not_a_post_checkpoint_output_exception':True,'not_demonstrated_double_consumption':True,
        'queue_must_remain_paused':True,'process_clusters':[c.cluster_id for c in cp.branch_clusters],
        'daughter_candidate_map':[{'front_id':b.branch_id,'candidate_id':b.local_state.get('candidate_id'),'tip_xy_m':b.tip}
            for b in cp.state.crack_network.branches if b.status=='active'],
        'last_accepted_clocks':[asdict(h) for h in cp.state.competition.hazard_states]}
    atomic_json(DEST/'state_owner_stop_diagnosis.json',owner_error)
    order=['Peak_300K','Peak_1000K','weakT_300K','weakT_1000K']
    fig,axes=plt.subplots(4,4,figsize=(12,10),sharex=True,sharey=True,constrained_layout=True)
    for c in cases:
        group,seed=c['case'].rsplit('_seed',1)
        ax=axes[order.index(group),int(seed)-3621]
        title=f'{group} / {seed}'
        if c['status']=='NOT_LAUNCHED':
            ax.text(.5,.5,'NOT LAUNCHED\nqueue paused',ha='center',va='center',transform=ax.transAxes)
        else:
            state=restore_branch_checkpoint(case_folder(c['case'])/'checkpoint/latest.json').state
            a0=max(b.tip[0] for b in state.crack_network.branches)-c['forward_extension_um']*1e-6
            for b in state.crack_network.branches:
                ax.plot([(x-a0)*1e6 for x,y in b.path],[y*1e6 for x,y in b.path],
                    color='black' if b.generation==0 else 'tab:blue',lw=1.4)
            title+='\n'+('STATE ERROR — last accepted' if c['status']=='SOFTWARE_OR_UNCLASSIFIED_STOP' else '20 µm daughter growth completed')
        ax.set_title(title,fontsize=8);ax.set_xlim(-2,45);ax.set_ylim(-28,28);ax.set_aspect('equal')
        if int(seed)==3621:ax.set_ylabel('Transverse (µm)')
        if group==order[-1]:ax.set_xlabel('Forward extension (µm)')
    fig.suptitle('V13 short gate — interim, 13 completed / 1 state error / 2 unlaunched\nBRANCHING_KINETICS_MODEL_UNCALIBRATED',fontsize=11)
    fig.savefig(DEST/'interim_16_case_topologies.png',dpi=170);plt.close(fig)
    lines=['# V13 marked-event audit and paused short ensemble','',
        '**BRANCHING_KINETICS_MODEL_UNCALIBRATED**','',
        '## Disposition','',
        'The two completed Peak seed-3621 branches pass all 29 bookkeeping checks each. The evidence supports **INTENDED_MARKED_EVENT_SEMANTICS**, not double consumption or double geometry accounting. No marked-event semantic change is warranted by this audit. No production source, state, clock, RNG, or mechanics was changed.','',
        'The queue is already paused at Peak/300 K, seed 3624, after accepted step 419. Thirteen cases reached the prescribed daughter-growth stop; this additional case encountered a state-owner error; the two Weak-T seed-3624 cases remain unlaunched. They are not negative branching observations. No completed case was restarted.','',
        '## Peak birth and daughter clock proof','',
        '| Case | Initial pair release (J/m) | Initial pair cost (J/m) | First companion growth step | First primary growth step | Checks |',
        '|---|---:|---:|---:|---:|---:|']
    for a in audit['cases']:
        d={d['role']:d for d in a['daughters']}
        lines.append(f"| {a['case']} | {a['pair_release_J_per_m']:.12g} | {a['pair_cost_J_per_m']:.12g} | {d['companion']['first_post_birth_accepted_event']['step']} | {d['primary']['first_post_birth_accepted_event']['step']} | 29/29 |")
    lines+=['',
        'The canonical primary event is consumed once. At birth, the companion event `(100)#event:2` is absent from the consumed-event ledger. Both daughter mappings retain the exact shared competition inventory with candidate-specific ownership; no independent duplicate clock is created. The companion keeps its accumulated action, cumulative threshold, ordinal and threshold RNG identity. Its later event 2 is consumed once to extend its existing 5 µm daughter to 10 µm. The primary next event is `(010)#event:5`. Full candidate IDs, actions, thresholds, RNG identities and first accepted event records are in `marked_event_bookkeeping.json`.','',
        'The pair contains two initial 5 µm arms, adding only one 5 µm companion segment relative to the already accepted single state. The pair release/cost ledger replaces the single transaction increment rather than adding the primary cost twice. Final geometry and cumulative energy equal the birth pair plus the subsequent accepted growth transactions. Clocks, baseline event counters, process state and physical time are unchanged by the mark.','',
        'This is a zero-time marked-parent approximation: the companion threshold predicts the mark but is not credited with a cleavage event at birth. That predictor use and its later single physical consumption are distinct. The audit proves bookkeeping consistency, not physical calibration of the approximation.','',
        '## Current case evidence','',
        '| Case | First branch event | Primary reach at birth (µm) | Last accepted reach (µm) | Status |','|---|---:|---:|---:|---|']
    for c in cases:
        def fmt(v):return '—' if v is None else f'{v:.6g}'
        lines.append(f"| {c['case']} | {fmt(c['first_branch_event_number'])} | {fmt(c['first_branch_primary_forward_um'])} | {fmt(c.get('forward_extension_um'))} | {c['status']} |")
    lines+=['',
        'All 14 launched cases have an accepted first branch, but only 13 completed the prescribed post-birth growth. First-branch event numbers vary from 2 to 5 across the available seed realizations. At seed 3622, Peak/1000 K branches at event 5 whereas the other three paired cases branch at event 4. Thus the available results already contain seed variation and a paired case difference; identical seed-3621 Peak geometry is not evidence of deterministic selection. These are provisional observations, not a final regime classification or calibrated probabilities.','',
        'The requested final four-way classification is withheld: the ensemble is incomplete due to an unqualified state-owner error, not a certified set of legitimate physical terminal gates. Assigning the “insufficient due to exact gates” category would mislabel the software/state issue.','',
        '## Event-by-event table and archive gaps','',
        f"`prebranch_opportunity_table.json` contains {len(events['rows'])} archived opportunities, including single-arm losses. For {events['missing_primary_time_rows']} rows the primary post-event time and pair margin were not evaluated/archived because execution short-circuited on companion expiry or inadmissibility. Those fields are explicit nulls with reasons; ratios are only computed where both inputs exist. Both candidate clock actions/thresholds/ordinals are joined from the same accepted endpoint. Pre-topology interval rates are not substituted for missing post-primary rates.", '',
        'A complete retrospective table of those unevaluated quantities cannot be claimed from this read-only export. No mechanics or missing full process state was reconstructed. The compatibility-clock launch difference remains disclosed; no full byte-identity claim is made between independent historical launches.','',
        '## State-owner stop','',
        '`selected_event_owner` accepts a multi-candidate action only when all candidate observations share one pre-event tip. The failed update selected a multi-candidate action whose observations map to distinct daughters. This is an incompatibility between a post-birth multi-tip transaction and the single-tip event-owner check; it is not an output error and not evidence of threshold duplication. The traceback establishes the call path; the exact failed proposal was not serialized, so no missing failed-interval values are invented.','',
        'Per the requested boundary, the queue remains paused. A narrow owner-transaction correction and exact-checkpoint continuation need a separate authorization; no fix, bypass, reseed, or further launch is applied here.','',
        '![Last accepted topologies; not a completed 16-case atlas](interim_16_case_topologies.png)','']
    (DEST/'V13_MARKED_EVENT_BOOKKEEPING_AND_QUEUE_STATUS.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
