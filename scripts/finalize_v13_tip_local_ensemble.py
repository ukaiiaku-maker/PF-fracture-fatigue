"""One final export of the completed bounded ensemble; no mechanics or clocks."""
from collections import Counter
import itertools
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from scripts.run_v13_primary_race_short_ensemble import OUT, CASES, case_folder
from scripts.run_v13_tip_local_recovery import DEST, EXPECTED, verify, ROOT
from scripts.audit_v13_marked_event_bookkeeping import opportunities, rows
from scripts.run_pf_current_source_multifront_field_atlas_v12 import atomic_json, sha256

FINAL = DEST/'final_16_case_record'


def authoritative(case):
    return DEST/'continuation' if case == 'Peak_300K_seed3624' else case_folder(case)


def main():
    # No partial publication, no replacement of the historical paused audit.
    unchanged = verify()
    terminals = {c: json.loads((authoritative(c)/'terminal.json').read_text()) for c in CASES}
    assert all(t['status'] in ('TERMINATED','EXISTING_GATE_STOP') for t in terminals.values())
    FINAL.mkdir()
    archived_path = OUT/'marked_event_readonly_audit/prebranch_opportunity_table.json'
    archived = json.loads(archived_path.read_text())['rows']
    events = list(archived)
    assert len(archived) == 58
    assert sum(e['T_primary_next_s'] is None for e in archived) == 36
    for case in ('weakT_300K_seed3624','weakT_1000K_seed3624'):
        assert not any(e['case'] == case for e in archived)
        events.extend(opportunities(case))
    assert events[:58] == archived
    cases = []
    topologies = []
    transactions = []
    protected = {}
    for case in CASES:
        folder = authoritative(case)
        cp = restore_branch_checkpoint(folder/'checkpoint/latest.json')
        births = [e for e in events if e['case'] == case and e['outcome'] == 'PAIR_ACCEPTED']
        first = births[0] if births else None
        terminal = terminals[case]
        plan = json.loads((OUT/'short_ensemble_plan.json').read_text())
        expected_seed = int(case.rsplit('seed',1)[1])
        assert cp.state.competition.global_hazard_seed == expected_seed
        assert set(cp.front_competitions) == set(cp.state.crack_network.active_tip_ids)
        for name in ('terminal.json','launch.json','v13_primary_race.jsonl','branch_action_trials.jsonl',
                     'directional_rates.jsonl','checkpoint/latest.json','checkpoint/latest.json.state.pkl','final_accepted_fields.npz'):
            p = folder/name
            if p.exists():
                protected[str(p)] = sha256(p)
        launch = json.loads((folder/'launch.json').read_text())
        c = dict(terminal,case=case,seed=expected_seed,material=case.split('_')[0],
            temperature_K=int(case.split('_')[1][:-1]),authoritative_output=str(folder),
            source_commit=launch['source_commit'],continued_from_step419=case=='Peak_300K_seed3624',
            source_checkpoint_sha256=EXPECTED if case=='Peak_300K_seed3624' else None,
            first_branch_event_number=first['accepted_event_number'] if first else None,
            single_opportunities_before_first_branch=first['accepted_event_number']-1 if first else None,
            first_branch_primary_forward_um=first['primary_forward_reach_um'] if first else None,
            branch_junction_forward_um=(first['branch_junction_xy_m'][0]-0.0005)*1e6 if first else None,
            accepted_steps=cp.state.event_counters['accepted_steps'],
            primary_raw_lambda_tau_c=first['primary_raw_rate_per_s']*first['tau_c_s'] if first else None,
            companion_raw_lambda_tau_c=first['companion_raw_rate_per_s']*first['tau_c_s'] if first else None,
            primary_effective_lambda_tau_c=first['primary_effective_rate_per_s']*first['tau_c_s'] if first else None,
            companion_effective_lambda_tau_c=first['companion_effective_rate_per_s']*first['tau_c_s'] if first else None)
        cases.append(c)
        topologies.append({'case':case,'network':cp.state.crack_network.to_dict(),
            'front_competition_ids':sorted(cp.front_competitions),
            'process_owner_ids':[v.cluster_id for v in cp.branch_clusters],
            'checkpoint_state_sha256':terminal['last_checkpoint_state_sha256']})
        # Explicitly distinguish the event owner from the trial's broader
        # participating-front list. Participation alone is NOT consumption.
        obs = {(r['step'],r['candidate_id']):r for r in rows(folder/'directional_rates.jsonl')}
        for t in rows(folder/'branch_action_trials.jsonl'):
            if not t['accepted']:
                continue
            observations = [obs[(t['step'],cid)] for cid in t['candidate_ids']]
            owners = {o['tip_id'] for o in observations}
            assert len(owners) == 1
            ids = [(o['tip_id'],cid,int(eid.rsplit(':',1)[1]))
                   for o,cid,eid in zip(observations,t['candidate_ids'],t['pending_event_ids'])]
            transactions.append(dict(case=case,step=t['step'],action_id=t['trial_id'],
                event_owner_tip_id=next(iter(owners)),process_owner_id=observations[0]['process_owner_id'],
                scoped_event_identities=ids,consumed_event_ids=t['pending_event_ids'],
                trial_participating_front_ids=t['participating_front_ids'],
                completion_times_s=t['completion_times_s'],physical_time_s=t['physical_time_s'],
                topology_fingerprint_before=t['topology_fingerprint_before'],topology_fingerprint_after=t['topology_fingerprint_after']))
    resumed = [t for t in transactions if t['case']=='Peak_300K_seed3624']
    assert min(t['step'] for t in resumed) == 420
    first = next(t for t in resumed if t['step']==420)
    assert first['event_owner_tip_id'] == 'b042c2d7b4cc6a46'
    later = next(t for t in resumed if any('(010)' in e and e.endswith(':0000000000000002') for e in t['consumed_event_ids']))
    assert later['step'] > 420
    for c in cases:
        ids = [e for t in transactions if t['case']==c['case'] for e in t['consumed_event_ids']]
        assert len(ids) == len(set(ids))
    contrasts = []
    for seed in (3621,3622,3623,3624):
        for a,b in itertools.combinations([c for c in cases if c['seed']==seed],2):
            if a['first_branch_event_number'] is not None and b['first_branch_event_number'] is not None:
                contrasts.append(dict(seed=seed,a=a['case'],b=b['case'],
                    delta_first_branch_event_number_b_minus_a=b['first_branch_event_number']-a['first_branch_event_number'],
                    delta_primary_forward_um_b_minus_a=b['first_branch_primary_forward_um']-a['first_branch_primary_forward_um'],
                    delta_branch_junction_forward_um_b_minus_a=b['branch_junction_forward_um']-a['branch_junction_forward_um']))
    groups = ['Peak_300K','Peak_1000K','weakT_300K','weakT_1000K']
    sensitivity = {'interpretation':'descriptive balanced-table sum of squares only; no independent error estimate, significance test, or probability calibration',
                   'groups':groups,'seeds':[3621,3622,3623,3624]}
    for field in ('first_branch_event_number','branch_junction_forward_um','first_branch_primary_forward_um'):
        matrix = np.array([[next(c[field] for c in cases if c['case']==f'{g}_seed{s}')
                            for s in sensitivity['seeds']] for g in groups],dtype=float)
        mean = matrix.mean()
        group_mean = matrix.mean(axis=1,keepdims=True)
        seed_mean = matrix.mean(axis=0,keepdims=True)
        total = float(np.sum((matrix-mean)**2))
        ss_seed = float(4*np.sum((seed_mean-mean)**2))
        ss_group = float(4*np.sum((group_mean-mean)**2))
        ss_interaction = float(np.sum((matrix-group_mean-seed_mean+mean)**2))
        sensitivity[field] = dict(values_by_group_then_seed=matrix.tolist(),
            within_group_seed_ranges=np.ptp(matrix,axis=1).tolist(),
            within_seed_group_ranges=np.ptp(matrix,axis=0).tolist(),
            SS_total=total,SS_seed=ss_seed,SS_material_temperature_group=ss_group,
            SS_seed_group_interaction=ss_interaction,
            fractions=dict(seed=ss_seed/total,material_temperature_group=ss_group/total,
                           seed_group_interaction=ss_interaction/total))
    all_births = all(c['branch_count'] == 1 for c in cases)
    saturation = all(c['primary_effective_lambda_tau_c'] is not None and
                     c['primary_effective_lambda_tau_c'] >= .99 for c in cases)
    companion_saturation = sum(c['companion_effective_lambda_tau_c'] is not None and
                              c['companion_effective_lambda_tau_c'] >= .99 for c in cases)
    pending_at_birth = [e for e in events if e['outcome']=='PAIR_ACCEPTED' and
                        e['companion'].get('pending_event_id') is not None]
    seed_variation = len({c['first_branch_event_number'] for c in cases}) > 1
    assert all_births and seed_variation and saturation, 'reassess formal classification from actual data'
    classification = 'STOCHASTIC_BUT_SATURATION_DOMINATED'
    summary = dict(status='COMPLETE',cases=16,scientific_terminal_count=16,
        formal_classification=classification,boundary='BRANCHING_KINETICS_MODEL_UNCALIBRATED',
        observed_branch_count=sum(c['branch_count'] for c in cases),observed_incidence='16/16 bounded realizations; not calibrated probability',
        terminal_reasons=dict(Counter(c['reason'] for c in cases)),
        first_branch_event_range=[min(c['first_branch_event_number'] for c in cases),max(c['first_branch_event_number'] for c in cases)],
        primary_near_saturation_at_birth_count=16,companion_near_saturation_at_birth_count=companion_saturation,
        inherited_completed_companion_at_birth_count=len(pending_at_birth),
        saturation_definition='effective lambda * tau_c >= 0.99; descriptive, not a changed physical threshold',
        original_unevaluated_rows_preserved=36,total_opportunities=len(events),
        total_missing_primary_time_rows=sum(e['T_primary_next_s'] is None for e in events),
        unchanged_original_input_artifacts=unchanged,completed_cases_not_rerun=13,
        physical_launches_this_authorization=3,continuation_from_step419_count=1,
        frozen_failed_interval_replay_count=1,frozen_corrected_interval_replay_count=1,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        correction='CROSS_TIP_EVENTS_INCORRECTLY_GROUPED_AS_ONE_ACTION',owner_guard_unchanged=True,
        resumed_first_event=first,resumed_preserved_pending_event=later,
        canonical_plan_sha256=sha256(OUT/'short_ensemble_plan.json'),
        family_sha256=plan['family_sha256'],material_parameters_changed=False)
    atomic_json(FINAL/'summary.json',summary)
    atomic_json(FINAL/'case_table.json',{'rows':cases})
    atomic_json(FINAL/'prebranch_opportunity_table.json',{'rows':events,'original_58_rows_unchanged':True,
        'historical_unevaluated_primary_time_rows':36,'missing_values_are_not_backfilled':True})
    atomic_json(FINAL/'paired_seed_contrasts.json',{'rows':contrasts,'calibrated_probability_claimed':False})
    atomic_json(FINAL/'seed_and_material_sensitivity.json',sensitivity)
    atomic_json(FINAL/'event_owner_transactions.json',{'rows':transactions,
        'scope':'accepted transactions in each authoritative output; step419 continuation excludes original accepted prefix, preserved separately',
        'trial_participating_front_ids_are_not_event_owners':True})
    atomic_json(FINAL/'final_topologies.json',{'rows':topologies})
    atomic_json(FINAL/'authoritative_case_registry.json',{c['case']:c['authoritative_output'] for c in cases})
    atomic_json(FINAL/'provenance.json',{'files':protected,'frozen_checkpoint_sha256':EXPECTED,
        'historical_opportunity_table_sha256':sha256(archived_path),
        'qualification_sha256':sha256(DEST/'qualification.json'),'source_commit':summary['source_commit']})
    order = ['Peak_300K','Peak_1000K','weakT_300K','weakT_1000K']
    fig,axes=plt.subplots(4,4,figsize=(12,10),sharex=True,sharey=True,constrained_layout=True)
    for c,top in zip(cases,topologies):
        group,seed=c['case'].rsplit('_seed',1)
        ax=axes[order.index(group),int(seed)-3621]
        for b in top['network']['branches']:
            ax.plot([(x-.0005)*1e6 for x,y in b['path_m']],[y*1e6 for x,y in b['path_m']],
                    color='black' if b['generation']==0 else 'tab:blue',lw=1.4)
        ax.set_title(f"{group} / {seed}\nbirth event {c['first_branch_event_number']}; reach {c['forward_extension_um']:.2f} µm",fontsize=8)
        ax.set_xlim(-2,45);ax.set_ylim(-28,28);ax.set_aspect('equal')
        if int(seed)==3621:ax.set_ylabel('Transverse (µm)')
        if group==order[-1]:ax.set_xlabel('Forward extension (µm)')
    fig.suptitle('V13 completed short gate — sixteen bounded realizations\nBRANCHING_KINETICS_MODEL_UNCALIBRATED',fontsize=11)
    fig.savefig(FINAL/'final_16_case_topologies.png',dpi=180);plt.close(fig)
    lines=['# V13 tip-local correction and completed short ensemble','',
        '**BRANCHING_KINETICS_MODEL_UNCALIBRATED**','',
        f'Formal classification: **{classification}**. All sixteen cases formed a first branch and reached a scientific terminal condition. These are bounded capability calculations, not calibrated branch probabilities.','',
        '## Narrow defect and correction','',
        'The sealed step-419 replay reproduced a two-arm action consuming `(010)#2` and `(100)#3` from different pre-event daughter tips. Their completion times were 3424.8555772088716 and 3424.8555771377714 s, respectively (about 71.1 ns apart). They share process owner `jbe1d699748fbb56`, but not an event owner. This is outcome A: cross-tip events incorrectly grouped as one action. There was no coalescence or new branch in that failed proposal.','',
        'Proposal grouping now uses the pre-event tip and its branch-birth opportunity. Multifront proposal identity contains `(front_id, candidate_id, ordinal)`. Legacy clock/event keys remain unchanged to preserve checkpoint thresholds and RNG. Same-tip correlation rules are unchanged; tip-local winners compete globally by completion time. The original owner guard is byte-identical. No rates, material fields, correlation time, pair-energy rule, or marked-parent semantics changed.','',
        f"The live continuation resumed exactly once from step 419, SHA-256 `{EXPECTED}`. Step 420 consumed only the earlier `(100)#3`; the preserved `(010)#2` was consumed later at step {later['step']}. The original failed output remains evidence, not the authoritative terminal result. The thirteen completed cases were not rerun; {unchanged} original input artifacts retain their hashes.",'',
        '## Completed case table','',
        '| Case | Birth event | Junction forward (µm) | Primary reach at birth (µm) | Final reach (µm) | Terminal reason |',
        '|---|---:|---:|---:|---:|---|']
    for c in cases:
        lines.append(f"| {c['case']} | {c['first_branch_event_number']} | {c['branch_junction_forward_um']:.5f} | {c['first_branch_primary_forward_um']:.5f} | {c['forward_extension_um']:.5f} | {c['reason']} |")
    lines += ['', '## Interpretation and limits','',
        f'First branching occurs at cleavage events 2–5 and its location varies across seeds: the current race is not exactly deterministic in branch location. The primary effective rate is within 1% of the retained `1/tau_c` saturation asymptote at all sixteen births; the companion is near that asymptote at {companion_saturation}/16. High incidence in this short regime coexists with stochastic onset spacing.','',
        f'{len(pending_at_birth)} births use an already-completed inherited companion clock (`T_j = 0`). A pending completion is not erased by a later small instantaneous rate: Peak/1000 K seed 3624, for example, has companion effective lambda*tau_c about 3.83e-27 at the mark, but the archived pending event already exists. Thus the formal saturation-dominated label describes the primary race and high bounded incidence; it does not assert that every companion is currently saturated or establish a causal saturation-only explanation. The retained-clock mechanism also matters.','',
        'Paired material/temperature differences exist (notably Peak/1000 K at seed 3622), but are smaller than the overall seed/onset spread in this bounded sample. `paired_seed_contrasts.json` contains all paired differences and `seed_and_material_sensitivity.json` gives a descriptive balanced-table decomposition, not a significance test. Four seeds do not establish calibrated material-dependent probabilities or predictive recursive-branching physics. Junction position and the accepted primary endpoint are different geometric quantities and are reported separately.','',
        f"The complete opportunity table contains {len(events)} rows. All original 58 rows are preserved exactly, including the 36 with unevaluated primary-continuation time/pair margin. Nulls were not filled from pre-event rates or inferred mechanics. New cases contribute only their actual archived evaluations.",'',
        'The previously disclosed compatibility-clock constructor difference across historical independent launches remains disclosed; no full byte-identity claim is made between those launches. The resumed case retains the exact saved directional clocks, process state, thresholds, ordinals, and RNG; no reinitialization or reseed was used.','',
        '## Verification','',
        'Ten narrow step-419 regressions pass. Twenty-two existing frozen production-callback, unchanged-parent/AST, and step-controller tests pass. The forensic live-engine pickle needed a diagnostic-only reader for already registered instance-bound methods; this did not change production checkpoint serialization or physics. Compileall and `git diff --check` passed. No broad new qualification campaign was run.','',
        '![All sixteen final accepted crack topologies](final_16_case_topologies.png)','',
        'Data: `case_table.json`, `prebranch_opportunity_table.json`, `paired_seed_contrasts.json`, `event_owner_transactions.json`, `final_topologies.json`, `authoritative_case_registry.json`, and `provenance.json`. Earlier paused audit reports remain historical, not the final disposition.','']
    (FINAL/'V13_TIP_LOCAL_CORRECTION_AND_FINAL_ENSEMBLE.md').write_text('\n'.join(lines))
    assert verify() == unchanged
    assert all(sha256(Path(p)) == h for p,h in protected.items())
    atomic_json(FINAL/'verification.json',{'status':'PASS','original_immutable_files':unchanged,
        'final_inputs_unchanged':len(protected),'old_opportunity_rows_exact':58,'old_null_rows_exact':36,
        'sixteen_checkpoint_owner_registries_valid':True,'accepted_consumptions_unique':True})
    manifest = {p.name:sha256(p) for p in sorted(FINAL.iterdir()) if p.is_file()}
    atomic_json(FINAL/'bundle_manifest.json',manifest)
    entries = {f'final_16_case_record/{p.name}':p for p in sorted(FINAL.iterdir())}
    for rel in ('qualification.json','frozen_failure/failed_proposal.json','frozen_failure/failed_interval.pkl',
                'frozen_failure/input_checkpoint/step0000419.json',
                'frozen_failure/input_checkpoint/latest.json.state.pkl',
                'frozen_corrected/checkpoint/latest.json','frozen_corrected/checkpoint/latest.json.state.pkl',
                'frozen_corrected/renewal_calls.json','frozen_corrected/frozen_result.json','focused_tests.xml'):
        entries['qualification/'+rel] = DEST/rel
    for rel in ('arrhenius_fracture/tip_local_proposals_v13.py','arrhenius_fracture/production_step_loop_v11.py',
                'arrhenius_fracture/directional_competition_v11.py','scripts/run_v13_tip_local_recovery.py',
                'scripts/finalize_v13_tip_local_ensemble.py','tests/test_v13_tip_local_proposals.py'):
        entries['source/'+rel] = ROOT/rel
    with zipfile.ZipFile(DEST/'V13_FINAL_16_CASE_REVIEW.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,p in entries.items():archive.write(p,name)
        archive.writestr('ARCHIVE_SHA256_MANIFEST.json',json.dumps({name:sha256(p) for name,p in entries.items()},indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
