"""Preregistered physical orientation/rate screen, with no committed branch.

Every observed canonical cleavage is evaluated on an isolated exact state.
Stop a parent after the first predicted admissible branch, or 75 um without
one. That is screening, not an observed branch-incidence ensemble.
"""
import argparse
from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from scripts.run_pf_current_source_multifront_field_atlas_v12 import (
    ROOT, ROWS, common_arguments, campaign_environment, registry_row, atomic_json, sha256)
from scripts.run_v13_primary_race_short_ensemble import OUT as OLD
from scripts.finalize_v13_tip_local_ensemble import authoritative
from scripts.v13_value_fingerprint import physical_state_fingerprint as fp

OUT = ROOT/'analysis_outputs/v13_transition_regime_screen'
GROUPS = ('Peak_300K','Peak_1000K','weakT_300K','weakT_1000K')
PLAN = OUT/'screen_plan.json'
BOUNDARY = 'BRANCHING_KINETICS_MODEL_UNCALIBRATED'


def prepare():
    from arrhenius_fracture.kernel_registry_v10227 import validate_family
    OUT.mkdir(exist_ok=True)
    if PLAN.exists():
        raise RuntimeError('plan already sealed; no overwrite')
    families = {}
    for f in (ROOT/'runtime_inputs/pf_canonical_kernel_families_v2').glob('*/family.json'):
        theta = int(json.loads((f.parent/'mechanical_configuration.json').read_text())['theta_deg'])
        if theta not in (15,30):continue
        validation = validate_family(f)
        assert validation['maximum_extension_um'] >= 100
        families[str(theta)] = dict(path=str(f),sha256=sha256(f),validation=validation,
            files={str(p):sha256(p) for p in f.parent.iterdir() if p.is_file()})
    assert set(families) == {'15','30'}
    legacy = OLD/'tip_local_recovery'
    archive = legacy/'V13_FINAL_16_CASE_REVIEW.zip'
    assert sha256(archive) == '9ed156a525f561f898af654d5b4e2fadfd8e310ea0d450eb804685fbe7af3d26'
    cases = json.loads((legacy/'final_16_case_record/case_table.json').read_text())['rows']
    files = {str(archive):sha256(archive)}
    for c in cases:
        for p in authoritative(c['case']).rglob('*'):
            if p.is_file():files[str(p)] = sha256(p)
    for p in (legacy/'final_16_case_record').iterdir():
        if p.is_file():files[str(p)] = sha256(p)
    atomic_json(OUT/'accepted_ensemble_freeze.json',dict(record_commit='f09ec8468d4d95df73385742611f3de44455f865',
        classification='STOCHASTIC_BUT_SATURATION_DOMINATED',boundary=BOUNDARY,files=files,
        per_case_execution_commits={c['case']:c['source_commit'] for c in cases},
        corrected_execution_source_commit='01011a4fa90888db9a8d10b8b7f47847748d92a5',
        final_report_producer_code_commit='9b63ad71d8902d8dbd96c2f81cd57d0efc569482',
        provenance_supersession='The historical frozen diagnostic used an uncommitted patch on 651d32e. That patch was subsequently qualified and committed as 01011a4; all three corrected physical launches used 01011a4. The final record is f09ec84. The old diagnostic phrase is historical, not the final execution provenance.'))
    # Only states with actual complete canonical event payloads and qualified
    # persistent bindings are reusable for frozen marked-race evaluation.
    reuse = []
    for p in (ROOT/'analysis_outputs/v13_physical_companion_qualification/physical_parents').glob('*/parent/parent_record.json'):
        case=p.parents[1].name
        if case not in GROUPS:continue
        r=json.loads(p.read_text())
        context=p.parent/'event_context.pkl'
        assert r['fresh_initialization'] and not r['historical_process_state_used']
        assert sha256(context)==r['event_context_sha256']
        reuse.append(dict(case=case,theta_deg=40,context=str(context),sha256=sha256(context),role='first_cleavage_control'))
    for p in (ROOT/'analysis_outputs/v13_clock_and_pair_mechanism/later_parents').glob('*/events/*/parent/parent_record.json'):
        r=json.loads(p.read_text());context=p.parent/'event_context.pkl'
        assert r['clean_history'] and not r['branching_enabled'] and sha256(context)==r['event_context_sha256']
        reuse.append(dict(case=p.parents[3].name,theta_deg=40,context=str(context),sha256=sha256(context),role='later_sparse_control'))
    atomic_json(OUT/'reusable_state_inventory.json',dict(qualified_states=reuse,
        excluded='Older scalar canonical tables are not complete canonical-event payloads. Historical restored V12 atlas bindings failed the intended-persistent-source audit; no state or threshold substitution.',
        no_theta40_trajectory_rerun=True))
    plan=dict(schema='v13.transition-screen/1',groups=GROUPS,screen_seed=3621,
        orientation_order_deg=[15,30],families=families,rate_multiplier=1.,
        screening_cases=[f'theta{theta}_rate1x_{g}_seed3621' for theta in (15,30) for g in GROUPS],
        target_unbranched_um=75.,stop_on_first_predicted_admissible_branch=True,
        new_committed_branches_in_screen=False,maximum_workers=2,maximum_intervals=2000,
        no_new_theta40_seeds=True,no_family_build=True,no_branch_parameter_changes=True,
        rate_second_stage='Only if no useful orientation: test 0.1x and 10x at the nonzero orientation closest to mixed screening outcomes; separately sealed before launch.',
        mixed_condition_gate='At least one predicted first branch and one 75-um no-predicted-branch parent across four groups; finite chi on both sides of 1, at least one nonsaturated channel, and an admissible pair for positive predictions. Existing early gates do not count as 75-um negatives.',
        pair_boundary_preference='Report margin/release and margin/cost. Prefer smaller positive normalized pair margins; no new physical threshold.',
        followup_gate='Only after useful transition: eight CRN seeds x four groups, first committed branch +20 um daughter growth /75 um without branch /existing legitimate gate.',
        boundary=BOUNDARY)
    atomic_json(PLAN,plan)
    print('Sealed plan;',len(files),'immutable files;',len(reuse),'reusable frozen theta40 states')


def verify_frozen():
    r=json.loads((OUT/'accepted_ensemble_freeze.json').read_text())
    changed=[p for p,h in r['files'].items() if sha256(Path(p))!=h]
    if changed:raise RuntimeError('accepted ensemble changed: '+repr(changed))
    return len(r['files'])


def reuse_controls():
    """Existing complete later diagnostics reused; first controls evaluated once."""
    inventory=json.loads((OUT/'reusable_state_inventory.json').read_text())['qualified_states']
    destination=OUT/'reused_theta40_controls';destination.mkdir()
    family=Path(json.loads((OLD/'short_ensemble_plan.json').read_text())['family'])
    os.environ.update(campaign_environment(family))
    result=[]
    for item in inventory:
        context=Path(item['context'])
        assert sha256(context)==item['sha256']
        if item['role']=='later_sparse_control':
            event=context.parents[1].name
            source=OLD/item['case']/event/'race.json'
            r=json.loads(source.read_text())
            assert r['source_context_sha256']==item['sha256']
            ti,tj=r['T_i_next_s'],r['T_j_s']
            record=dict(r,chi_B=tj/min(ti,r['tau_c_s']),
                primary_effective_lambda_tau_c=r['primary']['effective_rate_per_s']*r['tau_c_s'],
                companion_effective_lambda_tau_c=r['companion']['effective_rate_per_s']*r['tau_c_s'],
                source_record_sha256=sha256(source),new_mechanics=False)
        else:
            payload=pickle.loads(context.read_bytes())
            folder=destination/item['case'];folder.mkdir()
            record=diagnose(payload,folder)
            record['new_frozen_diagnostics_only']=True
        record.update(case=item['case'],theta_deg=40,source_context_sha256=item['sha256'],role=item['role'])
        result.append(record)
    atomic_json(destination/'controls.json',dict(rows=result,new_theta40_trajectories=0,
        archived_later_records_reused=8,new_first_parent_frozen_evaluations=4,immutable_ensemble_files=verify_frozen()))


def accepted_live(checkpoint):
    for manifest in Path(checkpoint.provider_runtime.cache_root).glob('*/manifest.json'):
        m=json.loads(manifest.read_text())
        if m['topology_fingerprint']!=checkpoint.provider_runtime.routing.topology_fingerprint:continue
        p=manifest.parent/'provider_state.pkl'
        assert sha256(p)==m['state_sha256']
        live=pickle.loads(p.read_bytes())
        if np.array_equal(live['base_equilibrium']['displacement'],checkpoint.state.displacement):
            return live
    raise RuntimeError('no hash-matched exact accepted provider state')


def diagnose(payload,folder):
    from scripts.v13_frozen_support import initialized_engine
    from arrhenius_fracture.primary_race_production_v13 import evaluate_production_mark
    from types import SimpleNamespace
    cp=payload['accepted_single_checkpoint']
    before=fp(cp)
    selected=next(t for t in payload['canonical_result'].trials if t.selected)
    isolated=replace(cp,provider_runtime=replace(cp.provider_runtime,cache_root=str(folder/'diagnostic_cache')))
    with initialized_engine(cp.shared_process_state) as engine:
        outcome=evaluate_production_mark(checkpoint=isolated,selected=selected,
            solved_pre_event=payload['solved_pre_event_state'],engine=engine,
            args=SimpleNamespace(**payload['args']),cfg=payload['configuration'],context=payload['context'],
            accepted_live=accepted_live(cp),evaluate_expired_diagnostics=True)
    r=dict(outcome.record)
    ti,tj=r.get('T_i_next_s'),r.get('T_j_s')
    denominator=min(ti,r['tau_c_s']) if ti is not None else None
    chi=tj/denominator if denominator is not None and denominator>0 and tj is not None else None
    r.update(chi_B=chi if chi is None or math.isfinite(chi) else None,
        T_primary_next_status='UNAVAILABLE' if ti is None else 'FINITE' if math.isfinite(ti) else 'INFINITE',
        T_companion_status='UNAVAILABLE' if tj is None else 'FINITE' if math.isfinite(tj) else 'INFINITE',
        chi_B_status='FINITE' if chi is not None and math.isfinite(chi) else 'INFINITE' if chi is not None else 'UNAVAILABLE_INADMISSIBLE_OBSERVATION_OR_ZERO_RACE_BOUND',
        primary_effective_lambda_tau_c=r['primary']['effective_rate_per_s']*r['tau_c_s'] if 'primary' in r else None,
        companion_effective_lambda_tau_c=r['companion']['effective_rate_per_s']*r['tau_c_s'] if 'companion' in r else None,
        accumulated_clock_inventory=[asdict(h) for h in cp.state.competition.hazard_states],
        process_state_fingerprint=fp(cp.shared_process_state),parent_state_fingerprint=before,
        accepted_event_count=len(cp.state.competition.consumed_event_ids),
        predicted_branch=outcome.record['outcome']=='PAIR_ACCEPTED',committed_branch=False,
        exact_pair_admissible=r.get('exact_pair_admissible'),pair_margin_J_per_m=r.get('pair_margin_J_per_m'),
        parent_unchanged=fp(cp)==before)
    assert r['parent_unchanged']
    # Strict JSON: infinity is a distinct status, never a fabricated finite time.
    def portable(v):
        if isinstance(v,float) and not math.isfinite(v):return None
        if isinstance(v,dict):return {k:portable(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [portable(x) for x in v]
        return v
    return portable(r)


class Capture:
    def __init__(self,folder):self.folder=folder;self.count=0;self.predicted=False
    def begin(self,state,engine,physical_time,opening,runtime):
        from arrhenius_fracture.current_source_runtime_bindings import callable_id
        from arrhenius_fracture.persistent_site_source_v10221 import _persistent_emit
        assert callable_id(engine.mpz._emit)==callable_id(_persistent_emit)
        assert len(state.crack_network.active_tip_ids)==1 and not state.crack_network.branching_enabled
    def accept(self,*,checkpoint,context,result,solved_pre_event,pre_event_sigma,args,cfg):
        atomic_json(self.folder/'progress.json',dict(step=context.step,extension_um=checkpoint.projected_extension_m*1e6,
            time_s=checkpoint.physical_time_s,accepted_events=len(checkpoint.state.competition.consumed_event_ids)))
        selected=[t for t in result.trials if t.selected]
        if not selected:return False
        assert len(selected)==1 and selected[0].proposal.action_type=='one_arm'
        self.count+=1
        folder=self.folder/'opportunities'/f'event{self.count:05d}';folder.mkdir(parents=True)
        payload=dict(accepted_single_checkpoint=checkpoint,context=context,canonical_result=result,
            solved_pre_event_state=solved_pre_event,pre_event_sigma=pre_event_sigma,args=vars(args),configuration=cfg)
        raw=pickle.dumps(payload,protocol=5)
        (folder/'event_context.pkl').write_bytes(raw)
        r=diagnose(payload,folder)
        r['event_context_sha256']=hashlib.sha256(raw).hexdigest()
        atomic_json(folder/'race.json',r)
        self.predicted=r['predicted_branch']
        return self.predicted


def worker(case):
    plan=json.loads(PLAN.read_text())
    if case not in plan['screening_cases']:raise RuntimeError('unregistered screen case')
    folder=OUT/'parents'/case;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'launch_claim.json').open('x') as stream:json.dump(dict(pid=os.getpid(),case=case),stream)
    theta,rate,material,temp,seed=case.split('_')
    theta=int(theta[5:]);T=int(temp[:-1]);seed=int(seed[4:])
    family=Path(plan['families'][str(theta)]['path'])
    for p,h in plan['families'][str(theta)]['files'].items():
        if sha256(Path(p))!=h:raise RuntimeError('family or validation artifact changed')
    manifest=json.loads((family.parent/'direct_kernel_validation_manifest.json').read_text())
    if manifest['family_sha256']!=sha256(family) or manifest['passed'] is not True:
        raise RuntimeError('final direct validation does not qualify this exact family')
    env=campaign_environment(family)
    env.update(CLEAVAGE_HAZARD_SEED=str(seed),MPLCONFIGDIR='/tmp/pf-current-source-v13-parent-mpl')
    os.environ.update(env)
    for k in ('V11_BRANCH_RESTART_CHECKPOINT','PF_V5_4_1_RESTORE_ONLY_SENTINEL_OUT','PF_QUALIFIED_DAUGHTER_STOP_UM','KERNEL_CACHE_ROOT'):
        os.environ.pop(k,None)
    canonical,alias=ROWS[material]
    argv=common_arguments(alias,T,family,folder)
    for flag,value in (('--steps','2000'),('--target-crack-extension-um','75'),('--crystal-theta-deg',str(theta))):
        argv[argv.index(flag)+1]=value
    argv+=['--maximum-fronts','1']
    from arrhenius_fracture import sharp_front_v11_branching as production,sharp_front_v10_2_27 as paper
    paper.DEFAULT_REGISTRY=ROOT/'runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_registry.csv'
    paper.SELECTION_RECORD=ROOT/'runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_selection.json'
    paper.VALID_OPTIONS={a:c for c,a in ROWS.values()}
    capture=Capture(folder)
    production.run_2d=partial(production.run_2d,parent_capture=capture)
    row,h=registry_row(canonical)
    atomic_json(folder/'launch.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        arguments=argv,material_row=row,material_row_sha256=h,case=case,seed=seed,
        family_sha256=sha256(family),plan_sha256=sha256(PLAN),fresh_parent=True,branch_disabled=True))
    try:
        production.main(argv)
        m=json.loads((folder/'checkpoint/latest.json').read_text())
        atomic_json(folder/'terminal.json',dict(status='SCREEN_TERMINATED',
            reason='FIRST_PREDICTED_BRANCH_STATE_CAPTURED' if capture.predicted else m['termination_reason'],
            predicted_branch=capture.predicted,committed_branches=0,extension_um=m['projected_extension_m']*1e6,
            state_sha256=m['state_sha256'],accepted_events=capture.count,boundary=BOUNDARY))
    except Exception as exc:
        known=any(s in str(exc) for s in ('opening_scale_not_resolved_above_probe_uncertainty',
            'directional adaptive stepping reached its minimum fraction','family endpoint','outside qualified',
            'Newton failed','equilibrium did not converge','nonpositive_opening'))
        atomic_json(folder/'terminal.json',dict(status='EXISTING_GATE_STOP' if known else 'SOFTWARE_OR_UNCLASSIFIED_STOP',
            reason=str(exc),traceback=traceback.format_exc(),predicted_branch=False,committed_branches=0))
        if not known:raise


def queue():
    plan=json.loads(PLAN.read_text());active={};pending=list(plan['screening_cases'])
    with (OUT/'queue_claim.json').open('x') as stream:json.dump(dict(pid=os.getpid(),maximum_workers=2),stream)
    if any((OUT/'parents'/c/'launch_claim.json').exists() for c in pending):raise RuntimeError('no duplicate parent/retry')
    while active or pending:
        while pending and len(active)<2:
            if shutil.disk_usage(OUT).free < 1024**3:raise RuntimeError('less than 1 GiB durable space')
            c=pending.pop(0);folder=OUT/'parents'/c;folder.mkdir(parents=True)
            log=(folder/'worker.log').open('w')
            env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONHASHSEED='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            active[c]=(subprocess.Popen([sys.executable,__file__,'--case',c],stdout=log,stderr=subprocess.STDOUT,env=env),log)
        for c,(p,log) in list(active.items()):
            if p.poll() is not None:
                log.close();del active[c]
                t=OUT/'parents'/c/'terminal.json'
                if not t.exists() or json.loads(t.read_text())['status']=='SOFTWARE_OR_UNCLASSIFIED_STOP':
                    atomic_json(OUT/'queue_pause.json',dict(case=c,pending=pending));pending=[]
        atomic_json(OUT/'queue_status.json',dict(active={c:p.pid for c,(p,_) in active.items()},pending=pending))
        if active:time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--case');p.add_argument('--queue',action='store_true');p.add_argument('--reuse-controls',action='store_true')
    args=p.parse_args()
    prepare() if args.prepare else worker(args.case) if args.case else queue() if args.queue else reuse_controls() if args.reuse_controls else print(verify_frozen())
