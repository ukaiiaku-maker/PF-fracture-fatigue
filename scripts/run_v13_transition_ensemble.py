"""Eight CRN seeds x four groups, launchable only after the screen gate."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from scripts.run_v13_transition_screen import OUT,GROUPS,PLAN,BOUNDARY
from scripts.run_pf_current_source_multifront_field_atlas_v12 import (
    ROOT,ROWS,campaign_environment,common_arguments,registry_row,atomic_json,sha256)
from scripts.run_v13_primary_race_short_ensemble import terminal_record

DEST=OUT/'transition_ensemble'
ENSEMBLE_PLAN=OUT/'transition_ensemble_plan.json'


def prepare(gate_path,condition):
    gate=json.loads(gate_path.read_text())
    if condition not in gate['useful_conditions'] or not gate['followup_ensemble_authorized_by_gate']:
        raise RuntimeError('condition has not passed the transition screen')
    if condition.startswith('theta40_'):raise RuntimeError('no additional theta40 ensemble is authorized')
    if ENSEMBLE_PLAN.exists():raise RuntimeError('ensemble already preregistered')
    theta=int(condition.split('_')[0][5:])
    families=json.loads(PLAN.read_text())['families']
    rate=float(condition.split('_')[1][4:-1].replace('p','.'))
    seeds=list(range(3621,3629))
    atomic_json(ENSEMBLE_PLAN,dict(schema='v13.transition-ensemble/1',condition=condition,theta_deg=theta,
        rate_multiplier=rate,seeds=seeds,groups=GROUPS,maximum_workers=2,
        cases=[f'{condition}_{g}_seed{s}' for s in seeds for g in GROUPS],
        family=families[str(theta)],source_gate=str(gate_path.resolve()),source_gate_sha256=sha256(gate_path),
        right_censor_threshold_um=75.,daughter_growth_after_first_branch_um=20.,
        maximum_intervals=2000,no_empirical_branch_probability=True,no_second_branch=True,
        no_new_material_rows=True,boundary=BOUNDARY,
        interpretation='Observed first-branch incidence/extension, not recursive spacing. Early gates are separately censored, not 75-um nonbranching successes.'))


class UnbranchedBoundary:
    """Stop at first accepted endpoint >=75 um only while still single-front.

    Never shorten a physical cleavage event or change the accepted endpoint.
    Both the prespecified 75-um censor and the discrete endpoint are retained.
    """
    censored=False
    def begin(self,*args):pass
    def accept(self,*,checkpoint,**kwargs):
        if not any(b.generation>0 for b in checkpoint.state.crack_network.branches) and checkpoint.projected_extension_m>=75e-6:
            self.censored=True
        return self.censored


def worker(case):
    p=json.loads(ENSEMBLE_PLAN.read_text())
    if case not in p['cases']:raise RuntimeError('unregistered followup case')
    gate=Path(p['source_gate'])
    if sha256(gate)!=p['source_gate_sha256']:raise RuntimeError('screen gate changed')
    folder=DEST/case;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'launch_claim.json').open('x') as stream:json.dump(dict(case=case,pid=os.getpid()),stream)
    theta,rate,material,temp,seed=case.split('_');T=int(temp[:-1]);seed=int(seed[4:])
    family=Path(p['family']['path'])
    for path,h in p['family']['files'].items():
        if sha256(Path(path))!=h:raise RuntimeError('family changed')
    env=campaign_environment(family)
    env.update(CLEAVAGE_HAZARD_SEED=str(seed),PF_QUALIFIED_DAUGHTER_STOP_UM='25',
        MPLCONFIGDIR='/tmp/pf-current-source-v13-parent-mpl')
    os.environ.update(env)
    for key in ('V11_BRANCH_RESTART_CHECKPOINT','PF_V5_4_1_RESTORE_ONLY_SENTINEL_OUT','KERNEL_CACHE_ROOT'):
        os.environ.pop(key,None)
    canonical,alias=ROWS[material]
    argv=common_arguments(alias,T,family,folder)
    # 125 um is only a nonbinding overall guard once a branch has occurred.
    # The observer enforces the separate 75-um unbranched censor.
    for flag,value in (('--steps',str(p['maximum_intervals'])),('--target-crack-extension-um','125'),
                       ('--dt',str(8.4/p['rate_multiplier'])),('--crystal-theta-deg',str(p['theta_deg']))):
        argv[argv.index(flag)+1]=value
    argv+=['--maximum-fronts','1','--v13-inherited-primary-race']
    from arrhenius_fracture import sharp_front_v11_branching as production,sharp_front_v10_2_27 as paper
    paper.DEFAULT_REGISTRY=ROOT/'runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_registry.csv'
    paper.SELECTION_RECORD=ROOT/'runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_selection.json'
    paper.VALID_OPTIONS={a:c for c,a in ROWS.values()}
    boundary=UnbranchedBoundary()
    production.run_2d=partial(production.run_2d,parent_capture=boundary)
    row,h=registry_row(canonical)
    atomic_json(folder/'launch.json',dict(case=case,seed=seed,material_row=row,material_row_sha256=h,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        arguments=argv,plan_sha256=sha256(ENSEMBLE_PLAN),family_sha256=sha256(family),
        fresh_initialization=True,common_random_numbers_across_groups=True,boundary=BOUNDARY))
    try:
        production.main(argv)
        m=json.loads((folder/'checkpoint/latest.json').read_text())
        reason='75_um_without_committed_branch' if boundary.censored else m['termination_reason']
        terminal_record(folder,reason,'TERMINATED')
    except Exception as exc:
        known=any(s in str(exc) for s in ('opening_scale_not_resolved_above_probe_uncertainty',
            'directional adaptive stepping reached its minimum fraction','family endpoint','outside qualified',
            'Newton failed','equilibrium did not converge','nonpositive_opening'))
        atomic_json(folder/'exception.json',dict(reason=str(exc),traceback=traceback.format_exc()))
        terminal_record(folder,str(exc),'EXISTING_GATE_STOP' if known else 'SOFTWARE_OR_UNCLASSIFIED_STOP')
        if not known:raise


def queue():
    p=json.loads(ENSEMBLE_PLAN.read_text());DEST.mkdir()
    with (DEST/'queue_claim.json').open('x') as stream:json.dump(dict(pid=os.getpid(),maximum_workers=2),stream)
    active={};pending=list(p['cases'])
    while pending or active:
        while pending and len(active)<2:
            if shutil.disk_usage(DEST).free<1024**3:raise RuntimeError('less than 1 GiB free; no next case')
            c=pending.pop(0);folder=DEST/c;folder.mkdir()
            log=(folder/'worker.log').open('w')
            env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONHASHSEED='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            active[c]=(subprocess.Popen([sys.executable,__file__,'--case',c],stdout=log,stderr=subprocess.STDOUT,env=env),log)
        for c,(process,log) in list(active.items()):
            if process.poll() is not None:
                log.close();del active[c]
                t=DEST/c/'terminal.json'
                if not t.exists() or json.loads(t.read_text())['status']=='SOFTWARE_OR_UNCLASSIFIED_STOP':
                    atomic_json(DEST/'queue_pause.json',dict(case=c,pending=pending));pending=[]
        atomic_json(DEST/'queue_status.json',dict(active={c:r.pid for c,(r,_) in active.items()},pending=pending))
        if active:time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',type=Path);p.add_argument('--condition');p.add_argument('--case');p.add_argument('--queue',action='store_true')
    a=p.parse_args()
    prepare(a.prepare,a.condition) if a.prepare else worker(a.case) if a.case else queue() if a.queue else None
