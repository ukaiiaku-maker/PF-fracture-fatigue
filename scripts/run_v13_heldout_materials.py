"""Final held-out material block; delegates physics and stopping to frozen V13.

Only output paths and the preregistered material groups differ from the
accepted transition worker. No orientation selection or model adjustment.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts import run_v13_transition_ensemble as frozen_worker
from scripts.run_v13_transition_screen import OUT as ACCEPTED, BOUNDARY, verify_frozen
from scripts.run_pf_current_source_multifront_field_atlas_v12 import ROOT, ROWS, atomic_json, sha256, registry_row

OUT=ROOT/'analysis_outputs/v13_heldout_material_classes'
DEST=OUT/'ensemble'
PLAN=OUT/'heldout_plan.json'
SEAL=OUT/'accepted_record_freeze.json'
PHYSICS='35f4a03c622112cd5bc7cf9bc08c5b7fe2387e46'
RECORD='b2e0e2d85e048553ad2593d895a98974356213e2'
GROUPS=('DBTT_300K','DBTT_1000K','ceramic_300K','ceramic_1000K')
EXPECTED_ROWS={'DBTT':'v913_zeroD_sobol_0202500','ceramic':'oneD_v2_focused_ceramic_like_0018'}
HELPERS=('scripts/run_v13_transition_ensemble.py','scripts/run_v13_transition_screen.py',
    'scripts/run_pf_current_source_multifront_field_atlas_v12.py',
    'scripts/run_v13_primary_race_short_ensemble.py','scripts/v13_value_fingerprint.py')


def build_plan(base):
    p=dict(base)
    assert p['theta_deg']==15 and p['rate_multiplier']==1.
    assert p['seeds']==list(range(3621,3629))
    assert p['right_censor_threshold_um']==75. and p['daughter_growth_after_first_branch_um']==20.
    p.update(schema='v13.heldout-materials/1',groups=list(GROUPS),
        cases=[f"{p['condition']}_{g}_seed{s}" for s in p['seeds'] for g in GROUPS],
        held_out_material_classes=True,parameterizations=EXPECTED_ROWS,
        frozen_physics_commit=PHYSICS,accepted_transition_record_commit=RECORD,
        no_condition_retuning=True,no_peak_or_weakT_reruns=True,
        full_repository_suite_once_after_all_cases=True,stop_simulation_program_after_this_block=True)
    return p


def verify_source():
    p=json.loads(PLAN.read_text())
    for name,h in p['source_and_registry_files'].items():
        if sha256(ROOT/name)!=h:raise RuntimeError('frozen source or registry changed: '+name)
    for material,canonical in EXPECTED_ROWS.items():
        assert ROWS[material][0]==canonical
        row,h=registry_row(canonical)
        assert h==p['material_rows'][material]['sha256'] and row==p['material_rows'][material]['row']
    return p


def verify_accepted():
    seal=json.loads(SEAL.read_text())
    for name,h in seal['files'].items():
        if sha256(Path(name))!=h:raise RuntimeError('accepted transition record changed: '+name)
    return dict(transition_files=len(seal['files']),theta40_files=verify_frozen())


def prepare():
    if OUT.exists():raise RuntimeError('held-out block already prepared; no overwrite')
    assert subprocess.check_output(['git','rev-parse','b2e0e2d'],text=True).strip()==RECORD
    assert json.loads((ACCEPTED/'transition_ensemble/queue_status.json').read_text())==dict(active={},pending=[])
    assert json.loads((ACCEPTED/'final_transition_record/summary.json').read_text())['status']=='COMPLETE'
    subprocess.run(['git','diff','--exit-code',PHYSICS,'--','arrhenius_fracture',*HELPERS],check=True)
    p=build_plan(json.loads((ACCEPTED/'transition_ensemble_plan.json').read_text()))
    source=subprocess.check_output(['git','ls-files','arrhenius_fracture'],text=True).splitlines()+list(HELPERS)
    source+=['runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_registry.csv',
        'runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_selection.json']
    p['source_and_registry_files']={name:sha256(ROOT/name) for name in source if (ROOT/name).is_file()}
    p['material_rows']={}
    for material,canonical in EXPECTED_ROWS.items():
        assert ROWS[material][0]==canonical
        row,h=registry_row(canonical);p['material_rows'][material]=dict(row=row,sha256=h)
    for name,h in p['family']['files'].items():assert sha256(Path(name))==h
    assert sha256(Path(p['source_gate']))==p['source_gate_sha256']
    if shutil.disk_usage(ROOT).free<1024**3:raise RuntimeError('less than 1 GiB durable free space')
    files={str(f):sha256(f) for f in sorted(ACCEPTED.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
    atomic_json(SEAL,dict(accepted_transition_record_commit=RECORD,frozen_physics_commit=PHYSICS,
        accepted_theta40_record_commit='f09ec8468d4d95df73385742611f3de44455f865',files=files,
        archive_sha256=sha256(ACCEPTED/'V13_TRANSITION_REVIEW.zip'),boundary=BOUNDARY))
    atomic_json(PLAN,p)
    print(json.dumps(dict(verification=verify_accepted(),source_files=len(p['source_and_registry_files']),
        cases=len(p['cases']),free_GiB=shutil.disk_usage(ROOT).free/1024**3),indent=2))


def worker(case):
    p=verify_source()
    if case not in p['cases']:raise RuntimeError('not an authorized held-out case')
    # Same function, same parent capture, same material aliases and CLI semantics.
    frozen_worker.DEST=DEST
    frozen_worker.ENSEMBLE_PLAN=PLAN
    frozen_worker.worker(case)


def queue():
    p=verify_source();verify_accepted()
    DEST.mkdir()
    atomic_json(DEST/'queue_claim.json',dict(pid=os.getpid(),maximum_workers=2,
        launcher_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        frozen_physics_commit=PHYSICS))
    run_pending(list(p['cases']))


def continuation_pending(p):
    status=json.loads((DEST/'queue_status.json').read_text())
    pause=json.loads((DEST/'queue_pause.json').read_text())
    record=json.loads((OUT/'HELDOUT_QUEUE_PAUSE.json').read_text())
    if status!={'active':{},'pending':[],'paused':True} or pause.get('reason')!='less than 1 GiB free':
        raise RuntimeError('not a drained storage pause')
    pending=record['unlaunched_cases']
    completed=record['completed_cases']
    if pause['pending']!=pending or [r['case'] for r in completed]+pending!=p['cases']:
        raise RuntimeError('pause history does not partition the preregistered queue')
    for row in completed:
        if sha256(DEST/row['case']/'terminal.json')!=row['terminal_sha256']:
            raise RuntimeError('completed terminal changed: '+row['case'])
    for case in pending:
        if (DEST/case).exists():raise RuntimeError('refusing to relaunch existing path: '+case)
    return list(pending)


def continue_queue():
    p=verify_source();accepted=verify_accepted()
    pending=continuation_pending(p)
    for name,h in p['family']['files'].items():
        if sha256(Path(name))!=h:raise RuntimeError('family changed: '+name)
    if sha256(Path(p['source_gate']))!=p['source_gate_sha256']:raise RuntimeError('source gate changed')
    if shutil.disk_usage(DEST).free<1024**3:raise RuntimeError('less than 1 GiB free')
    claim=dict(pid=os.getpid(),maximum_workers=2,pending=pending,accepted_verification=accepted,
        launcher_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        frozen_physics_commit=PHYSICS,pause_record_sha256=sha256(OUT/'HELDOUT_QUEUE_PAUSE.json'),
        original_queue_pause=json.loads((DEST/'queue_pause.json').read_text()))
    # Exclusive claim prevents two continuations; original claim and pause record remain intact.
    with (DEST/'continuation_claim.json').open('x') as f:json.dump(claim,f,indent=2)
    run_pending(pending)


def run_pending(pending):
    active={};paused=False
    while pending or active:
        while pending and len(active)<2 and not paused:
            if shutil.disk_usage(DEST).free<1024**3:
                atomic_json(DEST/'queue_pause.json',dict(reason='less than 1 GiB free',pending=pending));pending=[];paused=True;break
            c=pending.pop(0);folder=DEST/c;folder.mkdir()
            log=(folder/'worker.log').open('w')
            env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONHASHSEED='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            active[c]=(subprocess.Popen([sys.executable,__file__,'--case',c],stdout=log,stderr=subprocess.STDOUT,env=env),log)
        for c,(process,log) in list(active.items()):
            if process.poll() is not None:
                log.close();del active[c]
                terminal=DEST/c/'terminal.json'
                if not terminal.exists() or json.loads(terminal.read_text())['status']=='SOFTWARE_OR_UNCLASSIFIED_STOP':
                    atomic_json(DEST/'queue_pause.json',dict(case=c,pending=pending));pending=[];paused=True
        atomic_json(DEST/'queue_status.json',dict(active={c:r.pid for c,(r,_) in active.items()},pending=pending,paused=paused))
        if active:time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--queue',action='store_true');parser.add_argument('--continue-queue',action='store_true');parser.add_argument('--case')
    args=parser.parse_args()
    prepare() if args.prepare else queue() if args.queue else continue_queue() if args.continue_queue else worker(args.case) if args.case else print(verify_accepted())
