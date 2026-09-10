"""Fresh physical jobs from the frozen transfer cohort; at most three workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/prospective_paris_candidates'
RUN=ROOT/'runs/prospective_paris_transfer_v1'
PY='/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python'
FAMILY='/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_2_28_kernel_cache/4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/family.json'
BRANCH='codex/v10.2.30-prospective-paris-candidate-design'


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run_job(job,head,freeze_sha):
    if git('rev-parse','HEAD')!=head or git('status','--porcelain'):
        raise RuntimeError('source changed while physical jobs active')
    if shutil.disk_usage(ROOT).free < 3*2**30:raise RuntimeError('insufficient disk')
    parent=RUN/job['candidate_id'];parent.mkdir(parents=True,exist_ok=True)
    key=job['job_key']
    for attempt in range(100):
        out=parent/f'{key}__attempt{attempt}'
        record=parent/f'{key}__attempt{attempt}__launch.json'
        if not out.exists() and not record.exists():break
    else:raise RuntimeError('attempt limit exceeded')
    rec=dict(job,attempt=attempt,result_path=str(out),launch_head=head,freeze_sha256=freeze_sha,
             result_path_virgin_at_launch=True,resume=False,launch_time_unix=time.time())
    with record.open('x') as f:json.dump(rec,f,indent=2)
    env=os.environ.copy()
    if env.get('V10230_RESTART_CHECKPOINT_DIR'):raise RuntimeError('resume environment forbidden')
    env.update(PYTHON_BIN=PY,CONDA_ENV='arrhenius-sharp-front-v10-codex',CONDA_DEFAULT_ENV='arrhenius-sharp-front-v10-codex',
        EXPECTED_BRANCH=BRANCH,EXPECTED_HEAD=head,FAMILY_JSON=FAMILY,
        V10230_ENTRY_MODULE='arrhenius_fracture.sharp_front_v10_2_30_prospective_transfer_fixed_deltaK',
        PARAMETER_OPTION=job['candidate_id'],TARGET_DELTAK=str((1-job['R'])*job['Kmax']),R_RATIO=str(job['R']),
        TARGET_EXT_UM='100',CYCLES_MAX='1000000000000',HAZARD_SEED=str(job['seed']),MAX_WALL_SECONDS='43200',
        RUN_LABEL=key,OUTROOT=str(out),TARGET_FRACTION='prospective_transfer')
    t0=time.monotonic()
    with (parent/f'{out.name}__stdout.log').open('x') as stdout,(parent/f'{out.name}__stderr.log').open('x') as stderr:
        p=subprocess.run(['bash','scripts/run_v10_2_30_weakt_high_cycle_1e12.sh'],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
    rec.update(exit_code=p.returncode,wall_seconds=time.monotonic()-t0,completion_time_unix=time.time())
    summary=out/'developed_fatigue_growth_summary.json'
    rec['status']='NUMERICALLY_UNRESOLVED'
    if p.returncode==0 and summary.is_file():
        d=json.loads(summary.read_text())
        rec.update(status='PHYSICAL_TARGET_REACHED' if d['target_reached'] else 'NON_TARGET_REQUIRES_AUDIT',
                   developed_rate=d['developed_interval']['da_dN'],event_count=d['event_count'],final_extension_um=d['final_projected_extension_um'])
    (parent/f'{out.name}__result.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(rec),flush=True)
    return rec


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['PILOT','GRID','SEED2','R_TRANSFER']);parser.add_argument('--candidate');args=parser.parse_args()
    if git('branch','--show-current')!=BRANCH or git('status','--porcelain'):raise SystemExit('clean campaign branch required')
    head=git('rev-parse','HEAD');freeze=ART/'transfer_candidate_freeze_v1.json'
    process_lines=subprocess.check_output(['ps','-axo','pid,command'],text=True).splitlines()
    conflicting=[line for line in process_lines if ' -m arrhenius_fracture.sharp_front_v10_2_30_' in line]
    if conflicting:raise SystemExit('physical workers already active: '+str(conflicting))
    if args.stage!='PILOT':
        decisions=ART/'transfer_stage_decisions.json'
        assert subprocess.check_output(['git','show','HEAD:artifacts/prospective_paris_candidates/transfer_stage_decisions.json'],cwd=ROOT)==decisions.read_bytes()
        admitted=json.loads(decisions.read_text())['admitted_stages']
        if not args.candidate or args.stage not in admitted.get(args.candidate,[]):
            raise SystemExit('stage has not passed the preceding committed evidence gate')
    assert subprocess.check_output(['git','show','HEAD:artifacts/prospective_paris_candidates/transfer_candidate_freeze_v1.json'],cwd=ROOT)==freeze.read_bytes()
    data=json.loads(freeze.read_text())
    for name,expected in data['launch_source_hashes'].items():
        if sha(ROOT/name)!=expected:raise SystemExit('frozen launch source changed: '+name)
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(ROOT))
    from arrhenius_fracture.prospective_paris_transfer_engine_v10230 import build_transfer_manifest
    for row in data['candidates']:build_transfer_manifest(row['candidate_id'])
    jobs=[j for j in data['jobs'] if j['stage']==args.stage and (not args.candidate or j['candidate_id']==args.candidate)]
    if not jobs:raise SystemExit('no frozen jobs match')
    for j in jobs:
        if list((RUN/j['candidate_id']).glob(j['job_key']+'__attempt*__result.json')):
            raise SystemExit('job already attempted: audit result and authorize a new explicit attempt before rerunning')
    print(json.dumps(dict(head=head,stage=args.stage,jobs=len(jobs),max_workers=3)),flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run_job,j,head,sha(freeze)) for j in jobs]
        for future in as_completed(futures):future.result()


if __name__=='__main__':main()
