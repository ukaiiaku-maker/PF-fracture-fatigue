#!/usr/bin/env python3
"""One independently executed final-campaign phase or disjoint lifecycle shard."""
import argparse
import hashlib
from importlib.metadata import version,PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def run_phase(phase,output,section='all',shard_index=0,shard_count=1,base_worktree=None):
    if output.exists():raise ValueError('refusing to overwrite phase execution')
    def git(*args):return subprocess.check_output(('git',*args),cwd=ROOT,text=True).strip()
    sha=git('rev-parse','HEAD')
    if git('status','--porcelain'):raise ValueError('phase requires a clean exact-head worker')
    output.mkdir(parents=True);steps=[]
    def run(label,script,*args):
        command=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)]
        print('Actual phase operation '+label,flush=True)
        with (output/(label+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        steps.append({'operation':label,'script':script,'returncode':result.returncode})
        return result.returncode==0
    if phase=='static':
        run('static','qualify_v5_static_numerical_families_v1.py',output/'evidence',
            '--shard-index',shard_index,'--shard-count',shard_count)
    elif phase=='lifecycle':
        run('lifecycle','run_voiding_v5_closure_lifecycle.py',output/'evidence','--qualified-fine-history',
            '--section',section,'--shard-index',shard_index,'--shard-count',shard_count)
    elif phase=='source':
        generated=run('matched_production','qualify_voiding_v5_production_transfer.py',output/'production','--compress-checkpoints')
        if generated:
            run('production_ontology','validate_voiding_v5_closure_bundle.py',output/'production')
            run('production_topology','qualify_voiding_v5_closure_production_topology.py',output/'production',output/'production_topology.json')
            positive=run('real_source_pair','qualify_voiding_v5_source_positive_v1.py',output/'production',output/'positive')
            if positive:
                run('positive_ontology','validate_v5_source_resolution_evidence_v1.py','positive',output/'positive')
                if run('source_causality','qualify_v5_child_source_causality_v1.py',output/'positive',output/'causality'):
                    run('source_causality_ontology','validate_v5_source_resolution_evidence_v1.py','causality',output/'causality')
    elif phase=='recovery':
        run('patch_operator','qualify_cavity_boundary_patch_recovery_v1.py',output/'operator.json')
        run('kirsch_coarse','qualify_cavity_patch_kirsch_fem_v1.py',output/'kirsch_coarse')
        run('kirsch_fine','qualify_cavity_patch_kirsch_fem_v1.py',output/'kirsch_fine','--fine-extension')
        run('recovery_transfer','qualify_cavity_recovery_transfer_budget_v1.py',output/'transfer.json')
    elif phase=='causal-neutrality':
        if base_worktree is None:raise ValueError('historical clean base is required')
        run('causal_neutrality','qualify_v5_disabled_causal_neutrality_v1.py',output/'evidence','--base-worktree',base_worktree)
    else:raise ValueError('unknown phase')
    clean=not git('status','--porcelain') and git('rev-parse','HEAD')==sha
    packages={}
    for name in ('numpy','scipy','triangle','gmsh','numba','llvmlite'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    report={'schema':'v5.source-resolution-final-phase-execution/1','phase':phase,'section':section,
        'shard_index':shard_index,'shard_count':shard_count,'executed_code_sha':sha,
        'python_version':platform.python_version(),'operations':steps,'clean_exact_head_at_end':clean,
        'numerical_package_versions':packages,
        'numerical_environment':{name:os.environ.get(name) for name in
            ('PYTHONHASHSEED','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        'execution_completed':clean and bool(steps) and all(step['returncode']==0 for step in steps),
        'scientific_pass':'SEPARATELY_RECOMPUTED_NOT_INFERRED_FROM_PROCESS_EXIT'}
    (output/'execution.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (output/'sha256_manifest.json').write_text(json.dumps({str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob('*')) if p.is_file()},sort_keys=True,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('static','lifecycle','source','recovery','causal-neutrality'))
    parser.add_argument('output',type=Path);parser.add_argument('--section',default='all')
    parser.add_argument('--shard-index',type=int,default=0);parser.add_argument('--shard-count',type=int,default=1)
    parser.add_argument('--base-worktree',type=Path);args=parser.parse_args()
    result=run_phase(args.phase,args.output,args.section,args.shard_index,args.shard_count,args.base_worktree)
    print(json.dumps(result),flush=True);sys.exit(0 if result['execution_completed'] else 1)
