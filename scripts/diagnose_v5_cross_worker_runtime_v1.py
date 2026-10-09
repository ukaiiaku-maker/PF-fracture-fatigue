#!/usr/bin/env python3
"""Read-only dispatch intervention on trusted retained checkpoint evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def inspect(root):
    from validate_v5_source_resolution_evidence_v1 import verify_inventory
    from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
    from arrhenius_fracture.closure_lifecycle_evidence import stagewise_topology
    from arrhenius_fracture.closure_mechanics_evidence import canonical_data
    from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
    from v5_numerical_runtime_v1 import runtime_record
    import numpy as np
    verify_inventory(root);verify_inventory(root/'evidence')
    row=json.loads((root/'evidence/lifecycle_rows.json').read_text())['rows'][0]
    state=restore_checkpoint(root/'evidence'/row['terminal_checkpoint'])
    topology=canonical_data(stagewise_topology(state))
    # Small fixed algebra sentinel does not stand in for a production solve.
    x=np.arange(4096,dtype=float).reshape(64,64)/4096
    algebra=x.T@x+np.eye(64)
    solution=np.linalg.solve(algebra,np.arange(64,dtype=float))
    return {'record_kind':'READ_ONLY_RUNTIME_INTERVENTION_NOT_PHYSICAL_QUALIFICATION',
        'runtime':runtime_record(),'checkpoint_fingerprint':complete_accepted_state_fingerprint(state),
        'case_identity':row['case_identity'],'topology':topology,
        'recorded_topology':row['stagewise_topology'],
        'topology_exact':topology==row['stagewise_topology'],
        'topology_sha256':hashlib.sha256(json.dumps(topology,sort_keys=True).encode()).hexdigest(),
        'algebra_sentinel_sha256':hashlib.sha256(solution.tobytes()).hexdigest()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path);parser.add_argument('--child',action='store_true');args=parser.parse_args()
    if args.child:result=inspect(args.source)
    else:
        from numpy._core._multiarray_umath import __cpu_dispatch__,__cpu_baseline__
        from v5_numerical_runtime_v1 import policy_environment
        policy=policy_environment(__cpu_dispatch__,__cpu_baseline__);results={}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        for mode in ('native','haswell-only','numpy-baseline-only','fixed-policy','fixed-policy-repeat'):
            environment=dict(os.environ)
            for name in ('OPENBLAS_CORETYPE','NPY_DISABLE_CPU_FEATURES'):environment.pop(name,None)
            if mode in ('haswell-only','fixed-policy','fixed-policy-repeat'):environment['OPENBLAS_CORETYPE']=policy['OPENBLAS_CORETYPE']
            if mode in ('numpy-baseline-only','fixed-policy','fixed-policy-repeat'):environment['NPY_DISABLE_CPU_FEATURES']=policy['NPY_DISABLE_CPU_FEATURES']
            path=args.output.with_name(mode+'.json')
            subprocess.run([sys.executable,__file__,str(args.source),str(path),'--child'],env=environment,check=True)
            results[mode]=json.loads(path.read_text())
        from v5_numerical_runtime_v1 import require_pinned
        require_pinned(results['fixed-policy']['runtime'])
        if results['fixed-policy']!=results['fixed-policy-repeat']:raise ValueError('fixed runtime repeated reconstruction differs')
        result={'schema':'v5.cross-worker-runtime-intervention/1','results':results,
            'fixed_policy_repeat_exact':True,'scientific_qualification':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print('Recorded '+str(args.output),flush=True)
