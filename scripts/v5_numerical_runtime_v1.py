#!/usr/bin/env python3
"""Prospective x86 CI runtime policy; never changes scientific predicates.

Bootstrap writes variables for NEW processes. Importing NumPy then setting its
dispatch variables in the same process would not configure its kernels.
"""
import argparse
import json
import os
from pathlib import Path
import platform

ENVIRONMENT_KEYS=('PYTHONHASHSEED','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS',
    'MKL_NUM_THREADS','OPENBLAS_CORETYPE','NPY_DISABLE_CPU_FEATURES')


def policy_environment(dispatch,baseline):
    return {'PYTHONHASHSEED':'0','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1',
        'MKL_NUM_THREADS':'1','OPENBLAS_CORETYPE':'HASWELL',
        'NPY_DISABLE_CPU_FEATURES':','.join(sorted(set(dispatch)-set(baseline)))}


def runtime_record():
    import numpy
    import scipy.linalg
    from numpy._core._multiarray_umath import __cpu_dispatch__,__cpu_baseline__,__cpu_features__
    from threadpoolctl import threadpool_info
    # Host names, addresses and library installation paths are provenance, not
    # the selected numerical kernels. Record every loaded BLAS implementation.
    libraries=[{key:row.get(key) for key in ('prefix','internal_api','user_api',
        'version','num_threads','threading_layer','architecture')}
        for row in threadpool_info() if row['user_api']=='blas']
    return {'schema':'v5.pinned-x86-numerical-runtime/1','machine':platform.machine(),
        'numpy_version':numpy.__version__,'scipy_version':scipy.__version__,
        'numpy_baseline':list(__cpu_baseline__),'numpy_dispatch':list(__cpu_dispatch__),
        'enabled_dispatch':[name for name in __cpu_dispatch__ if __cpu_features__.get(name,False)],
        'blas_libraries':sorted(libraries,key=lambda row:json.dumps(row,sort_keys=True)),
        'environment':{name:os.environ.get(name) for name in ENVIRONMENT_KEYS}}


def require_pinned(record):
    expected=policy_environment(record['numpy_dispatch'],record['numpy_baseline'])
    if record['machine']!='x86_64' or record['environment']!=expected or record['enabled_dispatch']:
        raise ValueError('numerical worker is not using the frozen x86 baseline dispatch policy')
    libraries=record['blas_libraries']
    if not libraries or any(row['internal_api']!='openblas' or row['num_threads']!=1
        or str(row['architecture']).lower()!='haswell' for row in libraries):
        raise ValueError('loaded BLAS kernels do not match the frozen single-thread Haswell policy')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--github-env',type=Path);args=parser.parse_args()
    if args.github_env:
        if platform.machine()!='x86_64':raise ValueError('policy is prospectively scoped to x86 CI workers')
        from numpy._core._multiarray_umath import __cpu_dispatch__,__cpu_baseline__
        values=policy_environment(__cpu_dispatch__,__cpu_baseline__)
        with args.github_env.open('a') as output:
            for name,value in values.items():output.write(name+'='+value+'\n')
        print(json.dumps({'configured_for_subsequent_processes':values},sort_keys=True))
    else:print(json.dumps(require_pinned(runtime_record()),sort_keys=True))
