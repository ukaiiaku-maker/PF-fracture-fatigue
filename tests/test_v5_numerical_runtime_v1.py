import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from v5_numerical_runtime_v1 import policy_environment,require_pinned


def record():
    return {'machine':'x86_64','numpy_dispatch':['X86_V3','X86_V4'],
        'numpy_baseline':['X86_V2'],'enabled_dispatch':[],
        'environment':policy_environment(['X86_V4','X86_V3'],['X86_V2']),
        'blas_libraries':[{'internal_api':'openblas','num_threads':1,'architecture':'Haswell'}]}


def test_policy_does_not_disable_required_baseline():
    assert policy_environment(['X86_V4','X86_V3','X86_V2'],['X86_V2'])['NPY_DISABLE_CPU_FEATURES']=='X86_V3,X86_V4'
    assert require_pinned(record())==record()


@pytest.mark.parametrize('change',['kernel','threads','dispatch','environment','architecture','missing'])
def test_runtime_fails_closed_on_actual_kernel_or_environment_mismatch(change):
    value=copy.deepcopy(record())
    if change=='kernel':value['blas_libraries'][0]['architecture']='Zen'
    if change=='threads':value['blas_libraries'][0]['num_threads']=2
    if change=='dispatch':value['enabled_dispatch']=['X86_V3']
    if change=='environment':value['environment']['OPENBLAS_CORETYPE']='ZEN'
    if change=='architecture':value['machine']='arm64'
    if change=='missing':value['blas_libraries']=[]
    with pytest.raises(ValueError):require_pinned(value)
