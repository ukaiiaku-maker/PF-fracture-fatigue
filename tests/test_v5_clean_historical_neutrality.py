from copy import deepcopy
import numpy as np

from scripts.qualify_v5_disabled_neutrality_clean import compare_case,normalize,CASES,STAGE_II_V3_BASE


def sample():
    return {"case_id":"monotonic","input_configuration":CASES["monotonic"],
        "terminal_components":{"sharp_wake_model_id":"sharp_wake_mechanically_separating_v12",
            "displacement":normalize(np.array([1.,2.])),"support":{"source_commit":"a"}},
        "model_identity":"sharp_wake_mechanically_separating_v12",
        "complete_fingerprint":"a","failure":None,"restart_exact":True}


def test_neutrality_requires_v12_identity_not_v11_twice():
    base=sample();current=deepcopy(base)
    assert compare_case(base,current)["passed"]
    base["model_identity"]=current["model_identity"]="sharp_wake_causal_v11"
    assert not compare_case(base,current)["passed"]


def test_neutrality_does_not_strip_provenance_or_inactive_schema_difference():
    base=sample();current=deepcopy(base)
    current["terminal_components"]["support"]["source_commit"]="b"
    current["terminal_components"]["void_state"]=None
    result=compare_case(base,current)
    assert not result["passed"]
    assert {row["path"] for row in result["full_component_differences"]} == {
        "/support/source_commit","/void_state"}


def test_complete_arrays_preserve_shape_dtype_and_every_element():
    a=np.array([1.,2.]);b=np.array([1.,3.])
    assert normalize(a)!=normalize(b)
    assert normalize(a)!=normalize(a.reshape(1,2))
    assert normalize(a)!=normalize(a.astype(np.float32))


def test_registry_is_four_physical_cases_and_authoritative_corrected_base():
    assert set(CASES)=={"monotonic","fixed_mesh_oblique","checkpoint_restart","unload_reload"}
    assert STAGE_II_V3_BASE=="326e3f5973ef623781ab8568798c495a3f68238c"
