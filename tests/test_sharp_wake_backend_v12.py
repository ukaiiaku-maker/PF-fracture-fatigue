import pytest
from arrhenius_fracture.sharp_wake_backend_v12 import *

def sample():
    return V12SharpWakeSupportState("g","c",2,"graph",1.25,"arcs",(2,5),"damage",("tip-1",),"tx-2","tx-1","abc")

def test_default_remains_v11_and_selection_is_explicit():
    assert select_sharp_wake_model()==V11_MODEL_ID
    assert select_sharp_wake_model(V12_MODEL_ID)==V12_MODEL_ID
    with pytest.raises(ValueError): select_sharp_wake_model("v12")

def test_remesh_transfer_excludes_stale_element_and_mesh_ownership():
    transfer=sample().provenance_for_remesh()
    assert "selected_support_elements" not in transfer
    assert "mesh_geometry_fingerprint" not in transfer
    assert "mesh_connectivity_fingerprint" not in transfer
    assert transfer["previous_accepted_transaction"]=="tx-2"

def test_support_ids_fail_closed():
    values=sample().__dict__.copy(); values["selected_support_elements"]=(2,2)
    with pytest.raises(ValueError): V12SharpWakeSupportState(**values)
