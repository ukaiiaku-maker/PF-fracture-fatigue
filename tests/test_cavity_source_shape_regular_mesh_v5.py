import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v4_mesh_failure_audit_is_geometry_only_and_locates_crack_support_cause():
    record = json.loads(
        (ROOT / "artifacts/v5_cavity_source_recovery_v5/v4_mesh_failure_audit.json").read_text()
    )
    assert record["new_fem_solve_run"] is False
    assert record["cause_decision"]["classification"] == "C_CRACK_SUPPORT_CAVITY_INTERACTION"
    evidence = record["cause_decision"]["evidence"]
    assert evidence["N128_base_mesh_quality_above_acceptance"] is True
    assert evidence["N128_quality_drops_below_acceptance_after_fixed_crack_path_insertion"] is True
    assert evidence["N128_worst_element_location"] == "crack_support_region"
    assert evidence["retained_connected_mesh_collapses_further_in_same_construction_path"] is True
    assert len(record["levels"]) == 3
    assert all(len(level["lowest_quality_elements"]) == 20 for level in record["levels"])
    retained = record["retained_v4_interpretation"]
    assert retained["V4_SOURCE_GEOMETRY"] == "PASS"
    assert retained["V4_POINT_SOURCE_FORMULATION"] == "REMAINS_VIABLE"
    assert retained["FINITE_ACTIVATION_ZONE_REQUIRED"] == "NOT_ESTABLISHED"
    assert retained["V4_SOURCE_TENSOR_RELATIVE_CHANGE_64_TO_128"] == 0.006818275586047867
    assert "not an independent raw adjacent-element traction" in retained[
        "zero_recovered_boundary_traction_interpretation"
    ]
