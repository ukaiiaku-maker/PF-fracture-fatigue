import json
from pathlib import Path
import pytest

from arrhenius_fracture.cavity_source_shape_regular_mesh_v5 import (
    V6_LOCAL_LEVELS, build_shape_regular_source_patch_hole_mesh,
)

ROOT = Path(__file__).resolve().parents[1]


def _hole(level):
    return build_shape_regular_source_patch_hole_mesh(
        1.0e-3, 1.0e-3, (7.0e-4, 0.0), 5.5e-5, 5.0e-5,
        local_level=level, polygon_sectors=128,
    )


def test_v6_contract_freezes_exactly_D_and_E_before_fem_results():
    record = json.loads(
        (ROOT / "artifacts/v6_cavity_source_raw_traction_closure/v6_contract.json").read_text()
    )
    assert record["contract"] == "CAVITY_SOURCE_RAW_TRACTION_CLOSURE_V6"
    assert record["prospectively_frozen_before_D_or_E_fem_evaluation"] is True
    assert [row["level"] for row in record["new_levels"]] == ["D", "E"]
    assert [row["first_strip_radial_subdivisions"] for row in record["new_levels"]] == [16, 32]
    assert record["maximum_new_levels"] == 2
    assert record["additional_level_after_D_or_E"] == "PROHIBITED"
    assert record["acceptance"]["raw_traction_E_max"] == 0.05
    assert record["acceptance"]["strict_raw_traction_order"] == "C>D>E"


def test_v6_D_and_E_preserve_fixed_polygon_boundary_and_quality_target():
    holes = [_hole(level) for level in ("D", "E")]
    assert [hole.validation["first_strip_radial_subdivisions"] for hole in holes] == [16, 32]
    assert len({hole.validation["polygon_geometry_fingerprint"] for hole in holes}) == 1
    assert len({hole.validation["discrete_cavity_boundary_fingerprint"] for hole in holes}) == 1
    assert all(hole.validation["minimum_quality"] >= 0.10 for hole in holes)
    assert set(V6_LOCAL_LEVELS) == {"D", "E"}


def test_v6_contract_preserves_v5_values_and_scope():
    record = json.loads(
        (ROOT / "artifacts/v6_cavity_source_raw_traction_closure/v6_contract.json").read_text()
    )
    assert [record["retained_levels"][key]["raw_traction"] for key in ("A", "B", "C")] == [
        0.14906976345973477, 0.09144336014017264, 0.06715193304848284,
    ]
    assert record["conditional_N256_diagnostic"]["raw_traction"] == 0.038672813574278736
    assert "NOT_A_QUALIFICATION_LEVEL" in record["conditional_N256_diagnostic"]["role"]
    assert all(value is False for value in record["scope"].values())


def _readiness():
    return json.loads(
        (ROOT / "artifacts/v6_cavity_source_raw_traction_closure/central_dbtt_v6_readiness.json").read_text()
    )


def test_production_result_is_blocked_only_when_both_traction_routes_fail():
    record = _readiness()
    assert record["DBTT_SOURCE_READINESS"] == "BLOCKED_WITH_EXACT_V6_FAILURE_CLASS"
    assert record["exact_v6_failure_class"] == ["TRACTION_FREE_BOUNDARY_VERIFICATION"]
    assert record["UNIFIED_2D_ONE_VOID_CODE"] == "BLOCKED_TRACTION_FREE_BOUNDARY_VERIFICATION"
    assert record["RAW_CST_TRACTION"] == "DIAGNOSTIC_FAIL"
    assert record["TRACTION_FREE_BOUNDARY_VERIFICATION"] == "FAIL"
    predicates = record["fixed_geometry_local_family"]["predicates"]
    assert {key for key, passed in predicates.items() if not passed} == {"raw_traction_E"}
    fallback = record["fixed_geometry_local_family"]["fallback_predicates"]
    assert {key for key, passed in fallback.items() if not passed} == {
        "normalized_recovered_boundary_traction",
    }


def test_v6_D_E_report_physical_observables_and_converged_retained_quantities():
    record = _readiness()
    d, e = record["fixed_geometry_local_family"]["new_rows"]
    assert d["local_level"] == "D" and e["local_level"] == "E"
    assert d["raw_adjacent_element_traction_normalized"] == pytest.approx(0.05804714016621954)
    assert e["raw_adjacent_element_traction_normalized"] == pytest.approx(0.055397417629352)
    assert 0.06715193304848284 > d["raw_adjacent_element_traction_normalized"] > e[
        "raw_adjacent_element_traction_normalized"
    ]
    assert d["global_minimum_quality"] >= 0.05
    assert e["global_minimum_quality"] >= 0.05
    assert d["discrete_cavity_boundary_fingerprints"]["complete_boundary"] == (
        e["discrete_cavity_boundary_fingerprints"]["complete_boundary"]
    )
    assert d["equilibrated_boundary_traction_recovery_v1"]["normalized_boundary_traction"] == pytest.approx(0.06630974471574953)
    assert e["equilibrated_boundary_traction_recovery_v1"]["normalized_boundary_traction"] == pytest.approx(0.06850161639893818)
    assert all(all(row["physical_equilibrium_observable_predicates"].values()) for row in (d, e))
    assert all(row["observables"]["reaction_N_per_m"] != 0.0 for row in (d, e))
    assert all(row["observables"]["compliance_m2_per_N"] < 1.0e-9 for row in (d, e))
    comparisons = record["fixed_geometry_local_family"]["D_to_E_comparisons"]
    assert comparisons["source_tensor_relative_D_to_E"] <= 0.05
    assert comparisons["reaction_relative_D_to_E"] <= 0.05
    assert comparisons["compliance_relative_D_to_E"] <= 0.05
    assert comparisons["potential_energy_relative_D_to_E"] <= 0.05
    assert comparisons["ligament_energy_release_relative_D_to_E"] <= 0.05


def test_v6_failure_keeps_oracle_and_downstream_campaigns_closed():
    record = _readiness()
    assert record["oracle_states_accepted"] == 0
    assert record["oracle_generated_in_this_record"] is False
    assert record["paired_trajectories_run"] == 0
    assert record["fatigue_started"] is False
    assert record["next_bounded_step"] == "STOP_SOURCE_QUALIFICATION_PRIMARY_AND_FALLBACK_FAILED"
    assert record["preserved_v5"]["DBTT_SOURCE_READINESS"] == "BLOCKED"
    assert record["preserved_v5"]["FINITE_ACTIVATION_ZONE_REQUIRED"] == "NOT_ESTABLISHED"
    assert record["preserved_v5"]["V5_WEAK_TRACTION_FREE_BOUNDARY"] == "PASS"
    assert record["preserved_v5"]["V5_RAW_ADJACENT_ELEMENT_TRACTION"] == (
        "FAIL_0.0671519_GT_0.05"
    )
    assert all(value is False for value in record["scope"].values())
