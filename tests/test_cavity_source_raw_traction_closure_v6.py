import json
from pathlib import Path

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
