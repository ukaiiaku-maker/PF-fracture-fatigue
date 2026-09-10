from copy import deepcopy

from arrhenius_fracture.static_numerical_family_v2 import build_failure_atlas


def retained_stub():
    row = {
        "input_configuration": {
            "boundary_segments": 128, "cavity_center_m": [0.7, 0.0],
            "cavity_radius_m": 0.05, "specimen_width_m": 1.0,
            "specimen_height_m": 1.0, "crack_path_m": [[0.0, 0.0], [0.5, 0.0]],
            "crack_enabled": True, "cavity_enabled": True, "opening_m": 0.1,
        },
        "measurements": {
            "mesh_quality": 0.1, "reaction": 1.0, "compliance": 2.0, "energy": 3.0,
            "cavity_fields": {"normalized_traction": 0.01},
            "fixed_tip_probe": {"tensor_Pa": [[1.0, 0.0], [0.0, 1.0]]},
        },
        "recovery": [{"arc_fraction": 0.0, "recovery": {"tensor_Pa": [[1.0, 0.0], [0.0, 1.0]]}}],
    }
    rows = {}
    ids = []
    for level in (128, 256, 512):
        key = str(level); ids.append(key); rows[key] = deepcopy(row)
        rows[key]["input_configuration"]["boundary_segments"] = level
    family = {
        "family": "centered:{mesh}", "source_ids": ids, "passed": False,
        "gates": {"two_quality_valid_fine_levels": True, "fixed_tip_fine_accuracy": False},
        "convergence": {"fixed_tip_tensor": [0.1, 0.08]},
    }
    return {"rows": rows, "decision": {"families": [family], "derivatives": []}}


def test_atlas_preserves_sequences_and_identifies_first_failure():
    result = build_failure_atlas(retained_stub())
    row = result["families"][0]
    assert row["mesh_levels"] == [128, 256, 512]
    assert row["reaction_sequence_N_per_m"] == [1.0, 1.0, 1.0]
    assert row["first_failed_predicate"] == "fixed_tip_fine_accuracy"
    assert row["failure_classification"] == "FIXED_TIP_RECOVERY"
    assert row["geometry_match_across_nominal_refinement"]


def test_retained_rows_cannot_be_reclassified_as_v2_execution():
    result = build_failure_atlas(retained_stub())
    assert result["STATIC_FAMILY_V1"] == {"passed": 0, "total": 1, "derivatives_passed": 0, "derivatives_total": 0}
    assert result["STATIC_FAMILY_V2"] == "NOT_RUN_SENTINELS_NOT_YET_STABLE"
    assert not result["full_v2_matrix_authorized"]
