from scripts.run_independent_seed_pf_pairs import CASES, SEED, case_matrix


def test_seed_matrix_has_exactly_two_matched_pairs():
    assert SEED != 1733
    assert CASES == (
        ("DBTT_1200K", "accommodation_enabled"),
        ("DBTT_1200K", "opening_only"),
        ("Peak_900K", "accommodation_enabled"),
        ("Peak_900K", "opening_only"),
    )


def test_seed_matrix_preserves_spatial_mechanics_and_short_endpoint():
    matrix = case_matrix()
    assert matrix["scope"] == "two-seed pilot variability check only"
    assert not matrix["physical_loading_rate_changed"]
    assert not matrix["numerical_resolution_changed"]
    assert all(row["mechanics"] == "spatial_pf" for row in matrix["cases"])
    assert all(row["target_crack_extension_um"] == 10.0 for row in matrix["cases"])
