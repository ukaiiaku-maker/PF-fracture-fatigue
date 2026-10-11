from scripts.run_final_fracture_campaign import (
    BASELINE_TEMPERATURES,
    configured_case,
    rows,
    validate_matrix,
)


def test_baseline_matrix_is_exact_finite_product():
    audit = validate_matrix(rows())
    assert audit["passed"] is True
    assert audit["baseline_row_count"] == 4 * len(BASELINE_TEMPERATURES) * 2
    assert audit["unique_baseline_identities"] == 96


def test_baseline_case_consumes_common_endpoint_and_material():
    row = next(r for r in rows() if r["case_id"] == "baseline_weak_t_1050K_opening_only")
    options, native = configured_case(row)
    assert native[native.index("--parameter-option") + 1] == "v913_paper_weakT01_0129902_persistent_sites"
    assert native[native.index("--temperatures") + 1] == "1050"
    assert native[native.index("--steps") + 1] == "500"
    assert native[native.index("--target-crack-extension-um") + 1] == "10"
    assert "--no-tip-plasticity" in native
    assert "--no-active-shielding" in native
    assert options.random_seed == 1733
