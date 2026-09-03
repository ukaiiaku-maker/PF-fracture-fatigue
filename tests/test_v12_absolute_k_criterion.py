from arrhenius_fracture.absolute_k_criterion_v12 import (
    CONTOURS_M, LIMITS, MESH_LEVELS_M, WILLIAMS_TERMS,
    all_limits_pass, classify_absolute_k,
)


def test_contours_are_geometry_admissible_before_results():
    for h in MESH_LEVELS_M:
        for inner, outer in CONTOURS_M:
            assert inner / h >= LIMITS["minimum_r_inner_over_h_tip"]
            assert (outer - inner) / inner <= LIMITS["maximum_q_support_width_over_r_inner"]
            assert 300e-6 - outer >= LIMITS["minimum_root_exterior_patch_clearance_m"]


def test_williams_basis_contains_singular_modes_and_regular_term():
    assert WILLIAMS_TERMS == ("K_I", "K_II", "T_STRESS_SIGMA_XX")


def test_missing_evidence_fails_closed():
    assert not all_limits_pass({"a": True}, ("a", "b"))
    result = classify_absolute_k(
        conforming_pass=True, primal_pass=True, corridor_v3_pass=True,
        standard_pass=False, energy_pass=False, williams_pass=False,
        production_consumes_absolute_k=False,
    )
    assert result.aggregate == "NOT_QUALIFIED"
    assert not result.production_may_continue


def test_qualified_unavailable_path_depends_on_production_consumption():
    common = dict(conforming_pass=True, primal_pass=True, corridor_v3_pass=True,
                  standard_pass=False, energy_pass=True, williams_pass=True)
    diagnostic = classify_absolute_k(**common, production_consumes_absolute_k=False)
    blocked = classify_absolute_k(**common, production_consumes_absolute_k=True)
    assert diagnostic.aggregate == "QUALIFIED_BY_ENERGY_AND_WILLIAMS_STANDARD_INTEGRAL_UNAVAILABLE"
    assert diagnostic.production_may_continue
    assert not blocked.production_may_continue
