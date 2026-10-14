import pytest
from pathlib import Path

from scripts.qualify_generic_fem_convergence import qualify


def test_retained_three_level_fem_response_is_reported_not_tuned():
    root = (Path(__file__).resolve().parents[1]
            / "common_framework_readiness/generic_fem_three_level_a015a36")
    receipt = qualify([root / f"level_{index}" for index in range(1, 4)], 0.02)
    assert receipt["status"] == "FAIL_INVALID_REFINEMENT_AXIS"
    assert receipt["no_fitted_scale_factor"] is True
    assert receipt["maximum_medium_to_fine_relative_change"] == pytest.approx(
        0.1281599292814948
    )
    levels = receipt["levels"]
    assert [level["node_count"] for level in levels] == [1493, 1539, 1868]
    assert max(level["common_to_native_J_relative_difference"] for level in levels) < 0.002
    axis = receipt["refinement_axis"]
    assert axis["quantity"] == "selected path-envelope support spacing"
    assert axis["values_m"] == pytest.approx([12.25e-6, 12.25e-6, 11.025e-6])
    assert axis["strictly_refined"] is False
    assert axis["requested_tip_h_is_not_a_proxy_for_envelope_support"] is True
    assert receipt["common_observer"] == {
        "provenance": "algebraic_KJ_squared_over_Eprime",
        "independent_field_evaluation": False,
        "validation_role": "unit conversion only",
    }
