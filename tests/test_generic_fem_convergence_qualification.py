import pytest
from pathlib import Path

from scripts.qualify_generic_fem_convergence import qualify


def test_retained_three_level_fem_response_is_reported_not_tuned():
    root = (Path(__file__).resolve().parents[1]
            / "common_framework_readiness/generic_fem_three_level_a015a36")
    receipt = qualify([root / f"level_{index}" for index in range(1, 4)], 0.02)
    assert receipt["status"] == "FAIL_NOT_CONVERGED"
    assert receipt["no_fitted_scale_factor"] is True
    assert receipt["maximum_medium_to_fine_relative_change"] == pytest.approx(
        0.1281599292814948
    )
    levels = receipt["levels"]
    assert [level["node_count"] for level in levels] == [1493, 1539, 1868]
    assert max(level["common_to_native_J_relative_difference"] for level in levels) < 0.002
