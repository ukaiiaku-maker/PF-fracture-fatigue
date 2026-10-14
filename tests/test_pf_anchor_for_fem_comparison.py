import pytest

from scripts.run_pf_anchor_for_fem_comparison import configured_request, matrix


pytestmark = pytest.mark.external_evidence


def test_pf_anchor_matches_declared_common_condition():
    plan = matrix()
    assert plan["temperature_K"] == 1000.0
    assert plan["crystal_theta_deg"] == 0.0
    assert plan["hazard_seed"] == 8666
    assert plan["nominal_dU_m"] == 2e-7
    assert plan["nominal_dt_s"] == 8.4


def test_pf_anchor_retains_spatial_accommodation_and_short_endpoint():
    options, native = configured_request()
    assert options.mechanics.value == "spatial_pf"
    assert options.output.value == "selected_fields"
    assert "--tip-plasticity" in native
    assert "--active-shielding" in native
    assert native[native.index("--temperatures") + 1] == "1000"
    assert native[native.index("--target-crack-extension-um") + 1] == "10"
