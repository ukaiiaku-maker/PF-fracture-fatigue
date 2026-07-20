from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path("scripts/build_v10_2_14_campaign_ready_active_only_atlas_v2.py")
    spec = importlib.util.spec_from_file_location("endpoint_builder_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_endpoint_active_projection_is_campaign_ready_without_loo_cv():
    module = _module()
    payload = {
        "measured_station_projection": {
            "piecewise_linear_spatial_projection": True,
            "active_kernel_mechanically_measured": True,
            "wake_kernel_mechanically_measured": False,
            "wake_shielding_supported": False,
            "projected_schema_matches_measured_schema": True,
            "subelement_rows_claimed_as_direct_fem": False,
            "spatial_cross_validation_passed": False,
            "projection_checks": [
                {
                    "state_id": "E000",
                    "region": "active",
                    "system": 0,
                    "measured_bins": [0, 199],
                    "full_grid_count": 200,
                    "cross_validation_available": False,
                },
                {
                    "state_id": "E000",
                    "region": "wake",
                    "system": 0,
                    "measured_bins": [],
                    "full_grid_count": 200,
                    "cross_validation_required": False,
                    "wake_kernel_forced_zero": True,
                },
            ],
        }
    }
    ready, details = module.exact_endpoint_projection_ready(payload)
    assert ready is True
    assert details["all_active_curves_have_exact_endpoint_coverage"] is True
    assert "spatial_projection_cross_validation_passed" not in module.BASE.MECHANICAL_GATES
    assert module.ENDPOINT_GATE in module.BASE.MECHANICAL_GATES


def test_endpoint_gate_rejects_missing_last_endpoint():
    module = _module()
    payload = {
        "measured_station_projection": {
            "piecewise_linear_spatial_projection": True,
            "active_kernel_mechanically_measured": True,
            "wake_kernel_mechanically_measured": False,
            "wake_shielding_supported": False,
            "projected_schema_matches_measured_schema": True,
            "subelement_rows_claimed_as_direct_fem": False,
            "projection_checks": [
                {
                    "region": "active",
                    "system": 0,
                    "measured_bins": [0, 198],
                    "full_grid_count": 200,
                }
            ],
        }
    }
    ready, _ = module.exact_endpoint_projection_ready(payload)
    assert ready is False


def test_existing_2d_overnight_launcher_does_not_use_endpoint_builder():
    text = Path("scripts/run_v10_2_15_stage3_overnight.sh").read_text()
    assert "build_v10_2_14_campaign_ready_active_only_atlas_v2.py" not in text
    assert "SIGNED_KERNEL_FAMILY_JSON" not in text
    assert "arrhenius_fracture.sharp_front_v10_1_7_5" in text
