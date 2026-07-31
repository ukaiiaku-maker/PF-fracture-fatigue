from __future__ import annotations

import importlib.util
from pathlib import Path


def test_v1041_campaign_requires_detailed_balance_audit():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "build_v10_4_bulk_rate_orientation_launcher.py"
    spec = importlib.util.spec_from_file_location("v1041_builder_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = (
        root / "scripts" / "run_v10_2_28_paper_four_class_theta30_1000um.sh"
    ).read_text()
    generated = module.transform(source)

    assert '"bulk_net_slip_model": "detailed_balance_forward_minus_reverse"' in generated
    assert '"zero_stress_net_plastic_rate_exactly_zero": True' in generated
    assert '"v10_4_0_outputs_physics_compatible": False' in generated
    assert "v10_4_1_bulk_detailed_balance_audit.json" in generated
    assert 'detailed_balance_audit.get("new_fitted_parameters") != 0' in generated
    assert (
        'bulk_model_audit.get("zero_stress_net_plastic_rate_exactly_zero") '
        'is not True'
    ) in generated
