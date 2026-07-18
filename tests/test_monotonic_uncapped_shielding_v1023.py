import json
from types import SimpleNamespace

import numpy as np

from arrhenius_fracture.campaign_calibrated_tip import CampaignCalibratedTipEngine
from arrhenius_fracture.sharp_front_v10_1_5 import _rewrite_mode_audits


def _fake_engine(raw: float, legacy_cap_MPa: float):
    engine = CampaignCalibratedTipEngine.__new__(CampaignCalibratedTipEngine)
    engine.manifest = SimpleNamespace(max_K_shield_MPa_sqrt_m=legacy_cap_MPa)
    engine.mpz = SimpleNamespace(
        site_capacity=np.array([8.0, 8.0]),
        available_sites=np.array([5.0, 6.0]),
        campaign_source_last_refresh_length_m=1.5e-6,
        campaign_source_last_refresh_fraction=0.1,
    )
    engine._active_shielding_raw_uncapped = lambda: float(raw)
    return engine


def test_monotonic_engine_passes_field_above_legacy_reference():
    engine = _fake_engine(raw=4.0e6, legacy_cap_MPa=0.9)
    assert engine._active_shielding_signed() == 4.0e6
    diagnostics = engine._campaign_diagnostics()
    assert diagnostics["campaign_active_K_shield_effective_Pa_sqrt_m"] == 4.0e6
    assert diagnostics["campaign_active_K_shield_cap_Pa_sqrt_m"] == 0.0
    assert diagnostics["campaign_shielding_cap_applied"] is False


def test_monotonic_audit_files_report_shared_uncapped_core(tmp_path):
    (tmp_path / "v10_1_driver_modes.json").write_text(json.dumps({"schema": "old"}))
    (tmp_path / "v10_1_1_source_model.json").write_text(json.dumps({"schema": "old"}))

    _rewrite_mode_audits(["--out", str(tmp_path)])

    modes = json.loads((tmp_path / "v10_1_driver_modes.json").read_text())
    source = json.loads((tmp_path / "v10_1_1_source_model.json").read_text())
    assert modes["manifest_K_shield_cap_enabled"] is False
    assert modes["legacy_manifest_K_shield_cap_reference_only"] is True
    assert modes["shared_monotonic_and_fatigue_core"] is True
    assert source["cleavage_shielding_bound"] == "none; signed raw elastic dislocation field"
    assert source["legacy_manifest_K_shield_cap_reference_only"] is True
    assert source["shared_monotonic_and_fatigue_core"] is True
