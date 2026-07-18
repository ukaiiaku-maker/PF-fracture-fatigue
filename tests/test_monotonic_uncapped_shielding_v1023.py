from types import SimpleNamespace

import numpy as np

from arrhenius_fracture.campaign_calibrated_tip import CampaignCalibratedTipEngine


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
    """The non-fatigue path must use the permanently uncapped shared method."""
    engine = _fake_engine(raw=4.0e6, legacy_cap_MPa=0.9)
    assert engine._active_shielding_signed() == 4.0e6
    diagnostics = engine._campaign_diagnostics()
    assert diagnostics["campaign_active_K_shield_effective_Pa_sqrt_m"] == 4.0e6
    assert diagnostics["campaign_active_K_shield_cap_Pa_sqrt_m"] == 0.0
    assert diagnostics["campaign_legacy_K_shield_cap_reference_Pa_sqrt_m"] == 0.9e6
    assert diagnostics["campaign_shielding_cap_applied"] is False
    assert diagnostics["campaign_shielding_population_limited"] is True
