from types import SimpleNamespace

import numpy as np

from arrhenius_fracture.campaign_calibrated_tip import CampaignCalibratedTipEngine
from arrhenius_fracture.physical_shielding_v1022 import (
    install_uncapped_physical_shielding,
    physical_shielding_audit_payload,
    reset_physical_shielding_audit,
)


def _fake_engine(raw: float, legacy_cap_MPa: float = 1.0):
    engine = CampaignCalibratedTipEngine.__new__(CampaignCalibratedTipEngine)
    engine.manifest = SimpleNamespace(max_K_shield_MPa_sqrt_m=legacy_cap_MPa)
    engine.mpz = SimpleNamespace(
        site_capacity=np.array([10.0, 10.0]),
        available_sites=np.array([7.0, 8.0]),
        campaign_source_last_refresh_length_m=2.0e-6,
        campaign_source_last_refresh_fraction=0.2,
    )
    engine._active_shielding_raw_uncapped = lambda: float(raw)
    return engine


def test_uncapped_field_can_exceed_legacy_manifest_reference_without_method_patch():
    engine = _fake_engine(2.5e6, legacy_cap_MPa=1.0)
    original = CampaignCalibratedTipEngine._active_shielding_signed
    with install_uncapped_physical_shielding():
        assert CampaignCalibratedTipEngine._active_shielding_signed is original
        assert engine._active_shielding_signed() == 2.5e6
    assert CampaignCalibratedTipEngine._active_shielding_signed is original


def test_diagnostics_record_raw_equal_effective_without_clip():
    reset_physical_shielding_audit()
    engine = _fake_engine(-3.0e6, legacy_cap_MPa=1.0)
    original = CampaignCalibratedTipEngine._active_shielding_signed
    with install_uncapped_physical_shielding():
        assert CampaignCalibratedTipEngine._active_shielding_signed is original
        diagnostics = engine._campaign_diagnostics()
        assert diagnostics["campaign_active_K_shield_raw_Pa_sqrt_m"] == -3.0e6
        assert diagnostics["campaign_active_K_shield_effective_Pa_sqrt_m"] == -3.0e6
        assert diagnostics["campaign_active_K_shield_cap_Pa_sqrt_m"] == 0.0
        assert diagnostics["campaign_legacy_K_shield_cap_reference_Pa_sqrt_m"] == 1.0e6
        assert diagnostics["campaign_shielding_cap_applied"] is False
        assert diagnostics["campaign_diagnostic_context_constitutive_patch"] is False

    audit = physical_shielding_audit_payload()
    assert audit["constitutive_K_shield_clip_applied"] is False
    assert audit["diagnostic_context_modifies_constitutive_method"] is False
    assert audit["n_samples_above_legacy_cap_reference"] == 1
    assert audit["maximum_abs_raw_minus_effective_Pa_sqrt_m"] == 0.0
    assert audit["maximum_raw_to_legacy_cap_ratio"] == 3.0


def test_no_new_fitted_saturation_parameter_is_introduced():
    reset_physical_shielding_audit()
    audit = physical_shielding_audit_payload()
    assert audit["new_fitted_saturation_parameter_introduced"] is False
    assert "finite_crack_tip_source_capacity" in audit["population_saturation_controls"]
    assert "Taylor_backstress_reduces_emission_rate" in audit["population_saturation_controls"]
