from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.campaign_calibrated_tip import (
    CampaignCalibratedTipEngine,
    SOURCE_MODEL,
    _campaign_local_density_m2,
)
from arrhenius_fracture.kinetic_tip_cell import KineticTipConfig
from arrhenius_fracture.material_manifest import MaterialManifest, default_manifest_path
from arrhenius_fracture.unified_mpz import MPZConfig


def _engine(backstress_scale=1.0, refresh_scale=1.0):
    manifest = MaterialManifest.from_csv(default_manifest_path("weakT"))
    fcfg = SimpleNamespace(
        r0=1.0e-6,
        sigma_cap=0.0,
        m_hits=1.0,
        tau_c=1.0e-6,
        L_pz=1.0e-6,
        da=5.0e-6,
    )
    CampaignCalibratedTipEngine.configure_default(KineticTipConfig())
    CampaignCalibratedTipEngine.configure_campaign(backstress_scale, refresh_scale)
    return CampaignCalibratedTipEngine(
        fcfg,
        None,
        None,
        80.0e9,
        0.3,
        2.5e-10,
        manifest,
        MPZConfig(length_m=20.0e-6, n_bins=40, wake_length_m=20.0e-6),
    )


def test_campaign_model_uses_promoted_bounded_source_budget():
    eng = _engine(backstress_scale=0.0)
    state = eng.mpz
    expected = state.n_systems * state.manifest.source_sites_per_system
    assert state.source_model == SOURCE_MODEL
    assert np.sum(state.site_capacity) == pytest.approx(expected)
    assert np.sum(state.available_sites) == pytest.approx(expected)
    assert state.campaign_source_budget_remaining_total == pytest.approx(expected)


def test_stationary_tip_cannot_emit_more_than_promoted_budget():
    eng = _engine(backstress_scale=0.0)
    state = eng.mpz
    state.emission_rate_per_site = lambda stress, T: 1.0e30
    budget = float(np.sum(state.site_capacity))

    emitted_1 = state._emit(1.0, 5.0e9, 700.0)
    emitted_2 = state._emit(1.0, 5.0e9, 700.0)

    assert emitted_1 == pytest.approx(budget)
    assert emitted_2 == pytest.approx(0.0)
    assert np.sum(state.available_sites) == pytest.approx(0.0)
    assert state.continuum_source_last_clear_rate_s == pytest.approx(0.0)


def test_crack_advance_refreshes_budget_over_promoted_length_only():
    eng = _engine(backstress_scale=0.0, refresh_scale=2.0)
    state = eng.mpz
    state.available_sites[:] = 0.0
    state.tip_source_activity[:] = 0.0
    d = 1.0e-6

    result = state.advance(d)
    L = max(state.manifest.source_refresh_length_m * 2.0, state.dx)
    expected_fraction = 1.0 - np.exp(-d / L)

    assert result["campaign_source_refresh_fraction"] == pytest.approx(expected_fraction)
    assert result["campaign_source_refresh_length_m"] == pytest.approx(L)
    assert np.allclose(state.available_sites, state.site_capacity * expected_fraction)


def test_backstress_density_kernel_does_not_dilute_with_blunted_radius():
    eng = _engine()
    state = eng.mpz
    state.mobile[:, :2] = 25.0
    rho_1 = _campaign_local_density_m2(state)
    state._continuum_tip_radius_m = 1.0e-2
    rho_2 = _campaign_local_density_m2(state)
    assert np.allclose(rho_1, rho_2)
    assert np.all(rho_1 > 0.0)


def test_active_cleavage_shielding_is_bounded_by_promoted_manifest_value():
    eng = _engine()
    eng.mpz.mobile[:, :2] = 1.0e9
    raw = eng._active_shielding_raw_uncapped()
    effective = eng._active_shielding_signed()
    cap = eng.manifest.max_K_shield_MPa_sqrt_m * 1.0e6

    assert abs(raw) > cap
    assert abs(effective) == pytest.approx(cap)


def test_campaign_diagnostics_identify_no_temporal_recycling():
    eng = _engine(backstress_scale=1.25, refresh_scale=0.75)
    diag = eng._campaign_diagnostics()
    assert diag["campaign_source_model"] == SOURCE_MODEL
    assert diag["campaign_backstress_scale"] == pytest.approx(1.25)
    assert diag["campaign_refresh_length_scale"] == pytest.approx(0.75)
    assert diag["campaign_temporal_source_recycling"] is False
