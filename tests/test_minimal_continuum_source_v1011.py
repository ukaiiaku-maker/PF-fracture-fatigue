from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.continuum_source_tip import (
    ContinuumSourceKineticTipEngine,
    SOURCE_MODEL,
    _source_hardening_activity,
)
from arrhenius_fracture.kinetic_tip_cell import KineticTipConfig
from arrhenius_fracture.material_manifest import MaterialManifest, default_manifest_path
from arrhenius_fracture.sharp_front_v10_1 import _prepare_args_v1011
from arrhenius_fracture.unified_mpz import MPZConfig


def _engine() -> ContinuumSourceKineticTipEngine:
    manifest = MaterialManifest.from_csv(default_manifest_path("DBTT"))
    fcfg = SimpleNamespace(
        r0=1.0e-6,
        sigma_cap=0.0,
        m_hits=1.0,
        tau_c=1.0e-6,
        L_pz=1.0e-6,
        da=5.0e-6,
    )
    ContinuumSourceKineticTipEngine.configure_default(KineticTipConfig())
    return ContinuumSourceKineticTipEngine(
        fcfg, None, None, 80.0e9, 0.3, 2.5e-10, manifest,
        MPZConfig(length_m=20.0e-6, n_bins=40, wake_length_m=20.0e-6),
    )


def _constant_transport(state, velocity=1.0e-6):
    n = state.n_bins
    return {
        "peierls": np.ones(n),
        "taylor": np.zeros(n),
        "taylor_single": np.zeros(n),
        "m": np.ones(n),
        "jump": np.ones(n),
        "velocity": np.full(n, velocity),
        "encounter": np.zeros(n),
    }


def test_source_model_is_activity_not_finite_distributed_inventory():
    eng = _engine()
    state = eng.mpz
    assert state.source_model == SOURCE_MODEL
    assert state.tip_source_activity.shape == (state.n_systems,)
    assert state.mobile.shape == (state.n_systems, state.n_bins)
    assert state.retained.shape == (state.n_systems, state.n_bins)
    assert not hasattr(state, "distributed_sources")
    assert not hasattr(state, "source_density")


def test_emission_exhausts_activity_but_not_reference_multiplicity():
    eng = _engine()
    state = eng.mpz
    reference = state.reference_source_multiplicity
    state.emission_rate_per_site = lambda stress, T: 10.0
    state._transport_rates = lambda *args, **kwargs: _constant_transport(state, 0.0)

    emitted = state._emit(1.0, 1.0e9, 700.0)
    assert emitted > 0.0
    assert np.all(state.tip_source_activity < 1.0)
    assert np.all(state.site_capacity == pytest.approx(reference))
    assert state.available_site_fraction < 1.0


def test_peierls_clearing_reactivates_tip_channels():
    eng = _engine()
    state = eng.mpz
    state.tip_source_activity[:] = 0.0
    state.emission_rate_per_site = lambda stress, T: 0.0
    state._transport_rates = lambda *args, **kwargs: _constant_transport(state, 2.0e-6)
    state._continuum_tip_radius_m = 1.0e-6

    state._emit(1.0, 1.0e9, 700.0)
    assert np.all(state.tip_source_activity > 0.8)


def test_crack_advance_recovers_activity_over_current_tip_radius():
    eng = _engine()
    state = eng.mpz
    state.tip_source_activity[:] = 0.0
    state._continuum_tip_radius_m = 1.0e-6

    result = state.advance(1.0e-6)
    expected = 1.0 - np.exp(-1.0)
    assert np.all(state.tip_source_activity == pytest.approx(expected))
    assert result["tip_source_geometry_fraction"] == pytest.approx(expected)
    assert result["tip_source_activity_recovered_geometry"] > 0.0


def test_taylor_storage_suppresses_tip_cycling_without_new_scale():
    eng = _engine()
    state = eng.mpz
    baseline = _source_hardening_activity(state)
    state.retained[:, :2] = 100.0
    hardened = _source_hardening_activity(state)

    assert np.all(baseline == pytest.approx(1.0))
    assert np.all(hardened < baseline)
    assert np.all(hardened > 0.0)


def test_v1011_cli_defaults_to_continuum_and_strips_option():
    args, bulk, jmode, kmode, cfg, source = _prepare_args_v1011([
        "--material-class", "weakT",
        "--tip-source-model", "continuum",
    ])
    assert source == "continuum"
    assert bulk == "tip_only"
    assert jmode == "root_signed"
    assert kmode == "moving_velocity"
    assert cfg.plasticity_enabled
    assert "--tip-source-model" not in args


def test_finite_site_compatibility_mode_remains_explicit():
    _args, _bulk, _jmode, _kmode, _cfg, source = _prepare_args_v1011([
        "--material-class", "weakT",
        "--tip-source-model", "finite_sites",
    ])
    assert source == "finite_sites"
