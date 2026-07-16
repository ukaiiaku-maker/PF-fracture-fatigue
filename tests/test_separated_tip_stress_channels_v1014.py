from types import SimpleNamespace

import pytest

from arrhenius_fracture import (
    ContinuumSourceKineticTipEngine,
    SeparatedSourceKineticTipEngine,
)
from arrhenius_fracture.kinetic_tip_cell import KineticTipConfig
from arrhenius_fracture.material_manifest import MaterialManifest, default_manifest_path
from arrhenius_fracture.unified_mpz import MPZConfig


def _engine():
    manifest = MaterialManifest.from_csv(default_manifest_path("weakT"))
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
        fcfg,
        None,
        None,
        80.0e9,
        0.3,
        2.5e-10,
        manifest,
        MPZConfig(length_m=20.0e-6, n_bins=40, wake_length_m=20.0e-6),
    )


def test_public_continuum_engine_routes_to_separated_channels():
    assert ContinuumSourceKineticTipEngine is SeparatedSourceKineticTipEngine
    eng = _engine()
    assert eng.separated_tip_stress_channels


def test_plastic_half_step_uses_opening_stress_when_cleavage_is_fully_shielded():
    eng = _engine()
    K = 20.0e6
    opening = eng.sigma_opening_tip(K)
    eng.K_shield = lambda: 2.0 * K
    assert eng.sigma_tip(K) == pytest.approx(0.0)
    assert opening > 0.0

    captured = {}

    def fake_evolve(dt, T, stress, b):
        captured["stress"] = float(stress)
        return {
            "dN_emit": 0.0,
            "dN_trapped": 0.0,
            "dN_released": 0.0,
            "dN_recovered": 0.0,
            "dN_escaped": 0.0,
            "source_sites_refreshed": 0.0,
        }

    eng.mpz.evolve = fake_evolve
    eng._separated_current_K_Pa_sqrt_m = K
    result = eng._plastic_half_step(1.0, 700.0, 0.0)

    assert captured["stress"] == pytest.approx(opening)
    assert result["sigma_opening_tip_Pa"] == pytest.approx(opening)
    assert result["sigma_cleave_input_Pa"] == pytest.approx(0.0)


def test_step_reports_opening_and_cleavage_stresses_separately():
    eng = _engine()
    K = 20.0e6
    eng.K_shield = lambda: 2.0 * K

    eng._integrate_coupled = lambda *args, **kwargs: {
        "fired": False,
        "n_fire": 0,
        "v_crack": 0.0,
        "dB": 0.0,
        "da": 0.0,
        "dt_consumed": 1.0,
        "dt_unused": 0.0,
        "packet_mean": 0.0,
        "packet_variance_m2": 0.0,
        "lambda_c": 0.0,
        "lambda_c_raw": 0.0,
        "Gc_J": 0.0,
        "sigma_tip": 0.0,
        "plastic": {},
        "advance": {},
        "microsteps": 1,
    }

    result = eng.step(K, 700.0, 1.0)

    assert result["sigma_tip"] == pytest.approx(eng.sigma_opening_tip(K))
    assert result["sigma_opening_tip_Pa"] > 0.0
    assert result["sigma_cleave_eff_Pa"] == pytest.approx(0.0)
    assert result["stress_channels_separated"] is True
