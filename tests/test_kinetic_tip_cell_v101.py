from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.config import EV_TO_J
from arrhenius_fracture.kinetic_tip_cell import (
    KineticMovingTipFrontEngine,
    KineticTipConfig,
)
from arrhenius_fracture.material_manifest import MaterialManifest, default_manifest_path
from arrhenius_fracture.sharp_front_v10_1 import _prepare_args
from arrhenius_fracture.unified_mpz import MPZConfig


def _engine(cfg: KineticTipConfig) -> KineticMovingTipFrontEngine:
    manifest = MaterialManifest.from_csv(default_manifest_path("weakT"))
    fcfg = SimpleNamespace(
        r0=1.0e-6,
        sigma_cap=0.0,
        m_hits=1.0,
        tau_c=1.0e-6,
        L_pz=1.0e-6,
        da=5.0e-6,
    )
    KineticMovingTipFrontEngine.configure_default(cfg)
    eng = KineticMovingTipFrontEngine(
        fcfg, None, None, 80.0e9, 0.3, 2.5e-10, manifest,
        MPZConfig(length_m=20.0e-6, n_bins=40, wake_length_m=20.0e-6),
    )
    eng.lambda_cleave = lambda sig, T: (2.0, 2.0, 1.0 * EV_TO_J)
    eng.lambda_emit = lambda sig, T: (0.0, sig, 1.0 * EV_TO_J)
    return eng


def test_action_produces_continuous_advance_before_checkpoint():
    eng = _engine(KineticTipConfig(
        plasticity_enabled=False,
        active_shielding=False,
        max_action_substep=0.1,
        max_translation_substep_m=1.0e-6,
    ))
    result = eng.step(0.0, 700.0, 0.1)
    assert not result["fired"]
    assert result["B"] == pytest.approx(0.2)
    assert result["kinetic_micro_advance_step_m"] == pytest.approx(1.0e-6)
    assert eng.mpz.advance_total_m == pytest.approx(1.0e-6)
    assert eng.a_adv == pytest.approx(0.0)


def test_checkpoint_does_not_translate_mpz_twice():
    eng = _engine(KineticTipConfig(
        plasticity_enabled=False,
        active_shielding=False,
        max_action_substep=0.1,
        max_translation_substep_m=1.0e-6,
    ))
    eng.step(0.0, 700.0, 0.1)
    result = eng.step(0.0, 700.0, 0.4)
    assert result["fired"]
    assert result["n_fire"] == 1
    assert eng.n_adv == 1
    assert eng.a_adv == pytest.approx(5.0e-6)
    assert eng.micro_advance_total_m == pytest.approx(5.0e-6)
    assert eng.mpz.advance_total_m == pytest.approx(5.0e-6)
    assert result["B"] == pytest.approx(0.0, abs=1.0e-10)


def test_source_exposure_occurs_during_partial_advance():
    eng = _engine(KineticTipConfig(
        plasticity_enabled=False,
        active_shielding=False,
        max_action_substep=0.1,
        max_translation_substep_m=1.0e-6,
    ))
    eng.mpz.available_sites[:] = 0.0
    before = eng.mpz.available_site_fraction
    result = eng.step(0.0, 700.0, 0.1)
    assert not result["fired"]
    assert before == 0.0
    assert eng.mpz.available_site_fraction > 0.0
    assert result["source_sites_refreshed"] > 0.0


def test_active_mobile_population_has_switchable_signed_shielding():
    on = _engine(KineticTipConfig(
        plasticity_enabled=False,
        active_shielding=True,
        signed_active_shielding=True,
        mobile_shield_fraction=1.0,
    ))
    off = _engine(KineticTipConfig(
        plasticity_enabled=False,
        active_shielding=False,
        signed_active_shielding=True,
        mobile_shield_fraction=1.0,
    ))
    on.mpz.mobile[0, 0] = 1.0
    off.mpz.mobile[0, 0] = 1.0
    assert on.K_shield() > 0.0
    assert off.K_shield() == 0.0


def test_at_most_one_checkpoint_per_outer_state():
    eng = _engine(KineticTipConfig(
        plasticity_enabled=False,
        active_shielding=False,
        max_action_substep=0.05,
        max_translation_substep_m=0.25e-6,
    ))
    eng.lambda_cleave = lambda sig, T: (100.0, 100.0, 1.0 * EV_TO_J)
    result = eng.step(0.0, 700.0, 1.0)
    assert result["fired"]
    assert result["n_fire"] == 1
    assert eng.n_adv == 1
    assert result["kinetic_dt_unused_s"] > 0.9
    assert eng.mpz.advance_total_m == pytest.approx(5.0e-6)


def test_v101_cli_controls_are_removed_before_legacy_parser():
    args, bulk, jmode, kmode, cfg = _prepare_args([
        "--material-class", "weakT",
        "--tip-kinetics-mode", "moving_velocity",
        "--no-active-shielding",
        "--no-tip-plasticity",
        "--mobile-shield-fraction", "0.75",
        "--kinetic-max-action-substep", "0.01",
    ])
    assert bulk == "tip_only"
    assert jmode == "root_signed"
    assert kmode == "moving_velocity"
    assert not cfg.active_shielding
    assert not cfg.plasticity_enabled
    assert cfg.mobile_shield_fraction == pytest.approx(0.75)
    assert cfg.max_action_substep == pytest.approx(0.01)
    assert "--tip-kinetics-mode" not in args
    assert "--no-active-shielding" not in args
    assert "--mobile-shield-fraction" not in args
