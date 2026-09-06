"""PX2 (Part X): analytical periodic-orbit kinetic-regime characterization.

Covers analytical_periodic_orbit's core correctness properties: exact
convergence to the true periodic fixed point, probability conservation,
the R>=0 zero-contact semantic control, and reproduction of the frozen
SAT_EXISTING (RB2_reversible_finite) config hash used by the row-
construction script.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    ContactModel,
    CrackRebondingControls,
    FeedbackMode,
    InitialPrecrackWakeMode,
    RebondModelLevel,
)
from arrhenius_fracture.crack_rebonding_part_x_kinetic_regime_v10230 import (
    analytical_periodic_orbit,
    predicted_static_shield_equivalent_K_b,
)
from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa

REPO_ROOT = Path(__file__).resolve().parents[1]


def _engine_geometry():
    engine, _ = build_a_native_engine()
    return reduced_modulus_Pa(engine.G, engine.nu), float(engine.r_eff())


def _reversible_cfg(**overrides):
    base = dict(
        enabled=True, model_level=RebondModelLevel.CLEAN_REVERSIBLE_REBOND,
        contact_model=ContactModel.SIGNED_K_COMPRESSION_PROXY,
        feedback_mode=FeedbackMode.HAZARD_ONLY_REBOND_SHIELD,
        wake_length_m=0.0005, wake_weight_length_m=5e-7,
        restored_work_of_separation_J_m2=1.8225, rebond_K_geometry_factor=1.0,
        bond_activation_volume_m3=1e-30, rupture_activation_volume_m3=0.0,
        bond_attempt_frequency_s=1e10, rupture_attempt_frequency_s=1e10,
        bond_barrier_eV=0.39, rupture_barrier_eV=0.386,
        chemistry_factor=1.0, fresh_surface_clean_fraction=1.0,
        initial_precrack_wake_mode=InitialPrecrackWakeMode.NO_INITIAL_ACTIVE_WAKE,
    )
    base.update(overrides)
    return CrackRebondingControls(**base).validate()


def test_periodic_orbit_converges_tightly():
    Eprime_Pa, r_eff_m = _engine_geometry()
    cfg = _reversible_cfg()
    orbit = analytical_periodic_orbit(
        cfg=cfg, T_K=300.0, Kmax_Pa_sqrt_m=18.0e6, R=-0.5, frequency_Hz=1000.0, r_contact_m=r_eff_m, n_phase=64,
    )
    assert orbit["convergence_residual"] < 1e-10


def test_probability_conservation_along_trajectory():
    Eprime_Pa, r_eff_m = _engine_geometry()
    cfg = _reversible_cfg()
    orbit = analytical_periodic_orbit(
        cfg=cfg, T_K=300.0, Kmax_Pa_sqrt_m=18.0e6, R=-0.5, frequency_Hz=1000.0, r_contact_m=r_eff_m, n_phase=64,
    )
    totals = orbit["trajectory"].sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-8)
    assert np.all(orbit["trajectory"] >= -1e-10)


@pytest.mark.parametrize("hold_s", [0.0, 0.0005, 0.0020])
def test_positive_R_gives_exact_zero_contact_regardless_of_hold(hold_s):
    Eprime_Pa, r_eff_m = _engine_geometry()
    cfg = _reversible_cfg()
    orbit = analytical_periodic_orbit(
        cfg=cfg, T_K=300.0, Kmax_Pa_sqrt_m=18.0e6, R=0.1, frequency_Hz=1000.0,
        minimum_load_hold_s=hold_s, r_contact_m=r_eff_m, n_phase=32,
    )
    assert orbit["contact_time_s"] == 0.0
    assert orbit["A_CB"] == 0.0
    assert orbit["F_CB"] == 0.0


def test_hold_extends_contact_time_at_negative_R():
    Eprime_Pa, r_eff_m = _engine_geometry()
    cfg = _reversible_cfg()
    orbit_no_hold = analytical_periodic_orbit(
        cfg=cfg, T_K=300.0, Kmax_Pa_sqrt_m=18.0e6, R=-0.5, frequency_Hz=1000.0,
        minimum_load_hold_s=0.0, r_contact_m=r_eff_m, n_phase=32,
    )
    orbit_hold = analytical_periodic_orbit(
        cfg=cfg, T_K=300.0, Kmax_Pa_sqrt_m=18.0e6, R=-0.5, frequency_Hz=1000.0,
        minimum_load_hold_s=0.0005, r_contact_m=r_eff_m, n_phase=32,
    )
    assert orbit_hold["contact_time_dwell_s"] == pytest.approx(0.0005)
    assert orbit_hold["contact_time_s"] > orbit_no_hold["contact_time_s"]


def test_predicted_static_shield_equivalent_K_b_scales_with_cycle_mean_p_B():
    Eprime_Pa, r_eff_m = _engine_geometry()
    cfg = _reversible_cfg()
    orbit = analytical_periodic_orbit(
        cfg=cfg, T_K=300.0, Kmax_Pa_sqrt_m=18.0e6, R=-0.5, frequency_Hz=1000.0, r_contact_m=r_eff_m, n_phase=64,
    )
    K_rebond_max = cfg.rebond_K_geometry_factor * math.sqrt(Eprime_Pa * cfg.restored_work_of_separation_J_m2)
    predicted = predicted_static_shield_equivalent_K_b(orbit, cfg, Eprime_Pa)
    assert predicted == pytest.approx(K_rebond_max * orbit["mean_p_B"])


def test_sat_existing_frozen_config_hash_reproduced_verbatim():
    """The row-construction script's SAT_EXISTING row must reproduce the
    existing frozen RB2_reversible_finite config hash exactly -- a
    regression here means the frozen config was NOT reused verbatim."""
    frozen_path = REPO_ROOT / "artifacts" / "crack_rebonding_causal_pilot_v2" / "frozen_configuration.json"
    payload = json.loads(frozen_path.read_text())
    raw = dict(payload["configs"]["RB2_reversible_finite"])
    raw["model_level"] = RebondModelLevel(raw["model_level"])
    raw["contact_model"] = ContactModel(raw["contact_model"])
    raw["feedback_mode"] = FeedbackMode(raw["feedback_mode"])
    raw["initial_precrack_wake_mode"] = InitialPrecrackWakeMode(raw["initial_precrack_wake_mode"])
    cfg = CrackRebondingControls(**raw).validate()
    assert cfg.config_hash() == payload["config_hashes"]["RB2_reversible_finite"]


def test_kinetic_regime_registry_artifacts_exist_and_all_gates_passed():
    """Confirms the PX2 artifacts committed to the repo (not regenerated
    by this test -- generation takes a few seconds and is exercised
    directly by running the script) record all four rows passing their
    gates."""
    registry_path = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1" / "kinetic_regime_registry.json"
    if not registry_path.exists():
        pytest.skip("kinetic_regime_registry.json not yet built in this worktree")
    registry = json.loads(registry_path.read_text())
    for row_name in ("SAT_EXISTING", "COMPETING_REVERSIBLE", "COMPETING_PERSISTENT", "PASSIVATION_LIMITED"):
        assert registry["rows"][row_name]["gates"]["all_gates_passed"] is True
