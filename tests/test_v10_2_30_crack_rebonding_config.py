from __future__ import annotations

import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    ContactModel,
    CrackRebondingControls,
    FeedbackMode,
    RebondModelLevel,
    crack_rebonding_config_from_environment,
)


def test_default_is_disabled_and_valid():
    cfg = CrackRebondingControls().validate()
    assert cfg.enabled is False
    assert cfg.model_level is RebondModelLevel.REBOND_OFF
    assert cfg.feedback_mode is FeedbackMode.HAZARD_ONLY_REBOND_SHIELD


@pytest.mark.parametrize(
    "field,value",
    [
        ("topological_healing_enabled", True),
        ("negative_crack_advance_enabled", True),
        ("crack_segment_deletion_enabled", True),
        ("minimum_load_hold_s", 1.0e-6),
        ("stochastic_healing_enabled", True),
        ("chemistry_factor", 1.5),
        ("chemistry_factor", -0.1),
        ("wake_length_m", -1.0e-6),
        ("wake_weight_length_m", 0.0),
        ("contact_radius_min_m", 0.0),
        ("contact_pressure_cap_Pa", 0.0),
        ("opening_stress_cap_Pa", -1.0),
        ("fresh_surface_clean_fraction", 1.5),
        ("bond_barrier_eV", -0.1),
        ("bond_attempt_frequency_s", -1.0),
        ("bond_activation_volume_m3", -1.0e-30),
        ("healing_cooperative_order", 0.5),
        ("healing_correlation_time_s", 0.0),
        ("healing_correlation_time_s", -1.0e-6),
        ("restored_work_of_separation_J_m2", -1.0),
        ("rebond_K_geometry_factor", -1.0),
        ("rebonding_block_max_dpB", 0.0),
        ("rebonding_block_action_consistency_tol", -0.1),
    ],
)
def test_rejects_invalid_fields(field, value):
    cfg = CrackRebondingControls(**{field: value})
    with pytest.raises(ValueError):
        cfg.validate()


def test_wake_weight_length_must_not_exceed_wake_length():
    cfg = CrackRebondingControls(wake_length_m=1.0e-6, wake_weight_length_m=2.0e-6)
    with pytest.raises(ValueError):
        cfg.validate()


def test_resolved_gap_traction_always_rejected_even_with_oracle_fields():
    cfg = CrackRebondingControls(
        contact_model=ContactModel.RESOLVED_GAP_TRACTION,
        contact_oracle_id="future-oracle",
        contact_oracle_version="1.0",
        contact_oracle_source_hash="deadbeef",
    )
    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize(
    "mode",
    [FeedbackMode.COMMON_POSITIVE_LOCAL_K_REDUCTION, FeedbackMode.HAZARD_AND_ENERGY_GATE_COUPLED],
)
def test_unimplemented_feedback_modes_raise_not_implemented(mode):
    cfg = CrackRebondingControls(feedback_mode=mode)
    with pytest.raises(NotImplementedError):
        cfg.validate()


def test_zero_bond_attempt_frequency_is_valid_zero_effect_config():
    cfg = CrackRebondingControls(bond_attempt_frequency_s=0.0).validate()
    assert cfg.bond_attempt_frequency_s == 0.0


def test_zero_chemistry_factor_is_valid():
    cfg = CrackRebondingControls(chemistry_factor=0.0).validate()
    assert cfg.chemistry_factor == 0.0


def test_config_hash_deterministic_and_sensitive_to_changes():
    a = CrackRebondingControls().validate()
    b = CrackRebondingControls().validate()
    c = CrackRebondingControls(chemistry_factor=0.5).validate()
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()


def test_env_loader_defaults_to_disabled():
    cfg = crack_rebonding_config_from_environment(env={})
    assert cfg.enabled is False


def test_env_loader_reads_enabled_flag():
    cfg = crack_rebonding_config_from_environment(env={"V10230_CRACK_REBONDING_ENABLED": "1"})
    assert cfg.enabled is True


def test_env_loader_overrides_take_precedence_over_env():
    cfg = crack_rebonding_config_from_environment(
        env={"V10230_CRACK_REBONDING_ENABLED": "1"}, overrides={"enabled": False}
    )
    assert cfg.enabled is False


def test_env_loader_reads_enum_and_float_fields():
    cfg = crack_rebonding_config_from_environment(
        env={
            "V10230_CRACK_REBONDING_ENABLED": "1",
            "V10230_CRACK_REBONDING_MODEL_LEVEL": "CLEAN_REVERSIBLE_REBOND",
            "V10230_CRACK_REBONDING_CHEMISTRY_FACTOR": "0.5",
        }
    )
    assert cfg.model_level is RebondModelLevel.CLEAN_REVERSIBLE_REBOND
    assert cfg.chemistry_factor == pytest.approx(0.5)
