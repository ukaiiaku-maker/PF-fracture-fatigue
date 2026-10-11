import numpy as np

from arrhenius_fracture.reduced_persistent_reference_r2 import (
    HISTORICAL_REPLAY_ID, TAYLOR_LAW_ID, _stable_forward_minus_zero,
    advance_signed, build_restored_engine, runtime_receipt,
)
from arrhenius_fracture.reduced_shared_state_v1023 import SharedReducedConfig


def config():
    return SharedReducedConfig(
        mpz_length_um=50.0, mpz_n_bins=80, wake_length_um=50.0,
        max_internal_steps=100000,
    )


def test_default_is_persistent_reversible_and_historical_replay_is_separate():
    engine = build_restored_engine(config())
    receipt = runtime_receipt(engine, stage="initialized")
    assert receipt["historical_replay_identity"] == HISTORICAL_REPLAY_ID
    assert receipt["source_lifecycle"] == "persistent_reusable"
    assert receipt["reversible_binding"]["all_expected_methods_bound"] is True
    assert receipt["persistent_invariants"] == {
        "inventory_active": False,
        "depletion_on_emission": False,
        "advance_refresh": False,
        "available_equals_capacity": True,
    }
    assert receipt["taylor_contract"]["law_id"] == TAYLOR_LAW_ID
    assert receipt["taylor_contract"]["finite_window_s"] is None


def test_actual_forward_state_can_physically_return_and_reduce_net_blunting():
    engine = build_restored_engine(config())
    forward = advance_signed(
        engine, K_Pa_sqrt_m=10.0e6, temperature_K=1200.0, dt_s=1.0e-3,
    )
    before = engine.r_eff()
    reverse = advance_signed(
        engine, K_Pa_sqrt_m=-10.0e6, temperature_K=1200.0, dt_s=1.0e-3,
    )
    assert forward["dN_emit"] > 0.0
    assert reverse["reversible_mobile_velocity_m_s"] < 0.0
    assert reverse["dN_physical_returned"] > 0.0
    assert engine.mpz.cumulative_gross_return_activity > 0.0
    assert engine.r_eff() <= before


def test_stable_taylor_net_rate_is_zero_at_zero_and_positive_under_drive():
    engine = build_restored_engine(config())
    surface = engine.manifest.taylor.as_surface(engine.manifest.emission)
    rates = _stable_forward_minus_zero(surface, np.array([0.0, 1.0e9]), 900.0)
    assert rates[0] == 0.0
    assert rates[1] > 0.0
