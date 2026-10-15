from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.current_source_runtime_bindings import (
    rehydrate_current_source_runtime_bindings,
)
from arrhenius_fracture.persistent_site_source_v10221 import SOURCE_MODEL
from arrhenius_fracture.signed_burgers_shared_v1025 import (
    CHANNEL_RESOLVED_TRANSPORT,
    MODEL_ID,
    VALIDATED_SCALAR_TRANSPORT,
    _signed_evolve_channel,
    _signed_evolve_scalar,
    _signed_evolve_wake,
)
from arrhenius_fracture.unified_mpz import UnifiedMPZState


def _restorable_current_source_engine(transport_mode: str):
    """Numerical restoration fixture; it is not a natural fracture event."""
    engine_type = type("AuditedPersistentSiteStateResolvedTipEngine", (), {})
    engine = engine_type()
    mpz = object.__new__(UnifiedMPZState)
    engine.mpz = mpz
    mpz.source_model = SOURCE_MODEL
    mpz.state_model = MODEL_ID
    mpz._signed_transport_mode = transport_mode
    mpz._persistent_site_cfg = {"fixture": True}
    mpz._persistent_r0_m = 1.0e-6
    mpz._persistent_b = 2.5e-10
    mpz._persistent_active_arc_factor = 1.0
    mpz._signed_kernel = {"fixture": True}
    mpz._anisotropic_emission_config = {"fixture": True}
    for name in (
        "mobile_positive", "mobile_negative", "retained_positive",
        "retained_negative", "wake_mobile_positive", "wake_mobile_negative",
        "wake_retained_positive", "wake_retained_negative",
    ):
        setattr(mpz, name, np.arange(4.0).reshape(1, 4))
    return engine


@pytest.mark.parametrize(
    ("transport_mode", "expected"),
    [
        (VALIDATED_SCALAR_TRANSPORT, _signed_evolve_scalar),
        (CHANNEL_RESOLVED_TRANSPORT, _signed_evolve_channel),
    ],
)
def test_current_source_restore_rebinds_selected_kinetics_without_state_repair(
    transport_mode, expected,
):
    engine = _restorable_current_source_engine(transport_mode)
    before = {
        name: value.copy()
        for name, value in vars(engine.mpz).items()
        if isinstance(value, np.ndarray)
    }

    restored = rehydrate_current_source_runtime_bindings(engine)
    assert restored is engine
    assert engine.mpz.evolve.__func__ is expected
    assert engine.mpz._signed_transport_mode == transport_mode
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(engine.mpz, name), value)

    # Rebinding after another restore is idempotent and cannot select the
    # other historical transport body as a silent fallback.
    rehydrate_current_source_runtime_bindings(engine)
    assert engine.mpz.evolve.__func__ is expected


def _occupied_wake_fixture():
    state = SimpleNamespace()
    state.wake_n_bins = 4
    state.wake_dx = 1.0
    state.wake_mobile_positive = np.array([[1.0, 0.5, 0.0, 0.0]])
    state.wake_mobile_negative = np.array([[0.0, 0.0, 0.25, 0.75]])
    state.wake_retained_positive = np.array([[0.2, 0.0, 0.0, 0.0]])
    state.wake_retained_negative = np.array([[0.0, 0.3, 0.0, 0.0]])
    state.wake_slip_positive = np.zeros((1, 4))
    state.wake_slip_negative = np.zeros((1, 4))
    state.wake_mobile = state.wake_mobile_positive + state.wake_mobile_negative
    state.wake_retained = (
        state.wake_retained_positive + state.wake_retained_negative
    )
    state.wake_slip = state.wake_slip_positive + state.wake_slip_negative
    state.wake_mobile_count = float(np.sum(state.wake_mobile))
    state.wake_retained_count = float(np.sum(state.wake_retained))
    state.wake_discarded_mobile_total = 0.0
    state.cfg = SimpleNamespace(mobile_recovery_rate_s=0.4)
    state.manifest = SimpleNamespace(retained_recovery_rate_s=0.2)
    state.local_forest_density_m2 = lambda wake: np.ones(4)
    state._transport_rates = lambda *_args: {
        "encounter": np.full(4, 0.7),
        "taylor": np.full(4, 0.3),
        "velocity": np.full(4, 0.8),
    }
    state._exchange = UnifiedMPZState._exchange
    state._advect_forward = UnifiedMPZState._advect_forward
    return state


def test_occupied_wake_keeps_signed_species_and_closes_fate_balance():
    """Injected occupied-state fixture, not a natural wake-population claim."""
    state = _occupied_wake_fixture()
    positive_before = float(np.sum(
        state.wake_mobile_positive + state.wake_retained_positive
    ))
    negative_before = float(np.sum(
        state.wake_mobile_negative + state.wake_retained_negative
    ))
    total_before = positive_before + negative_before

    result = _signed_evolve_wake(state, 0.5, 900.0, 2.5e-10)

    positive_after = float(np.sum(
        state.wake_mobile_positive + state.wake_retained_positive
    ))
    negative_after = float(np.sum(
        state.wake_mobile_negative + state.wake_retained_negative
    ))
    removed = (
        result["wake_dN_recovered"]
        + result["wake_dN_mobile_transport_loss"]
    )
    assert positive_after > 0.0
    assert negative_after > 0.0
    assert positive_after + negative_after + removed == pytest.approx(total_before)
    assert result["wake_dN_trapped"] >= 0.0
    assert result["wake_dN_released"] >= 0.0
    assert state.wake_discarded_mobile_total == pytest.approx(
        result["wake_dN_mobile_transport_loss"]
    )
