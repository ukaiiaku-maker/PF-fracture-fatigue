from __future__ import annotations

from types import MethodType

import pytest

from arrhenius_fracture.sharp_front_v10_1_7_5 import (
    CHANNEL_RESOLVED_TRANSPORT,
    VALIDATED_SCALAR_TRANSPORT,
    make_transport_installer,
    normalize_transport_mode,
)


class _DummyState:
    def evolve(self, *args, **kwargs):
        return "validated"


def _fake_anisotropic_install(state, config) -> None:
    def channel_evolve(self, *args, **kwargs):
        return "channel"

    state._emit = object()
    state.evolve = MethodType(channel_evolve, state)


def test_transport_mode_aliases() -> None:
    assert normalize_transport_mode(None) == VALIDATED_SCALAR_TRANSPORT
    assert normalize_transport_mode("emission-only") == VALIDATED_SCALAR_TRANSPORT
    assert normalize_transport_mode("validated") == VALIDATED_SCALAR_TRANSPORT
    assert normalize_transport_mode("channel") == CHANNEL_RESOLVED_TRANSPORT
    assert normalize_transport_mode("full") == CHANNEL_RESOLVED_TRANSPORT


def test_invalid_transport_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid ANISOTROPIC_TRANSPORT_MODE"):
        normalize_transport_mode("mixed_unknown")


def test_validated_mode_restores_inherited_evolve_but_keeps_emitter() -> None:
    state = _DummyState()
    installer = make_transport_installer(
        _fake_anisotropic_install,
        VALIDATED_SCALAR_TRANSPORT,
    )
    installer(state, None)
    assert state.evolve() == "validated"
    assert hasattr(state, "_emit")
    assert state._anisotropic_transport_mode == VALIDATED_SCALAR_TRANSPORT
    assert state._anisotropic_transport_channel_resolved is False


def test_channel_mode_keeps_channel_resolved_evolve() -> None:
    state = _DummyState()
    installer = make_transport_installer(
        _fake_anisotropic_install,
        CHANNEL_RESOLVED_TRANSPORT,
    )
    installer(state, None)
    assert state.evolve() == "channel"
    assert hasattr(state, "_emit")
    assert state._anisotropic_transport_mode == CHANNEL_RESOLVED_TRANSPORT
    assert state._anisotropic_transport_channel_resolved is True
