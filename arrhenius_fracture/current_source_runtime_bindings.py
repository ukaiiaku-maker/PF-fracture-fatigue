"""Explicit rehydration of persistent signed-process runtime methods.

Model IDs select a closed registry. No constructor, installer, random draw,
state initialization, grid binding, or physical update is performed here.
"""
from __future__ import annotations

from types import MethodType

from .persistent_site_source_v10221 import (
    SOURCE_MODEL, _persistent_emit, _persistent_advance, _persistent_diagnostics,
)
from .signed_burgers_shared_v1025 import (
    MODEL_ID, VALIDATED_SCALAR_TRANSPORT, CHANNEL_RESOLVED_TRANSPORT,
    _signed_evolve_scalar, _signed_evolve_channel, _active_K, _wake_K,
    _total_K, _signed_diagnostics,
)


class RuntimeBindingError(RuntimeError):
    """The serialized model cannot select a complete known implementation."""


ENGINE_IDS = frozenset({
    "AuditedPersistentSiteStateResolvedTipEngine",
    "PersistentSiteStateResolvedTipEngine",
})
_COMMON = {
    "_emit": _persistent_emit,
    "advance": _persistent_advance,
    "active_K_shielding": _active_K,
    "wake_K_shielding": _wake_K,
    "shielding_K": _total_K,
    "diagnostics": _persistent_diagnostics,
    "_persistent_base_diagnostics": _signed_diagnostics,
}
BINDING_REGISTRY = {
    (SOURCE_MODEL, MODEL_ID, VALIDATED_SCALAR_TRANSPORT): {
        **_COMMON, "evolve": _signed_evolve_scalar,
    },
    (SOURCE_MODEL, MODEL_ID, CHANNEL_RESOLVED_TRANSPORT): {
        **_COMMON, "evolve": _signed_evolve_channel,
    },
}


def callable_id(function):
    function = getattr(function, "__func__", function)
    return f"{function.__module__}.{function.__qualname__}"


def runtime_binding_inventory(engine):
    return {
        owner_name: {
            name: callable_id(value)
            for name, value in sorted(vars(owner).items()) if callable(value)
        }
        for owner_name, owner in (("engine", engine), ("mpz", engine.mpz))
    }


def rehydrate_current_source_runtime_bindings(engine):
    """Bind only methods; reject unknown/inconsistent IDs before any mutation."""
    if type(engine).__name__ not in ENGINE_IDS:
        raise RuntimeBindingError(f"unknown current-source engine ID: {type(engine).__name__}")
    mpz = engine.mpz
    if type(mpz).__name__ != "UnifiedMPZState":
        raise RuntimeBindingError(f"unknown process class: {type(mpz).__name__}")
    key = tuple(getattr(mpz, name, None) for name in (
        "source_model", "state_model", "_signed_transport_mode",
    ))
    if key not in BINDING_REGISTRY:
        raise RuntimeBindingError(f"unknown or inconsistent serialized process model IDs: {key!r}")
    bindings = BINDING_REGISTRY[key]
    # These fields identify a completely installed persistent signed stack.
    # Presence is checked, but no defaults, copies, or state repairs are made.
    required = (
        "_persistent_site_cfg", "_persistent_r0_m", "_persistent_b",
        "_persistent_active_arc_factor", "_signed_kernel",
        "_anisotropic_emission_config", "mobile_positive", "mobile_negative",
        "retained_positive", "retained_negative", "wake_mobile_positive",
        "wake_mobile_negative", "wake_retained_positive", "wake_retained_negative",
    )
    missing = [name for name in required if not hasattr(mpz, name)]
    if missing:
        raise RuntimeBindingError(f"incomplete serialized persistent process stack: {missing}")
    inventory = runtime_binding_inventory(engine)
    if inventory["engine"] or set(inventory["mpz"]) - set(bindings):
        raise RuntimeBindingError("unregistered dynamically installed process callable")
    # An attribute expected to be a method must not overwrite serialized data.
    for name in bindings:
        if name in vars(mpz) and not callable(vars(mpz)[name]):
            raise RuntimeBindingError(f"runtime method collides with serialized field: {name}")
    for name, function in bindings.items():
        setattr(mpz, name, MethodType(function, mpz))
    return engine
