"""Runtime identity contract for the intended persistent-source baseline.

This module is an enforcement layer, not a new constitutive model.  It inspects
the methods bound on the actual evolving MPZ object and fails closed if the
requested persistent source is replaced by finite inventory, split/reduced
transport, or a different release/advance binding.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

from .persistent_site_source_v10221 import (
    SOURCE_MODEL, _coupled_persistent_source_transport, _persistent_advance,
    _persistent_emit,
)
from .coupled_transport_exchange import joint_transport_exchange
from .signed_burgers_shared_v1025 import (
    JOINT_CTMC_EXCHANGE_TRANSPORT, _signed_evolve_scalar,
)
from .taylor_release import completion_rate, runtime_selection
from .unified_mpz import UnifiedMPZState


MODEL_ID = "persistent-native-runtime-contract/1"


def _function(bound: Any) -> Any:
    return getattr(bound, "__func__", bound)


def _identity(function: Any) -> dict[str, Any]:
    fn = _function(function)
    source = inspect.getsourcefile(fn)
    path = None if source is None else Path(source).resolve()
    return {
        "callable": f"{getattr(fn, '__module__', type(fn).__module__)}."
                    f"{getattr(fn, '__qualname__', type(fn).__qualname__)}",
        "source_file": None if path is None else str(path),
        "source_sha256": None if path is None or not path.is_file()
                         else hashlib.sha256(path.read_bytes()).hexdigest(),
        "body_sha256": hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest(),
    }


def persistent_binding_receipt(mpz: Any) -> dict[str, Any]:
    reversible = bool(getattr(mpz, "_reversible_transport_installed", False))
    expected = {
        "emit": _persistent_emit,
        "advance": _persistent_advance,
        "evolve": _signed_evolve_scalar,
        "coupled_source_transport": _coupled_persistent_source_transport,
    }
    # Reversible transport intentionally wraps evolve/advance after the base
    # persistent state is installed.  Its own authoritative receipt verifies
    # those wrappers and the preserved base advance callable.
    if reversible:
        from .persistent_site_reversible_transport_v10230 import reversible_binding_receipt
        reversible_receipt = reversible_binding_receipt(mpz)
        matches = {
            "emit": _function(getattr(mpz, "_emit", None)) is _persistent_emit,
            "base_advance": _function(getattr(mpz, "_reversible_base_advance", None)) is _persistent_advance,
            "coupled_source_transport": _function(
                getattr(mpz, "_coupled_persistent_source_transport", None)
            ) is _coupled_persistent_source_transport,
            "reversible_overlay": bool(reversible_receipt["all_expected_methods_bound"]),
        }
    else:
        reversible_receipt = None
        matches = {
            name: _function(getattr(
                mpz,
                "_emit" if name == "emit" else
                "_coupled_persistent_source_transport" if name == "coupled_source_transport" else
                name,
                None,
            )) is fn
            for name, fn in expected.items()
        }
    inventory = bool(getattr(mpz, "persistent_source_inventory_active", False))
    observed = {
        "emit": _identity(getattr(mpz, "_emit", None)),
        "evolve": _identity(getattr(mpz, "evolve", None)),
        "advance": _identity(getattr(mpz, "advance", None)),
        "exchange": _identity(getattr(mpz, "_exchange", None)),
        "emission_rate_per_site": _identity(getattr(mpz, "emission_rate_per_site", None)),
    }
    return {
        "schema": MODEL_ID,
        "object_type": f"{type(mpz).__module__}.{type(mpz).__qualname__}",
        "source_model": str(getattr(mpz, "source_model", "")),
        "signed_transport_mode": str(getattr(mpz, "_signed_transport_mode", "")),
        "exchange_transport_mode": str(getattr(mpz, "_exchange_transport_mode", "")),
        "persistent_source_inventory_active": inventory,
        "available_sites_are_diagnostic_only": True,
        "bindings": observed, "expected_method_match": matches,
        "inner_operator_identities": {
            "joint_transport_exchange": _identity(joint_transport_exchange),
            "taylor_completion_rate": _identity(completion_rate),
            "transport_rate_constructor": _identity(UnifiedMPZState._transport_rates),
        },
        "taylor_completion_constructor": dict(zip(
            ("mode", "window_s", "hard_cap"), runtime_selection()
        )),
        "reversible_overlay": reversible_receipt,
        "physical_contract": {
            "injection": "continuous mobile source inside the coupled CTMC with an implicit persistent-site backstress/end-state root",
            "transport": "signed finite-volume Peierls transport in the same CTMC generator",
            "exchange_release": "simultaneous encounter/Taylor release/recapture, transport, escape, and gross/net ledgers",
            "backstress": "unsigned active mobile plus retained line content",
            "shielding": "signed retained line content",
            "advance": "signed moving-frame wake inheritance with persistent sites not consumed",
            "time_unit": "s",
        },
        "all_expected_methods_bound": (
            str(getattr(mpz, "source_model", "")) == SOURCE_MODEL
            and str(getattr(mpz, "_signed_transport_mode", "")) == "validated_scalar"
            and str(getattr(mpz, "_exchange_transport_mode", "")) == JOINT_CTMC_EXCHANGE_TRANSPORT
            and not inventory and all(matches.values())
        ),
    }


def require_persistent_binding(mpz: Any, *, stage: str) -> dict[str, Any]:
    receipt = persistent_binding_receipt(mpz)
    if not receipt["all_expected_methods_bound"]:
        failed = [name for name, okay in receipt["expected_method_match"].items() if not okay]
        raise RuntimeError(
            f"persistent native physics binding mismatch at {stage}: "
            f"source_model={receipt['source_model']!r}, "
            f"transport={receipt['signed_transport_mode']!r}, "
            f"inventory={receipt['persistent_source_inventory_active']!r}, "
            f"failed_bindings={failed}; no substitute will be started"
        )
    return receipt


__all__ = ["MODEL_ID", "persistent_binding_receipt", "require_persistent_binding"]
