"""R1 model identities and live geometry/source/void state contracts.

This module is deliberately an ownership/selection layer.  It does not copy a
mechanics law, install a compatibility monkeypatch, or change any calibrated
material row.  Each identity points at an existing, independently runnable
implementation and the state audit reads the native owners in place.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


class RestorationModelIdentity(str, Enum):
    RESTORED_REDUCED_REFERENCE = "RESTORED_REDUCED_REFERENCE"
    HISTORICAL_FINITE_SOURCE_REPLAY_V10_2_3 = "HISTORICAL_FINITE_SOURCE_REPLAY_V10_2_3"
    FULL_SPATIAL_REFERENCE = "FULL_SPATIAL_REFERENCE"
    SEAM_DIAGNOSTIC_V4_10 = "SEAM_DIAGNOSTIC_V4_10"


@dataclass(frozen=True)
class RestorationBinding:
    identity: RestorationModelIdentity
    implementation: str
    entry: str
    mechanics_strategy: str
    geometry_owner: str
    source_owner: str
    cavity_owner: str | None
    default: bool
    physical_scope: str


BINDINGS: dict[RestorationModelIdentity, RestorationBinding] = {
    RestorationModelIdentity.RESTORED_REDUCED_REFERENCE: RestorationBinding(
        RestorationModelIdentity.RESTORED_REDUCED_REFERENCE,
        "R2 persistent/reversible signed MPZ with corrected mean-rate Taylor consumer",
        "scripts.run_r2_reduced_runtime_contract:main",
        "prescribed_K_reduced_provider_with_production_state",
        "FrontEngine.r_eff plus material-frame signed MPZ moving window",
        "persistent_site_source_v10221",
        None,
        True,
        "reduced external mechanics with the production signed MPZ constitutive state",
    ),
    RestorationModelIdentity.HISTORICAL_FINITE_SOURCE_REPLAY_V10_2_3: RestorationBinding(
        RestorationModelIdentity.HISTORICAL_FINITE_SOURCE_REPLAY_V10_2_3,
        "unchanged v10.2.3 finite campaign source replay",
        "scripts.run_v10_2_3_shared_reduced:main",
        "recorded_2d_K_dt_temperature_and_channel_factor_schedule",
        "historical FrontEngine.r_eff and finite moving MPZ window",
        "campaign_calibrated_tip finite depletion/advance refresh",
        None,
        False,
        "same-law replay control only; not the persistent/reversible default",
    ),
    RestorationModelIdentity.FULL_SPATIAL_REFERENCE: RestorationBinding(
        RestorationModelIdentity.FULL_SPATIAL_REFERENCE,
        "anisotropic PF/sharp-front shared option family with V11-V13 and V5 extensions",
        "arrhenius_fracture.shared_execution:execute_shared",
        "full_spatial_pf",
        "sharp-front topology plus analytical local kinetic radius (not a rounded FEM notch)",
        "selected persistent, finite-inventory, or current-source owner",
        "voiding_production_v5 when full_order_v5 is selected",
        False,
        "full spatial mechanics, topology, remesh, source/provider rebinding, and optional cavities",
    ),
    RestorationModelIdentity.SEAM_DIAGNOSTIC_V4_10: RestorationBinding(
        RestorationModelIdentity.SEAM_DIAGNOSTIC_V4_10,
        "V4.10 phase-exact moving circular seam/cut diagnostic",
        "scripts.run_v4_10_moving_tip_fatigue_v10230:main",
        "linear_elastic_circular_seam_cut",
        "offset straight seam in a fixed circular body",
        "V4.10 cut-history ownership",
        None,
        False,
        "opt-in numerical/mechanics diagnostic; never a replacement for either restored reference",
    ),
}


def resolve_restoration_identity(value: str | RestorationModelIdentity) -> RestorationBinding:
    """Resolve one explicit identity; aliases and silent fallbacks are forbidden."""
    identity = value if isinstance(value, RestorationModelIdentity) else RestorationModelIdentity(value)
    return BINDINGS[identity]


def registry_payload() -> dict[str, Any]:
    return {
        "schema": "fracture-fatigue-restoration-registry-r2/1",
        "default_identity": RestorationModelIdentity.RESTORED_REDUCED_REFERENCE.value,
        "models": {
            key.value: {**asdict(binding), "identity": binding.identity.value}
            for key, binding in BINDINGS.items()
        },
        "no_refit": True,
        "no_mechanics_mixing": True,
    }


def _array_sum(owner: Any, name: str) -> float | None:
    value = getattr(owner, name, None)
    if value is None:
        return None
    try:
        import numpy as np
        result = float(np.sum(np.asarray(value, dtype=float)))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def live_geometry_source_void_contract(
    process_state: Any,
    *,
    identity: RestorationModelIdentity | str,
) -> dict[str, Any]:
    """Read, but never reconstruct, the native state ownership graph.

    Persistent-site geometry is obtained from the donor's own ``_source_geometry``
    routine so radius, front width, source area, and multiplicity cannot drift
    into parallel formulas.  Cavity radii remain separate state owned by V5.
    """
    binding = resolve_restoration_identity(identity)
    source_geometry = None
    if hasattr(process_state, "_persistent_site_cfg"):
        from .persistent_site_source_v10221 import _source_geometry
        source_geometry = dict(_source_geometry(process_state))

    effective_radius = None
    if source_geometry is not None:
        effective_radius = float(source_geometry["tip_radius_m"])
    elif callable(getattr(process_state, "r_eff", None)):
        effective_radius = float(process_state.r_eff())
    elif hasattr(process_state, "r_tip_m"):
        effective_radius = float(process_state.r_tip_m)

    cavity_radii: list[float] = []
    void_state = getattr(process_state, "void_state", None)
    for cavity in getattr(void_state, "cavities", ()) if void_state is not None else ():
        cavity_radii.append(float(cavity.radius_m))

    if effective_radius is not None and not math.isfinite(effective_radius):
        raise ValueError("native effective tip radius is nonfinite")
    if any(not math.isfinite(value) or value < 0.0 for value in cavity_radii):
        raise ValueError("native cavity radius is invalid")

    payload = {
        "schema": "geometry-source-void-state-contract-r1/1",
        "model_identity": binding.identity.value,
        "mechanics_strategy": binding.mechanics_strategy,
        "effective_kinetic_tip_radius_m": effective_radius,
        "source_geometry": source_geometry,
        "cavity_radii_m": cavity_radii,
        "tip_radius_and_cavity_radius_have_distinct_owners": True,
        "signed_state": {
            name: _array_sum(process_state, name)
            for name in (
                "mobile_positive", "mobile_negative", "retained_positive",
                "retained_negative", "accumulated_slip_positive",
                "accumulated_slip_negative", "returned_slip_positive",
                "returned_slip_negative",
            )
        },
        "moving_frame": {
            "wake_mobile": _array_sum(process_state, "wake_mobile"),
            "wake_retained": _array_sum(process_state, "wake_retained"),
            "source_slip_wake_transfer": _array_sum(
                process_state, "cumulative_source_slip_wake_transfer"
            ),
        },
        "contribution_ownership": {
            "resolved_mechanics": binding.mechanics_strategy,
            "unresolved_source_resistance": binding.source_owner,
            "kinetic_sampling_geometry": binding.geometry_owner,
            "cavity_geometry": binding.cavity_owner,
            "mixed_seam_plus_persistent_radius": False,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    payload["contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def write_registry(path: str | Path) -> None:
    Path(path).write_text(json.dumps(registry_payload(), indent=2, sort_keys=True) + "\n")


__all__ = [
    "BINDINGS", "RestorationBinding", "RestorationModelIdentity",
    "live_geometry_source_void_contract", "registry_payload",
    "resolve_restoration_identity", "write_registry",
]
