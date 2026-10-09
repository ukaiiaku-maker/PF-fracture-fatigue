"""Fail-closed signed-kernel family migration for v11 restart checkpoints.

The production checkpoint intentionally contains the complete process state,
including the signed-kernel object that was active when it was written.  A
continuation against an append-only replacement atlas must restore all physical
and stochastic state without allowing that archived family object to overwrite
the family selected by the new launch.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
from typing import Any, Mapping

import numpy as np

from .signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
)


MIGRATION_SCHEMA = "v11.restart-signed-kernel-family-migration/1"
_ENGINE_FAMILY_FIELDS = frozenset({"_state_kernel_family"})
_MPZ_FAMILY_FIELDS = frozenset({"_signed_kernel"})
_MIGRATION_FIELDS = frozenset({
    "_restart_family_migration_audit",
    "_restart_family_migration_count",
})


def _sha_pickle(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"pickle_sha256": _sha_pickle(value), "type": type(value).__name__}


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def family_digest(family: ActiveOnlySigned2DShieldingKernelFamily) -> str:
    """Hash the bound operator and interpolation contract, not its file path."""
    payload = {
        "states": [
            {
                "state_id": state.state_id,
                "coordinates": state.coordinates,
                "active_I": state.active_I,
                "wake_I": state.wake_I,
                "active_II": state.active_II,
                "wake_II": state.wake_II,
            }
            for state in family.states
        ],
        "active_x_m": family.active_x_m,
        "wake_x_m": family.wake_x_m,
        "activation_to_line_content": family.activation_to_line_content,
        "source_capacity_bounds": family.source_capacity_bounds,
        "interpolation": family.interpolation,
        "append_only_legacy_domain_policy": family.metadata.get(
            "append_only_legacy_domain_policy"
        ),
    }
    return canonical_hash(payload)


def _payload_without_family(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": payload.get("schema"),
        "engine_type": payload.get("engine_type"),
        "engine_fields": {
            key: value for key, value in payload.get("engine_fields", {}).items()
            if key not in _ENGINE_FAMILY_FIELDS and key not in _MIGRATION_FIELDS
        },
        "mpz_type": payload.get("mpz_type"),
        "mpz_fields": {
            key: value for key, value in payload.get("mpz_fields", {}).items()
            if key not in _MPZ_FAMILY_FIELDS
        },
    }


def process_state_digest(payload: Mapping[str, Any]) -> str:
    """Hash every serialized process field except family/provenance identity."""
    return canonical_hash(_payload_without_family(payload))


def rng_digest(payload: Mapping[str, Any]) -> str:
    fields = payload.get("engine_fields", {})
    return canonical_hash({
        key: value for key, value in fields.items()
        if "rng" in key.lower() or "threshold" in key.lower()
    })


def _family_references(owner: Any) -> dict[str, ActiveOnlySigned2DShieldingKernelFamily]:
    return {
        name: value for name, value in owner.__dict__.items()
        if isinstance(value, ActiveOnlySigned2DShieldingKernelFamily)
    }


def _drop_serializable_fields_absent_from_checkpoint(
    owner: Any, archived_fields: Mapping[str, Any], protected: frozenset[str]
) -> None:
    """Remove newer initializer defaults that were not in the archived state."""
    for name, value in list(owner.__dict__.items()):
        if name in archived_fields or name in protected or callable(value):
            continue
        try:
            pickle.dumps(value, protocol=5)
        except Exception:
            continue
        delattr(owner, name)


def migrate_shared_engine_family(
    engine: Any,
    payload: Mapping[str, Any],
    *,
    source_family: ActiveOnlySigned2DShieldingKernelFamily,
    target_family: ActiveOnlySigned2DShieldingKernelFamily,
    source_family_sha256: str,
    target_family_sha256: str,
    target_family_physics_fingerprint: str,
) -> tuple[Any, dict[str, Any]]:
    """Restore a checkpoint and replace only its archived family identity.

    ``engine`` must be a freshly initialized production engine.  No stochastic
    method is called here and no time, load, hazard, or MPZ update is performed.
    """
    if payload.get("schema") != "v11.shared-production-engine-state/1":
        raise ValueError("unsupported shared production-engine state")
    if payload.get("engine_type") != type(engine).__name__:
        raise RuntimeError("restart engine type differs from initialized production engine")
    if int(getattr(engine, "_restart_family_migration_count", 0)) != 0:
        raise RuntimeError("restart family migration may be applied exactly once")

    engine_fields = payload.get("engine_fields", {})
    mpz_fields = payload.get("mpz_fields", {})
    archived_engine_family = engine_fields.get("_state_kernel_family")
    archived_mpz_family = mpz_fields.get("_signed_kernel")
    if not isinstance(archived_engine_family, ActiveOnlySigned2DShieldingKernelFamily):
        raise RuntimeError("checkpoint is missing its archived engine family")
    if archived_engine_family is not archived_mpz_family:
        raise RuntimeError("checkpoint engine and MPZ family aliases are not identical")

    # Qualify the serialized source operator against the independently loaded,
    # pinned source atlas on the checkpoint's actual grid before discarding it.
    source_grid = type("RestoredGrid", (), {
            "n_systems": int(mpz_fields["n_systems"]),
            "n_bins": int(mpz_fields["n_bins"]),
            "x": np.asarray(mpz_fields["x"]),
            "wake_x": np.asarray(mpz_fields["wake_x"]),
            "wake_n_bins": int(mpz_fields["wake_n_bins"]),
            "site_capacity": np.asarray(mpz_fields["site_capacity"]),
        })()
    source_bound = source_family.clone_for_engine()
    source_bound.validate_state(source_grid)
    archived_source_digest = family_digest(archived_engine_family)
    independently_loaded_source_digest = family_digest(source_bound)
    if archived_source_digest != independently_loaded_source_digest:
        raise RuntimeError("checkpoint family differs from the pinned source atlas")

    before_process = process_state_digest(payload)
    before_rng = rng_digest(payload)

    # Restore everything except family objects and migration bookkeeping.
    _drop_serializable_fields_absent_from_checkpoint(
        engine, engine_fields, _ENGINE_FAMILY_FIELDS | _MIGRATION_FIELDS | {"mpz"}
    )
    _drop_serializable_fields_absent_from_checkpoint(
        engine.mpz, mpz_fields, _MPZ_FAMILY_FIELDS
    )
    for name, value in engine_fields.items():
        if name not in _ENGINE_FAMILY_FIELDS and name not in _MIGRATION_FIELDS:
            setattr(engine, name, copy.deepcopy(value))
    for name, value in mpz_fields.items():
        if name not in _MPZ_FAMILY_FIELDS:
            setattr(engine.mpz, name, copy.deepcopy(value))

    bound_target = target_family.clone_for_engine()
    bound_target.validate_state(engine.mpz)
    engine._state_kernel_family = bound_target
    engine.mpz._signed_kernel = bound_target

    coordinates = dict(getattr(engine, "_signed_last_state_coordinates", {}))
    required = {"r_eff_over_r0", "opening_strength_fraction", "crack_extension_m"}
    if set(coordinates) != required:
        raise RuntimeError("checkpoint has incomplete signed-kernel coordinates")
    bound_target.resolve(**{key: float(coordinates[key]) for key in required})

    # Capture using a local import to avoid a module-level cycle.
    from .sharp_front_v11_branching import _capture_shared_engine
    after_payload = _capture_shared_engine(engine)
    after_process = process_state_digest(after_payload)
    after_rng = rng_digest(after_payload)
    if before_process != after_process:
        raise RuntimeError("restart family migration changed process state")
    if before_rng != after_rng:
        raise RuntimeError("restart family migration changed RNG/threshold state")

    references = {
        "engine": _family_references(engine),
        "mpz": _family_references(engine.mpz),
    }
    all_refs = {id(value) for fields in references.values() for value in fields.values()}
    if all_refs != {id(bound_target)}:
        raise RuntimeError("more than one signed-kernel family remains reachable")
    if set(references["engine"]) != {"_state_kernel_family"}:
        raise RuntimeError("unexpected engine family alias remains reachable")
    if set(references["mpz"]) != {"_signed_kernel"}:
        raise RuntimeError("unexpected MPZ family alias remains reachable")

    audit = {
        "schema": MIGRATION_SCHEMA,
        "qualification": "PASS",
        "migration_count": 1,
        "source_family_sha256": source_family_sha256,
        "target_family_sha256": target_family_sha256,
        "target_family_physics_fingerprint": target_family_physics_fingerprint,
        "source_bound_family_digest": independently_loaded_source_digest,
        "checkpoint_archived_family_digest": archived_source_digest,
        "target_bound_family_digest": family_digest(bound_target),
        "restored_runtime_grid": {
            "n_systems": int(engine.mpz.n_systems),
            "n_bins": int(engine.mpz.n_bins),
            "active_x_sha256": canonical_hash(np.asarray(engine.mpz.x)),
            "wake_x_sha256": canonical_hash(np.asarray(engine.mpz.wake_x)),
        },
        "resolved_checkpoint_coordinates": {
            key: float(coordinates[key]) for key in sorted(required)
        },
        "process_state_sha256_before": before_process,
        "process_state_sha256_after": after_process,
        "rng_threshold_sha256_before": before_rng,
        "rng_threshold_sha256_after": after_rng,
        "engine_mpz_same_bound_family_object": (
            engine._state_kernel_family is engine.mpz._signed_kernel
        ),
        "old_family_reachable_after_migration": False,
        "excluded_archived_family_fields": {
            "engine": sorted(_ENGINE_FAMILY_FIELDS),
            "mpz": sorted(_MPZ_FAMILY_FIELDS),
        },
        "physical_or_stochastic_advance_performed": False,
    }
    engine._restart_family_migration_count = 1
    engine._restart_family_migration_audit = copy.deepcopy(audit)
    return engine, audit


__all__ = [
    "MIGRATION_SCHEMA", "canonical_hash", "family_digest",
    "migrate_shared_engine_family", "process_state_digest", "rng_digest",
]
