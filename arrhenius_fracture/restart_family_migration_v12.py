"""One-shot V12 accepted-boundary signed-family migration.

The migration changes only the directly serialized family aliases and their
explicit identity metadata. Physical, kinetic, topology, ledger, clock, and
RNG state are preserved exactly.
"""
from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np

from .general_multifront_v12 import (
    MultiFrontRuntimeState, ProcessEngineState, ProcessRegionReservoir,
    canonical_hash,
)
from .restart_family_migration_v11 import (
    family_digest, process_state_digest, rng_digest,
)
from .signed_kernel_family_v10214 import ActiveOnlySigned2DShieldingKernelFamily


SCHEMA = "v12.accepted-boundary-family-migration/1"


def _sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _grid(payload: Mapping[str, Any]) -> Any:
    mpz = payload["mpz_fields"]
    return type("RestoredV12Grid", (), {
        "n_systems": int(mpz["n_systems"]),
        "n_bins": int(mpz["n_bins"]),
        "x": np.asarray(mpz["x"]),
        "wake_x": np.asarray(mpz["wake_x"]),
        "wake_n_bins": int(mpz["wake_n_bins"]),
        "site_capacity": np.asarray(mpz["site_capacity"]),
    })()


def _migrate_engine(
    state: ProcessEngineState,
    *,
    source_family: ActiveOnlySigned2DShieldingKernelFamily,
    target_family: ActiveOnlySigned2DShieldingKernelFamily,
    target_identity: str,
) -> tuple[ProcessEngineState, dict[str, Any]]:
    payload = state.complete_checkpoint_payload()
    engine_fields = payload["engine_fields"]
    mpz_fields = payload["mpz_fields"]
    archived = engine_fields.get("_state_kernel_family")
    if archived is not mpz_fields.get("_signed_kernel"):
        raise RuntimeError("V12 engine and MPZ archived-family aliases differ")
    source_bound = source_family.clone_for_engine()
    source_bound.validate_state(_grid(payload))
    archived_digest = family_digest(archived)
    source_digest = family_digest(source_bound)
    if archived_digest != source_digest:
        raise RuntimeError("V12 engine family differs from the pinned source family")

    process_before = process_state_digest(payload)
    rng_before = rng_digest(payload)
    target_bound = target_family.clone_for_engine()
    target_bound.validate_state(_grid(payload))
    coordinates = copy.deepcopy(
        engine_fields.get("_signed_last_state_coordinates", {})
    )
    required = {"r_eff_over_r0", "opening_strength_fraction", "crack_extension_m"}
    if set(coordinates) != required:
        raise RuntimeError("V12 engine has incomplete signed-family coordinates")
    target_bound.resolve(**{key: float(coordinates[key]) for key in required})
    migrated_payload = copy.deepcopy(payload)
    migrated_payload["engine_fields"]["_state_kernel_family"] = target_bound
    migrated_payload["mpz_fields"]["_signed_kernel"] = target_bound
    if process_state_digest(migrated_payload) != process_before:
        raise RuntimeError("V12 family migration changed process state")
    if rng_digest(migrated_payload) != rng_before:
        raise RuntimeError("V12 family migration changed RNG/threshold state")
    migrated = ProcessEngineState.from_v11_payload(
        engine_id=state.engine_id,
        source_state_id=state.source_state_id,
        payload=migrated_payload,
        active_ledgers=state.active_ledgers,
        wake_ledgers=state.wake_ledgers,
        signed_system_ledgers=state.signed_system_ledgers,
        update_count=state.update_count,
        event_renewal_count=state.event_renewal_count,
        local_process_coordinate_m=state.local_process_coordinate_m,
        family_identity=target_identity,
    )
    check = migrated.complete_checkpoint_payload()
    if (
        check["engine_fields"]["_state_kernel_family"]
        is not check["mpz_fields"]["_signed_kernel"]
    ):
        raise RuntimeError("migrated V12 family aliases are not identical")
    return migrated, {
        "engine_id": state.engine_id,
        "source_state_id": state.source_state_id,
        "source_bound_family_digest": source_digest,
        "checkpoint_archived_family_digest": archived_digest,
        "target_bound_family_digest": family_digest(target_bound),
        "process_state_sha256_before": process_before,
        "process_state_sha256_after": process_state_digest(check),
        "rng_threshold_sha256_before": rng_before,
        "rng_threshold_sha256_after": rng_digest(check),
        "resolved_coordinates": {
            key: float(coordinates[key]) for key in sorted(coordinates)
        },
        "old_direct_family_alias_reachable_after_migration": False,
        "engine_mpz_same_target_family_object": True,
    }


def migrate_runtime_family(
    runtime: MultiFrontRuntimeState,
    *,
    source_family: ActiveOnlySigned2DShieldingKernelFamily,
    target_family: ActiveOnlySigned2DShieldingKernelFamily,
    source_family_path: str | Path,
    target_family_path: str | Path,
) -> tuple[MultiFrontRuntimeState, dict[str, Any]]:
    """Migrate every active and archived owner without advancing state."""
    target_identity = str(Path(target_family_path).expanduser().resolve())
    active: dict[str, ProcessEngineState] = {}
    engine_audits = []
    for engine_id, engine in sorted(runtime.process_engines.items()):
        active[engine_id], audit = _migrate_engine(
            engine, source_family=source_family, target_family=target_family,
            target_identity=target_identity,
        )
        audit["storage_role"] = "active_owner"
        engine_audits.append(audit)
    reservoirs: dict[str, ProcessRegionReservoir] = {}
    for reservoir_id, reservoir in sorted(runtime.reservoirs.items()):
        archived, audit = _migrate_engine(
            reservoir.archived_engine,
            source_family=source_family, target_family=target_family,
            target_identity=target_identity,
        )
        reservoirs[reservoir_id] = replace(reservoir, archived_engine=archived)
        audit["storage_role"] = "archived_reservoir"
        audit["reservoir_id"] = reservoir_id
        engine_audits.append(audit)

    provenance_before = copy.deepcopy(dict(runtime.compatibility_provenance or {}))
    migration = {
        "schema": SCHEMA,
        "qualification": "PASS",
        "migration_count": 1,
        "source_family": str(Path(source_family_path).expanduser().resolve()),
        "source_family_sha256": _sha_file(source_family_path),
        "target_family": target_identity,
        "target_family_sha256": _sha_file(target_family_path),
        "active_owner_count": len(active),
        "archived_reservoir_count": len(reservoirs),
        "engines": engine_audits,
        "physical_or_stochastic_advance_performed": False,
        "old_serialized_direct_family_alias_reachable": False,
    }
    provenance_after = copy.deepcopy(provenance_before)
    provenance_after["family_identity"] = target_identity
    provenance_after["source_provider_cache_identity"] = target_identity
    provenance_after["v12_1000um_family_migration"] = copy.deepcopy(migration)
    migrated = replace(
        runtime,
        process_engines=active,
        reservoirs=reservoirs,
        compatibility_provenance=provenance_after,
    )
    migrated.validate()
    migration["registry_sha256_before"] = runtime.registry_fingerprint
    migration["registry_sha256_after"] = migrated.registry_fingerprint
    migration["topology_sha256_before"] = runtime.topology_fingerprint
    migration["topology_sha256_after"] = migrated.topology_fingerprint
    migration["scheduler_sha256_before"] = canonical_hash(runtime.scheduler.to_dict())
    migration["scheduler_sha256_after"] = canonical_hash(migrated.scheduler.to_dict())
    migration["front_runtime_sha256_before"] = canonical_hash({
        key: value.to_dict() for key, value in runtime.front_runtimes.items()
    })
    migration["front_runtime_sha256_after"] = canonical_hash({
        key: value.to_dict() for key, value in migrated.front_runtimes.items()
    })
    migration["crack_network_exact"] = (
        runtime.crack_network.to_dict() == migrated.crack_network.to_dict()
    )
    migration["resource_policy_exact"] = (
        runtime.resource_policy.to_dict() == migrated.resource_policy.to_dict()
    )
    migration["ledger_totals_exact"] = (
        runtime.total_conserved_ledgers() == migrated.total_conserved_ledgers()
        and runtime.total_signed_system_ledgers()
        == migrated.total_signed_system_ledgers()
    )
    return migrated, migration


__all__ = ["SCHEMA", "migrate_runtime_family"]
