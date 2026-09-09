#!/usr/bin/env python3
"""Migrate one accepted V12 checkpoint to the qualified 1600-um family."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.kernel_registry_v10227 import family_physics_fingerprint
from arrhenius_fracture.multifront_checkpoint_v12 import (
    load_accepted_boundary_checkpoint_v12, write_accepted_boundary_checkpoint_v12,
)
from arrhenius_fracture.production_multifront_v12 import accepted_fem_state_fingerprint
from arrhenius_fracture.restart_family_migration_v12 import migrate_runtime_family
from arrhenius_fracture.signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
)
from arrhenius_fracture.stateful_multifront_production_v12 import (
    accepted_context_state_identity, stress_field_identity,
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--source-family", type=Path, required=True)
    parser.add_argument("--expected-source-family-sha256", required=True)
    parser.add_argument("--target-family", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        key: value.expanduser().resolve()
        for key, value in {
            "source_checkpoint": args.source_checkpoint,
            "source_family": args.source_family,
            "target_family": args.target_family,
            "target_checkpoint": args.target_checkpoint,
            "audit": args.audit,
        }.items()
    }
    if paths["target_checkpoint"].exists() or paths["audit"].exists():
        raise FileExistsError("refusing to overwrite migrated V12 products")
    source_checkpoint_sha = sha256(paths["source_checkpoint"])
    if source_checkpoint_sha != args.expected_source_sha256:
        raise RuntimeError("source V12 checkpoint SHA-256 mismatch")
    source_family_sha = sha256(paths["source_family"])
    if source_family_sha != args.expected_source_family_sha256:
        raise RuntimeError("source family SHA-256 mismatch")
    source_family = ActiveOnlySigned2DShieldingKernelFamily.from_json(
        paths["source_family"]
    )
    target_family = ActiveOnlySigned2DShieldingKernelFamily.from_json(
        paths["target_family"]
    )
    source = load_accepted_boundary_checkpoint_v12(paths["source_checkpoint"])
    fem_sha_before = accepted_fem_state_fingerprint(source.accepted_fem_state)
    sigma_sha_before = hashlib.sha256(
        np.ascontiguousarray(source.accepted_stress_field).tobytes()
    ).hexdigest()
    context_sha_before = hashlib.sha256(
        pickle.dumps(dict(source.context_restart), protocol=5)
    ).hexdigest()

    runtime, audit = migrate_runtime_family(
        source.runtime,
        source_family=source_family,
        target_family=target_family,
        source_family_path=paths["source_family"],
        target_family_path=paths["target_family"],
    )
    audit.update({
        "source_checkpoint": str(paths["source_checkpoint"]),
        "source_checkpoint_sha256": source_checkpoint_sha,
        "source_payload_sha256": source.payload_sha256,
        "target_family_physics_fingerprint": family_physics_fingerprint(
            paths["target_family"]
        ),
        "accepted_fem_sha256_before": fem_sha_before,
        "accepted_stress_sha256_before": sigma_sha_before,
        "context_restart_sha256_before": context_sha_before,
        "physical_time_s_before": source.physical_time_s,
        "accepted_opening_m_before": source.accepted_opening_m,
        "step_count_before": source.step_count,
    })
    provenance = dict(runtime.compatibility_provenance or {})
    provenance["v12_1000um_family_migration"] = dict(audit)
    runtime = replace(runtime, compatibility_provenance=provenance)
    accepted_id = accepted_context_state_identity(
        source.accepted_fem_state, runtime,
        physical_time_s=source.physical_time_s,
        accepted_opening_m=source.accepted_opening_m,
        step_count=source.step_count,
    )
    stress_id = stress_field_identity(
        source.accepted_fem_state, source.accepted_stress_field,
        mechanics_source_identity=source.mechanics_source_identity,
        accepted_state_id=accepted_id,
    )
    runtime = replace(
        runtime, accepted_state_id=accepted_id, stress_field_state_id=stress_id
    )
    paths["target_checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    record = write_accepted_boundary_checkpoint_v12(
        runtime, paths["target_checkpoint"],
        accepted_fem_state=source.accepted_fem_state,
        accepted_stress_field=source.accepted_stress_field,
        mechanics_source_identity=source.mechanics_source_identity,
        physical_time_s=source.physical_time_s,
        accepted_opening_m=source.accepted_opening_m,
        step_count=source.step_count,
        context_restart=source.context_restart,
    )
    restored = load_accepted_boundary_checkpoint_v12(paths["target_checkpoint"])
    audit.update({
        "target_checkpoint": str(paths["target_checkpoint"]),
        "target_checkpoint_sha256": sha256(paths["target_checkpoint"]),
        "target_payload_sha256": restored.payload_sha256,
        "target_runtime_sha256": restored.runtime.registry_fingerprint,
        "accepted_fem_sha256_after": accepted_fem_state_fingerprint(
            restored.accepted_fem_state
        ),
        "accepted_stress_sha256_after": hashlib.sha256(
            np.ascontiguousarray(restored.accepted_stress_field).tobytes()
        ).hexdigest(),
        "context_restart_sha256_after": hashlib.sha256(
            pickle.dumps(dict(restored.context_restart), protocol=5)
        ).hexdigest(),
        "physical_time_s_after": restored.physical_time_s,
        "accepted_opening_m_after": restored.accepted_opening_m,
        "step_count_after": restored.step_count,
        "accepted_boundary_reload_passed": True,
        "accepted_fem_exact": (
            fem_sha_before
            == accepted_fem_state_fingerprint(restored.accepted_fem_state)
        ),
        "accepted_stress_exact": sigma_sha_before == hashlib.sha256(
            np.ascontiguousarray(restored.accepted_stress_field).tobytes()
        ).hexdigest(),
        "context_restart_exact": context_sha_before == hashlib.sha256(
            pickle.dumps(dict(restored.context_restart), protocol=5)
        ).hexdigest(),
        "time_load_step_exact": (
            restored.physical_time_s == source.physical_time_s
            and restored.accepted_opening_m == source.accepted_opening_m
            and restored.step_count == source.step_count
        ),
        "writer_record": record,
    })
    paths["audit"].parent.mkdir(parents=True, exist_ok=True)
    paths["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
