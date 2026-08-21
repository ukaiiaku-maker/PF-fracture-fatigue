#!/usr/bin/env python3
"""Derive the minimal calibrated common physics needed by v7 transport.

The historical calibrated file is immutable.  This builder verifies its exact
SHA-256, changes only the dimensionless mobile-transport velocity scale from
zero to the existing v7 unscaled value of one, and writes a deterministic
derived input plus a four-model provenance contract.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


PARENT_SHA256 = "59ebf68b366045c91fb08bfe870703c52e1f4e6b51c3f1c480b28dfc8ef8e9c6"
LEGACY1D_SHA256 = "b5d7e54631928c6e7fb5b5d7304e15ec60a9e3e38c7ef5c3d50fc69ea98ec813"
DERIVED_FILENAME = "calibrated_v7_transport_extension.json"
PROVENANCE_FILENAME = "calibrated_v7_transport_extension.provenance.json"
CHANGED_FIELD = "common_physics.mobile_transport_velocity_scale"


def canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def derive_payload(parent_bytes: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_sha = sha256_bytes(parent_bytes)
    if parent_sha != PARENT_SHA256:
        raise ValueError(
            f"historical calibrated common-physics SHA mismatch: {parent_sha}"
        )
    parent = json.loads(parent_bytes)
    old_value = parent["common_physics"]["mobile_transport_velocity_scale"]
    if old_value != 0.0:
        raise ValueError("historical mobile transport is not disabled")

    derived = copy.deepcopy(parent)
    derived["common_physics"]["mobile_transport_velocity_scale"] = 1.0
    derived["derived_configuration"] = {
        "configuration_id": "calibrated_v7_transport_extension",
        "parent_sha256": PARENT_SHA256,
        "changed_common_physics_fields": [
            {
                "field": "mobile_transport_velocity_scale",
                "old_value": 0.0,
                "new_value": 1.0,
                "reason": (
                    "strictly positive velocity scale is mathematically required "
                    "for the existing v7 signed Peierls velocity to transport "
                    "mobile line content; one is its unscaled dimensionless value"
                ),
            }
        ],
        "candidate_parameters_refit": False,
    }
    return parent, derived


def changed_leaf_paths(parent: Any, derived: Any, prefix: str = "") -> list[str]:
    """Return changed leaf paths, ignoring derived-only provenance metadata."""
    if isinstance(parent, dict) and isinstance(derived, dict):
        paths: list[str] = []
        for key in parent:
            child = f"{prefix}.{key}" if prefix else key
            paths.extend(changed_leaf_paths(parent[key], derived[key], child))
        return paths
    return [] if parent == derived else [prefix]


def build(parent_path: Path, output_directory: Path, legacy1d_path: Path | None) -> dict[str, Any]:
    parent_bytes = parent_path.read_bytes()
    parent, derived = derive_payload(parent_bytes)
    changed = changed_leaf_paths(parent, derived)
    if changed != [CHANGED_FIELD]:
        raise RuntimeError(f"derived common physics changed unexpected fields: {changed}")

    output_directory.mkdir(parents=True, exist_ok=True)
    derived_path = output_directory / DERIVED_FILENAME
    derived_bytes = canonical_bytes(derived)
    derived_path.write_bytes(derived_bytes)
    derived_sha = sha256_bytes(derived_bytes)

    legacy_record: dict[str, Any] = {"sha256": LEGACY1D_SHA256}
    if legacy1d_path is not None:
        legacy_sha = sha256_bytes(legacy1d_path.read_bytes())
        if legacy_sha != LEGACY1D_SHA256:
            raise ValueError(f"legacy-1D common-physics SHA mismatch: {legacy_sha}")
        legacy_record["source"] = str(legacy1d_path.resolve())

    provenance = {
        "schema": "v914_calibrated_v7_transport_extension_provenance_v1",
        "configuration_id": "calibrated_v7_transport_extension",
        "parent": {
            "source": str(parent_path.resolve()),
            "sha256": PARENT_SHA256,
            "immutable": True,
        },
        "derived": {
            "source": DERIVED_FILENAME,
            "sha256": derived_sha,
            "changed_fields": [
                {
                    "field": CHANGED_FIELD,
                    "old_value": 0.0,
                    "new_value": 1.0,
                    "reason": derived["derived_configuration"][
                        "changed_common_physics_fields"
                    ][0]["reason"],
                }
            ],
            "all_other_parent_fields_preserved": True,
        },
        "models": {
            "A_HISTORICAL": {
                "physics_sha256": PARENT_SHA256,
                "mobile_transport_enabled": False,
                "return_cancellation": "historical_not_applicable",
            },
            "B_CAL_TRANS_IRREV": {
                "physics_sha256": derived_sha,
                "mobile_transport_enabled": True,
                "return_cancellation": False,
            },
            "C_CAL_TRANS_REV": {
                "physics_sha256": derived_sha,
                "mobile_transport_enabled": True,
                "return_cancellation": True,
            },
            "D_LEGACY1D_V7": {
                **legacy_record,
                "role": "regression_diagnostic_only",
            },
        },
    }
    provenance_path = output_directory / PROVENANCE_FILENAME
    provenance_path.write_bytes(canonical_bytes(provenance))
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--legacy1d", type=Path, default=None)
    args = parser.parse_args()
    provenance = build(args.parent, args.out, args.legacy1d)
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
