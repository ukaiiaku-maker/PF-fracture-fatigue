#!/usr/bin/env python3
"""Verify two independent append-only theta-40 kernel builds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

NEW_STATE_IDS = ("E0000420", "E0000425", "E0000600", "E0000745")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    primary = args.primary.expanduser().resolve()
    reproduction = args.reproduction.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)

    manifests = [
        _load(root / "append_only_kernel_build_manifest.json")
        for root in (primary, reproduction)
    ]
    if not all(manifest.get("passed") is True for manifest in manifests):
        raise RuntimeError("at least one append-only build is not qualified")
    if manifests[0]["producer_code_commit"] != manifests[1]["producer_code_commit"]:
        raise RuntimeError("reproduction used a different producer commit")

    comparisons = []
    relative_paths = [Path("family.json")]
    for state_id in NEW_STATE_IDS:
        relative_paths.extend(
            [
                Path("new_prescribed_geometry_snapshots") / state_id / "snapshot.json",
                Path("new_prescribed_geometry_snapshots") / state_id / "state_arrays.npz",
                Path("load_invariance") / state_id / "active_station_responses_load_1.csv",
            ]
        )
    for relative in relative_paths:
        first = primary / relative
        second = reproduction / relative
        first_sha = _sha(first)
        second_sha = _sha(second)
        if first_sha != second_sha or first.read_bytes() != second.read_bytes():
            raise RuntimeError(f"reproducibility failed for {relative}")
        comparisons.append(
            {
                "path": str(relative),
                "sha256": first_sha,
                "byte_identical": True,
            }
        )

    payload = {
        "schema": "pf_branching_theta40_append_only_kernel_reproducibility_v1",
        "passed": True,
        "primary_root": str(primary),
        "reproduction_root": str(reproduction),
        "producer_code_commit": manifests[0]["producer_code_commit"],
        "family_sha256": manifests[0]["new_family_sha256"],
        "family_physics_fingerprint": manifests[0][
            "new_family_physics_fingerprint"
        ],
        "family_hash_equal": (
            manifests[0]["new_family_sha256"]
            == manifests[1]["new_family_sha256"]
        ),
        "physics_fingerprint_equal": (
            manifests[0]["new_family_physics_fingerprint"]
            == manifests[1]["new_family_physics_fingerprint"]
        ),
        "new_state_arrays_byte_identical": True,
        "new_state_response_tables_byte_identical": True,
        "comparisons": comparisons,
    }
    if not payload["family_hash_equal"] or not payload["physics_fingerprint_equal"]:
        raise RuntimeError("family identity differs between deterministic builds")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
