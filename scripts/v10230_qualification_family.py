#!/usr/bin/env python3
"""Resolve and fail-closed validate the immutable qualification FEM family."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
DESCRIPTOR = REPOSITORY / "runtime_inputs/v10_2_30/qualification_family_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(descriptor_path: Path = DESCRIPTOR) -> dict:
    descriptor = json.loads(descriptor_path.read_text())
    if descriptor.get("schema") != "v10.2.30_qualification_family_source_v1":
        raise ValueError("unsupported qualification family descriptor")
    path = Path(descriptor["path"])
    if not path.is_file():
        raise ValueError(f"authoritative qualification family is missing: {path}")
    observed_hash = sha256(path)
    if observed_hash != descriptor["sha256"] or path.stat().st_size != descriptor["bytes"]:
        raise ValueError("qualification family content hash or size differs from authority")
    family = json.loads(path.read_text())
    ids = family.get("physical_state_ids")
    levels = family.get("cumulative_crack_path_extension_levels_m")
    if len(family.get("states", [])) != descriptor["expected_state_count"]:
        raise ValueError("qualification family has wrong state count")
    if ids != descriptor["expected_state_ids"] or levels != descriptor["expected_extension_levels_m"]:
        raise ValueError("qualification family state identity or coverage grid differs")
    if max(levels) != descriptor["expected_coverage_m"] or not family.get("state_coverage", {}).get("coverage_passed"):
        raise ValueError("qualification family lacks authoritative 1175 um coverage")
    if not family.get("candidate_independent") or not family.get("production_parameterization_allowed"):
        raise ValueError("qualification family is not authorized for production parameterizations")
    return {**descriptor, "path": str(path.resolve()), "observed_sha256": observed_hash,
            "observed_state_count": len(family["states"]), "observed_coverage_m": max(levels)}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--descriptor", type=Path, default=DESCRIPTOR)
    args = parser.parse_args()
    print(json.dumps(validate(args.descriptor), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
