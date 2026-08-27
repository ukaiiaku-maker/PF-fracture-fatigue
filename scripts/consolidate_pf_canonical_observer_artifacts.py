#!/usr/bin/env python3
"""Fail-closed consolidation of redundant canonical PF observer artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


SOURCE_NAMES = (
    "anisotropic_emission_audit_v10174.json.zst",
    "kinetic_tip_cell_audit_v101.json.zst",
    "v10_2_17_final_signed_stochastic_stack.json.zst",
)
DEST_NAME = "canonical_pf_state_observer_v2.json.zst"
MANIFEST_NAME = "canonical_pf_state_observer_v2_manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: object) -> bytes:
    # Production audit JSON uses Infinity sentinels for disabled rates.
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=True) + "\n").encode()


def decompress(path: Path) -> bytes:
    return subprocess.run(
        ["zstd", "-q", "-dc", str(path)], check=True, capture_output=True
    ).stdout


def compress(raw_path: Path, dest_path: Path) -> None:
    subprocess.run(
        ["zstd", "-q", "-f", "-10", str(raw_path), "-o", str(dest_path)],
        check=True,
    )


def consolidate_case(case: Path) -> str:
    """Consolidate one complete case, returning a restart-safe status."""
    dest = case / DEST_NAME
    manifest_path = case / MANIFEST_NAME
    sources = [case / name for name in SOURCE_NAMES]
    if dest.exists() and manifest_path.exists():
        if any(path.exists() for path in sources):
            raise RuntimeError(f"partial consolidation state in {case}")
        manifest = json.loads(manifest_path.read_text())
        if sha256_bytes(decompress(dest)) != manifest["canonical_decompressed_sha256"]:
            raise RuntimeError(f"canonical hash mismatch in {case}")
        return "already_consolidated"
    if not all(path.exists() for path in sources):
        return "not_ready"
    if not (case / "canonical_case_result.json").is_file():
        return "not_complete"
    result = json.loads((case / "canonical_case_result.json").read_text())
    if result.get("status") not in {"COMPLETE", "REUSED_COMPLETE"}:
        return "not_complete"

    raws = [decompress(path) for path in sources]
    docs = [json.loads(raw) for raw in raws]
    if not all(isinstance(doc, dict) for doc in docs):
        raise RuntimeError(f"non-object observer artifact in {case}")
    records = docs[0].get("records")
    if any(doc.get("records") != records for doc in docs[1:]):
        raise RuntimeError(f"observer records differ in {case}")

    merged: dict[str, Any] = dict(docs[2])
    source_schemas = {}
    for path, doc in zip(sources, docs):
        source_schemas[path.name] = doc.get("schema")
        for key, value in doc.items():
            if key == "schema":
                continue
            if key in merged and merged[key] != value:
                raise RuntimeError(f"shared key {key!r} differs in {case}")
            merged.setdefault(key, value)

    provenance = {
        "schema": "canonical_pf_state_observer_consolidation_v2",
        "method": "exact_json_value_union_after_fail_closed_shared_key_equality",
        "records_exactly_equal_across_sources": True,
        "records_canonical_sha256": sha256_bytes(canonical_bytes(records)),
        "source_artifacts": [
            {
                "name": path.name,
                "compressed_sha256": sha256_bytes(path.read_bytes()),
                "compressed_size_bytes": path.stat().st_size,
                "decompressed_sha256": sha256_bytes(raw),
                "decompressed_size_bytes": len(raw),
                "schema": doc.get("schema"),
            }
            for path, raw, doc in zip(sources, raws, docs)
        ],
        "source_schemas": source_schemas,
        "source_files_removed_only_after_verification": True,
    }
    merged["canonical_observer_consolidation"] = provenance
    output = canonical_bytes(merged)
    with tempfile.TemporaryDirectory(dir=case) as temp_dir:
        raw_path = Path(temp_dir) / "canonical_pf_state_observer_v2.json"
        zst_path = Path(temp_dir) / DEST_NAME
        raw_path.write_bytes(output)
        compress(raw_path, zst_path)
        subprocess.run(["zstd", "-q", "-t", str(zst_path)], check=True)
        if decompress(zst_path) != output:
            raise RuntimeError(f"round-trip mismatch in {case}")
        os.replace(zst_path, dest)

    manifest = {
        **provenance,
        "canonical_artifact": DEST_NAME,
        "canonical_compressed_sha256": sha256_bytes(dest.read_bytes()),
        "canonical_compressed_size_bytes": dest.stat().st_size,
        "canonical_decompressed_sha256": sha256_bytes(output),
        "canonical_decompressed_size_bytes": len(output),
        "union_keys": sorted(merged),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    check_manifest = json.loads(manifest_path.read_text())
    if sha256_bytes(decompress(dest)) != check_manifest["canonical_decompressed_sha256"]:
        raise RuntimeError(f"durable verification failed in {case}")
    for path in sources:
        path.unlink()
    return "consolidated"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    counts: dict[str, int] = {}
    for case in sorted(path for path in args.root.iterdir() if path.is_dir()):
        status = consolidate_case(case)
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
