#!/usr/bin/env python3
"""Build and fail-closed verify the complete deterministic V5.1 archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


MANIFEST = "pf_branching_v5_1_archive_manifest.json"
ARCHIVE_NAME = "Archive_PF_BRANCHING_V5_1_COMPLETE.zip"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name and not path.is_absolute() and ".." not in path.parts)


def source_members(source: Path) -> dict[str, Path]:
    excluded = {ARCHIVE_NAME, MANIFEST}
    members = {
        path.relative_to(source).as_posix(): path
        for path in sorted(item for item in source.rglob("*") if item.is_file())
        if path.name not in excluded
    }
    if not members:
        raise RuntimeError("refusing to build an empty V5.1 archive")
    return members


def build_archive(source: Path, archive: Path) -> dict:
    source, archive = source.resolve(), archive.resolve()
    members = source_members(source)
    manifest = {
        "schema": "pf_branching_v5_1_complete_archive_manifest",
        "archive_root": ".",
        "declared_file_count": len(members),
        "files_sha256": {name: sha256(path) for name, path in members.items()},
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9,
    ) as stream:
        for name, data in [(MANIFEST, manifest_bytes), *(
            (name, path.read_bytes()) for name, path in members.items()
        )]:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            stream.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return verify_archive(archive)


def verify_archive(archive: Path) -> dict:
    archive = archive.resolve()
    if not archive.is_file():
        raise RuntimeError(f"archive is missing: {archive}")
    with zipfile.ZipFile(archive, "r") as stream:
        names = stream.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("archive contains duplicate paths")
        if any(not _safe_relative(name) for name in names):
            raise RuntimeError("archive contains an unsafe path")
        if MANIFEST not in names:
            raise RuntimeError("archive manifest is missing")
        manifest = json.loads(stream.read(MANIFEST))
        declared = manifest.get("files_sha256", {})
        actual = set(names) - {MANIFEST}
        if set(declared) != actual:
            missing = sorted(set(declared) - actual)
            undeclared = sorted(actual - set(declared))
            raise RuntimeError(
                f"archive presence mismatch: missing={missing}, undeclared={undeclared}"
            )
        for name, expected in declared.items():
            observed = sha256_bytes(stream.read(name))
            if observed != expected:
                raise RuntimeError(f"archive member SHA-256 mismatch: {name}")
        required = {
            "PF_CURRENT_SOURCE_BRANCHING_EXECUTION_SEAL_V5_1.md",
            "pf_branching_v5_execution_source.patch",
            "pf_branching_v5_record_delta.patch",
            "pf_branching_v5_1_source_provenance.json",
            "pf_branching_v5_1_archive_presence_audit.json",
            "pf_branching_v5_1_tensor_conditioning_audit.csv",
            "pf_branching_qualified_daughter_stop_audit.json",
            "pf_branching_matched_pair_launch_preflight.json",
            "run_pf_current_source_branching_corrected_pair_v5_1.py",
            "pinned_inputs/theta40_signed_kernel_family.json",
            "pinned_inputs/theta40_mechanical_configuration.json",
            "immutable_source/pf_branching_v5_execution_source_725709f.tar",
        }
        absent = sorted(required - actual)
        if absent:
            raise RuntimeError(f"archive required members missing: {absent}")
        preflight = json.loads(stream.read("pf_branching_matched_pair_launch_preflight.json"))
        family_hash = sha256_bytes(stream.read(
            "pinned_inputs/theta40_signed_kernel_family.json"
        ))
        mechanical_hash = sha256_bytes(stream.read(
            "pinned_inputs/theta40_mechanical_configuration.json"
        ))
        for case in preflight["cases"]:
            if case["signed_kernel_family_sha256"] != family_hash:
                raise RuntimeError("launcher family hash does not match pinned archive input")
            if case["mechanical_configuration_sha256"] != mechanical_hash:
                raise RuntimeError(
                    "launcher mechanical hash does not match pinned archive input"
                )
        snapshots = [name for name in actual if name.startswith("source_snapshot_")]
        provenance = json.loads(stream.read("pf_branching_v5_1_source_provenance.json"))
        declared_snapshots = set(provenance["source_snapshot_sha256"])
        if set(snapshots) != declared_snapshots:
            raise RuntimeError("undeclared or absent source snapshot")
    return {
        "schema": "pf_branching_v5_1_archive_verification",
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "declared_file_count": len(declared),
        "all_manifest_paths_exist": True,
        "all_hashes_match": True,
        "no_undeclared_source_snapshots": True,
        "pinned_input_hashes_match_launcher": True,
        "result": "PASS",
    }


def rewrite_archive_for_negative_test(
    source_archive: Path, destination: Path, *, remove: str | None = None,
    modify: str | None = None,
) -> None:
    """Test-only helper that preserves the original manifest while corrupting content."""
    with zipfile.ZipFile(source_archive, "r") as source, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED,
    ) as target:
        for info in source.infolist():
            if info.filename == remove:
                continue
            data = source.read(info.filename)
            if info.filename == modify:
                data += b"\nCORRUPTED\n"
            target.writestr(info, data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--archive", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (
        build_archive(args.source, args.archive)
        if args.operation == "build" else verify_archive(args.archive)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
