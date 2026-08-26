#!/usr/bin/env python3
"""Create and verify a full recoverable archive of one scoped legacy run tree."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


REQUIRED_NAMES = {
    "run_args.json",
    "v10_2_27_case_contract.json",
    "v10_2_22_parameter_selection.json",
    "selected_material_manifest_v10_2_22.csv",
    "stage3_case_status.json",
    "stochastic_avalanche_geometry_events.json",
    "summary.json",
    "COMPLETE",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def members(source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(source)
        rows.append({
            "relative_path": rel.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "required_preservation_member": (
                path.name in REQUIRED_NAMES
                or path.name.startswith("steps_") and path.suffix == ".csv"
                or path.name.startswith("crack_path_") and path.suffix == ".csv"
            ),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assert_safe_source(source: Path, runs_root: Path, worktrees: set[Path]) -> None:
    source = source.resolve()
    runs_root = runs_root.resolve()
    if source.parent != runs_root:
        raise RuntimeError("legacy archive source must be one direct child of runs root")
    if source in worktrees or any(source == root or source in root.parents for root in worktrees):
        raise RuntimeError("refusing to archive a Git worktree as a run directory")
    if not source.is_dir():
        raise RuntimeError("legacy archive source is not a directory")


def git_worktrees(repo: Path) -> set[Path]:
    output = subprocess.check_output(
        ["git", "-C", str(repo), "worktree", "list", "--porcelain"], text=True
    )
    return {
        Path(line.split(" ", 1)[1]).resolve()
        for line in output.splitlines() if line.startswith("worktree ")
    }


def create_archive(source: Path, archive_path: Path, level: int) -> None:
    if archive_path.exists():
        raise RuntimeError(f"archive already exists: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tar = subprocess.Popen(
        ["tar", "-C", str(source.parent), "-cf", "-", source.name],
        stdout=subprocess.PIPE,
    )
    assert tar.stdout is not None
    zstd = subprocess.run(
        ["zstd", f"-{level}", "-T0", "-o", str(archive_path)],
        stdin=tar.stdout,
        check=False,
    )
    tar.stdout.close()
    tar_rc = tar.wait()
    if tar_rc != 0 or zstd.returncode != 0:
        raise RuntimeError(f"archive creation failed: tar={tar_rc} zstd={zstd.returncode}")


def verify_full_extraction(source: Path, archive_path: Path,
                           expected: list[dict[str, Any]]) -> dict[str, Any]:
    subprocess.run(["zstd", "-t", str(archive_path)], check=True)
    with tempfile.TemporaryDirectory(prefix="pf-legacy-verify-") as temp:
        root = Path(temp)
        zstd = subprocess.Popen(["zstd", "-dc", str(archive_path)], stdout=subprocess.PIPE)
        assert zstd.stdout is not None
        untar = subprocess.run(["tar", "-C", str(root), "-xf", "-"],
                               stdin=zstd.stdout, check=False)
        zstd.stdout.close()
        zstd_rc = zstd.wait()
        if zstd_rc != 0 or untar.returncode != 0:
            raise RuntimeError("archive test extraction failed")
        extracted = root / source.name
        checked = 0
        for row in expected:
            path = extracted / row["relative_path"]
            if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
                raise RuntimeError(f"extracted member mismatch: {path}")
            if sha256(path) != row["sha256"]:
                raise RuntimeError(f"extracted hash mismatch: {path}")
            checked += 1
    return {"full_test_extraction": True, "verified_member_count": checked}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--pf-repo", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--compression-level", type=int, default=9)
    args = parser.parse_args()
    source = args.source.resolve()
    assert_safe_source(source, args.runs_root, git_worktrees(args.pf_repo))
    manifest_dir = args.manifest_dir.resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows = members(source)
    if not rows:
        raise RuntimeError("refusing to archive an empty source")
    write_csv(manifest_dir / "pf_legacy_archive_members.csv", rows)
    source_bytes = sum(int(row["size_bytes"]) for row in rows)
    create_archive(source, args.archive.resolve(), args.compression_level)
    verification = verify_full_extraction(source, args.archive.resolve(), rows)
    payload = {
        "schema": "pf_verified_legacy_archive_v1",
        "source_path": str(source),
        "archive_path": str(args.archive.resolve()),
        "source_file_count": len(rows),
        "source_size_bytes": source_bytes,
        "archive_size_bytes": args.archive.stat().st_size,
        "archive_sha256": sha256(args.archive.resolve()),
        "compression_ratio": 1.0 - args.archive.stat().st_size / source_bytes,
        "archive_format": "tar.zst",
        "test_extraction": verification,
        "source_deleted": False,
        "eligible_for_source_deletion_after_archive_is_moved_to_scoped_runs_root": True,
    }
    (manifest_dir / "pf_legacy_archive_hashes.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (manifest_dir / "PF_LEGACY_ARCHIVE_MANIFEST.md").write_text(
        "# PF legacy archive manifest\n\n"
        f"- Source: `{source}`\n"
        f"- Archive: `{args.archive.resolve()}`\n"
        f"- Files: {len(rows)}\n"
        f"- Source bytes: {source_bytes}\n"
        f"- Archive bytes: {args.archive.stat().st_size}\n"
        f"- SHA-256: `{payload['archive_sha256']}`\n"
        "- Full test extraction: passed; every member size and SHA-256 reproduced.\n"
        "- Source deletion status: not performed by this utility.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
