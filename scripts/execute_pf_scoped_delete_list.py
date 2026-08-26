#!/usr/bin/env python3
"""Fail-closed executor for explicit, verified PF run deletion manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import shutil
import subprocess


ALLOWED_CLASSIFICATIONS = {
    "DELETE_EXACT_DUPLICATE",
    "DELETE_FAILED_OR_EMPTY",
    "DELETE_REGENERABLE_CACHE",
    "DELETE_SUPERSEDED_EXTRACTED_COPY",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def worktrees(repo: Path) -> set[Path]:
    output = subprocess.check_output(
        ["git", "-C", str(repo), "worktree", "list", "--porcelain"], text=True
    )
    return {Path(line.split(" ", 1)[1]).resolve()
            for line in output.splitlines() if line.startswith("worktree ")}


def active_commands() -> str:
    return subprocess.check_output(["ps", "-axo", "command="], text=True)


def validate_row(row: dict[str, str], runs_root: Path, repo: Path,
                 member_manifest: Path) -> tuple[Path, Path]:
    target = Path(row["absolute_path"]).resolve()
    runs_root = runs_root.resolve()
    if target.parent != runs_root:
        raise RuntimeError("delete target must be one direct child of scoped runs root")
    if row["classification"] not in ALLOWED_CLASSIFICATIONS:
        raise RuntimeError("delete classification is not allowed")
    if row.get("status") != "ELIGIBLE_AFTER_EXPLICIT_SAFETY_CHECK":
        raise RuntimeError("delete row is not explicitly eligible")
    if not target.is_dir():
        raise RuntimeError("delete target is absent or not a directory")
    git_roots = worktrees(repo)
    if target in git_roots or any(target == root or target in root.parents for root in git_roots):
        raise RuntimeError("refusing to delete a Git worktree")
    if str(target) in active_commands():
        raise RuntimeError("refusing to delete a path referenced by an active process")
    archive = Path(row["replacement_or_archive"]).resolve()
    if archive.parent != runs_root or not archive.is_file():
        raise RuntimeError("replacement archive is absent or outside scoped runs root")
    if row["classification"] == "DELETE_SUPERSEDED_EXTRACTED_COPY":
        if sha256(member_manifest.resolve()) != row["content_manifest_sha256"]:
            raise RuntimeError("member-manifest hash mismatch")
    return target, archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete-list", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--pf-repo", type=Path, required=True)
    parser.add_argument("--member-manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    with args.delete_list.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("delete list is empty")
    validated = [
        validate_row(row, args.runs_root, args.pf_repo, args.member_manifest)
        for row in rows
    ]
    if not args.execute:
        for target, archive in validated:
            print(f"VALIDATED {target} -> {archive}")
        return 0
    for target, archive in validated:
        shutil.rmtree(target)
        if target.exists():
            raise RuntimeError(f"delete target remains after removal: {target}")
        print(f"DELETED_VERIFIED_EXTRACTED_COPY {target} archive={archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
