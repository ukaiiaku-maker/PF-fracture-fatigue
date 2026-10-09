#!/usr/bin/env python3
"""Strict cross-worker comparison for V5 finalization-v2 evidence trees."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SERIALIZATION_HASH_KEYS = {"state_sha256", "production_void_state_sha256"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_manifest(root: Path, relative: str) -> None:
    path = root / relative
    manifest = json.loads(path.read_text())
    for name, expected in manifest.items():
        target = path.parent / name
        if not target.is_file():
            raise AssertionError(f"{root}: manifest target missing: {target}")
        actual = _sha256(target)
        if actual != expected:
            raise AssertionError(f"{root}: manifest mismatch: {target}: {actual} != {expected}")


def _compare(left, right, path: str = "$") -> None:
    if isinstance(left, bool) or isinstance(right, bool):
        if left is not right:
            raise AssertionError(f"{path}: {left!r} != {right!r}")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-15):
            raise AssertionError(f"{path}: {left!r} != {right!r}")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{path}: keys differ")
        for key in left:
            if key not in SERIALIZATION_HASH_KEYS:
                _compare(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise AssertionError(f"{path}: lengths differ")
        for index, (a, b) in enumerate(zip(left, right)):
            _compare(a, b, f"{path}[{index}]")
        return
    if left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")


def compare_trees(left: Path, right: Path) -> None:
    for root in (left, right):
        _validate_manifest(root, "sha256_manifest.json")
        _validate_manifest(root, "static_mechanics/sha256_manifest.json")
    left_files = {p.relative_to(left).as_posix() for p in left.rglob("*") if p.is_file()}
    right_files = {p.relative_to(right).as_posix() for p in right.rglob("*") if p.is_file()}
    if left_files != right_files:
        raise AssertionError(f"file sets differ: {sorted(left_files ^ right_files)}")
    for name in sorted(left_files):
        if name.endswith(".state.pkl") or name.endswith("sha256_manifest.json"):
            continue
        if not name.endswith(".json"):
            if _sha256(left / name) != _sha256(right / name):
                raise AssertionError(f"{name}: binary content differs")
            continue
        _compare(json.loads((left / name).read_text()), json.loads((right / name).read_text()), name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    compare_trees(args.left, args.right)
    print("V5 finalization-v2 canonical comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
