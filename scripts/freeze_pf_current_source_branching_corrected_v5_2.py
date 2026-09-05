#!/usr/bin/env python3
"""Freeze an immutable content manifest for a completed V5.2 raw run tree."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.raw_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"raw root is not a directory: {root}")
    if args.output.resolve().is_relative_to(root):
        raise SystemExit("freeze manifest must be outside the raw tree")

    rows = []
    tree = hashlib.sha256()
    total_bytes = 0
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = file_sha256(path)
        rows.append({"path": relative, "bytes": size, "sha256": digest})
        encoded = relative.encode()
        tree.update(len(encoded).to_bytes(8, "big"))
        tree.update(encoded)
        tree.update(size.to_bytes(8, "big"))
        tree.update(bytes.fromhex(digest))
        total_bytes += size

    payload = {
        "schema": "pf_branching_v5_2_raw_tree_freeze_v1",
        "raw_root": str(root),
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "tree_sha256": tree.hexdigest(),
        "files": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("raw_root", "file_count", "total_bytes", "tree_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
