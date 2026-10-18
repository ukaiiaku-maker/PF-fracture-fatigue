#!/usr/bin/env python3
"""Verify every manifest entry in a portable R10 twelve-hour archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def verify(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("archive contains duplicate member names")
        manifest = json.loads(archive.read("MANIFEST.json"))
        declared = {row["path"]: row for row in manifest["files"]}
        actual = set(names) - {"MANIFEST.json"}
        if actual != set(declared):
            missing = sorted(set(declared) - actual)
            extra = sorted(actual - set(declared))
            raise RuntimeError(f"manifest/archive mismatch: missing={missing}, extra={extra}")
        total = 0
        for name in sorted(actual):
            digest = hashlib.sha256()
            size = 0
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            row = declared[name]
            if size != int(row["bytes"]):
                raise RuntimeError(f"size mismatch for {name}")
            if digest.hexdigest() != row["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch for {name}")
            total += size
    return {
        "archive": str(path),
        "manifest_schema": manifest.get("schema"),
        "base_revision": manifest.get("base_revision"),
        "head_revision": manifest.get("head_revision"),
        "verified_files": len(declared),
        "verified_uncompressed_bytes": total,
        "status": "PASS",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.archive), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
