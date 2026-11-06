"""Restore immutable Ast input fixtures into ignored runtime paths.

No solver or optimizer is executed. Existing differing files are rejected.
Run from any directory; the checkout root is resolved from this script.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures/ast2018_per_temperature_handoff_20261104"


def write_identical_or_new(path: Path, content: bytes) -> bool:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Existing input differs; preserve and inspect it: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return True


def prepare() -> int:
    manifest = json.loads((FIXTURES / "input_manifest.json").read_text())
    archive_path = ROOT / manifest["experimental_archive"]["path"]
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != manifest["experimental_archive"]["sha256"]:
        raise RuntimeError("Experimental extraction archive hash differs")
    writes = 0
    for row in manifest["inputs"]:
        data = (FIXTURES / row["fixture"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise RuntimeError(f"Fixture hash differs: {row['fixture']}")
        destination = (ROOT / row["destination"]).resolve()
        if not destination.is_relative_to(ROOT / "runs"):
            raise RuntimeError("Fixture destination must be inside ignored runs/")
        writes += write_identical_or_new(destination, data)
    data_root = ROOT / "runs/ast2018_shape_fit_v5/production/input"
    with ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if name.endswith("/") or not name.startswith("Ast2018_joint_fit_data/"):
                continue
            destination = (data_root / name).resolve()
            if not destination.is_relative_to(data_root):
                raise RuntimeError("Experimental archive path leaves input directory")
            writes += write_identical_or_new(destination, archive.read(name))
    print(f"Ast input preparation complete: {writes} files created; no forward calls")
    return writes


if __name__ == "__main__":
    prepare()
