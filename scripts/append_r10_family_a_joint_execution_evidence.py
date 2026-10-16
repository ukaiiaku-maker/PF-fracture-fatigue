#!/usr/bin/env python3
"""Append the executed Family-A 300/450 K gate to the existing R10 receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.destination.is_dir():
        raise FileNotFoundError(args.destination)

    copy(ROOT / "scripts/analyze_r10_family_a_joint_execution.py",
         args.destination / "source/scripts/analyze_r10_family_a_joint_execution.py")
    analysis = args.run_root / "joint_execution"
    copy(analysis / "family_a_joint_fracture_screen.csv",
         args.destination / "executed/family_a_joint_execution/family_a_joint_fracture_screen.csv")
    copy(analysis / "joint_execution_decision.json",
         args.destination / "executed/family_a_joint_execution/joint_execution_decision.json")
    for label in ("open", "emit_m060", "emit_p060"):
        for temperature in (300, 450):
            name = f"T_{temperature:04d}K.json"
            copy(args.run_root / "full_state" / label / name,
                 args.destination / "executed/family_a_joint_execution" / label / name)

    files = []
    for path in sorted(args.destination.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files.append({
                "path": str(path.relative_to(args.destination)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    (args.destination / "MANIFEST.json").write_text(json.dumps({
        "schema": "r10.v2-p25-developed-completion-package/2",
        "files": files,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"file_count": len(files), "destination": str(args.destination.resolve())},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
