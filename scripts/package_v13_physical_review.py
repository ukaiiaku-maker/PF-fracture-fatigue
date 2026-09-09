#!/usr/bin/env python3
"""Publish a compact review packet without copying immutable large state files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    gate = json.loads((root / "physical_ensemble_gate.json").read_text())
    if gate["status"] == "PASS_SHORT_ENSEMBLE_WARRANTED":
        raise RuntimeError("physical ensemble remains required; do not publish a final disposition yet")
    inventory = []
    for folder in ("physical_parents", "companions"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file():
                inventory.append({"path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size, "sha256": sha(path)})
    manifest = {"schema": "v13.physical-companion-artifact-inventory/1",
        "boundary": "BRANCHING_KINETICS_MODEL_UNCALIBRATED",
        "durable_root": str(root),
        "record_producer_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "preregistered_plan_sha256": sha(root / "preregistered_plan.json"),
        "files": inventory,
        "compact_packet_excludes": "Binary checkpoints, full fields, FEM caches and Git bundles remain on Data with hashes in this inventory/source manifests.",
        "historical_atlas_modified": False,
        "sampler_only_monte_carlo_performed": False,
        "long_atlas_launched": False}
    (root / "artifact_inventory.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    selected = [root / name for name in (
        "V13_PHYSICAL_COMPANION_QUALIFICATION.md", "preregistered_plan.json",
        "physical_ensemble_gate.json", "artifact_inventory.json", "verification.json")]
    selected.extend(sorted((root / "source_preservation").glob("*.json")))
    selected.append(root / "companions/summary.json")
    for folder in (root / "physical_parents").iterdir():
        if folder.is_dir():
            selected.extend(path for path in (folder / "parent/parent_record.json",
                folder / "launch.json", folder / "terminal.json") if path.exists())
    for folder in (root / "companions").iterdir():
        if folder.is_dir():
            selected.extend(path for path in (folder / "companion_qualification.json",
                folder / "companion_stop.json") if path.exists())
    target = root / "V13_PHYSICAL_COMPANION_REVIEW.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, path.relative_to(root))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("review archive verification failed")
    (root / "review_archive_manifest.json").write_text(json.dumps({
        "archive": target.name, "bytes": target.stat().st_size, "sha256": sha(target),
        "members": len(selected), "zip_integrity": "PASS"}, indent=2) + "\n")
    print(target, target.stat().st_size, sha(target))


if __name__ == "__main__":
    main()
