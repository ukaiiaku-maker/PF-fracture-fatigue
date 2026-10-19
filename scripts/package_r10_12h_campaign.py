#!/usr/bin/env python3
"""Build a compact, hash-manifested export of the autonomous R10 campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = "6018f4c40df68139fd0a54bda6ab10c48510eca0"
RESTART_NAMES = {
    "high_cycle_live_checkpoint.json",
    "high_cycle_live_state.npz",
}


def rooted(path: Path) -> Path:
    """Resolve CLI paths relative to the repository, independent of caller cwd."""
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def changed_sources(base_revision: str) -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_revision}..HEAD"],
        cwd=ROOT, check=True, capture_output=True, text=True)
    paths = []
    for name in result.stdout.splitlines():
        path = ROOT / name
        if path.is_file():
            paths.append(path)
    return sorted(set(paths))


def campaign_files(campaign: Path) -> list[Path]:
    selected: set[Path] = set()
    for name in ("campaign_ledger.json", "acquisition_round_29_decision.json"):
        path = campaign / name
        if path.is_file():
            selected.add(path)
    for directory in (campaign / "analysis", campaign / "figures",
                      campaign / "campaign_delivery",
                      campaign / "reference_reproduction"):
        if directory.is_dir():
            selected.update(path for path in directory.rglob("*") if path.is_file())
    for path in campaign.rglob("*"):
        if not path.is_file():
            continue
        if path.name in {"result.json", "progress.json"} | RESTART_NAMES:
            selected.add(path)
        elif (path.name.endswith("summary.json") or
              path.name in {"qualification.json", "selection_summary.json"}):
            selected.add(path)
        elif path.suffix == ".csv" or path.name.endswith(".csv.gz"):
            selected.add(path)
        elif (any(part.startswith("mechanism_controls") or
                  part.startswith("spatial_2d") for part in path.parts) and
              path.suffix in {".json", ".npz"}):
            selected.add(path)
    # Append-only checkpoint histories dominate the working archive and are
    # reproducible from the compact checkpoint plus progress/result records.
    return sorted(path for path in selected
                  if path.name != "high_cycle_live_history.jsonl")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.campaign = rooted(args.campaign)
    if args.report is not None:
        args.report = rooted(args.report)
    args.out = rooted(args.out)
    evidence = campaign_files(args.campaign)
    sources = changed_sources(args.base_revision)
    if args.report and args.report.is_file():
        evidence.append(args.report)
    entries = []
    for path in evidence:
        relative = path.relative_to(ROOT)
        entries.append((path, Path("evidence") / relative))
    for path in sources:
        entries.append((path, Path("source_snapshot") / path.relative_to(ROOT)))
    verifier = ROOT / "scripts" / "verify_r10_12h_campaign_archive.py"
    if verifier.is_file():
        entries.append((verifier, Path("VERIFY_ARCHIVE.py")))
    # Deduplicate archive names while preserving the evidence copy of a report.
    unique = {str(archive): (path, archive) for path, archive in entries}
    entries = [unique[key] for key in sorted(unique)]
    manifest = {
        "schema": "r10.12h-portable-evidence-manifest/1",
        "base_revision": args.base_revision,
        "head_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip(),
        "files": [{
            "path": str(archive),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        } for path, archive in entries],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for path, name in entries:
            archive.write(path, str(name))
        archive.writestr("MANIFEST.json", json.dumps(
            manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({
        "archive": str(args.out),
        "bytes": args.out.stat().st_size,
        "sha256": sha256(args.out),
        "files": len(entries),
        "source_files": len(sources),
        "evidence_files": len(evidence),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
