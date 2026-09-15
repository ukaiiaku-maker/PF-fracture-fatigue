#!/usr/bin/env python3
"""Build a deterministic compact archive for the V2 spatial-transfer campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "analysis_outputs/v2_2d_single_crack_300K_1200K"
SOURCE_FILES = (
    "V2_2D_SINGLE_CRACK_CAMPAIGN_PROTOCOL.md",
    "v2_2d_single_crack_campaign_manifest.json",
    "v2_named_parameterization_provenance.json",
    "v2_named_parameterization_registry.csv",
    "v2_named_parameterization_registry.json",
    "arrhenius_fracture/sharp_front.py",
    "arrhenius_fracture/sharp_front_v10_2_30_v2_named_single_crack.py",
    "arrhenius_fracture/run_state_checkpoint_v10230.py",
    "arrhenius_fracture/sparse_accepted_field_export_v10230.py",
    "arrhenius_fracture/v2_named_parameterizations.py",
    "scripts/build_v2_2d_single_crack_review_archive_v10230.py",
    "scripts/run_v2_2d_single_crack_campaign_v10230.py",
    "scripts/report_v2_2d_single_crack_campaign_v10230.py",
    "scripts/verify_v2_2d_single_crack_campaign_v10230.py",
    "tests/test_v10_2_30_v2_2d_campaign.py",
)
CASE_AUDIT_FILES = (
    "command.json",
    "run_args.json",
    "terminal_record.json",
    "worker_status.json",
    "v2_exact_parameter_binding.json",
    "run_state_checkpoint.json",
)
FIXED_TIME = (2026, 9, 15, 0, 0, 0)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def add_bytes(zf: zipfile.ZipFile, name: str, data: bytes, hashes: dict[str, str]) -> None:
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    zf.writestr(info, data, compresslevel=9)
    hashes[name] = digest(data)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--campaign-root", required=True, type=Path)
    p.add_argument("--archive", required=True, type=Path)
    args = p.parse_args()
    campaign = args.campaign_root.resolve()
    archive = args.archive.resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    checkpoint_index = []
    with zipfile.ZipFile(archive, "w", allowZip64=True) as zf:
        for path in sorted(REVIEW.iterdir()):
            if path.is_file() and path.name not in {"SHA256_MANIFEST.json", "COMPACT_REVIEW_ARCHIVE.json"}:
                add_bytes(zf, f"review/{path.name}", path.read_bytes(), hashes)
        for rel in SOURCE_FILES:
            path = ROOT / rel
            add_bytes(zf, f"source/{rel}", path.read_bytes(), hashes)
        for path in sorted(campaign.glob("*.json")) + sorted(campaign.glob("*.csv")):
            add_bytes(zf, f"campaign/{path.name}", path.read_bytes(), hashes)
        for case in sorted(p for p in campaign.iterdir() if p.is_dir()):
            for name in CASE_AUDIT_FILES:
                path = case / name
                if path.is_file():
                    add_bytes(zf, f"campaign/{case.name}/{name}", path.read_bytes(), hashes)
            portable = case / "portable_fields"
            for path in sorted(portable.iterdir()) if portable.is_dir() else ():
                if path.is_file():
                    add_bytes(zf, f"campaign/{case.name}/portable_fields/{path.name}", path.read_bytes(), hashes)
            pointer = json.loads((case / "run_state_checkpoint.json").read_text())
            checkpoint_index.append({
                "case_id": case.name,
                "external_case_path": str(case),
                "checkpoint_pointer_sha256": digest((case / "run_state_checkpoint.json").read_bytes()),
                "generation": pointer["generation"],
                "generation_file_sha256": pointer["files"],
            })
        add_bytes(zf, "RAW_CHECKPOINT_INDEX.json", (json.dumps(checkpoint_index, indent=2, sort_keys=True) + "\n").encode(), hashes)
        manifest = {
            "schema": "v10.2.30_v2_single_crack_compact_review_archive_v1",
            "campaign_root": str(campaign),
            "portable_fields_included": True,
            "large_native_checkpoints_externally_retained": True,
            "files": hashes,
        }
        add_bytes(zf, "ARCHIVE_CONTENT_MANIFEST.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(), {})
    archive_record = {
        "schema": "v10.2.30_v2_single_crack_compact_review_archive_record_v1",
        "archive_path": str(archive),
        "archive_sha256": digest(archive.read_bytes()),
        "archive_bytes": archive.stat().st_size,
        "portable_fields_included": True,
        "large_native_checkpoints_externally_retained_at": str(campaign),
    }
    record_path = REVIEW / "COMPACT_REVIEW_ARCHIVE.json"
    record_path.write_text(json.dumps(archive_record, indent=2, sort_keys=True) + "\n")
    files = sorted(p for p in REVIEW.iterdir() if p.is_file() and p.name != "SHA256_MANIFEST.json")
    (REVIEW / "SHA256_MANIFEST.json").write_text(json.dumps({p.name: digest(p.read_bytes()) for p in files}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(archive_record, sort_keys=True))


if __name__ == "__main__":
    main()
