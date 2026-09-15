#!/usr/bin/env python3
"""Create the deterministic V2.3 payload manifest and sealed archive."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/v2_3_exact_row_cycle_hazard_fatigue"
ARCHIVE = OUT / "Archive_V2_3_EXACT_ROW_CYCLE_HAZARD_FATIGUE.zip"
MANIFEST = OUT / "SHA256_MANIFEST.json"
ROOT_ARTIFACTS = (
    "V2_2_ACCEPTED_INTERPRETATION_ADDENDUM.md",
    "V2_3_EXACT_ROW_CYCLE_HAZARD_FATIGUE_PROTOCOL.md",
    "v2_3_exact_row_cycle_hazard_fatigue_protocol.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    excluded = {ARCHIVE.resolve(), MANIFEST.resolve()}
    payload = {
        str(path.relative_to(OUT)): path
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.resolve() not in excluded
    }
    for name in ROOT_ARTIFACTS:
        payload[name] = ROOT / name
    files = {name: sha(path) for name, path in sorted(payload.items())}
    MANIFEST.write_text(json.dumps({
        "schema": "v10.2.30_v2_3_complete_artifact_manifest_v1",
        "files": files,
        "manifest_scope": "all_archive_payload_files_except_this_manifest",
        "raw_run_files_embedded": False,
    }, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo(MANIFEST.name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, MANIFEST.read_bytes())
    print(json.dumps({
        "status": "PASS", "payload_files": len(files), "archive_sha256": sha(ARCHIVE)
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
