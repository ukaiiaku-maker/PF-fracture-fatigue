#!/usr/bin/env python3
"""Create the deterministic V2.2 payload manifest and archive."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/v2_2_1d_rising_resistance_and_fatigue"
ARCHIVE = OUT / "Archive_V2_2_1D_RISING_RESISTANCE_AND_FATIGUE.zip"
MANIFEST = OUT / "SHA256_MANIFEST.json"
ROOT_ARTIFACTS = (
    "ONE_D_RISING_RESISTANCE_PROTOCOL.md",
    "one_d_rising_resistance_protocol.json",
    "fatigue_entrypoint_capability_audit.csv",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    excluded = {ARCHIVE.resolve(), MANIFEST.resolve()}
    payload = {str(path.relative_to(OUT)): path for path in sorted(OUT.rglob("*"))
               if path.is_file() and path.resolve() not in excluded}
    for name in ROOT_ARTIFACTS:
        payload[name] = ROOT / name
    files = {name: sha(path) for name, path in sorted(payload.items())}
    MANIFEST.write_text(json.dumps({
        "schema": "v10.2.30_v2_2_complete_artifact_manifest_v1",
        "files": files,
        "manifest_scope": "all_archive_payload_files_except_this_manifest",
    }, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(payload.items()):
            archive.write(path, name)
        archive.write(MANIFEST, MANIFEST.name)
    print(json.dumps({"status": "PASS", "payload_files": len(files),
                      "archive_sha256": sha(ARCHIVE)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
