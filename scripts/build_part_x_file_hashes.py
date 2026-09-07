"""Final closure D: sha256 of every tracked file under artifacts/
crack_rebonding_part_x_v1/ (excluding this file's own prior output, to
avoid a self-referential hash), for tamper-evident reproducibility of
the whole Part X artifact package.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"
SELF_NAME = "part_x_file_hashes.json"


def main() -> None:
    hashes = {}
    for path in sorted(ARTIFACTS_DIR.rglob("*")):
        if path.is_dir() or path.name == SELF_NAME:
            continue
        rel = str(path.relative_to(ARTIFACTS_DIR))
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()

    out = {
        "schema": "v10230_part_x_file_hashes_v1",
        "n_files": len(hashes), "files": hashes,
    }
    (ARTIFACTS_DIR / SELF_NAME).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"Wrote {ARTIFACTS_DIR / SELF_NAME}: {len(hashes)} files hashed")


if __name__ == "__main__":
    main()
