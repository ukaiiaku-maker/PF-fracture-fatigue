#!/usr/bin/env python3
"""Write the prospective V2 child-tip constitutive consumer graph."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.r_tip_constitutive_handoff_v2 import audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-dirty-development", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    status = subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True)
    if status.strip() and not args.allow_dirty_development:
        raise ValueError("canonical audit requires a clean committed implementation")
    payload = {
        **audit(),
        "executed_code_sha": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "worktree_status": status,
        "canonical_evidence": not bool(status.strip()),
        "protocol_sha256": hashlib.sha256(
            (ROOT / "docs/V5_ONE_VOID_SCIENTIFIC_CLOSURE_V2_PROTOCOL.md").read_bytes()
        ).hexdigest(),
    }
    args.output.mkdir(parents=True)
    report = args.output / "report.json"
    report.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    (args.output / "sha256_manifest.json").write_text(json.dumps({
        "report.json": hashlib.sha256(report.read_bytes()).hexdigest(),
    }, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"classification": payload["scientific_classification"]}, sort_keys=True))


if __name__ == "__main__":
    main()
