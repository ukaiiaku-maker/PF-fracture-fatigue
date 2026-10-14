#!/usr/bin/env python3
"""Audit optional/full external evidence before selecting external regressions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.external_evidence import EvidenceManifest


MANIFEST = ROOT / "common_framework_release/EXTERNAL_EVIDENCE_MANIFEST.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("portable", "full"), required=True)
    parser.add_argument("--evidence-id", action="append")
    parser.add_argument("--no-hash", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.profile == "portable":
        report = {
            "schema": "external-evidence-audit/1", "decision": "COMPLETE",
            "profile": "portable", "records": [],
            "interpretation": "portable/core tests intentionally do not require external evidence",
        }
    else:
        report = EvidenceManifest(MANIFEST).audit(
            args.evidence_id, verify_hashes=not args.no_hash,
        )
        report["profile"] = "full"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
