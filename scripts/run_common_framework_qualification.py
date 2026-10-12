#!/usr/bin/env python3
"""Run explicit portable or complete common-framework qualification layers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.external_evidence import load_and_verify


PORTABLE = (
    "tests/test_cohesive_landscape.py",
    "tests/test_candidate_drive_contract.py",
    "tests/test_prospective_campaign.py",
    "tests/test_external_evidence.py",
    "tests/test_shared_options_resolution.py",
    "tests/test_shared_execution.py",
    "tests/test_observer_contract.py",
    "tests/test_loading_protocol.py",
)


def _pytest(paths: tuple[str, ...]) -> int:
    existing = tuple(path for path in paths if (ROOT / path).is_file())
    return subprocess.run([sys.executable, "-m", "pytest", "-q", *existing], cwd=ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layer", choices=("portable", "full"))
    parser.add_argument("--external-manifest", type=Path,
                        default=ROOT / "common_framework_release/EXTERNAL_EVIDENCE_MANIFEST.json")
    args = parser.parse_args()
    if args.layer == "portable":
        return _pytest(PORTABLE)

    if not args.external_manifest.is_file():
        print(json.dumps({"status": "INCOMPLETE", "reason": "external evidence manifest absent", "path": str(args.external_manifest)}))
        return 3
    evidence = load_and_verify(args.external_manifest)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if evidence["status"] != "PASS":
        return 3
    return subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
