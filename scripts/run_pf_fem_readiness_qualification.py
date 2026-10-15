#!/usr/bin/env python3
"""Report the completed compact three-level PF/FEM qualification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arrhenius_fracture.qualification_manifest import capacity_preflight


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--levels", type=int, required=True)
    p.add_argument("--opening-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.levels < 3 or not args.opening_only:
        raise SystemExit("qualification requires >=3 levels and opening-only first")
    receipt = capacity_preflight(ROOT, 18.0)
    payload = {"qualification": "Q-PF-FEM-3LEVEL", "capacity": receipt.__dict__,
               "production_execution_authorized": False,
               "status": "QUALIFIED_COMPACT_INDEPENDENT_SUPPORTS_REPRESENTATION_OFFSET",
               "evidence": "common_framework_readiness/fem_observer_restart_continuation_20261014/PF_FEM_COMMON_MECHANICS_COMPARISON.json"}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
