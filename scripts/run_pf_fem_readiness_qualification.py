#!/usr/bin/env python3
"""Fail-closed preflight for the not-yet-qualified three-level PF/FEM job."""
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
               "status": "FEM_OBSERVER_AND_RESTART_READY_OPENING_ONLY_ADAPTER_UNRESOLVED"}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not args.dry_run:
        raise SystemExit("execution disabled: opening-only current-companion state-update ownership is not qualified")


if __name__ == "__main__":
    main()
