#!/usr/bin/env python3
"""Fail-closed preflight for native recursive-branch qualification."""
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
    p.add_argument("--second-generation", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not args.second_generation:
        raise SystemExit("recursive readiness requires a second-generation birth")
    payload = {"qualification": "Q-BRANCH-RECURSIVE",
               "native_birth_entry": "scripts/run_v11_branching_case.py",
               "capacity": capacity_preflight(ROOT, 24.0).__dict__,
               "production_execution_authorized": False,
               "status": "UNRESOLVED_SECOND_GENERATION_NATIVE_ADAPTER"}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not args.dry_run:
        raise SystemExit("execution disabled: recursive second-generation adapter is not implemented")


if __name__ == "__main__":
    main()
