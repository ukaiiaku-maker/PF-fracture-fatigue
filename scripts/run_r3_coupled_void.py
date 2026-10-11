#!/usr/bin/env python3
"""Run the exact clean-tree coupled reduced fracture/void execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.coupled_void_execution_r3 import execute_coupled_void_r3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute_coupled_void_r3(args.output)
    print(json.dumps({
        "passed": result["passed"],
        "terminal_void_phase": result["active_cavity"]["terminal_phase"],
        "terminal_radius_m": result["active_cavity"]["terminal_radius_m"],
        "coupled_event_count": result["active_cavity"]["event_count"],
        "fracture_adapter_renewals": result["fracture_adapter"]["actual_renewals"],
        "global_status": result["qualification"]["global_status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

