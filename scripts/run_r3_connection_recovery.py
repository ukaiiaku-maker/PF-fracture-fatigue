#!/usr/bin/env python3
"""Execute and export the R3 accepted/vetoed crack-void transactions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.connection_recovery_r3 import run_connection_recovery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_connection_recovery(args.output)
    print(json.dumps({
        "passed": result["passed"],
        "veto_margin_J_per_m": result["veto"]["trial"]["energy_margin_J_per_m"],
        "connection_margin_J_per_m": result["accepted"]["trial"]["energy_margin_J_per_m"],
        "child_margin_J_per_m": result["accepted"]["downstream_child"]["trial"]["energy_margin_J_per_m"],
        "continuation_margin_J_per_m": result["accepted"]["downstream_continuation"]["trial"]["energy_margin_J_per_m"],
        "connection_restart_exact": result["accepted"]["checkpoint"]["restart_exact"],
        "continuation_restart_exact": result["accepted"]["downstream_continuation"]["restart_exact"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

