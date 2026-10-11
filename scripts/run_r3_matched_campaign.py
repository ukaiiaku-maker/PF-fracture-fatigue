#!/usr/bin/env python3
"""Run the 900 K recorded replay and autonomous reduced load history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.matched_persistent_execution_r3 import run_matched_campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_matched_campaign(args.output)
    print(json.dumps({
        "passed": result["passed"],
        "replay_events": [
            row["reduced_step"]["fired"]
            for row in result["recorded_operator_replay"]
        ],
        "restart_exact": result["restart"]["exact"],
        "autonomous_terminal_checkpoints": result[
            "autonomous_reduced"
        ]["terminal"]["checkpoint_count"],
        "unload_reload_exercised": result[
            "autonomous_reduced"
        ]["unload_reload_exercised"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
