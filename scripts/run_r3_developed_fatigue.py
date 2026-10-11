#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrhenius_fracture.developed_fatigue_r3 import run_developed_fatigue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-events", type=int, default=4)
    parser.add_argument("--phase-count", type=int, default=24)
    parser.add_argument("--maximum-cycles", type=int, default=80)
    args = parser.parse_args()
    result = run_developed_fatigue(
        args.output, target_events=args.target_events,
        phase_count=args.phase_count, maximum_cycles=args.maximum_cycles,
    )
    print(json.dumps({
        "target_events": result["target_events"],
        "terminal_cycle": result["terminal_cycle"],
        "terminal_extension_m": result["terminal_extension_m"],
        "checkpoint_exact": result["checkpoint_exact"],
        "restart_continuation_exact": result["restart_continuation_exact"],
        "all_waveform_time_consumed": result["all_waveform_time_consumed"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
