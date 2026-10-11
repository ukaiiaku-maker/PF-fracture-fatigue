#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrhenius_fracture.r2_autonomous_reduced_fatigue import run_autonomous_reduced_fatigue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_autonomous_reduced_fatigue(args.output)
    print(json.dumps({
        "event_cycle": result["event_cycle"],
        "event_advance_m": result["event_advance_m"],
        "checkpoint_exact": result["checkpoint_exact"],
        "restart_continuation_exact": result["restart_continuation_exact"],
        "post_event_return_reload": result["post_event_return_reload"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
