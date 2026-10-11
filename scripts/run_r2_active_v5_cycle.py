#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrhenius_fracture.r2_active_v5_cycle import run_active_v5_cycle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=1.0e-8)
    args = parser.parse_args()
    print(json.dumps(run_active_v5_cycle(args.output, duration_s=args.duration_s), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
