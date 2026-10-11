#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from arrhenius_fracture.reduced_void_reference_r2 import execute_reference


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--duration-s", type=float, default=0.001)
    args = parser.parse_args(argv)
    summary = execute_reference(
        args.output, steps=args.steps, duration_s=args.duration_s,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
