#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from arrhenius_fracture.resolved_stochastic_fatigue_r4 import (
    execute_resolved_stochastic_fatigue_r4,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-extension-um", type=float, default=20.0)
    parser.add_argument("--minimum-events", type=int, default=4)
    parser.add_argument("--phase-count", type=int, default=24)
    args = parser.parse_args()
    result = execute_resolved_stochastic_fatigue_r4(
        args.output, target_events=args.minimum_events,
        target_extension_m=args.target_extension_um * 1.0e-6,
        phase_count=args.phase_count,
    )
    print(f"R4_RESOLVED_STOCHASTIC_FATIGUE = {'PASS' if result['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
