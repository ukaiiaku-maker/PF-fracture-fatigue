#!/usr/bin/env python3
"""Run the R1 native V5 seeded finite-activity functional fixture."""
from __future__ import annotations

import argparse
import json

from arrhenius_fracture.r1_seeded_v5_finite_activity import (
    run_seeded_finite_activity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run_seeded_finite_activity(args.out)
    print(json.dumps({
        "schema": result["schema"],
        "classification": result["classification"],
        "terminal_cavity_radius_m": result["terminal_cavity_radius_m"],
        "restart_exact": result["restart_exact"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
