#!/usr/bin/env python3
"""Generate measured signed interaction-integral rows from one FEM snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrhenius_fracture.physical_fem_station_responses_v10212 import generate_station_responses


def _magnitudes(raw: str) -> list[float]:
    return [float(token) for token in raw.replace(",", " ").split()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--magnitudes", type=_magnitudes, default=[0.25, 0.5])
    parser.add_argument("--ribbon-width-m", type=float)
    parser.add_argument("--minimum-station-spacing-m", type=float)
    args = parser.parse_args()
    report = generate_station_responses(
        args.snapshot,
        out_csv=args.out,
        magnitudes=args.magnitudes,
        ribbon_width_m=args.ribbon_width_m,
        minimum_station_spacing_m=args.minimum_station_spacing_m,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
