#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrhenius_fracture.frozen_geometry_load_invariance_v10213 import (
    evaluate_frozen_geometry_load_invariance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--load-scales", type=float, nargs="+", required=True)
    parser.add_argument("--magnitudes", type=float, nargs="+", required=True)
    parser.add_argument("--linearity-tolerance", type=float, default=0.03)
    parser.add_argument("--load-invariance-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--minimum-residual-stiffness-fraction", type=float, default=1.0e-3
    )
    parser.add_argument("--ribbon-width-m", type=float)
    parser.add_argument("--minimum-station-spacing-m", type=float)
    args = parser.parse_args()

    payload = evaluate_frozen_geometry_load_invariance(
        args.snapshot,
        outroot=args.outroot,
        load_scales=args.load_scales,
        perturbation_magnitudes=args.magnitudes,
        ribbon_width_m=args.ribbon_width_m,
        minimum_station_spacing_m=args.minimum_station_spacing_m,
        linearity_tolerance=args.linearity_tolerance,
        load_invariance_tolerance=args.load_invariance_tolerance,
        minimum_residual_stiffness_fraction=(
            args.minimum_residual_stiffness_fraction
        ),
    )
    print(json.dumps({
        "schema": payload["schema"],
        "parent_state_id": payload["parent_state_id"],
        "load_invariance_passed": payload["load_invariance_passed"],
        "active_kernel_mechanically_measured": payload[
            "active_kernel_mechanically_measured"
        ],
        "wake_shielding_supported": payload["wake_shielding_supported"],
        "maximum_within_load_relative_spread": payload["checks"][
            "maximum_within_load_relative_spread"
        ],
        "maximum_relative_load_variation": payload["checks"][
            "maximum_relative_load_variation"
        ],
        "report": str((args.outroot / "frozen_geometry_load_invariance.json").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
