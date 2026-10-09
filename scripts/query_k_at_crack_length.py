#!/usr/bin/env python3
"""Query a stage-aware fracture history or a legacy v10.2.27 steps CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrhenius_fracture.fracture_observation_history import (
    LengthCoordinate, ObservationHistory, ObservationStage, ValidityStatus,
    from_legacy_steps_csv,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("--target-um", type=float, required=True)
    parser.add_argument("--coordinate", choices=[item.value for item in LengthCoordinate],
                        default=LengthCoordinate.EXTENSION.value)
    parser.add_argument("--initial-length-um", type=float, default=0.0)
    parser.add_argument("--front-id")
    parser.add_argument("--segment-id")
    parser.add_argument("--stage", choices=[item.value for item in ObservationStage])
    parser.add_argument("--legacy-steps", action="store_true")
    parser.add_argument("--validity", choices=[item.value for item in ValidityStatus])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    history = (from_legacy_steps_csv(args.history, front_id=args.front_id or "primary",
                                     segment_id=args.segment_id or "legacy",
                                     initial_length_m=args.initial_length_um * 1e-6)
               if args.legacy_steps else ObservationHistory.load(args.history))
    result = history.at_length(
        args.target_um * 1e-6, front_id=args.front_id,
        segment_id=args.segment_id, stage=args.stage,
        coordinate=args.coordinate, validity=args.validity,
    ).to_dict()
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
