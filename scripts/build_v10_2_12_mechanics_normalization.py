#!/usr/bin/env python3
"""Build the v10.2.12 mechanics-derived source normalization artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrhenius_fracture.mechanics_normalization_v10212 import (
    SourceGeometryAssumptions,
    derive_mechanical_normalization,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine-config",
        type=Path,
        required=True,
        help="Complete engine JSON or a captured snapshot.json containing engine_config",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-spacing-b", type=float, default=10.0)
    parser.add_argument("--maximum-spacing-b", type=float, default=100.0)
    parser.add_argument("--source-region-length-m", type=float)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if not args.engine_config.is_file():
        raise SystemExit(f"engine/snapshot JSON is missing: {args.engine_config}")
    raw = json.loads(args.engine_config.read_text())
    payload_in = raw.get("engine_config", raw)
    if not isinstance(payload_in, dict):
        raise SystemExit("engine_config must be a JSON object")
    assumptions = SourceGeometryAssumptions(
        minimum_spacing_b=args.minimum_spacing_b,
        maximum_spacing_b=args.maximum_spacing_b,
        source_region_length_m=args.source_region_length_m,
    )
    payload = derive_mechanical_normalization(payload_in, assumptions=assumptions)
    payload["input_json"] = str(args.engine_config.resolve())
    payload["input_was_snapshot"] = "engine_config" in raw
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"out": str(args.out), **payload}, indent=2))


if __name__ == "__main__":
    main()
