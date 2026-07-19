#!/usr/bin/env python3
"""Build the v10.2.12 mechanics-derived source normalization artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrhenius_fracture.mechanics_normalization_v10212 import (
    SourceGeometryAssumptions,
    derive_from_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-spacing-b", type=float, default=10.0)
    parser.add_argument("--maximum-spacing-b", type=float, default=100.0)
    parser.add_argument("--source-region-length-m", type=float)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    assumptions = SourceGeometryAssumptions(
        minimum_spacing_b=args.minimum_spacing_b,
        maximum_spacing_b=args.maximum_spacing_b,
        source_region_length_m=args.source_region_length_m,
    )
    payload = derive_from_json(args.engine_config, assumptions=assumptions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"out": str(args.out), **payload}, indent=2))


if __name__ == "__main__":
    main()
