#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrhenius_fracture.mechanical_closure_v1024 import (
    build_atlas_from_trace_roots,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-da-um", type=float, default=5.0)
    parser.add_argument("--thin-stride", type=int, default=1)
    args = parser.parse_args()

    payload = build_atlas_from_trace_roots(
        args.trace_root,
        args.out,
        checkpoint_da_m=float(args.checkpoint_da_um) * 1.0e-6,
        thin_stride=int(args.thin_stride),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
