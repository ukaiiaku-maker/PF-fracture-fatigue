#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrhenius_fracture.candidate_c_dual_reference_r2 import write_candidate_c_receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_candidate_c_receipt(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
