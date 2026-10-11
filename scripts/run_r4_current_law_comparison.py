#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from arrhenius_fracture.current_law_spatial_comparison_r4 import (
    execute_current_law_comparison,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute_current_law_comparison(args.spatial, args.output)
    print(f"R4_CURRENT_LAW_COMPARISON = {'PASS' if result['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
