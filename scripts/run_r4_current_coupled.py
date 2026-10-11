#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from arrhenius_fracture.current_coupled_execution_r4 import (
    execute_current_coupled_r4,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute_current_coupled_r4(args.output)
    print(f"R4_CURRENT_COUPLED = {'PASS' if result['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
