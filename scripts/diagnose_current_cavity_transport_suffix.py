#!/usr/bin/env python3
"""Run one bounded paired suffix from an immutable cavity checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.shared_execution import execute_shared
from scripts.run_current_cavity_post_ramp_pair import NATIVE_ARGS, options

DEFAULT_RESTART = (
    ROOT / "integration_evidence" / "active_fracture_comparisons_0a5d55e" /
    "cavity_post_ramp_dwell_10us_run4" / "checkpoint" / "accepted_v5_pair.json"
)


def replace_argument(args: list[str], flag: str, value: str) -> None:
    args[args.index(flag) + 1] = value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart", type=Path, default=DEFAULT_RESTART)
    parser.add_argument("--duration-s", type=float, default=1.953125e-8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    native = list(NATIVE_ARGS)
    replace_argument(native, "--duration-s", repr(args.duration_s))
    execute_shared(
        options(), native, args.out.resolve(),
        source_policy="qualified", restart_from=args.restart.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
