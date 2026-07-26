#!/usr/bin/env python3
"""Command-line entry point for v10.2.27 mechanical kernel resolution."""
from __future__ import annotations

import os
import sys

from arrhenius_fracture.kernel_resolver_v10227 import main


def _option(argv: list[str], name: str, default: str) -> str:
    try:
        index = argv.index(name)
    except ValueError:
        return default
    if index + 1 >= len(argv):
        raise SystemExit(f"{name} requires a value")
    return argv[index + 1]


def _validate_cache_mode(argv: list[str]) -> None:
    mode = _option(argv, "--mode", "auto")
    snapshot_root = str(os.environ.get("KERNEL_SNAPSHOT_ROOT", "")).strip()
    load_root = str(os.environ.get("KERNEL_LOAD_INVARIANCE_ROOT", "")).strip()
    if mode == "build" and (snapshot_root or load_root):
        raise SystemExit(
            "--mode build clears generated cache directories before launching the "
            "kernel builder and cannot be combined with KERNEL_SNAPSHOT_ROOT or "
            "KERNEL_LOAD_INVARIANCE_ROOT. Use --mode auto for artifact reuse, or "
            "unset the reuse roots for a clean recalculation."
        )


if __name__ == "__main__":
    arguments = list(sys.argv[1:])
    _validate_cache_mode(arguments)
    raise SystemExit(main(arguments))
