#!/usr/bin/env python3
"""Command-line entry point for v10.2.27 mechanical kernel resolution."""
from __future__ import annotations

import os
import sys

from arrhenius_fracture.kernel_configuration_v10227 import (
    endpoint_resolving_tip_h_fine_m,
)
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


def _install_automatic_endpoint_resolution(argv: list[str]) -> list[str]:
    """Add a resolvable graded-tip spacing for generated configurations.

    An explicit mechanical JSON remains authoritative. For ordinary generated
    configurations, the spacing follows the first centre of the requested uniform
    active grid, so changing process-zone length or bin count remains automatic.
    """
    result = list(argv)
    _validate_cache_mode(result)
    if "--mechanical-config" in result or "--tip-h-fine-um" in result:
        return result
    length_um = float(_option(result, "--process-zone-length-um", "50"))
    bins = int(_option(result, "--process-zone-bins", "80"))
    tip_h_um = 1.0e6 * endpoint_resolving_tip_h_fine_m(length_um * 1.0e-6, bins)
    result.extend(["--tip-h-fine-um", f"{tip_h_um:.17g}"])
    return result


if __name__ == "__main__":
    raise SystemExit(main(_install_automatic_endpoint_resolution(sys.argv[1:])))
