#!/usr/bin/env python3
"""Command-line entry point for v10.2.28 direct kernel resolution."""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
root_text = str(ROOT)
if not sys.path or sys.path[0] != root_text:
    sys.path.insert(0, root_text)

from arrhenius_fracture.kernel_resolver_v10228 import main


def _option(argv: list[str], name: str, default: str) -> str:
    prefix = name + "="
    for index, token in enumerate(argv):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name:
            if index + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value")
            return argv[index + 1]
    return default


def _reject_legacy_capture_inputs(argv: list[str]) -> None:
    if _option(argv, "--mode", "auto") == "build":
        forbidden = {
            "KERNEL_CAPTURE_SEED_FAMILY",
            "KERNEL_CAPTURE_PARAMETER_OPTION",
            "KERNEL_CAPTURE_HAZARD_SEED",
            "KERNEL_SNAPSHOT_ROOT",
            "KERNEL_LOAD_INVARIANCE_ROOT",
        }
        supplied = sorted(name for name in forbidden if str(os.environ.get(name, "")).strip())
        if supplied:
            raise SystemExit(
                "v10.2.28 direct build forbids legacy capture inputs: "
                + ", ".join(supplied)
            )


if __name__ == "__main__":
    arguments = list(sys.argv[1:])
    _reject_legacy_capture_inputs(arguments)
    raise SystemExit(main(arguments))
