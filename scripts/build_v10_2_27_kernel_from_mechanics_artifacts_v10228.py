#!/usr/bin/env python3
"""Run the v10.2.27 mechanics-artifact builder with the v10.2.28 atlas adapter.

The mechanics-artifact builder launches the extended atlas assembler in a child
subprocess. This wrapper installs the command rewrite in that same process so the
finite-metadata adapter is actually used. No mechanics, kernel coefficients, or
coverage rules are modified.
"""
from __future__ import annotations

from pathlib import Path
import runpy
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORIGINAL_BUILDER = ROOT / "scripts" / "build_v10_2_27_kernel_from_mechanics_artifacts.py"
ORIGINAL_ASSEMBLER = ROOT / "scripts" / "build_v10_2_27_extended_active_only_atlas.py"
FINITE_ASSEMBLER = (
    ROOT / "scripts" / "build_v10_2_27_extended_active_only_atlas_finite_metadata.py"
)


def rewrite_assembler_command(command: Any) -> Any:
    """Replace only the exact extended-atlas executable in argv-style commands."""
    if not isinstance(command, (list, tuple)):
        return command
    patched = list(command)
    replacements = 0
    for index, token in enumerate(patched):
        if str(token) == str(ORIGINAL_ASSEMBLER):
            patched[index] = str(FINITE_ASSEMBLER)
            replacements += 1
    if replacements > 1:
        raise RuntimeError("extended atlas assembler appeared more than once in command")
    return patched


def main() -> None:
    original_run = subprocess.run

    def run_with_finite_atlas_metadata(command, *args, **kwargs):
        return original_run(rewrite_assembler_command(command), *args, **kwargs)

    subprocess.run = run_with_finite_atlas_metadata
    try:
        runpy.run_path(str(ORIGINAL_BUILDER), run_name="__main__")
    finally:
        subprocess.run = original_run


if __name__ == "__main__":
    main()
