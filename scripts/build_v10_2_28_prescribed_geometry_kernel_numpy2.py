#!/usr/bin/env python3
"""Run the direct-kernel builder with local compatibility adapters."""
from __future__ import annotations

from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.prescribed_geometry_numpy2_compat_v10228 import (
    install_numpy2_orientation_compat,
)

install_numpy2_orientation_compat()

_original_run = subprocess.run
_original_assembler = str(
    ROOT / "scripts" / "build_v10_2_27_extended_active_only_atlas.py"
)
_finite_assembler = str(
    ROOT / "scripts" / "build_v10_2_27_extended_active_only_atlas_finite_metadata.py"
)


def _run_with_finite_atlas_metadata(command, *args, **kwargs):
    if isinstance(command, (list, tuple)):
        patched = list(command)
        for index, token in enumerate(patched):
            if str(token) == _original_assembler:
                patched[index] = _finite_assembler
        command = patched
    return _original_run(command, *args, **kwargs)


subprocess.run = _run_with_finite_atlas_metadata
try:
    runpy.run_path(
        str(ROOT / "scripts" / "build_v10_2_28_prescribed_geometry_kernel.py"),
        run_name="__main__",
    )
finally:
    subprocess.run = _original_run
