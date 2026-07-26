#!/usr/bin/env python3
"""Run the v10.2.28 direct-kernel builder with NumPy 2 geometry compatibility."""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.prescribed_geometry_numpy2_compat_v10228 import (
    install_numpy2_orientation_compat,
)

install_numpy2_orientation_compat()
runpy.run_path(
    str(ROOT / "scripts" / "build_v10_2_28_prescribed_geometry_kernel.py"),
    run_name="__main__",
)
