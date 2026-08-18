#!/usr/bin/env python3
"""Run matched baseline/reversible cases with v4 physical-return semantics."""
from __future__ import annotations

import os
from pathlib import Path
import sys

_SCRIPTS = Path(__file__).resolve().parent
_DEFAULT_V914 = Path(
    os.environ.get(
        "V914_ROOT",
        "/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search",
    )
)
for _path in (str(_SCRIPTS), str(_DEFAULT_V914)):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(_DEFAULT_V914))
sys.path.insert(0, str(_SCRIPTS))

import run_v914_minimal_reversible_case as _runner
from v914_minimal_reversible_explicit_v4 import (
    run_minimal_reversible_explicit,
)


_runner.run_minimal_reversible_explicit = run_minimal_reversible_explicit


if __name__ == "__main__":
    raise SystemExit(_runner.main())
