#!/usr/bin/env python3
"""Final measured ceramic neighbor for the 1e-3 m/cycle endpoint."""
from __future__ import annotations

import importlib

try:
    from scripts import v10230_four_class_1e3_rate_supervisor as base
except ModuleNotFoundError:
    import v10230_four_class_1e3_rate_supervisor as base

base = importlib.reload(base)
base.TARGET_POINTS = {"ceramic": (1.264,)}
base.ladder.MANIFEST_NAME = "ceramic_1e3_rate_final_matrix.json"
base.ladder.matrix = base.matrix


if __name__ == "__main__":
    raise SystemExit(base.ladder.main())
