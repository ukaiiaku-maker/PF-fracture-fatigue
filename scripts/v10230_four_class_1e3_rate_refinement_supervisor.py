#!/usr/bin/env python3
"""Measured neighbor refinements for the four-class 1e-3 rate endpoint."""
from __future__ import annotations

import importlib

try:
    from scripts import v10230_four_class_1e3_rate_supervisor as base
except ModuleNotFoundError:
    import v10230_four_class_1e3_rate_supervisor as base

base = importlib.reload(base)
base.TARGET_POINTS = {
    "peak": (1.175,),
    "dbtt": (1.125, 1.128),
    "weakT": (1.185, 1.1875),
}
base.ladder.MANIFEST_NAME = "four_class_1e3_rate_refinement_matrix.json"
base.ladder.matrix = base.matrix


if __name__ == "__main__":
    raise SystemExit(base.ladder.main())
