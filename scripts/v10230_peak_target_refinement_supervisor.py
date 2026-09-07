#!/usr/bin/env python3
"""Actual Peak rate-target refinement at f=1.135 and 1.140."""
from __future__ import annotations
import importlib
try:
    from scripts import v10230_driving_force_ladder_supervisor as ladder
except ModuleNotFoundError:
    import v10230_driving_force_ladder_supervisor as ladder
ladder = importlib.reload(ladder)
ladder.LABELS = ("peak",)
ladder.FRACTIONS = (1.135, 1.140)
ladder.MANIFEST_NAME = "peak_target_refinement_matrix.json"
if __name__ == "__main__":
    raise SystemExit(ladder.main())
