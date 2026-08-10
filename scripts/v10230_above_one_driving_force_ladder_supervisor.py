#!/usr/bin/env python3
"""Supervise the qualified Peak/DBTT driving-force ladder above f=1.

This launch-only specialization retains every setting from the qualified
v10.2.30 ladder and changes only the requested driving-force fractions.
"""
from __future__ import annotations

try:
    from scripts import v10230_driving_force_ladder_supervisor as ladder
except ModuleNotFoundError:
    import v10230_driving_force_ladder_supervisor as ladder


ladder.FRACTIONS = (1.025, 1.050)
ladder.MANIFEST_NAME = "above_one_driving_force_ladder_matrix.json"


if __name__ == "__main__":
    raise SystemExit(ladder.main())
