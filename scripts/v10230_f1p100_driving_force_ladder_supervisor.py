#!/usr/bin/env python3
"""Supervise the qualified Peak/DBTT f=1.10 transition check."""
from __future__ import annotations

try:
    from scripts import v10230_driving_force_ladder_supervisor as ladder
except ModuleNotFoundError:
    import v10230_driving_force_ladder_supervisor as ladder


ladder.FRACTIONS = (1.100,)
ladder.MANIFEST_NAME = "f1p100_driving_force_ladder_matrix.json"


if __name__ == "__main__":
    raise SystemExit(ladder.main())
