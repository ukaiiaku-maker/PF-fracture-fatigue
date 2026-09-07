#!/usr/bin/env python3
"""Supervise the qualified Weak-T/ceramic high-rate common ladder.

This launch-only specialization retains the qualified v10.2.30 production
settings and changes only the selected material labels and driving-force
fractions.
"""
from __future__ import annotations

try:
    from scripts import v10230_driving_force_ladder_supervisor as ladder
except ModuleNotFoundError:
    import v10230_driving_force_ladder_supervisor as ladder


ladder.LABELS = ("weakT", "ceramic")
ladder.FRACTIONS = (0.950, 0.975, 1.000, 1.025, 1.050, 1.100)
ladder.MANIFEST_NAME = "weakt_ceramic_high_rate_ladder_matrix.json"


if __name__ == "__main__":
    raise SystemExit(ladder.main())
