#!/usr/bin/env python3
"""Qualified Weak-T/ceramic adaptive high-rate ladder, launch-only."""
from __future__ import annotations

try:
    from scripts import v10230_driving_force_ladder_supervisor as ladder
except ModuleNotFoundError:
    import v10230_driving_force_ladder_supervisor as ladder

ladder.LABELS = ("weakT", "ceramic")
ladder.FRACTIONS = (1.150, 1.200)
ladder.MANIFEST_NAME = "weakt_ceramic_adaptive_high_rate_matrix.json"

if __name__ == "__main__":
    raise SystemExit(ladder.main())
