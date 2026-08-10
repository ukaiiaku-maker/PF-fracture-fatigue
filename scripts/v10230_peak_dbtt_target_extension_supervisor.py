#!/usr/bin/env python3
"""Measured-rate Peak/DBTT extension toward da/dN = 1e-5 m/cycle."""
from __future__ import annotations

import importlib

try:
    from scripts import v10230_driving_force_ladder_supervisor as ladder
except ModuleNotFoundError:
    import v10230_driving_force_ladder_supervisor as ladder

ladder = importlib.reload(ladder)
ladder.LABELS = ("peak", "dbtt")
ladder.FRACTIONS = (1.105, 1.150)
_full_matrix = ladder.matrix

def matrix():
    return [row for row in _full_matrix()
            if (row["label"], row["fraction"]) in {("dbtt", 1.105), ("peak", 1.150)}]

ladder.matrix = matrix
ladder.MANIFEST_NAME = "peak_dbtt_target_extension_matrix.json"

if __name__ == "__main__":
    raise SystemExit(ladder.main())
