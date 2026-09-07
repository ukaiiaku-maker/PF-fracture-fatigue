#!/usr/bin/env python3
"""Two-row measured-rate refinement around da/dN = 1e-5 m/cycle."""
from __future__ import annotations

try:
    from scripts import v10230_driving_force_ladder_supervisor as ladder
except ModuleNotFoundError:
    import v10230_driving_force_ladder_supervisor as ladder

ladder.LABELS = ("weakT", "ceramic")
ladder.FRACTIONS = (1.145, 1.205)
_full_matrix = ladder.matrix


def matrix():
    return [row for row in _full_matrix()
            if (row["label"], row["fraction"]) in {("weakT", 1.145), ("ceramic", 1.205)}]


ladder.matrix = matrix
ladder.MANIFEST_NAME = "weakt_ceramic_target_refinement_matrix.json"

if __name__ == "__main__":
    raise SystemExit(ladder.main())
