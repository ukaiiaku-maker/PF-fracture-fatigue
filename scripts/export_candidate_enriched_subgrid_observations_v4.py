#!/usr/bin/env python3
"""Extend the saved-state candidate closure through the 8.47499-s boundary.

The 8.3-s state is declared as the prospective validation row before the
8.47499-s boundary is admitted to training.  Both are complete accepted
checkpoints; no new FEM solve or trajectory projection is performed here.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_candidate_enriched_subgrid_observations_v3 as base


base.base.OUT = base.base.ROOT/"artifacts/aligned_subgrid_closure_candidate_v4/observations.json"
base.base.SPLIT = {
    **base.base.SPLIT,
    "restart_000008300ms.json": "HOLDOUT_CANDIDATE_TWO_SITE_REPAIRED_WINDOW",
    "restart_000008475ms.json": "TRAIN_CANDIDATE_TWO_SITE_REPAIRED_BOUNDARY",
}


if __name__ == "__main__":
    base.base.main()
