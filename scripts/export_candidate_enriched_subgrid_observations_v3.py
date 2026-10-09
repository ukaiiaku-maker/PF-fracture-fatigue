#!/usr/bin/env python3
"""Extend the saved-state candidate closure through the 7.8-s boundary."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_candidate_enriched_subgrid_observations_v2 as base


base.OUT = base.ROOT/"artifacts/aligned_subgrid_closure_candidate_v3/observations.json"
base.SPLIT = {
    **base.SPLIT,
    "restart_000007300ms.json": "HOLDOUT_CANDIDATE_TWO_SITE_NEXT_DOMAIN",
    "restart_000007800ms.json": "TRAIN_CANDIDATE_TWO_SITE_NEXT_BOUNDARY",
}


if __name__ == "__main__":
    base.main()
