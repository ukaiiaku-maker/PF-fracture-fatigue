#!/usr/bin/env python3
"""Extend fixed-opening support to 17.47499 s with two later holdouts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_candidate_enriched_subgrid_observations_v4 as prior

prior.base.base.OUT = prior.base.base.ROOT/(
    "artifacts/aligned_subgrid_closure_candidate_v5/observations.json")
prior.base.base.SPLIT = {
    **prior.base.base.SPLIT,
    "restart_000010475ms.json": "TRAIN_CANDIDATE_TWO_SITE_LONGER_SUPPORT_A",
    "restart_000011475ms.json": "TRAIN_CANDIDATE_TWO_SITE_LONGER_SUPPORT_A2",
    "restart_000011975ms.json": "TRAIN_CANDIDATE_TWO_SITE_LONGER_SUPPORT_A2B",
    "restart_000012475ms.json": "HOLDOUT_CANDIDATE_TWO_SITE_LONGER_DOMAIN_A",
    "restart_000012975ms.json": "TRAIN_CANDIDATE_TWO_SITE_LONGER_SUPPORT_A2C",
    "restart_000013475ms.json": "TRAIN_CANDIDATE_TWO_SITE_LONGER_SUPPORT_A3",
    "restart_000014475ms.json": "TRAIN_CANDIDATE_TWO_SITE_LONGER_SUPPORT_B",
    "restart_000016475ms.json": "TRAIN_CANDIDATE_TWO_SITE_LONGER_SUPPORT_B2",
    "restart_000016975ms.json": "HOLDOUT_CANDIDATE_TWO_SITE_LONGER_DOMAIN_B",
    "restart_000017475ms.json": "TRAIN_CANDIDATE_TWO_SITE_LONGER_BOUNDARY",
}

if __name__ == "__main__":
    prior.base.base.main()
