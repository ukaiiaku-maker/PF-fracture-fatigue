#!/usr/bin/env python3
"""Extend fixed-opening support through 43.47499 s with later holdouts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_candidate_enriched_subgrid_observations_v5 as prior


prior.prior.base.base.OUT = prior.prior.base.base.ROOT/(
    "artifacts/aligned_subgrid_closure_candidate_v6/observations.json")
prior.prior.base.base.SPLIT = {
    **prior.prior.base.base.SPLIT,
    "restart_000022475ms.json": "TRAIN_CANDIDATE_TWO_SITE_EXTENDED_SUPPORT_A",
    "restart_000025475ms.json": "TRAIN_CANDIDATE_TWO_SITE_EXTENDED_SUPPORT_A2",
    "restart_000026475ms.json": "TRAIN_CANDIDATE_TWO_SITE_EXTENDED_SUPPORT_A3",
    "restart_000027475ms.json": "HOLDOUT_CANDIDATE_TWO_SITE_EXTENDED_DOMAIN_A",
    "restart_000028475ms.json": "TRAIN_CANDIDATE_TWO_SITE_EXTENDED_SUPPORT_A4",
    "restart_000029475ms.json": "TRAIN_CANDIDATE_TWO_SITE_EXTENDED_SUPPORT_A5",
    "restart_000034475ms.json": "TRAIN_CANDIDATE_TWO_SITE_EXTENDED_SUPPORT_B",
    "restart_000037475ms.json": "TRAIN_CANDIDATE_TWO_SITE_EXTENDED_SUPPORT_B2",
    "restart_000038475ms.json": "TRAIN_CANDIDATE_TWO_SITE_EXTENDED_SUPPORT_B3",
    "restart_000039475ms.json": "HOLDOUT_CANDIDATE_TWO_SITE_EXTENDED_DOMAIN_B",
    "restart_000040475ms.json": "TRAIN_CANDIDATE_TWO_SITE_EXTENDED_SUPPORT_B4",
    "restart_000041475ms.json": "TRAIN_CANDIDATE_TWO_SITE_EXTENDED_SUPPORT_B5",
    "restart_000043475ms.json": "TRAIN_CANDIDATE_TWO_SITE_EXTENDED_BOUNDARY",
}


if __name__ == "__main__":
    prior.prior.base.base.main()
