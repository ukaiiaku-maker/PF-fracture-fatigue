#!/usr/bin/env python3
"""Extend the bounded subgrid mechanics dataset from saved candidate states only."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from scripts import export_aligned_subgrid_closure_observations_v1 as exporter


CANDIDATE = ROOT/"runs/managed_sustained_growth_v1_20260920/output"
ORIGINAL = ROOT/"artifacts/aligned_subgrid_closure_v1/observations.json"
OUT = ROOT/"artifacts/aligned_subgrid_closure_candidate_v2/observations.json"
SPLIT = {
    "restart_000002400ms.json": "TRAIN_CANDIDATE_ONE_SITE",
    "restart_000002600ms.json": "TRAIN_CANDIDATE_ONE_SITE",
    "restart_000002800ms.json": "TRAIN_CANDIDATE_TWO_SITE",
    "restart_000003500ms.json": "HOLDOUT_CANDIDATE_TWO_SITE_EARLY",
    "restart_000003900ms.json": "TRAIN_CANDIDATE_TWO_SITE",
    "restart_000005300ms.json": "TRAIN_CANDIDATE_TWO_SITE",
    "restart_000006300ms.json": "HOLDOUT_CANDIDATE_TWO_SITE_LATE",
    "restart_000006800ms.json": "TRAIN_CANDIDATE_TWO_SITE_BOUNDARY",
}


def maximum_equivalent_plastic_strain(state) -> float:
    ep = np.asarray(state.ep_gp, float)
    return float(np.max(np.sqrt(ep[0]**2+ep[1]**2+2.0*ep[2]**2)))


def main() -> None:
    exporter.RUN = CANDIDATE
    progress = json.loads((CANDIDATE/"progress.json").read_text())
    history_by_fp = {}
    checkpoint_by_time = {}
    for name in SPLIT:
        state = restore_checkpoint(CANDIDATE/name)
        checkpoint_by_time[round(float(state.junction_process_state[
            "production_time_s"]), 8)] = (name, state)
    for history in progress["history"]:
        key = round(float(history["end_time_s"]), 8)
        if key not in checkpoint_by_time:
            continue
        _, state = checkpoint_by_time[key]
        row = dict(history)
        row["maximum_equivalent_plastic_strain"] = maximum_equivalent_plastic_strain(state)
        row["plastic_work_J_per_m"] = float(history["adaptive_local_error"][
            "accepted_half_path_plastic_work_J_per_m"])
        history_by_fp[complete_accepted_state_fingerprint(state)] = row
    candidate_rows = [exporter._row(name, split, history_by_fp)
                      for name, split in SPLIT.items()]

    original = json.loads(ORIGINAL.read_text())
    # Retain only the parameter-independent pre-birth anchors.  Candidate
    # states own every observation after the changed emission row becomes active.
    original_rows = [row for row in original["rows"]
                     if row["checkpoint"] in {
                         "restart_001900ms.json", "restart_002100ms.json"}]
    for row in original_rows:
        row["split"] = "TRAIN_COMMON_PREBIRTH"
    rows = original_rows+candidate_rows
    ordered = sorted(rows, key=lambda row: row["accepted_time_s"])
    histories = np.asarray([row["plastic_history"][
        "maximum_equivalent_plastic_strain"] for row in ordered], float)
    times = np.asarray([row["accepted_time_s"] for row in ordered], float)
    if np.any(np.diff(histories) <= 0.0) or np.any(np.diff(times) <= 0.0):
        raise RuntimeError("candidate enrichment coordinates are not strictly ordered")
    rates = np.gradient(histories, times)
    if np.any(~np.isfinite(rates)) or np.any(rates <= 0.0):
        raise RuntimeError("candidate history-rate coordinate is invalid")
    for row, rate in zip(ordered, rates):
        row["plastic_history"]["maximum_equivalent_plastic_strain_rate_s"] = float(rate)
        row["plastic_history"]["history_evolution_coordinate"] = (
            "STATE_ODE_DH_DT_AS_FUNCTION_OF_ACCEPTED_MAX_EQUIVALENT_PLASTIC_STRAIN")
        row["observation_sha256"] = exporter._sha({
            key: value for key, value in row.items() if key != "observation_sha256"})
    payload = {
        "schema": "unified-2d.aligned-subgrid-closure-dataset/2",
        "scope": "DBTT_1200K_ALIGNED_PRECONNECTION_SUBGRID_CANDIDATE_ENRICHED",
        "split_declared_by_checkpoint_before_fit": {
            **{row["checkpoint"]: row["split"] for row in original_rows}, **SPLIT},
        "enrichment_policy": (
            "SAVED_ACCEPTED_STATES_ONLY; PREBIRTH_COMMON ANCHORS PLUS CANDIDATE; "
            "NO_RADIUS_VERSUS_TIME FIT"),
        "history_evolution_policy": (
            "BOUNDED STATE ODE dH/dt=f(H), PIECEWISE LINEAR IN ACCEPTED HISTORY"),
        "far_field_correction_policy": (
            "EXACT_ZERO_WHILE PRODUCTION REPRESENTATION HAS NO MESH HOLE"),
        "local_source_policy": "ACCEPTED_3D_ELASTOPLASTIC_CONSTITUTIVE_TENSOR",
        "rows": ordered,
    }
    payload["dataset_sha256"] = exporter._sha(payload)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(exporter._plain(payload), indent=2,
                              sort_keys=True, allow_nan=False)+"\n")
    print(OUT)
    print("OBSERVATIONS", len(ordered))
    print("DATASET_SHA256", payload["dataset_sha256"])
    print("HISTORY_DOMAIN", histories[0], histories[-1])


if __name__ == "__main__":
    main()
