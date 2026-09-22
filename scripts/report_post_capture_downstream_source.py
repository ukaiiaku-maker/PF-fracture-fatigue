#!/usr/bin/env python3
"""Persist the qualified downstream observation at the saved upper capture."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.voiding_production_v5 import (
    advance_directional_clocks,
    downstream_source_observation,
    preview_directional_first_passage,
)
from scripts.screen_informative_off_axis_operating_point import TEMPERATURE_K


CHECKPOINT = ROOT / "artifacts/informative_off_axis_2s_continuation/above/interval_001_restart.json"
OUTPUT = ROOT / "artifacts/informative_off_axis_2s_continuation/above/downstream_source_qualification.json"
TOTAL_HORIZON_S = 2.0


def finite(value):
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def main():
    state = restore_checkpoint(CHECKPOINT)
    before = complete_accepted_state_fingerprint(state)
    source = downstream_source_observation(state, continuation=False)
    remaining = TOTAL_HORIZON_S - float(state.junction_process_state["production_time_s"])
    preview = preview_directional_first_passage(
        state, source["tensor_Pa"], source_kind=source["source_kind"],
        source_front_id=source["source_front_id"],
        source_cavity_id=source["source_cavity_id"],
        source_boundary_site_id=source["source_boundary_site_id"],
        source_position_m=source["start_m"],
        source_probe_identity=source["probe_identity"],
        temperature_K=TEMPERATURE_K,
        maximum_advance_duration_s=remaining,
        front_load_rows=source["front_load_rows"],
    )
    _, rates = advance_directional_clocks(state, preview)
    after = complete_accepted_state_fingerprint(state)
    proof = source["probe_identity"]["qualification_proof"]
    result = {
        "schema": "informative.post-capture-downstream-source/1",
        "checkpoint": str(CHECKPOINT),
        "accepted_time_s": float(state.junction_process_state["production_time_s"]),
        "remaining_horizon_s": remaining,
        "source_kind": source["source_kind"],
        "source_position_m": source["start_m"],
        "source_tensor_Pa": source["tensor_Pa"],
        "physical_arc_identity": source["probe_identity"]["physical_arc_identity"],
        "qualification_proof": proof,
        "directional_rates": rates,
        "crossing_time_s": preview.crossing_time_s,
        "event_ids": preview.event_ids,
        "qualified_physical_outcome": (
            "NO_DOWNSTREAM_BIRTH_WITHIN_REMAINING_HORIZON"
            if not math.isfinite(preview.crossing_time_s) else
            "DOWNSTREAM_BIRTH_WITHIN_REMAINING_HORIZON"
        ),
        "accepted_state_preserved": before == after,
    }
    OUTPUT.write_text(json.dumps(finite(result), indent=2, sort_keys=True) + "\n")
    print(result["qualified_physical_outcome"])
    print("ACCEPTED_STATE_PRESERVED", result["accepted_state_preserved"])


if __name__ == "__main__":
    main()
