"""Compare the fresh R4 spatial path with the same current reduced law."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

from .current_coupled_execution_r4 import build_current_r4_engine, complete_state
from .matched_persistent_execution_r3 import (
    stress_contract,
)
from .reduced_persistent_reference_r2 import (
    ReducedPersistentReversibleTipEngine, advance_signed, runtime_receipt,
)
from .stochastic_avalanche_tip import (
    clear_pending_geometry_events, pop_pending_geometry_event,
)


MODEL_ID = "r4.fresh-spatial-current-law-comparison/1"


def _events():
    rows = []
    while True:
        item = pop_pending_geometry_event()
        if item is None:
            return rows
        rows.append(item)


def _build_spatial_event_law_engine():
    engine_type = ReducedPersistentReversibleTipEngine
    old_hazard = copy.deepcopy(engine_type._hazard_config_default)
    old_avalanche = copy.deepcopy(engine_type._avalanche_config_default)
    try:
        engine_type.configure_hazard(mode="exponential", seed=1733)
        engine_type.configure_avalanche(
            mode="threshold_scaled", minimum_factor=0.5,
            maximum_factor=4.0, geometry_subsegment_fraction=0.1,
        )
        return build_current_r4_engine()
    finally:
        engine_type._hazard_config_default = old_hazard
        engine_type._avalanche_config_default = old_avalanche


def _run_reduced(K_values: list[float], labels: list[str], dt_s: float):
    engine = _build_spatial_event_law_engine()
    rows = []
    clear_pending_geometry_events()
    for label, K in zip(labels, K_values, strict=True):
        before = stress_contract(engine, K)
        result = advance_signed(
            engine, K_Pa_sqrt_m=K, temperature_K=900.0, dt_s=dt_s,
        )
        rows.append({
            "stage": label, "applied_K_Pa_sqrt_m": K,
            "stress_before": before, "result": result,
            "geometry_events": _events(), "complete_state": complete_state(engine),
        })
    checkpoint = copy.deepcopy(engine)
    post = advance_signed(
        engine, K_Pa_sqrt_m=K_values[-1], temperature_K=900.0, dt_s=dt_s,
    )
    post_events = _events()
    restarted_post = advance_signed(
        checkpoint, K_Pa_sqrt_m=K_values[-1], temperature_K=900.0, dt_s=dt_s,
    )
    restarted_events = _events()
    return {
        "rows": rows,
        "restart": {
            "uninterrupted_result": post, "restarted_result": restarted_post,
            "uninterrupted_events": post_events,
            "restarted_events": restarted_events,
            "bitwise_complete_state_equal": (
                complete_state(engine) == complete_state(checkpoint)
            ),
        },
        "runtime_receipt": runtime_receipt(
            engine, stage="r4-reduced-matched-to-fresh-spatial-law"
        ),
    }


def execute_current_law_comparison(
    spatial_directory: str | Path, output: str | Path,
) -> dict[str, Any]:
    spatial_root = Path(spatial_directory).resolve()
    spatial_rows = list(csv.DictReader(
        (spatial_root / "steps_0900K.csv").open()
    ))
    if len(spatial_rows) != 5:
        raise RuntimeError("R4 spatial comparison requires five accepted rows")
    openings = [float(row["Uapp_m"]) for row in spatial_rows]
    expected = [1.0e-5, 2.0e-5, 3.0e-5, 2.0e-5, 3.0e-5]
    if openings != expected:
        raise RuntimeError(f"unexpected spatial opening history: {openings!r}")
    labels = ["load_10um", "load_20um", "load_30um",
              "unload_20um", "reload_30um"]
    prescribed_K = [float(row["KJ_Pa_sqrtm"]) for row in spatial_rows]
    # Independent reduced mechanics: retain the virgin 10 um compliance and
    # scale K linearly with prescribed opening.  It does not ingest later
    # spatial state, crack geometry, stress, or event time.
    virgin_scale = prescribed_K[0] / openings[0]
    autonomous_K = [virgin_scale * opening for opening in openings]
    dt = float(spatial_rows[0]["dt_cur_s"])
    replay = _run_reduced(prescribed_K, labels, dt)
    autonomous = _run_reduced(autonomous_K, labels, dt)
    comparisons = []
    for index, (label, spatial, replay_row, auto_row) in enumerate(zip(
        labels, spatial_rows, replay["rows"], autonomous["rows"], strict=True
    ), start=1):
        spatial_sigma = float(spatial["sigma_tip_Pa"])
        comparisons.append({
            "step": index, "stage": label, "opening_m": openings[index - 1],
            "spatial_K_Pa_sqrt_m": prescribed_K[index - 1],
            "spatial_sigma_tip_Pa": spatial_sigma,
            "spatial_N_em": float(spatial["N_em"]),
            "spatial_fired": bool(float(spatial["n_fire"]) > 0.0),
            "spatial_projected_tip_m": float(spatial["a_tip_m"]),
            "replay_sigma_tip_Pa": float(replay_row["result"]["sigma_tip"]),
            "replay_N_em": float(replay_row["result"]["N_em"]),
            "replay_fired": bool(replay_row["result"]["fired"]),
            "autonomous_reduced_K_Pa_sqrt_m": autonomous_K[index - 1],
            "autonomous_reduced_sigma_tip_Pa": float(auto_row["result"]["sigma_tip"]),
            "autonomous_reduced_N_em": float(auto_row["result"]["N_em"]),
            "autonomous_reduced_fired": bool(auto_row["result"]["fired"]),
        })
    summary = json.loads((spatial_root / "summary.json").read_text())[0]
    result = {
        "schema": MODEL_ID,
        "spatial_source": str(spatial_root),
        "spatial_current_law_receipt": json.loads(
            (spatial_root / "r4_current_law_spatial.json").read_text()
        ),
        "loading_controller": {
            "spatial": "PRESCRIBED_ABSOLUTE_OPENING_PER_ACCEPTED_NONADAPTIVE_STEP",
            "reduced_prescribed_drive_replay": "FRESH_SPATIAL_K_ROWS_AS_INPUT",
            "reduced_autonomous": "VIRGIN_10UM_LINEAR_K_OPENING_CLOSURE",
            "opening_history_m": openings, "duration_per_stage_s": dt,
            "event_terminates_stage": True,
            "unused_kinetic_time_is_not_relabelled_as_hold": True,
        },
        "event_strategy": {
            "hazard": "exponential", "hazard_seed": 1733,
            "reward": "threshold_scaled", "minimum_factor": 0.5,
            "maximum_factor": 4.0,
        },
        "spatial_summary": summary,
        "prescribed_drive_replay": replay,
        "autonomous_reduced": autonomous,
        "comparisons": comparisons,
        "qualifications": {
            "same_material_selection": True,
            "same_persistent_reversible_taylor_strategy": True,
            "same_event_strategy": True,
            "spatial_has_at_least_two_advances": int(summary["n_geometry_events"]) >= 2,
            "positive_unload_reload_executed": openings[-2:] == [2.0e-5, 3.0e-5],
            "spatial_sigma_cap_Pa": 30.0e9,
            "spatial_sigma_cap_active_any_row": any(
                float(row["sigma_tip_Pa"]) >= 30.0e9 for row in spatial_rows
            ),
            "autonomous_reduced_is_not_finite_specimen_prediction": True,
            "reduced_spatial_complete_state_equality_claimed": False,
        },
    }
    result["passed"] = bool(
        result["qualifications"]["spatial_has_at_least_two_advances"]
        and result["qualifications"]["positive_unload_reload_executed"]
        and replay["restart"]["bitwise_complete_state_equal"]
        and autonomous["restart"]["bitwise_complete_state_equal"]
    )
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


__all__ = ["MODEL_ID", "execute_current_law_comparison"]
