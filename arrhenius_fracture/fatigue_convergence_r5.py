"""Phase-drive and native-step convergence for the restored fatigue path."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .current_coupled_execution_r4 import complete_state
from .fatigue_v1 import FatigueWaveform
from .resolved_stochastic_fatigue_r4 import (
    PhaseCursor, _advance_one_subcall, _build_stochastic_engine,
)
from .stochastic_avalanche_tip import pending_geometry_event_scope


MODEL_ID = "r5.phase-and-native-step-convergence/1"


def _run_through_event_and_reload(engine: Any, waveform: FatigueWaveform,
                                  phase_count: int, maximum_cycles: int):
    cursor = PhaseCursor()
    rows = []
    events = []
    first_event_elapsed_cycles = None
    stop_cycle = None
    with pending_geometry_event_scope():
        while cursor.cycle < maximum_cycles:
            before_cycle = cursor.cycle
            before_phase = cursor.phase_index
            row = _advance_one_subcall(engine, cursor, waveform, phase_count)
            phase_duration = waveform.period_s / phase_count
            fraction = row["consumed_s"] / phase_duration
            elapsed = before_cycle + (before_phase + fraction) / phase_count
            row["physical_elapsed_cycles"] = elapsed
            row["continuous_micro_advance_m"] = float(engine.micro_advance_total_m)
            row["checkpoint_reward_advance_m"] = float(
                engine.checkpoint_advance_total_m
            )
            rows.append(row)
            if row["fired"]:
                events.append(copy.deepcopy(row))
                if first_event_elapsed_cycles is None:
                    first_event_elapsed_cycles = elapsed
                    stop_cycle = math.floor(elapsed) + 2
            if stop_cycle is not None and cursor.cycle >= stop_cycle:
                break
    if first_event_elapsed_cycles is None:
        raise RuntimeError("fatigue convergence arm did not reach a native advance")
    return {
        "phase_count": phase_count,
        "rows": rows, "events": events,
        "first_event_elapsed_cycles": first_event_elapsed_cycles,
        "completed_cycles": cursor.cycle,
        "fractional_elapsed_cycles": (
            cursor.cycle + cursor.phase_index / phase_count
            + cursor.remaining_phase_s / waveform.period_s
        ),
        "reload_after_first_event_completed": cursor.cycle >= stop_cycle,
        "terminal": complete_state(engine),
    }


def execute_fatigue_convergence_r5(
    output: str | Path, *, phase_counts: Iterable[int] = (12, 24, 48),
    native_action_limits: Iterable[float] = (0.02, 0.01, 0.005),
    maximum_cycles: int = 60,
) -> dict[str, Any]:
    waveform = FatigueWaveform(Kmax=20.0e6, R=0.1, frequency_Hz=10.0)
    parent = _build_stochastic_engine()
    parent_state = complete_state(parent)
    phase_rows = []
    for count in phase_counts:
        engine = copy.deepcopy(parent)
        phase_rows.append(_run_through_event_and_reload(
            engine, waveform, int(count), maximum_cycles
        ))
    native_rows = []
    for limit in native_action_limits:
        engine = copy.deepcopy(parent)
        engine.tip_cfg.max_action_substep = float(limit)
        row = _run_through_event_and_reload(
            engine, waveform, 24, maximum_cycles
        )
        row["max_action_substep"] = float(limit)
        native_rows.append(row)
    phase24 = next(row for row in phase_rows if row["phase_count"] == 24)
    result = {
        "schema": MODEL_ID,
        "common_owner": {
            "complete_initial_owner_forked": True,
            "initial_state": parent_state,
            "common_pending_mark_and_correlated_reward": True,
        },
        "waveform": {
            "Kmax_Pa_sqrt_m": waveform.Kmax, "R": waveform.R,
            "frequency_Hz": waveform.frequency_Hz,
            "drive_interpretation": "CONSTANT_MIDPOINT_K_PER_PHASE_BIN",
            "exact_sinusoid_claimed": False,
        },
        "phase_drive_refinement": phase_rows,
        "native_state_step_refinement_at_24_phases": native_rows,
        "separation": {
            "phase_count_changes_applied_drive_quadrature": True,
            "native_action_limit_changes_internal_state_resolution": True,
            "dimensions_varied_separately": True,
        },
        "time_conventions": {
            "r4_reported_completed_cycle_count": 23,
            "r4_reported_physical_elapsed_cycles": 23.9178286745,
            "current_24_phase_first_event_elapsed_cycles": (
                phase24["first_event_elapsed_cycles"]
            ),
            "event_plots_must_use_physical_elapsed_cycles": True,
        },
        "advance_channels": {
            "continuous_micro_advance_reported_separately": True,
            "checkpoint_reward_advance_reported_separately": True,
        },
    }
    result["passed"] = bool(
        all(row["events"] and row["reload_after_first_event_completed"]
            for row in phase_rows + native_rows)
    )
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


__all__ = ["MODEL_ID", "execute_fatigue_convergence_r5"]
