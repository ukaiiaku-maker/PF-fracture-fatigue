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
                                  phase_count: int, maximum_cycles: int,
                                  endpoint_cycles: float = 3.0,
                                  outer_refreshes_per_phase: int = 1):
    cursor = PhaseCursor()
    rows = []
    events = []
    first_event_elapsed_cycles = None
    elapsed_s = 0.0
    endpoint_s = float(endpoint_cycles) * waveform.period_s
    endpoint_tolerance_s = max(64.0*math.ulp(endpoint_s), 1.0e-15)
    with pending_geometry_event_scope():
        while cursor.cycle < maximum_cycles:
            if (first_event_elapsed_cycles is not None
                    and endpoint_s-elapsed_s <= endpoint_tolerance_s):
                break
            refresh_cap = (
                waveform.period_s / phase_count
                / int(outer_refreshes_per_phase)
            )
            if first_event_elapsed_cycles is not None:
                refresh_cap = min(refresh_cap, endpoint_s-elapsed_s)
            row = _advance_one_subcall(
                engine, cursor, waveform, phase_count,
                maximum_subcall_s=refresh_cap,
            )
            elapsed_s += float(row["consumed_s"])
            elapsed = elapsed_s / waveform.period_s
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
            if (first_event_elapsed_cycles is not None
                    and endpoint_s-elapsed_s <= endpoint_tolerance_s):
                break
    if first_event_elapsed_cycles is None:
        raise RuntimeError("fatigue convergence arm did not reach a native advance")
    return {
        "phase_count": phase_count,
        "outer_refreshes_per_phase": int(outer_refreshes_per_phase),
        "rows": rows, "events": events,
        "first_event_elapsed_cycles": first_event_elapsed_cycles,
        "completed_cycles": cursor.cycle,
        "fractional_elapsed_cycles": elapsed_s / waveform.period_s,
        "common_endpoint_cycles": endpoint_cycles,
        "reload_after_first_event_completed": (
            endpoint_s-elapsed_s <= endpoint_tolerance_s
        ),
        "elapsed_time_monotone": all(
            rows[index]["physical_elapsed_cycles"]
            > rows[index-1]["physical_elapsed_cycles"]
            for index in range(1, len(rows))
        ),
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
    outer_rows = []
    for refreshes in (1, 2):
        engine = copy.deepcopy(parent)
        outer_rows.append(_run_through_event_and_reload(
            engine, waveform, 24, maximum_cycles,
            outer_refreshes_per_phase=refreshes,
        ))
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
        "outer_source_refresh_refinement_at_24_phases": outer_rows,
        "separation": {
            "phase_count_changes_applied_drive_quadrature": True,
            "native_action_limit_changes_internal_state_resolution": True,
            "outer_refresh_changes_source_state_refresh_at_fixed_phase_drive": True,
            "dimensions_varied_separately": True,
        },
        "qualification": {
            "status": "REFINEMENT_EXECUTED",
            "accuracy_qualified": False,
            "reason": (
                "phase refinement remains outside a declared prospective "
                "first-event tolerance"
            ),
            "common_physical_endpoint_cycles": 3.0,
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
            and row["elapsed_time_monotone"]
            and math.isclose(row["fractional_elapsed_cycles"], 3.0,
                             rel_tol=0.0, abs_tol=2.0e-12)
            for row in phase_rows + native_rows + outer_rows)
    )
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


__all__ = ["MODEL_ID", "execute_fatigue_convergence_r5"]
