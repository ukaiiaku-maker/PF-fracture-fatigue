"""R5 state/step correspondence qualifications for the current backend."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .current_coupled_execution_r4 import build_current_r4_engine, complete_state
from .reduced_persistent_reference_r2 import advance_signed
from .stochastic_avalanche_tip import pending_geometry_event_scope


MODEL_ID = "r5.current-backend-state-correspondence/1"


@dataclass(frozen=True)
class FixedHorizonResult:
    requested_s: float
    consumed_s: float
    calls: tuple[dict[str, Any], ...]
    geometry_events: tuple[dict[str, Any], ...]


def advance_fixed_horizon(
    engine: Any, *, applied_K_Pa_sqrt_m: float, temperature_K: float,
    duration_s: float, maximum_outer_substep_s: float,
) -> FixedHorizonResult:
    """Advance an accepted owner over all requested time, including remainders.

    The native engine is event-terminated.  This controller commits the event,
    refreshes the accepted state, and resumes the unused physical interval.  A
    private descriptor queue is owned for the entire transaction.
    """
    requested = float(duration_s)
    maximum = float(maximum_outer_substep_s)
    if not (math.isfinite(requested) and requested > 0.0):
        raise ValueError("duration must be positive and finite")
    if not (math.isfinite(maximum) and maximum > 0.0):
        raise ValueError("maximum outer substep must be positive and finite")
    remaining = requested
    rows: list[dict[str, Any]] = []
    tolerance = max(64.0 * math.ulp(requested), 1.0e-15 * requested)
    with pending_geometry_event_scope() as pending:
        while remaining > tolerance:
            call_requested = min(remaining, maximum)
            result = advance_signed(
                engine, K_Pa_sqrt_m=float(applied_K_Pa_sqrt_m),
                temperature_K=float(temperature_K), dt_s=call_requested,
            )
            consumed = float(result["kinetic_dt_consumed_s"])
            unused = float(result["kinetic_dt_unused_s"])
            if consumed <= 0.0 or consumed > call_requested + tolerance:
                raise RuntimeError("current backend failed to consume valid physical time")
            rows.append({
                "requested_s": call_requested, "consumed_s": consumed,
                "unused_s": unused, "fired": bool(result["fired"]),
                "N_em": float(result["N_em"]),
                "r_eff_m": float(result["r_eff"]),
                "hazard_action": float(result["hazard_action_current"]),
                "internal_substeps": int(result["kinetic_internal_substeps"]),
            })
            remaining = max(remaining - consumed, 0.0)
            if len(rows) > 100000:
                raise RuntimeError("fixed-horizon event continuation exceeded its bound")
        events = tuple(copy.deepcopy(item) for item in pending)
    return FixedHorizonResult(
        requested_s=requested, consumed_s=requested - remaining,
        calls=tuple(rows), geometry_events=events,
    )


def _observable(engine: Any, run: FixedHorizonResult) -> dict[str, Any]:
    last = run.calls[-1]
    return {
        "N_em": last["N_em"], "r_eff_m": last["r_eff_m"],
        "hazard_action": last["hazard_action"],
        "event_count": len(run.geometry_events), "backend_call_count": len(run.calls),
        "native_internal_substeps": sum(row["internal_substeps"] for row in run.calls),
        "complete_state": complete_state(engine),
    }


def qualify_no_event_partition(
    output: str | Path, *, partitions: Iterable[int] = (1, 2, 4, 8, 16, 32, 64, 128),
) -> dict[str, Any]:
    """Trace outer source-coupling error at the audited constant-K stage."""
    K = 16.9223230932e6
    duration = 8.4
    parent = build_current_r4_engine()
    rows = []
    for count in partitions:
        n = int(count)
        engine = copy.deepcopy(parent)
        run = advance_fixed_horizon(
            engine, applied_K_Pa_sqrt_m=K, temperature_K=900.0,
            duration_s=duration, maximum_outer_substep_s=duration / n,
        )
        observed = _observable(engine, run)
        rows.append({
            "partition_count": n, "outer_substep_s": duration / n,
            **observed,
        })
    reference = rows[-1]
    for row in rows:
        row["relative_N_em_error_vs_finest"] = (
            (row["N_em"] - reference["N_em"]) / reference["N_em"]
        )
        row["absolute_r_eff_error_m_vs_finest"] = (
            row["r_eff_m"] - reference["r_eff_m"]
        )
        row["absolute_hazard_action_error_vs_finest"] = (
            row["hazard_action"] - reference["hazard_action"]
        )
    result = {
        "schema": MODEL_ID,
        "experiment": {
            "loading_kind": "CONSTANT_APPLIED_K_HOLD",
            "K_Pa_sqrt_m": K, "temperature_K": 900.0,
            "physical_horizon_s": duration, "fracture_event_expected": False,
            "external_partition_is_outer_source_coupling_resolution": True,
            "native_max_action_substep_does_not_activate": True,
        },
        "rows": rows,
        "diagnosis": {
            "one_call_N_em": rows[0]["N_em"],
            "two_call_N_em": rows[1]["N_em"],
            "one_call_r_eff_m": rows[0]["r_eff_m"],
            "two_call_r_eff_m": rows[1]["r_eff_m"],
            "cause": (
                "OUTER_NONLINEAR_SOURCE_BACKSTRESS_AND_POST_HALF_STEP_RATE_"
                "REFRESH; FRACTURE_ACTION_CONTROL_IS_INACTIVE_ON_THIS_INTERVAL"
            ),
            "backward_euler_source_dependence_in_scope": True,
            "transport_and_moving_window_evaluated_at_each_outer_refresh": True,
            "root_localization_active": False,
        },
        "qualification": {
            "common_initial_complete_owner": True,
            "same_physical_loading_horizon": True,
            "all_event_free": all(row["event_count"] == 0 for row in rows),
            "monotone_population_refinement": all(
                rows[i + 1]["N_em"] <= rows[i]["N_em"]
                for i in range(len(rows) - 1)
            ),
            "finest_last_doubling_relative_N_em_change": abs(
                (rows[-1]["N_em"] - rows[-2]["N_em"]) / rows[-1]["N_em"]
            ),
        },
    }
    result["passed"] = bool(
        result["qualification"]["all_event_free"]
        and result["qualification"]["monotone_population_refinement"]
        and math.isclose(rows[0]["N_em"], 4360.146965128, rel_tol=2e-12)
        and math.isclose(rows[1]["N_em"], 3418.293054431, rel_tol=2e-12)
    )
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


__all__ = [
    "FixedHorizonResult", "MODEL_ID", "advance_fixed_horizon",
    "qualify_no_event_partition",
]
