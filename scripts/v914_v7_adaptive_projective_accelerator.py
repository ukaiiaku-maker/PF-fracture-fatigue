"""Adaptive promotion for the v9.14 intrinsic reverse-glide v7 accelerator.

The authoritative within-cycle map remains ``advance_v7_cycle``.  This module
changes only *when* inter-cycle projection is permitted.  A fixed warm-up count
is not sufficient when the early cyclic state contains a strong transient.
Projection is therefore enabled only after the actual resolved endpoint state
has been predicted acceptably from the two preceding resolved endpoints for a
specified number of consecutive cycles.

The readiness criterion is numerical, not constitutive.  It compares the full
spatial mobile/retained/source-slip/returned-slip fields and key derived endpoint
observables.  Hazard is intentionally excluded because skipped-cycle hazard is
reconstructed between resolved anchors rather than projected as a state field.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from v914_v7_cycle_map import advance_v7_cycle, cycle_endpoint_summary
from v914_v7_projective_accelerator import (
    ACCELERATOR_ID as STRIDE2_ACCELERATOR_ID,
    ProjectiveAcceleratorControls,
    _add_cumulative_hazard,
    _telemetry_record,
    log_bridge_hazards,
)
from v914_v7_projective_state import (
    ACTIVE_MONOTONIC_ARRAYS,
    ACTIVE_NONNEGATIVE_ARRAYS,
    PROJECTOR_ID,
    project_v7_state_secant,
)


ACCELERATOR_ID = "v9.14_v7_adaptive_promotion_projective_accelerator_v1_stride2"


@dataclass(frozen=True)
class AdaptivePromotionControls:
    minimum_exact_cycles: int = 4
    readiness_relative_tolerance: float = 0.05
    readiness_consecutive_passes: int = 2
    block_stride: int = 2
    max_projection_constraint_correction: float = 0.10

    def validate(self) -> None:
        if int(self.minimum_exact_cycles) < 3:
            raise ValueError("minimum_exact_cycles must be at least 3")
        if int(self.readiness_consecutive_passes) < 1:
            raise ValueError("readiness_consecutive_passes must be positive")
        if int(self.block_stride) != 2:
            raise ValueError("adaptive promotion v1 supports only block_stride=2")
        for name, value in (
            ("readiness_relative_tolerance", self.readiness_relative_tolerance),
            (
                "max_projection_constraint_correction",
                self.max_projection_constraint_correction,
            ),
        ):
            x = float(value)
            if not math.isfinite(x) or x <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


def _norm_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    a = np.asarray(reference, dtype=float)
    b = np.asarray(candidate, dtype=float)
    if a.shape != b.shape:
        return math.inf
    return float(np.linalg.norm(b - a)) / max(
        float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0
    )


def _scaled_relative(a: float, b: float, floor: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), float(floor))


def readiness_prediction(
    state_nm2,
    state_nm1,
    state_n,
    *,
    frequency_Hz: float,
    cycle_index: int,
) -> dict[str, Any]:
    """Measure one-cycle projective predictability against an already-resolved state.

    ``state_nm2`` and ``state_nm1`` are consecutive resolved endpoints.  A
    parameter-free one-cycle secant prediction of ``state_n`` is constructed and
    compared with the actual resolved ``state_n``.  This check is available at
    essentially no additional constitutive cost because all three states have
    already been resolved through the shared cycle map.
    """
    predicted, projection = project_v7_state_secant(
        state_nm2,
        state_nm1,
        anchor_gap_cycles=1,
        skip_cycles=1,
        frequency_Hz=float(frequency_Hz),
    )

    field_errors = {
        name: _norm_error(getattr(state_n, name), getattr(predicted, name))
        for name in ACTIVE_NONNEGATIVE_ARRAYS + ACTIVE_MONOTONIC_ARRAYS
    }

    actual_summary = cycle_endpoint_summary(state_n, 0.0, int(cycle_index))
    predicted_summary = cycle_endpoint_summary(predicted, 0.0, int(cycle_index))
    floors = {
        "shielding_MPa_sqrt_m": 1.0e-3,
        "mobile_line_content": 1.0,
        "retained_line_content": 1.0,
        "returned_source_slip": 1.0e-3,
        "tip_radius_m": 1.0e-9,
        "cumulative_source_slip": 1.0,
    }
    summary_errors = {
        name: _scaled_relative(actual_summary[name], predicted_summary[name], floor)
        for name, floor in floors.items()
    }

    maximum = max(
        [0.0, *field_errors.values(), *summary_errors.values()]
    )
    return {
        "cycle_index": int(cycle_index),
        "maximum_relative_error": float(maximum),
        "full_state_relative_norm_error": field_errors,
        "summary_relative_error": summary_errors,
        "projection_constraint_correction": float(
            projection["maximum_relative_constraint_correction"]
        ),
        "projector_id": projection["projector_id"],
    }


def run_adaptive_projective_multicycle(
    initial_state,
    loading,
    cycle_controls,
    cycles: int,
    *,
    promotion_controls: AdaptivePromotionControls | None = None,
    cycle_map_fn: Callable[..., tuple[Any, float, dict[str, Any]]] = advance_v7_cycle,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Resolve transients exactly, then promote automatically to stride-2 projection.

    Before promotion every physical cycle is resolved through the authoritative
    cycle map.  Promotion requires consecutive successful retrospective
    one-cycle state predictions.  After promotion the usual stride-2 pattern is
    used: one projected endpoint followed by one resolved anchor.  If a proposed
    projection requires excessive physical constraint correction, the algorithm
    falls back to an exact cycle and returns to readiness checking rather than
    accepting the state or terminating the fatigue calculation.
    """
    ncycles = int(cycles)
    if ncycles < 1:
        raise ValueError("cycles must be positive")
    controls = promotion_controls or AdaptivePromotionControls()
    controls.validate()

    state = copy.deepcopy(initial_state)
    records: list[dict[str, Any]] = []
    telemetry_records: list[dict[str, Any]] = []
    resolved_states: list[tuple[int, Any, float]] = []
    readiness_history: list[dict[str, Any]] = []
    readiness_streak = 0
    promoted = False
    promotion_cycle: int | None = None
    resolved_cycle_count = 0
    projected_cycle_count = 0
    fallback_exact_count = 0
    maximum_projection_correction = 0.0

    cycle = 0
    while cycle < ncycles:
        # Exact mode is used during the transient and after any failed projection.
        if not promoted:
            cycle += 1
            state, hazard, telemetry = cycle_map_fn(state, loading, cycle_controls)
            resolved_cycle_count += 1
            summary = cycle_endpoint_summary(state, hazard, cycle)
            summary["resolution"] = "shared_cycle_map_readiness"
            summary["projector_id"] = ""
            summary["projection_max_relative_constraint_correction"] = 0.0
            records.append(summary)
            telemetry_records.append(
                _telemetry_record(cycle, telemetry, summary["resolution"])
            )
            resolved_states.append((cycle, copy.deepcopy(state), float(hazard)))
            if len(resolved_states) > 3:
                resolved_states = resolved_states[-3:]

            if len(resolved_states) == 3:
                c0, s0, _ = resolved_states[0]
                c1, s1, _ = resolved_states[1]
                c2, s2, _ = resolved_states[2]
                if c1 == c0 + 1 and c2 == c1 + 1:
                    readiness = readiness_prediction(
                        s0,
                        s1,
                        s2,
                        frequency_Hz=float(loading.frequency_Hz),
                        cycle_index=c2,
                    )
                    passed = bool(
                        readiness["maximum_relative_error"]
                        <= float(controls.readiness_relative_tolerance)
                        and readiness["projection_constraint_correction"]
                        <= float(controls.max_projection_constraint_correction)
                    )
                    readiness["pass"] = passed
                    readiness_history.append(readiness)
                    readiness_streak = readiness_streak + 1 if passed else 0

            if (
                cycle >= int(controls.minimum_exact_cycles)
                and readiness_streak >= int(controls.readiness_consecutive_passes)
                and cycle < ncycles - 1
            ):
                promoted = True
                promotion_cycle = cycle
            continue

        # Promoted stride-2 mode.  A single remaining cycle is always exact.
        remaining = ncycles - cycle
        if remaining == 1:
            cycle += 1
            state, hazard, telemetry = cycle_map_fn(state, loading, cycle_controls)
            resolved_cycle_count += 1
            summary = cycle_endpoint_summary(state, hazard, cycle)
            summary["resolution"] = "shared_cycle_map_exact_tail"
            summary["projector_id"] = ""
            summary["projection_max_relative_constraint_correction"] = 0.0
            records.append(summary)
            telemetry_records.append(
                _telemetry_record(cycle, telemetry, summary["resolution"])
            )
            break

        if len(resolved_states) < 2:
            raise RuntimeError("adaptive projector lacks two resolved anchor states")
        previous_cycle, previous_state, previous_hazard = resolved_states[-2]
        current_cycle, current_state, current_hazard = resolved_states[-1]
        anchor_gap = current_cycle - previous_cycle
        if anchor_gap < 1:
            raise RuntimeError("invalid adaptive projective anchor ordering")

        projected_cycle = cycle + 1
        projected_state, projection = project_v7_state_secant(
            previous_state,
            current_state,
            anchor_gap_cycles=anchor_gap,
            skip_cycles=1,
            frequency_Hz=float(loading.frequency_Hz),
        )
        correction = float(projection["maximum_relative_constraint_correction"])
        maximum_projection_correction = max(maximum_projection_correction, correction)

        if correction > float(controls.max_projection_constraint_correction):
            # Do not alter the threshold.  Resolve the next physical cycle and
            # resume readiness checking from the actual current state.
            cycle += 1
            state, hazard, telemetry = cycle_map_fn(
                current_state, loading, cycle_controls
            )
            resolved_cycle_count += 1
            fallback_exact_count += 1
            summary = cycle_endpoint_summary(state, hazard, cycle)
            summary["resolution"] = "shared_cycle_map_projection_fallback"
            summary["projector_id"] = ""
            summary["projection_max_relative_constraint_correction"] = correction
            records.append(summary)
            telemetry_records.append(
                _telemetry_record(cycle, telemetry, summary["resolution"])
            )
            resolved_states = [
                (current_cycle, copy.deepcopy(current_state), current_hazard),
                (cycle, copy.deepcopy(state), float(hazard)),
            ]
            readiness_streak = 0
            promoted = False
            continue

        projected_cycle_count += 1
        projected_summary = cycle_endpoint_summary(
            projected_state, 0.0, projected_cycle
        )
        projected_summary["resolution"] = "projected_skip"
        projected_summary["projector_id"] = PROJECTOR_ID
        projected_summary["projection_anchor_gap_cycles"] = int(anchor_gap)
        projected_summary["projection_max_relative_constraint_correction"] = correction

        anchor_cycle = projected_cycle + 1
        anchor_state, anchor_hazard, telemetry = cycle_map_fn(
            projected_state, loading, cycle_controls
        )
        resolved_cycle_count += 1
        anchor_summary = cycle_endpoint_summary(
            anchor_state, anchor_hazard, anchor_cycle
        )
        anchor_summary["resolution"] = "shared_cycle_map_anchor"
        anchor_summary["projector_id"] = PROJECTOR_ID
        anchor_summary["projection_anchor_gap_cycles"] = int(anchor_gap)
        anchor_summary["projection_max_relative_constraint_correction"] = correction

        bridge = log_bridge_hazards(current_hazard, anchor_hazard, 1)
        projected_summary["hazard_action"] = float(bridge[0])
        projected_summary["hazard_reconstruction"] = (
            "log_bridge_between_resolved_anchors"
        )
        anchor_summary["hazard_reconstruction"] = "resolved_shared_cycle_map"

        records.extend((projected_summary, anchor_summary))
        telemetry_records.append(
            {
                "cycle_index": projected_cycle,
                "resolution": "projected_skip",
                "cycle_map_id": None,
                "accepted_intervals": 0,
                "refined_intervals": 0,
                "maximum_depth_reached": 0,
                "minimum_accepted_phase_width": math.nan,
                "projector_id": PROJECTOR_ID,
                "projection_anchor_gap_cycles": int(anchor_gap),
                "projection_max_relative_constraint_correction": correction,
            }
        )
        telemetry_records.append(
            _telemetry_record(anchor_cycle, telemetry, "shared_cycle_map_anchor")
        )

        cycle = anchor_cycle
        state = copy.deepcopy(anchor_state)
        resolved_states = [
            (current_cycle, copy.deepcopy(current_state), current_hazard),
            (anchor_cycle, copy.deepcopy(anchor_state), float(anchor_hazard)),
        ]

    records.sort(key=lambda row: int(row["cycle_index"]))
    _add_cumulative_hazard(records)
    metadata = {
        "accelerator_id": ACCELERATOR_ID,
        "base_stride2_accelerator_id": STRIDE2_ACCELERATOR_ID,
        "projector_id": PROJECTOR_ID,
        "minimum_exact_cycles": int(controls.minimum_exact_cycles),
        "readiness_relative_tolerance": float(
            controls.readiness_relative_tolerance
        ),
        "readiness_consecutive_passes": int(
            controls.readiness_consecutive_passes
        ),
        "promotion_cycle": promotion_cycle,
        "readiness_history": readiness_history,
        "resolved_cycle_count": resolved_cycle_count,
        "projected_cycle_count": projected_cycle_count,
        "fallback_exact_count": fallback_exact_count,
        "cycle_resolution_fraction": resolved_cycle_count / ncycles,
        "ideal_cycle_map_speedup": ncycles / max(resolved_cycle_count, 1),
        "maximum_projection_constraint_correction": maximum_projection_correction,
        "skipped_cycle_hazard_rule": "log_bridge_between_resolved_anchor_hazards",
        "within_cycle_law": "advance_v7_cycle_at_every_resolved_cycle",
    }
    return state, records, telemetry_records, metadata


__all__ = [
    "ACCELERATOR_ID",
    "AdaptivePromotionControls",
    "readiness_prediction",
    "run_adaptive_projective_multicycle",
]
