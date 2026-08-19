"""Projective inter-cycle accelerator for intrinsic reverse-glide v7.

The within-cycle physics remains exactly ``advance_v7_cycle``.  Acceleration is
restricted to cycle-end state projection between resolved anchors.  The initial
qualification version intentionally supports only a two-cycle block:

    resolved anchor n -> project endpoint n+1 -> resolve cycle n+2 exactly.

The skipped-cycle hazard is reconstructed *after* the right anchor is resolved
by log-linear interpolation between the two positive anchor hazards.  This
keeps the hazard positive and avoids the demonstrably poor raw affine hazard
extrapolation while preserving the same Arrhenius cycle map at every resolved
anchor.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from v914_v7_cycle_map import advance_v7_cycle, cycle_endpoint_summary
from v914_v7_multicycle_accelerator import run_exact_multicycle
from v914_v7_projective_state import (
    ACTIVE_MONOTONIC_ARRAYS,
    ACTIVE_NONNEGATIVE_ARRAYS,
    PROJECTOR_ID,
    project_v7_state_secant,
)


ACCELERATOR_ID = "v9.14_v7_projective_cycle_accelerator_v1_stride2"


@dataclass(frozen=True)
class ProjectiveAcceleratorControls:
    warmup_cycles: int = 4
    block_stride: int = 2
    max_projection_constraint_correction: float = 0.10

    def validate(self) -> None:
        if int(self.warmup_cycles) < 2:
            raise ValueError("warmup_cycles must be at least 2")
        # Deliberately fail closed until the first prospective stride-2 gate is
        # complete.  Larger strides are a later numerical promotion, not a new
        # constitutive parameterization.
        if int(self.block_stride) != 2:
            raise ValueError("v1 projective accelerator is qualified only for block_stride=2")
        value = float(self.max_projection_constraint_correction)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("max_projection_constraint_correction must be positive and finite")


def log_bridge_hazards(left: float, right: float, count: int) -> list[float]:
    """Positive log-linear hazards at ``count`` points between two anchors."""
    n = int(count)
    if n < 0:
        raise ValueError("count must be nonnegative")
    if n == 0:
        return []
    floor = 1.0e-300
    a = max(float(left), floor)
    b = max(float(right), floor)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("anchor hazards must be finite")
    la = math.log(a)
    lb = math.log(b)
    return [
        math.exp((1.0 - theta) * la + theta * lb)
        for theta in ((j + 1.0) / (n + 1.0) for j in range(n))
    ]


def _telemetry_record(cycle: int, telemetry: dict[str, Any], resolution: str) -> dict[str, Any]:
    return {
        "cycle_index": int(cycle),
        "resolution": resolution,
        "cycle_map_id": telemetry.get("cycle_map_id"),
        "accepted_intervals": int(telemetry.get("accepted_intervals", 0)),
        "refined_intervals": int(telemetry.get("refined_intervals", 0)),
        "maximum_depth_reached": int(telemetry.get("maximum_depth_reached", 0)),
        "minimum_accepted_phase_width": float(
            telemetry.get("minimum_accepted_phase_width", math.nan)
        ),
    }


def _add_cumulative_hazard(records: list[dict[str, Any]]) -> None:
    total = 0.0
    for row in records:
        total += max(float(row["hazard_action"]), 0.0)
        row["cumulative_hazard_action"] = total


def run_projective_multicycle(
    initial_state,
    loading,
    cycle_controls,
    cycles: int,
    *,
    accelerator_controls: ProjectiveAcceleratorControls | None = None,
    cycle_map_fn: Callable[..., tuple[Any, float, dict[str, Any]]] = advance_v7_cycle,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run warm-up exact cycles followed by stride-2 full-state projection.

    A skipped endpoint is projected from the secant between the two most recent
    resolved anchors, normalized by their physical cycle separation.  The next
    cycle is then resolved by the authoritative v7 cycle map.  No projected state
    is allowed to cross a crack-extension event.
    """
    ncycles = int(cycles)
    if ncycles < 1:
        raise ValueError("cycles must be positive")
    controls = accelerator_controls or ProjectiveAcceleratorControls()
    controls.validate()

    state = copy.deepcopy(initial_state)
    records: list[dict[str, Any]] = []
    telemetry_records: list[dict[str, Any]] = []
    resolved_cycle_count = 0
    projected_cycle_count = 0
    maximum_projection_correction = 0.0

    warmup = min(int(controls.warmup_cycles), ncycles)
    warmup_states: list[tuple[int, Any, float]] = []
    for cycle in range(1, warmup + 1):
        state, hazard, telemetry = cycle_map_fn(state, loading, cycle_controls)
        resolved_cycle_count += 1
        summary = cycle_endpoint_summary(state, hazard, cycle)
        summary["resolution"] = "shared_cycle_map_warmup"
        summary["projector_id"] = ""
        summary["projection_max_relative_constraint_correction"] = 0.0
        records.append(summary)
        telemetry_records.append(_telemetry_record(cycle, telemetry, summary["resolution"]))
        warmup_states.append((cycle, copy.deepcopy(state), float(hazard)))

    if ncycles <= warmup:
        _add_cumulative_hazard(records)
        return state, records, telemetry_records, {
            "accelerator_id": ACCELERATOR_ID,
            "projector_id": PROJECTOR_ID,
            "warmup_cycles": warmup,
            "block_stride": int(controls.block_stride),
            "resolved_cycle_count": resolved_cycle_count,
            "projected_cycle_count": projected_cycle_count,
            "cycle_resolution_fraction": resolved_cycle_count / ncycles,
            "ideal_cycle_map_speedup": ncycles / max(resolved_cycle_count, 1),
            "maximum_projection_constraint_correction": maximum_projection_correction,
        }

    # Need two resolved endpoints to define the first cycle-end secant.
    previous_anchor_cycle, previous_anchor_state, previous_anchor_hazard = warmup_states[-2]
    current_anchor_cycle, current_anchor_state, current_anchor_hazard = warmup_states[-1]
    state = copy.deepcopy(current_anchor_state)

    while current_anchor_cycle < ncycles:
        remaining = ncycles - current_anchor_cycle
        if remaining == 1:
            # Do not leave an unbracketed projected hazard at the end of a run.
            next_cycle = current_anchor_cycle + 1
            state, hazard, telemetry = cycle_map_fn(
                current_anchor_state, loading, cycle_controls
            )
            resolved_cycle_count += 1
            summary = cycle_endpoint_summary(state, hazard, next_cycle)
            summary["resolution"] = "shared_cycle_map_exact_tail"
            summary["projector_id"] = ""
            summary["projection_max_relative_constraint_correction"] = 0.0
            records.append(summary)
            telemetry_records.append(
                _telemetry_record(next_cycle, telemetry, summary["resolution"])
            )
            current_anchor_cycle = next_cycle
            current_anchor_state = copy.deepcopy(state)
            current_anchor_hazard = float(hazard)
            break

        anchor_gap = current_anchor_cycle - previous_anchor_cycle
        if anchor_gap < 1:
            raise RuntimeError("invalid projective anchor ordering")

        skipped_cycle = current_anchor_cycle + 1
        projected_state, projection = project_v7_state_secant(
            previous_anchor_state,
            current_anchor_state,
            anchor_gap_cycles=anchor_gap,
            skip_cycles=1,
            frequency_Hz=float(loading.frequency_Hz),
        )
        correction = float(projection["maximum_relative_constraint_correction"])
        maximum_projection_correction = max(maximum_projection_correction, correction)
        if correction > float(controls.max_projection_constraint_correction):
            raise RuntimeError(
                "projective state required excessive physical constraint correction: "
                f"cycle={skipped_cycle}, correction={correction:.6g}, "
                f"limit={controls.max_projection_constraint_correction:.6g}"
            )
        projected_cycle_count += 1

        skipped_summary = cycle_endpoint_summary(projected_state, 0.0, skipped_cycle)
        skipped_summary["resolution"] = "projected_skip"
        skipped_summary["projector_id"] = PROJECTOR_ID
        skipped_summary["projection_anchor_gap_cycles"] = int(anchor_gap)
        skipped_summary["projection_max_relative_constraint_correction"] = correction

        anchor_cycle = skipped_cycle + 1
        anchor_state, anchor_hazard, telemetry = cycle_map_fn(
            projected_state, loading, cycle_controls
        )
        resolved_cycle_count += 1
        anchor_summary = cycle_endpoint_summary(anchor_state, anchor_hazard, anchor_cycle)
        anchor_summary["resolution"] = "shared_cycle_map_anchor"
        anchor_summary["projector_id"] = PROJECTOR_ID
        anchor_summary["projection_anchor_gap_cycles"] = int(anchor_gap)
        anchor_summary["projection_max_relative_constraint_correction"] = correction

        bridge = log_bridge_hazards(current_anchor_hazard, anchor_hazard, 1)
        skipped_summary["hazard_action"] = float(bridge[0])
        skipped_summary["hazard_reconstruction"] = "log_bridge_between_resolved_anchors"
        anchor_summary["hazard_reconstruction"] = "resolved_shared_cycle_map"

        records.extend((skipped_summary, anchor_summary))
        telemetry_records.append(
            {
                "cycle_index": skipped_cycle,
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

        previous_anchor_cycle = current_anchor_cycle
        previous_anchor_state = copy.deepcopy(current_anchor_state)
        previous_anchor_hazard = current_anchor_hazard
        current_anchor_cycle = anchor_cycle
        current_anchor_state = copy.deepcopy(anchor_state)
        current_anchor_hazard = float(anchor_hazard)
        state = copy.deepcopy(anchor_state)

    records.sort(key=lambda row: int(row["cycle_index"]))
    _add_cumulative_hazard(records)
    metadata = {
        "accelerator_id": ACCELERATOR_ID,
        "projector_id": PROJECTOR_ID,
        "warmup_cycles": warmup,
        "block_stride": int(controls.block_stride),
        "resolved_cycle_count": resolved_cycle_count,
        "projected_cycle_count": projected_cycle_count,
        "cycle_resolution_fraction": resolved_cycle_count / ncycles,
        "ideal_cycle_map_speedup": ncycles / max(resolved_cycle_count, 1),
        "maximum_projection_constraint_correction": maximum_projection_correction,
        "skipped_cycle_hazard_rule": "log_bridge_between_resolved_anchor_hazards",
        "within_cycle_law": "advance_v7_cycle_at_every_resolved_cycle",
    }
    return state, records, telemetry_records, metadata


def _scaled_relative(a: float, b: float, floor: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), float(floor))


def _field_norm_error(exact_state, accelerated_state, name: str) -> float:
    a = np.asarray(getattr(exact_state, name), dtype=float)
    b = np.asarray(getattr(accelerated_state, name), dtype=float)
    if a.shape != b.shape:
        return math.inf
    return float(np.linalg.norm(b - a)) / max(
        float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0
    )


def compare_projective_to_exact(
    exact_final_state,
    exact_records: list[dict[str, Any]],
    accelerated_final_state,
    accelerated_records: list[dict[str, Any]],
    *,
    warmup_cycles: int,
) -> dict[str, Any]:
    """Numerical qualification metrics; none of these tolerances alter physics."""
    if len(exact_records) != len(accelerated_records):
        raise ValueError("exact/accelerated history lengths differ")
    exact_by_cycle = {int(row["cycle_index"]): row for row in exact_records}
    accel_by_cycle = {int(row["cycle_index"]): row for row in accelerated_records}
    if set(exact_by_cycle) != set(accel_by_cycle):
        raise ValueError("exact/accelerated cycle indices differ")

    floors = {
        "shielding_MPa_sqrt_m": 1.0e-3,
        "mobile_line_content": 1.0,
        "retained_line_content": 1.0,
        "returned_source_slip": 1.0e-3,
        "physical_return_fraction": 1.0e-6,
        "tip_radius_m": 1.0e-9,
        "cumulative_source_slip": 1.0,
        "raw_return_fraction": 1.0e-6,
    }
    anchor_fields = (
        "shielding_MPa_sqrt_m",
        "mobile_line_content",
        "retained_line_content",
        "returned_source_slip",
        "tip_radius_m",
        "cumulative_source_slip",
    )
    anchor_max = {name: 0.0 for name in anchor_fields}
    all_cycle_max = {name: 0.0 for name in anchor_fields}
    max_post_warmup_log10_hazard_error = 0.0

    for cycle in sorted(exact_by_cycle):
        e = exact_by_cycle[cycle]
        a = accel_by_cycle[cycle]
        for name in anchor_fields:
            err = _scaled_relative(e[name], a[name], floors[name])
            all_cycle_max[name] = max(all_cycle_max[name], err)
            if cycle > int(warmup_cycles) and a.get("resolution") != "projected_skip":
                anchor_max[name] = max(anchor_max[name], err)
        if cycle > int(warmup_cycles):
            eh = max(float(e["hazard_action"]), 1.0e-300)
            ah = max(float(a["hazard_action"]), 1.0e-300)
            max_post_warmup_log10_hazard_error = max(
                max_post_warmup_log10_hazard_error,
                abs(math.log10(ah / eh)),
            )

    warmup = int(warmup_cycles)
    exact_post_hazard = sum(
        max(float(row["hazard_action"]), 0.0)
        for row in exact_records
        if int(row["cycle_index"]) > warmup
    )
    accel_post_hazard = sum(
        max(float(row["hazard_action"]), 0.0)
        for row in accelerated_records
        if int(row["cycle_index"]) > warmup
    )
    post_hazard_relerr = _scaled_relative(
        exact_post_hazard, accel_post_hazard, 1.0e-30
    )
    total_exact_hazard = sum(max(float(row["hazard_action"]), 0.0) for row in exact_records)
    total_accel_hazard = sum(max(float(row["hazard_action"]), 0.0) for row in accelerated_records)
    total_hazard_relerr = _scaled_relative(
        total_exact_hazard, total_accel_hazard, 1.0e-30
    )

    final_full_state = {
        name: _field_norm_error(exact_final_state, accelerated_final_state, name)
        for name in ACTIVE_NONNEGATIVE_ARRAYS + ACTIVE_MONOTONIC_ARRAYS
    }

    # Initial engineering qualification only.  These gates judge the numerical
    # approximation and are not fitted material/kinetic parameters.
    limits = {
        "anchor_shielding": 0.05,
        "anchor_mobile": 0.02,
        "anchor_retained": 0.05,
        "anchor_returned": 0.10,
        "anchor_tip_radius": 0.02,
        "anchor_source_slip": 0.02,
        "post_warmup_cumulative_hazard": 0.10,
        "final_full_state": 0.05,
    }
    checks = {
        "anchor_shielding": anchor_max["shielding_MPa_sqrt_m"] <= limits["anchor_shielding"],
        "anchor_mobile": anchor_max["mobile_line_content"] <= limits["anchor_mobile"],
        "anchor_retained": anchor_max["retained_line_content"] <= limits["anchor_retained"],
        "anchor_returned": anchor_max["returned_source_slip"] <= limits["anchor_returned"],
        "anchor_tip_radius": anchor_max["tip_radius_m"] <= limits["anchor_tip_radius"],
        "anchor_source_slip": anchor_max["cumulative_source_slip"] <= limits["anchor_source_slip"],
        "post_warmup_cumulative_hazard": post_hazard_relerr <= limits["post_warmup_cumulative_hazard"],
        "final_full_state": max(final_full_state.values(), default=0.0) <= limits["final_full_state"],
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "limits": limits,
        "anchor_max_relative_error": anchor_max,
        "all_cycle_max_relative_error": all_cycle_max,
        "final_full_state_relative_norm_error": final_full_state,
        "post_warmup_exact_hazard_action": exact_post_hazard,
        "post_warmup_accelerated_hazard_action": accel_post_hazard,
        "post_warmup_cumulative_hazard_relative_error": post_hazard_relerr,
        "total_cumulative_hazard_relative_error": total_hazard_relerr,
        "max_post_warmup_cycle_log10_hazard_error": max_post_warmup_log10_hazard_error,
    }


__all__ = [
    "ACCELERATOR_ID",
    "ProjectiveAcceleratorControls",
    "compare_projective_to_exact",
    "log_bridge_hazards",
    "run_projective_multicycle",
    "run_exact_multicycle",
]
