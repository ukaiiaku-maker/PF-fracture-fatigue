"""Adaptive outer-phase integration for the v9.14 finite-tip reversibility audit.

This module changes no constitutive or fracture physics.  It replaces a fixed
uniform outer phase grid by deterministic step doubling so the nonlinear
feedback

    retained GND -> shielding -> signed transport -> storage/return

is refreshed only as finely as required by the evolving state.  The accepted
solution is always the two-half-step path.  A coarse one-step path is used only
as a local truncation/error indicator and is never committed.

The helper is intentionally audit-only.  It does not localize stochastic crack
events.  It is therefore appropriate for the one-cycle, zero-fracture-event
qualification cases used to establish numerical convergence of the reversible
transport/shielding transient before any production-integrator promotion.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from arrhenius_fracture import fatigue_v914 as base


@dataclass(frozen=True)
class AdaptiveFeedbackControls:
    """Numerical tolerances for deterministic state step doubling."""

    state_rtol: float = 1.0e-2
    tip_radius_rtol: float = 1.0e-3
    hazard_rtol: float = 1.0e-2
    base_phase_intervals: int = 256
    max_refinement_depth: int = 12
    shielding_scale_floor_MPa_sqrt_m: float = 0.1
    line_content_scale_floor: float = 1.0
    returned_slip_scale_floor: float = 1.0e-4
    hazard_scale_floor: float = 1.0e-14

    def validate(self) -> None:
        for value, name in (
            (self.state_rtol, "state_rtol"),
            (self.tip_radius_rtol, "tip_radius_rtol"),
            (self.hazard_rtol, "hazard_rtol"),
            (
                self.shielding_scale_floor_MPa_sqrt_m,
                "shielding_scale_floor_MPa_sqrt_m",
            ),
            (self.line_content_scale_floor, "line_content_scale_floor"),
            (self.returned_slip_scale_floor, "returned_slip_scale_floor"),
            (self.hazard_scale_floor, "hazard_scale_floor"),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if int(self.base_phase_intervals) < 1:
            raise ValueError("base_phase_intervals must be positive")
        if int(self.max_refinement_depth) < 0:
            raise ValueError("max_refinement_depth must be nonnegative")


def _relative_difference(a: float, b: float, floor: float) -> float:
    a = float(a)
    b = float(b)
    floor = max(float(floor), 1.0e-300)
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return abs(a - b) / max(abs(a), abs(b), floor)


def state_observables(state) -> dict[str, float]:
    """Return endpoint observables used by the adaptive error controller."""

    diagnostics = state.reversibility_diagnostics()
    cell_area = float(state.cell_area_m2)
    return {
        "shielding_MPa_sqrt_m": float(state.K_shield_MPa_sqrt_m()),
        "mobile_line_content": float(np.sum(state.mobile_m2) * cell_area),
        "retained_line_content": float(np.sum(state.retained_m2) * cell_area),
        "returned_source_slip": float(
            diagnostics.get("reversible_returned_source_slip_count", 0.0)
        ),
        "physical_return_fraction": float(
            diagnostics.get(
                "reversible_physical_return_fraction_of_emitted", 0.0
            )
        ),
        "tip_radius_m": float(state.tip_radius_m()),
    }


def state_error(
    coarse_state,
    fine_state,
    coarse_hazard: float,
    fine_hazard: float,
    controls: AdaptiveFeedbackControls,
) -> dict[str, float]:
    """Compare one full step with two half steps at the same endpoint."""

    coarse = state_observables(coarse_state)
    fine = state_observables(fine_state)
    return {
        "shielding": _relative_difference(
            coarse["shielding_MPa_sqrt_m"],
            fine["shielding_MPa_sqrt_m"],
            controls.shielding_scale_floor_MPa_sqrt_m,
        ),
        "mobile": _relative_difference(
            coarse["mobile_line_content"],
            fine["mobile_line_content"],
            controls.line_content_scale_floor,
        ),
        "retained": _relative_difference(
            coarse["retained_line_content"],
            fine["retained_line_content"],
            controls.line_content_scale_floor,
        ),
        "returned": _relative_difference(
            coarse["returned_source_slip"],
            fine["returned_source_slip"],
            controls.returned_slip_scale_floor,
        ),
        "tip_radius": _relative_difference(
            coarse["tip_radius_m"],
            fine["tip_radius_m"],
            1.0e-12,
        ),
        "hazard": _relative_difference(
            coarse_hazard,
            fine_hazard,
            controls.hazard_scale_floor,
        ),
    }


def error_passes(
    error: dict[str, float], controls: AdaptiveFeedbackControls
) -> bool:
    return bool(
        error["shielding"] <= controls.state_rtol
        and error["mobile"] <= controls.state_rtol
        and error["retained"] <= controls.state_rtol
        and error["returned"] <= controls.state_rtol
        and error["tip_radius"] <= controls.tip_radius_rtol
        and error["hazard"] <= controls.hazard_rtol
    )


def phase_sample(state, loading, phase: float, depth: int) -> dict[str, float]:
    """Record committed adaptive-path state at one phase location."""

    K = float(loading.K_at_phase(float(phase)))
    rates = state.local_rates(K, loading.temperature_K)
    obs = state_observables(state)
    diagnostics = state.reversibility_diagnostics()
    return {
        "phase": float(phase),
        "depth": int(depth),
        "K_MPa_sqrt_m": K,
        "shielding_MPa_sqrt_m": obs["shielding_MPa_sqrt_m"],
        "transport_K_signed_MPa_sqrt_m": float(
            rates["reversible_transport_K_signed_MPa_sqrt_m"]
        ),
        "mobile_line_content": obs["mobile_line_content"],
        "retained_line_content": obs["retained_line_content"],
        "returned_source_slip": obs["returned_source_slip"],
        "physical_return_fraction": obs["physical_return_fraction"],
        "tip_radius_m": obs["tip_radius_m"],
        "reverse_mobile_exposure_fraction": float(
            diagnostics.get(
                "reversible_reverse_mobile_exposure_fraction", 0.0
            )
        ),
    }


def adaptive_one_cycle(
    state,
    loading,
    controls: AdaptiveFeedbackControls,
    *,
    advance_phase_fn: Callable[..., float] = base._advance_phase,
) -> tuple[Any, float, dict[str, Any]]:
    """Advance exactly one cycle with recursive deterministic step doubling.

    The input state is not mutated.  The returned state follows only accepted
    two-half-step trajectories.  If the requested tolerance cannot be met by
    ``max_refinement_depth``, the routine raises rather than silently accepting
    an unconverged interval.
    """

    controls.validate()
    committed = copy.deepcopy(state)
    total_hazard = 0.0
    telemetry: dict[str, Any] = {
        "attempted_intervals": 0,
        "accepted_intervals": 0,
        "refined_intervals": 0,
        "maximum_depth_reached": 0,
        "minimum_accepted_phase_width": math.inf,
        "maximum_accepted_error": {
            "shielding": 0.0,
            "mobile": 0.0,
            "retained": 0.0,
            "returned": 0.0,
            "tip_radius": 0.0,
            "hazard": 0.0,
        },
        "samples": [phase_sample(committed, loading, 0.0, 0)],
    }

    def advance_interval(start_state, p0: float, p1: float, depth: int):
        telemetry["attempted_intervals"] += 1
        telemetry["maximum_depth_reached"] = max(
            telemetry["maximum_depth_reached"], int(depth)
        )
        midpoint = 0.5 * (p0 + p1)

        coarse = copy.deepcopy(start_state)
        coarse_hazard = float(advance_phase_fn(coarse, loading, p0, p1))

        fine = copy.deepcopy(start_state)
        h0 = float(advance_phase_fn(fine, loading, p0, midpoint))
        midpoint_state = copy.deepcopy(fine)
        h1 = float(advance_phase_fn(fine, loading, midpoint, p1))
        fine_hazard = h0 + h1

        if (
            coarse_hazard < 0.0
            or fine_hazard < 0.0
            or not math.isfinite(coarse_hazard)
            or not math.isfinite(fine_hazard)
        ):
            raise FloatingPointError("invalid adaptive phase hazard")

        error = state_error(
            coarse, fine, coarse_hazard, fine_hazard, controls
        )

        if error_passes(error, controls):
            telemetry["accepted_intervals"] += 1
            width = float(p1 - p0)
            telemetry["minimum_accepted_phase_width"] = min(
                telemetry["minimum_accepted_phase_width"], width
            )
            for name, value in error.items():
                telemetry["maximum_accepted_error"][name] = max(
                    telemetry["maximum_accepted_error"][name], float(value)
                )
            # Record both half-step states because the narrow reverse transient
            # can occur between nominal outer-grid endpoints.
            telemetry["samples"].append(
                phase_sample(midpoint_state, loading, midpoint, depth + 1)
            )
            telemetry["samples"].append(
                phase_sample(fine, loading, p1, depth + 1)
            )
            return fine, fine_hazard

        if depth >= controls.max_refinement_depth:
            raise RuntimeError(
                "adaptive feedback refinement failed closed at "
                f"phase=[{p0:.16g},{p1:.16g}], depth={depth}, error={error}"
            )

        telemetry["refined_intervals"] += 1
        first_state, first_hazard = advance_interval(
            start_state, p0, midpoint, depth + 1
        )
        second_state, second_hazard = advance_interval(
            first_state, midpoint, p1, depth + 1
        )
        return second_state, first_hazard + second_hazard

    nbase = int(controls.base_phase_intervals)
    for index in range(nbase):
        p0 = index / nbase
        p1 = (index + 1) / nbase
        committed, increment = advance_interval(committed, p0, p1, 0)
        total_hazard += increment

    if math.isinf(telemetry["minimum_accepted_phase_width"]):
        telemetry["minimum_accepted_phase_width"] = math.nan

    # Sort and deduplicate samples by phase/depth while preserving the finest
    # committed representation at repeated phase locations.
    by_phase: dict[float, dict[str, float]] = {}
    for sample in telemetry["samples"]:
        phase = float(sample["phase"])
        previous = by_phase.get(phase)
        if previous is None or sample["depth"] >= previous["depth"]:
            by_phase[phase] = sample
    telemetry["samples"] = [by_phase[p] for p in sorted(by_phase)]

    return committed, total_hazard, telemetry


__all__ = [
    "AdaptiveFeedbackControls",
    "adaptive_one_cycle",
    "error_passes",
    "phase_sample",
    "state_error",
    "state_observables",
]
