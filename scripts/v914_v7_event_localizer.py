"""Adaptive within-cycle first-passage localization for v9.14 intrinsic v7.

This module preserves the authoritative v7 constitutive phase advance and the
same step-doubling acceptance test used by ``advance_v7_cycle``.  It is used only
when an accumulated stochastic hazard threshold is known to lie inside a
physical cycle.  No new hazard law, event law, or constitutive parameter is
introduced.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Callable

from arrhenius_fracture import fatigue_v914 as base
from v914_adaptive_feedback_v6 import (
    AdaptiveFeedbackControls,
    error_passes,
    state_error,
)


LOCALIZER_ID = "v9.14_v7_adaptive_first_passage_localizer_v1"


def _adaptive_interval(
    start_state,
    loading,
    controls: AdaptiveFeedbackControls,
    p0: float,
    p1: float,
    *,
    depth: int = 0,
    advance_phase_fn: Callable[..., float] = base._advance_phase,
) -> tuple[Any, float]:
    """Advance one phase interval with the same accepted fine path as v7."""
    midpoint = 0.5 * (float(p0) + float(p1))

    coarse = copy.deepcopy(start_state)
    coarse_hazard = float(advance_phase_fn(coarse, loading, p0, p1))

    fine = copy.deepcopy(start_state)
    h0 = float(advance_phase_fn(fine, loading, p0, midpoint))
    h1 = float(advance_phase_fn(fine, loading, midpoint, p1))
    fine_hazard = h0 + h1
    if (
        coarse_hazard < 0.0
        or fine_hazard < 0.0
        or not math.isfinite(coarse_hazard)
        or not math.isfinite(fine_hazard)
    ):
        raise FloatingPointError("invalid v7 first-passage interval hazard")

    error = state_error(coarse, fine, coarse_hazard, fine_hazard, controls)
    if error_passes(error, controls):
        return fine, fine_hazard
    if int(depth) >= int(controls.max_refinement_depth):
        raise RuntimeError(
            "v7 first-passage adaptive refinement failed closed at "
            f"phase=[{p0:.16g},{p1:.16g}], depth={depth}, error={error}"
        )

    first_state, first_hazard = _adaptive_interval(
        start_state,
        loading,
        controls,
        p0,
        midpoint,
        depth=depth + 1,
        advance_phase_fn=advance_phase_fn,
    )
    second_state, second_hazard = _adaptive_interval(
        first_state,
        loading,
        controls,
        midpoint,
        p1,
        depth=depth + 1,
        advance_phase_fn=advance_phase_fn,
    )
    return second_state, first_hazard + second_hazard


def _phase_boundaries(p0: float, p1: float, nbase: int) -> list[float]:
    lo = min(max(float(p0), 0.0), 1.0)
    hi = min(max(float(p1), 0.0), 1.0)
    if hi < lo:
        raise ValueError("phase interval must be ordered")
    if hi == lo:
        return [lo, hi]
    values = [lo]
    for k in range(1, int(nbase)):
        p = k / float(nbase)
        if lo < p < hi:
            values.append(p)
    values.append(hi)
    return values


def advance_v7_phase_span(
    state,
    loading,
    controls: AdaptiveFeedbackControls,
    p0: float,
    p1: float,
    *,
    advance_phase_fn: Callable[..., float] = base._advance_phase,
) -> tuple[Any, float]:
    """Advance an arbitrary phase span while respecting the v7 base partition."""
    controls.validate()
    committed = copy.deepcopy(state)
    total = 0.0
    boundaries = _phase_boundaries(p0, p1, int(controls.base_phase_intervals))
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b <= a:
            continue
        committed, increment = _adaptive_interval(
            committed,
            loading,
            controls,
            a,
            b,
            depth=0,
            advance_phase_fn=advance_phase_fn,
        )
        total += increment
    return committed, total


def localize_v7_action_in_phase_span(
    state,
    loading,
    controls: AdaptiveFeedbackControls,
    *,
    start_phase: float,
    end_phase: float,
    required_action: float,
    phase_tolerance: float = 1.0e-13,
    max_bisections: int = 64,
    advance_phase_fn: Callable[..., float] = base._advance_phase,
) -> tuple[Any, float, float]:
    """Return the first phase where the required hazard action is reached.

    The input state is the committed state at ``start_phase``.  The returned
    state follows the same adaptive fine-path semantics used by the cycle map.
    """
    required = float(required_action)
    if not math.isfinite(required) or required <= 0.0:
        raise ValueError("required_action must be positive and finite")
    lo = float(start_phase)
    hi = float(end_phase)
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError("phase bounds must satisfy 0 <= start < end <= 1")

    whole_state, whole_hazard = advance_v7_phase_span(
        state,
        loading,
        controls,
        lo,
        hi,
        advance_phase_fn=advance_phase_fn,
    )
    if whole_hazard + 1.0e-15 * max(required, 1.0) < required:
        raise RuntimeError("phase span does not bracket the requested hazard action")

    committed = copy.deepcopy(state)
    remaining = required
    left = lo
    right = hi
    final_state = whole_state
    final_hazard = whole_hazard

    for _ in range(int(max_bisections)):
        if right - left <= float(phase_tolerance):
            final_state, final_hazard = advance_v7_phase_span(
                committed,
                loading,
                controls,
                left,
                right,
                advance_phase_fn=advance_phase_fn,
            )
            return final_state, right, required - remaining + final_hazard

        mid = 0.5 * (left + right)
        mid_state, mid_hazard = advance_v7_phase_span(
            committed,
            loading,
            controls,
            left,
            mid,
            advance_phase_fn=advance_phase_fn,
        )
        if mid_hazard >= remaining:
            right = mid
            final_state = mid_state
            final_hazard = mid_hazard
        else:
            committed = mid_state
            remaining -= mid_hazard
            left = mid

    final_state, final_hazard = advance_v7_phase_span(
        committed,
        loading,
        controls,
        left,
        right,
        advance_phase_fn=advance_phase_fn,
    )
    return final_state, right, required - remaining + final_hazard


def localize_v7_action_in_cycle(
    state,
    loading,
    controls: AdaptiveFeedbackControls,
    required_action: float,
    *,
    start_phase: float = 0.0,
    phase_tolerance: float = 1.0e-13,
) -> tuple[Any, float, float]:
    """Locate a threshold crossing between ``start_phase`` and cycle end."""
    required = float(required_action)
    if required <= 0.0:
        raise ValueError("required_action must be positive")
    committed = copy.deepcopy(state)
    accumulated = 0.0
    boundaries = _phase_boundaries(
        start_phase, 1.0, int(controls.base_phase_intervals)
    )
    for p0, p1 in zip(boundaries[:-1], boundaries[1:]):
        trial, increment = _adaptive_interval(
            committed,
            loading,
            controls,
            p0,
            p1,
            depth=0,
        )
        if accumulated + increment >= required:
            localized, phase, localized_action = localize_v7_action_in_phase_span(
                committed,
                loading,
                controls,
                start_phase=p0,
                end_phase=p1,
                required_action=required - accumulated,
                phase_tolerance=phase_tolerance,
            )
            return localized, phase, accumulated + localized_action
        committed = trial
        accumulated += increment
    raise RuntimeError("cycle does not bracket the requested hazard action")


__all__ = [
    "LOCALIZER_ID",
    "advance_v7_phase_span",
    "localize_v7_action_in_cycle",
    "localize_v7_action_in_phase_span",
]
