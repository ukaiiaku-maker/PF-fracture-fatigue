"""Projected-ligament-equivalent kernel coordinate for v10.2.28.

The direct provider parameterizes each straight prescribed crack by path length.
For an orientation campaign, however, the production crack may switch between
forward-admissible cleavage traces while the requested stopping quantity remains
projected ligament extension.  The equivalent direct-provider coordinate is

    s_equiv = Delta x_projected / cos(theta_provider),

where ``cos(theta_provider)`` is the forward component of the provider's chosen
straight {100} trace.  This module tracks the selected production direction and
integrates projected micro-advance without changing any hazard, material, or
shielding coefficient.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
from typing import Any, Callable

import numpy as np


_COORDINATE_MODE = "projected_ligament_equivalent"
_LAST_SELECTED_DIRECTION = np.array([1.0, 0.0], dtype=float)


def _normalized_direction(value: Any, fallback: Any = None) -> np.ndarray:
    try:
        direction = np.asarray(value, dtype=float).reshape(2)
    except Exception:
        direction = np.asarray(
            [1.0, 0.0] if fallback is None else fallback,
            dtype=float,
        ).reshape(2)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1.0e-30:
        direction = np.asarray(
            [1.0, 0.0] if fallback is None else fallback,
            dtype=float,
        ).reshape(2)
        norm = max(float(np.linalg.norm(direction)), 1.0e-30)
    direction = direction / norm
    if direction[0] < 0.0:
        direction = -direction
    return direction


def record_selected_direction(value: Any, fallback: Any = None) -> np.ndarray:
    """Record the primary selected crack direction for the single-front campaign."""
    global _LAST_SELECTED_DIRECTION
    _LAST_SELECTED_DIRECTION = _normalized_direction(value, fallback=fallback)
    return _LAST_SELECTED_DIRECTION.copy()


def selected_direction() -> np.ndarray:
    return _LAST_SELECTED_DIRECTION.copy()


def selected_direction_x() -> float:
    return max(float(_LAST_SELECTED_DIRECTION[0]), 0.0)


def _forward_argument(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "forward" in kwargs:
        return kwargs["forward"]
    return args[2] if len(args) >= 3 else np.array([1.0, 0.0], dtype=float)


def _tracked_selector(original: Callable[..., list[dict[str, Any]]]):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        winners = original(*args, **kwargs)
        forward = _forward_argument(args, kwargs)
        if winners:
            record_selected_direction(winners[0].get("t"), fallback=forward)
        else:
            record_selected_direction(forward, fallback=[1.0, 0.0])
        return winners

    wrapped._v10228_direction_tracker = True
    return wrapped


def install_direction_tracker() -> None:
    """Install a process-local tracker on the two production direction selectors."""
    from . import crystal

    for name in ("cleave_direction_competition", "cleavage_branch_candidates"):
        original = getattr(crystal, name)
        if bool(getattr(original, "_v10228_direction_tracker", False)):
            continue
        setattr(crystal, name, _tracked_selector(original))


@dataclass
class ProjectedLigamentEquivalentCoordinate:
    """Incrementally map actual micro-path advance to the direct-provider axis.

    ``update`` is called whenever the state-resolved kernel is queried.  Any raw
    micro-advance accumulated since the previous query belongs to the direction
    that was active at the previous query.  The newly selected direction is then
    installed for subsequent micro-advance.  This ordering matches the kinetic
    integrator, which resolves the post-advance stress before incrementing its
    cumulative micro-advance counter.
    """

    raw_anchor_m: float = 0.0
    projected_anchor_m: float = 0.0
    direction_x: float = 1.0

    def update(
        self,
        raw_path_extension_m: float,
        selected_direction_x_value: float,
        nominal_forward_cosine: float,
    ) -> float:
        raw = max(float(raw_path_extension_m), 0.0)
        nominal = float(nominal_forward_cosine)
        if not np.isfinite(nominal) or nominal <= 1.0e-12:
            raise RuntimeError(
                "projected-ligament kernel coordinate requires a positive nominal "
                "forward cosine"
            )
        current = float(selected_direction_x_value)
        if not np.isfinite(current) or current < 0.0:
            raise RuntimeError(
                "projected-ligament kernel coordinate requires a finite non-negative "
                "selected-direction forward component"
            )

        delta = raw - float(self.raw_anchor_m)
        if delta < -1.0e-15:
            raise RuntimeError(
                "moving-tip micro-advance decreased while resolving the projected-"
                "ligament kernel coordinate"
            )
        delta = max(delta, 0.0)
        projected = float(self.projected_anchor_m) + delta * float(self.direction_x)

        self.raw_anchor_m = raw
        self.projected_anchor_m = projected
        self.direction_x = current
        return projected / nominal


__all__ = [
    "_COORDINATE_MODE",
    "ProjectedLigamentEquivalentCoordinate",
    "install_direction_tracker",
    "record_selected_direction",
    "selected_direction",
    "selected_direction_x",
]
