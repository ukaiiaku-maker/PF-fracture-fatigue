"""Geometry realization for the stochastic avalanche-length pilot.

The first v10.1.7.3 implementation attempted to realize one event by repeatedly
calling the sharp-wake backend with ten small requested increments.  That is not
valid: the sharp-wake backend has a finite realizable increment associated with
its damage-band resolution, so nominal 0.5 micrometre requests were promoted to
approximately 2 micrometres and a nominal 5 micrometre event became 20
micrometres.

This wrapper now performs one geometry commit per sampled event and verifies that
the backend realized the requested length.  The event-length stochasticity is
therefore tested without a hidden geometry multiplication.  True 10-percent
subincrements require a driver-level FEM re-equilibration loop and are explicitly
not claimed here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .crack_backend import CrackAdvanceResult
from .stochastic_avalanche_tip import pop_pending_geometry_event


class AvalancheSubsegmentBackend:
    """Wrap sharp-wake with one checked geometry commit per variable event."""

    name = "stochastic_avalanche_event"

    def __init__(
        self,
        base_backend,
        default_subsegment_fraction: float = 0.1,
        relative_length_tolerance: float = 0.05,
        absolute_length_tolerance_m: float = 1.0e-9,
    ):
        if str(getattr(base_backend, "name", "")) != "sharp_wake":
            raise ValueError(
                "the v10.1.7.3 avalanche pilot currently supports only sharp_wake"
            )
        self.base_backend = base_backend
        self.cohesive_network = getattr(base_backend, "cohesive_network", None)
        self.default_subsegment_fraction = float(default_subsegment_fraction)
        if not (0.0 < self.default_subsegment_fraction <= 1.0):
            raise ValueError("default_subsegment_fraction must lie in (0, 1]")
        self.relative_length_tolerance = max(float(relative_length_tolerance), 0.0)
        self.absolute_length_tolerance_m = max(float(absolute_length_tolerance_m), 0.0)
        self.advance_log: list[dict[str, Any]] = []

    def __getattr__(self, name: str):
        return getattr(self.base_backend, name)

    def advance(self, **kwargs) -> CrackAdvanceResult:
        mesh0 = kwargs["mesh"]
        boundary0 = kwargs["boundary"]
        damage0 = np.asarray(kwargs["damage"], dtype=float)
        displacement0 = np.asarray(kwargs["displacement"], dtype=float)
        p0 = np.asarray(kwargs["p0"], dtype=float)
        p1_requested = np.asarray(kwargs["p1"], dtype=float)
        direction = np.asarray(kwargs.get("direction", p1_requested - p0), dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            return CrackAdvanceResult(
                mesh0, boundary0, damage0, displacement0, 0.0, False,
                reason="zero_direction",
            )
        direction /= norm

        descriptor = pop_pending_geometry_event()
        fixed_requested_length = float(np.linalg.norm(p1_requested - p0))
        event_requested_length = (
            float(descriptor["event_advance_m"])
            if descriptor is not None
            else fixed_requested_length
        )
        if not math.isfinite(event_requested_length) or event_requested_length <= 0.0:
            return CrackAdvanceResult(
                mesh0, boundary0, damage0, displacement0, 0.0, False,
                reason="nonpositive_event_length",
            )

        fraction = (
            float(descriptor.get(
                "geometry_subsegment_fraction", self.default_subsegment_fraction
            ))
            if descriptor is not None
            else self.default_subsegment_fraction
        )
        fraction = min(max(fraction, 1.0e-6), 1.0)
        requested_subsegments = max(int(math.ceil(1.0 / fraction)), 1)

        # One geometry call is deliberate.  Repeated calls without a FEM solve in
        # between are not physical subincrements and can multiply the crack length
        # when the backend has a finite minimum realizable increment.
        target = p0 + event_requested_length * direction
        call = dict(kwargs)
        call.update({"p0": p0, "p1": target, "direction": direction})
        result = self.base_backend.advance(**call)
        if not result.inserted or result.moved <= 0.0:
            return CrackAdvanceResult(
                mesh0, boundary0, damage0, displacement0, 0.0, False,
                angle_error_deg=float(result.angle_error_deg),
                selected_edge_length=float(result.selected_edge_length),
                reason=f"event_geometry:{result.reason}",
            )

        moved = float(result.moved)
        length_error = moved - event_requested_length
        tolerance = max(
            self.absolute_length_tolerance_m,
            self.relative_length_tolerance * event_requested_length,
        )
        if abs(length_error) > tolerance:
            # Return the original transaction state.  The caller will restore the
            # completed hazard event rather than silently accepting a different
            # crack-growth reward.
            return CrackAdvanceResult(
                mesh0, boundary0, damage0, displacement0, 0.0, False,
                angle_error_deg=float(result.angle_error_deg),
                selected_edge_length=float(result.selected_edge_length),
                reason=(
                    "event_length_mismatch:"
                    f"requested={event_requested_length:.9e},realized={moved:.9e}"
                ),
            )

        endpoint = p0 + moved * direction
        row = {
            "event_index": len(self.advance_log),
            "front_id": int(kwargs.get("front_id", 0)),
            "x0": float(p0[0]),
            "y0": float(p0[1]),
            "x1": float(endpoint[0]),
            "y1": float(endpoint[1]),
            "requested_fixed_length_m": fixed_requested_length,
            "requested_event_advance_m": event_requested_length,
            "event_advance_m": moved,
            "event_length_error_m": length_error,
            "event_length_relative_error": (
                length_error / max(event_requested_length, 1.0e-300)
            ),
            "event_length_factor": float(
                descriptor.get(
                    "event_length_factor",
                    moved / max(fixed_requested_length, 1.0e-300),
                )
                if descriptor is not None else 1.0
            ),
            "threshold_action": float(
                descriptor.get("threshold_action", 1.0)
                if descriptor is not None else 1.0
            ),
            "hazard_seed": int(
                descriptor.get("hazard_seed", 0)
                if descriptor is not None else 0
            ),
            "hazard_event_index": int(
                descriptor.get("hazard_event_index", len(self.advance_log))
                if descriptor is not None else len(self.advance_log)
            ),
            "requested_subsegment_fraction": fraction,
            "requested_subsegments": requested_subsegments,
            "realized_geometry_commits": 1,
            "mechanics_re_equilibrated_between_subsegments": False,
            "geometry_realization": "single_checked_outer_commit",
        }
        self.advance_log.append(row)

        return CrackAdvanceResult(
            mesh=result.mesh,
            boundary=result.boundary,
            damage=result.damage,
            displacement=result.displacement,
            moved=moved,
            inserted=True,
            angle_error_deg=float(result.angle_error_deg),
            selected_edge_length=float(result.selected_edge_length),
            reason="ok",
            elem_parent_map=result.elem_parent_map,
        )

    def write_diagnostics(self, out_dir: str) -> None:
        try:
            self.base_backend.write_diagnostics(out_dir)
        except Exception:
            pass
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "stochastic_avalanche_geometry_events.json").write_text(
            json.dumps(self.advance_log, indent=2)
        )


def build_avalanche_backend(
    args,
    geom,
    original_builder: Callable,
    default_subsegment_fraction: float = 0.1,
):
    base = original_builder(args, geom)
    return AvalancheSubsegmentBackend(
        base,
        default_subsegment_fraction=default_subsegment_fraction,
    )


__all__ = [
    "AvalancheSubsegmentBackend",
    "build_avalanche_backend",
]
