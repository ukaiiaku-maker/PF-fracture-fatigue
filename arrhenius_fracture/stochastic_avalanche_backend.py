"""Geometry realization for the stochastic avalanche-length pilot.

The first v10.1.7.3 implementation attempted to realize one event by repeatedly
calling the sharp-wake backend with ten small requested increments. That is not
valid: the sharp-wake backend has a finite realizable increment associated with
its damage-band resolution, so nominal 0.5 micrometre requests were promoted to
approximately 2 micrometres and a nominal 5 micrometre event became 20
micrometres.

This wrapper performs one geometry commit per sampled event and verifies that the
backend realized the requested length. It deliberately preserves the
``sharp_wake`` semantic backend identity because the 2-D driver uses that name to
enable tip-following remeshing. The same driver also treats the ``p1`` array it
passes to a sharp-wake backend as the authoritative new front position. A
variable-length wrapper must therefore update that array transactionally to the
actual event endpoint; otherwise the damage wake, kinetic moving frame, and front
position follow different crack lengths.

Extra pilot diagnostics are written through an explicit registry hook rather
than by changing the backend name.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .crack_backend import CrackAdvanceResult
from .stochastic_avalanche_tip import pop_pending_geometry_event


_LAST_AVALANCHE_BACKEND = None


class AvalancheSubsegmentBackend:
    """Wrap sharp-wake with one checked geometry commit per variable event."""

    # This is a behavioral identity in sharp_front.py, not merely a display label.
    # Keeping it equal to sharp_wake preserves tip-following remeshing and the
    # sharp-wake endpoint convention.
    name = "sharp_wake"
    diagnostic_name = "stochastic_avalanche_event"

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

    def _rollback_base_log(self) -> None:
        """Undo the base backend's diagnostic append after a vetoed transaction."""
        log = getattr(self.base_backend, "advance_log", None)
        if isinstance(log, list) and log:
            log.pop()

    @staticmethod
    def _mutable_driver_endpoint(value: Any) -> np.ndarray | None:
        """Return the driver's writable endpoint array, or ``None`` if unsupported."""
        if not isinstance(value, np.ndarray):
            return None
        if value.shape != (2,) or not value.flags.writeable:
            return None
        return value

    def advance(self, **kwargs) -> CrackAdvanceResult:
        mesh0 = kwargs["mesh"]
        boundary0 = kwargs["boundary"]
        damage0 = np.asarray(kwargs["damage"], dtype=float)
        displacement0 = np.asarray(kwargs["displacement"], dtype=float)
        p0 = np.asarray(kwargs["p0"], dtype=float)

        # ``sharp_front._advance_polyline`` retains this exact ndarray and, for a
        # backend named sharp_wake, uses it as the authoritative front endpoint
        # after ``advance`` returns. Keep the reference and update it only after a
        # successful checked geometry transaction.
        driver_endpoint = self._mutable_driver_endpoint(kwargs["p1"])
        if driver_endpoint is None:
            return CrackAdvanceResult(
                mesh0, boundary0, damage0, displacement0, 0.0, False,
                reason="driver_endpoint_not_mutable",
            )
        p1_requested = np.asarray(driver_endpoint, dtype=float).copy()
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

        # One geometry call is deliberate. Repeated calls without a FEM solve in
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
            # Return the original transaction state. The caller will restore the
            # completed hazard event rather than silently accepting a different
            # crack-growth reward.
            self._rollback_base_log()
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
        driver_endpoint[...] = endpoint
        endpoint_error = float(np.linalg.norm(np.asarray(driver_endpoint) - endpoint))
        if endpoint_error > max(self.absolute_length_tolerance_m, 1.0e-15):
            self._rollback_base_log()
            return CrackAdvanceResult(
                mesh0, boundary0, damage0, displacement0, 0.0, False,
                angle_error_deg=float(result.angle_error_deg),
                selected_edge_length=float(result.selected_edge_length),
                reason=f"driver_endpoint_sync_failed:error={endpoint_error:.9e}",
            )

        row = {
            "event_index": len(self.advance_log),
            "front_id": int(kwargs.get("front_id", 0)),
            "x0": float(p0[0]),
            "y0": float(p0[1]),
            "x1": float(endpoint[0]),
            "y1": float(endpoint[1]),
            "driver_requested_x1": float(p1_requested[0]),
            "driver_requested_y1": float(p1_requested[1]),
            "driver_endpoint_synchronized": True,
            "driver_endpoint_sync_error_m": endpoint_error,
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
            "backend_semantic_identity": "sharp_wake",
            "tip_following_remeshing_preserved": True,
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


def write_last_avalanche_backend_diagnostics(out_dir: str) -> None:
    """Write diagnostics for the backend built in the current one-case process."""
    if _LAST_AVALANCHE_BACKEND is None:
        raise RuntimeError("no stochastic avalanche backend was constructed")
    _LAST_AVALANCHE_BACKEND.write_diagnostics(out_dir)


def build_avalanche_backend(
    args,
    geom,
    original_builder: Callable,
    default_subsegment_fraction: float = 0.1,
):
    global _LAST_AVALANCHE_BACKEND
    base = original_builder(args, geom)
    wrapped = AvalancheSubsegmentBackend(
        base,
        default_subsegment_fraction=default_subsegment_fraction,
    )
    _LAST_AVALANCHE_BACKEND = wrapped
    return wrapped


__all__ = [
    "AvalancheSubsegmentBackend",
    "build_avalanche_backend",
    "write_last_avalanche_backend_diagnostics",
]
