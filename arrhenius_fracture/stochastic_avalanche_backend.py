"""Geometry realization for the stochastic avalanche-length pilot.

The stochastic pilot uses one checked sharp-wake geometry commit per completed
renewal. Repeated calls without a FEM solve are not physical subincrements and
can multiply the realized crack length when a backend has a finite geometry
resolution.

A crucial regression requirement is that deterministic/fixed mode reproduce the
unwrapped sharp-wake transaction exactly. In that mode this wrapper now passes
the original ``p0``, ``p1`` and ``direction`` objects directly to the base
backend and does not normalize or reconstruct the endpoint. Stochastic variable
lengths still replace the requested endpoint transactionally and synchronize the
driver-owned endpoint after the checked commit.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .crack_backend import CrackAdvanceResult
from .stochastic_avalanche_tip import pop_pending_geometry_event


_LAST_AVALANCHE_BACKEND = None


def _deterministic_fixed_mode() -> bool:
    """Return whether geometry must be an exact pass-through regression control."""
    hazard_mode = os.environ.get("CLEAVAGE_HAZARD_MODE", "deterministic").strip().lower()
    length_mode = os.environ.get("CLEAVAGE_EVENT_LENGTH_MODE", "fixed").strip().lower()
    return hazard_mode == "deterministic" and length_mode == "fixed"


class AvalancheSubsegmentBackend:
    """Wrap sharp-wake with one checked geometry commit per variable event."""

    # This is a behavioral identity in sharp_front.py, not merely a display label.
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

    def _failed(
        self,
        mesh,
        boundary,
        damage,
        displacement,
        reason: str,
        result: CrackAdvanceResult | None = None,
    ) -> CrackAdvanceResult:
        return CrackAdvanceResult(
            mesh,
            boundary,
            damage,
            displacement,
            0.0,
            False,
            angle_error_deg=float(getattr(result, "angle_error_deg", 0.0)),
            selected_edge_length=float(getattr(result, "selected_edge_length", 0.0)),
            reason=reason,
        )

    def advance(self, **kwargs) -> CrackAdvanceResult:
        mesh0 = kwargs["mesh"]
        boundary0 = kwargs["boundary"]
        damage0 = np.asarray(kwargs["damage"], dtype=float)
        displacement0 = np.asarray(kwargs["displacement"], dtype=float)
        p0 = np.asarray(kwargs["p0"], dtype=float)

        # sharp_front retains this exact ndarray and uses it as the authoritative
        # front endpoint for a backend named sharp_wake.
        driver_endpoint = self._mutable_driver_endpoint(kwargs["p1"])
        if driver_endpoint is None:
            return self._failed(
                mesh0, boundary0, damage0, displacement0,
                "driver_endpoint_not_mutable",
            )

        p1_requested = np.asarray(driver_endpoint, dtype=float).copy()
        fixed_vector = p1_requested - p0
        fixed_requested_length = float(np.linalg.norm(fixed_vector))
        if not math.isfinite(fixed_requested_length) or fixed_requested_length <= 0.0:
            return self._failed(
                mesh0, boundary0, damage0, displacement0,
                "nonpositive_fixed_length",
            )

        raw_direction = np.asarray(
            kwargs.get("direction", fixed_vector), dtype=float
        ).copy()
        direction_norm = float(np.linalg.norm(raw_direction))
        if not math.isfinite(direction_norm) or direction_norm <= 0.0:
            return self._failed(
                mesh0, boundary0, damage0, displacement0,
                "zero_direction",
            )

        descriptor = pop_pending_geometry_event()
        kinetic_event_length = (
            float(descriptor["event_advance_m"])
            if descriptor is not None
            else fixed_requested_length
        )
        if not math.isfinite(kinetic_event_length) or kinetic_event_length <= 0.0:
            return self._failed(
                mesh0, boundary0, damage0, displacement0,
                "nonpositive_event_length",
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

        # The deterministic control must be bitwise-equivalent at the geometry
        # transaction boundary. Do not normalize direction, allocate a replacement
        # endpoint, or write back a reconstructed endpoint in this mode.
        deterministic_passthrough = bool(descriptor is not None and _deterministic_fixed_mode())
        if deterministic_passthrough:
            event_requested_length = fixed_requested_length
            result = self.base_backend.advance(**kwargs)
            endpoint = np.asarray(driver_endpoint, dtype=float).copy()
            transaction_mode = "exact_driver_passthrough"
        else:
            # One geometry call is deliberate. Repeated calls without a FEM solve in
            # between are not physical subincrements and can multiply crack length.
            direction = raw_direction / direction_norm
            event_requested_length = kinetic_event_length
            target = p0 + event_requested_length * direction
            call = dict(kwargs)
            call.update({"p0": p0, "p1": target, "direction": direction})
            result = self.base_backend.advance(**call)
            endpoint = p0 + float(result.moved) * direction
            transaction_mode = "variable_length_endpoint_replacement"

        if not result.inserted or result.moved <= 0.0:
            return self._failed(
                mesh0, boundary0, damage0, displacement0,
                f"event_geometry:{result.reason}", result,
            )

        moved = float(result.moved)
        length_error = moved - event_requested_length
        tolerance = max(
            self.absolute_length_tolerance_m,
            self.relative_length_tolerance * event_requested_length,
        )
        if abs(length_error) > tolerance:
            self._rollback_base_log()
            return self._failed(
                mesh0, boundary0, damage0, displacement0,
                "event_length_mismatch:"
                f"requested={event_requested_length:.9e},realized={moved:.9e}",
                result,
            )

        if not deterministic_passthrough:
            driver_endpoint[...] = endpoint

        endpoint_error = float(np.linalg.norm(np.asarray(driver_endpoint) - endpoint))
        if endpoint_error > max(self.absolute_length_tolerance_m, 1.0e-15):
            self._rollback_base_log()
            return self._failed(
                mesh0, boundary0, damage0, displacement0,
                f"driver_endpoint_sync_failed:error={endpoint_error:.9e}",
                result,
            )

        endpoint_adjustment = float(np.linalg.norm(endpoint - p1_requested))
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
            "driver_direction_norm": direction_norm,
            "deterministic_geometry_passthrough": deterministic_passthrough,
            "requested_endpoint_preserved": bool(endpoint_adjustment <= 1.0e-15),
            "endpoint_adjustment_m": endpoint_adjustment,
            "geometry_transaction_mode": transaction_mode,
            "requested_fixed_length_m": fixed_requested_length,
            "kinetic_event_advance_m": kinetic_event_length,
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
