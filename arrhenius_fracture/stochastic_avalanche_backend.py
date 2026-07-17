"""Geometry realization for the stochastic avalanche-length pilot.

This wrapper is intentionally restricted to the sharp-wake PF backend.  It
replaces one requested fixed geometry increment by the event length supplied by
the stochastic renewal-reward engine and realizes that distance as equal
subsegments.  The subsegments update topology/damage conservatively but do not
re-equilibrate the FEM field between subsegments; the next outer solve sees the
completed event geometry.
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
    """Wrap the sharp-wake backend with variable event lengths and subsegments."""

    name = "stochastic_avalanche_segmented"

    def __init__(self, base_backend, default_subsegment_fraction: float = 0.1):
        if str(getattr(base_backend, "name", "")) != "sharp_wake":
            raise ValueError(
                "the v10.1.7.3 avalanche pilot currently supports only sharp_wake"
            )
        self.base_backend = base_backend
        self.cohesive_network = getattr(base_backend, "cohesive_network", None)
        self.default_subsegment_fraction = float(default_subsegment_fraction)
        if not (0.0 < self.default_subsegment_fraction <= 1.0):
            raise ValueError("default_subsegment_fraction must lie in (0, 1]")
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
        requested_length = float(np.linalg.norm(p1_requested - p0))
        total_length = (
            float(descriptor["event_advance_m"])
            if descriptor is not None
            else requested_length
        )
        if not math.isfinite(total_length) or total_length <= 0.0:
            return CrackAdvanceResult(
                mesh0, boundary0, damage0, displacement0, 0.0, False,
                reason="nonpositive_event_length",
            )
        fraction = (
            float(descriptor.get("geometry_subsegment_fraction", self.default_subsegment_fraction))
            if descriptor is not None
            else self.default_subsegment_fraction
        )
        fraction = min(max(fraction, 1.0e-6), 1.0)
        n_segments = max(int(math.ceil(1.0 / fraction)), 1)
        segment_length = total_length / n_segments

        mesh = mesh0
        boundary = boundary0
        damage = damage0.copy()
        displacement = displacement0.copy()
        current = p0.copy()
        moved_total = 0.0
        last_result = None

        for index in range(n_segments):
            remaining = max(total_length - moved_total, 0.0)
            this_length = remaining if index == n_segments - 1 else min(segment_length, remaining)
            if this_length <= 0.0:
                break
            target = current + this_length * direction
            call = dict(kwargs)
            call.update({
                "mesh": mesh,
                "boundary": boundary,
                "damage": damage,
                "displacement": displacement,
                "p0": current,
                "p1": target,
                "direction": direction,
            })
            result = self.base_backend.advance(**call)
            if not result.inserted or result.moved <= 0.0:
                return CrackAdvanceResult(
                    mesh0, boundary0, damage0, displacement0, 0.0, False,
                    angle_error_deg=float(result.angle_error_deg),
                    selected_edge_length=float(result.selected_edge_length),
                    reason=f"subsegment_{index}:{result.reason}",
                )
            mesh = result.mesh
            boundary = result.boundary
            damage = result.damage
            displacement = result.displacement
            moved = float(result.moved)
            moved_total += moved
            current = current + moved * direction
            last_result = result

        if last_result is None or moved_total <= 0.0:
            return CrackAdvanceResult(
                mesh0, boundary0, damage0, displacement0, 0.0, False,
                reason="no_subsegment_committed",
            )

        row = {
            "event_index": len(self.advance_log),
            "front_id": int(kwargs.get("front_id", 0)),
            "x0": float(p0[0]),
            "y0": float(p0[1]),
            "x1": float(current[0]),
            "y1": float(current[1]),
            "requested_fixed_length_m": requested_length,
            "event_advance_m": moved_total,
            "event_length_factor": float(
                descriptor.get("event_length_factor", moved_total / max(requested_length, 1.0e-300))
                if descriptor is not None else 1.0
            ),
            "threshold_action": float(
                descriptor.get("threshold_action", 1.0) if descriptor is not None else 1.0
            ),
            "hazard_seed": int(
                descriptor.get("hazard_seed", 0) if descriptor is not None else 0
            ),
            "hazard_event_index": int(
                descriptor.get("hazard_event_index", len(self.advance_log))
                if descriptor is not None else len(self.advance_log)
            ),
            "n_subsegments": n_segments,
            "subsegment_fraction": fraction,
            "mechanics_re_equilibrated_between_subsegments": False,
        }
        self.advance_log.append(row)

        return CrackAdvanceResult(
            mesh=mesh,
            boundary=boundary,
            damage=damage,
            displacement=displacement,
            moved=moved_total,
            inserted=True,
            angle_error_deg=float(last_result.angle_error_deg),
            selected_edge_length=moved_total,
            reason="ok",
            elem_parent_map=None,
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
