"""Provenance-checked wrapper around the v10.2.12 station projection.

The PF ``r_eff`` variable is an analytical local-tip stress/blunting state, not a
geometric coordinate of the fixed-crack FEM problem. It remains in the physical
snapshot audit, but the shielding atlas compatibility coordinate is held at one.
The actual interpolation axes are opening fraction and crack extension.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from .interaction_integral_v1029 import MODEL_ID as REQUIRED_INTERACTION_SCHEMA
from .spatial_station_projection_v10212 import (
    MODEL_ID as PROJECTION_MODEL_ID,
    STATION_SCHEMA,
    expand_station_response_files as _expand_station_response_files,
)

MODEL_ID = "v10.2.12_checked_measured_station_to_mpz_grid_projection"
KERNEL_RADIUS_COMPATIBILITY_COORDINATE = 1.0


def _interaction_schemas(paths: Iterable[str | Path]) -> set[str]:
    schemas: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if "interaction_integral_schema" not in (reader.fieldnames or []):
                raise ValueError(
                    f"{path} is missing interaction_integral_schema provenance"
                )
            for row in reader:
                schema = str(row.get("interaction_integral_schema", "")).strip()
                if not schema:
                    raise ValueError(
                        f"{path} contains an empty interaction-integral schema"
                    )
                schemas.add(schema)
    return schemas


def _collapse_radius_axis(expanded: list[dict[str, Any]]) -> dict[str, Any]:
    observed = sorted({float(row["r_eff_over_r0"]) for row in expanded})
    state_coordinates: dict[str, tuple[float, float]] = {}
    coordinate_owners: dict[tuple[float, float], str] = {}
    for row in expanded:
        state_id = str(row["state_id"])
        physical_coordinate = (
            float(row["opening_strength_fraction"]),
            float(row["crack_extension_m"]),
        )
        old = state_coordinates.setdefault(state_id, physical_coordinate)
        if old != physical_coordinate:
            raise ValueError(f"state {state_id} has inconsistent physical coordinates")
        row["r_eff_over_r0"] = KERNEL_RADIUS_COMPATIBILITY_COORDINATE
    for state_id, coordinate in state_coordinates.items():
        owner = coordinate_owners.setdefault(coordinate, state_id)
        if owner != state_id:
            raise ValueError(
                "two physical snapshots collapse to the same opening/extension kernel state: "
                f"{owner!r} and {state_id!r} at {coordinate}; keep one reviewed state"
            )
    return {
        "kernel_radius_axis_policy": "disabled_constant_compatibility",
        "kernel_radius_compatibility_coordinate": KERNEL_RADIUS_COMPATIBILITY_COORDINATE,
        "observed_analytical_r_eff_over_r0_values": observed,
        "observed_analytical_r_eff_over_r0_min": min(observed),
        "observed_analytical_r_eff_over_r0_max": max(observed),
        "active_physical_kernel_axes": [
            "opening_strength_fraction",
            "crack_extension_m",
        ],
        "analytical_r_eff_used_for_spatial_interpolation": False,
        "finite_radius_fem_geometry_claimed": False,
    }


def expand_station_response_files(
    paths: Iterable[str | Path],
    **kwargs: Any,
):
    resolved_paths = [Path(path) for path in paths]
    schemas = _interaction_schemas(resolved_paths)
    if schemas != {REQUIRED_INTERACTION_SCHEMA}:
        raise ValueError(
            "all measured station responses must use exactly "
            f"{REQUIRED_INTERACTION_SCHEMA}; found {sorted(schemas)}"
        )
    expanded, physical_inputs, report = _expand_station_response_files(
        resolved_paths, **kwargs
    )
    radius_audit = _collapse_radius_axis(expanded)
    projected_schemas = {
        str(row.get("interaction_integral_schema", "")).strip()
        for row in expanded
    }
    if projected_schemas != schemas:
        raise RuntimeError(
            "spatial projection changed interaction-integral provenance: "
            f"measured={sorted(schemas)}, projected={sorted(projected_schemas)}"
        )
    report = {
        **report,
        **radius_audit,
        "schema": MODEL_ID,
        "underlying_projection_schema": PROJECTION_MODEL_ID,
        "interaction_integral_schema": REQUIRED_INTERACTION_SCHEMA,
        "single_interaction_integral_schema_required": True,
        "projected_schema_matches_measured_schema": True,
    }
    return expanded, physical_inputs, report


__all__ = [
    "MODEL_ID",
    "STATION_SCHEMA",
    "REQUIRED_INTERACTION_SCHEMA",
    "KERNEL_RADIUS_COMPATIBILITY_COORDINATE",
    "expand_station_response_files",
]
