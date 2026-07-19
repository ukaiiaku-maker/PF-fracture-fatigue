"""Provenance-checked wrapper around measured-station projection.

Archived v10.2.12 atlases retain the reviewed v10.2.9 isotropic interaction
schema. New v10.2.14 production responses use intrinsic stiffness isotropy. A
build may use either reviewed schema, but mixing schemas within one response set
is prohibited.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from .interaction_integral_v1029 import MODEL_ID as LEGACY_INTERACTION_SCHEMA
from .interaction_integral_v10214 import MODEL_ID as INTRINSIC_INTERACTION_SCHEMA
from .spatial_station_projection_v10212 import (
    MODEL_ID as PROJECTION_MODEL_ID,
    STATION_SCHEMA,
    expand_station_response_files as _expand_station_response_files,
)

MODEL_ID = "v10.2.14_checked_measured_station_to_mpz_grid_projection"
REQUIRED_INTERACTION_SCHEMA = INTRINSIC_INTERACTION_SCHEMA
ACCEPTED_INTERACTION_SCHEMAS = {
    LEGACY_INTERACTION_SCHEMA,
    INTRINSIC_INTERACTION_SCHEMA,
}
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
                "two physical snapshots collapse to the same opening/extension "
                f"kernel state: {owner!r} and {state_id!r} at {coordinate}; "
                "keep one reviewed state"
            )
    return {
        "kernel_radius_axis_policy": "disabled_constant_compatibility",
        "kernel_radius_compatibility_coordinate": (
            KERNEL_RADIUS_COMPATIBILITY_COORDINATE
        ),
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
    if len(schemas) != 1 or not schemas.issubset(ACCEPTED_INTERACTION_SCHEMAS):
        raise ValueError(
            "measured station responses must use exactly one uniform reviewed "
            f"interaction schema from {sorted(ACCEPTED_INTERACTION_SCHEMAS)}; "
            f"found {sorted(schemas)}"
        )
    selected_schema = next(iter(schemas))
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
        "interaction_integral_schema": selected_schema,
        "accepted_interaction_integral_schemas": sorted(
            ACCEPTED_INTERACTION_SCHEMAS
        ),
        "single_interaction_integral_schema_required": True,
        "projected_schema_matches_measured_schema": True,
        "intrinsic_stiffness_isotropy_required_for_v10214": True,
    }
    return expanded, physical_inputs, report


__all__ = [
    "MODEL_ID",
    "STATION_SCHEMA",
    "LEGACY_INTERACTION_SCHEMA",
    "INTRINSIC_INTERACTION_SCHEMA",
    "REQUIRED_INTERACTION_SCHEMA",
    "ACCEPTED_INTERACTION_SCHEMAS",
    "KERNEL_RADIUS_COMPATIBILITY_COORDINATE",
    "expand_station_response_files",
]
