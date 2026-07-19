"""Provenance-checked wrapper around the v10.2.12 station projection."""
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
    "expand_station_response_files",
]
