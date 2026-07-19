"""Mechanics-derived activation-to-line and source-capacity normalization.

The signed population carried by the shared fracture/fatigue engine is line
content, whereas ``source_sites_per_system`` counts independent nucleation
opportunities.  This module supplies the missing dimensional bridge without
fitting toughness, fatigue life, or a shielding attenuation factor.

One accepted emission activation produces the slip packet serialized by the
production kinetic-tip configuration.  The associated signed line content is
therefore the packet displacement divided by the Burgers magnitude.  Source
capacity is bounded independently from the process-zone source length and a
reviewed admissible line/source spacing interval.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

MODEL_ID = "v10.2.12_mechanics_derived_activation_line_source_normalization"


@dataclass(frozen=True)
class SourceGeometryAssumptions:
    minimum_spacing_b: float = 10.0
    maximum_spacing_b: float = 100.0
    source_region_length_m: float | None = None
    plane_strain_line_convention: str = "one_through_thickness_line_per_in_plane_source_position"

    def validate(self) -> "SourceGeometryAssumptions":
        minimum = float(self.minimum_spacing_b)
        maximum = float(self.maximum_spacing_b)
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError("source spacing bounds must be finite")
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError("require 0 < minimum_spacing_b <= maximum_spacing_b")
        if self.source_region_length_m is not None:
            length = float(self.source_region_length_m)
            if not math.isfinite(length) or length <= 0.0:
                raise ValueError("source_region_length_m must be positive and finite")
        convention = str(self.plane_strain_line_convention).strip()
        if not convention:
            raise ValueError("plane-strain line convention must be explicit")
        return self


def _finite_positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def derive_mechanical_normalization(
    engine_payload: dict[str, Any],
    *,
    assumptions: SourceGeometryAssumptions | None = None,
) -> dict[str, Any]:
    """Build a normalization artifact from serialized production geometry.

    ``packet_length_m / b_m`` is the line content represented by one accepted
    activation.  Capacity bounds count admissible in-plane source positions over
    the process-zone source length.  They deliberately do not multiply by an
    arbitrary specimen thickness: in plane strain one line represents a
    through-thickness dislocation line.
    """
    assumptions = (assumptions or SourceGeometryAssumptions()).validate()
    if not isinstance(engine_payload, dict):
        raise ValueError("engine payload must be a dictionary")
    front = dict(engine_payload.get("front_config", {}))
    mpz = dict(engine_payload.get("mpz_config", {}))
    tip = dict(engine_payload.get("tip_config", {}))
    b_m = _finite_positive(engine_payload.get("b_m"), "b_m")
    packet_length_m = _finite_positive(
        tip.get("packet_length_m"), "tip_config.packet_length_m"
    )
    n_systems = max(int(mpz.get("n_systems", 2)), 1)
    L_pz = _finite_positive(front.get("L_pz"), "front_config.L_pz")
    source_length = (
        L_pz
        if assumptions.source_region_length_m is None
        else _finite_positive(
            assumptions.source_region_length_m, "source_region_length_m"
        )
    )

    minimum_spacing_m = float(assumptions.minimum_spacing_b) * b_m
    maximum_spacing_m = float(assumptions.maximum_spacing_b) * b_m
    lower = max(int(math.floor(source_length / maximum_spacing_m)), 1)
    upper = max(int(math.ceil(source_length / minimum_spacing_m)), lower)
    activation_to_line = packet_length_m / b_m

    historical = engine_payload.get("material_manifest", {})
    old_sites = None
    if isinstance(historical, dict):
        old_sites = historical.get("source_sites_per_system")
    historical_outside = None
    if old_sites is not None:
        try:
            old_value = float(old_sites)
            historical_outside = bool(old_value < lower or old_value > upper)
        except (TypeError, ValueError):
            historical_outside = None

    return {
        "schema": MODEL_ID,
        "normalization_source": "front_thickness_source_geometry",
        "mechanical_line_content_source": "kinetic_packet_displacement_divided_by_burgers_magnitude",
        "activation_to_line_content_by_system": [
            float(activation_to_line) for _ in range(n_systems)
        ],
        "source_capacity_bounds_per_system": [
            [float(lower), float(upper)] for _ in range(n_systems)
        ],
        "n_systems": n_systems,
        "burgers_m": b_m,
        "kinetic_packet_length_m": packet_length_m,
        "source_region_length_m": source_length,
        "minimum_source_spacing_b": float(assumptions.minimum_spacing_b),
        "maximum_source_spacing_b": float(assumptions.maximum_spacing_b),
        "minimum_source_spacing_m": minimum_spacing_m,
        "maximum_source_spacing_m": maximum_spacing_m,
        "plane_strain_line_convention": assumptions.plane_strain_line_convention,
        "capacity_counts_in_plane_source_positions_only": True,
        "out_of_plane_thickness_multiplier_applied": False,
        "historical_source_sites_per_system": old_sites,
        "historical_source_sites_outside_mechanical_bounds": historical_outside,
        "source_sites_are_nucleation_opportunities": True,
        "emitted_population_units": "signed_dislocation_line_content",
        "fitted_to_toughness_or_fatigue": False,
        "shielding_attenuation_factor_fitted": False,
        "constitutive_K_shield_cap_present": False,
    }


def derive_from_json(
    engine_config: str | Path,
    *,
    assumptions: SourceGeometryAssumptions | None = None,
) -> dict[str, Any]:
    path = Path(engine_config)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    return derive_mechanical_normalization(payload, assumptions=assumptions)


__all__ = [
    "MODEL_ID",
    "SourceGeometryAssumptions",
    "derive_mechanical_normalization",
    "derive_from_json",
]
