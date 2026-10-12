"""Candidate-specific mechanical-drive normalization contracts.

This module does not choose the unresolved physical meaning of a complete
ligament event.  It makes the alternatives explicit and prevents a raw finite
energy, a local derivative, and cavity free span from being fed interchangeably
to one rate law.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class CandidateClass(str, Enum):
    ORDINARY_INCREMENT = "ordinary_increment"
    FINITE_SOLID_LIGAMENT = "finite_solid_ligament"
    DOWNSTREAM_INITIATION = "downstream_initiation"


class LigamentKineticInterpretation(str, Enum):
    LOCAL_INCREMENTAL_FRONT = "local_incremental_front"
    COLLECTIVE_FINITE_LIGAMENT = "collective_finite_ligament"


@dataclass(frozen=True)
class FiniteSeparationOperation:
    candidate_class: CandidateClass
    released_energy_J_per_m_thickness: float
    new_solid_length_m: float
    cavity_free_span_m: float = 0.0
    total_geometric_span_m: float | None = None

    def __post_init__(self) -> None:
        for name in ("released_energy_J_per_m_thickness", "new_solid_length_m", "cavity_free_span_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.new_solid_length_m <= 0.0:
            raise ValueError("new_solid_length_m must be positive")
        if self.total_geometric_span_m is not None:
            span = float(self.total_geometric_span_m)
            if not math.isfinite(span) or span <= 0.0:
                raise ValueError("total_geometric_span_m must be finite and positive")
            if not math.isclose(
                self.new_solid_length_m + self.cavity_free_span_m, span,
                rel_tol=5e-12, abs_tol=1e-15,
            ):
                raise ValueError("solid length plus cavity free span must equal geometric span")

    @property
    def energy_release_J_m2(self) -> float:
        """Finite energy release divided only by newly separated solid area."""
        return self.released_energy_J_per_m_thickness / self.new_solid_length_m

    def rate_contract(
        self,
        *,
        interpretation: LigamentKineticInterpretation,
        activation_area_m2: float,
        physical_site_area_m2: float,
    ) -> dict[str, float | str]:
        A = float(activation_area_m2)
        site = float(physical_site_area_m2)
        if not math.isfinite(A) or A <= 0.0 or not math.isfinite(site) or site <= 0.0:
            raise ValueError("activation and site areas must be finite and positive")
        if interpretation is LigamentKineticInterpretation.LOCAL_INCREMENTAL_FRONT:
            multiplicity = A / site
            support = A
        elif interpretation is LigamentKineticInterpretation.COLLECTIVE_FINITE_LIGAMENT:
            multiplicity = 1.0
            support = A
        else:  # pragma: no cover - Enum construction normally prevents this
            raise ValueError(f"unsupported ligament interpretation: {interpretation}")
        return {
            "candidate_class": self.candidate_class.value,
            "interpretation": interpretation.value,
            "energy_release_J_m2": self.energy_release_J_m2,
            "activation_area_m2": support,
            "multiplicity": multiplicity,
            "cavity_free_span_charged_as_solid": "false",
        }


def finite_difference_energy_release(
    energy_before_J_per_m: float,
    energy_after_J_per_m: float,
    new_solid_length_m: float,
) -> float:
    """Comparable finite-operation energy release per new solid area."""
    operation = FiniteSeparationOperation(
        CandidateClass.FINITE_SOLID_LIGAMENT,
        released_energy_J_per_m_thickness=float(energy_before_J_per_m) - float(energy_after_J_per_m),
        new_solid_length_m=new_solid_length_m,
    )
    return operation.energy_release_J_m2


__all__ = [
    "CandidateClass", "FiniteSeparationOperation", "LigamentKineticInterpretation",
    "finite_difference_energy_release",
]
