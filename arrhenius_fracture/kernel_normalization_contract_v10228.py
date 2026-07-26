"""Candidate-independent line-content normalization used by the direct provider.

The values reproduce the existing production defaults.  They are separated from
elastic influence-function construction so no material option or stochastic state
is needed to generate the mechanical operator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

MODEL_ID = "v10.2.28_unchanged_code_defined_activation_line_conversion_v1"
DEFAULT_BURGERS_M = 2.74e-10
DEFAULT_KINETIC_PACKET_LENGTH_M = 2.5e-10


@dataclass(frozen=True)
class KernelNormalizationContract:
    burgers_m: float = DEFAULT_BURGERS_M
    kinetic_packet_length_m: float = DEFAULT_KINETIC_PACKET_LENGTH_M

    def validate(self) -> "KernelNormalizationContract":
        for name in ("burgers_m", "kinetic_packet_length_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        return self

    @property
    def activation_to_line_content(self) -> float:
        self.validate()
        return float(self.kinetic_packet_length_m / self.burgers_m)

    def audit_payload(self) -> dict[str, Any]:
        return {
            "schema": MODEL_ID,
            "burgers_m": float(self.burgers_m),
            "kinetic_packet_length_m": float(self.kinetic_packet_length_m),
            "activation_to_line_content": self.activation_to_line_content,
            "production_default_values_preserved": True,
            "material_parameter_option_dependent": False,
            "hazard_seed_dependent": False,
            "elastic_influence_operator_dependent": False,
            "fitted_to_toughness_or_fatigue": False,
        }


__all__ = [
    "MODEL_ID",
    "DEFAULT_BURGERS_M",
    "DEFAULT_KINETIC_PACKET_LENGTH_M",
    "KernelNormalizationContract",
]
