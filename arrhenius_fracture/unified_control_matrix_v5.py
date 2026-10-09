"""Bounded paper controls for the unified 2-D fracture model line."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .unified_fracture_material_v5 import (
    UnifiedFractureMaterialBundle, canonical_front_engine,
)


class PaperControl(str, Enum):
    SINGLE_TIP = "SINGLE_TIP"
    MULTI_TIP = "MULTI_TIP"
    MULTI_TIP_PLUS_VOIDING = "MULTI_TIP_PLUS_VOIDING"


@dataclass(frozen=True)
class PaperControlConfiguration:
    control: PaperControl
    material_bundle: UnifiedFractureMaterialBundle
    branching_enabled: bool
    voiding_enabled: bool

    @property
    def composite_identity(self) -> Mapping[str, Any]:
        extensions = []
        versions = {}
        if self.branching_enabled:
            extensions.append("multi_tip")
            versions["multi_tip"] = "v13.current-source-qualified-multitip/1"
        if self.voiding_enabled:
            extensions.append("voiding")
            versions["voiding"] = "v5.production-one-void-trajectory/5"
        return MappingProxyType({
            "core_model_id": self.material_bundle.hazard_algorithm_id,
            "material_bundle_id": self.material_bundle.bundle_id,
            "extensions_enabled": tuple(extensions),
            "extension_versions": MappingProxyType(versions),
        })

    def front_engine(self, elastic_material: Any):
        return canonical_front_engine(self.material_bundle, elastic_material)


def paper_control_matrix(
    bundle: UnifiedFractureMaterialBundle,
) -> Mapping[PaperControl, PaperControlConfiguration]:
    return {
        PaperControl.SINGLE_TIP: PaperControlConfiguration(
            PaperControl.SINGLE_TIP, bundle, False, False,
        ),
        PaperControl.MULTI_TIP: PaperControlConfiguration(
            PaperControl.MULTI_TIP, bundle, True, False,
        ),
        PaperControl.MULTI_TIP_PLUS_VOIDING: PaperControlConfiguration(
            PaperControl.MULTI_TIP_PLUS_VOIDING, bundle, True, True,
        ),
    }


DELTA_DEFINITIONS = {
    "branching_increment": "O[MULTI_TIP] - O[SINGLE_TIP]",
    "voiding_increment": "O[MULTI_TIP_PLUS_VOIDING] - O[MULTI_TIP]",
    "combined_increment": "O[MULTI_TIP_PLUS_VOIDING] - O[SINGLE_TIP]",
}


__all__ = [
    "DELTA_DEFINITIONS", "PaperControl", "PaperControlConfiguration",
    "paper_control_matrix",
]
