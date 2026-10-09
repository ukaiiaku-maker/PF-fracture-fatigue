"""Typed option contract for the shared fracture--fatigue code family.

This module selects already-versioned physical strategies.  It intentionally
does not flatten source geography, source lifecycle, or mechanics backend into
one ambiguous ``model`` switch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import numbers
from typing import Any, Mapping


SCHEMA = "shared-fracture-fatigue-options/1"


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class MechanicsBackend(_ValueEnum):
    SPATIAL_PF = "spatial_pf"
    FULL_FEM_CZM = "full_fem_czm"


class LoadingMode(_ValueEnum):
    MONOTONIC = "monotonic"
    EXPLICIT_CYCLIC = "explicit_cyclic"
    FIXED_DELTA_K = "fixed_delta_k"
    HOLD_DWELL = "hold_dwell"


class SourceLocation(_ValueEnum):
    TIP_SURFACE = "tip_surface"
    BULK_UNIFORM = "bulk_uniform"
    TIP_PLUS_BULK = "tip_plus_bulk"


class SourceLifecycle(_ValueEnum):
    PERSISTENT_REUSABLE = "persistent_reusable"
    FINITE_INVENTORY = "finite_inventory"
    CURRENT_SOURCE = "current_source"


class PlasticityMode(_ValueEnum):
    TIP_PROCESS_ONLY = "tip_process_only"
    FULL_FIELD_FIXED_RHO = "full_field_fixed_rho"
    FULL_FIELD_EVOLVING_RHO = "full_field_evolving_rho"
    TIP_PLUS_FULL_FIELD = "tip_plus_full_field"


class CrackPathMode(_ValueEnum):
    SINGLE_FRONT = "single_front"
    BRANCHING_MULTIFRONT = "branching_multifront"


class CavityMode(_ValueEnum):
    DISABLED = "disabled"
    FULL_ORDER_V5 = "full_order_v5"


class KineticsStrategy(_ValueEnum):
    NATIVE_SIGNED = "native_signed"
    REVERSIBLE_SIGNED = "reversible_signed"
    OPERATOR_CONSISTENT_V3_6 = "operator_consistent_v3_6"


class IntegrationStrategy(_ValueEnum):
    ADAPTIVE_TRANSACTIONAL = "adaptive_transactional"
    EXPLICIT_EVENT = "explicit_event"
    COUPLED_FULL_HALF = "coupled_full_half"


class OutputProfile(_ValueEnum):
    COMPACT_SMOKE = "compact_smoke"
    SELECTED_FIELDS = "selected_fields"
    PUBLICATION_RESTART = "publication_restart"


@dataclass(frozen=True)
class SharedOptions:
    mechanics: MechanicsBackend = MechanicsBackend.SPATIAL_PF
    loading: LoadingMode = LoadingMode.MONOTONIC
    source_location: SourceLocation = SourceLocation.TIP_SURFACE
    source_lifecycle: SourceLifecycle = SourceLifecycle.PERSISTENT_REUSABLE
    plasticity: PlasticityMode = PlasticityMode.TIP_PROCESS_ONLY
    crack_path: CrackPathMode = CrackPathMode.SINGLE_FRONT
    cavity: CavityMode = CavityMode.DISABLED
    kinetics: KineticsStrategy = KineticsStrategy.NATIVE_SIGNED
    integration: IntegrationStrategy = IntegrationStrategy.ADAPTIVE_TRANSACTIONAL
    output: OutputProfile = OutputProfile.COMPACT_SMOKE
    material_manifest: str | None = None
    kernel_manifest: str | None = None
    random_seed: int = 1733
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        for name, enum_type in _ENUM_FIELDS.items():
            value = getattr(self, name)
            if not isinstance(value, enum_type):
                object.__setattr__(self, name, enum_type(value))
        if self.schema != SCHEMA:
            raise ValueError(f"unsupported option schema: {self.schema!r}")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, numbers.Integral):
            raise TypeError("random_seed must be an integer and not Boolean")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        object.__setattr__(self, "random_seed", int(self.random_seed))
        self.validate()

    def validate(self) -> None:
        errors: list[str] = []
        bulk_location = self.source_location in {
            SourceLocation.BULK_UNIFORM, SourceLocation.TIP_PLUS_BULK,
        }
        bulk_plasticity = self.plasticity in {
            PlasticityMode.FULL_FIELD_FIXED_RHO,
            PlasticityMode.FULL_FIELD_EVOLVING_RHO,
            PlasticityMode.TIP_PLUS_FULL_FIELD,
        }
        if bulk_location and not bulk_plasticity:
            errors.append("bulk source geography requires a full-field plasticity mode")
        if self.source_location is SourceLocation.TIP_SURFACE and self.plasticity in {
            PlasticityMode.FULL_FIELD_FIXED_RHO,
            PlasticityMode.FULL_FIELD_EVOLVING_RHO,
        }:
            errors.append("full-field-only plasticity requires bulk_uniform source geography")
        if self.cavity is CavityMode.FULL_ORDER_V5:
            if self.mechanics is not MechanicsBackend.SPATIAL_PF:
                errors.append("full_order_v5 is a spatial-PF strategy")
            # Multi-owner infrastructure is a construction dependency, not
            # permission to nucleate daughter cracks or a source-law choice.
        if self.kinetics is KineticsStrategy.OPERATOR_CONSISTENT_V3_6:
            if self.mechanics is not MechanicsBackend.SPATIAL_PF:
                errors.append("operator_consistent_v3_6 is a spatial-PF strategy")
            if self.crack_path is not CrackPathMode.SINGLE_FRONT:
                errors.append("operator_consistent_v3_6 currently owns one fixed crack front")
            if self.source_location is not SourceLocation.TIP_SURFACE:
                errors.append("operator_consistent_v3_6 uses material tip-slip patches")
        if self.mechanics is MechanicsBackend.FULL_FEM_CZM:
            if self.cavity is not CavityMode.DISABLED:
                errors.append("the selected clean FEM/CZM pair has no qualified V5 cavity adapter")
            if self.kinetics is KineticsStrategy.OPERATOR_CONSISTENT_V3_6:
                errors.append("the V3.6 fixed-support projection is PF-only")
        if errors:
            raise ValueError("incompatible shared options: " + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            key: (value.value if isinstance(value, Enum) else value)
            for key, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SharedOptions":
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown shared option fields: {sorted(unknown)}")
        return cls(**dict(payload))

    @classmethod
    def load(cls, path: str | Path) -> "SharedOptions":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


_ENUM_FIELDS = {
    "mechanics": MechanicsBackend,
    "loading": LoadingMode,
    "source_location": SourceLocation,
    "source_lifecycle": SourceLifecycle,
    "plasticity": PlasticityMode,
    "crack_path": CrackPathMode,
    "cavity": CavityMode,
    "kinetics": KineticsStrategy,
    "integration": IntegrationStrategy,
    "output": OutputProfile,
}


PRESETS: dict[str, SharedOptions] = {
    "pf_monotonic_tip_persistent": SharedOptions(),
    "pf_cyclic_tip_persistent": SharedOptions(
        loading=LoadingMode.EXPLICIT_CYCLIC,
        kinetics=KineticsStrategy.REVERSIBLE_SIGNED,
        output=OutputProfile.PUBLICATION_RESTART,
    ),
    "pf_monotonic_bulk": SharedOptions(
        source_location=SourceLocation.BULK_UNIFORM,
        source_lifecycle=SourceLifecycle.FINITE_INVENTORY,
        plasticity=PlasticityMode.FULL_FIELD_EVOLVING_RHO,
        output=OutputProfile.SELECTED_FIELDS,
    ),
    "pf_branching_current": SharedOptions(
        source_lifecycle=SourceLifecycle.CURRENT_SOURCE,
        crack_path=CrackPathMode.BRANCHING_MULTIFRONT,
        output=OutputProfile.PUBLICATION_RESTART,
    ),
    "pf_branching_void_v5": SharedOptions(
        source_lifecycle=SourceLifecycle.CURRENT_SOURCE,
        crack_path=CrackPathMode.BRANCHING_MULTIFRONT,
        cavity=CavityMode.FULL_ORDER_V5,
        plasticity=PlasticityMode.TIP_PLUS_FULL_FIELD,
        source_location=SourceLocation.TIP_PLUS_BULK,
        output=OutputProfile.PUBLICATION_RESTART,
    ),
    "pf_operator_consistent_cyclic": SharedOptions(
        loading=LoadingMode.EXPLICIT_CYCLIC,
        kinetics=KineticsStrategy.OPERATOR_CONSISTENT_V3_6,
        integration=IntegrationStrategy.COUPLED_FULL_HALF,
        output=OutputProfile.SELECTED_FIELDS,
    ),
    "fem_czm_theta0": SharedOptions(
        mechanics=MechanicsBackend.FULL_FEM_CZM,
        source_lifecycle=SourceLifecycle.PERSISTENT_REUSABLE,
        output=OutputProfile.PUBLICATION_RESTART,
        random_seed=8666,
    ),
}


__all__ = [
    "SCHEMA", "MechanicsBackend", "LoadingMode", "SourceLocation",
    "SourceLifecycle", "PlasticityMode", "CrackPathMode", "CavityMode",
    "KineticsStrategy", "IntegrationStrategy", "OutputProfile",
    "SharedOptions", "PRESETS",
]
