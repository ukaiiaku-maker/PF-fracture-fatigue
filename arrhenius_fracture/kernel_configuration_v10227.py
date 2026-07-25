"""Canonical mechanical-configuration identity for v10.2.27 shielding kernels.

Material kinetics, temperatures with fixed mechanics, random seeds, and plotting
settings deliberately do not enter this identity. The fingerprint represents
only the mechanical problem whose signed FEM response is being cached.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "v10.2.27_mechanical_kernel_configuration_v1"
DEFAULT_PROFILE_ID = "v10_2_27_default_single_front_frontfix"
NON_MECHANICAL_KEYS = {
    "candidate_id",
    "hazard_seed",
    "material_class",
    "material_option",
    "option",
    "options",
    "plot_settings",
    "random_seed",
    "save_snapshots",
    "seed",
    "snapshot_columns",
    "steps",
    "target_extension_um",
    "temperatures_K",
}


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("mechanical configuration contains a non-finite float")
        if value == 0.0:
            return 0.0
        return float(f"{value:.16g}")
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported configuration value: {type(value).__name__}")


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(_normalize(payload), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class MechanicalKernelConfiguration:
    profile_id: str = DEFAULT_PROFILE_ID
    theta_deg: float = 30.0
    nominal_crack_angle_deg: float = 0.0
    branching_mode: str = "single_front"
    maximum_fronts: int = 1
    mechanics_backend: str = "v10.2.27_sharp_front_fem"
    specimen_geometry_id: str = "v10.2.27_default_specimen"
    boundary_condition_id: str = "v10.2.27_mode_I_displacement"
    mesh_policy_id: str = "v10.2.27_front_direction_fix_mesh"
    process_zone_policy_id: str = "dynamic_tip_radius_physical_front_width"
    active_station_policy_id: str = "v10.2.14_measured_active_stations"
    signed_channel_convention: str = "v10.2.27_signed_mobile_retained_channels"
    front_direction_convention: str = "v10.2.27_front_direction_fix"
    normalization_policy: str = "derive_v10.2.12_from_snapshot_engine_config"
    elasticity_policy: str = "runtime_engine_configuration"
    initial_crack_length_m: float | None = None
    interaction_length_m: float | None = None
    temperature_dependent_mechanics: bool = False
    temperature_K: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "MechanicalKernelConfiguration":
        if not str(self.profile_id).strip():
            raise ValueError("profile_id must be non-empty")
        if not math.isfinite(float(self.theta_deg)):
            raise ValueError("theta_deg must be finite")
        if not math.isfinite(float(self.nominal_crack_angle_deg)):
            raise ValueError("nominal_crack_angle_deg must be finite")
        if self.branching_mode not in {"single_front", "topology_cached", "direct_fem"}:
            raise ValueError(f"unsupported branching_mode={self.branching_mode!r}")
        if int(self.maximum_fronts) < 1:
            raise ValueError("maximum_fronts must be at least one")
        if self.branching_mode == "single_front" and int(self.maximum_fronts) != 1:
            raise ValueError("single_front mode requires maximum_fronts=1")
        if self.branching_mode != "single_front" and int(self.maximum_fronts) < 2:
            raise ValueError("branch-aware modes require maximum_fronts>=2")
        for name in ("initial_crack_length_m", "interaction_length_m"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0.0):
                raise ValueError(f"{name} must be positive and finite when supplied")
        if self.temperature_dependent_mechanics:
            if self.temperature_K is None or not math.isfinite(float(self.temperature_K)):
                raise ValueError(
                    "temperature_K is required when temperature_dependent_mechanics is true"
                )
        return self

    def canonical_payload(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["schema"] = SCHEMA
        if not self.temperature_dependent_mechanics:
            payload["temperature_K"] = None
        return _normalize(payload)

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.canonical_payload())).hexdigest()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MechanicalKernelConfiguration":
        source = dict(payload)
        source.pop("schema", None)
        known = set(cls.__dataclass_fields__)
        extra = dict(source.pop("extra", {}) or {})
        for key in list(source):
            if key in NON_MECHANICAL_KEYS:
                source.pop(key)
        unknown = {key: source.pop(key) for key in list(source) if key not in known}
        extra.update(unknown)
        source["extra"] = extra
        return cls(**source).validate()


def load_configuration(path: str | Path) -> MechanicalKernelConfiguration:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"mechanical configuration must be a JSON object: {source}")
    return MechanicalKernelConfiguration.from_mapping(payload)


__all__ = [
    "SCHEMA",
    "DEFAULT_PROFILE_ID",
    "NON_MECHANICAL_KEYS",
    "MechanicalKernelConfiguration",
    "canonical_json_bytes",
    "load_configuration",
]
