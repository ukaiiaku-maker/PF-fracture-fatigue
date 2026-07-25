"""Canonical mechanical identity for v10.2.27 signed FEM shielding kernels.

Only quantities that can change the static FEM response or the crack geometry
used to sample it belong in this identity. Material kinetics, stochastic seeds,
plotting settings, and the requested coverage do not.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "v10.2.27_mechanical_kernel_configuration_v2"
DEFAULT_PROFILE_ID = "v10_2_27_current_single_front_frontfix"
PROFILE_ALIASES = {
    "v10_2_27_default_single_front_frontfix": DEFAULT_PROFILE_ID,
}
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
    specimen_geometry_id: str = "v10.2.27_rectangular_single_edge_notch"
    specimen_length_x_m: float = 2.0e-3
    specimen_length_y_m: float = 4.0e-3
    initial_crack_length_m: float = 0.5e-3
    notch_half_thickness_m: float = 0.08e-3
    boundary_condition_id: str = "v10.2.27_mode_I_displacement"
    mesh_policy_id: str = "v10.2.27_front_direction_fix_mesh"
    mesh_nx: int = 36
    mesh_ny: int = 72
    tip_h_fine_m: float = 1.0e-6
    tip_ratio: float = 1.20
    process_zone_policy_id: str = "dynamic_tip_radius_physical_front_width"
    process_zone_length_m: float = 50.0e-6
    process_zone_bins: int = 80
    active_station_policy_id: str = "v10.2.14_measured_active_stations"
    interaction_length_m: float = 2.0e-6
    atlas_anchor_spacing_m: float = 200.0e-6
    minimum_elements_per_process_zone: float = 3.0
    da_phys_m: float = 5.0e-6
    signed_channel_convention: str = "v10.2.27_signed_mobile_retained_channels"
    front_direction_convention: str = "v10.2.27_front_direction_fix"
    normalization_policy: str = "derive_v10.2.12_from_captured_engine_config"
    elasticity_policy: str = "runtime_engine_configuration"
    kernel_provider_id: str = "v10.2.27_current_configuration_fem_recalculation_v1"
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
        for name in ("mesh_nx", "mesh_ny", "process_zone_bins"):
            if int(getattr(self, name)) < 2:
                raise ValueError(f"{name} must be at least two")
        for name in (
            "specimen_length_x_m",
            "specimen_length_y_m",
            "initial_crack_length_m",
            "notch_half_thickness_m",
            "tip_h_fine_m",
            "tip_ratio",
            "process_zone_length_m",
            "interaction_length_m",
            "atlas_anchor_spacing_m",
            "minimum_elements_per_process_zone",
            "da_phys_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.initial_crack_length_m >= self.specimen_length_x_m:
            raise ValueError("initial_crack_length_m must be smaller than specimen_length_x_m")
        if 2.0 * self.notch_half_thickness_m >= self.specimen_length_y_m:
            raise ValueError("notch thickness must be smaller than specimen_length_y_m")
        if self.temperature_dependent_mechanics:
            if self.temperature_K is None or not math.isfinite(float(self.temperature_K)):
                raise ValueError(
                    "temperature_K is required when temperature_dependent_mechanics is true"
                )
        _normalize(dict(self.extra))
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
        unknown = sorted(key for key in source if key not in known)
        if unknown:
            raise ValueError(
                "unknown mechanical-configuration keys: "
                + ", ".join(unknown)
                + "; put deliberate mechanical extensions inside the explicit 'extra' object"
            )
        if "profile_id" in source:
            profile = str(source["profile_id"])
            source["profile_id"] = PROFILE_ALIASES.get(profile, profile)
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
    "PROFILE_ALIASES",
    "NON_MECHANICAL_KEYS",
    "MechanicalKernelConfiguration",
    "canonical_json_bytes",
    "load_configuration",
]
