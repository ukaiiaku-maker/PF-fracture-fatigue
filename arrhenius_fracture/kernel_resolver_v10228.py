"""Direct prescribed-geometry resolver for v10.2.28 signed FEM kernels.

The mature v10.2.27 cache/coverage/locking implementation is reused.  Only the
mechanical identity, default provider command, and promotion evidence are
replaced; fixed-point trajectory promotion is not accepted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from . import kernel_resolver_v10227 as _legacy
from .kernel_configuration_v10227 import MechanicalKernelConfiguration
from .kernel_normalization_contract_v10228 import (
    DEFAULT_BURGERS_M,
    DEFAULT_KINETIC_PACKET_LENGTH_M,
)
from .kernel_registry_v10227 import validate_family as _validate_family_base

ROOT = Path(__file__).resolve().parents[1]
BUILD_SCHEMA = "v10.2.28_direct_prescribed_geometry_kernel_build_v1"
VALIDATION_SCHEMA = "v10.2.28_direct_kernel_validation_v1"
PROVIDER_ID = "v10.2.28_direct_prescribed_geometry_fem_v1"

_LEGACY_CONFIGURATION = _legacy._configuration
_LEGACY_VALIDATE_PROMOTION = _legacy._validate_promotion_evidence
_LEGACY_REGISTRY_PROMOTION = _legacy._registry_entry_is_promoted
_LEGACY_CLEAR_CACHE = _legacy._clear_generated_cache
_LEGACY_VALIDATE_FAMILY = _legacy.validate_family
_LEGACY_REQUIRED_MAX_EXTENSION = _legacy.required_max_extension_um


def required_max_extension_um(**kwargs) -> float:
    """Return strict coverage with one extra physical checkpoint of guard space.

    The v10.2.27 bound covers the target projection plus one maximum stochastic
    event.  The moving process-zone state can begin resolving the next partial
    checkpoint before the outer driver observes the completed target event.  Add
    one full ``da_phys`` checkpoint so strict interpolation never depends on
    floating-point endpoint coincidence or clips a legitimate final-state query.
    """
    base = _LEGACY_REQUIRED_MAX_EXTENSION(**kwargs)
    checkpoint = max(float(kwargs.get("da_phys_um", 0.0)), 0.0)
    return base + checkpoint


def _configuration(args: argparse.Namespace) -> MechanicalKernelConfiguration:
    base = _LEGACY_CONFIGURATION(args)
    payload = base.canonical_payload()
    payload.update(
        {
            "profile_id": "v10_2_28_direct_prescribed_geometry_single_front",
            "mechanics_backend": "v10.2.28_sharp_front_fem_direct_kernel",
            "mesh_policy_id": "v10.2.28_production_evolution_plus_direct_measurement_mesh",
            "measurement_mesh_policy_id": "v10.2.28_direct_prescribed_geometry_endpoint_mesh",
            "normalization_policy": "v10.2.28_unchanged_code_defined_activation_line_conversion",
            "kernel_provider_id": PROVIDER_ID,
        }
    )
    extra = dict(payload.get("extra", {}) or {})
    extra.setdefault("prescribed_crack_path_policy", "forward_100_cleavage_trace")
    extra.setdefault("burgers_m", DEFAULT_BURGERS_M)
    extra.setdefault("kinetic_packet_length_m", DEFAULT_KINETIC_PACKET_LENGTH_M)
    extra.update(
        {
            "geometry_anchor_source": "direct_prescription_not_stochastic_capture",
            "prior_kernel_family_required": False,
            "material_parameter_option_required": False,
            "hazard_seed_required": False,
        }
    )
    payload["extra"] = extra
    return MechanicalKernelConfiguration.from_mapping(payload)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _family_declares_direct_provider(path: str | Path) -> bool:
    payload = _load_object(Path(path).expanduser().resolve())
    return bool(
        payload.get("kernel_provider_id") == PROVIDER_ID
        and payload.get("direct_prescribed_geometry") is True
        and payload.get("prior_kernel_family_required") is False
        and payload.get("material_parameter_option_required") is False
        and payload.get("hazard_seed_required") is False
        and payload.get("stochastic_trajectory_required") is False
        and payload.get("production_physics_modified") is False
    )


def _validate_direct_family(path, *args, **kwargs) -> dict[str, Any]:
    audit = _validate_family_base(path, *args, **kwargs)
    if not _family_declares_direct_provider(path):
        raise ValueError("family lacks v10.2.28 direct-provider provenance")
    return audit


def _validate_direct_evidence(
    cache_dir: Path,
    audit: dict[str, Any],
    configuration_fingerprint: str,
) -> dict[str, Any]:
    build_path = cache_dir / "kernel_build_manifest.json"
    validation_path = cache_dir / "direct_kernel_validation_manifest.json"
    if not build_path.is_file() or not validation_path.is_file():
        raise ValueError("cached family lacks direct prescribed-geometry evidence")
    build = _load_object(build_path)
    validation = _load_object(validation_path)
    failures: list[str] = []
    expected_build = {
        "schema": BUILD_SCHEMA,
        "configuration_fingerprint": configuration_fingerprint,
        "family_sha256": audit["file_sha256"],
        "family_physics_fingerprint": audit["physics_fingerprint"],
        "direct_provider_validated": True,
        "production_parameterization_promotion_allowed": True,
        "prior_kernel_family_required": False,
        "stochastic_trajectory_required": False,
        "production_physics_modified": False,
    }
    for key, expected in expected_build.items():
        if build.get(key) != expected:
            failures.append(f"build:{key}")
    expected_validation = {
        "schema": VALIDATION_SCHEMA,
        "passed": True,
        "configuration_fingerprint": configuration_fingerprint,
        "family_sha256": audit["file_sha256"],
        "family_physics_fingerprint": audit["physics_fingerprint"],
        "direct_prescribed_geometry": True,
        "material_option_independent_by_construction": True,
        "hazard_seed_independent_by_construction": True,
        "prior_family_independent_by_construction": True,
        "stochastic_event_independent_by_construction": True,
        "fracture_hazard_imported_or_invoked": False,
        "source_emission_imported_or_invoked": False,
        "moving_process_zone_imported_or_invoked": False,
        "load_invariance_passed": True,
        "positive_negative_multi_amplitude_linearity_passed": True,
    }
    for key, expected in expected_validation.items():
        if validation.get(key) != expected:
            failures.append(f"validation:{key}")
    if failures:
        raise ValueError(
            "direct kernel promotion evidence is invalid: "
            + ", ".join(sorted(failures))
        )
    return {
        "kernel_build_manifest": str(build_path),
        "direct_kernel_validation_manifest": str(validation_path),
        "direct_provider_validated": True,
        "production_parameterization_promotion_allowed": True,
        # Compatibility key consumed by the inherited registry writer only.
        "converged_iteration": None,
    }


def _registry_entry_is_direct(entry: dict[str, Any], audit: dict[str, Any]) -> bool:
    configuration = dict(entry.get("configuration") or {})
    return bool(
        entry.get("production_parameterization_promotion_allowed") is True
        and entry.get("self_consistency_physics_fingerprint")
        == audit["physics_fingerprint"]
        and configuration.get("kernel_provider_id") == PROVIDER_ID
    )


def _clear_generated_cache(cache_dir: Path) -> None:
    for name in (
        "family.json",
        "mechanics_normalization.json",
        "coverage_audit.json",
        "kernel_build_manifest.json",
        "direct_kernel_validation_manifest.json",
        "portable_load_invariance_reports",
        "portable_assembly",
        "prescribed_geometry_snapshots",
        "load_invariance",
    ):
        path = cache_dir / name
        shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)


def _has_option(arguments: list[str], name: str) -> bool:
    return any(token == name or token.startswith(name + "=") for token in arguments)


def build_parser() -> argparse.ArgumentParser:
    return _legacy.build_parser()


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv or [])
    if _has_option(arguments, "--snapshot-archive") or _has_option(
        arguments, "--load-invariance-archive"
    ):
        raise SystemExit(
            "v10.2.28 direct provider does not accept captured mechanics archives"
        )
    if not _has_option(arguments, "--builder-command"):
        arguments.extend(
            [
                "--builder-command",
                "bash scripts/run_v10_2_28_prescribed_geometry_kernel.sh",
            ]
        )
    if not _has_option(arguments, "--cache-root"):
        arguments.extend(
            ["--cache-root", str(ROOT / "runs" / "v10_2_28_kernel_cache")]
        )
    if not _has_option(arguments, "--tracked-registry"):
        arguments.extend(
            [
                "--tracked-registry",
                str(ROOT / "artifacts" / "v10_2_28_kernel_registry.json"),
            ]
        )

    _legacy.required_max_extension_um = required_max_extension_um
    _legacy._configuration = _configuration
    _legacy._validate_promotion_evidence = _validate_direct_evidence
    _legacy._registry_entry_is_promoted = _registry_entry_is_direct
    _legacy._clear_generated_cache = _clear_generated_cache
    _legacy.validate_family = _validate_direct_family
    try:
        return _legacy.main(arguments)
    finally:
        _legacy.required_max_extension_um = _LEGACY_REQUIRED_MAX_EXTENSION
        _legacy._configuration = _LEGACY_CONFIGURATION
        _legacy._validate_promotion_evidence = _LEGACY_VALIDATE_PROMOTION
        _legacy._registry_entry_is_promoted = _LEGACY_REGISTRY_PROMOTION
        _legacy._clear_generated_cache = _LEGACY_CLEAR_CACHE
        _legacy.validate_family = _LEGACY_VALIDATE_FAMILY


__all__ = [
    "BUILD_SCHEMA",
    "VALIDATION_SCHEMA",
    "PROVIDER_ID",
    "build_parser",
    "main",
    "required_max_extension_um",
    "_configuration",
    "_validate_direct_evidence",
]
