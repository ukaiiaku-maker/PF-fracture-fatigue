#!/usr/bin/env python3
"""Resolve or recalculate the signed FEM kernel required by a campaign.

The public contract is mechanical configuration plus coverage. Existing family
files and portable archives are optional caches; neither is required for a new
single-front configuration.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.kernel_configuration_v10227 import (  # noqa: E402
    DEFAULT_PROFILE_ID,
    MechanicalKernelConfiguration,
    load_configuration,
)
from arrhenius_fracture.kernel_registry_v10227 import (  # noqa: E402
    LOCAL_REGISTRY_SCHEMA,
    kernel_lock,
    load_registry,
    select_entry,
    update_local_registry,
    validate_family,
)


def required_max_extension_um(
    *,
    target_extension_um: float,
    theta_deg: float,
    da_phys_um: float,
    event_minimum_factor: float,
    event_maximum_factor: float,
    margin_events: float,
) -> float:
    cosine = abs(math.cos(math.radians(float(theta_deg))))
    if cosine <= 1.0e-12:
        raise ValueError("theta places the projected-extension direction at zero cosine")
    a = max(float(event_minimum_factor), 0.0)
    b = max(float(event_maximum_factor), a)
    clipped_mean = max(a + math.exp(-a) - math.exp(-b), 1.0e-300)
    maximum_event = float(da_phys_um) * float(event_maximum_factor) / clipped_mean
    return float(target_extension_um) / cosine + float(margin_events) * maximum_event


def _coverage_ok(audit: dict[str, Any], required_um: float) -> bool:
    return float(audit["maximum_extension_um"]) + 1.0e-6 >= required_um


def _configuration(args: argparse.Namespace) -> MechanicalKernelConfiguration:
    if args.mechanical_config is not None:
        base = load_configuration(args.mechanical_config)
    else:
        base = MechanicalKernelConfiguration(
            profile_id=args.mechanical_profile or DEFAULT_PROFILE_ID
        )
    payload = base.canonical_payload()
    payload.update(
        {
            "profile_id": args.mechanical_profile
            or payload.get("profile_id", DEFAULT_PROFILE_ID),
            "theta_deg": float(args.theta_deg),
            "branching_mode": args.branching_mode,
            "maximum_fronts": int(args.maximum_fronts),
        }
    )
    optional = {
        "process_zone_length_m": (args.process_zone_length_um, 1.0e-6),
        "process_zone_bins": (args.process_zone_bins, 1),
        "mesh_nx": (args.mesh_nx, 1),
        "mesh_ny": (args.mesh_ny, 1),
        "tip_h_fine_m": (args.tip_h_fine_um, 1.0e-6),
        "tip_ratio": (args.tip_ratio, 1.0),
        "atlas_anchor_spacing_m": (args.atlas_anchor_spacing_um, 1.0e-6),
        "minimum_elements_per_process_zone": (
            args.minimum_elements_per_process_zone,
            1.0,
        ),
        "da_phys_m": (args.da_phys_um, 1.0e-6),
        "interaction_length_m": (args.interaction_length_um, 1.0e-6),
        "initial_crack_length_m": (args.initial_crack_length_um, 1.0e-6),
    }
    integer_fields = {"process_zone_bins", "mesh_nx", "mesh_ny"}
    for key, (value, scale) in optional.items():
        if value is None:
            continue
        converted = float(value) * scale
        payload[key] = int(round(converted)) if key in integer_fields else converted
    if args.temperature_dependent_mechanics:
        if args.temperature_K is None:
            raise ValueError(
                "--temperature-K is required with --temperature-dependent-mechanics"
            )
        payload["temperature_dependent_mechanics"] = True
        payload["temperature_K"] = float(args.temperature_K)
    return MechanicalKernelConfiguration.from_mapping(payload)


def _read_local_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema": LOCAL_REGISTRY_SCHEMA, "entries": [], "recipes": []}
    payload = json.loads(path.read_text())
    if payload.get("schema") != LOCAL_REGISTRY_SCHEMA:
        raise ValueError(f"local kernel registry schema mismatch: {path}")
    payload.setdefault("recipes", [])
    return payload


def _emit(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _run_builder(command: list[str], env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        _emit(completed.stdout.rstrip())
    if completed.stderr:
        _emit(completed.stderr.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(
            f"kernel builder failed ({completed.returncode}): {' '.join(command)}"
        )


def _clear_generated_cache(cache_dir: Path) -> None:
    for name in (
        "family.json",
        "mechanics_normalization.json",
        "coverage_audit.json",
        "kernel_build_manifest.json",
        "portable_load_invariance_reports",
        "snapshots",
        "load_invariance",
        "capture",
    ):
        path = cache_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def _builder_environment(
    *,
    configuration_path: Path,
    fingerprint: str,
    cache_dir: Path,
    family_out: Path,
    required_um: float,
    configuration: MechanicalKernelConfiguration,
    args: argparse.Namespace,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "V10227_KERNEL_CONFIGURATION": str(configuration_path),
            "V10227_KERNEL_CONFIGURATION_FINGERPRINT": fingerprint,
            "V10227_KERNEL_CACHE_DIR": str(cache_dir),
            "V10227_KERNEL_FAMILY_OUT": str(family_out),
            "V10227_KERNEL_REQUIRED_MAX_EXTENSION_UM": f"{required_um:.17g}",
            "V10227_KERNEL_TARGET_EXTENSION_UM": f"{args.target_extension_um:.17g}",
            "V10227_KERNEL_THETA_DEG": f"{args.theta_deg:.17g}",
            "V10227_KERNEL_DA_PHYS_UM": f"{1.0e6 * configuration.da_phys_m:.17g}",
            "V10227_KERNEL_EVENT_MINIMUM_FACTOR": (
                f"{args.event_minimum_factor:.17g}"
            ),
            "V10227_KERNEL_EVENT_MAXIMUM_FACTOR": (
                f"{args.event_maximum_factor:.17g}"
            ),
            "V10227_KERNEL_MARGIN_EVENTS": f"{args.margin_events:.17g}",
        }
    )
    if args.snapshot_archive is not None:
        environment["KERNEL_SNAPSHOT_ARCHIVE"] = str(
            args.snapshot_archive.expanduser().resolve()
        )
    if args.load_invariance_archive is not None:
        environment["KERNEL_LOAD_INVARIANCE_ARCHIVE"] = str(
            args.load_invariance_archive.expanduser().resolve()
        )
    return environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta-deg", type=float, required=True)
    parser.add_argument("--target-extension-um", type=float, required=True)
    parser.add_argument(
        "--branching-mode",
        choices=("single_front", "topology_cached", "direct_fem"),
        default="single_front",
    )
    parser.add_argument("--maximum-fronts", type=int, default=1)
    parser.add_argument("--mechanical-profile")
    parser.add_argument("--mechanical-config", type=Path)
    parser.add_argument("--initial-crack-length-um", type=float)
    parser.add_argument("--interaction-length-um", type=float)
    parser.add_argument("--process-zone-length-um", type=float)
    parser.add_argument("--process-zone-bins", type=int)
    parser.add_argument("--mesh-nx", type=int)
    parser.add_argument("--mesh-ny", type=int)
    parser.add_argument("--tip-h-fine-um", type=float)
    parser.add_argument("--tip-ratio", type=float)
    parser.add_argument("--atlas-anchor-spacing-um", type=float)
    parser.add_argument("--minimum-elements-per-process-zone", type=float)
    parser.add_argument("--temperature-dependent-mechanics", action="store_true")
    parser.add_argument("--temperature-K", type=float)
    parser.add_argument("--family-override", type=Path)
    parser.add_argument(
        "--mode", choices=("auto", "reuse-only", "build"), default="auto"
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "runs" / "v10_2_27_kernel_cache",
    )
    parser.add_argument(
        "--tracked-registry",
        type=Path,
        default=ROOT / "artifacts" / "v10_2_27_kernel_registry.json",
    )
    parser.add_argument("--builder-command")
    parser.add_argument("--snapshot-archive", type=Path)
    parser.add_argument("--load-invariance-archive", type=Path)
    parser.add_argument("--da-phys-um", type=float)
    parser.add_argument("--event-minimum-factor", type=float, default=0.5)
    parser.add_argument("--event-maximum-factor", type=float, default=4.0)
    parser.add_argument("--margin-events", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if (args.snapshot_archive is None) != (args.load_invariance_archive is None):
        raise SystemExit(
            "--snapshot-archive and --load-invariance-archive are optional caches "
            "but must be supplied together"
        )

    configuration = _configuration(args)
    fingerprint = configuration.fingerprint()
    required_um = required_max_extension_um(
        target_extension_um=args.target_extension_um,
        theta_deg=args.theta_deg,
        da_phys_um=1.0e6 * configuration.da_phys_m,
        event_minimum_factor=args.event_minimum_factor,
        event_maximum_factor=args.event_maximum_factor,
        margin_events=args.margin_events,
    )

    if args.family_override is not None:
        audit = validate_family(
            args.family_override,
            expected_configuration_fingerprint=fingerprint,
        )
        if not _coverage_ok(audit, required_um):
            raise SystemExit(
                "explicit kernel override lacks required coverage: "
                f"maximum={audit['maximum_extension_um']:.9g} um, "
                f"required={required_um:.9g} um"
            )
        result = {
            "resolution": "explicit_override",
            "configuration_fingerprint": fingerprint,
            "required_max_extension_um": required_um,
            **audit,
        }
        print(json.dumps(result, indent=2, sort_keys=True) if args.json else audit["family"])
        return 0

    cache_root = args.cache_root.expanduser().resolve()
    cache_dir = cache_root / fingerprint
    family_out = cache_dir / "family.json"
    config_path = cache_dir / "mechanical_configuration.json"
    local_registry_path = cache_root / "registry.json"
    lock_path = cache_root / "locks" / f"{fingerprint}.lock"
    tracked_registry = load_registry(args.tracked_registry)

    with kernel_lock(lock_path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(configuration.canonical_payload(), indent=2, sort_keys=True)
            + "\n"
        )

        rebuild_reason = None
        if args.mode == "build":
            rebuild_reason = "explicit build requested"
        elif family_out.is_file():
            try:
                audit = validate_family(
                    family_out,
                    expected_configuration_fingerprint=fingerprint,
                )
                if _coverage_ok(audit, required_um):
                    result = {
                        "resolution": "local_cache",
                        "configuration_fingerprint": fingerprint,
                        "required_max_extension_um": required_um,
                        **audit,
                    }
                    print(
                        json.dumps(result, indent=2, sort_keys=True)
                        if args.json
                        else audit["family"]
                    )
                    return 0
                rebuild_reason = (
                    "cached coverage is too short: "
                    f"{audit['maximum_extension_um']:.9g} < {required_um:.9g} um"
                )
            except (FileNotFoundError, ValueError) as exc:
                rebuild_reason = f"cached family is invalid: {exc}"

        if rebuild_reason is None and args.mode != "build":
            registries = [tracked_registry, _read_local_registry(local_registry_path)]
            for registry in registries:
                selected = select_entry(
                    registry,
                    configuration_fingerprint=fingerprint,
                    required_max_extension_um=required_um,
                    repo_root=ROOT,
                )
                if selected is None:
                    continue
                entry, source = selected
                audit = validate_family(
                    source,
                    expected_file_sha256=entry.get("family_sha256"),
                    expected_physics_fingerprint=entry.get("physics_fingerprint"),
                    expected_configuration_fingerprint=fingerprint,
                )
                result = {
                    "resolution": "tracked_registry"
                    if registry is tracked_registry
                    else "local_registry",
                    "configuration_fingerprint": fingerprint,
                    "required_max_extension_um": required_um,
                    **audit,
                }
                print(
                    json.dumps(result, indent=2, sort_keys=True)
                    if args.json
                    else audit["family"]
                )
                return 0

        if args.mode == "reuse-only":
            raise SystemExit(
                "no validated kernel covers the requested mechanical configuration; "
                f"configuration_fingerprint={fingerprint}, "
                f"required_max_extension_um={required_um:.9g}"
            )

        if (
            configuration.branching_mode != "single_front"
            or configuration.maximum_fronts != 1
        ) and not args.builder_command:
            raise SystemExit(
                "no branch-aware kernel provider is registered for this configuration. "
                "A fixed extension atlas cannot represent branch interaction; register "
                "a topology_cached or direct_fem provider through --builder-command. "
                f"configuration_fingerprint={fingerprint}"
            )

        if rebuild_reason:
            _emit(f"Recalculating signed FEM kernel because {rebuild_reason}")
            _clear_generated_cache(cache_dir)
            config_path.write_text(
                json.dumps(configuration.canonical_payload(), indent=2, sort_keys=True)
                + "\n"
            )

        environment = _builder_environment(
            configuration_path=config_path,
            fingerprint=fingerprint,
            cache_dir=cache_dir,
            family_out=family_out,
            required_um=required_um,
            configuration=configuration,
            args=args,
        )
        if args.builder_command:
            command = shlex.split(args.builder_command)
            _emit(
                "Executing registered kernel provider for configuration "
                f"{fingerprint[:12]}"
            )
        else:
            command = [
                "bash",
                str(ROOT / "scripts" / "build_v10_2_27_kernel_for_configuration.sh"),
            ]
            _emit(
                "Recalculating snapshots, load invariance, normalization, and signed "
                f"kernel for configuration {fingerprint[:12]}"
            )
        _run_builder(command, env=environment)

        audit = validate_family(
            family_out,
            expected_configuration_fingerprint=fingerprint,
        )
        if not _coverage_ok(audit, required_um):
            raise SystemExit(
                "newly recalculated kernel lacks required coverage: "
                f"maximum={audit['maximum_extension_um']:.9g} um, "
                f"required={required_um:.9g} um"
            )
        entry = {
            "configuration_fingerprint": fingerprint,
            "configuration": configuration.canonical_payload(),
            "family": audit["family"],
            "family_sha256": audit["file_sha256"],
            "physics_fingerprint": audit["physics_fingerprint"],
            "minimum_extension_um": audit["minimum_extension_um"],
            "maximum_extension_um": audit["maximum_extension_um"],
            "provider_mode": configuration.branching_mode,
        }
        update_local_registry(local_registry_path, entry)
        result = {
            "resolution": "recalculated",
            "configuration_fingerprint": fingerprint,
            "required_max_extension_um": required_um,
            **audit,
        }
        print(json.dumps(result, indent=2, sort_keys=True) if args.json else audit["family"])
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
