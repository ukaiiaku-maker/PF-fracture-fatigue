#!/usr/bin/env python3
"""Resolve, restore, or build the signed FEM kernel required by a campaign.

Production runners request a mechanical configuration and coverage; they do not
require a pre-existing FAMILY_JSON pathname.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
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
    resolve_repo_path,
    select_entry,
    select_recipe,
    sha256_file,
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
        payload = base.canonical_payload()
    else:
        payload = MechanicalKernelConfiguration(
            profile_id=args.mechanical_profile or DEFAULT_PROFILE_ID
        ).canonical_payload()
    payload.update(
        {
            "profile_id": args.mechanical_profile
            or payload.get("profile_id", DEFAULT_PROFILE_ID),
            "theta_deg": float(args.theta_deg),
            "branching_mode": args.branching_mode,
            "maximum_fronts": int(args.maximum_fronts),
        }
    )
    if args.initial_crack_length_um is not None:
        payload["initial_crack_length_m"] = (
            float(args.initial_crack_length_um) * 1.0e-6
        )
    if args.interaction_length_um is not None:
        payload["interaction_length_m"] = float(args.interaction_length_um) * 1.0e-6
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
    parser.add_argument("--da-phys-um", type=float, default=5.0)
    parser.add_argument("--event-minimum-factor", type=float, default=0.5)
    parser.add_argument("--event-maximum-factor", type=float, default=4.0)
    parser.add_argument("--margin-events", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    configuration = _configuration(args)
    fingerprint = configuration.fingerprint()
    required_um = required_max_extension_um(
        target_extension_um=args.target_extension_um,
        theta_deg=args.theta_deg,
        da_phys_um=args.da_phys_um,
        event_minimum_factor=args.event_minimum_factor,
        event_maximum_factor=args.event_maximum_factor,
        margin_events=args.margin_events,
    )

    if args.family_override is not None:
        audit = validate_family(args.family_override)
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
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(audit["family"])
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

        if args.mode == "build" and family_out.exists():
            family_out.unlink()

        if family_out.is_file() and args.mode != "build":
            try:
                audit = validate_family(family_out)
                if _coverage_ok(audit, required_um):
                    result = {
                        "resolution": "local_cache",
                        "configuration_fingerprint": fingerprint,
                        "required_max_extension_um": required_um,
                        **audit,
                    }
                    if args.json:
                        print(json.dumps(result, indent=2, sort_keys=True))
                    else:
                        print(audit["family"])
                    return 0
            except ValueError as exc:
                _emit(f"Ignoring invalid local kernel cache: {exc}")

        if args.mode != "build":
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
                )
                result = {
                    "resolution": "tracked_registry"
                    if registry is tracked_registry
                    else "local_registry",
                    "configuration_fingerprint": fingerprint,
                    "required_max_extension_um": required_um,
                    **audit,
                }
                if args.json:
                    print(json.dumps(result, indent=2, sort_keys=True))
                else:
                    print(audit["family"])
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
        ):
            raise SystemExit(
                "no branch-aware kernel provider is registered for this configuration. "
                "A fixed single-front extension atlas cannot represent branch interaction; "
                "register a topology_cached or direct_fem builder. "
                f"configuration_fingerprint={fingerprint}"
            )

        recipe = select_recipe(tracked_registry, configuration.canonical_payload())
        snapshot_archive = args.snapshot_archive
        load_archive = args.load_invariance_archive
        if recipe is not None:
            if snapshot_archive is None and recipe.get("snapshot_archive"):
                snapshot_archive = resolve_repo_path(ROOT, recipe["snapshot_archive"])
            if load_archive is None and recipe.get("load_invariance_archive"):
                load_archive = resolve_repo_path(
                    ROOT, recipe["load_invariance_archive"]
                )

        if snapshot_archive is not None and load_archive is not None:
            missing = [
                str(path)
                for path in (snapshot_archive, load_archive)
                if not Path(path).is_file()
            ]
            if missing:
                raise SystemExit(
                    "registered mechanics artifacts are missing; restore or commit these "
                    "compact inputs rather than searching for an old run directory: "
                    + ", ".join(missing)
                )
            if recipe is not None:
                expected_snapshot_sha = recipe.get("snapshot_archive_sha256")
                expected_load_sha = recipe.get("load_invariance_archive_sha256")
                if (
                    expected_snapshot_sha
                    and sha256_file(snapshot_archive) != expected_snapshot_sha
                ):
                    raise SystemExit("registered snapshot archive SHA-256 mismatch")
                if expected_load_sha and sha256_file(load_archive) != expected_load_sha:
                    raise SystemExit(
                        "registered load-invariance archive SHA-256 mismatch"
                    )
            _emit(
                "Building signed kernel from portable mechanics artifacts for "
                f"configuration {fingerprint[:12]}"
            )
            _run_builder(
                [
                    sys.executable,
                    str(
                        ROOT
                        / "scripts"
                        / "build_v10_2_27_kernel_from_mechanics_artifacts.py"
                    ),
                    "--snapshot-archive",
                    str(Path(snapshot_archive).expanduser().resolve()),
                    "--load-invariance-archive",
                    str(Path(load_archive).expanduser().resolve()),
                    "--mechanical-config",
                    str(config_path),
                    "--outroot",
                    str(cache_dir),
                    "--family-out",
                    str(family_out),
                    "--target-extension-um",
                    f"{args.target_extension_um:.17g}",
                    "--theta-deg",
                    f"{args.theta_deg:.17g}",
                    "--da-phys-um",
                    f"{args.da_phys_um:.17g}",
                    "--event-minimum-factor",
                    f"{args.event_minimum_factor:.17g}",
                    "--event-maximum-factor",
                    f"{args.event_maximum_factor:.17g}",
                    "--margin-events",
                    f"{args.margin_events:.17g}",
                ]
            )
        elif args.builder_command:
            environment = os.environ.copy()
            environment.update(
                {
                    "V10227_KERNEL_CONFIGURATION": str(config_path),
                    "V10227_KERNEL_CONFIGURATION_FINGERPRINT": fingerprint,
                    "V10227_KERNEL_CACHE_DIR": str(cache_dir),
                    "V10227_KERNEL_FAMILY_OUT": str(family_out),
                    "V10227_KERNEL_REQUIRED_MAX_EXTENSION_UM": f"{required_um:.17g}",
                    "V10227_KERNEL_TARGET_EXTENSION_UM": (
                        f"{args.target_extension_um:.17g}"
                    ),
                }
            )
            _emit(
                "Executing registered mechanical kernel builder for configuration "
                f"{fingerprint[:12]}"
            )
            _run_builder(shlex.split(args.builder_command), env=environment)
        else:
            raise SystemExit(
                "no kernel or automatic builder is registered for the requested mechanical "
                "configuration. Supply a mechanical configuration builder, not a reference "
                "FAMILY_JSON path. Builder protocol environment variables include "
                "V10227_KERNEL_CONFIGURATION and V10227_KERNEL_FAMILY_OUT. "
                f"configuration_fingerprint={fingerprint}"
            )

        audit = validate_family(family_out)
        if not _coverage_ok(audit, required_um):
            raise SystemExit(
                "newly built kernel lacks required coverage: "
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
            "resolution": "built",
            "configuration_fingerprint": fingerprint,
            "required_max_extension_um": required_um,
            **audit,
        }
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(audit["family"])
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
