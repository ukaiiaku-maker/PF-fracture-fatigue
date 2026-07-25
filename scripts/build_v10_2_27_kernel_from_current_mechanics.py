#!/usr/bin/env python3
"""Validate current mechanics artifacts, build the kernel, and bind provenance."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.kernel_configuration_v10227 import (
    MechanicalKernelConfiguration,
    load_configuration,
)
from arrhenius_fracture.kernel_registry_v10227 import validate_family

BASE_PATH = ROOT / "scripts" / "build_v10_2_27_kernel_from_mechanics_artifacts.py"
SPEC = importlib.util.spec_from_file_location("v10227_portable_builder", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load mechanics builder from {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


def _validate_artifacts(
    snapshot_root: Path,
    states: list[dict[str, Any]],
    configuration: MechanicalKernelConfiguration,
) -> None:
    manifest_path = snapshot_root / "kernel_capture_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "snapshot mechanics lack a v10.2.27 configuration manifest. Legacy "
            "archives cannot be assumed compatible; recalculate from the requested "
            "mechanical configuration instead."
        )
    manifest = json.loads(manifest_path.read_text())
    expected_fingerprint = configuration.fingerprint()
    observed_fingerprint = str(
        manifest.get("mechanical_configuration_fingerprint", "")
    )
    if observed_fingerprint != expected_fingerprint:
        raise ValueError(
            "snapshot mechanical-configuration fingerprint mismatch: "
            f"{observed_fingerprint or '<missing>'} != {expected_fingerprint}"
        )
    if manifest.get("mechanical_configuration") != configuration.canonical_payload():
        raise ValueError("snapshot mechanical-configuration payload does not match request")

    for row in states:
        metadata = json.loads(Path(row["snapshot_json"]).read_text())
        engine = row["engine_config"]
        front = dict(engine.get("front_config", {}))
        mpz = dict(engine.get("mpz_config", {}))
        anisotropic = dict(engine.get("anisotropic_config", {}))
        checks = {
            "theta_deg": (
                float(anisotropic.get("crystal_theta_deg", float("nan"))),
                float(configuration.theta_deg),
                1.0e-10,
            ),
            "front_process_zone_length_m": (
                float(front.get("L_pz", float("nan"))),
                float(configuration.process_zone_length_m),
                1.0e-12,
            ),
            "mpz_length_m": (
                float(mpz.get("length_m", float("nan"))),
                float(configuration.process_zone_length_m),
                1.0e-12,
            ),
            "crack_advance_m": (
                float(front.get("da", float("nan"))),
                float(configuration.da_phys_m),
                1.0e-12,
            ),
            "interaction_length_m": (
                float(metadata.get("interaction_ell_m", float("nan"))),
                float(configuration.interaction_length_m),
                1.0e-12,
            ),
        }
        for name, (actual, expected, tolerance) in checks.items():
            if not math.isfinite(actual) or not math.isclose(
                actual, expected, rel_tol=1.0e-10, abs_tol=tolerance
            ):
                raise ValueError(
                    f"{row['state_id']} mechanics mismatch for {name}: "
                    f"{actual} != {expected}"
                )
        if int(mpz.get("n_bins", -1)) != int(configuration.process_zone_bins):
            raise ValueError(
                f"{row['state_id']} MPZ bins mismatch: "
                f"{mpz.get('n_bins')} != {configuration.process_zone_bins}"
            )
        if len(metadata.get("active_x_m", [])) != int(configuration.process_zone_bins):
            raise ValueError(
                f"{row['state_id']} active station count does not match "
                f"process_zone_bins={configuration.process_zone_bins}"
            )
        if metadata.get("fixed_crack_geometry") is not True:
            raise ValueError(f"{row['state_id']} is not a fixed-crack FEM snapshot")
        if metadata.get("active_kernel_supported") is not True:
            raise ValueError(f"{row['state_id']} does not support the active kernel")
        if metadata.get("wake_kernel_supported") is not False:
            raise ValueError(f"{row['state_id']} unexpectedly supports wake shielding")
        if configuration.temperature_dependent_mechanics:
            actual_temperature = float(metadata.get("temperature_K", float("nan")))
            if not math.isclose(
                actual_temperature,
                float(configuration.temperature_K),
                rel_tol=0.0,
                abs_tol=1.0e-8,
            ):
                raise ValueError(
                    f"{row['state_id']} temperature mismatch: "
                    f"{actual_temperature} != {configuration.temperature_K}"
                )


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    if completed.stdout:
        print(completed.stdout, end="", file=sys.stderr)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"current-configuration kernel build failed ({completed.returncode}): "
            + " ".join(command)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--snapshot-archive", type=Path)
    parser.add_argument("--load-invariance-root", type=Path)
    parser.add_argument("--load-invariance-archive", type=Path)
    parser.add_argument("--mechanical-config", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--family-out", type=Path, required=True)
    parser.add_argument("--target-extension-um", type=float, required=True)
    parser.add_argument("--theta-deg", type=float, required=True)
    parser.add_argument("--da-phys-um", type=float, required=True)
    parser.add_argument("--event-minimum-factor", type=float, default=0.5)
    parser.add_argument("--event-maximum-factor", type=float, default=4.0)
    parser.add_argument("--margin-events", type=float, default=1.0)
    args = parser.parse_args()

    configuration = load_configuration(args.mechanical_config)
    if configuration.branching_mode != "single_front" or configuration.maximum_fronts != 1:
        raise SystemExit("current built-in kernel provider is single-front only")

    outroot = args.outroot.expanduser().resolve()
    family_out = args.family_out.expanduser().resolve()
    outroot.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="v10227_current_kernel_") as temp:
        workspace = Path(temp)
        snapshot_root = BASE._materialize(
            root=args.snapshot_root,
            archive=args.snapshot_archive,
            workspace=workspace,
            kind="snapshots",
        )
        load_root = BASE._materialize(
            root=args.load_invariance_root,
            archive=args.load_invariance_archive,
            workspace=workspace,
            kind="load_invariance",
        )
        states = BASE._state_records(snapshot_root, load_root)
        _validate_artifacts(snapshot_root, states, configuration)

        command = [
            sys.executable,
            str(BASE_PATH),
            "--snapshot-root",
            str(snapshot_root),
            "--load-invariance-root",
            str(load_root),
            "--mechanical-config",
            str(args.mechanical_config.expanduser().resolve()),
            "--outroot",
            str(outroot),
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
        _run(command)

    payload = json.loads(family_out.read_text())
    payload.update(
        {
            "mechanical_configuration": configuration.canonical_payload(),
            "mechanical_configuration_fingerprint": configuration.fingerprint(),
            "kernel_provider_id": configuration.kernel_provider_id,
            "kernel_recalculated_from_current_configuration": True,
            "historical_reference_condition_required": False,
        }
    )
    family_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    audit = validate_family(
        family_out,
        expected_configuration_fingerprint=configuration.fingerprint(),
    )

    manifest_path = outroot / "kernel_build_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest.update(
        {
            "schema": "v10.2.27_current_configuration_kernel_build_v2",
            "configuration": configuration.canonical_payload(),
            "configuration_fingerprint": configuration.fingerprint(),
            "family": audit["family"],
            "family_sha256": audit["file_sha256"],
            "family_physics_fingerprint": audit["physics_fingerprint"],
            "historical_reference_condition_required": False,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reused": False, **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
