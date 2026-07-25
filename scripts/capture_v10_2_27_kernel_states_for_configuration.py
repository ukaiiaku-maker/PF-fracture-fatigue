#!/usr/bin/env python3
"""Generate frozen FEM kernel states for one mechanical configuration."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.kernel_configuration_v10227 import load_configuration

SOURCE_REGISTRY = (
    ROOT / "arrhenius_fracture" / "data" / "materials"
    / "v10_2_27_v913_four_class_paper_registry.csv"
)


def _write_mechanics_only_manifest(source_registry: Path, destination: Path) -> Path:
    """Create an internal geometry driver, not a scientific material reference.

    The artificial cleavage surface is dormant at zero stress but becomes active
    under the FEM loading ramp. Its 1.2 eV floor remains slow enough that adaptive
    stepping cannot produce a multi-event jump at the minimum step fraction. This
    makes every fixed 5 um geometry increment observable by the snapshot matcher.
    """
    with source_registry.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    if not rows or not fieldnames:
        raise ValueError(f"current parameter registry is empty: {source_registry}")
    template = dict(rows[0])
    template.update({
        "option_key": "v10_2_27_mechanics_only_geometry_driver",
        "candidate_id": "v10_2_27_mechanics_only_geometry_driver",
        "material_class": "mechanics_only",
        "role": "internal deterministic geometry construction",
        "mechanism_summary": (
            "Artificial stress-triggered cleavage and suppressed emission used only "
            "to advance an unshielded single front through prescribed geometries."
        ),
        "validation_status": "not a material parameterization",
        "target_class": "mechanics_only",
        "cleave_G00_eV": "4.0",
        "cleave_gT_eV_per_K": "0",
        "cleave_sigc0_GPa": "2.0",
        "cleave_sT_GPa_per_K": "0",
        "cleave_exp_a": "1.0",
        "cleave_exp_n": "1.0",
        "cleave_floor_frac": "0.30",
        "emit_G00_eV": "100",
        "emit_gT_eV_per_K": "0",
        "emit_sigc0_GPa": "100",
        "emit_sT_GPa_per_K": "0",
        "emit_exp_a": "1",
        "emit_exp_n": "1",
        "emit_floor_frac": "0.95",
        "source_sites_per_system": "1",
        "encounter_efficiency": "0",
        "retained_recovery_rate_s": "0",
        "source_refresh_length_um": "0",
        "c_blunt": "0",
        "max_K_shield_MPa_sqrt_m": "0",
    })
    for field in ("target_class", "max_K_shield_MPa_sqrt_m"):
        if field not in fieldnames:
            fieldnames.append(field)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(template)
    return destination.resolve()


def _run(command: list[str], *, env: dict[str, str]) -> None:
    print("RUN:", " ".join(command), file=sys.stderr, flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            "automatic current-configuration FEM capture failed with exit "
            f"code {completed.returncode}"
        )


def _round_up(value: float, quantum: float) -> float:
    return math.ceil((value - 1.0e-12 * quantum) / quantum) * quantum


def _state_rows(
    *, required_max_um: float, spacing_um: float, da_um: float,
    temperature_K: float, interaction_length_m: float,
) -> tuple[list[dict[str, object]], float]:
    if required_max_um <= 0.0:
        raise ValueError("required maximum extension must be positive")
    spacing = max(_round_up(spacing_um, da_um), da_um)
    maximum = _round_up(required_max_um, spacing)
    count = max(int(round(maximum / spacing)), 1)
    tolerance_m = max(0.51 * da_um, 1.0e-3) * 1.0e-6
    rows = []
    for index in range(count + 1):
        extension_um = index * spacing
        rows.append({
            "state_id": f"E{int(round(extension_um)):07d}",
            "temperature_K": float(temperature_K),
            "cumulative_crack_path_extension_m": extension_um * 1.0e-6,
            "extension_tolerance_m": tolerance_m,
            "interaction_ell_m": float(interaction_length_m),
        })
    return rows, maximum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanical-config", type=Path, required=True)
    parser.add_argument("--snapshot-out", type=Path, required=True)
    parser.add_argument("--run-out", type=Path, required=True)
    parser.add_argument("--required-max-extension-um", type=float, required=True)
    parser.add_argument("--target-extension-um", type=float, required=True)
    parser.add_argument("--theta-deg", type=float, required=True)
    parser.add_argument("--capture-temperature-K", type=float)
    parser.add_argument("--registry", type=Path, default=SOURCE_REGISTRY)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    configuration = load_configuration(args.mechanical_config)
    if configuration.branching_mode != "single_front" or configuration.maximum_fronts != 1:
        raise SystemExit(
            "automatic fixed-extension capture is single-front only; branching "
            "requires a topology_cached or direct_fem provider"
        )
    if abs(float(configuration.theta_deg) - float(args.theta_deg)) > 1.0e-10:
        raise SystemExit("capture theta does not match the mechanical configuration")

    snapshot_out = args.snapshot_out.expanduser().resolve()
    run_out = args.run_out.expanduser().resolve()
    if args.force:
        shutil.rmtree(snapshot_out, ignore_errors=True)
        shutil.rmtree(run_out, ignore_errors=True)
    elif snapshot_out.exists() or run_out.exists():
        raise SystemExit("capture output exists; resolver must request a forced rebuild")
    # PhysicalFEMCapture owns creation of snapshot_out and intentionally rejects
    # any pre-existing output root. Create only its parent here.
    snapshot_out.parent.mkdir(parents=True, exist_ok=True)
    run_out.mkdir(parents=True, exist_ok=True)

    capture_temperature = (
        float(configuration.temperature_K)
        if configuration.temperature_dependent_mechanics
        else float(args.capture_temperature_K or 700.0)
    )
    mechanics_manifest = _write_mechanics_only_manifest(
        args.registry.expanduser().resolve(),
        run_out / "mechanics_only_geometry_driver_manifest.csv",
    )
    da_um = 1.0e6 * configuration.da_phys_m
    rows, atlas_max_um = _state_rows(
        required_max_um=float(args.required_max_extension_um),
        spacing_um=1.0e6 * configuration.atlas_anchor_spacing_m,
        da_um=da_um,
        temperature_K=capture_temperature,
        interaction_length_m=configuration.interaction_length_m,
    )
    state_table = run_out / "kernel_state_table.csv"
    with state_table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cosine = abs(math.cos(math.radians(float(args.theta_deg))))
    if cosine <= 1.0e-12:
        raise SystemExit("theta produces zero projected-extension cosine")
    projected_stop_um = atlas_max_um * cosine + 4.0 * da_um
    estimated_advances = int(math.ceil(atlas_max_um / da_um))
    steps = max(4000, 12 * estimated_advances + 2000)

    command = [
        sys.executable, "-u", "-m", "arrhenius_fracture.sharp_front_v10_2_13_capture",
        "--atlas-state-table", str(state_table),
        "--atlas-outroot", str(snapshot_out),
        "--minimum-elements-per-process-zone",
        f"{configuration.minimum_elements_per_process_zone:.17g}",
        "--mode", "2d",
        "--material-manifest", str(mechanics_manifest),
        "--temperatures", f"{capture_temperature:.17g}",
        "--out", str(run_out / "mechanics_run"),
        "--steps", str(steps),
        "--nx", str(configuration.mesh_nx),
        "--ny", str(configuration.mesh_ny),
        "--dU", "2e-7", "--dt", "8.4", "--n-stagger", "2",
        "--tip-h-fine", f"{configuration.tip_h_fine_m:.17g}",
        "--tip-ratio", f"{configuration.tip_ratio:.17g}",
        "--da-phys", f"{configuration.da_phys_m:.17g}",
        "--target-crack-extension-um", f"{projected_stop_um:.17g}",
        "--mpz-length-um", f"{1.0e6 * configuration.process_zone_length_m:.17g}",
        "--mpz-n-bins", str(configuration.process_zone_bins),
        "--front-state-model", "moving_pz",
        "--tip-source-model", "continuum",
        "--tip-kinetics-mode", "moving_velocity",
        "--bulk-plasticity-mode", "tip_only",
        "--directional-j-mode", "root_signed",
        "--no-tip-plasticity", "--no-active-shielding",
        "--signed-active-shielding", "--mobile-shield-fraction", "0",
        "--no-wake-shielding", "--crystal-aniso", "--crystal-compete",
        "--crystal-theta-deg", f"{float(args.theta_deg):.17g}",
        "--crystal-material", "w", "--j-decomposition", "cluster",
        "--max-fronts", "1", "--crack-backend", "sharp_wake",
        "--adaptive-events", "--adaptive-event-target", "0.15",
        "--adaptive-min-frac", "1e-10",
        "--print-every", "100", "--save-snapshots", "0", "--no-plots",
    ]
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": str(ROOT) + (
            ":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
        ),
        "PYTHONUNBUFFERED": "1",
        "CLEAVAGE_HAZARD_MODE": "deterministic",
        "CLEAVAGE_HAZARD_SEED": "0",
        "CLEAVAGE_EVENT_LENGTH_MODE": "fixed",
        "CLEAVAGE_EVENT_MIN_FACTOR": "1",
        "CLEAVAGE_EVENT_MAX_FACTOR": "1",
        "ANISOTROPIC_TRANSPORT_MODE": "validated_scalar",
        "ANISOTROPIC_USE_AVALANCHE_BACKEND": "1",
        "ANISOTROPIC_EMISSION_ENABLED": "1",
        "V10227_SPECIMEN_LX_M": f"{configuration.specimen_length_x_m:.17g}",
        "V10227_SPECIMEN_LY_M": f"{configuration.specimen_length_y_m:.17g}",
        "V10227_INITIAL_CRACK_LENGTH_M": f"{configuration.initial_crack_length_m:.17g}",
        "V10227_NOTCH_HALF_THICKNESS_M": f"{configuration.notch_half_thickness_m:.17g}",
    })
    _run(command, env=environment)

    complete = snapshot_out / "capture_complete.json"
    if not complete.is_file():
        raise SystemExit(f"capture did not produce {complete}")
    payload = json.loads(complete.read_text())
    if int(payload.get("captured_states", -1)) != len(rows):
        raise SystemExit(
            f"capture completed {payload.get('captured_states')} of {len(rows)} states"
        )
    manifest_payload = {
        "schema": "v10.2.27_current_configuration_kernel_capture_v3",
        "mechanical_configuration": configuration.canonical_payload(),
        "mechanical_configuration_fingerprint": configuration.fingerprint(),
        "trajectory_driver": {
            "driver": "internally_generated_mechanics_only_manifest",
            "capture_temperature_K": capture_temperature,
            "scientific_material_reference_condition": False,
            "existing_material_parameterization_required": False,
            "cleavage_surface": {
                "G00_eV": 4.0,
                "sigc0_GPa": 2.0,
                "exp_a": 1.0,
                "exp_n": 1.0,
                "floor_fraction": 0.30,
                "zero_stress_thermally_dormant": True,
            },
            "emission_suppressed": True,
            "adaptive_event_target": 0.15,
            "adaptive_minimum_fraction": 1.0e-10,
        },
        "requested_target_extension_um": float(args.target_extension_um),
        "required_kernel_path_extension_um": float(args.required_max_extension_um),
        "captured_atlas_max_extension_um": atlas_max_um,
        "state_count": len(rows),
        "state_table": str(state_table),
        "snapshot_root": str(snapshot_out),
        "run_root": str(run_out),
    }
    text = json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
    (run_out / "current_configuration_capture_manifest.json").write_text(text)
    (snapshot_out / "kernel_capture_manifest.json").write_text(text)
    print(json.dumps(manifest_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
