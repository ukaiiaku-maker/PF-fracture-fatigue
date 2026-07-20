#!/usr/bin/env python3
"""Build the v10.2.14 active-only campaign atlas from completed E-state mechanics.

This is the direct campaign path requested by the project owner.  It does not
rerun FEM mechanics, refit any material parameter, or fabricate response data.
It assembles the already completed E000/E200/E500/E800 load-invariance outputs,
derives the mechanics normalization from a serialized engine configuration, and
marks the resulting family campaign-ready after all machine-verifiable mechanics
gates pass.  No claim of an independent manual review is made.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.mechanics_normalization_v10212 import (
    SourceGeometryAssumptions,
    derive_from_json,
)
from arrhenius_fracture.signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
    SCHEMA as V10214_SCHEMA,
)

BASE_PATH = ROOT / "scripts" / "build_v10_2_14_active_only_real_signed_atlas.py"
SPEC = importlib.util.spec_from_file_location("v10214_atlas_helpers", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load atlas helpers from {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

MODEL_ID = "v10.2.15_campaign_ready_v10.2.14_active_only_atlas"
MECHANICAL_GATES = (
    "physical_fem_station_inputs_present",
    "frozen_geometry_load_invariance_passed",
    "opening_axis_collapsed_after_validation",
    "multiple_cumulative_path_extensions_present",
    "spatial_projection_cross_validation_passed",
    "subelement_rows_not_claimed_as_direct_fem",
    "interaction_integral_provenance_consistent",
    "mechanics_derived_activation_to_line_normalization",
    "mechanics_derived_source_capacity_bounds",
    "no_fitted_shielding_attenuation",
    "base_extension_only_kernel_gates_passed",
)


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON object required: {path}")
    return payload


def _run_review_source(
    responses: list[Path], reports: list[Path], normalization: Path, source_out: Path
) -> None:
    command = [sys.executable, str(BASE.SOURCE_BUILDER)]
    for response in responses:
        command.extend(["--responses", str(response)])
    for report in reports:
        command.extend(["--load-invariance", str(report)])
    command.extend(
        ["--normalization", str(normalization), "--out", str(source_out)]
    )
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise SystemExit(
            "v10.2.13 mechanical source build failed:\n"
            + completed.stdout
            + completed.stderr
        )
    if completed.stdout:
        print(completed.stdout, end="")


def _promote(source_out: Path, out: Path, metadata: dict) -> None:
    payload = _load(source_out)
    states = payload.get("states")
    if not isinstance(states, list) or len(states) < 2:
        raise SystemExit("mechanical source does not contain a valid state family")
    for row in states:
        if not isinstance(row, dict):
            raise SystemExit("invalid state row in mechanical source")
        for key in (
            "wake_kernel_I_Pa_sqrt_m_per_signed_line",
            "wake_kernel_II_Pa_sqrt_m_per_signed_line",
        ):
            if not BASE._array_is_zero(row.get(key)):
                raise SystemExit(
                    f"refusing active-only campaign build because {key} is nonzero "
                    f"in state {row.get('state_id')!r}"
                )

    gates = payload.get("real_atlas_authorization_gates", {})
    failed = [name for name in MECHANICAL_GATES if gates.get(name) is not True]
    if failed:
        raise SystemExit(
            "completed mechanics did not satisfy campaign gates: " + ", ".join(failed)
        )

    source_schema = payload.get("schema")
    payload.update(
        {
            "schema": V10214_SCHEMA,
            "production_parameterization_allowed": True,
            "campaign_parameterization_allowed": True,
            "authorization_basis": (
                "completed_machine_verified_v10.2.14_mechanics_and_project_owner_direction"
            ),
            "independent_manual_review_performed": False,
            "active_kernel_mechanically_measured": True,
            "wake_kernel_mechanically_measured": False,
            "wake_shielding_supported": False,
            "wake_kernel_forced_zero": True,
            "same_kernel_family_for_monotonic_and_fatigue": True,
            "constitutive_K_shield_cap_present": False,
            "signed_burgers_population_required": True,
            "kernel_radius_axis_policy": "disabled_constant_compatibility",
            "opening_axis_policy": "validation_only_collapsed_constant_compatibility",
            "kernel_opening_compatibility_coordinate": 0.0,
            "full_mpz_grid_values_are_spatial_projection": True,
            "direct_fem_measurements_exist_only_at_recorded_station_indices": True,
            "frozen_geometry_load_invariance_passed": True,
            "campaign_promotion": {
                "model_id": MODEL_ID,
                "source_schema": source_schema,
                "automatic_from_completed_mechanics": True,
                "mechanics_rerun_performed": False,
                "material_parameter_refit_performed": False,
                "manual_review_gate_required": False,
                **metadata,
            },
        }
    )

    candidate = out.with_suffix(out.suffix + ".candidate")
    candidate.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    try:
        family = ActiveOnlySigned2DShieldingKernelFamily.from_json(candidate)
        if family.metadata.get("production_parameterization_allowed") is not True:
            raise SystemExit("campaign atlas lost production-allowed metadata")
        out.parent.mkdir(parents=True, exist_ok=True)
        candidate.replace(out)
    finally:
        candidate.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-invariance-root", type=Path, required=True)
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-source-spacing-b", type=float, default=10.0)
    parser.add_argument("--maximum-source-spacing-b", type=float, default=100.0)
    parser.add_argument("--source-region-length-um", type=float)
    args = parser.parse_args()

    out = args.out.expanduser().resolve()
    if out.exists():
        family = ActiveOnlySigned2DShieldingKernelFamily.from_json(out)
        print(json.dumps({"out": str(out), "reused": True, "states": len(family.states)}, indent=2))
        return

    engine_config = args.engine_config.expanduser().resolve()
    if not engine_config.is_file():
        raise SystemExit(f"engine config is missing: {engine_config}")
    source_root = BASE._find_state_root(args.load_invariance_root)
    assembly_root = out.parent / f"{out.stem}_assembly_inputs"
    assembly_root.mkdir(parents=True, exist_ok=True)
    responses, reports, audit_rows = BASE._prepare_reports(source_root, assembly_root)

    assumptions = SourceGeometryAssumptions(
        minimum_spacing_b=args.minimum_source_spacing_b,
        maximum_spacing_b=args.maximum_source_spacing_b,
        source_region_length_m=(
            None
            if args.source_region_length_um is None
            else float(args.source_region_length_um) * 1.0e-6
        ),
    )
    normalization_payload = derive_from_json(engine_config, assumptions=assumptions)
    normalization = out.parent / f"{out.stem}_mechanics_normalization.json"
    normalization.write_text(json.dumps(normalization_payload, indent=2, sort_keys=True) + "\n")

    source_out = out.parent / f"{out.stem}_v10_2_13_mechanical_source.json"
    source_out.unlink(missing_ok=True)
    _run_review_source(responses, reports, normalization, source_out)
    _promote(
        source_out,
        out,
        {
            "load_invariance_root": str(source_root),
            "engine_config": str(engine_config),
            "normalization": str(normalization),
            "input_states": audit_rows,
        },
    )

    family = ActiveOnlySigned2DShieldingKernelFamily.from_json(out)
    print(
        json.dumps(
            {
                "out": str(out),
                "schema": family.metadata.get("schema"),
                "states": len(family.states),
                "state_ids": [state.state_id for state in family.states],
                "production_parameterization_allowed": True,
                "mechanics_rerun_performed": False,
                "material_parameter_refit_performed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
