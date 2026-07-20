#!/usr/bin/env python3
"""Assemble an authorized v10.2.14 active-only atlas from completed FEM outputs.

This script does not rerun mechanics and does not fabricate response values.  It
reuses completed E000/E200/E500/E800 frozen-geometry load-invariance outputs,
derives the source normalization from a serialized production engine config,
builds the reviewed v10.2.13 extension-only artifact, and promotes it to the
v10.2.14 active-only schema only after every inherited authorization gate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from arrhenius_fracture.mechanics_normalization_v10212 import (
    SourceGeometryAssumptions,
    derive_from_json,
)
from arrhenius_fracture.signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
    SCHEMA as V10214_SCHEMA,
)

SOURCE_BUILDER = (
    REPOSITORY_ROOT / "scripts" / "build_v10_2_13_extension_only_real_signed_atlas.py"
)
MODEL_ID = "v10.2.15_assemble_v10.2.14_active_only_real_signed_atlas"
REQUIRED_STATES = ("E000", "E200", "E500", "E800")
REPORT_NAME = "frozen_geometry_load_invariance.json"
REFERENCE_RESPONSE_NAME = "active_station_responses_load_1.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON object required: {path}")
    return payload


def _find_state_root(supplied: Path) -> Path:
    root = supplied.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"load-invariance root is not a directory: {root}")
    if all((root / state / REPORT_NAME).is_file() for state in REQUIRED_STATES):
        return root
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_dir()
        and all((path / state / REPORT_NAME).is_file() for state in REQUIRED_STATES)
    ]
    if len(candidates) != 1:
        raise SystemExit(
            "expected exactly one directory containing E000/E200/E500/E800; "
            f"found {len(candidates)} under {root}"
        )
    return candidates[0]


def _local_response_for_scale(state_dir: Path, scale: float) -> Path:
    labels = {
        0.5: "active_station_responses_load_0p5.csv",
        1.0: REFERENCE_RESPONSE_NAME,
        1.5: "active_station_responses_load_1p5.csv",
    }
    for expected, name in labels.items():
        if abs(float(scale) - expected) <= 1.0e-12:
            return state_dir / name
    raise SystemExit(f"unsupported load scale {scale:g} in {state_dir / REPORT_NAME}")


def _prepare_reports(
    source_root: Path,
    assembly_root: Path,
) -> tuple[list[Path], list[Path], list[dict[str, Any]]]:
    rewritten_reports: list[Path] = []
    responses: list[Path] = []
    audit_rows: list[dict[str, Any]] = []
    seen_extensions: set[float] = set()

    for state in REQUIRED_STATES:
        state_dir = source_root / state
        report_path = state_dir / REPORT_NAME
        response_path = state_dir / REFERENCE_RESPONSE_NAME
        response_audit = response_path.with_suffix(".audit.json")
        for required in (report_path, response_path, response_audit):
            if not required.is_file():
                raise SystemExit(f"required completed mechanics artifact is missing: {required}")

        report = _json(report_path)
        if report.get("parent_state_id") != state:
            raise SystemExit(
                f"{report_path} parent_state_id must be {state!r}; "
                f"got {report.get('parent_state_id')!r}"
            )
        if report.get("load_invariance_passed") is not True:
            raise SystemExit(f"load invariance did not pass for {state}: {report_path}")
        if report.get("active_kernel_mechanically_measured") is not True:
            raise SystemExit(f"active kernel is not marked mechanically measured for {state}")
        if report.get("wake_kernel_mechanically_measured") is not False:
            raise SystemExit(f"wake kernel must be unmeasured for active-only state {state}")
        if report.get("wake_shielding_supported") is not False:
            raise SystemExit(f"wake shielding must be unsupported for active-only state {state}")
        if report.get("opening_is_production_interpolation_axis") is not False:
            raise SystemExit(f"opening must be validation-only for {state}")

        extension = float(report["cumulative_crack_path_extension_m"])
        rounded_extension = round(extension, 14)
        if rounded_extension in seen_extensions:
            raise SystemExit(f"duplicate cumulative crack extension for {state}: {extension}")
        seen_extensions.add(rounded_extension)

        generated = report.get("generated_load_cases")
        if not isinstance(generated, list) or not generated:
            raise SystemExit(f"generated_load_cases missing from {report_path}")
        reference_found = False
        rewritten_cases: list[dict[str, Any]] = []
        for case in generated:
            if not isinstance(case, dict):
                raise SystemExit(f"invalid generated load case in {report_path}")
            scale = float(case["load_scale"])
            local_response = _local_response_for_scale(state_dir, scale).resolve()
            local_audit = local_response.with_suffix(".audit.json")
            for required in (local_response, local_audit):
                if not required.is_file():
                    raise SystemExit(f"local load case artifact is missing: {required}")
            local_case = dict(case)
            local_case["responses"] = str(local_response)
            local_case["response_audit"] = str(local_audit)
            rewritten_cases.append(local_case)
            if abs(scale - 1.0) <= 1.0e-12:
                reference_found = True
                if local_response != response_path.resolve():
                    raise SystemExit(f"reference response mismatch for {state}")
        if not reference_found:
            raise SystemExit(f"no load_scale=1 reference response in {report_path}")

        rewritten = dict(report)
        rewritten["generated_load_cases"] = rewritten_cases
        rewritten["source_report_path"] = str(report_path.resolve())
        rewritten["source_report_sha256"] = _sha256(report_path)
        destination_dir = assembly_root / state
        destination_dir.mkdir(parents=True, exist_ok=True)
        rewritten_path = destination_dir / REPORT_NAME
        rewritten_path.write_text(json.dumps(rewritten, indent=2, sort_keys=True) + "\n")

        rewritten_reports.append(rewritten_path)
        responses.append(response_path.resolve())
        audit_rows.append(
            {
                "state_id": state,
                "cumulative_crack_path_extension_m": extension,
                "source_report": str(report_path.resolve()),
                "source_report_sha256": _sha256(report_path),
                "reference_response": str(response_path.resolve()),
                "reference_response_sha256": _sha256(response_path),
                "reference_response_audit_sha256": _sha256(response_audit),
                "maximum_within_load_relative_spread": report.get("checks", {}).get(
                    "maximum_within_load_relative_spread"
                ),
                "maximum_relative_load_variation": report.get("checks", {}).get(
                    "maximum_relative_load_variation"
                ),
            }
        )

    if len(seen_extensions) < 2:
        raise SystemExit("active-only atlas requires at least two crack extensions")
    return responses, rewritten_reports, audit_rows


def _run_source_builder(
    responses: list[Path],
    reports: list[Path],
    normalization: Path,
    review: Path,
    source_out: Path,
) -> None:
    command = [sys.executable, str(SOURCE_BUILDER)]
    for response in responses:
        command.extend(["--responses", str(response)])
    for report in reports:
        command.extend(["--load-invariance", str(report)])
    command.extend(
        [
            "--normalization",
            str(normalization),
            "--out",
            str(source_out),
            "--independent-review",
            str(review),
            "--authorize-production-parameterization",
        ]
    )
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "v10.2.13 reviewed source build failed:\n"
            + completed.stdout
            + completed.stderr
        )
    if completed.stdout:
        print(completed.stdout, end="")


def _array_is_zero(value: Any, tolerance: float = 0.0) -> bool:
    if isinstance(value, list):
        return all(_array_is_zero(item, tolerance) for item in value)
    try:
        return abs(float(value)) <= tolerance
    except (TypeError, ValueError):
        return False


def _promote(
    source_path: Path,
    final_path: Path,
    source_root: Path,
    engine_config: Path,
    normalization: Path,
    review: Path,
    audit_rows: list[dict[str, Any]],
) -> None:
    payload = _json(source_path)
    if payload.get("production_parameterization_allowed") is not True:
        raise SystemExit("reviewed v10.2.13 source is not production-authorized")
    states = payload.get("states")
    if not isinstance(states, list) or len(states) < 2:
        raise SystemExit("reviewed source does not contain a valid state family")
    for row in states:
        if not isinstance(row, dict):
            raise SystemExit("invalid state row in reviewed source")
        for key in (
            "wake_kernel_I_Pa_sqrt_m_per_signed_line",
            "wake_kernel_II_Pa_sqrt_m_per_signed_line",
        ):
            if not _array_is_zero(row.get(key)):
                raise SystemExit(
                    f"refusing active-only promotion because {key} is nonzero in "
                    f"state {row.get('state_id')!r}"
                )

    source_schema = payload.get("schema")
    payload.update(
        {
            "schema": V10214_SCHEMA,
            "production_parameterization_allowed": True,
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
            "promotion": {
                "model_id": MODEL_ID,
                "source_schema": source_schema,
                "source_path": str(source_path.resolve()),
                "source_sha256": _sha256(source_path),
                "load_invariance_root": str(source_root.resolve()),
                "engine_config": str(engine_config.resolve()),
                "engine_config_sha256": _sha256(engine_config),
                "normalization": str(normalization.resolve()),
                "normalization_sha256": _sha256(normalization),
                "independent_review": str(review.resolve()),
                "independent_review_sha256": _sha256(review),
                "automatic_authorization": False,
                "mechanics_rerun_performed": False,
                "input_states": audit_rows,
            },
        }
    )

    candidate = final_path.with_suffix(final_path.suffix + ".candidate")
    candidate.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    try:
        family = ActiveOnlySigned2DShieldingKernelFamily.from_json(candidate)
        if family.metadata.get("production_parameterization_allowed") is not True:
            raise SystemExit("promoted family lost production authorization metadata")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        candidate.replace(final_path)
    finally:
        candidate.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-invariance-root", type=Path, required=True)
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument("--independent-review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-source-spacing-b", type=float, default=10.0)
    parser.add_argument("--maximum-source-spacing-b", type=float, default=100.0)
    parser.add_argument("--source-region-length-um", type=float)
    args = parser.parse_args()

    out = args.out.expanduser().resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing atlas: {out}")
    engine_config = args.engine_config.expanduser().resolve()
    review = args.independent_review.expanduser().resolve()
    for required in (engine_config, review, SOURCE_BUILDER):
        if not required.is_file():
            raise SystemExit(f"required input is missing: {required}")

    source_root = _find_state_root(args.load_invariance_root)
    assembly_root = out.parent / f"{out.stem}_assembly_inputs"
    if assembly_root.exists():
        shutil.rmtree(assembly_root)
    assembly_root.mkdir(parents=True)

    responses, reports, audit_rows = _prepare_reports(source_root, assembly_root)
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
    normalization.write_text(
        json.dumps(normalization_payload, indent=2, sort_keys=True) + "\n"
    )

    source_out = out.parent / f"{out.stem}_authorized_v10_2_13_source.json"
    if source_out.exists():
        source_out.unlink()
    _run_source_builder(responses, reports, normalization, review, source_out)
    _promote(
        source_out,
        out,
        source_root,
        engine_config,
        normalization,
        review,
        audit_rows,
    )

    family = ActiveOnlySigned2DShieldingKernelFamily.from_json(out)
    print(
        json.dumps(
            {
                "out": str(out),
                "schema": family.metadata.get("schema"),
                "states": len(family.states),
                "state_ids": [state.state_id for state in family.states],
                "production_parameterization_allowed": family.metadata.get(
                    "production_parameterization_allowed"
                ),
                "mechanics_rerun_performed": False,
                "normalization": str(normalization),
                "authorized_v10_2_13_source": str(source_out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
