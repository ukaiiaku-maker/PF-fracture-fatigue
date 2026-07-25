#!/usr/bin/env python3
"""Assemble an extended v10.2.14 active-only atlas from arbitrary completed states.

This builder does not rerun FEM mechanics and does not extrapolate kernel data. It
accepts one load-scale=1 station-response CSV and one passed frozen-geometry
load-invariance report per mechanically captured crack-extension state, delegates
all inherited mechanical gates to the audited v10.2.13/v10.2.14 builders, and
promotes the result through the exact-endpoint v10.2.14 campaign path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V2_PATH = ROOT / "scripts" / "build_v10_2_14_campaign_ready_active_only_atlas_v2.py"
SPEC = importlib.util.spec_from_file_location("v10214_campaign_builder_v2", V2_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load v10.2.14 campaign builder from {V2_PATH}")
V2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V2)
BASE = V2.BASE

from arrhenius_fracture.signed_kernel_family_v10214 import (  # noqa: E402
    ActiveOnlySigned2DShieldingKernelFamily,
    SCHEMA as V10214_SCHEMA,
)

MODEL_ID = "v10.2.27_extended_active_only_signed_atlas_assembler"
REPORT_SCHEMA = "v10.2.14_active_frozen_geometry_load_invariance"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state_rows(
    responses: list[Path],
    reports: list[Path],
) -> list[dict[str, Any]]:
    if len(responses) != len(reports):
        raise ValueError(
            f"responses/reports count mismatch: {len(responses)} != {len(reports)}"
        )
    if len(reports) < 2:
        raise ValueError("extended atlas requires at least two measured states")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_extensions: set[float] = set()

    for response_raw, report_raw in zip(responses, reports):
        response = response_raw.expanduser().resolve()
        report = report_raw.expanduser().resolve()
        if not response.is_file():
            raise FileNotFoundError(response)
        if not response.with_suffix(".audit.json").is_file():
            raise FileNotFoundError(response.with_suffix(".audit.json"))
        if not report.is_file():
            raise FileNotFoundError(report)

        payload = json.loads(report.read_text())
        if payload.get("schema") != REPORT_SCHEMA:
            raise ValueError(
                f"load-invariance report schema mismatch for {report}: "
                f"{payload.get('schema')!r}"
            )
        if payload.get("load_invariance_passed") is not True:
            raise ValueError(f"load invariance did not pass: {report}")
        if payload.get("active_kernel_mechanically_measured") is not True:
            raise ValueError(f"active kernel is not mechanically measured: {report}")
        if payload.get("wake_kernel_mechanically_measured") is not False:
            raise ValueError(f"wake kernel must remain unmeasured: {report}")
        if payload.get("wake_shielding_supported") is not False:
            raise ValueError(f"wake shielding must remain unsupported: {report}")

        state_id = str(payload.get("parent_state_id", "")).strip()
        extension = float(payload["cumulative_crack_path_extension_m"])
        if not state_id or state_id in seen_ids:
            raise ValueError(f"invalid or duplicate state id {state_id!r}")
        rounded_extension = round(extension, 14)
        if rounded_extension in seen_extensions:
            raise ValueError(
                f"duplicate cumulative crack-path extension {extension:.12g} m"
            )

        reference = None
        for case in payload.get("generated_load_cases", []):
            if abs(float(case.get("load_scale", -1.0)) - 1.0) <= 1.0e-12:
                reference = Path(case["responses"]).expanduser().resolve()
                break
        if reference is None:
            raise ValueError(f"no load_scale=1 response in {report}")
        if reference != response:
            raise ValueError(
                f"response/report mismatch for {state_id}: {response} != {reference}"
            )

        seen_ids.add(state_id)
        seen_extensions.add(rounded_extension)
        rows.append(
            {
                "state_id": state_id,
                "cumulative_crack_path_extension_m": extension,
                "response": str(response),
                "response_sha256": _sha256(response),
                "response_audit_sha256": _sha256(response.with_suffix(".audit.json")),
                "load_invariance_report": str(report),
                "load_invariance_report_sha256": _sha256(report),
                "maximum_relative_load_variation": payload.get("checks", {}).get(
                    "maximum_relative_load_variation"
                ),
                "maximum_within_load_relative_spread": payload.get("checks", {}).get(
                    "maximum_within_load_relative_spread"
                ),
            }
        )

    rows.sort(key=lambda row: float(row["cumulative_crack_path_extension_m"]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, action="append", required=True)
    parser.add_argument("--load-invariance", type=Path, action="append", required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-max-extension-um", type=float, default=1200.0)
    args = parser.parse_args()

    out = args.out.expanduser().resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing atlas: {out}")
    normalization = args.normalization.expanduser().resolve()
    if not normalization.is_file():
        raise SystemExit(f"mechanics normalization is missing: {normalization}")

    try:
        states = load_state_rows(args.responses, args.load_invariance)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    maximum_um = 1.0e6 * max(
        float(row["cumulative_crack_path_extension_m"]) for row in states
    )
    if maximum_um + 1.0e-9 < float(args.minimum_max_extension_um):
        raise SystemExit(
            "measured state family is too short: "
            f"maximum={maximum_um:.9g} um, "
            f"required={float(args.minimum_max_extension_um):.9g} um"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    source_out = out.parent / f"{out.stem}_v10_2_13_mechanical_source.json"
    source_out.unlink(missing_ok=True)

    responses = [Path(row["response"]) for row in states]
    reports = [Path(row["load_invariance_report"]) for row in states]
    BASE._run_review_source(responses, reports, normalization, source_out)
    BASE._promote(
        source_out,
        out,
        {
            "extended_atlas_assembler_model_id": MODEL_ID,
            "arbitrary_measured_state_list": True,
            "hardcoded_E000_E200_E500_E800_state_set_used": False,
            "minimum_required_max_extension_um": float(
                args.minimum_max_extension_um
            ),
            "measured_max_extension_um": maximum_um,
            "mechanics_normalization": str(normalization),
            "mechanics_normalization_sha256": _sha256(normalization),
            "input_states": states,
        },
    )

    family = ActiveOnlySigned2DShieldingKernelFamily.from_json(out)
    family_extensions = sorted(
        float(state.coordinates[2]) for state in family.states
    )
    payload = {
        "schema": MODEL_ID,
        "out": str(out),
        "out_sha256": _sha256(out),
        "family_schema": family.metadata.get("schema"),
        "expected_family_schema": V10214_SCHEMA,
        "state_count": len(family.states),
        "state_ids": [state.state_id for state in family.states],
        "crack_path_extensions_um": [
            1.0e6 * value for value in family_extensions
        ],
        "maximum_crack_path_extension_um": 1.0e6 * family_extensions[-1],
        "production_parameterization_allowed": (
            family.metadata.get("production_parameterization_allowed") is True
        ),
        "active_kernel_mechanically_measured": (
            family.metadata.get("active_kernel_mechanically_measured") is True
        ),
        "wake_shielding_supported": (
            family.metadata.get("wake_shielding_supported") is True
        ),
    }
    audit = out.with_name(f"{out.stem}_extended_assembly_audit.json")
    audit.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if payload["family_schema"] != V10214_SCHEMA:
        raise SystemExit("extended atlas lost the v10.2.14 schema")
    if payload["production_parameterization_allowed"] is not True:
        raise SystemExit("extended atlas is not production-authorized")
    if payload["active_kernel_mechanically_measured"] is not True:
        raise SystemExit("extended atlas lost mechanically measured active-kernel status")
    if payload["wake_shielding_supported"] is not False:
        raise SystemExit("extended atlas unexpectedly enabled wake shielding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
