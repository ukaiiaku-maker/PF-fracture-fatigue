#!/usr/bin/env python3
"""Acceptance audit for all disabled campaign bindings and domain rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _value(args: list[str], flag: str):
    for index, token in enumerate(args):
        if token == flag and index + 1 < len(args):
            return args[index + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def _binding_errors(row: dict) -> list[str]:
    errors = []
    args = list(row.get("native_args", []))
    execution = row.get("execution", {})
    if args != list(execution.get("native_args", [])):
        errors.append("execution/native_args mismatch")
    fitted = row.get("block") == "P9" and row.get("cohesive_law", {}).get("fit")
    if fitted:
        q = row.get("qualification_execution", {})
        command = list(q.get("launcher", []))
        checks = {
            "material_class": row["family"],
            "opening_model": row["cohesive_law"]["kind"],
            "temperature_K": f"{float(row['temperature_K']):g}",
            "temperature_policy": row["cohesive_law"]["temperature_policy"],
            "constraint": row["mechanics_state"]["plane_state"],
        }
        flags = {
            "material_class": "--material-class",
            "opening_model": "--opening-model",
            "temperature_K": "--temperature-K",
            "temperature_policy": "--temperature-policy",
            "constraint": "--constraint",
        }
        for name, expected in checks.items():
            if _value(command, flags[name]) != expected:
                errors.append(f"qualification {name} is not translated")
        if q.get("production_case_substitution") is not False:
            errors.append("constructed specimen is not explicitly excluded as production substitution")
        return errors
    if row.get("command_status") == "RESOLVED_DISABLED" and not args:
        errors.append("resolved row has empty native arguments")
        return errors
    if not args:
        return errors
    expected_temperature = f"{float(row['temperature_K']):g}"
    actual_temperature = _value(args, "--temperatures") or _value(args, "--temperature-K")
    if actual_temperature != expected_temperature:
        errors.append("temperature_K is not translated")
    if (row.get("parameter_option")
            and _value(args, "--parameter-option") != row["parameter_option"]):
        errors.append("parameter_option is not translated")
    if row.get("parameter_registry"):
        actual = _value(args, "--parameter-registry") or _value(args, "--material-registry")
        if actual is None or not actual.endswith(row["parameter_registry"]):
            errors.append("parameter_registry is not translated")
    if row.get("block") == "P7":
        endpoint = row["physical_endpoint"]
        expected_target_um = float(endpoint["target_extension_m"]) * 1.0e6
        expected_exposure_s = float(endpoint["no_event_exposure_ceiling_s"])
        if float(_value(args, "--target-extension-um")) != expected_target_um:
            errors.append("P7 target extension is not translated")
        if float(_value(args, "--exposure-limit-s")) != expected_exposure_s:
            errors.append("P7 exposure censor is not translated")
        if "--steps" in args:
            errors.append("P7 complete-case request is only a compact fixed-step interval")
        qualification = row.get("qualification_execution", {})
        qargs = list(qualification.get("launcher", []))
        if _value(qargs, "--steps") != "1" or _value(qargs, "--duration-s") != "0.2":
            errors.append("P7 compact qualification interval is absent")
        if qualification.get("production_case_substitution") is not False:
            errors.append("P7 compact interval is not excluded as production substitution")
    if row.get("backend") == "full_fem_czm":
        if _value(args, "--constraint") != row["mechanics_state"]["plane_state"]:
            errors.append("FEM constraint is not translated")
        expected_angle = float(row["mechanics_state"].get("grip_mode_angle_deg", 0.0))
        actual_angle = float(_value(args, "--grip-mode-angle-deg") or 0.0)
        if actual_angle != expected_angle:
            errors.append("FEM signed grip mixity is not translated")
        expected_eps = row["mechanics_state"].get("eps_zz")
        expected_sigma = row["mechanics_state"].get("sigma_zz_Pa")
        actual_eps = _value(args, "--eps-zz")
        actual_sigma = _value(args, "--sigma-zz-Pa")
        if row["mechanics_state"]["plane_state"] == "generalized_plane_strain" and expected_eps is not None and (
            actual_eps is None or float(actual_eps) != float(expected_eps)
        ):
            errors.append("FEM eps_zz control is not translated")
        if row["mechanics_state"]["plane_state"] == "generalized_plane_strain" and expected_sigma is not None and (
            actual_sigma is None or float(actual_sigma) != float(expected_sigma)
        ):
            errors.append("FEM sigma_zz control is not translated")
        expected = "active_tip" if row["arm"] == "accommodation" else "opening_only"
        if _value(args, "--accommodation-mode") != expected:
            errors.append("FEM accommodation arm is not translated")
    if row.get("orientation_deg") is not None and "--crystal-theta-deg" in args:
        if float(_value(args, "--crystal-theta-deg")) != float(row["orientation_deg"]):
            errors.append("orientation is not translated")
    return errors


def audit(path: Path) -> dict:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    per_case = {row["case_id"]: _binding_errors(row) for row in rows}
    per_case = {key: value for key, value in per_case.items() if value}
    blocked = [row for row in rows if row["readiness"]["material_domain"]
               == "BLOCKED_RAW_CONSTITUTIVE_SCALE_NONPOSITIVE"]
    boundary = [row for row in rows if row["readiness"]["material_domain"]
                == "QUALIFICATION_REQUIRED_NEAR_SCALE_ZERO"]
    fitted = [row for row in rows if row.get("block") == "P9"
              and row.get("cohesive_law", {}).get("fit") is not None]
    commands = [row for row in rows if row.get("native_args")]
    return {
        "schema": "campaign-v3-acceptance-audit/1",
        "campaign": str(path.resolve()),
        "campaign_sha256": hashlib.sha256(raw).hexdigest(),
        "case_count": len(rows),
        "native_command_count": len(commands),
        "fitted_bilinear_production_blocked_count": len(fitted),
        "fitted_bilinear_qualification_request_count": sum(
            bool(row.get("qualification_execution", {}).get("launcher")) for row in fitted
        ),
        "raw_domain_blocked_rows": [{
            "case_id": row["case_id"], "block": row["block"],
            "temperature_K": row["temperature_K"], "family": row["family"],
            "parameter_option": row["parameter_option"],
        } for row in blocked],
        "near_boundary_rows": [{
            "case_id": row["case_id"], "block": row["block"],
            "temperature_K": row["temperature_K"], "family": row["family"],
            "parameter_option": row["parameter_option"],
        } for row in boundary],
        "raw_blocked_near_boundary_overlap_case_ids": sorted(
            {row["case_id"] for row in blocked} & {row["case_id"] for row in boundary}
        ),
        "binding_errors": per_case,
        "checks": {
            "case_count_699": len(rows) == 699,
            "native_command_count_699": len(commands) == 699,
            "five_selective_constraint_mixity_rows": sum(
                bool(row.get("selective_native_comparison")) for row in rows
            ) == 5,
            "exact_raw_domain_blocked_count_3": len(blocked) == 3,
            "exact_near_boundary_count_20": len(boundary) == 20,
            "all_36_fitted_rows_have_qualification_requests": len(fitted) == 36 and all(
                row.get("qualification_execution", {}).get("launcher") for row in fitted
            ),
            "all_static_argument_bindings_resolve": not per_case,
            "production_disabled_all_rows": all(
                row.get("production_execution_authorized") is False for row in rows
            ),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", type=Path,
        default=ROOT / "common_framework_readiness/PROSPECTIVE_CAMPAIGN_V3.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit(args.campaign)
    report["status"] = "PASS" if all(report["checks"].values()) else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
