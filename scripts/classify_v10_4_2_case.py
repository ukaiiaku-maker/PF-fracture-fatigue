#!/usr/bin/env python3
"""Classify a v10.4.2/v10.4.3 case as fracture or plastic-flow terminal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from classify_v10_2_15_stage3_case import classify as classify_stage3


PLASTIC_STATUS = "plastic_flow_no_sharp_fracture"
PLASTIC_AUDIT_SCHEMAS = {
    "v10.4.2_plastic_flow_terminal_audit_v1",
    "v10.4.3_plastic_dominance_terminal_audit_v1",
}


def _load_json(path: Path):
    return json.loads(path.read_text())


def _v1043_energy_balance_pass(audit: dict) -> bool:
    explicit = audit.get("energy_balance_pass")
    if explicit is not None:
        return explicit is True
    criteria = audit.get("criteria")
    if isinstance(criteria, dict):
        return criteria.get("energy_balance_bounded") is True
    window_error = audit.get("window_energy_balance_relative_error")
    cumulative_error = audit.get("cumulative_energy_balance_relative_error")
    tolerance = (audit.get("thresholds") or {}).get(
        "energy_balance_relative_tolerance",
        0.01,
    )
    if window_error is None or cumulative_error is None:
        return False
    return max(float(window_error), float(cumulative_error)) <= float(tolerance)


def classify(case_root: Path, target_extension_um: float) -> dict:
    root = case_root.expanduser().resolve()
    marker = root / "PLASTIC_FLOW"
    audit_path = root / "plastic_flow_terminal_audit.json"
    if marker.is_file() or audit_path.is_file():
        if not marker.is_file() or not audit_path.is_file():
            raise ValueError("plastic terminal requires both PLASTIC_FLOW and audit")
        audit = _load_json(audit_path)
        schema = audit.get("schema")
        if schema not in PLASTIC_AUDIT_SCHEMAS:
            raise ValueError(f"unexpected plastic-flow terminal audit schema: {schema}")
        if audit.get("classification") != PLASTIC_STATUS:
            raise ValueError("plastic-flow audit classification mismatch")
        if audit.get("terminal") is not True:
            raise ValueError("plastic-flow audit is not terminal")
        if audit.get("sharp_fracture_occurred") is not False:
            raise ValueError("plastic terminal cannot contain sharp fracture")
        if audit.get("plastic_work_enters_fracture_measure") is not False:
            raise ValueError("plastic work must not enter fracture measure")
        if audit.get("plastic_work_enters_cleavage_hazard") is not False:
            raise ValueError("plastic work must not enter cleavage hazard")
        if audit.get("contour_shielding_enters_fracture_hazard") is not False:
            raise ValueError("contour shielding must remain diagnostic")
        is_v1043 = schema.startswith("v10.4.3_")
        if is_v1043:
            if audit.get("plastic_terminal_is_model_limit_censor") is not True:
                raise ValueError(
                    "v10.4.3 terminal must be identified as a model-limit censor"
                )
            if audit.get("future_fracture_beyond_terminal_resolved") is not False:
                raise ValueError(
                    "v10.4.3 may not claim post-terminal fracture resolution"
                )
            if not _v1043_energy_balance_pass(audit):
                raise ValueError(
                    "v10.4.3 terminal requires a passing energy balance"
                )
        energy_error = max(
            float(audit.get("window_energy_balance_relative_error", 0.0) or 0.0),
            float(
                audit.get("cumulative_energy_balance_relative_error", 0.0)
                or 0.0
            ),
            float(audit.get("energy_balance_relative_error", 0.0) or 0.0),
        )
        return {
            "schema": (
                "v10.4.3_case_status_v1" if is_v1043
                else "v10.4.2_case_status_v1"
            ),
            "case_root": str(root),
            "temperature_K": float(audit["temperature_K"]),
            "target_extension_um": float(target_extension_um),
            "projected_extension_um": float(
                audit.get("projected_extension_um", 0.0) or 0.0
            ),
            "first_passage_recorded": False,
            "Kc_first_MPa_sqrt_m": None,
            "status": PLASTIC_STATUS,
            "complete": False,
            "target_extension_complete": False,
            "terminal": True,
            "campaign_terminal": True,
            "sharp_fracture_occurred": False,
            "ductile_fracture_simulated": False,
            "failure_regime": audit.get(
                "failure_regime",
                "bulk_plastic_dominance_model_limit" if is_v1043
                else "bulk_plastic_flow",
            ),
            "plastic_terminal_is_model_limit_censor": bool(
                audit.get("plastic_terminal_is_model_limit_censor", False)
            ),
            "interpretation": audit.get("interpretation"),
            "future_fracture_beyond_terminal_resolved": audit.get(
                "future_fracture_beyond_terminal_resolved"
            ),
            "J_pl_diss_J_per_m2": audit.get("J_pl_diss_J_per_m2"),
            "K_pl_equivalent_MPa_sqrt_m": audit.get(
                "K_pl_equivalent_MPa_sqrt_m"
            ),
            "J_contour_shielding_J_per_m2": audit.get(
                "J_contour_shielding_J_per_m2"
            ),
            "plastic_accommodation_ratio_median": audit.get(
                "plastic_accommodation_ratio_median"
            ),
            "normalized_tangent_stiffness": audit.get(
                "normalized_tangent_stiffness"
            ),
            "active_plastic_area_fraction_median": audit.get(
                "active_plastic_area_fraction_median"
            ),
            "energy_balance_relative_error": energy_error,
            "plastic_flow_terminal_audit": str(audit_path),
        }

    payload = classify_stage3(root, target_extension_um)
    payload["schema"] = "v10.4.3_case_status_v1"
    payload["target_extension_complete"] = payload.get("complete") is True
    payload["terminal"] = payload.get("complete") is True
    payload["campaign_terminal"] = payload.get("complete") is True
    payload["sharp_fracture_occurred"] = (
        payload.get("first_passage_recorded") is True
    )
    payload["ductile_fracture_simulated"] = False
    payload["plastic_terminal_is_model_limit_censor"] = False
    return payload


def write_status(case_root: Path, payload: dict) -> Path:
    root = case_root.expanduser().resolve()
    output = root / "stage3_case_status.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for name in ("COMPLETE", "INCOMPLETE", "CENSORED", "PLASTIC_FLOW"):
        path = root / name
        if path.exists() and not (
            name == "PLASTIC_FLOW"
            and payload.get("status") == PLASTIC_STATUS
        ):
            path.unlink()

    if payload.get("status") == PLASTIC_STATUS:
        marker = root / "PLASTIC_FLOW"
    elif payload.get("status") == "complete_target_extension":
        marker = root / "COMPLETE"
    elif payload.get("status") == "incomplete_after_first_passage":
        marker = root / "INCOMPLETE"
    else:
        marker = root / "CENSORED"
    marker.write_text(str(payload["status"]) + "\n")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", required=True, type=Path)
    parser.add_argument("--target-extension-um", required=True, type=float)
    args = parser.parse_args()
    payload = classify(args.case_root, args.target_extension_um)
    write_status(args.case_root, payload)
    print(
        json.dumps(
            {
                key: payload.get(key)
                for key in (
                    "status",
                    "temperature_K",
                    "projected_extension_um",
                    "Kc_first_MPa_sqrt_m",
                    "J_pl_diss_J_per_m2",
                    "J_contour_shielding_J_per_m2",
                    "plastic_accommodation_ratio_median",
                    "campaign_terminal",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
