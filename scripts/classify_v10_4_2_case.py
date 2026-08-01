#!/usr/bin/env python3
"""Classify a v10.4.2 case as crack-growth complete or plastic-flow terminal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from classify_v10_2_15_stage3_case import classify as classify_stage3


PLASTIC_STATUS = "plastic_flow_no_sharp_fracture"


def _load_json(path: Path):
    return json.loads(path.read_text())


def classify(case_root: Path, target_extension_um: float) -> dict:
    root = case_root.expanduser().resolve()
    marker = root / "PLASTIC_FLOW"
    audit_path = root / "plastic_flow_terminal_audit.json"
    if marker.is_file() or audit_path.is_file():
        if not marker.is_file() or not audit_path.is_file():
            raise ValueError("plastic terminal requires both PLASTIC_FLOW and audit")
        audit = _load_json(audit_path)
        if audit.get("schema") != "v10.4.2_plastic_flow_terminal_audit_v1":
            raise ValueError("unexpected plastic-flow terminal audit schema")
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
        return {
            "schema": "v10.4.2_case_status_v1",
            "case_root": str(root),
            "temperature_K": float(audit["temperature_K"]),
            "target_extension_um": float(target_extension_um),
            "projected_extension_um": 0.0,
            "first_passage_recorded": False,
            "Kc_first_MPa_sqrt_m": None,
            "status": PLASTIC_STATUS,
            "complete": False,
            "target_extension_complete": False,
            "terminal": True,
            "campaign_terminal": True,
            "sharp_fracture_occurred": False,
            "ductile_fracture_simulated": False,
            "failure_regime": "bulk_plastic_flow",
            "J_pl_diss_J_per_m2": audit.get("J_pl_diss_J_per_m2"),
            "K_pl_equivalent_MPa_sqrt_m": audit.get(
                "K_pl_equivalent_MPa_sqrt_m"
            ),
            "J_contour_shielding_J_per_m2": audit.get(
                "J_contour_shielding_J_per_m2"
            ),
            "plastic_flow_terminal_audit": str(audit_path),
        }

    payload = classify_stage3(root, target_extension_um)
    payload["schema"] = "v10.4.2_case_status_v1"
    payload["target_extension_complete"] = payload.get("complete") is True
    payload["terminal"] = payload.get("complete") is True
    payload["campaign_terminal"] = payload.get("complete") is True
    payload["sharp_fracture_occurred"] = payload.get("first_passage_recorded") is True
    payload["ductile_fracture_simulated"] = False
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
    print(json.dumps({
        key: payload.get(key)
        for key in (
            "status",
            "temperature_K",
            "projected_extension_um",
            "Kc_first_MPa_sqrt_m",
            "J_pl_diss_J_per_m2",
            "J_contour_shielding_J_per_m2",
            "campaign_terminal",
        )
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
