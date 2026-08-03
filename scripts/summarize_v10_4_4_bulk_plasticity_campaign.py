#!/usr/bin/env python3
"""Summarize fracture-target and plasticity-dominated campaign terminals."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_summary(case_root: Path) -> dict:
    path = case_root / "summary.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    return payload if isinstance(payload, dict) else {}


def _case_record(case_root: Path, row: dict, theta: float) -> dict:
    status_path = case_root / "stage3_case_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    terminal_path = case_root / "plastic_flow_terminal_audit.json"
    terminal = json.loads(terminal_path.read_text()) if terminal_path.is_file() else {}
    summary = _first_summary(case_root)

    plasticity_marker = case_root / "PLASTICITY_DOMINATED"
    legacy_marker = case_root / "PLASTIC_FLOW"
    plasticity_dominated = plasticity_marker.is_file() or legacy_marker.is_file()
    fracture_complete = (
        status.get("complete") is True
        and status.get("status") != "plasticity_dominated"
        and not plasticity_dominated
    )
    if plasticity_dominated:
        classification = str(
            terminal.get(
                "campaign_classification",
                terminal.get(
                    "classification",
                    plasticity_marker.read_text().strip()
                    if plasticity_marker.is_file()
                    else "plasticity_dominated",
                ),
            )
        )
    elif fracture_complete:
        classification = "fracture_target_reached"
    elif (case_root / "RUN_FAILED").exists():
        classification = "numerical_failure"
    else:
        classification = str(status.get("status", "incomplete"))

    return {
        "option": row["option"],
        "temperature_K": float(row["temperature_K"]),
        "seed": int(row["seed"]),
        "theta_deg": theta,
        "case_root": str(case_root),
        "complete": bool((case_root / "COMPLETE").is_file()),
        "classification": classification,
        "fracture_target_reached": fracture_complete,
        "plasticity_dominated": plasticity_dominated,
        "projected_extension_um": _finite(
            status.get(
                "projected_extension_um",
                summary.get("geometry_projected_extension_m", 0.0) * 1.0e6
                if summary.get("geometry_projected_extension_m") is not None
                else None,
            )
        ),
        "first_passage_K_MPa_sqrt_m": _finite(
            summary.get("Kc_first_MPa_sqrt_m")
        ),
        "J_elastic_positive_J_per_m2": _finite(
            terminal.get("J_elastic_positive_J_per_m2")
        ),
        "J_plastic_dissipation_J_per_m2": _finite(
            terminal.get("J_plastic_dissipation_J_per_m2")
        ),
        "J_apparent_total_J_per_m2": _finite(
            terminal.get("J_apparent_total_J_per_m2")
        ),
        "K_elastic_equivalent_MPa_sqrt_m": _finite(
            terminal.get("K_elastic_equivalent_MPa_sqrt_m")
        ),
        "K_plastic_equivalent_MPa_sqrt_m": _finite(
            terminal.get("K_plastic_equivalent_MPa_sqrt_m")
        ),
        "K_apparent_plasticity_limited_MPa_sqrt_m": _finite(
            terminal.get("K_apparent_plasticity_limited_MPa_sqrt_m")
        ),
        "plastic_work_fraction_window": _finite(
            terminal.get("plastic_work_fraction_window")
        ),
        "elastic_storage_fraction_window": _finite(
            terminal.get("elastic_storage_fraction_window")
        ),
        "normalized_tangent_stiffness": _finite(
            terminal.get("normalized_tangent_stiffness")
        ),
        "terminal_cleavage_action_B": _finite(
            terminal.get("terminal_cleavage_action_B", terminal.get("B_final"))
        ),
        "terminal_nominal_progress": _finite(
            terminal.get("terminal_nominal_progress")
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outroot", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.outroot.resolve()
    manifest_paths = [
        root / "v10_2_27_campaign_manifest.json",
        root / "v10_2_30_hazard_energy_gate_campaign_lock.json",
    ]
    manifest = {}
    for path in manifest_paths:
        if path.is_file():
            manifest.update(json.loads(path.read_text()))
    theta = float(manifest.get("crystal_theta_deg", 0.0))

    seed_map = root / "v10_2_27_case_seed_map.csv"
    if not seed_map.is_file():
        raise SystemExit(f"missing case seed map: {seed_map}")
    with seed_map.open(newline="") as stream:
        cases = list(csv.DictReader(stream))

    records = []
    for row in cases:
        temperature = float(row["temperature_K"])
        case_root = (
            root
            / row["option"]
            / f"T{temperature:g}K_th{theta:g}_seed{int(row['seed'])}"
        )
        records.append(_case_record(case_root, row, theta))

    fields = list(records[0]) if records else []
    csv_path = root / "v10_4_4_bulk_plasticity_campaign_summary.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    complete = sum(bool(row["complete"]) for row in records)
    fracture = sum(bool(row["fracture_target_reached"]) for row in records)
    plasticity = sum(bool(row["plasticity_dominated"]) for row in records)
    failed = sum(row["classification"] == "numerical_failure" for row in records)
    payload = {
        "schema": "v10.4.4_bulk_plasticity_campaign_summary_v1",
        "planned_cases": len(records),
        "complete_cases": complete,
        "fracture_target_cases": fracture,
        "plasticity_dominated_cases": plasticity,
        "numerical_failure_cases": failed,
        "incomplete_cases": len(records) - complete,
        "all_cases_complete": complete == len(records),
        "records": records,
    }
    json_path = root / "v10_4_4_bulk_plasticity_campaign_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(
        "Bulk-plasticity campaign summary: "
        f"planned={len(records)} complete={complete} fracture={fracture} "
        f"plasticity_dominated={plasticity} numerical_failure={failed}"
    )
    print(csv_path)
    print(json_path)
    return 0 if complete == len(records) and failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
