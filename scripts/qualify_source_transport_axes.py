#!/usr/bin/env python3
"""Audit active source support and distinguish native transport capabilities."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TIP_ONLY = ROOT / "integration_evidence/current_science_pilots_6abbf10/pf/DBTT_1200K/accommodation_enabled"
GEOGRAPHY = ROOT / "integration_evidence/native_source_geography_9d9f3edf"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(root: Path) -> tuple[Path, list[dict[str, str]]]:
    paths = list(root.glob("steps_*K.csv"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected exactly one step history in {root}")
    with paths[0].open(newline="") as stream:
        return paths[0], list(csv.DictReader(stream))


def _maximum(rows: list[dict[str, str]], name: str) -> float:
    return max(float(row[name]) for row in rows if row.get(name) not in (None, ""))


def _support_case(label: str, root: Path) -> dict[str, Any]:
    steps_path, rows = _rows(root)
    max_tip = _maximum(rows, "N_em")
    max_bulk_work = _maximum(rows, "W_bulk_plastic_cumulative_J_per_m")
    audit_path = root / "v10_4_bulk_coupled_model_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else None
    expected = {
        "tip_only": (True, False),
        "bulk_uniform": (False, True),
        "tip_plus_bulk": (True, True),
    }[label]
    observed = (max_tip > 0.0, max_bulk_work > 0.0)
    if observed != expected:
        raise ValueError(f"{label}: observed active owners {observed}, expected {expected}")
    if audit is not None:
        if audit["source_geography"] != label:
            raise ValueError(f"{label}: geography audit mismatch")
        if audit["tip_emission_enabled"] != expected[0]:
            raise ValueError(f"{label}: tip-emission binding mismatch")
    files = [steps_path]
    if audit_path.is_file():
        files.append(audit_path)
    return {
        "status": "PASS_ACTIVE_NATIVE_SUPPORT",
        "root": str(root.relative_to(ROOT)), "accepted_steps": len(rows),
        "maximum_tip_mobile_plus_retained_count": max_tip,
        "maximum_bulk_plastic_work_J_per_m": max_bulk_work,
        "tip_process_active": observed[0], "bulk_process_active": observed[1],
        "full_mechanics_history_present": all(
            row.get(name) not in (None, "") for row in rows
            for name in ("Ftop_N", "KJ_Pa_sqrtm", "U_elastic_J_per_m")
        ),
        "native_capabilities": None if audit is None else {
            "bulk_density_transport": audit["bulk_density_transport"],
            "direct_tip_bulk_population_exchange": audit["direct_tip_bulk_population_exchange"],
            "bulk_net_slip_model": audit["bulk_net_slip_model"],
            "zero_stress_net_plastic_rate_exactly_zero": audit["zero_stress_net_plastic_rate_exactly_zero"],
        },
        "sha256": {str(path.relative_to(ROOT)): sha256(path) for path in files},
    }


def qualify() -> dict[str, Any]:
    support = {
        "tip_only": _support_case("tip_only", TIP_ONLY),
        "bulk_uniform": _support_case(
            "bulk_uniform", GEOGRAPHY / "bulk_uniform/uninterrupted_to_step2"
        ),
        "tip_plus_bulk": _support_case(
            "tip_plus_bulk", GEOGRAPHY / "tip_plus_bulk/uninterrupted_to_step2"
        ),
    }
    for geography in ("bulk_uniform", "tip_plus_bulk"):
        comparison = json.loads(
            (GEOGRAPHY / geography / "restart_comparison.json").read_text()
        )
        if comparison["status"] != "PASS" or not comparison["complete_terminal_state_equal"]:
            raise ValueError(f"{geography}: restart comparison failed")
        support[geography]["fresh_process_restart"] = "PASS_COMPLETE_TERMINAL_STATE_EQUAL"
    return {
        "schema": "source-transport-axis-qualification/1",
        "decision": "COMPLETE_GENERIC_NATIVE_OPTIONS_PHYSICAL_TRANSFER_SELECTION_PENDING",
        "production_execution_authorized": False,
        "support_qualification": support,
        "capability_matrix": [
            {
                "owner": "tip moving process zone", "transport": "signed Peierls field plus Taylor exchange, wake/return and moving-frame advection",
                "population_transfer": "within native tip/wake state", "executable": True,
            },
            {
                "owner": "V10.4 bulk_uniform or tip_plus_bulk", "transport": "local full-field Peierls-Taylor constitutive evolution",
                "bulk_density_transport": False, "direct_tip_bulk_population_exchange": False,
                "executable": True,
            },
            {
                "owner": "V5 cavity vacancy field", "transport": "positive conservative vacancy transport",
                "classification": "distinct vacancy/cavity capability; not dislocation-density transport or a tip/bulk transfer coefficient",
                "executable": True,
            },
            {
                "owner": "conservative tip/bulk dislocation population transfer",
                "transport": "parameterized positive conservative line-content map",
                "classification": "GENERIC_OPERATOR_QUALIFIED_PHYSICAL_COEFFICIENT_UNAVAILABLE",
                "executable": True,"production_executable":False,
                "physical_transfer_coefficient":None,
            },
        ],
        "planned_interventions": {
            "native": "EXECUTABLE",
            "retained_shielding_off": "QUALIFIED_ORTHOGONAL_NATIVE_CONSUMER",
            "source_backstress_off": "QUALIFIED_ORTHOGONAL_NATIVE_CONSUMER",
            "blunting_fixed": "QUALIFIED_DEVELOPED_RADIUS_CONSUMER",
            "transport_retention_off": "QUALIFIED_SOURCE_PRESERVING_NATIVE_CONSUMER",
            "history_reset_reequilibrated": "QUALIFIED_COUNTERFACTUAL_RESET_AND_NATIVE_REEQUILIBRATION",
        },
        "interpretation": (
            "Source support and transport are now reported as independent axes. "
            "The active support evidence cannot be promoted into nonexistent bulk "
            "density transport or conservative tip/bulk population exchange."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = qualify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"].startswith("COMPLETE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
