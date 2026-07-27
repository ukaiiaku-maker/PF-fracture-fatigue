#!/usr/bin/env python3
"""Audit a v10.2.30 three-DeltaK energy-gated fatigue qualification."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


CONTROL = "v10_2_30_fixed_deltaK_control.json"
EVENTS = "hazard_energy_gated_events_v10_2_30.json"
GEOMETRY = "stochastic_avalanche_geometry_events.json"
CASE_META = "qualification_case.json"
KINETIC = "kinetic_tip_cell_audit_v101.json"
SUMMARY = "summary.json"


def _load(path: Path, default: Any):
    if not path.is_file():
        return default
    return json.loads(path.read_text())


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _closest_trial(event: dict[str, Any], committed: float) -> dict[str, Any]:
    rows = [row for row in event.get("trial_rows", []) if isinstance(row, dict)]
    if not rows:
        return {}
    return min(
        rows,
        key=lambda row: abs(_finite(row.get("trial_length_m")) - committed),
    )


def _cycles(case: Path) -> float:
    payload = _load(case / KINETIC, {})
    records = payload.get("records", []) if isinstance(payload, dict) else []
    return sum(
        max(_finite(row.get("cycles_consumed")), 0.0)
        for row in records
        if isinstance(row, dict) and row.get("loading_mode") == "cyclic"
    )


def _projected_extension(case: Path, geometry: list[dict[str, Any]]) -> float:
    summary = _load(case / SUMMARY, [])
    if isinstance(summary, list) and summary:
        row = summary[0]
        value = row.get("geometry_projected_extension_m")
        if value is not None:
            return max(_finite(value), 0.0)
    if not geometry:
        return 0.0
    return max(_finite(geometry[-1].get("x1")) - _finite(geometry[0].get("x0")), 0.0)


def summarize_case(control_path: Path) -> dict[str, Any]:
    case = control_path.parent
    control = _load(control_path, {})
    meta = _load(case / CASE_META, {})
    attempts = _load(case / EVENTS, [])
    geometry = _load(case / GEOMETRY, [])
    if not isinstance(attempts, list):
        attempts = []
    if not isinstance(geometry, list):
        geometry = []
    committed = [
        row
        for row in attempts
        if isinstance(row, dict)
        and bool(row.get("inserted", False))
        and _finite(row.get("event_advance_m")) > 0.0
    ]
    arrested = [row for row in attempts if row not in committed]
    errors: list[str] = []

    if control.get("Gc0_athermal_active") is not False:
        errors.append("Gc0_athermal_active is not explicitly false")
    if control.get("independent_fracture_energy_active") is not False:
        errors.append("independent fracture energy is active or unaudited")
    if control.get("fixed_deltaK_exact_within_relative_1e-12") is not True:
        errors.append("fixed-DeltaK control is outside tolerance")
    if control.get("persistent_site_source") is not True:
        errors.append("persistent-site source is not active")
    if control.get("finite_source_inventory") is not False:
        errors.append("finite source inventory is active")
    if control.get("source_refresh") is not False:
        errors.append("source refresh is active")
    if len(geometry) != len(committed):
        errors.append(
            f"geometry/committed-event count mismatch: {len(geometry)} != {len(committed)}"
        )

    proposal = []
    actual = []
    ratios = []
    residuals = []
    gamma = []
    full_proposal = 0
    truncated = 0
    direction_sources: set[str] = set()
    for index, event in enumerate(committed):
        p = _finite(event.get("stochastic_proposed_event_length_m"))
        a = _finite(event.get("event_advance_m"))
        if not p > 0.0 or not a > 0.0:
            errors.append(f"event {index} has nonpositive proposal or commit")
            continue
        if a > p * (1.0 + 1.0e-10):
            errors.append(f"event {index} committed beyond stochastic proposal")
        proposal.append(p)
        actual.append(a)
        ratios.append(a / p)
        g = _finite(event.get("orientation_gamma_relative"), 1.0)
        gamma.append(g)
        if not g > 0.0:
            errors.append(f"event {index} has nonpositive orientation factor")
        direction = event.get("direction_audit", {})
        if isinstance(direction, dict):
            source = direction.get("source")
            if source:
                direction_sources.add(str(source))
        if event.get("athermal_Gc_used") is not False:
            errors.append(f"event {index} used an athermal Gc")
        if event.get("independent_toughness_floor_used") is not False:
            errors.append(f"event {index} used an independent toughness floor")
        if event.get("mesh_resolved_commit_required") is not True:
            errors.append(f"event {index} lacks mesh-resolved commit contract")
        row = _closest_trial(event, a)
        if not row:
            errors.append(f"event {index} lacks trial energy rows")
        else:
            residual = _finite(row.get("energy_residual_J_per_m"))
            tolerance = max(_finite(row.get("energy_tolerance_J_per_m")), 0.0)
            residuals.append(residual)
            if residual + tolerance < -1.0e-15:
                errors.append(f"event {index} violates energy closure")
            if row.get("topology_changed") is not True:
                errors.append(f"event {index} committed without topology change")
        if a >= p * (1.0 - 1.0e-8):
            full_proposal += 1
        else:
            truncated += 1

    for index, event in enumerate(arrested):
        if event.get("inserted") is True:
            errors.append(f"arrested attempt {index} is marked inserted")
        if event.get("athermal_Gc_used") is not False:
            errors.append(f"arrested attempt {index} used an athermal Gc")

    cycles = _cycles(case)
    projected = _projected_extension(case, geometry)
    status = (
        "propagated"
        if committed
        else ("energy_arrested" if arrested else "right_censored_no_event")
    )
    return {
        "case": str(case),
        "parameter_option": control.get("parameter_option"),
        "temperature_K": _finite(meta.get("temperature_K", 300.0)),
        "deltaK_fraction": _finite(meta.get("deltaK_fraction")),
        "target_deltaK_MPa_sqrt_m": _finite(
            control.get("target_deltaK_MPa_sqrt_m")
        ),
        "R": _finite(control.get("R")),
        "frequency_Hz": _finite(control.get("frequency_Hz")),
        "hazard_seed": int(control.get("cleavage_hazard_seed", 0)),
        "trial_fraction": _finite(meta.get("energy_gate_trial_fraction", 0.0)),
        "status": status,
        "attempted_events": len(attempts),
        "committed_events": len(committed),
        "arrested_attempts": len(arrested),
        "full_proposal_events": full_proposal,
        "truncated_events": truncated,
        "mean_proposed_event_um": (
            1.0e6 * sum(proposal) / len(proposal) if proposal else 0.0
        ),
        "mean_committed_event_um": (
            1.0e6 * sum(actual) / len(actual) if actual else 0.0
        ),
        "mean_committed_over_proposed": (
            sum(ratios) / len(ratios) if ratios else 0.0
        ),
        "minimum_energy_residual_J_per_m": min(residuals, default=0.0),
        "mean_orientation_gamma_relative": (
            sum(gamma) / len(gamma) if gamma else 0.0
        ),
        "direction_sources": sorted(direction_sources),
        "cycles_consumed": cycles,
        "projected_extension_um": projected * 1.0e6,
        "projected_da_dN_m_per_cycle": (
            projected / cycles if cycles > 0.0 else 0.0
        ),
        "fixed_deltaK_exact": control.get(
            "fixed_deltaK_exact_within_relative_1e-12"
        ),
        "errors": errors,
        "pass": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--require-bracket", action="store_true")
    parser.add_argument("--require-truncation", action="store_true")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()

    controls = sorted(args.root.rglob(CONTROL))
    if not controls:
        raise SystemExit(f"no {CONTROL} files found below {args.root}")
    rows = [summarize_case(path) for path in controls]
    failures = [row for row in rows if not row["pass"]]
    propagated = [row for row in rows if row["committed_events"] > 0]
    censored = [row for row in rows if row["committed_events"] == 0]
    truncated = [row for row in rows if row["truncated_events"] > 0]
    campaign_errors = []
    if args.require_bracket and (not propagated or not censored):
        campaign_errors.append(
            "qualification does not bracket growth: require at least one propagated and one censored case"
        )
    if args.require_truncation and not truncated:
        campaign_errors.append("no energetically truncated committed event was observed")

    payload = {
        "schema": "v10.2.30_energy_gated_three_deltaK_qualification",
        "root": str(args.root),
        "case_count": len(rows),
        "propagated_case_count": len(propagated),
        "censored_case_count": len(censored),
        "truncated_case_count": len(truncated),
        "failed_case_count": len(failures),
        "campaign_errors": campaign_errors,
        "pass": not failures and not campaign_errors,
        "cases": rows,
    }
    out_json = args.out_json or args.root / "v10_2_30_qualification_summary.json"
    out_csv = args.out_csv or args.root / "v10_2_30_qualification_summary.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    scalar_keys = [
        key
        for key, value in rows[0].items()
        if not isinstance(value, (list, dict))
    ]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
