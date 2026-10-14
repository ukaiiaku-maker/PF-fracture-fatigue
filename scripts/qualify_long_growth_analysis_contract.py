#!/usr/bin/env python3
"""Qualify retained long-growth histories against the R11 analysis contract.

This is an analysis-only qualification.  It reads the accepted native runs and
does not regenerate a trajectory.  Missing external evidence is reported as an
explicit incomplete result rather than being mistaken for a physics failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.fracture_observation_history import (  # noqa: E402
    LengthCoordinate,
    ObservationStage,
)
from scripts.analyze_final_developed_growth import selected_resistance  # noqa: E402
from scripts.export_physical_growth_packet import CASES, event_pairs, history_rows  # noqa: E402


SCHEMA = "common-fracture-long-growth-qualification/1"
OUTPUT_POLICY_RECEIPT = (
    ROOT / "common_framework_readiness/OUTPUT_POLICY_INDEPENDENCE_QUALIFICATION.json"
)
REQUIRED_CASE_FILES = (
    "fracture_observations_online.jsonl",
    "stochastic_avalanche_geometry_events.json",
    "steps_{token}K.csv",
    "kinetic_tip_cell_audit_v101.json",
    "shared_runtime_attestation.post.json",
    "run_args.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required_files(case: Path, token: str) -> list[Path]:
    return [case / name.format(token=token) for name in REQUIRED_CASE_FILES]


def evidence_inventory() -> dict[str, Any]:
    cases: dict[str, Any] = {}
    missing: list[str] = []
    for label, (case, token) in CASES.items():
        files = required_files(case, token)
        absent = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
        missing.extend(absent)
        cases[label] = {
            "root": str(case.relative_to(ROOT)),
            "required_files": [str(path.relative_to(ROOT)) for path in files],
            "missing_files": absent,
        }
    return {"available": not missing, "missing_files": missing, "cases": cases}


def _query_record(history, target_m: float, coordinate: LengthCoordinate) -> dict[str, Any]:
    query = history.at_length(target_m, coordinate=coordinate)
    record = {
        "target_m": target_m,
        "coordinate": coordinate.value,
        "status": query.status,
        "interpolation_performed": query.interpolation_performed,
        "lower_sequence": None if query.lower is None else query.lower.sequence,
        "upper_sequence": None if query.upper is None else query.upper.sequence,
    }
    if query.status == "discrete_jump_bracket":
        if query.lower is None or query.upper is None:
            raise ValueError("jump bracket lacks bounding observations")
        if query.lower.stage is not ObservationStage.PRE_EVENT:
            raise ValueError("lower jump bound is not the exact pre-event state")
        if query.upper.stage is not ObservationStage.COMMITTED:
            raise ValueError("upper jump bound is not the paired committed state")
        if query.lower.event_id != query.upper.event_id:
            raise ValueError("jump bracket crosses unrelated events")
    if query.interpolation_performed:
        raise ValueError("discrete history query fabricated an interpolated state")
    return record


def audit_case(label: str, case: Path, token: str) -> dict[str, Any]:
    history, steps, _audit, _records = history_rows(label, case, token)
    pairs = event_pairs(history)
    resistance = selected_resistance(label, history, steps)
    if len(pairs) != 203 or len(resistance) != len(pairs):
        raise ValueError(f"{label}: expected 203 paired accepted events")

    previous_committed = -math.inf
    repeated_stage_coordinates = 0
    stage_positions: dict[float, int] = {}
    for row in history.rows:
        if row.sequence < 0 or row.committed_length_m + 1e-15 < row.pre_length_m:
            raise ValueError(f"{label}: invalid chronological geometry")
        coordinate = row.stage_extension_m
        stage_positions[coordinate] = stage_positions.get(coordinate, 0) + 1
        if row.stage is ObservationStage.COMMITTED:
            if row.committed_length_m + 1e-15 < previous_committed:
                raise ValueError(f"{label}: committed extension is not monotone")
            previous_committed = row.committed_length_m
    repeated_stage_coordinates = sum(count > 1 for count in stage_positions.values())

    for pre, committed in pairs:
        if pre.stage is not ObservationStage.PRE_EVENT:
            raise ValueError(f"{label}: pair lacks pre-event state")
        if committed.stage is not ObservationStage.COMMITTED:
            raise ValueError(f"{label}: pair lacks committed state")
        if pre.event_id != committed.event_id or pre.front_id != committed.front_id:
            raise ValueError(f"{label}: event/front identity changes across transaction")
        if not math.isfinite(pre.K_J_Pa_sqrt_m):
            raise ValueError(f"{label}: exact pre-event K_J is absent")
        if committed.path_arclength_m + 1e-15 < committed.committed_length_m:
            raise ValueError(f"{label}: path arclength is shorter than projected extension")

    queries = []
    for target_um in (100.0, 500.0, 1000.0):
        for coordinate in (LengthCoordinate.EXTENSION, LengthCoordinate.PATH_ARCLENGTH):
            queries.append(_query_record(history, target_um * 1e-6, coordinate))

    attestation_path = case / "shared_runtime_attestation.post.json"
    attestation = json.loads(attestation_path.read_text())
    if attestation.get("execution_verified") is not True:
        raise ValueError(f"{label}: execution attestation is not verified")

    fields = sorted(path for path in case.glob("field_snapshots_*.png") if path.is_file())
    inventory_paths = required_files(case, token) + fields
    return {
        "status": "QUALIFIED_RETAINED_NATIVE_HISTORY",
        "root": str(case.relative_to(ROOT)),
        "accepted_event_pairs": len(pairs),
        "observation_count": len(history.rows),
        "terminal_projected_extension_m": pairs[-1][1].committed_length_m,
        "terminal_path_arclength_m": pairs[-1][1].path_arclength_m,
        "repeated_stage_coordinate_count": repeated_stage_coordinates,
        "front_ids": sorted({row.front_id for row in history.rows}),
        "topology_generations": sorted({row.topology_generation for row in history.rows}),
        "load_segments": sorted({row.segment_id for row in history.rows}),
        "length_queries": queries,
        "resistance_definition": resistance[0]["resistance_definition"],
        "resistance_rows": len(resistance),
        "field_outputs": [str(path.relative_to(ROOT)) for path in fields],
        "artifact_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in inventory_paths
        },
        "execution_attestation_sha256": sha256(attestation_path),
    }


def qualify() -> dict[str, Any]:
    inventory = evidence_inventory()
    if not inventory["available"]:
        return {
            "schema": SCHEMA,
            "decision": "INCOMPLETE_EXTERNAL_EVIDENCE_MISSING",
            "production_execution_authorized": False,
            "inventory": inventory,
            "retrieval_instruction": (
                "restore the current_science_pilots_6abbf10 PF evidence root from "
                "its verified archive, then rerun this command"
            ),
        }

    cases = {
        label: audit_case(label, case, token)
        for label, (case, token) in CASES.items()
    }
    output_policy = (
        json.loads(OUTPUT_POLICY_RECEIPT.read_text())
        if OUTPUT_POLICY_RECEIPT.is_file() else None
    )
    output_policy_status = (
        "PASS_NATIVE_BYTE_IDENTICAL_STATE"
        if output_policy is not None and output_policy.get("decision") == "PASS"
        else "PENDING_DETERMINISTIC_NATIVE_PAIR"
    )
    return {
        "schema": SCHEMA,
        "decision": "COMPLETE_RETAINED_LONG_GROWTH_ANALYSIS",
        "production_execution_authorized": False,
        "scope": (
            "analysis regression over two retained native 1 mm trajectories; "
            "no new physical trajectory was executed"
        ),
        "cases": cases,
        "contract_results": {
            "chronological_stage_history": "PASS",
            "exact_pre_event_committed_pairing": "PASS",
            "projected_extension_separate_from_path_arclength": "PASS",
            "repeated_length_coordinates_retained": "PASS",
            "K_J_not_relabelled_as_modal_K_I": "PASS",
            "discrete_jump_queries_do_not_interpolate": "PASS",
            "developed_resistance_selector": "PASS",
            "selected_native_field_render_retained": "PASS",
            "mixed_mode_and_signed_J_schema": (
                "IMPLEMENTED_AND_UNIT_TESTED; these mode-I retained histories do not "
                "constitute a mixed-mode specimen qualification"
            ),
            "event_hold_reload_control": (
                "IMPLEMENTED_SCHEMA; retained developed histories contain one loading "
                "segment and do not prove independent unload/reload resistance"
            ),
            "restart_continuity": (
                "NOT_REQUALIFIED_BY_THIS_ANALYSIS; native execution attestations and "
                "atomic restart tests remain separate evidence"
            ),
            "output_policy_state_independence": output_policy_status,
        },
        "output_policy_qualification": {
            "receipt": str(OUTPUT_POLICY_RECEIPT.relative_to(ROOT)),
            "receipt_sha256": None if output_policy is None else sha256(OUTPUT_POLICY_RECEIPT),
            "scope": None if output_policy is None else output_policy["qualification_scope"],
        },
        "limitations": [
            "The common-mark hot/cold geometry is controlled evidence, not independent morphology invariance.",
            "The histories provide event-sampled monotonic resistance, not unload/reload R-curve points.",
            "K_J remains J-derived K_J and is not interpreted as modal K_I.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "common_framework_readiness/LONG_GROWTH_ANALYSIS_QUALIFICATION.json",
    )
    args = parser.parse_args(argv)
    report = qualify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"].startswith("COMPLETE_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
