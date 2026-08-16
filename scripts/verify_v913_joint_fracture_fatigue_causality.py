#!/usr/bin/env python3
"""Fail-closed completion gate for the joint fracture/fatigue causality study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


FRACTURE_FIGURES = (
    "DBTT_causal_design_K_vs_T",
    "PeakT_causal_design_K_vs_T",
    "causal_mechanism_transition_map",
    "causal_response_vs_deltaThetaSigma",
    "causal_response_vs_delta_mu",
    "causal_response_vs_overlap",
    "causal_response_vs_plastic_bottleneck",
)
FATIGUE_FIGURES = (
    "prospective_joint_da_dN_vs_deltaK",
    "prospective_joint_da_dN_vs_f",
    "prospective_joint_fracture_and_fatigue_panels",
    "prospective_joint_knee_LCF_comparison",
)
JOINT_FIGURES = (
    "fracture_PC_vs_fatigue_PC",
    "shared_barrier_feature_heatmap",
    "fracture_fatigue_barrier_phase_map",
    "joint_candidate_pareto_map",
    "canonical_and_prospective_joint_map",
    "joint_mechanism_atlas",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--require-clean-worktree", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify_figure_triplets(root: Path, stems: tuple[str, ...], errors: list[str]) -> None:
    for stem in stems:
        for suffix in (".png", ".pdf", "_plot_data.csv"):
            path = root / f"{stem}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing or empty artifact: {path}")


def verify_counts(root: Path) -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []
    fracture = root / "prospective_fracture_analysis_final"
    fatigue = root / "prospective_fatigue" / "analysis_final"
    registry = root / "prospective_fatigue_registry"
    loads = root / "prospective_fatigue" / "load_selection"
    joint = root / "final_joint_analysis"

    required = (
        fracture / "prospective_fracture_analysis_manifest.json",
        fatigue / "prospective_joint_fatigue_analysis_manifest.json",
        registry / "prospective_fatigue_registry_manifest.json",
        registry / "prospective_fatigue_registry_roundtrip_audit.csv",
        loads / "prospective_fatigue_load_selection_manifest.json",
        loads / "prospective_fatigue_adaptive_loads.csv",
        fatigue / "prospective_joint_fatigue_rates.csv",
        fatigue / "prospective_joint_accelerated_explicit_overlap.csv",
        fatigue / "prospective_joint_fatigue_morphology.csv",
        joint / "JOINT_FRACTURE_FATIGUE_BARRIER_CAUSALITY_REPORT.md",
        joint / "joint_candidate_pareto_front.csv",
        joint / "final_joint_analysis_manifest.json",
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty artifact: {path}")
    if errors:
        return errors, {}

    fm = read_json(required[0])
    fam = read_json(required[1])
    rm = read_json(required[2])
    lm = read_json(required[4])
    expected_fracture = {
        "analyzed_candidates": 31,
        "complete_K300_cases": 31,
        "complete_historical_cases": 310,
        "full_state_snapshots": 5456,
        "qualified_candidates": 31,
        "physics_changed": False,
    }
    expected_fatigue = {
        "candidate_count": 7,
        "rate_case_count": 85,
        "developed_target_rates": 78,
        "cycle_or_hazard_censors": 7,
        "partial_or_numerical_unresolved": 0,
        "event_count": 1222,
        "accelerated_explicit_overlap_candidates": 7,
        "physics_changed": False,
    }
    expected_registry = {
        "candidate_count": 7,
        "active_parameter_count": 29,
        "all_round_trip_identity": True,
        "parameter_refit_for_fatigue": False,
        "physics_changed": False,
    }
    expected_loads = {
        "candidate_count": 7,
        "load_count": 70,
        "minimum_loads_per_candidate": 10,
        "maximum_loads_per_candidate": 10,
        "accelerated_explicit_overlap_per_candidate": True,
        "fixed_dense_grid_used": False,
        "physics_changed": False,
    }
    for label, actual, expected in (
        ("fracture", fm, expected_fracture),
        ("fatigue", fam, expected_fatigue),
        ("registry", rm, expected_registry),
        ("loads", lm, expected_loads),
    ):
        for key, value in expected.items():
            if actual.get(key) != value:
                errors.append(f"{label} manifest {key}: expected {value!r}, got {actual.get(key)!r}")

    roundtrip = read_csv(required[3])
    if len(roundtrip) != 7:
        errors.append(f"round-trip row count: expected 7, got {len(roundtrip)}")
    for row in roundtrip:
        if row.get("round_trip_identity") != "True" or row.get("active_parameter_count") != "29":
            errors.append(f"failed 29-field round trip: {row.get('candidate_id')}")
        if row.get("parameter_refit_for_fatigue") != "False" or row.get("physics_changed") != "False":
            errors.append(f"physics/refit violation: {row.get('candidate_id')}")

    load_rows = read_csv(required[5])
    load_counts = Counter(row["candidate_id"] for row in load_rows)
    if len(load_rows) != 70 or set(load_counts.values()) != {10} or len(load_counts) != 7:
        errors.append(f"adaptive load layout invalid: rows={len(load_rows)} counts={dict(load_counts)}")
    overlap_counts = Counter(
        row["candidate_id"] for row in load_rows
        if row["accelerated_required"] == "True" and row["explicit_required"] == "True"
    )
    if set(overlap_counts.values()) != {1} or set(overlap_counts) != set(load_counts):
        errors.append(f"accelerated/explicit overlap layout invalid: {dict(overlap_counts)}")

    rates = read_csv(required[6])
    statuses = Counter(row["status_class"] for row in rates)
    if len(rates) != 85 or statuses != {"developed_target_reached": 78, "cycle_or_hazard_censor": 7}:
        errors.append(f"fatigue status semantics invalid: rows={len(rates)} statuses={dict(statuses)}")
    if any("INVALID_DUPLICATE_WRITER" in row["result_path"] for row in rates):
        errors.append("quarantined duplicate-writer output entered final rates")
    if any(row["status_class"] == "cycle_or_hazard_censor" and row["plot_marker_semantics"] != "downward_triangle" for row in rates):
        errors.append("a true censor does not use downward-triangle semantics")

    overlap = read_csv(required[7])
    ratios = [float(row["explicit_over_accelerated_rate_ratio"]) for row in overlap]
    if len(overlap) != 7 or any(row["same_seed"] != "True" for row in overlap):
        errors.append("same-seed overlap is incomplete")
    if any(not math.isfinite(value) or not 0.9 <= value <= 1.1 for value in ratios):
        errors.append(f"explicit/accelerated parity outside [0.9, 1.1]: {ratios}")

    explicit_seeds: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rates:
        if row["integration_mode"] == "explicit":
            explicit_seeds[(row["candidate_id"], row["normalized_f"])].add(row["seed"])
    three_seed_candidates = {
        candidate for (candidate, _), seeds in explicit_seeds.items() if len(seeds) >= 3
    }
    if len(three_seed_candidates) != 4:
        errors.append(f"expected four three-seed overlap candidates, got {sorted(three_seed_candidates)}")

    morphology = read_csv(required[8])
    if len(morphology) != 7:
        errors.append(f"morphology row count: expected 7, got {len(morphology)}")
    for row in morphology:
        if (
            row["finite_developed_points"] != "9"
            or row["cycle_or_hazard_censors"] != "1"
            or row["partial_or_numerical_unresolved"] != "0"
            or row["localized_knee_detected"] != "False"
            or row["fatigue_shape_class"] != "SMOOTH_ARRHENIUS_HCF_TO_LCF_NO_LOCALIZED_KNEE"
        ):
            errors.append(f"fatigue morphology acceptance failed: {row.get('candidate_id')}")

    verify_figure_triplets(fracture, FRACTURE_FIGURES, errors)
    verify_figure_triplets(fatigue, FATIGUE_FIGURES, errors)
    verify_figure_triplets(joint, JOINT_FIGURES, errors)
    report = required[9].read_text()
    if any(f"{number}. **" not in report for number in range(1, 17)):
        errors.append("final report does not contain all 16 explicit answers")
    if "MODEL_INTERNAL_PHYSICAL_PLAUSIBILITY" not in report:
        errors.append("final report lacks the required realism qualification")

    evidence = {
        "fracture_candidates": fm.get("analyzed_candidates"),
        "historical_fracture_cases": fm.get("complete_historical_cases"),
        "K300_cases": fm.get("complete_K300_cases"),
        "first_passage_state_snapshots": fm.get("full_state_snapshots"),
        "fatigue_candidates": fam.get("candidate_count"),
        "fatigue_rate_cases": fam.get("rate_case_count"),
        "fatigue_events": fam.get("event_count"),
        "developed_target_rates": fam.get("developed_target_rates"),
        "true_censors": fam.get("cycle_or_hazard_censors"),
        "partial_or_unresolved": fam.get("partial_or_numerical_unresolved"),
        "overlap_ratio_min": min(ratios),
        "overlap_ratio_max": max(ratios),
        "three_seed_candidates": sorted(three_seed_candidates),
    }
    return errors, evidence


def main() -> int:
    args = parse_args()
    errors, evidence = verify_counts(args.root)
    if args.require_clean_worktree:
        status = subprocess.run(
            ["git", "status", "--porcelain"], check=True, text=True, capture_output=True
        ).stdout.strip()
        if status:
            errors.append("worktree is not clean")
    result = {
        "schema": "v913_joint_fracture_fatigue_causality_verification_v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "evidence": evidence,
    }
    output = args.root / "joint_causality_verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
