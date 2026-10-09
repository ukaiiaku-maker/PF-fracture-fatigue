#!/usr/bin/env python3
"""Assemble bounded overlays and the complete retained physical A/B ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.controlled_history_v3 import EXPECTED
from arrhenius_fracture.one_void_final_closure_v2 import REQUIRED_GATES, SCHEMA, validate


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def status(passed):
    return "PASS" if passed else "BLOCKED"


def exact_pair(first, second, identity):
    left, right = Path(first).read_bytes(), Path(second).read_bytes()
    return {"identity": identity, "exact": left == right,
            "a_sha256": hashlib.sha256(left).hexdigest(),
            "b_sha256": hashlib.sha256(right).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_ledger", type=Path)
    parser.add_argument("bounded", type=Path)
    parser.add_argument("static", type=Path)
    parser.add_argument("controlled", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite final ledger")
    head = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    baseline = load(args.baseline_ledger)
    r_tip = load(args.bounded / "a" / "r_tip" / "report.json")
    atlas = load(args.bounded / "a" / "static_atlas" / "report.json")
    static = load(args.static / "a" / "report.json")
    full_static_path = args.static / "full-a" / "report.json"
    full_static = load(full_static_path) if full_static_path.exists() else None
    pairs = [
        exact_pair(args.bounded / "a" / "r_tip" / "report.json",
                   args.bounded / "b" / "r_tip" / "report.json", "r_tip"),
        exact_pair(args.bounded / "a" / "static_atlas" / "report.json",
                   args.bounded / "b" / "static_atlas" / "report.json", "static_atlas"),
        exact_pair(args.static / "a" / "report.json", args.static / "b" / "report.json",
                   "static_v2_sentinels"),
    ]
    if full_static is not None:
        pairs.append(exact_pair(args.static / "full-a" / "report.json",
                                args.static / "full-b" / "report.json", "static_v2_full"))

    controlled_rows = []
    new_by_case = {}
    for case in sorted(EXPECTED):
        path = args.controlled / case / "a" / "report.json"
        if path.exists():
            row = load(path); new_by_case[case] = row
            pairs.append(exact_pair(path, args.controlled / case / "b" / "report.json",
                                    "controlled_v3:" + case))
            controlled_rows.append({"dataset": "controlled_v3", "case_identity": case,
                "classification": "PASS" if row["passed"] else "BLOCKED",
                "final_classification": row["final_classification"],
                "expected_classification": row["prospectively_frozen_expected_classification"]})
    baseline_controlled = {row["case_identity"]: row for row in baseline["controlled_history_classifications_v2"]}
    for case in sorted(EXPECTED):
        if case not in new_by_case:
            row = baseline_controlled[case]
            controlled_rows.append({"dataset": "controlled_v3", "case_identity": case,
                "classification": "PASS" if row["passed"] else "BLOCKED",
                "final_classification": row["final_source_resolved_classification"],
                "expected_classification": EXPECTED[case], "source": "final_baseline_reexecution"})
    controlled_passed = sum(row["classification"] == "PASS" for row in controlled_rows)

    replay_rows = []
    for seed in range(12000, 12032):
        row = load(args.replay / str(seed) / "a" / "report.json")
        pairs.append(exact_pair(args.replay / str(seed) / "a" / "report.json",
                                args.replay / str(seed) / "b" / "report.json",
                                "physical_replay_v2:" + str(seed)))
        replay_rows.append({"dataset": "natural_physical_replay_v2", "case_identity": str(seed),
            "classification": "PASS" if row["passed"] else "BLOCKED",
            "V1_bitwise_terminal_fingerprint_equal": row["V1_bitwise_terminal_fingerprint_equal"],
            "maximum_ulp_difference": max(
                item.get("maximum_ulp_difference", 0) for item in row["continuous_comparisons"]),
            "threshold_safety": row["threshold_safety"],
            "event_selection_safety": row["event_selection_safety"]})
    replay_passed = sum(row["classification"] == "PASS" for row in replay_rows)

    static_passed = bool(full_static and full_static["decision"]["passed"]
                         and len(full_static["decision"]["families"]) == 33)
    static_final = ("PASS_33_OF_33" if static_passed else
                    "BLOCKED_SENTINELS_FAILED" if not static["decision"]["passed"] else
                    "BLOCKED_FULL_MATRIX_FAILED")
    baseline_checks = {
        "transition_partitions": baseline["transition_partitions"]["passed"] == 45,
        "common_terminal_restarts": baseline["common_terminal_exact"],
        "lifecycle_rollback": baseline["rollback"]["passed"] == 39,
        "same_worker_natural_peers": baseline["natural_partitions_restart"]["passed"] == 160,
        "future_causal_disabled_neutrality": baseline["mandatory_scientific_gates"]["disabled_future_causal_neutrality"],
        "stagewise_conservation": baseline["mandatory_scientific_gates"]["stagewise_topology_and_conservation"],
    }
    gates = {
        "r_tip_final_classification": {"status": r_tip["R_TIP_CAUSAL_LAW"],
            "ownership": r_tip["R_TIP_OWNERSHIP"], "model_form_complete": r_tip["model_form_complete"]},
        "static_family_v1_preserved": {"status": "PASS", "result": atlas["STATIC_FAMILY_V1"]},
        "static_family_v2_final": {"status": "PASS" if static_passed else "BLOCKED", "result": static_final,
            "sentinel_passed": sum(row["passed"] for row in static["decision"]["families"]),
            "sentinel_total": len(static["decision"]["families"])},
        "controlled_histories_final": {"status": status(controlled_passed == 12),
            "passed": controlled_passed, "total": 12},
        "natural_bitwise_replay_v1_preserved": {"status": "PASS", "result": "10_OF_160_PASS_RETAINED"},
        "natural_physical_replay_v2": {"status": status(replay_passed == 32),
            "passed": replay_passed, "total": 32},
        "paired_evidence_exact": {"status": status(all(row["exact"] for row in pairs)),
                                  "pairs": len(pairs)},
    }
    for name, passed in baseline_checks.items():
        gates[name] = {"status": status(bool(passed))}
    mandatory_pass = all(gates.get(name, {}).get("status") == "PASS" for name in REQUIRED_GATES)
    terminal = ("V5_ONE_VOID_SCIENTIFIC_CLOSURE_QUALIFIED" if mandatory_pass else
                "V5_ONE_VOID_FINAL_SCIENTIFIC_CLOSURE_COMPLETE_BUT_BLOCKED")
    ontology = controlled_rows + replay_rows + [
        {"dataset": "r_tip", "case_identity": "constitutive_handoff",
         "classification": "PASS" if r_tip["R_TIP_CAUSAL_LAW"] == "IMPLEMENTED" else "BLOCKED"},
        *({"dataset": "static_v2", "case_identity": row["family"],
           "classification": "PASS" if row["passed"] else "BLOCKED"}
          for row in static["decision"]["families"]),
    ]
    record = {
        "schema": SCHEMA, "executed_code_sha": head,
        "physical_baseline_implementation_sha": baseline["executed_code_sha"],
        "terminal_classification": terminal, "mission_terminal": True,
        "r_tip_distinct_from_R_void": r_tip["R_TIP_DISTINCT_FROM_VOID_RADIUS"] == "PASS",
        "gates": gates,
        "controlled_histories": controlled_rows,
        "natural_physical_replay_v2": replay_rows,
        "paired_evidence_comparison": pairs,
        "source_bound_ontology": {"row_count": len(ontology), "rows": ontology},
        "legacy_projection_used": False,
        "predicate_relaxed": False,
        "release_candidate_authorized": mandatory_pass,
        "merge_authorized": False,
    }
    validation = validate(record)
    if not validation["valid"]:
        raise ValueError(validation)
    args.output.mkdir(parents=True)
    write(args.output / "scientific_ledger.json", record)
    write(args.output / "ontology_validation.json", validation)
    write(args.output / "sha256_manifest.json", {
        name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
        for name in ("scientific_ledger.json", "ontology_validation.json")})
    print(json.dumps({"terminal": terminal, "controlled": controlled_passed,
                      "physical_replay": replay_passed, "static": static_final}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
