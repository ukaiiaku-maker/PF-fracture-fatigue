#!/usr/bin/env python3
"""Corrected Stage-II qualification: real branching is physical; injection is a screen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.sharp_wake_backend_v12 import V11_MODEL_ID, V12_MODEL_ID
from arrhenius_fracture.stage2_criterion_v12 import ROLLBACK_STAGES
from arrhenius_fracture.v12_production_driver import (
    _observables, build_loaded_state, execute_physical_two_arm_event,
)

SCHEMA = "v12.production-integration-qualified-evidence/3"


def _head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()


def _run_cli(out: Path, *, explicit: bool) -> dict:
    command = [
        sys.executable, "-m", "arrhenius_fracture.v12_production_driver",
        "--trajectory", "sequential", "--out", str(out),
    ]
    if explicit:
        command[3:3] = ["--sharp-wake-model", V11_MODEL_ID]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(out.read_text())["row"]


def main(argv=None) -> int:
    out = Path(argv[0] if argv else "artifacts/v12_production_integration_v3")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    default = _run_cli(out / "v11_default.json", explicit=False)
    explicit = _run_cli(out / "v11_explicit.json", explicit=True)
    fields = (
        "fingerprint", "graph_length_m", "damage_l1", "displacement_l2_m",
        "plastic_history_l2", "density_l2_m2", "reaction_N_per_m",
        "stored_energy_J_per_m", "hazards", "thresholds", "rng_state",
        "tip_process_state", "event_counters", "competition_consumed",
    )
    comparison = {name: default["final"][name] == explicit["final"][name] for name in fields}
    rows.append({
        "gate": "V11_DEFAULT_EXPLICIT_EXECUTABLE_NEUTRALITY", "case": "separate_cli_processes",
        "comparison": comparison, "passed": all(comparison.values()),
    })

    accepted = build_loaded_state(V12_MODEL_ID)
    final, branch = execute_physical_two_arm_event(accepted)
    rows.append({
        "gate": "V12_PHYSICAL_TWO_ARM_BRANCH", "case": "correlated_010_100_first_passages",
        "initial": _observables(accepted), "final": _observables(final),
        "measurement": branch,
        "passed": bool(
            branch["accepted"] and branch["action_type"] == "two_arm"
            and len(set(branch["candidate_ids"])) == 2
            and len(set(branch["branch_ids"])) == 2
            and branch["hazard_dissipation_J_per_m"] > 0.0
            and branch["energy_margin_J_per_m"] > 0.0
            and len(final.crack_network.active_tip_ids) == 2
        ),
    })

    # These rows prove atomic rollback after operations on the real two-arm path.
    # They are explicitly screens, never evidence of event reachability.
    for stage in ROLLBACK_STAGES:
        accepted = build_loaded_state(V12_MODEL_ID)
        before = _observables(accepted)
        operations = []
        exception = None
        try:
            execute_physical_two_arm_event(
                accepted, failure_stage=stage, operation_log=operations,
            )
        except RuntimeError as error:
            exception = str(error)
        after = _observables(accepted)
        rows.append({
            "gate": "FORCED_EVENT_ROLLBACK_SCREEN", "case": stage,
            "event_classification": "software_failure_injection",
            "operations": operations, "exception": exception,
            "fingerprints_equal": before["fingerprint"] == after["fingerprint"],
            "passed": exception == "injected:" + stage and before["fingerprint"] == after["fingerprint"],
        })

    gate_names = (
        "V11_DEFAULT_EXPLICIT_EXECUTABLE_NEUTRALITY",
        "V12_PHYSICAL_TWO_ARM_BRANCH", "FORCED_EVENT_ROLLBACK_SCREEN",
    )
    gates = {
        name: "PASS" if all(row["passed"] for row in rows if row["gate"] == name) else "FAIL"
        for name in gate_names
    }
    gates["STAGE_II_BASE_V3_LOCAL"] = "PASS" if all(value == "PASS" for value in gates.values()) else "FAIL"
    payload = {
        "schema": SCHEMA, "implementation_git_sha": _head(), "gates": gates,
        "ontology": {
            "physical_event_evidence": ["V12_PHYSICAL_TWO_ARM_BRANCH"],
            "software_screens_only": ["FORCED_EVENT_ROLLBACK_SCREEN"],
        },
        "rows": rows,
    }
    evidence = out / "case_rows.json"
    evidence.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "sha256_manifest.json").write_text(json.dumps({
        "case_rows.json": hashlib.sha256(evidence.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gates, indent=2, sort_keys=True))
    return 0 if gates["STAGE_II_BASE_V3_LOCAL"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
