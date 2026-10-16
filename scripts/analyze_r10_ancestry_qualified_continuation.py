#!/usr/bin/env python3
"""Record the first post-lineage, ancestry-qualified joint continuation.

This consumes executed current-backend artifacts.  It does not fit a new law
or turn a numerical censor into a constitutive result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


WEAKT = "v913_zeroD_sobol_0129902"
WEAKT_ALIAS = "v913_paper_weakT01_0129902_persistent_sites"
P40 = "P40_TRANSFER_CALIBRATED_GEN2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--original-registry", type=Path, required=True)
    p.add_argument("--parent-registry", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    a = p.parse_args(argv)
    a.out_root.mkdir(parents=True, exist_ok=True)

    with a.original_registry.open(newline="") as stream:
        weak = next(x for x in csv.DictReader(stream) if x["option_key"] == WEAKT_ALIAS)
    with a.parent_registry.open(newline="") as stream:
        p40_record = next(x for x in csv.DictReader(stream) if x["candidate_id"] == P40)
    p40 = json.loads(p40_record["complete_parameter_vector"])
    lineage_path = a.run_root / "constitutive_temperature_reconciliation/reconciliation.json"
    lineage = json.loads(lineage_path.read_text())
    fracture_root = a.run_root / "fracture/original_weakt_ancestry_qualified_refined"
    fracture = json.loads((fracture_root / "panel_summary.json").read_text())["cases"]
    fatigue_root = a.run_root / "bridges/original_weakt_ancestry_qualified_seed4810"
    preflight_path = fatigue_root / "production_preflight.json"
    preflight = json.loads(preflight_path.read_text())
    checkpoint_path = fatigue_root / "Kmax_16p6666667/checkpoint/high_cycle_live_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    progress_path = fatigue_root / "Kmax_16p6666667/checkpoint/trajectory_progress.json"
    progress = json.loads(progress_path.read_text())
    causal_path = a.run_root / "thermal_accessibility_iteration/thermal_accessibility_result.json"
    causal = json.loads(causal_path.read_text())

    expected_hash = next(
        x["complete_row_sha256"] for x in json.loads(
            (a.run_root / "continuation_state.json").read_text()
        )["retained_rows"] if x["candidate_id"] == WEAKT
    )
    if preflight["binding"]["complete_row_sha256"] != expected_hash:
        raise RuntimeError("fatigue attempt is not bound to the retained Weak-T row")
    p40_zeroed = [k for k in (
        "cleave_gT_eV_per_K", "cleave_sT_GPa_per_K",
        "emit_gT_eV_per_K", "emit_sT_GPa_per_K",
    ) if float(p40[k]) == 0.0]
    rows = [{
        "candidate_id": WEAKT,
        "family": "A_ORIGINAL_SINGLE_EXP_FLOOR",
        "selection_status": "EXECUTED_ANCESTRY_QUALIFIED_CANONICAL_ROW",
        "reason": "original complete Weak-T material; all four temperature/entropy dependencies retained",
    }, {
        "candidate_id": P40,
        "family": "A_ORIGINAL_SINGLE_EXP_FLOOR",
        "selection_status": "ELIGIBLE_ISOTHERMAL_CONTROL_AND_DESIGN_SEED_THERMAL_PERFORMANCE_FAILED",
        "reason": (
            "zero explicit opening/emission slopes are lawful constant-Q EXP_floor coordinates; "
            "the executed thermal panel, not coefficient syntax, rejects this row as the final material"
        ),
    }, {
        "candidate_id": "R10K_WEAKT_COUPLED_EMIT_SCALE_S3P0_V1",
        "family": "A_ORIGINAL_SINGLE_EXP_FLOOR",
        "selection_status": "PRESERVED_TESTED_NON_PROMOTABLE",
        "reason": "coherent final scale inherits unauthorized P3/P4 endpoint fields",
    }]
    with (a.out_root / "selection_ledger.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

    result = {
        "schema": "r10.ancestry-qualified-joint-continuation/1",
        "lineage_audit": {
            "path": str(lineage_path.resolve()), "sha256": sha256(lineage_path),
            "decision": lineage["decision"],
        },
        "selected_candidate": {
            "candidate_id": WEAKT, "alias": WEAKT_ALIAS,
            "complete_row_sha256": expected_hash,
            "basis": "canonical original-family retained row evaluated without endpoint reconstruction",
            "new_parameter_fit": False,
        },
        "fracture_execution": {
            "scope": "fresh current-backend 300/450 K discriminating panel",
            "cases": fracture,
            "outcome": "FAIL_WEAK_T_PHYSICAL_ENDPOINT_SPAN",
            "quantified_limit": (
                "300 K completed 55 um with K_init=14.4315409362 MPa sqrt(m); "
                "450 K had zero events and zero extension through Kmax=80 MPa sqrt(m)"
            ),
            "stop_reason": "PHYSICAL_K_ENDPOINT_REACHED_NO_EVENT",
            "final_accepted_K_MPa_sqrt_m": 80.0,
            "accepted_steps": 1600,
            "accepted_time_s": 16000.0,
            "terminal_hazard_action": 0.021197933391292337,
            "pending_mark_action": 0.9788020666087076,
            "minimum_span_from_300K": 80.0 / 14.431540936206051,
            "remaining_temperatures": "not run after the 450 K hard censor rejected the row",
        },
        "fatigue_execution": {
            "scope": "same-row native current-state attempt at DeltaK=15 MPa sqrt(m)",
            "outcome": "OPERATOR_STOPPED_RESOURCE_EVIDENCE_NO_DURABLE_PROGRESS",
            "checkpoint_reason": checkpoint["reason"],
            "checkpoint_timestamp_unix_s": checkpoint["timestamp_unix_s"],
            "events_committed": len(progress.get("events", [])),
            "result_written": False,
            "observed_wall_seconds_before_operator_stop": 283,
            "limitation": (
                "no post-initial atomic checkpoint or event was committed in 283 s; "
                "the process remained inside the native chained-DMD/current-kernel map; this is "
                "not a material failure, convergence result, or intrinsic runtime limit"
            ),
            "analytical_prior": "A0 moderate-slope width was 0.0 decades; not treated as a full-state result",
        },
        "joint_decision": "NO_PROMOTION_OR_2D_ESCALATION",
        "mechanism_interpretation": (
            "The finite 300 K initiation and physical-endpoint 450 K censor reject the current "
            "Weak-T objective, but event absence alone does not identify a microscopic cause. "
            "The operator-stopped fatigue attempt has no rate or Paris-slope interpretation."
        ),
        "p40_control_disposition": {
            "zero_temperature_fields": p40_zeroed,
            "fresh_fracture_diagnostic": str((
                a.run_root / "fracture/p40_transfer_ancestry_qualified_refined/panel_summary.json"
            ).resolve()),
            "status": (
                "lawful retained isothermal control/design seed; not selected as the final thermal "
                "material because its executed joint thermal fracture performance fails"
            ),
            "kinetic_temperature_dependence": (
                "for fixed Q and state, lambda=nu0*exp[-Q/(kB*T)] remains temperature dependent "
                "when explicit gT and sT are zero"
            ),
        },
        "preserved_class_mechanism_evidence": {
            "source": {"path": str(causal_path.resolve()), "sha256": sha256(causal_path)},
            "DBTT": causal["mechanism_classification"]["DBTT"],
            "Peak-T": causal["mechanism_classification"]["Peak-T"],
            "scope": (
                "existing causal classification retained as diagnostic evidence; it does not "
                "override the non-promotable disposition of endpoint-constructed descendants"
            ),
        },
        "search_status": (
            "continue within ancestry-qualified original/refined/bank rows; use permitted physical "
            "nucleation and motion parameters with their native couplings; no endpoint DOFs"
        ),
        "same_row_2D_status": "WITHHELD_AFTER_1D_FRACTURE_FAILURE_AND_FATIGUE_NUMERICAL_LIMIT",
    }
    (a.out_root / "joint_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
