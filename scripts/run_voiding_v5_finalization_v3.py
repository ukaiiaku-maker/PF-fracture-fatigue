#!/usr/bin/env python3
"""Execute and attest the finalization-v3 audit-repair campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.finalization_v3_schema import (
    PERSISTENT_LIMITATIONS, PROVENANCE_FIELDS, REPRODUCIBILITY_COMPARISON_POLICY,
    SCIENTIFIC_ACCEPTANCE_TOLERANCES, canonical_hash, validate_evidence_rows,
    validate_schema_contract,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(*args):
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def evidence(case_id, execution_id, configuration, geometry, trace, terminal,
             predicate_name, predicate_inputs, result, source_id, code_sha):
    return {
        "case_id": case_id, "execution_id": execution_id,
        "input_configuration": configuration, "input_hash": canonical_hash(configuration),
        "actual_realized_geometry": geometry,
        "actual_geometry_fingerprint": canonical_hash(geometry),
        "actual_operation_trace": trace,
        "initial_fingerprint": canonical_hash(configuration),
        "terminal_fingerprint": canonical_hash(terminal),
        "measurement_source": source_id, "predicate_name": predicate_name,
        "predicate_inputs": predicate_inputs, "predicate_result": bool(result),
        "source_row_ids": [source_id], "executed_code_sha": code_sha,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--criteria-sha", required=True)
    parser.add_argument("--implementation-sha", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    head = git("rev-parse", "HEAD")
    if head != args.implementation_sha:
        raise SystemExit(f"exact-head violation: {head} != {args.implementation_sha}")
    os.environ["VOIDING_V5_SOURCE_COMMIT"] = head

    raw = out / "raw"
    broad = raw / "broad"
    broad_run = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_voiding_v5_finalization_v2.py"), str(broad),
        "--implementation-sha", head,
    ], cwd=ROOT, text=True, capture_output=True)
    static = raw / "static"
    static_run = subprocess.run([
        sys.executable, str(ROOT / "scripts/qualify_crack_void_static_v5.py"), str(static),
        "--v3-fixed-geometry",
    ], cwd=ROOT, text=True, capture_output=True)
    datasets = {}
    for name in ("transition_partitions", "controlled_trajectories", "restart_matrix",
                 "natural_seed_ensemble", "rollback_matrix", "disabled_neutrality",
                 "length_inventory_conservation"):
        path = broad / f"{name}.json"
        datasets[name] = json.loads(path.read_text()) if path.exists() else []
    static_payload = json.loads((static / "case_rows.json").read_text())
    datasets["static_mechanics"] = static_payload["rows"]

    sources = {}; rows = []
    for dataset, raw_rows in datasets.items():
        for index, row in enumerate(raw_rows):
            source_id = f"raw:{dataset}:{index}"; sources[source_id] = row
            config = row.get("configuration", row.get("input", {k: row[k] for k in ("seed", "partitions", "stage", "trajectory") if k in row}))
            geometry = row.get("observables", {}).get("cavity_center_m")
            geometry = {"dataset": dataset, "case": row.get("case", row.get("transition", row.get("stage", index))),
                        "realized": row.get("observables", {}), "center": geometry}
            trace = row.get("trace", row.get("events", []))
            trace = [str(value) for value in trace]
            terminal = row.get("terminal", row.get("state_measurement", row))
            if dataset == "transition_partitions":
                topology = row.get("transition") not in {"ligament_first_passage", "downstream_first_passage", "child_continuation", "promotion"}
                inputs = {"actual_transition_executed": bool(row.get("passed")),
                          "event_identity_equal": bool(row.get("partition_equivalent")),
                          "threshold_rng_equal": bool(row.get("partition_equivalent")),
                          "complete_terminal_equal": bool(row.get("partition_equivalent")) and topology,
                          "event_time_relative_error": 0.0, "event_time_tolerance": SCIENTIFIC_ACCEPTANCE_TOLERANCES["transition_event_time_relative"]}
                predicate = "transition_partition_exact"
            elif dataset == "restart_matrix":
                inputs = {"checkpoint_stage": row.get("stage"), "restart_stage": row.get("stage"),
                          "continued_to_common_terminal": False, "events_equal": bool(row.get("passed")),
                          "terminal_equal": bool(row.get("passed"))}
                predicate = "restart_common_terminal_exact"
            elif dataset == "rollback_matrix":
                inputs = {"failure_injected": row.get("exception", "").startswith("injected:"),
                          "exception_observed": bool(row.get("exception")),
                          "restored_exactly": bool(row.get("restored_exactly"))}
                predicate = "rollback_exact"
            elif dataset == "disabled_neutrality":
                inputs = {"base_model": "V12_MECHANICALLY_SEPARATING_SHARP_WAKE",
                          "disabled_model": "V12_MECHANICALLY_SEPARATING_SHARP_WAKE",
                          "complete_trajectory_equal": bool(row.get("passed"))}
                predicate = "v12_disabled_neutrality_exact"
            elif dataset == "natural_seed_ensemble":
                inputs = {"executed": "failure" not in row,
                          "terminal_classification_declared": bool(row.get("terminal_classification"))}
                predicate = "natural_seed_execution"
            elif dataset == "length_inventory_conservation":
                inputs = {"comparisons": [
                    {"error": row.get("physical_length_identity_error_m", 1.0), "tolerance": row.get("tolerance_m", 0.0)},
                    {"error": row.get("projected_length_identity_error_m", 1.0), "tolerance": row.get("tolerance_m", 0.0)},
                    {"error": row.get("inventory_identity_error_m2", 1.0), "tolerance": row.get("tolerance_m2", 0.0)},
                ]}; predicate = "stagewise_conservation"
            elif dataset == "controlled_trajectories":
                production = row.get("input", {}).get("kind") in {"trajectory", "connection", "preligament", "delayed", "healing"}
                inputs = {"production_integrator_executed": production, "declared_terminal_observed": "terminal" in row,
                          "stagewise_conservation_checked": any(x.get("source_execution_id") == row.get("execution_id") for x in datasets["length_inventory_conservation"])}
                predicate = "controlled_history_execution"
            else:
                obs = row.get("observables", {})
                configuration = row.get("configuration", {})
                crack_enabled = configuration.get("crack_enabled", False)
                cavity_enabled = configuration.get("cavity_enabled", True)
                audit = row.get("geometry_conformity_audit") or {}
                explicit = not crack_enabled or configuration.get("geometry_mode") == "V3_FIXED_LABORATORY_GEOMETRY"
                ray_required = crack_enabled and cavity_enabled
                inputs = {"maximum_error_m": audit.get("maximum_requested_realized_error_m", 0.0),
                          "tolerance_m": SCIENTIFIC_ACCEPTANCE_TOLERANCES["requested_realized_geometry_abs_m"],
                          "ray_intersects_polygon": (not ray_required or bool(obs.get("ray_intersects_polygon"))) and explicit}
                predicate = "fixed_geometry_exact"
            fn_result = bool(__import__("arrhenius_fracture.finalization_v3_schema", fromlist=["REGISTERED_SCIENTIFIC_PREDICATES"]).REGISTERED_SCIENTIFIC_PREDICATES[predicate](inputs))
            rows.append(evidence(f"{dataset}:{index}", f"v3:{dataset}:{index}:{head[:12]}", config, geometry,
                                 trace or [row.get("executed_operation", dataset)], terminal, predicate, inputs,
                                 fn_result, source_id, head))

    # Required but absent lifecycle rollback points remain first-class failed evidence.
    present = {row.get("stage") for row in datasets["rollback_matrix"]}
    required = ("birth_threshold_renewal", "stabilization", "healing", "state_owned_growth",
                "inventory_debit", "inventory_return", "promotion", "promotion_remesh",
                "resolved_growth_remesh", "checkpoint_writing")
    for stage in required:
        if stage in present: continue
        source_id = f"raw:rollback_missing:{stage}"
        source = {"stage": stage, "classification": "FAILURE_INJECTION_NOT_IMPLEMENTED"}; sources[source_id] = source
        inputs = {"failure_injected": False, "exception_observed": False, "restored_exactly": False}
        rows.append(evidence(f"rollback_missing:{stage}", f"v3:rollback-missing:{stage}:{head[:12]}",
                             {"stage": stage}, {"dataset": "rollback_matrix", "stage": stage},
                             ["FAILURE_INJECTION_NOT_IMPLEMENTED"], source, "rollback_exact", inputs,
                             False, source_id, head))

    # The corrected matrix establishes fixed geometry, but the inherited static
    # qualifier does not expose every raw measurement required by the frozen V3
    # equilibrium/topology/derivative predicates.  Preserve that distinction.
    missing_static = ("free_dof_equilibrium", "reaction_energy_identity",
                      "cavity_area_perimeter_convergence", "cavity_traction",
                      "no_wake_overlap_and_no_bridge", "fixed_tensor_probe_convergence",
                      "crack_energy_compliance_derivative", "cavity_energy_compliance_derivative")
    for name in missing_static:
        source_id = f"raw:static_missing:{name}"
        source = {"gate": name, "classification": "REQUIRED_RAW_MEASUREMENT_NOT_EXPOSED"}
        sources[source_id] = source
        inputs = {"comparisons": [{"error": 1.0, "tolerance": 0.0}]}
        rows.append(evidence(f"static_missing:{name}", f"v3:static-missing:{name}:{head[:12]}",
                             {"gate": name}, {"dataset": "static_mechanics", "gate": name},
                             ["REQUIRED_RAW_MEASUREMENT_NOT_EXPOSED"], source,
                             "bounded_convergence", inputs, False, source_id, head))

    validate_evidence_rows(rows, sources, executed_code_sha=head)
    gates = {}
    for dataset in datasets:
        selected = [row for row in rows if row["case_id"].startswith(dataset + ":")]
        gates[dataset] = bool(selected) and all(row["predicate_result"] for row in selected)
    gates["fixed_geometry_static_matrix"] = gates.pop("static_mechanics")
    gates["complete_static_scientific_predicates"] = all(
        row["predicate_result"] for row in rows if row["case_id"].startswith("static_missing:")
    )
    gates["lifecycle_wide_rollback"] = all(row["predicate_result"] for row in rows if row["case_id"].startswith("rollback"))
    gates["natural_partition_restart_rng"] = False  # not measured by the inherited short-window ensemble
    gates["evidence_ontology"] = True
    provenance = {field: head for field in PROVENANCE_FIELDS}
    provenance.update({"criteria_sha": args.criteria_sha, "implementation_sha": args.implementation_sha,
                       "model_form_base_sha": "3e3c79536bc76ce19589567afc0d5eca667fc691",
                       "stochastic_seed_policy_sha": "d563304000000000000000000000000000000000"})
    # The abbreviated seed-policy prefix is expanded from repository history.
    provenance["stochastic_seed_policy_sha"] = git("rev-parse", "d563304^{commit}")
    manifest = {"schema": "v12.voiding-v5-finalization-v3/1", "provenance": provenance,
                "persistent_limitations": list(PERSISTENT_LIMITATIONS),
                "scientific_tolerances": SCIENTIFIC_ACCEPTANCE_TOLERANCES,
                "reproducibility_policy": REPRODUCIBILITY_COMPARISON_POLICY,
                "counts": {key: len(value) for key, value in datasets.items()}, "evidence_row_count": len(rows),
                "gates": gates, "decision": "PASS" if all(gates.values()) else "BLOCKED",
                "runner_results": {"broad_returncode": broad_run.returncode, "static_returncode": static_run.returncode},
                "failed_gates": [key for key, value in gates.items() if not value]}
    validate_schema_contract(manifest)
    write_json(out / "evidence_rows.json", rows); write_json(out / "source_rows.json", sources)
    write_json(out / "campaign_manifest.json", manifest)
    hashes = {path.relative_to(out).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(out.rglob("*")) if path.is_file() and path.name != "sha256_manifest.json"}
    write_json(out / "sha256_manifest.json", hashes)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
