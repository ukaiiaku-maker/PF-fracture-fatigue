#!/usr/bin/env python3
"""Reduce the retained V5 closure archive to bounded root-cause diagnostics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


FAILED_CONTROLLED = {
    "delayed_downstream": ("DOWNSTREAM_FRONT_ACTIVE_AFTER_DORMANT_INTERVAL_AND_RELOAD",
                           "case_construction_or_source_qualification"),
    "diffusion_limited": ("STABLE_SUBGRID_VOID_WITH_VACANCY_TRANSPORT_MINIMUM_ALL_INTERVALS",
                          "case_construction_or_physics"),
    "downstream_zero_drive": ("CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE_WITH_QUALIFIED_SOURCE",
                              "source_qualification"),
    "fixed_mesh_oblique": ("CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE",
                           "expectation"),
    "local_remesh_refinement": ("DOWNSTREAM_FRONT_ACTIVE_AFTER_QUALIFIED_LOCAL_REFINEMENT",
                                "source_qualification"),
    "long_ligament": ("DOWNSTREAM_FRONT_ACTIVE", "expectation"),
    "negative_offset": ("CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE_WITH_QUALIFIED_SOURCE",
                        "source_qualification"),
    "short_ligament": ("DOWNSTREAM_FRONT_ACTIVE", "expectation"),
}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def recursive_differences(first, second, path=""):
    if type(first) is not type(second):
        return [{"path": path, "reference": first, "peer": second}]
    if isinstance(first, dict):
        rows = []
        for key in sorted(set(first) | set(second)):
            child = path + "/" + key
            if key not in first or key not in second:
                rows.append({"path": child, "reference": first.get(key), "peer": second.get(key)})
            else:
                rows.extend(recursive_differences(first[key], second[key], child))
        return rows
    if isinstance(first, list):
        rows = []
        if len(first) != len(second):
            rows.append({"path": path + "/#length", "reference": len(first), "peer": len(second)})
        for index, (left, right) in enumerate(zip(first, second)):
            rows.extend(recursive_differences(left, right, path + "/" + str(index)))
        return rows
    return [] if first == second else [{"path": path, "reference": first, "peer": second}]


def tensor_sequence(rows, selector):
    result = []
    for row in rows:
        value = selector(row)
        result.append(value)
    return result


def static_diagnostics(report):
    families = report["decision"]["families"]
    if isinstance(families, dict):
        families = list(families.values())
    output = []
    for family in families:
        rows = [report["rows"][source] for source in family["source_ids"]]
        measures = [row["measurements"] for row in rows]
        failed = [key for key, value in family.get("gates", {}).items() if not value]
        output.append({
            "family": family["family"],
            "source_ids": family["source_ids"],
            "requested_geometry": [{key: row["input_configuration"].get(key) for key in
                ("geometry_mode", "crack_path_m", "cavity_center_m", "cavity_radius_m",
                 "opening_m", "boundary_segments", "radial_layers")} for row in rows],
            "realized_geometry": [{
                "geometry_generation": row["geometry_conformity_audit"]["geometry_generation"],
                "maximum_requested_realized_error_m":
                    row["geometry_conformity_audit"]["maximum_requested_realized_error_m"],
                "root_boundary_component": row["geometry_conformity_audit"]["root_boundary_component"],
            } for row in rows],
            "mesh_quality_sequence": [item["mesh_quality"] for item in measures],
            "two_quality_valid_fine_levels": family.get("gates", {}).get("two_quality_valid_fine_levels"),
            "reaction_sequence": [item["reaction"] for item in measures],
            "compliance_sequence": [item["compliance"] for item in measures],
            "energy_sequence": [item["energy"] for item in measures],
            "raw_full_boundary_traction_sequence": [
                item.get("cavity_fields", {}).get("normalized_traction") for item in measures],
            "fixed_tip_tensor_sequence_Pa": tensor_sequence(rows,
                lambda row: row["measurements"]["fixed_tip_probe"].get("tensor_Pa")),
            "cavity_fixed_arc_tensor_sequences_Pa": [[{
                "arc_fraction": arc["arc_fraction"],
                "tensor_Pa": arc.get("recovery", {}).get("tensor_Pa"),
                "status": arc.get("recovery", {}).get("status"),
            } for arc in row["recovery"]] for row in rows],
            "convergence": family.get("convergence", {}),
            "gates": family.get("gates", {}),
            "failed_predicates": failed,
            "first_failed_predicate": failed[0] if failed else None,
            "passed": family["passed"],
        })
    return {"schema": "v5.retained-static-family-diagnostics/1",
            "executed_code_sha": report["executed_code_sha"], "families": output,
            "family_count": len(output), "derivatives": report["decision"]["derivatives"],
            "passed": report["decision"]["passed"]}


def compact_audit(audit):
    if not isinstance(audit, dict):
        return audit
    return {"status": audit.get("status"), "source_kind": audit.get("source_kind"),
            "source_front_id": audit.get("source_front_id"),
            "cleavage": [{key: row.get(key) for key in
                ("candidate_id", "instantaneous_status", "resolved_opening_stress_Pa",
                 "effective_rate_s", "crossing_time_s", "current_threshold_action", "action", "winner")}
                for row in audit.get("cleavage", [])]}


def controlled_diagnostics(lifecycle):
    verdicts = {row["case_identity"]: row for row in lifecycle["decision"]["controlled_histories"]}
    rows = []
    for row in lifecycle["rows"]:
        if row["dataset"] != "controlled" or row["case_identity"] not in FAILED_CONTROLLED:
            continue
        expected, failure_class = FAILED_CONTROLLED[row["case_identity"]]
        trace = []
        for operation in row["actual_operations"]:
            trace.append({key: operation.get(key) for key in ("api", "events", "accepted", "opening_m",
                          "duration_s") if key in operation} | {"audit": compact_audit(operation.get("audit"))})
        actual = verdicts[row["case_identity"]]
        first = ("runtime_failure:" + row["failure"]["message"] if row.get("failure") else
                 "v1_expected_terminal_predicate_mismatch")
        rows.append({
            "case_identity": row["case_identity"],
            "prospective_v2_expected_terminal_classification": expected,
            "actual_v1_terminal_classification": actual["actual_phase"],
            "actual_v1_passed": actual["passed"],
            "first_failed_predicate": first,
            "failure_classification": failure_class,
            "failure": row.get("failure"), "input_configuration": row["input_configuration"],
            "operation_trace": trace, "conservation": row["conservation"],
            "topology_passed": row["stagewise_topology"]["passed"],
        })
    return {"schema": "v5.retained-controlled-history-diagnostics/1",
            "executed_code_sha": lifecycle["executed_code_sha"],
            "criteria_frozen_before_any_reexecution": True,
            "rows": rows, "case_count": len(rows)}


def natural_diagnostics(lifecycle):
    natural = [row for row in lifecycle["rows"] if row["dataset"] == "natural"]
    result = []
    paths = Counter()
    for case in sorted({row["case_identity"] for row in natural}, key=int):
        peers = {row["partition_count"]: row for row in natural if row["case_identity"] == case}
        for parts in (2, 4, 8, 16):
            differences = recursive_differences(peers[1]["terminal_measurements"],
                                                peers[parts]["terminal_measurements"])
            for item in differences:
                paths[item["path"]] += 1
            if differences:
                result.append({"seed": int(case), "partition_count": parts,
                               "first_difference": differences[0],
                               "difference_count": len(differences)})
    return {"schema": "v5.retained-natural-first-differences/1",
            "executed_code_sha": lifecycle["executed_code_sha"],
            "peer_count": 128, "failed_peer_count": len(result),
            "difference_path_counts": dict(paths), "failed_peers": result}


def neutrality_diagnostics(report):
    classification = {
        "latest_constrained_reaction_l2_N_per_m": "solver_diagnostic_noncausal_if_future_state_and_event_exact",
        "latest_energy_reaction_identity": "solver_diagnostic_noncausal_if_future_state_and_event_exact",
        "latest_free_dof_residual_l2_N_per_m": "solver_diagnostic_noncausal_if_future_state_and_event_exact",
        "latest_top_bottom_reaction_balance": "solver_diagnostic_noncausal_if_future_state_and_event_exact",
        "v12_boundary_terminal_certificates": "empty_representation_provenance_noncausal_if_recomputed_topology_exact",
        "boundary_terminal_certificates": "empty_representation_provenance_noncausal_if_recomputed_topology_exact",
        "source_commit": "audit_provenance_noncausal",
        "void_state": "disabled_none_representation_noncausal",
    }
    rows = []
    for row in report["rows"]:
        rows.append({"case_id": row["case_id"], "raw_differences": row["raw_differences"],
            "causal_differences": row["causal_differences"],
            "observed_causal_states_exact": row["observed_causal_states_exact"],
            "observed_histories_exact": row["observed_histories_exact"],
            "future_event_counts": row["observed_future_event_counts"],
            "legacy_v1_passed": row["passed"]})
    return {"schema": "v5.retained-disabled-neutrality-diagnostics/1",
            "implementation_sha": report["implementation_sha"],
            "historical_raw_full_state_identity": report["historical_raw_full_state_identity"],
            "field_classifications": classification,
            "consumer_occurrences": report["consumer_occurrences"], "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_stream_review", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite diagnostics")
    selected = json.loads(args.full_stream_review.read_text())["selected_original_records"]
    args.output.mkdir(parents=True)
    outputs = {
        "static_family_diagnostics.json": static_diagnostics(selected["a/static/report.json"]),
        "natural_partition_first_differences.json": natural_diagnostics(
            selected["a/lifecycle/lifecycle_rows.json"]),
        "controlled_history_diagnostics.json": controlled_diagnostics(
            selected["a/lifecycle/lifecycle_rows.json"]),
        "disabled_neutrality_diagnostics.json": neutrality_diagnostics(
            selected["a/causal-neutrality/evidence/report.json"]),
    }
    for name, value in outputs.items():
        write_json(args.output / name, value)
    manifest = {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                for name in outputs}
    write_json(args.output / "sha256_manifest.json", manifest)
    print(json.dumps({"static_families": outputs["static_family_diagnostics.json"]["family_count"],
                      "natural_failures": outputs["natural_partition_first_differences.json"]["failed_peer_count"],
                      "controlled_failures": outputs["controlled_history_diagnostics.json"]["case_count"],
                      "neutrality_cases": len(outputs["disabled_neutrality_diagnostics.json"]["rows"])},
                     sort_keys=True))


if __name__ == "__main__":
    main()
