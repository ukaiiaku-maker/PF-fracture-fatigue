#!/usr/bin/env python3
"""Derive a conservative closure ledger from source-validated evidence trees.

This is a derived audit, not another physical execution. It retains the
historical lifecycle score and separately applies the stricter frozen
positive/negative-offset terminal classification to actual candidate rates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from arrhenius_fracture.closure_production_evidence import candidate_measurements
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.finalization_v3_schema import SCIENTIFIC_ACCEPTANCE_TOLERANCES as LIMITS


def read_tree(path,filename):
    manifest=json.loads((path/"sha256_manifest.json").read_text())
    actual={str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file() and p.name!="sha256_manifest.json"}
    if manifest!=actual: raise ValueError("source tree manifest mismatch")
    return json.loads((path/filename).read_text())


def derive(root,junit):
    traction=read_tree(root/"traction_source_a","traction_convergence.json")
    mechanics=read_tree(root/"mechanics_a","mechanics_matrix.json")
    lifecycle=read_tree(root/"lifecycle_a","lifecycle_rows.json")
    production=read_tree(root/"production_a","transfer_manifest.json")
    kirsch=[]
    for row in traction["base_rows"]:
        cfg=row["input_configuration"]
        if cfg["crack_enabled"]: continue
        measurement=row["measurements"]
        nominal=abs(measurement["reaction_top_N_per_m"])/cfg["domain_width_m"]
        concentration=max(float(np.asarray(edge["canonical_edge_tangent"])@
            np.asarray(edge["adjacent_element_stress_tensor_Pa"])@
            np.asarray(edge["canonical_edge_tangent"])) for edge in measurement["edge_records"])/nominal
        error=abs(concentration-3.)/3.
        kirsch.append({"source_row_id":row["case_id"],"source_execution_id":row["execution_id"],
            "source_fingerprints":row["source_fingerprints"],"boundary_segments":cfg["boundary_segments"],
            "radial_layers":cfg["radial_layers"],"measured_hoop_concentration":concentration,
            "analytic_infinite_plate_maximum":3.,"relative_error":error,
            "frozen_limit":LIMITS["kirsch_relative"],"passed":error<=LIMITS["kirsch_relative"]})
    offsets=[]
    for row in lifecycle["rows"]:
        if row["dataset"]!="controlled" or row["case_identity"] not in ("positive_offset","negative_offset"): continue
        state=restore_checkpoint(root/"lifecycle_a"/row["terminal_checkpoint"])
        candidates=candidate_measurements(state)
        zero=all(c["rates_before_resolution_guard"]["effective_rate_s"]==0. for c in candidates)
        offsets.append({"source_execution_id":row["execution_id"],"case_identity":row["case_identity"],
            "source_checkpoint":row["terminal_checkpoint"],"source_fingerprint":row["terminal_fingerprint"],
            "actual_candidate_rates_s":[c["rates_before_resolution_guard"]["effective_rate_s"] for c in candidates],
            "actual_phase":state.void_state.cavities[0].phase.value,
            "valid_connection":bool(state.junction_process_state["latest_crack_void_connection_certificate"]["passed"]),
            "frozen_controlled_terminal_passed":bool(zero and row["failure"] is None and row["conservation"]["passed"]),
            "classification":"CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE" if zero else "CONNECTED_VOID_UNQUALIFIED_POSITIVE_SOURCE"})
    failures=[]; suite=ET.parse(junit).getroot()
    for test in suite.iter("testcase"):
        if test.find("failure") is not None or test.find("error") is not None:
            failures.append(test.attrib["classname"].replace(".","/")+".py::"+test.attrib["name"])
    inherited=json.loads((ROOT/"artifacts/voiding_v5_semantic_hardening/general_ci_inheritance.json").read_text())["base"]["failure_ids"]
    decision=lifecycle["decision"]
    return {"schema":"v12.closure-source-derived-ledger/1","audit_code_sha":subprocess.check_output(
        ("git","rev-parse","HEAD"),cwd=ROOT,text=True).strip(),"decision":"BLOCKED",
        "source_code_shas":{"traction":traction["executed_code_sha"],"mechanics":mechanics["executed_code_sha"],
            "production":production["executed_code_sha"],"lifecycle":lifecycle["executed_code_sha"]},
        "kirsch_derived_only":kirsch,"strict_offset_terminal_reassessment":offsets,
        "unique_static_mechanics_solves":len(mechanics["base_rows"]),
        "static_predicates":{"passed":sum(r["predicate_result"] for r in mechanics["derived_rows"]),
            "failed":sum(not r["predicate_result"] for r in mechanics["derived_rows"])},
        "actual_lifecycle_rows":len(lifecycle["rows"]),
        "transition_attempts":45,"transitions_observed":sum(r.get("transition_occurred",False) for r in lifecycle["rows"]),
        "exact_transition_partition_gates_passed":sum(r["passed"] for r in decision["transition_partitions"]),
        "natural_partition_restart_gates_passed":sum(r["passed"] for r in decision["natural_partitions_restart"]),
        "basic_conservation_rows_passed":sum(r["conservation"]["passed"] for r in lifecycle["rows"]),
        "reached_downstream_continued_terminal":decision["required_continued_front_restart_terminal"],
        "reachable_rollback_gates_passed":sum(r["passed"] for r in decision["rollback_attempts"]),
        "complete_lifecycle_rollback":"NOT_COMPLETE_PREVOID_FAULT_HOOKS_NOT_IMPLEMENTED_DOWNSTREAM_PREREQUISITE_UNAVAILABLE",
        "generalized_length_and_stagewise_topology":"PARTIAL_BASIC_CONSERVATION_NOT_COMPLETE_CERTIFICATION",
        "local_regression":{"implementation_sha":"6ba4094c5ef42857412f6528a60e22e6dd2be52e",
            "failure_ids":failures,"inherited_failure_ids":sorted(set(failures)&set(inherited)),
            "additional_failure_ids":sorted(set(failures)-set(inherited)),"classification":"NOT_GREEN_NOT_INHERITED_ONLY"},
        "exact_head_clean_worker_full_closure":"NOT_RUN_PUBLICATION_APPROVAL_BLOCKED"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_root",type=Path); parser.add_argument("junit",type=Path)
    args=parser.parse_args(); print(json.dumps(derive(args.evidence_root,args.junit),indent=2,sort_keys=True,allow_nan=False))
