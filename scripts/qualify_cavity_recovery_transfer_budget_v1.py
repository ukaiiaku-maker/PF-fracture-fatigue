#!/usr/bin/env python3
"""Frozen timing budgets on restored owned sources; never advances a clock.

Rates below are counterfactual evaluations of the existing cleavage law before
the production source-resolution guard. They are not accepted effective rates.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arrhenius_fracture.source_resolution_protocol_v1 import (
    ALLOCATED_LOG_RATE_BUDGET, ALLOCATED_WAITING_TIME_RELATIVE_ERROR,
    TOTAL_LOG_RATE_BUDGET, MAXIMUM_TOTAL_WAITING_TIME_RELATIVE_ERROR,
    TENSOR_RELATIVE_LIMIT, PRACTICAL_LEVELS, FINE_REFERENCE,
)
from arrhenius_fracture.sharp_front import KB
from arrhenius_fracture.cavity_boundary_patch_recovery_v1 import recover_boundary_tensor

TEMPERATURE_K = 900.


def canonical(value):
    if isinstance(value, np.ndarray): return canonical(value.tolist())
    if isinstance(value, np.generic): return canonical(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0 else "-infinity" if value < 0 else "NaN"
    if isinstance(value, dict): return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [canonical(v) for v in value]
    return value


def candidate_budget(rate, reference, *, temperature_K=TEMPERATURE_K):
    """Stable transfer diagnostics; no relative rate calculation or state write."""
    if temperature_K <= 0 or not math.isfinite(temperature_K):
        raise ValueError("temperature must be positive and finite")
    if rate["candidate_id"] != reference["candidate_id"]:
        return dict(passed=False, classification="CANDIDATE_ID_MISMATCH")
    first, second = float(rate["effective_rate_s"]), float(reference["effective_rate_s"])
    result = dict(candidate_id=rate["candidate_id"], positive=False, passed=False,
                  allocated_log_rate_limit=ALLOCATED_LOG_RATE_BUDGET,
                  allocated_waiting_time_relative_limit=ALLOCATED_WAITING_TIME_RELATIVE_ERROR)
    if not all(math.isfinite(x) and x >= 0 for x in (first, second)):
        return dict(result, classification="INVALID_RATE")
    barrier_error = abs(float(rate["hazard_barrier_J"])-float(reference["hazard_barrier_J"]))/(KB*temperature_K)
    result["effective_barrier_error_over_kBT"] = barrier_error
    if not math.isfinite(barrier_error):
        return dict(result, classification="NONFINITE_BARRIER")
    if min(first, second) <= np.finfo(float).tiny:
        # Exact inactivity is admissible but never a positive-source gate.
        same_zero = first == second == 0
        return dict(result, classification="BOTH_KINETICALLY_INACTIVE" if same_zero else "INACTIVE_OR_NEAR_ZERO_MISMATCH",
                    inactive_classification_equal=same_zero,
                    passed=same_zero and barrier_error <= ALLOCATED_LOG_RATE_BUDGET,
                    positive=False, waiting_time_relative_error=None, log_rate_error=None)
    log_error = abs(math.log(first)-math.log(second))
    wait, wait_ref = float(rate["crossing_time_s"]), float(reference["crossing_time_s"])
    if not all(math.isfinite(x) and x > 0 for x in (wait, wait_ref)):
        return dict(result, classification="NONFINITE_OR_ALREADY_COMPLETED_WAITING_TIME", log_rate_error=log_error)
    # Subtraction of logs avoids unstable tiny-rate ratios. This uses the real
    # preserved remaining action from each restored clock, not a new threshold.
    difference = math.log(wait)-math.log(wait_ref)
    wait_error = abs(math.expm1(difference)) if difference < 700 else math.inf
    return dict(result, classification="POSITIVE_EXISTING_THRESHOLD", positive=True,
        log_rate_error=log_error, waiting_time_relative_error=wait_error,
        total_budget_passed=log_error <= TOTAL_LOG_RATE_BUDGET and
            barrier_error <= TOTAL_LOG_RATE_BUDGET and
            wait_error <= MAXIMUM_TOTAL_WAITING_TIME_RELATIVE_ERROR,
        passed=log_error <= ALLOCATED_LOG_RATE_BUDGET and
               barrier_error <= ALLOCATED_LOG_RATE_BUDGET and
               wait_error <= ALLOCATED_WAITING_TIME_RELATIVE_ERROR)


def compare_sources(current, reference):
    a, b = np.asarray(current["tensor_Pa"]), np.asarray(reference["tensor_Pa"])
    tensor_error = float(np.linalg.norm(a-b)/max(np.linalg.norm(b), np.finfo(float).tiny))
    identity = {key: current[key] == reference[key] for key in (
        "candidate_records", "hazard_records", "rng_state", "owned_source_identity",
        "normal_xy", "tangent_xy", "pending_event_ids", "consumed_event_ids", "stored_raw_source_provenance")}
    rates = current["unguarded_candidate_rates"]
    references = {r["candidate_id"]: r for r in reference["unguarded_candidate_rates"]}
    rows = [candidate_budget(r, references[r["candidate_id"]]) if r["candidate_id"] in references else
            dict(candidate_id=r["candidate_id"], passed=False, classification="MISSING_REFERENCE_CANDIDATE") for r in rates]
    identity["complete_candidate_set"] = len(rates) == len(references) and all(r["candidate_id"] in references for r in rates)
    return dict(tensor_relative_error=tensor_error, tensor_gate=tensor_error <= TENSOR_RELATIVE_LIMIT,
        exact_identity_checks=identity,
        full_stored_raw_source_provenance_equal=current["stored_raw_source_provenance"] == reference["stored_raw_source_provenance"],
        compared_operator_ids=[current["operator_id"], reference["operator_id"]],
        current_tensor_fingerprint=current["tensor_fingerprint"], reference_tensor_fingerprint=reference["tensor_fingerprint"],
        candidate_rows=rows, positive_candidate_exists=any(r.get("positive", False) for r in rows),
        total_transfer_budget_passed=bool(rows) and all(identity.values()) and
            tensor_error <= TENSOR_RELATIVE_LIMIT and all(r.get("total_budget_passed", r["passed"]) for r in rows),
        allocated_transfer_budget_passed=bool(rows) and all(identity.values()) and
            tensor_error <= TENSOR_RELATIVE_LIMIT and all(r["passed"] for r in rows),
        source_qualified=False,
        interpretation="TRANSFER_BUDGET_ONLY_NO_GEOMETRY_MECHANICS_TOPOLOGY_OR_EVENT_CERTIFICATE")


def evaluate_checkpoint(path):
    from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
    from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
    from arrhenius_fracture.voiding_production_v5 import (
        _actual_cavity_boundary_edges, cavity_boundary_tensor, directional_clock_rates,
    )
    from arrhenius_fracture.fem import assemble_mechanics
    state = restore_checkpoint(path)  # verifies immutable payload digest before decoding
    before = fingerprint(state)
    cavity = state.void_state.cavities[0]
    point = np.asarray(cavity.connection_exit_m)
    normal = point-np.asarray(cavity.center_m); normal /= np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0]])
    _, _, stress, *_ = assemble_mechanics(state.mesh, state.displacement, state.ep_gp,
        state.rho_gp, state.damage, state.elasticity_D, state.material,
        cohesive_network=state.cohesive_network)
    mask = np.asarray(state.mesh.element_damage_gp) if state.mesh.element_damage_gp is not None else np.zeros(state.mesh.ne)
    recovery = recover_boundary_tensor(state.mesh.nodes, state.mesh.elems, stress,
        boundary_edges=_actual_cavity_boundary_edges(state), source_point=point, normal=normal,
        intact_mask=mask == 0,
        candidate_normals=[candidate.normal_xy for candidate in state.competition.candidates])
    node = int(np.argmin(np.linalg.norm(state.mesh.nodes-point, axis=1)))
    raw_tensor, raw_elements = cavity_boundary_tensor(state, boundary_node=node)
    source = state.junction_process_state["active_event_source"]
    identity = {key: source.get(key) for key in ("source_kind", "source_front_id", "source_cavity_id",
        "source_boundary_site_id", "source_position_m", "source_geometry_generation")}
    common = canonical(dict(candidate_records=[asdict(c) for c in state.competition.candidates],
        hazard_records=[asdict(h) for h in state.competition.hazard_states], rng_state=state.rng_state,
        owned_source_identity=identity, normal_xy=normal, tangent_xy=tangent,
        pending_event_ids=[event.event_id for event in state.competition.pending_events],
        consumed_event_ids=list(state.competition.consumed_event_ids), stored_raw_source_provenance=source))
    records = {}
    for name, tensor in (("raw", raw_tensor), ("recovered", np.asarray(recovery["tensor_Pa"]))):
        records[name] = dict(common, tensor_Pa=tensor.tolist(),
            operator_id="maximum-principal-incident-CST-at-fixed-owned-boundary-node" if name == "raw" else recovery["operator_id"],
            tensor_fingerprint=hashlib.sha256(np.asarray(tensor, dtype=np.float64).tobytes()).hexdigest(),
            unguarded_candidate_rates=canonical(directional_clock_rates(state, tensor, temperature_K=TEMPERATURE_K)))
    if fingerprint(state) != before:
        raise ValueError("read-only recovered source qualification mutated accepted state")
    return dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        accepted_state_fingerprint=before, accepted_state_unchanged=True,
        records=records, recovery_stencil=recovery, raw_element_ids=list(raw_elements))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--retained", type=Path, default=ROOT/"artifacts/voiding_v5_finalization_v3_closure/complete_attempt_20260907/a/production")
    args = parser.parse_args()
    if args.output.exists(): raise ValueError("refusing to overwrite evidence")
    manifest = json.loads((args.retained/"sha256_manifest.json").read_text())
    states = {}
    for segments, layers in (*PRACTICAL_LEVELS, FINE_REFERENCE):
        key = f"{segments}/{layers}"
        path = args.retained/"checkpoints"/f"production_{segments}_{layers}_connected.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[str(path.relative_to(args.retained))]:
            raise ValueError("retained checkpoint manifest hash mismatch")
        states[key] = evaluate_checkpoint(path)
    ref = states["512/192"]["records"]
    comparisons = {key: compare_sources(row["records"]["recovered"], ref["recovered"])
                   for key, row in states.items() if key != "512/192"}
    representation = compare_sources(ref["recovered"], ref["raw"])
    total_comparisons = {key: compare_sources(row["records"]["recovered"], ref["raw"])
                         for key, row in states.items() if key != "512/192"}
    result = dict(schema="v5.recovery-owned-source-timing-transfer/1", temperature_K=TEMPERATURE_K,
        git_head=subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        clean_worktree=not subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip(),
        implementation_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_operator_sha256=hashlib.sha256((ROOT/"arrhenius_fracture/cavity_boundary_patch_recovery_v1.py").read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((ROOT/"arrhenius_fracture/source_resolution_protocol_v1.py").read_bytes()).hexdigest(),
        limits=dict(total_log_rate=TOTAL_LOG_RATE_BUDGET, total_waiting_time_relative=MAXIMUM_TOTAL_WAITING_TIME_RELATIVE_ERROR,
                    allocated_log_rate=ALLOCATED_LOG_RATE_BUDGET, allocated_waiting_time_relative=ALLOCATED_WAITING_TIME_RELATIVE_ERROR,
                    tensor_relative=TENSOR_RELATIVE_LIMIT), states=states,
        recovered_resolution_comparisons=comparisons, fine_raw_to_recovered_representation=representation,
        combined_recovered_practical_to_raw_fine=total_comparisons,
        source_qualified=False, pending_or_completed_event_created=False, child_created=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(canonical(result), sort_keys=True, indent=2, allow_nan=False)+"\n")
    print(json.dumps(canonical(dict(recovered_resolution_comparisons=comparisons,
                                   fine_raw_to_recovered_representation=representation)), indent=2))


if __name__ == "__main__": main()
