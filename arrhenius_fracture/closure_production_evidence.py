"""Checkpoint-bound production transfer validation, including failed attempts.

Only trusted, locally produced checkpoints may be loaded by the caller. A
valid evidence object is not a scientific PASS. No rate/barrier mesh-transfer
limit is invented here, and a failed fine connection cannot be a reference.
"""
from dataclasses import asdict
import math
import numpy as np

from .closure_mechanics_evidence import canonical_data
from .topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
from .voiding_production_v5 import (
    cavity_boundary_tensor, directional_clock_rates, downstream_front_transaction,
    ligament_transaction,
    cavity_source_resolution_metrics,
    refine_downstream_source,
)
from .finalization_v3_schema import SCIENTIFIC_ACCEPTANCE_TOLERANCES as LIMITS

SCHEMA = "v12.production-source-transfer/2"
LEVELS = ((32,12),(64,24),(128,48),(512,192))
PATH = ((0.,0.),(.0005725993004046688,0.))
REFINEMENT_CASES = {
    "production:32:12": {"max_refinement_levels":3,"refinement_region":"complete_cavity_ring",
                          "quality_improvement":False},
    "production:512:192": {"max_refinement_levels":1,"refinement_region":"complete_cavity_ring",
                            "quality_improvement":True},
}


def execute_refinement_peer(connected, configuration):
    """Actual bounded negative/positive-peer attempt, including failed science."""
    initial = fingerprint(connected); operations = []
    result = {"configuration":configuration,"initial_state_fingerprint":initial}
    try:
        refined,audit = refine_downstream_source(connected,**configuration,operation_log=operations)
        result.update({"audit":audit,"refined_state_fingerprint":fingerprint(refined),
            "qualified":audit["status"] == "SOURCE_TENSOR_QUALIFIED",
            "preserved_clock_and_rng":refined.competition == connected.competition and refined.rng_state == connected.rng_state})
        try:
            advanced,trial,trace,source_audit = downstream_front_transaction(refined)
            result.update({"event_attempted":True,"event_accepted":trial is not None and trial.accepted,
                "event_state_fingerprint":fingerprint(advanced),"event_operations":trace,"event_audit":source_audit})
        except Exception as error:
            result.update({"event_attempted":True,"event_accepted":False,
                "event_failure":{"type":type(error).__name__,"message":str(error)}})
    except Exception as error:
        result.update({"qualified":False,"refinement_failure":{"type":type(error).__name__,"message":str(error)}})
    result["refinement_operations"] = operations
    result["caller_restored_exactly"] = fingerprint(connected) == initial
    return canonical_data(result)


def transfer_comparisons(rows):
    reference = rows[-1]; comparisons = []
    for row in rows[:-1]:
        result = {"case_id": row["case_id"], "reference_case_id": reference["case_id"], "qualified": False}
        if row["connection_executed"] and reference["connection_executed"]:
            peer = {r["candidate_identity"]["candidate_id"]: r for r in reference["candidates"]}
            errors = []
            for candidate in row["candidates"]:
                identity = candidate["candidate_identity"]["candidate_id"]; ref = peer.get(identity)
                if ref is None:
                    errors.append({"candidate_id": identity, "failure": "CANDIDATE_NOT_IN_REFERENCE"}); continue
                tensor_error = float(np.linalg.norm(np.asarray(candidate["source_tensor_Pa"])-ref["source_tensor_Pa"])/np.linalg.norm(ref["source_tensor_Pa"]))
                scalar_errors = {}
                for field in ("hazard_barrier_J", "raw_rate_s", "effective_rate_s", "crossing_time_s"):
                    a, b = candidate["rates_before_resolution_guard"][field], ref["rates_before_resolution_guard"][field]
                    if isinstance(a, str): a = float(a)
                    if isinstance(b, str): b = float(b)
                    scalar_errors[field] = 0. if a == b else abs(a-b)/max(abs(b), 1e-300)
                errors.append({"candidate_id": identity, "tensor_relative_error": tensor_error,
                    "candidate_identity_equal": candidate["candidate_identity"] == ref["candidate_identity"],
                    "threshold_equal": candidate["hazard"]["current_threshold_action"] == ref["hazard"]["current_threshold_action"],
                    "scalar_relative_errors": scalar_errors, "tensor_gate": tensor_error <= LIMITS["tensor_probe_relative"]})
            result["errors"] = errors
            result["classification"] = "DIAGNOSTIC_TRANSFER_ONLY_NO_RATE_BARRIER_ACCEPTANCE_LIMIT_IN_FROZEN_REGISTRY"
        else:
            result["classification"] = "FAIL_CLOSED_CONNECTION_OR_MATCHED_FINE_REFERENCE_UNAVAILABLE"
        comparisons.append(result)
    return canonical_data(comparisons)


def candidate_measurements(state):
    cavity=state.void_state.cavities[0]
    position=np.asarray(cavity.connection_exit_m)
    node=int(np.argmin(np.linalg.norm(state.mesh.nodes-position,axis=1)))
    tensor,elements=cavity_boundary_tensor(state,boundary_node=node)
    metrics = cavity_source_resolution_metrics(state)
    delta=position-np.asarray(cavity.center_m)
    arc=(math.atan2(delta[1],delta[0])%(2*math.pi))/(2*math.pi)
    return canonical_data([
        {"candidate_identity":asdict(candidate),"boundary_site":"connection_exit",
         "boundary_position_m":position.tolist(),"arc_fraction":arc,
         "probe_element_ids":list(elements),"source_tensor_Pa":tensor.tolist(),
         "source_resolution_metrics":metrics,
         "candidate_plane_shear_Pa":float(np.asarray(candidate.normal_xy)@tensor@np.asarray(candidate.direction_xy)),
         "hazard":asdict(hazard),"rates_before_resolution_guard":rate,
         "owned_source":state.junction_process_state["active_event_source"]}
        for candidate,hazard,rate in zip(state.competition.candidates,
            state.competition.hazard_states,directional_clock_rates(state,tensor))])


def validate_production(payload,sources,*,executed_code_sha):
    manifest=payload["transfer_manifest"]; rows=payload["rows"]
    if manifest["schema"]!=SCHEMA or manifest["executed_code_sha"]!=executed_code_sha:
        raise ValueError("production transfer implementation/schema mismatch")
    expected=[f"production:{n}:{r}" for n,r in LEVELS]
    if [row["case_id"] for row in rows]!=expected:
        raise ValueError("production physical registry mismatch")
    if manifest["physical_registry"]!=[list(level) for level in LEVELS]:
        raise ValueError("production resolution registry mismatch")
    passed=0; failures=0
    for row,(n,r) in zip(rows,LEVELS):
        if row["executed_code_sha"]!=executed_code_sha or row["first_passage_qualified"]:
            raise ValueError("production row provenance/qualification mismatch")
        cfg={"crack_path_m":PATH,"cavity_center_m":(7e-4,0.),
             "boundary_segments":n,"radial_layers":r,"seed":3621}
        if row["input_configuration"]!=canonical_data(cfg):
            raise ValueError("production input configuration mismatch")
        prefix="checkpoints/"+row["case_id"].replace(":","_")
        before=sources[prefix+"_pre.json"]
        if canonical_data(before.crack_network.branches[0].path)!=canonical_data(PATH):
            raise ValueError("production fixed crack is not checkpoint geometry")
        if not row["connection_executed"]:
            # Re-execute the actual failed ligament trial from its saved
            # accepted state; an authored exception string is not evidence.
            original=fingerprint(before)
            try: ligament_transaction(before)
            except Exception as error:
                observed={"type":type(error).__name__,"message":str(error)}
                if any(row["failure"][key]!=value for key,value in observed.items()):
                    raise ValueError("production failure does not reproduce") from error
            else: raise ValueError("claimed failed connection actually succeeds")
            if fingerprint(before)!=original:
                raise ValueError("failed fine trial mutated accepted checkpoint")
            failures+=1; continue
        connected=sources[prefix+"_connected.json"]
        if row["initial_state_fingerprint"]!=fingerprint(before) or row["connected_state_fingerprint"]!=fingerprint(connected):
            raise ValueError("production checkpoint fingerprint mismatch")
        if row["candidates"]!=candidate_measurements(connected):
            raise ValueError("production candidate tensor/rate/source mismatch")
        guarded,trial,operations,audit=downstream_front_transaction(connected)
        restored=sources[prefix+"_connected.json"]
        replay,replay_trial,_,replay_audit=downstream_front_transaction(restored)
        derived={"guarded_state_fingerprint":fingerprint(guarded),
            "guard_audit":canonical_data(audit),"guard_operations":canonical_data(operations),
            "guard_preserves_complete_state":fingerprint(guarded)==fingerprint(connected),
            "checkpoint_restart_exact":fingerprint(replay)==fingerprint(guarded) and replay_audit==audit,
            "child_created":len(guarded.crack_network.branches)!=len(connected.crack_network.branches),
            "active_tip_ids":list(guarded.crack_network.active_tip_ids),
            "event_transaction_created":trial is not None or replay_trial is not None}
        if any(row.get(key)!=value for key,value in derived.items()):
            raise ValueError("production guard claim differs from actual replay")
        if row["case_id"] in REFINEMENT_CASES:
            observed = execute_refinement_peer(connected,REFINEMENT_CASES[row["case_id"]])
            if row.get("source_refinement_peer") != observed:
                raise ValueError("source refinement peer does not reproduce from its owned checkpoint")
        elif "source_refinement_peer" in row:
            raise ValueError("unregistered source refinement peer")
        passed+=1
    # Successful fine connections are now derived evidence, not an authored
    # rejection requirement. Comparison never grants event authority by itself.
    comparisons = transfer_comparisons(rows)
    if manifest["comparisons"]!=comparisons or manifest["production_resolution_cavity_tensor_qualified"] is not False:
        raise ValueError("production reference qualification falsely claimed")
    if manifest["policy"]!="C_FIRST_PASSAGE_UNAVAILABLE_UNTIL_QUALIFIED":
        raise ValueError("production fail-closed policy mismatch")
    return {"valid":True,"physical_cases":len(rows),"connected_checkpoint_replays":passed,
            "reproduced_failed_connections":failures,"production_tensor_qualified":False}
