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
)

SCHEMA = "v12.production-source-transfer/1"
LEVELS = ((32,12),(64,24),(128,48),(512,192))
PATH = ((0.,0.),(.0005725993004046688,0.))


def candidate_measurements(state):
    cavity=state.void_state.cavities[0]
    position=np.asarray(cavity.connection_exit_m)
    node=int(np.argmin(np.linalg.norm(state.mesh.nodes-position,axis=1)))
    tensor,elements=cavity_boundary_tensor(state,boundary_node=node)
    delta=position-np.asarray(cavity.center_m)
    arc=(math.atan2(delta[1],delta[0])%(2*math.pi))/(2*math.pi)
    return canonical_data([
        {"candidate_identity":asdict(candidate),"boundary_site":"connection_exit",
         "boundary_position_m":position.tolist(),"arc_fraction":arc,
         "probe_element_ids":list(elements),"source_tensor_Pa":tensor.tolist(),
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
        passed+=1
    # The current fine reference fails. A future successful fine reference
    # needs a prospectively specified comparison contract, not this fail-closed
    # validator silently accepting newly authored errors or tolerances.
    if rows[-1]["connection_executed"]:
        raise ValueError("successful fine reference requires registered transfer predicates")
    comparisons=[{"case_id":row["case_id"],"reference_case_id":rows[-1]["case_id"],
        "qualified":False,"classification":"FAIL_CLOSED_CONNECTION_OR_MATCHED_FINE_REFERENCE_UNAVAILABLE"}
        for row in rows[:-1]]
    if manifest["comparisons"]!=comparisons or manifest["production_resolution_cavity_tensor_qualified"] is not False:
        raise ValueError("production reference qualification falsely claimed")
    if manifest["policy"]!="C_FIRST_PASSAGE_UNAVAILABLE_UNTIL_QUALIFIED":
        raise ValueError("production fail-closed policy mismatch")
    return {"valid":True,"physical_cases":len(rows),"connected_checkpoint_replays":passed,
            "reproduced_failed_connections":failures,"production_tensor_qualified":False}
