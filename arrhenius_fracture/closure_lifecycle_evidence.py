"""Actual-state closure attempts, including fail-closed prerequisites.

This contract separates evidence validity from a successful scientific gate.
It never invents an unavailable child checkpoint or a successful operation
trace. Every reported endpoint must resolve to a complete accepted FEM state.
"""
from dataclasses import replace
import math
import numpy as np

from .finalization_v3_schema import FROZEN_CASE_REGISTRY, canonical_hash
from .topology_transaction_v11 import (
    LiveFEMTopologyState, complete_accepted_state_fingerprint as fingerprint,
    equilibrate_fixed_load_with_production_fem as equilibrate,
)
from .voiding_production_v5 import (
    _geometry, _grow_hole_boundary, _complete_next_clock,
    local_site_tensor, cavity_boundary_tensor, crack_tip_tensor, directional_clock_rates,
    ligament_transaction, downstream_front_transaction, remesh_cavity,
)
from .voiding_v5 import (
    VoidingConfig, VoidPhase, advance_site, arrhenius_rates, create_subgrid_cavity,
    update_cavity_growth, promote_cavity,
)

SCHEMA = "v12.voiding-v5-closure-actual-lifecycle/1"
PARTITIONS = (1,2,4,8,16)
CFG = VoidingConfig(enabled=True, promotion_radius_m=5e-5)
PRECURSORS = {"birth_hit_1": "available_site", "birth_hit_2": "multi_hit_1",
              "stabilization": "multi_hit_2", "healing": "multi_hit_2",
              "subgrid_growth": "subgrid_void", "promotion": "subgrid_growth",
              "ligament": "resolved_growth", "downstream_child": "ligament_rupture",
              "child_continuation": "new_graph_front"}


def load_state(state, opening_m):
    u = state.displacement.copy()
    u[2*np.asarray(state.boundary.top_nodes)+1] = opening_m/2
    u[2*np.asarray(state.boundary.bot_nodes)+1] = -opening_m/2
    return equilibrate(replace(state, displacement=u))


def advance_transition(state, name, partitions, *, operations=None, config=CFG):
    """Execute a real stage from its accepted predecessor at a fixed load."""
    operations = [] if operations is None else operations
    initial = state
    if name in ("birth_hit_1", "birth_hit_2", "stabilization", "healing"):
        rates = arrhenius_rates(config, temperature_K=900., stress_tensor_Pa=local_site_tensor(state))
        site = state.void_state.sites[0]
        clock = site.birth if name.startswith("birth") else getattr(site, name)
        channel = "birth_s" if name.startswith("birth") else name+"_s"
        effective = rates[channel]*(site.candidate_weight if name.startswith("birth") else 1.)
        total = clock.crossing_time(effective)
        if not math.isfinite(total): raise RuntimeError("NO_KINETICALLY_ACTIVE_CANDIDATE")
        for part in range(partitions):
            dt = total/partitions if part+1 < partitions else total-total/partitions*(partitions-1)
            if part+1 == partitions:
                current = state.void_state.sites[0]
                owned = current.birth if name.startswith("birth") else getattr(current,name)
                # Localize the final real first passage from the remaining
                # owned clock, not a rounded sum of nominal interval lengths.
                # The actual duration is recorded for the event-time gate.
                dt = owned.crossing_time(effective)
            voids, events = advance_site(state.void_state, site.site_id, dt, rates=rates)
            state = replace(state, void_state=voids)
            operations.append({"api": "advance_site", "duration_s": dt, "rates": rates, "events": list(events)})
        return equilibrate(state), operations
    if name == "subgrid_growth":
        rates = arrhenius_rates(config, temperature_K=900., stress_tensor_Pa=local_site_tensor(state))
        cavity = state.void_state.cavities[0]
        total = (5e-5-cavity.radius_m)/(config.radial_growth_scale_m*rates["series_limited_growth_s"])
        for part in range(partitions):
            dt = total/partitions if part+1 < partitions else total-total/partitions*(partitions-1)
            state = replace(state, void_state=update_cavity_growth(state.void_state, cavity.cavity_id,
                rates=rates, dt_s=dt, radial_growth_scale_m=config.radial_growth_scale_m))
            operations.append({"api": "update_cavity_growth", "duration_s": dt, "rates": rates})
        return equilibrate(state), operations
    if name == "promotion":
        cavity = state.void_state.cavities[0]
        voids = promote_cavity(state.void_state, cavity.cavity_id, config.promotion_radius_m)
        hole, _ = _geometry(radius_m=cavity.radius_m, center_m=cavity.center_m)
        hole = _grow_hole_boundary(hole, cavity.radius_m, crack_path_m=state.crack_network.branches[0].path)
        trace = []
        state = remesh_cavity(state, hole, voids, "promotion", trace)
        operations.append({"api": "remesh_cavity", "operations": trace})
        return state, operations
    if name == "ligament":
        tensor, ids = crack_tip_tensor(state, branch_id=state.crack_network.branches[0].branch_id)
        rates = directional_clock_rates(state,tensor)
        total = min(r["crossing_time_s"] for r in rates)
        if not math.isfinite(total): raise RuntimeError("NO_KINETICALLY_ACTIVE_CANDIDATE")
        # Leave the final actual crossing to the real topology transaction.
        for _ in range(partitions-1):
            state, audit = _complete_next_clock(state,tensor,maximum_advance_duration_s=total/partitions,
                source_kind="sharp_front",source_front_id=state.crack_network.branches[0].branch_id,
                source_position_m=state.crack_network.branches[0].tip,
                source_probe_identity={"kind":"crack_tip_tensor","element_ids":list(ids)})
            if any(r["winner"] for r in audit):
                raise RuntimeError("unexpected early threshold crossing in pretransaction partition")
            operations.append({"api": "_complete_next_clock", "duration_s":total/partitions,"audit":audit})
        trace = []
        state, trial = ligament_transaction(state,operation_log=trace)
        operations.append({"api":"ligament_transaction", "accepted":trial.accepted,"operations":trace})
        return state, operations
    if name in ("downstream_child", "child_continuation"):
        for _ in range(partitions):
            state, trial, trace, audit = downstream_front_transaction(state,continuation=name=="child_continuation")
            operations.append({"api":"downstream_front_transaction", "accepted":trial is not None and trial.accepted,
                               "operations":trace,"audit":audit})
            if trial is not None: break
        return state, operations
    raise ValueError(name)


def transition_occurred(name, before, after):
    old,new = before.void_state.sites[0],after.void_state.sites[0]
    if name == "birth_hit_1": return new.hits == old.hits+1 and new.phase == VoidPhase.AVAILABLE_SITE
    if name == "birth_hit_2": return new.hits == old.hits+1 and new.phase == VoidPhase.EMBRYO
    if name == "stabilization": return old.phase == VoidPhase.EMBRYO and new.phase == VoidPhase.STABLE_SUBGRID_VOID
    if name == "healing": return old.phase == VoidPhase.EMBRYO and new.phase == VoidPhase.HEALED_SITE
    if not before.void_state.cavities or not after.void_state.cavities: return False
    a,b = before.void_state.cavities[0],after.void_state.cavities[0]
    if name == "subgrid_growth": return b.radius_m > a.radius_m and after.void_state.consumed_defect_inventory_area_m2 > before.void_state.consumed_defect_inventory_area_m2
    if name == "promotion": return a.phase == VoidPhase.STABLE_SUBGRID_VOID and b.phase == VoidPhase.RESOLVED_VOID and before.mesh.ne != after.mesh.ne
    if name == "ligament":
        def consumed(state):
            return {key for key,value in state.junction_process_state.get("directional_event_provenance",{}).items()
                    if value.get("status")=="CONSUMED_AT_OWNED_SOURCE"}
        # The new cavity source correctly owns a fresh competition. The root
        # event's consumed identity is retained in the source-provenance ledger,
        # not in that fresh competition's empty consumed-event tuple.
        return (a.phase == VoidPhase.RESOLVED_VOID and b.phase == VoidPhase.CONNECTED_VOID
                and not after.crack_network.active_tip_ids and bool(consumed(after)-consumed(before)))
    if name == "downstream_child": return len(after.crack_network.branches)==len(before.crack_network.branches)+1 and b.phase == VoidPhase.DOWNSTREAM_FRONT_ACTIVE and len(after.crack_network.active_tip_ids)==1
    if name == "child_continuation": return len(before.crack_network.active_tip_ids)==1 and a.phase == VoidPhase.DOWNSTREAM_FRONT_ACTIVE and after.crack_network.total_physical_crack_length_m > before.crack_network.total_physical_crack_length_m
    raise ValueError(name)


def conservation(state, initial):
    voids = state.void_state
    if voids is None:
        return {"void_inventory": "NOT_APPLICABLE_DISABLED", "passed": initial.void_state is None
            and tuple(state.crack_network.active_tip_ids)==tuple(state.v12_support_state.active_tip_identities)}
    total = voids.available_defect_inventory_area_m2+voids.consumed_defect_inventory_area_m2
    original = initial.void_state.available_defect_inventory_area_m2+initial.void_state.consumed_defect_inventory_area_m2
    inventory_error = abs(total-original)
    radius_error = max((abs(c.area_m2-math.pi*c.radius_m**2) for c in voids.cavities),default=0.)
    active = tuple(state.crack_network.active_tip_ids)
    support = tuple(state.v12_support_state.active_tip_identities)
    ledger = voids.length_ledgers
    added = state.crack_network.total_physical_crack_length_m-initial.crack_network.total_physical_crack_length_m
    expected = (ledger["fractured_ligament_length_m"]+ledger["ordinary_crack_fractured_length_m"]
        -initial.void_state.length_ledgers["fractured_ligament_length_m"]-initial.void_state.length_ledgers["ordinary_crack_fractured_length_m"])
    length_error = abs(added-expected)
    return {"inventory_error_m2":inventory_error,"cavity_area_identity_error_m2":radius_error,
            "length_error_m":length_error,"active_tip_ids":list(active),"support_tip_ids":list(support),
            "single_void":len(voids.cavities)<=1,
            "passed":inventory_error<=1e-24 and radius_error<=1e-24 and length_error<=1e-15 and active==support and len(active)<=1 and len(voids.cavities)<=1}


def resume_to_guard(state, operations):
    """Resume the actual accepted lifecycle to its attainable terminal state."""
    while state.void_state.sites[0].phase == VoidPhase.AVAILABLE_SITE:
        name = "birth_hit_1" if state.void_state.sites[0].hits == 0 else "birth_hit_2"
        previous = fingerprint(state)
        state,_ = advance_transition(state,name,1,operations=operations)
        if fingerprint(state)==previous: raise RuntimeError("birth resume made no progress")
    if state.void_state.sites[0].phase == VoidPhase.EMBRYO:
        state,_ = advance_transition(state,"stabilization",1,operations=operations)
    if state.void_state.sites[0].phase == VoidPhase.HEALED_SITE: return state
    if not state.void_state.cavities:
        state = equilibrate(replace(state,void_state=create_subgrid_cavity(state.void_state,"site-1",2.5e-5)))
        operations.append({"api":"create_subgrid_cavity","radius_m":2.5e-5})
    cavity = state.void_state.cavities[0]
    if cavity.radius_m < 5e-5:
        state,_ = advance_transition(state,"subgrid_growth",1,operations=operations)
    if state.void_state.cavities[0].phase == VoidPhase.STABLE_SUBGRID_VOID:
        state,_ = advance_transition(state,"promotion",1,operations=operations)
    cavity = state.void_state.cavities[0]
    if cavity.phase == VoidPhase.RESOLVED_VOID and cavity.radius_m < 5.5e-5:
        hole,_ = _geometry(radius_m=cavity.radius_m,center_m=cavity.center_m)
        hole = _grow_hole_boundary(hole,5.5e-5,crack_path_m=state.crack_network.branches[0].path)
        rates = arrhenius_rates(CFG,temperature_K=900.,stress_tensor_Pa=cavity_boundary_tensor(state)[0])
        dt = (5.5e-5-cavity.radius_m)/(CFG.radial_growth_scale_m*rates["series_limited_growth_s"])
        voids = update_cavity_growth(state.void_state,cavity.cavity_id,rates=rates,dt_s=dt,
                                     radial_growth_scale_m=CFG.radial_growth_scale_m)
        trace=[]; state=remesh_cavity(state,hole,voids,"resolved-growth",trace)
        operations.append({"api":"resolved_growth_remesh","duration_s":dt,"rates":rates,"operations":trace})
    if state.void_state.cavities[0].phase == VoidPhase.RESOLVED_VOID:
        state,_ = advance_transition(state,"ligament",1,operations=operations)
    if state.void_state.cavities[0].phase == VoidPhase.CONNECTED_VOID:
        state,_ = advance_transition(state,"downstream_child",1,operations=operations)
    if state.void_state.cavities[0].phase == VoidPhase.DOWNSTREAM_FRONT_ACTIVE:
        state,_ = advance_transition(state,"child_continuation",1,operations=operations)
    return state


def validate_lifecycle(payload, sources, *, executed_code_sha):
    if payload["schema"] != SCHEMA or payload["executed_code_sha"] != executed_code_sha:
        raise ValueError("lifecycle source/schema identity")
    rows = payload["rows"]
    if len({r["execution_id"] for r in rows}) != len(rows): raise ValueError("aliased lifecycle execution")
    expected = {(case,p) for case in FROZEN_CASE_REGISTRY["transitions"] for p in PARTITIONS}
    transitions = [r for r in rows if r["dataset"]=="transitions"]
    if len(transitions)!=45 or {(r["case_identity"],r["partition_count"]) for r in transitions} != expected:
        raise ValueError("actual transition registry mismatch")
    for dataset in ("restarts","controlled","neutrality"):
        observed = [r["case_identity"] for r in rows if r["dataset"]==dataset]
        if sorted(observed) != sorted(FROZEN_CASE_REGISTRY[dataset]):
            raise ValueError("actual "+dataset+" registry mismatch")
    natural=[r for r in rows if r["dataset"]=="natural"]
    if len(natural)!=160 or {(int(r["case_identity"]),r["partition_count"]) for r in natural} != {
        (seed,p) for seed in range(12000,12032) for p in PARTITIONS}:
        raise ValueError("natural seed/partition registry mismatch")
    for row in rows:
        before, after = sources[row["initial_checkpoint"]],sources[row["terminal_checkpoint"]]
        if not isinstance(before,LiveFEMTopologyState) or not isinstance(after,LiveFEMTopologyState):
            raise ValueError("not complete accepted FEM captures")
        if row["initial_fingerprint"] != fingerprint(before) or row["terminal_fingerprint"] != fingerprint(after):
            raise ValueError("actual accepted checkpoint mismatch")
        if row["executed_code_sha"] != executed_code_sha or row["input_hash"] != canonical_hash(row["input_configuration"]):
            raise ValueError("actual execution input/source mismatch")
        if row["conservation"] != conservation(after,before): raise ValueError("stagewise conservation recomputation")
        if row["dataset"] == "transitions":
            occurred = transition_occurred(row["case_identity"],before,after)
            if row["transition_occurred"] != occurred: raise ValueError("transition falsely claimed")
            if occurred:
                name=row["case_identity"]; ops=row["actual_operations"]
                if name in ("birth_hit_1","birth_hit_2","stabilization","healing"):
                    if len(ops)!=row["partition_count"] or any(op["api"]!="advance_site" for op in ops):
                        raise ValueError("missing actual site intervals")
                    expected_events=[r["event"] for r in after.void_state.event_history[len(before.void_state.event_history):]]
                    if [event for op in ops for event in op["events"]]!=expected_events:
                        raise ValueError("site operation trace is not the actual event history")
                required={"promotion":("remesh","field_projection","support_rebuild","equilibrium"),
                    "ligament":("accepted_snapshot","graph_edit","remesh","field_projection","support_rebuild",
                                "connected_surface_certification","equilibrium","energy_gate","topology_verification")}.get(name)
                if required:
                    trace=ops[-1]["operations"]
                    if any(trace.count(token)!=1 for token in required) or [trace.index(t) for t in required] != sorted(trace.index(t) for t in required):
                        raise ValueError("missing/reordered/duplicated actual topology operations")
        if row["dataset"] == "restarts":
            restarted=sources[row["restored_terminal_checkpoint"]]
            if row["restart_exact"] != (fingerprint(after)==fingerprint(restarted)):
                raise ValueError("restart peer mismatch")
        if row["dataset"] == "natural":
            restarted=sources[row["restarted_terminal_checkpoint"]]
            if row["midpoint_restart_exact"] != (fingerprint(after)==fingerprint(restarted)):
                raise ValueError("natural restart mismatch")
        if row["dataset"] == "neutrality":
            base=sources[row["base_terminal_checkpoint"]]
            if row["exact_neutrality"] != (fingerprint(after)==fingerprint(base)):
                raise ValueError("V12 neutrality source comparison mismatch")
        if row["dataset"] == "rollback":
            if row["restored_exactly"] != (fingerprint(after)==fingerprint(before)):
                raise ValueError("rollback source comparison mismatch")
    if payload["decision"] != lifecycle_decision(rows,sources):
        raise ValueError("lifecycle decision differs from actual checkpoint comparisons")
    return {"valid":True,"actual_state_rows":len(rows),"transition_attempts":len(transitions),
            "successful_transition_attempts":sum(r["transition_occurred"] for r in transitions)}


def lifecycle_decision(rows,sources):
    """Recompute success separately from valid evidence of a failed attempt."""
    transitions=[r for r in rows if r["dataset"]=="transitions"]
    partition=[]
    for row in transitions:
        reference=next(r for r in transitions if r["case_identity"]==row["case_identity"] and r["partition_count"]==1)
        elapsed=lambda r:sum(float(op.get("duration_s",0.)) for op in r["actual_operations"])
        duration,reference_duration=elapsed(row),elapsed(reference)
        time_error=abs(duration-reference_duration)/max(abs(reference_duration),1e-300)
        if row["case_identity"] in ("ligament","downstream_child","child_continuation"):
            times=[]
            for r in (row,reference):
                before=sources[r["initial_checkpoint"]]; after=sources[r["terminal_checkpoint"]]
                times.append(after.junction_process_state.get("production_time_s",0.)-before.junction_process_state.get("production_time_s",0.))
            time_error=abs(times[0]-times[1])/max(abs(times[1]),1e-300)
        partition.append({"execution_id":row["execution_id"],"reference_execution_id":reference["execution_id"],
            "actual_transition":row["transition_occurred"],
            "event_time_relative_error":time_error,"frozen_event_time_limit":1e-12,
            "complete_terminal_exact":row["terminal_fingerprint"]==reference["terminal_fingerprint"],
            "passed":bool(row["transition_occurred"] and time_error<=1e-12 and row["terminal_fingerprint"]==reference["terminal_fingerprint"] and row["conservation"]["passed"])})
    restarts=[r for r in rows if r["dataset"]=="restarts"]
    natural=[]
    for row in [r for r in rows if r["dataset"]=="natural"]:
        ref=next(r for r in rows if r["dataset"]=="natural" and r["case_identity"]==row["case_identity"] and r["partition_count"]==1)
        natural.append({"execution_id":row["execution_id"],"partition_exact":row["terminal_fingerprint"]==ref["terminal_fingerprint"],
            "midpoint_restart_exact":row["midpoint_restart_exact"],
            "passed":row["terminal_fingerprint"]==ref["terminal_fingerprint"] and row["midpoint_restart_exact"] and row["conservation"]["passed"]})
    controlled=[]
    for row in [r for r in rows if r["dataset"]=="controlled"]:
        state=sources[row["terminal_checkpoint"]]; case=row["case_identity"]
        phase=state.void_state.sites[0].phase if not state.void_state.cavities else state.void_state.cavities[0].phase
        success=False
        if case=="embryo_healing": success=phase==VoidPhase.HEALED_SITE
        elif case in ("diffusion_limited","accommodation_limited"):
            channel="vacancy_transport_s" if case=="diffusion_limited" else "plastic_accommodation_s"
            histories=[op for op in row["actual_operations"] if op.get("api")=="accepted_load_growth_interval"]
            success=bool(histories) and all(op["rates"][channel]==min(op["rates"][k] for k in
                ("surface_reaction_s","vacancy_transport_s","plastic_accommodation_s")) for op in histories)
        elif case=="local_remesh_refinement":
            success=any(op.get("api")=="actual_local_remesh" for op in row["actual_operations"])
        elif case in ("positive_offset","negative_offset","short_ligament","long_ligament","downstream_zero_drive"):
            success=phase==VoidPhase.CONNECTED_VOID and state.junction_process_state.get("latest_crack_void_connection_certificate",{}).get("passed",False)
            if case=="downstream_zero_drive":
                tensor=cavity_boundary_tensor(state)[0]
                success &= all(r["effective_rate_s"]==0. for r in directional_clock_rates(state,tensor))
        else:
            success=phase==VoidPhase.DOWNSTREAM_FRONT_ACTIVE and bool(state.crack_network.active_tip_ids)
        controlled.append({"case_identity":case,"actual_phase":phase.value,
            "passed":bool(success and row["failure"] is None and row["conservation"]["passed"])})
    rollback=[{"case_identity":r["case_identity"],"passed":r["restored_exactly"] and bool(r["failure"])
        and r["failure"]["message"].startswith("injected:")} for r in rows if r["dataset"]=="rollback"]
    neutrality=[{"case_identity":r["case_identity"],"passed":r["exact_neutrality"] and r["failure"] is None}
        for r in rows if r["dataset"]=="neutrality"]
    continued_restarts=all(bool(r["continued_front_terminal_reached"]) for r in restarts)
    same_terminal=bool(restarts) and len({r["terminal_fingerprint"] for r in restarts})==1
    downstream_rollback=any(r["dataset"]=="rollback" and r["case_identity"].startswith("downstream:") for r in rows)
    complete=(all(r["passed"] for r in partition+natural+controlled+rollback+neutrality)
              and continued_restarts and same_terminal and downstream_rollback)
    return {"transition_partitions":partition,"natural_partitions_restart":natural,
        "controlled_histories":controlled,"rollback_attempts":rollback,"V12_disabled_neutrality":neutrality,
        "restart_to_attainable_terminal_exact":all(r["restart_exact"] for r in restarts),
        "all_restart_stages_reach_identical_complete_terminal":same_terminal,
        "required_continued_front_restart_terminal":continued_restarts,
        "downstream_lifecycle_rollback": "NOT_EXERCISED_SOURCE_RESOLUTION_PREREQUISITE",
        "decision":"PASS" if complete else "BLOCKED"}
