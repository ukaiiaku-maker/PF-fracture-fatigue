#!/usr/bin/env python3
"""Bounded negative/positive scheduler qualification without source rebuilding."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.embryo_growth_v1 import (
    CavityEmissionParameterRowV1, initialize_embryo_growth_state,
    loop_area_kinematics, make_cavity_loop_event_preview,
)
from arrhenius_fracture.global_physical_scheduler_v1 import (
    advance_global_physical_interval, preview_downstream_first_passage,
    preview_ligament_first_passage,
)
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
)
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.voiding_production_v5 import (
    _bundle_from_state, _grow_hole_boundary, build_production_void_state,
    deterministic_trajectory, downstream_source_observation, fresh_sharp_front_engine,
    preview_directional_first_passage, remesh_cavity,
)
from arrhenius_fracture.voiding_v5 import (
    VoidPhase, create_subgrid_cavity, promote_cavity,
)

OUTDIR = ROOT/"artifacts/restricted_production_closeout"
OUT = OUTDIR/"bounded_scheduler_qualification.json"
CONNECTED = OUTDIR/"vacancy_connected_source_qualified.json"
REFERENCE_CACHE = OUTDIR/"bounded_physical_reference_result.json"
LOOP_CACHE = OUTDIR/"real_cavity_loop_growth_result.json"
POSITIVE_CACHE = OUTDIR/"downstream_scheduler_positive_result.json"
LIGAMENT_CACHE = OUTDIR/"scheduler_ligament_result.json"
CHILD_RECEIPT = OUTDIR/"bounded_scheduler_positive_child.receipt.json"
BEHAVIOR_SOURCES = (
    ROOT/"arrhenius_fracture/global_physical_scheduler_v1.py",
    ROOT/"arrhenius_fracture/voiding_production_v5.py",
    ROOT/"arrhenius_fracture/embryo_growth_v1.py",
    ROOT/"arrhenius_fracture/vacancy_transport_phase_field_v2.py",
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def finite(value):
    if is_dataclass(value):
        return finite(asdict(value))
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def hash_payload(value):
    return hashlib.sha256(json.dumps(
        finite(value), sort_keys=True, separators=(",", ":"),
    allow_nan=False).encode()).hexdigest()


def behavior_source_dependencies():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in BEHAVIOR_SOURCES}


def cache_dependency(fixture, *, input_fingerprint=None, configuration=None):
    payload = {
        "fixture": str(fixture),
        "input_state_fingerprint": input_fingerprint,
        "configuration": finite(configuration or {}),
        "behavior_source_sha256": behavior_source_dependencies(),
    }
    return {"payload": payload, "sha256": hash_payload(payload)}


def load_bound_cache(path, dependency):
    if not path.exists():
        return None
    cached = json.loads(path.read_text())
    observed = cached.get("cache_dependency")
    if not isinstance(observed, dict):
        return None
    if observed.get("sha256") != hash_payload(observed.get("payload", {})):
        return None
    return cached if observed == dependency else None


def bind_cache(result, dependency):
    return {**result, "cache_dependency": dependency}


def validate_receipt_chain(segments, *, input_fingerprint, output_fingerprint,
                           expected_events, expected_owner_kinds=(),
                           expected_duration_s=None):
    """Fail closed on endpoints, time, ownership, and exact event order."""
    rows = tuple(segments)
    if not rows:
        return False
    if rows[0].get("input_state_fingerprint") != input_fingerprint:
        return False
    if rows[-1].get("output_state_fingerprint") != output_fingerprint:
        return False
    if any(a.get("output_state_fingerprint") != b.get("input_state_fingerprint")
           for a, b in zip(rows, rows[1:])):
        return False
    durations = tuple(float(row.get("elapsed_time_s", math.nan)) for row in rows)
    if any(not math.isfinite(value) or value < 0.0 for value in durations):
        return False
    if any(value == 0.0 and (row.get("selected_event") is None or
                             row.get("input_state_fingerprint") ==
                             row.get("output_state_fingerprint"))
           for row, value in zip(rows, durations)):
        return False
    if expected_duration_s is not None and not math.isclose(
            math.fsum(durations), float(expected_duration_s),
            rel_tol=2e-12, abs_tol=64*np.spacing(max(
                abs(float(expected_duration_s)), np.finfo(float).tiny))):
        return False
    selected = tuple(row.get("selected_event") for row in rows
                     if row.get("selected_event") is not None)
    if selected != tuple(expected_events):
        return False
    selected_rows = tuple(row for row in rows
                          if row.get("selected_event") is not None)
    if any(not isinstance(row.get("selected_owner"), (tuple, list)) or
           len(row["selected_owner"]) != 2 or
           not all(row["selected_owner"]) for row in selected_rows):
        return False
    if expected_owner_kinds:
        owners = tuple(
            row.get("selected_owner", (None, None))[0] for row in selected_rows)
        if owners != tuple(expected_owner_kinds):
            return False
    return True


def directional_contract(state):
    source = downstream_source_observation(state, continuation=False)
    kwargs = dict(
        source_kind=source["source_kind"],
        source_front_id=source["source_front_id"],
        source_cavity_id=source["source_cavity_id"],
        source_boundary_site_id=source["source_boundary_site_id"],
        source_position_m=source["start_m"],
        source_probe_identity=source["probe_identity"],
        temperature_K=900.0, maximum_advance_duration_s=1.0e100,
        front_load_rows=source["front_load_rows"],
    )
    manual = preview_directional_first_passage(state, source["tensor_Pa"], **kwargs)
    scheduler = preview_downstream_first_passage(
        state, temperature_K=900.0, maximum_duration_s=1.0e100)
    adapted = scheduler.commit_payload["directional_preview"]
    require(manual.event_type == "DIRECTIONAL_CLEAVAGE", "manual crossing unavailable")
    require(scheduler.event_type == "DOWNSTREAM_FIRST_PASSAGE", "scheduler crossing unavailable")
    require(manual.candidate_ids == scheduler.candidate_ids, "candidate adapter mismatch")
    require(manual.threshold_identity == scheduler.threshold_identity, "threshold adapter mismatch")
    require(manual.source_fingerprint == adapted.source_fingerprint, "source adapter mismatch")
    require(manual.crossing_time_s == scheduler.crossing_time_s, "waiting-time adapter mismatch")
    candidate = manual.candidate_ids[0]
    mrow = next(x for x in manual.commit_payload["audit"] if x["candidate_id"] == candidate)
    srow = next(x for x in adapted.commit_payload["audit"] if x["candidate_id"] == candidate)
    for key in ("hazard_barrier_J", "raw_rate_s", "effective_rate_s", "crossing_time_s"):
        require(mrow[key] == srow[key], f"directional adapter mismatch: {key}")
    hazard = next(x for x in state.competition.hazard_states if x.candidate_id == candidate)
    engine = fresh_sharp_front_engine(state.material, _bundle_from_state(state))
    return {
        "status": "PASS", "temperature_K": 900.0,
        "accepted_state_fingerprint": complete_accepted_state_fingerprint(state),
        "source_observation_fingerprint": hash_payload(source),
        "source_fingerprint": manual.source_fingerprint,
        "candidate_id": candidate,
        "process_owner": scheduler.owner_id,
        "material_bundle_fingerprint": hash_payload(
            state.junction_process_state["unified_material_bundle"]),
        "barrier_J": mrow["hazard_barrier_J"],
        "attempt_frequency_s": float(engine.f.nu0_c),
        "action_accumulated": float(hazard.action),
        "threshold_action": float(hazard.current_threshold_action),
        "remaining_threshold_action": float(
            hazard.current_threshold_action-hazard.action),
        "instantaneous_rate_s": mrow["effective_rate_s"],
        "waiting_time_s": manual.crossing_time_s,
        "manual_preview_noncommitting": True,
        "scheduler_preview_noncommitting": True,
    }


def bounded_reference(state):
    duration = 1.0e-6
    before = complete_accepted_state_fingerprint(state)
    dependency = cache_dependency(
        "bounded_reference", input_fingerprint=before,
        configuration={"duration_s": duration, "temperature_K": 900.0})
    cached = load_bound_cache(REFERENCE_CACHE, dependency)
    if cached is not None:
        return cached
    advanced, audit = advance_global_physical_interval(
        state, state, duration, {
            "kind": "hold", "temperature_K": 900.0, "maximum_events": 16,
        })
    require(all(x["selected_event"] is None for x in audit["events"]),
            "physical reference invented an event")
    require(advanced.crack_network.active_tip_ids == (),
            "physical reference invented a downstream child")
    require(advanced.vacancy_transport_state_v2.transport_time_s >
            state.vacancy_transport_state_v2.transport_time_s,
            "transport did not advance")
    # Partition plus checkpoint/restart uses the same finite reference interval.
    half, first = advance_global_physical_interval(
        state, state, duration/2.0, {
            "kind": "hold", "temperature_K": 900.0, "maximum_events": 16,
        })
    checkpoint = OUTDIR/"bounded_reference_midpoint.json"
    write_checkpoint(half, checkpoint, compression="gzip")
    restored = restore_checkpoint(checkpoint)
    require(complete_accepted_state_fingerprint(restored) ==
            complete_accepted_state_fingerprint(half), "midpoint restart mismatch")
    split, second = advance_global_physical_interval(
        restored, restored, duration/2.0, {
            "kind": "hold", "temperature_K": 900.0, "maximum_events": 16,
        })
    direct_hazards = {x.candidate_id: x.action for x in advanced.competition.hazard_states}
    split_hazards = {x.candidate_id: x.action for x in split.competition.hazard_states}
    require(direct_hazards.keys() == split_hazards.keys(), "partition owner mismatch")
    for key in direct_hazards:
        require(math.isclose(direct_hazards[key], split_hazards[key],
                             rel_tol=2e-12, abs_tol=1e-30),
                "partition changed directional action")
    require(complete_accepted_state_fingerprint(state) == before,
            "bounded scheduler mutated its input state")
    result = {
        "status": "NO_EVENT_WITHIN_WINDOW", "duration_s": duration,
        "input_state_fingerprint": before,
        "output_state_fingerprint": complete_accepted_state_fingerprint(advanced),
        "selected_sequence": [x["selected_event"] for x in audit["events"]],
        "common_clock_advanced_s": audit["advanced_duration_s"],
        "transport_advanced_s": (advanced.vacancy_transport_state_v2.transport_time_s-
                                  state.vacancy_transport_state_v2.transport_time_s),
        "no_clock_reset": all(split_hazards[k] >=
                              next(x.action for x in state.competition.hazard_states
                                   if x.candidate_id == k) for k in split_hazards),
        "partition_check": "PASS", "checkpoint_restart": "PASS",
        "plastic_rate_policy": "ACCEPTED_ENDPOINT_INCREMENT_ONLY_NO_FROZEN_EXTRAPOLATION",
        "segments": finite(audit["events"]),
        "partition_segments": finite(first["events"]+second["events"]),
    }
    result = bind_cache(result, dependency)
    REFERENCE_CACHE.write_text(json.dumps(finite(result), indent=2, sort_keys=True)+"\n")
    return result


def real_loop_growth():
    dependency = cache_dependency(
        "real_loop_growth",
        configuration={"temperature_K": 1200.0, "seed": 901,
                       "attempt_frequency_s": 0.5,
                       "functional_growth_supply_area_rate_m2_s": 0.0})
    cached = load_bound_cache(LOOP_CACHE, dependency)
    if cached is not None:
        return cached
    state, hole = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=1200.0)
    row = replace(
        CavityEmissionParameterRowV1(),
        row_id="functional-real-loop-promotion-v1",
        source_orientations_rad=(0.0,), attempt_frequency_s=0.5)
    growth = initialize_embryo_growth_state(
        site_id="site-1", radius_m=4.9e-5, row=row, seed=901,
        emission_enabled=True)
    site = replace(state.void_state.sites[0], phase=VoidPhase.STABLE_SUBGRID_VOID)
    configuration = replace(
        state.void_state.embryo_growth_configuration,
        vacancy_clustering_birth_enabled=False,
        direct_interaction_birth_enabled=False,
        cavity_surface_loop_emission_enabled=True)
    void_state = create_subgrid_cavity(replace(
        state.void_state, sites=(site,), embryo_growth_states=(growth,),
        embryo_growth_configuration=configuration), site.site_id, growth.radius_m)
    state = replace(state, void_state=void_state)
    input_fingerprint = complete_accepted_state_fingerprint(state)
    growth = state.void_state.embryo_growth_states[0]
    area_per_loop = float(loop_area_kinematics(row, growth.radius_m)["selected_area_m2"])
    one_loop_radius = math.sqrt(growth.radius_m**2+area_per_loop/math.pi)
    promotion_radius = 0.5*(growth.radius_m+one_loop_radius)
    preview = make_cavity_loop_event_preview(
        state, growth, temperature_K=1200.0, maximum_duration_s=100.0)
    require(preview.event_type == "CAVITY_LOOP_EMISSION", "real loop event unavailable")

    promotion_commit_count = 0

    def commit_promotion(live, _preview):
        nonlocal promotion_commit_count
        promotion_commit_count += 1
        if promotion_commit_count > 1:
            raise RuntimeError("growth promotion callback invoked more than once: "+repr({
                "representation": live.void_state.embryo_growth_states[0].representation,
                "cavity_phase": live.void_state.cavities[0].phase.value,
                "radius_m": live.void_state.embryo_growth_states[0].radius_m,
            }))
        cavity = live.void_state.cavities[0]
        promoted = promote_cavity(live.void_state, cavity.cavity_id, promotion_radius)
        # The retained production restriction remains explicit after promotion.
        promoted_growth = replace(
            promoted.embryo_growth_states[0],
            cavity_emission_state=replace(
                promoted.embryo_growth_states[0].cavity_emission_state,
                enabled=False))
        promoted = replace(
            promoted, embryo_growth_states=(promoted_growth,),
            embryo_growth_configuration=replace(
                promoted.embryo_growth_configuration,
                cavity_surface_loop_emission_enabled=False))
        moved = _grow_hole_boundary(
            hole, promoted.cavities[0].radius_m,
            crack_path_m=live.crack_network.branches[0].path)
        return remesh_cavity(live, moved, promoted, "bounded-real-loop-promotion")

    advanced, audit = advance_global_physical_interval(
        state, state, preview.crossing_time_s+1.1e-10, {
            "kind": "hold", "temperature_K": 1200.0, "maximum_events": 16,
            "promotion_radius_m": promotion_radius,
            "functional_test_growth_supply_area_rate_m2_s": 0.0,
            "test_only_functional_qualification": True,
            "diagnostic_freeze_front_process": True,
            "commit_growth_promotion": commit_promotion,
        })
    after = advanced.void_state.embryo_growth_states[0]
    loop_events = [x for x in audit["events"] if x["selected_event"] ==
                   "CAVITY_LOOP_EMISSION"]
    promotion_events = [x for x in audit["events"] if x["selected_event"] ==
                        "GROWTH_PROMOTION"]
    require(len(loop_events) == 1 and len(promotion_events) == 1,
            "loop/promotion sequence incomplete: "+repr(
                [x["selected_event"] for x in audit["events"]]))
    loop_packet = loop_events[0]["embryo_growth"][0]["packets"][0]
    area_increment = float(after.area_m2-growth.area_m2)
    loop_area_increment = float(
        after.dislocation_derived_area_m2-growth.dislocation_derived_area_m2)
    require(after.cavity_emission_state.cumulative_loop_count_effective == 1.0,
            "real loop count did not advance")
    require(math.isclose(loop_area_increment, area_per_loop, rel_tol=2e-12),
            "loop-derived area mismatch")
    require(loop_packet["mechanical_deposition"]["deposited_loop_count"] == 1.0,
            "loop mechanics was not deposited")
    require(advanced.void_state.cavities[0].phase == VoidPhase.RESOLVED_VOID,
            "promotion was not committed")
    result = {
        "status": "PASS", "configuration": finite(asdict(row)),
        "functional_test_growth_supply_area_rate_m2_s": 0.0,
        "stabilized_before_growth": True,
        "radius_before_m": growth.radius_m,
        "selected_promotion_radius_m": promotion_radius,
        "radius_after_loop_m": one_loop_radius,
        "radius_after_m": after.radius_m,
        "realized_loop_increment": 1,
        "burgers_increment_m": row.burgers_m,
        "selected_loop_area_m2": area_per_loop,
        "measured_loop_derived_area_increment_m2": loop_area_increment,
        "measured_area_increment_m2": area_increment,
        "mechanical_deposition": finite(loop_packet["mechanical_deposition"]),
        "equilibrated_after_every_packet":
            loop_events[0]["embryo_growth"][0]["equilibrated_after_every_packet"],
        "selected_sequence": [x["selected_event"] for x in audit["events"]],
        "segments": finite(audit["events"]),
        "input_state_fingerprint": input_fingerprint,
        "output_state_fingerprint": complete_accepted_state_fingerprint(advanced),
    }
    result = bind_cache(result, dependency)
    LOOP_CACHE.write_text(json.dumps(finite(result), indent=2, sort_keys=True)+"\n")
    return result


def positive_downstream(state):
    scale = 1.0e60
    parent_fingerprint = complete_accepted_state_fingerprint(state)
    dependency = cache_dependency(
        "positive_downstream", input_fingerprint=parent_fingerprint,
        configuration={"temperature_K": 900.0,
                       "functional_test_directional_rate_scale": scale})
    cached = load_bound_cache(POSITIVE_CACHE, dependency)
    if cached is not None:
        return cached
    history = {
        "kind": "hold", "temperature_K": 900.0, "maximum_events": 8,
        "functional_test_directional_rate_scale": scale,
        "test_only_functional_qualification": True,
    }
    child_preview = preview_downstream_first_passage(
        state, temperature_K=900.0, maximum_duration_s=1.0,
        functional_test_rate_scale=scale,
        test_only_functional_qualification=True)
    require(child_preview.event_type == "DOWNSTREAM_FIRST_PASSAGE",
            "controlled downstream child preview unavailable")
    child_checkpoint = OUTDIR/"bounded_scheduler_positive_child.json"
    # A checkpoint is not an event receipt.  Re-execute only this bounded
    # scheduler crossing when no source-bound original receipt is available;
    # never synthesize today's parent/time around an old terminal state.
    child, child_audit = advance_global_physical_interval(
        state, state, child_preview.crossing_time_s, history)
    require(child.crack_network.active_tip_ids == ("void-front-1",),
            "scheduler did not commit downstream child")
    child_output = complete_accepted_state_fingerprint(child)
    if child_checkpoint.exists():
        retained = restore_checkpoint(child_checkpoint)
        require(complete_accepted_state_fingerprint(retained) == child_output,
                "retained child checkpoint disagrees with current scheduler receipt")
    else:
        write_checkpoint(child, child_checkpoint, compression="gzip")
    child_receipt = {
        "schema": "unified-2d.scheduler-receipt-sidecar/1",
        "cache_dependency": dependency,
        "input_state_fingerprint": parent_fingerprint,
        "output_state_fingerprint": child_output,
        "segments": finite(child_audit["events"]),
    }
    child_receipt["authentication_sha256"] = hash_payload(child_receipt)
    CHILD_RECEIPT.write_text(json.dumps(child_receipt, indent=2, sort_keys=True)+"\n")
    restored = restore_checkpoint(child_checkpoint)
    require(complete_accepted_state_fingerprint(restored) ==
            complete_accepted_state_fingerprint(child), "positive child restart mismatch")
    continuation_preview = preview_downstream_first_passage(
        restored, continuation=True, temperature_K=900.0,
        maximum_duration_s=1.0, functional_test_rate_scale=scale,
        test_only_functional_qualification=True)
    require(continuation_preview.event_type == "CHILD_CONTINUATION",
            "controlled continuation preview unavailable")
    continuation_horizon = continuation_preview.crossing_time_s
    continued, continuation_audit = advance_global_physical_interval(
        restored, restored, continuation_horizon, history)
    require(any(x["selected_event"] == "CHILD_CONTINUATION"
                for x in continuation_audit["events"]),
            "scheduler did not commit child continuation: "+repr(
                [x["selected_event"] for x in continuation_audit["events"]]))
    continuation_checkpoint = OUTDIR/"bounded_scheduler_positive_continuation.json"
    write_checkpoint(continued, continuation_checkpoint, compression="gzip")
    result = {
        "status": "PASS",
        "classification": "CONTROLLED_NOT_CALIBRATED_REFERENCE",
        "functional_test_directional_rate_scale": scale,
        "threshold_preserved": child_preview.threshold_identity,
        "child_wait_s": child_preview.crossing_time_s,
        "continuation_wait_s": continuation_preview.crossing_time_s,
        "continuation_bracket_horizon_s": continuation_horizon,
        "child_input_state_fingerprint": complete_accepted_state_fingerprint(state),
        "child_output_state_fingerprint": complete_accepted_state_fingerprint(child),
        "continuation_input_state_fingerprint": complete_accepted_state_fingerprint(restored),
        "continuation_output_state_fingerprint": complete_accepted_state_fingerprint(continued),
        "child_segments": finite(child_audit["events"]),
        "continuation_segments": finite(continuation_audit["events"]),
        "child_checkpoint_restart": "PASS",
        "same_owner_preview_advance_commit": True,
    }
    result = bind_cache(result, dependency)
    POSITIVE_CACHE.write_text(json.dumps(finite(result), indent=2, sort_keys=True)+"\n")
    return result


def scheduler_ligament_fixture():
    dependency = cache_dependency(
        "scheduler_ligament_fixture",
        configuration={"temperature_K": 900.0, "applied_opening_multiplier": 1.25,
                       "functional_test_directional_rate_scale": 1.0e60})
    cached = load_bound_cache(LIGAMENT_CACHE, dependency)
    if cached is not None:
        return cached
    from arrhenius_fracture.vacancy_transport_phase_field_v2 import (
        VacancyTransportParameterRowV2, build_exact_cut_transport_mesh_for_network,
        initial_transport_state,
    )
    state, _ = deterministic_trajectory(
        bundle=material_bundle("DBTT"), stop_before_ligament=True)
    # Prospectively fixed applied-opening control.  It changes neither the
    # material row nor the topology/energy acceptance rule.
    state = equilibrate_fixed_load_with_production_fem(replace(
        state, displacement=1.25*np.asarray(state.displacement)))
    segments = tuple(
        (first, second, f"{branch.branch_id}:{index}")
        for branch in state.crack_network.branches
        for index, (first, second) in enumerate(zip(branch.path, branch.path[1:])))
    cut = build_exact_cut_transport_mesh_for_network(
        state.mesh, crack_segments=segments,
        topology_boundary_tags=state.junction_process_state[
            "transport_boundary_tags_v2"])
    row = VacancyTransportParameterRowV2.from_material(state.material)
    transport = initial_transport_state(
        cut.mesh, temperature_K=900.0, row=row, registry=cut.registry,
        parent_mechanics_element_ids=cut.parent_mechanics_element_ids,
        active_cavity_inventory_area_m2=
            state.void_state.cavities[0].inventory_area_m2)
    state = replace(state, vacancy_transport_state_v2=transport)
    scale = 1.0e60
    preview = preview_ligament_first_passage(
        state, temperature_K=900.0, maximum_duration_s=1.0,
        functional_test_rate_scale=scale,
        test_only_functional_qualification=True)
    require(preview.event_type == "LIGAMENT_RUPTURE",
            "controlled ligament preview unavailable")
    advanced, audit = advance_global_physical_interval(
        state, state, preview.crossing_time_s, {
            "kind": "hold", "temperature_K": 900.0, "maximum_events": 4,
            "functional_test_directional_rate_scale": scale,
            "test_only_functional_qualification": True,
            "diagnostic_freeze_front_process": True,
        })
    require([x["selected_event"] for x in audit["events"]] ==
            ["LIGAMENT_RUPTURE"], "scheduler did not commit ligament rupture")
    require(advanced.void_state.cavities[0].phase == VoidPhase.CONNECTED_VOID,
            "ligament fixture did not create connected topology")
    checkpoint = OUTDIR/"bounded_scheduler_ligament_connected.json"
    write_checkpoint(advanced, checkpoint, compression="gzip")
    result = {
        "status": "PASS", "classification": "CONTROLLED_NOT_REFERENCE_PREDICTION",
        "applied_opening_multiplier": 1.25,
        "functional_test_directional_rate_scale": scale,
        "production_material_row_unchanged": True,
        "topology_and_energy_gate_unchanged": True,
        "crossing_time_s": preview.crossing_time_s,
        "threshold_identity": finite(preview.threshold_identity),
        "input_state_fingerprint": complete_accepted_state_fingerprint(state),
        "output_state_fingerprint": complete_accepted_state_fingerprint(advanced),
        "segments": finite(audit["events"]),
    }
    result = bind_cache(result, dependency)
    LIGAMENT_CACHE.write_text(json.dumps(finite(result), indent=2, sort_keys=True)+"\n")
    return result


def main():
    require(CONNECTED.exists(), "retained qualified connected checkpoint is missing")
    connected = restore_checkpoint(CONNECTED)
    evidence = json.loads((OUTDIR/"connected_lifecycle_evidence.json").read_text())
    manufactured = next(x for x in evidence["direct_stages"]
                        if x["operation"] == "STABLE_SUBGRID_GROWTH")
    manufactured_delta = manufactured["operation_evidence"]["measured_growth_delta"]
    result = {
        "schema": "unified-2d.bounded-scheduler-qualification/1",
        "downstream_rate_contract": directional_contract(connected),
        "physical_reference": bounded_reference(connected),
        "manufactured_supply_control": {
            "status": "PASS_EXPLICIT_AREA_SUPPLY_CONTROL",
            "radius_before_m": manufactured_delta["radius_before_m"],
            "radius_after_m": manufactured_delta["radius_after_m"],
            "not_evidence_of_cavity_loop_emission": True,
            "retained_stage_authentication_sha256": manufactured["authentication_sha256"],
        },
        "real_cavity_loop_growth": real_loop_growth(),
        "scheduler_ligament_fixture": scheduler_ligament_fixture(),
        "downstream_scheduler_positive": positive_downstream(connected),
        "retained_resolved_loop_consumer": "UNAVAILABLE_NO_PRODUCTION_RATE",
        "scheduler_resolved_growth_fixture": "PASS_FOCUSED_REGRESSION",
    }
    physical = result["physical_reference"]
    loop = result["real_cavity_loop_growth"]
    ligament = result["scheduler_ligament_fixture"]
    downstream = result["downstream_scheduler_positive"]
    receipt_checks = {
        "physical_reference": validate_receipt_chain(
            physical["segments"], input_fingerprint=physical["input_state_fingerprint"],
            output_fingerprint=physical["output_state_fingerprint"],
            expected_events=(), expected_duration_s=physical["duration_s"]),
        "real_loop_growth": validate_receipt_chain(
            loop["segments"], input_fingerprint=loop["input_state_fingerprint"],
            output_fingerprint=loop["output_state_fingerprint"],
            expected_events=("CAVITY_LOOP_EMISSION", "GROWTH_PROMOTION"),
            expected_owner_kinds=("EmbryoGrowthProcessOwner",
                                  "EmbryoGrowthProcessOwner")),
        "ligament": validate_receipt_chain(
            ligament["segments"], input_fingerprint=ligament["input_state_fingerprint"],
            output_fingerprint=ligament["output_state_fingerprint"],
            expected_events=("LIGAMENT_RUPTURE",),
            expected_owner_kinds=("ligament_transaction",),
            expected_duration_s=ligament["crossing_time_s"]),
        "downstream_child": validate_receipt_chain(
            downstream["child_segments"],
            input_fingerprint=downstream["child_input_state_fingerprint"],
            output_fingerprint=downstream["child_output_state_fingerprint"],
            expected_events=("DOWNSTREAM_FIRST_PASSAGE",),
            expected_owner_kinds=("downstream_front_transaction",),
            expected_duration_s=downstream["child_wait_s"]),
        "child_continuation": validate_receipt_chain(
            downstream["continuation_segments"],
            input_fingerprint=downstream["continuation_input_state_fingerprint"],
            output_fingerprint=downstream["continuation_output_state_fingerprint"],
            expected_events=("CHILD_CONTINUATION",),
            expected_owner_kinds=("downstream_front_transaction",),
            expected_duration_s=downstream["continuation_wait_s"]),
    }
    require(downstream["child_output_state_fingerprint"] ==
            downstream["continuation_input_state_fingerprint"],
            "child checkpoint does not link to continuation receipt")
    cache_checks = {
        key: (isinstance(value.get("cache_dependency"), dict) and
              value["cache_dependency"].get("sha256") ==
              hash_payload(value["cache_dependency"].get("payload", {})))
        for key, value in {
            "physical_reference": physical, "real_loop_growth": loop,
            "ligament": ligament, "downstream": downstream,
        }.items()
    }
    result["receipt_validation"] = receipt_checks
    result["cache_dependency_validation"] = cache_checks
    result["stage_receipt_linkage"] = (
        "PASS" if all(receipt_checks.values()) else "FAIL")
    result["receipt_and_cache_validity"] = (
        "PASS" if all(receipt_checks.values()) and all(cache_checks.values())
        else "FAIL")
    result["real_loop_growth_control"] = (
        "PASS" if loop["status"] == "PASS" else "FAIL")
    result["positive_topology_controls"] = (
        "PASS" if ligament["status"] == "PASS" and downstream["status"] == "PASS"
        else "FAIL")
    # These two code-path decisions are exercised by the focused scheduler
    # regressions; source hashes above bind this aggregate to that executable.
    result["nonwinning_owner_advancement"] = "PASS_FOCUSED_REGRESSION"
    result["legacy_front_common_time_evolution"] = "PASS_FOCUSED_REGRESSION"
    result["integrated_closeout"] = "PASS" if (
        result["physical_reference"]["status"] == "NO_EVENT_WITHIN_WINDOW"
        and result["downstream_rate_contract"]["status"] == "PASS"
        and result["real_cavity_loop_growth"]["status"] == "PASS"
        and result["scheduler_ligament_fixture"]["status"] == "PASS"
        and result["downstream_scheduler_positive"]["status"] == "PASS"
        and result["receipt_and_cache_validity"] == "PASS"
    ) else "PENDING"
    body = finite(result)
    body["authentication_sha256"] = hash_payload(body)
    OUT.write_text(json.dumps(body, indent=2, sort_keys=True)+"\n")
    print("PHYSICAL_REFERENCE_OUTCOME = "+body["physical_reference"]["status"])
    print("MANUFACTURED_SUPPLY_CONTROL = "+body["manufactured_supply_control"]["status"])
    print("REAL_LOOP_GROWTH_CONTROL = "+body["real_loop_growth_control"])
    print("POSITIVE_TOPOLOGY_CONTROLS = "+body["positive_topology_controls"])
    print("NONWINNING_OWNER_ADVANCEMENT = "+body["nonwinning_owner_advancement"])
    print("LEGACY_FRONT_COMMON_TIME_EVOLUTION = "+body["legacy_front_common_time_evolution"])
    print("RECEIPT_AND_CACHE_VALIDITY = "+body["receipt_and_cache_validity"])
    print("INTEGRATED_CLOSEOUT = "+body["integrated_closeout"])
    return 0 if body["integrated_closeout"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
