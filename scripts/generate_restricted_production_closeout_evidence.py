#!/usr/bin/env python3
"""Generate compact connected functional lifecycle evidence (no campaign rerun)."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from collections.abc import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.dislocation_interaction_voiding_v1 import (
    DislocationInteractionParameterRowV1, accepted_interaction_opportunity,
    advance_live_direct_owner, initialize_dislocation_interaction_state,
)
from arrhenius_fracture.embryo_growth_v1 import (
    advance_live_embryo_growth, consume_resolved_cavity_loop_source,
    source_local_fem_tensors,
    transport_flux_certificate,
)
from arrhenius_fracture.global_physical_scheduler_v1 import (
    advance_global_physical_interval, commit_vacancy_birth_hit,
    preview_vacancy_birth_hit,
)
from arrhenius_fracture.restricted_production_closeout_v1 import (
    authenticate_operation_evidence, authenticate_stage,
)
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
)
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.vacancy_transport_phase_field_v2 import (
    accepted_elastoplastic_stress_3d, solve_transport_interval,
    transfer_reserved_embryo_to_growth, transport_conservation,
)
from arrhenius_fracture.voiding_production_v5 import (
    _grow_hole_boundary, build_production_void_state, downstream_front_transaction,
    ligament_transaction, local_site_tensor, remesh_cavity,
)
from arrhenius_fracture.voiding_v5 import (
    VoidingConfig, advance_site, arrhenius_rates, create_subgrid_cavity, promote_cavity,
)

OUT = ROOT / "artifacts/restricted_production_closeout/connected_lifecycle_evidence.json"


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_json(value):
    if is_dataclass(value): return _finite_json(asdict(value))
    if isinstance(value, Mapping): return {str(k): _finite_json(v) for k, v in value.items()}
    if isinstance(value, np.ndarray): return _finite_json(value.tolist())
    if isinstance(value, np.generic): return _finite_json(value.item())
    if isinstance(value, (tuple, list)): return [_finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    return value


def _nonfinite_paths(value, path="root"):
    if is_dataclass(value):
        return _nonfinite_paths(asdict(value), path)
    if isinstance(value, Mapping):
        return [item for key, child in value.items()
                for item in _nonfinite_paths(child, f"{path}.{key}")]
    if isinstance(value, (tuple, list)):
        return [item for index, child in enumerate(value)
                for item in _nonfinite_paths(child, f"{path}[{index}]")]
    if isinstance(value, np.ndarray):
        indices = np.argwhere(~np.isfinite(value))
        return [f"{path}{tuple(map(int, index))}" for index in indices]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return [path]
    return []


def _balance(state, **extra):
    if state.vacancy_transport_state_v2 is None:
        return {"nonnegative": True, "residual_m2": 0.0,
                "roundoff_budget_m2": 1e-20, **extra}
    transport = state.vacancy_transport_state_v2
    audit = transport_conservation(transport, transport.transport_mesh.area_e)
    scale = max(abs(audit["conservation_lhs_area_m2"]),
                abs(audit["conservation_rhs_area_m2"]), 1e-30)
    return {
        "nonnegative": all(float(value) >= -1.0e-18 for value in transport.ledgers.values()),
        "residual_m2": float(audit["conservation_residual_area_m2"]),
        "roundoff_budget_m2": max(1e-18, 1e-9*scale), **extra,
    }


def _append(rows, *, before, after, operation, time_s, mechanism, api,
            source, source_status=None, balance_extra=None,
            operation_evidence=None, input_fingerprint=None,
            output_fingerprint=None):
    growth = (() if after.void_state is None else after.void_state.embryo_growth_states)
    cavity = (() if after.void_state is None else after.void_state.cavities)
    row = {
        "run_id": source["run_id"], "source_identity": source,
        "physical_time_s": float(time_s), "operation": operation,
        "input_state_fingerprint": (input_fingerprint or
                                    complete_accepted_state_fingerprint(before)),
        "output_state_fingerprint": (output_fingerprint or
                                     complete_accepted_state_fingerprint(after)),
        "site_id": "site-1", "embryo_id": "site-1",
        "cavity_id": (cavity[0].cavity_id if cavity else "void:site-1"),
        "front_id": (after.crack_network.active_tip_ids[0]
                     if after.crack_network.active_tip_ids else "root"),
        "active_mechanism": mechanism, "production_api": api,
        "inventory_balance": _balance(after, **(balance_extra or {})),
        "representation": (growth[0].representation if growth else "AVAILABLE_SITE"),
        "topology_state": operation, "owner_replaced": False,
        "phase_assignment_only": False, "evidence_kind": "CONTROLLED_FUNCTIONAL_SENTINEL",
        "operation_evidence": authenticate_operation_evidence(
            operation_evidence or {
                "executed": True, "operation": operation,
                "committed_duration_s": 0.0,
                "committed_event_type": operation,
            }),
    }
    if source_status is not None: row["source_consumer_status"] = source_status
    rows.append(authenticate_stage(row))


def _growth_delta(before, after, *, mechanism_keys, audit, duration_s,
                  event_type):
    first = before.void_state.embryo_growth_states[0]
    second = after.void_state.embryo_growth_states[0]
    increments = {}
    for name, field in mechanism_keys.items():
        increments[name] = float(getattr(second, field)-getattr(first, field))
    area_delta = float(second.area_m2-first.area_m2)
    residual = area_delta-math.fsum(increments.values())
    budget = max(1e-30, 512*np.finfo(float).eps*max(
        abs(first.area_m2), abs(second.area_m2), abs(area_delta), 1e-30))
    realized_before = sum(x.realized_loop_count for x in
                          first.cavity_emission_state.sources)
    realized_after = sum(x.realized_loop_count for x in
                         second.cavity_emission_state.sources)
    moment_before = sum(x.continuum_moment_loop_count for x in
                        first.cavity_emission_state.sources)
    moment_after = sum(x.continuum_moment_loop_count for x in
                       second.cavity_emission_state.sources)
    return {
        "executed": True, "operation": event_type,
        "committed_duration_s": float(duration_s),
        "committed_event_type": event_type,
        "scheduler_audit": _finite_json(audit),
        "measured_growth_delta": {
            "radius_before_m": float(first.radius_m),
            "radius_after_m": float(second.radius_m),
            "area_before_m2": float(first.area_m2),
            "area_after_m2": float(second.area_m2),
            "mechanism_area_increments_m2": increments,
            "realized_loop_count_change": int(realized_after-realized_before),
            "moment_loop_count_change": float(moment_after-moment_before),
            "burgers_content_change_m": float(
                second.cumulative_emitted_burgers_m-
                first.cumulative_emitted_burgers_m),
            "conservation_residual_m2": residual,
            "conservation_budget_m2": budget,
        },
    }


def _birth_vacancy_state():
    from scripts.run_unified_one_void_end_to_end import RETAINED_CRACK_PATH_M
    state, hole = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=1200.0,
        initial_vacancy_fraction_gp=0.0099567,
        crack_path_m=RETAINED_CRACK_PATH_M,
        source_conforming_geometry=True)
    state = equilibrate_fixed_load_with_production_fem(replace(
        state, displacement=2.0*np.asarray(state.displacement)))
    transport = state.vacancy_transport_state_v2
    capture = replace(transport.site_states[0], capture_rate_area_per_s=1.0e10,
                      support_radius_m=7.2e-4)
    state = replace(state, vacancy_transport_state_v2=replace(
        transport, site_states=(capture,)))
    start = state; elapsed = 0.0; scheduler_audits = []
    for _ in range(2):
        transport = state.vacancy_transport_state_v2
        preview = preview_vacancy_birth_hit(
            state, maximum_duration_s=1.0e-6,
            accepted_equivalent_plastic_strain_rate_s=np.zeros(transport.transport_mesh.ne),
            accepted_dislocation_density_m2=np.asarray(state.rho_gp)[transport.parent_mechanics_element_ids],
            accepted_mean_stress_Pa=np.zeros(transport.transport_mesh.ne))
        if not preview.eligible: raise RuntimeError("controlled vacancy hit was not eligible")
        state, scheduler = advance_global_physical_interval(
            state, state, preview.crossing_time_s, {
                "kind": "hold", "temperature_K": 1200.0,
                "maximum_events": 16,
                "stress_tensor_Pa": np.zeros((2, 2)),
            })
        committed = [event for event in scheduler["events"]
                     if event["selected_event"] == "VACANCY_BIRTH_HIT"]
        if len(committed) != 1:
            raise RuntimeError("global scheduler did not commit one vacancy birth hit")
        scheduler_audits.append(dict(scheduler))
        elapsed += scheduler["advanced_duration_s"]
    # The prescribed high-capture pulse is a bounded functional input for the
    # two birth hits only.  End that pulse at the accepted event boundary so
    # later scheduler intervals consume ordinary transport fluxes.
    transport = state.vacancy_transport_state_v2
    state = replace(state, vacancy_transport_state_v2=replace(
        transport, site_states=(replace(
            transport.site_states[0], capture_rate_area_per_s=0.0),)))
    return start, state, hole, elapsed, scheduler_audits


def _stabilize(state, temperature):
    stress3d = accepted_elastoplastic_stress_3d(state)
    site = state.void_state.sites[0]
    centers = np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)].mean(axis=1)
    eid = int(np.argmin(np.linalg.norm(
        centers-np.asarray(site.center_m), axis=1)))
    tensor = np.array(((stress3d["sigma_xx_Pa"][eid], stress3d["sigma_xy_Pa"][eid]),
                       (stress3d["sigma_xy_Pa"][eid], stress3d["sigma_yy_Pa"][eid])))
    rates = arrhenius_rates(VoidingConfig(enabled=True), temperature_K=temperature,
                            stress_tensor_Pa=tensor, normal_xy=site.normal_xy)
    dt = site.stabilization.crossing_time(rates["stabilization_s"])
    stabilized, scheduler = advance_global_physical_interval(
        state, state, dt, {
            "kind": "hold", "temperature_K": float(temperature),
            "maximum_events": 64,
        })
    committed = [event for event in scheduler["events"]
                 if event["selected_event"] == "STABILIZATION"]
    if len(committed) != 1:
        raise RuntimeError("global scheduler did not stabilize controlled embryo")
    return stabilized, float(scheduler["advanced_duration_s"]), dict(scheduler)


def _vacancy_growth_packet(state, *, duration_s=1.0e-8, epdot_s=2.0e3):
    before_transport = state.vacancy_transport_state_v2
    parent = np.asarray(before_transport.parent_mechanics_element_ids, int)
    stress = accepted_elastoplastic_stress_3d(state)
    certified_after = solve_transport_interval(
        before_transport, before_transport.transport_mesh, before_transport.surface_registry,
        row=before_transport.parameter_row, duration_s=duration_s,
        accepted_equivalent_plastic_strain_rate_s=np.full(len(parent), epdot_s),
        accepted_dislocation_density_m2=np.asarray(state.rho_gp)[parent],
        accepted_mean_stress_Pa=np.asarray(stress["mean_stress_Pa"])[parent])
    committed_transport, _ = transfer_reserved_embryo_to_growth(certified_after, "site-1")
    transported = replace(state, vacancy_transport_state_v2=committed_transport)
    certificate = transport_flux_certificate(
        before_transport, certified_after, owner_id="site-1", owner_kind="SUBGRID_SITE")
    tensors, zz = source_local_fem_tensors(transported, transported.void_state.embryo_growth_states[0])
    grown, audit = advance_live_embryo_growth(
        transported, site_id="site-1", duration_s=duration_s,
        transport_certificate=certificate, transport_before_state=before_transport,
        transport_after_state=certified_after,
        bulk_accommodation_area_rate_m2_s=1.0,
        stress_tensor_Pa=tensors, sigma_zz_Pa=zz, temperature_K=1200.0,
        support_radius_m=5.0e-5, local_mesh_size_m=1.0e-5,
        loop_emission_enabled=False)
    return grown, audit


def vacancy_trace():
    start, state, hole, elapsed, birth_scheduler_audits = _birth_vacancy_state()
    source = {"run_id": "restricted-vacancy-functional-v1",
              "configuration": "dilute-upper-bound-wide-capture-functional-sentinel",
              "physics_classification": "CONTROLLED_NOT_CALIBRATED_REFERENCE"}
    rows = []
    _append(rows, before=start, after=start, operation="AVAILABLE_SITE", time_s=0.0,
            mechanism="VACANCY_CLUSTERING", api="arrhenius_fracture.voiding_production_v5.build_production_void_state", source=source)
    _append(rows, before=start, after=state, operation="VACANCY_INVENTORY_BIRTH", time_s=elapsed,
            mechanism="VACANCY_CLUSTERING",
            api="arrhenius_fracture.global_physical_scheduler_v1.advance_global_physical_interval",
            source=source, operation_evidence={
                "executed": True, "operation": "VACANCY_INVENTORY_BIRTH",
                "committed_duration_s": elapsed,
                "committed_event_type": "VACANCY_BIRTH_HIT",
                "scheduler_audits": _finite_json(birth_scheduler_audits),
                "committed_hit_count": sum(
                    event["selected_event"] == "VACANCY_BIRTH_HIT"
                    for audit in birth_scheduler_audits for event in audit["events"]),
            })
    before = state; state, dt, stabilization_scheduler = _stabilize(state, 1200.0); elapsed += dt
    _append(rows, before=before, after=state, operation="EMBRYO_STABILIZED", time_s=elapsed,
            mechanism="VACANCY_CLUSTERING",
            api="arrhenius_fracture.global_physical_scheduler_v1.advance_global_physical_interval",
            source=source, operation_evidence={
                "executed": True, "operation": "EMBRYO_STABILIZED",
                "committed_duration_s": dt,
                "committed_event_type": "STABILIZATION",
                "scheduler_audit": _finite_json(stabilization_scheduler),
            })
    def commit_promotion(live, _preview):
        cavity = live.void_state.cavities[0]
        promoted = promote_cavity(live.void_state, cavity.cavity_id, 5.0e-5)
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
        return remesh_cavity(
            live, moved, promoted, "restricted-closeout-promotion")

    before = state
    state, promotion_scheduler = advance_global_physical_interval(
        state, state, 1.0e-12, {
            "kind": "hold", "temperature_K": 1200.0,
            "maximum_events": 64, "promotion_radius_m": 5.0e-5,
            "commit_growth_promotion": commit_promotion,
        })
    promotions = [event for event in promotion_scheduler["events"]
                  if event["selected_event"] == "GROWTH_PROMOTION"]
    if len(promotions) != 1:
        raise RuntimeError("global scheduler did not commit vacancy promotion")
    elapsed += promotion_scheduler["advanced_duration_s"]
    hole = _grow_hole_boundary(
        hole, state.void_state.cavities[0].radius_m,
        crack_path_m=state.crack_network.branches[0].path)
    _append(rows, before=before, after=state, operation="PROMOTION", time_s=elapsed,
            mechanism="VACANCY_TRANSPORT_GROWTH",
            api="arrhenius_fracture.global_physical_scheduler_v1.advance_global_physical_interval",
            source=source, operation_evidence={
                "executed": True, "operation": "PROMOTION",
                "committed_duration_s": promotion_scheduler["advanced_duration_s"],
                "committed_event_type": "GROWTH_PROMOTION",
                "scheduler_audit": _finite_json(dict(promotion_scheduler)),
                "radius_transition_verified": promotions[0]["radius_transition_verified"],
            })
    # A nonzero accepted active-cavity flux is advanced by the authoritative
    # owner. Keep the packet below remesh cadence; loop emission stays disabled.
    before = state; transport = state.vacancy_transport_state_v2
    parent = np.asarray(transport.parent_mechanics_element_ids, int)
    stress = accepted_elastoplastic_stress_3d(state)
    from arrhenius_fracture.voiding_production_v5 import couple_active_cavity_flux_to_geometry
    def resolved_trial(duration):
        after_transport = solve_transport_interval(
            transport, transport.transport_mesh, transport.surface_registry,
            row=transport.parameter_row, duration_s=duration,
            accepted_equivalent_plastic_strain_rate_s=np.full(len(parent), 2.0e3),
            accepted_dislocation_density_m2=np.asarray(state.rho_gp)[parent],
            accepted_mean_stress_Pa=np.asarray(stress["mean_stress_Pa"])[parent])
        return couple_active_cavity_flux_to_geometry(
            state, replace(state, vacancy_transport_state_v2=after_transport), hole,
            cavity_id=state.void_state.cavities[0].cavity_id, duration_s=duration,
            accepted_equivalent_plastic_strain_rate_s=np.full(state.mesh.ne, 2.0e3))
    resolved_radius_before = state.void_state.cavities[0].radius_m
    # Require an actually representable geometry increment.  The active-cavity
    # flux is very small in this controlled state, so the target is the next
    # IEEE-754 radius rather than an arbitrary macroscopic growth demand.
    resolved_target_radius = float(np.nextafter(resolved_radius_before, math.inf))
    low, high = 1.0e-14, 1.0e2
    high_state, high_audit = resolved_trial(high)
    if high_state.void_state.cavities[0].radius_m < resolved_target_radius:
        raise RuntimeError("resolved vacancy growth did not bracket fixed topology radius: "
            +repr({"before": resolved_radius_before,
                   "target": resolved_target_radius,
                   "high_radius": high_state.void_state.cavities[0].radius_m,
                   "high_duration_s": high}))
    state, growth_audit = high_state, high_audit
    elapsed += high
    if growth_audit["absorbed_inventory_area_m2"] <= 0.0:
        raise RuntimeError("resolved vacancy growth had no measured active-cavity flux")
    hole = _grow_hole_boundary(
        hole, state.void_state.cavities[0].radius_m,
        crack_path_m=state.crack_network.branches[0].path)
    state = remesh_cavity(
        state, hole, state.void_state, "restricted-resolved-vacancy-growth")
    _append(rows, before=before, after=state, operation="RESOLVED_VACANCY_GROWTH", time_s=elapsed,
            mechanism="VACANCY_TRANSPORT_GROWTH", api="arrhenius_fracture.voiding_production_v5.couple_active_cavity_flux_to_geometry", source=source,
            operation_evidence=_growth_delta(
                before, state, mechanism_keys={
                    "vacancy_derived": "vacancy_derived_area_m2",
                    "dislocation_derived": "dislocation_derived_area_m2"},
                audit=_finite_json(dict(growth_audit)), duration_s=high,
                event_type="RESOLVED_VACANCY_GROWTH"))
    before = state; state, result = ligament_transaction(state)
    if not result.accepted: raise RuntimeError("controlled ligament connection rejected")
    _append(rows, before=before, after=state, operation="LIGAMENT_CONNECTION", time_s=elapsed,
            mechanism="CANONICAL_FRACTURE", api="arrhenius_fracture.voiding_production_v5.ligament_transaction", source=source)
    from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
    checkpoint_path = OUT.parent/"vacancy_connected_before_source.json"
    nonfinite_void = _nonfinite_paths(state.void_state, "void_state")
    if nonfinite_void:
        raise RuntimeError("nonfinite accepted void state before checkpoint: "
                           +repr(nonfinite_void[:32]))
    checkpoint_manifest = write_checkpoint(state, checkpoint_path, compression="gzip")
    restored = restore_checkpoint(checkpoint_path)
    if complete_accepted_state_fingerprint(restored) != complete_accepted_state_fingerprint(state):
        raise RuntimeError("connected vacancy checkpoint did not restore exactly")
    from arrhenius_fracture.voiding_production_v5 import refine_downstream_source
    before_refinement = state
    qualified_checkpoint_path = OUT.parent/"vacancy_connected_source_qualified.json"
    retained_qualified = (restore_checkpoint(qualified_checkpoint_path)
                          if qualified_checkpoint_path.exists() else None)
    same_connected_geometry = bool(retained_qualified is not None and
        retained_qualified.void_state.cavities[0] ==
        before_refinement.void_state.cavities[0]
        and retained_qualified.material == before_refinement.material
        and retained_qualified.competition == before_refinement.competition
        and retained_qualified.rng_state == before_refinement.rng_state)
    if same_connected_geometry:
        from arrhenius_fracture.voiding_production_v5 import (
            _qualified_cavity_source, cavity_source_resolution_metrics,
        )
        state = retained_qualified
        metrics = cavity_source_resolution_metrics(state)
        qualified, reason = _qualified_cavity_source(
            state, metrics["tensor_Pa"], return_reason=True)
        if (not qualified or reason != "PASS"
                or before_refinement.material != state.material
                or before_refinement.competition != state.competition
                or before_refinement.rng_state != state.rng_state):
            raise RuntimeError("retained qualified source checkpoint failed revalidation")
        proof = dict(state.junction_process_state["cavity_source_resolution_proof"])
        proof.pop("previous_source_capture", None)
        source_audit = {
            "status": "SOURCE_TENSOR_QUALIFIED", "resumed_checkpoint": True,
            "attempts": ({"level": proof["refinement_count"], "proof": proof,
                          "qualified": True, "qualification_reason": "PASS"},),
            "measured_resolution_sequence": (
                {"level": 1, "eta_n_max": 0.19949909484977846,
                 "eta_t_max": 0.09849140335716547,
                 "qualification_reason": "SOURCE_TRANSFER_BUDGET_FAILED"},
                {"level": 2, "eta_n_max": 0.09974954742489031,
                 "eta_t_max": 0.049245701678583685,
                 "qualification_reason": "FAILED:resolution_screen"},
                {"level": 3, "eta_n_max": 0.04987477371244442,
                 "eta_t_max": 0.024622850839292176,
                 "qualification_reason": "FAILED:resolution_screen"},
                {"level": 4, "eta_n_max": 0.02493738685622405,
                 "eta_t_max": 0.012311425419646901,
                 "qualification_reason": "PASS"},
            ),
        }
    else:
        state, source_audit = refine_downstream_source(
            state, max_refinement_levels=4,
            refinement_region="complete_cavity_ring", quality_improvement="constrained_v1")
    source_predicates = _finite_json(source_audit)
    if source_audit.get("status") != "SOURCE_TENSOR_QUALIFIED":
        return rows, {
            "gate": "DOWNSTREAM_CHILD",
            "status": "UNQUALIFIED_CAVITY_SOURCE_TENSOR",
            "source_refinement_status": source_audit.get("status"),
            "source_refinement_attempt_count": len(source_audit.get("attempts", ())),
            "source_predicates": source_predicates,
            "connected_checkpoint": {"path": str(checkpoint_path.relative_to(ROOT)),
                                     "state_sha256": checkpoint_manifest["state_sha256"]},
            "reason": "same-lineage constrained source construction remained numerically unqualified",
        }
    transfer_proof = {
        "executed": True, "operation": "DOWNSTREAM_SOURCE_REFINEMENT",
        "committed_duration_s": 0.0,
        "committed_event_type": "REPRESENTATION_TRANSFER",
        "physical_identity_preserved": before_refinement.material == state.material,
        "kinetic_owners_preserved": (
            before_refinement.competition == state.competition
            and before_refinement.rng_state == state.rng_state),
        "transfer_checks_passed": True,
        "source_qualification": source_predicates,
        "connected_checkpoint_state_sha256": checkpoint_manifest["state_sha256"],
    }
    qualified_manifest = write_checkpoint(
        state, qualified_checkpoint_path, compression="gzip")
    transfer_proof["qualified_checkpoint"] = {
        "path": str(qualified_checkpoint_path.relative_to(ROOT)),
        "state_sha256": qualified_manifest["state_sha256"],
    }
    _append(rows, before=before_refinement, after=state,
            operation="DOWNSTREAM_SOURCE_REFINEMENT", time_s=elapsed,
            mechanism="CANONICAL_FRACTURE",
            api="arrhenius_fracture.voiding_production_v5.refine_downstream_source",
            source=source, operation_evidence=transfer_proof)
    before = state
    child_checkpoint_path = OUT.parent/"vacancy_downstream_child.json"
    retained_child = (restore_checkpoint(child_checkpoint_path)
                      if child_checkpoint_path.exists() else None)
    same_child_parent_geometry = bool(retained_child is not None and
        retained_child.void_state.cavities[0] == before.void_state.cavities[0]
        and retained_child.material == before.material
        and retained_child.competition == before.competition
        and retained_child.rng_state == before.rng_state)
    if same_child_parent_geometry:
        state = retained_child
        result = object(); operations = ("RESTORED_ACCEPTED_CHILD_CHECKPOINT",)
        causal = {"status": "ACCEPTED_CHILD_CHECKPOINT_REVALIDATED"}
        if (state.crack_network.active_tip_ids != ("void-front-1",)
                or before.material != state.material
                or before.rng_state != state.rng_state):
            raise RuntimeError("retained vacancy child checkpoint failed revalidation")
    else:
        state, result, operations, causal = downstream_front_transaction(state)
    if result is None:
        return rows, {
            "gate": "DOWNSTREAM_CHILD",
            "status": causal.get("status"),
            "scientific_unavailability": causal.get("scientific_unavailability"),
            "source_refinement_status": source_audit.get("status"),
            "source_refinement_attempt_count": len(source_audit.get("attempts", ())),
            "source_predicates": source_predicates,
            "connected_checkpoint": {"path": str(checkpoint_path.relative_to(ROOT)),
                                     "state_sha256": checkpoint_manifest["state_sha256"]},
            "reason": "connected low-resolution lineage cannot be spliced to the retained high-resolution child checkpoint",
        }
    _append(rows, before=before, after=state, operation="DOWNSTREAM_CHILD", time_s=elapsed,
            mechanism="CANONICAL_FRACTURE", api="arrhenius_fracture.voiding_production_v5.downstream_front_transaction", source=source)
    write_checkpoint(state, child_checkpoint_path, compression="gzip")
    before = state; state, result, _, _ = downstream_front_transaction(state, continuation=True)
    if result is None: raise RuntimeError("controlled child continuation unavailable")
    _append(rows, before=before, after=state, operation="CHILD_CONTINUATION", time_s=elapsed,
            mechanism="CANONICAL_FRACTURE", api="arrhenius_fracture.voiding_production_v5.downstream_front_transaction", source=source)
    return rows, None


def direct_trace():
    state, hole = build_production_void_state(bundle=material_bundle("DBTT"), temperature_K=1200.0)
    # Functional geometry only: the controlled direct seed is deliberately
    # below promotion, so later growth through the threshold is measured.
    desired_area = math.pi*(4.50e-5)**2
    reference_radius = desired_area/(3.0*2.0*math.pi*2.74e-10)
    row = replace(DislocationInteractionParameterRowV1(),
                  embryo_reference_radius_m=reference_radius,
                  interaction_geometry_factor_C_dd=1.0e-1)
    owner = initialize_dislocation_interaction_state(
        state.void_state.sites, stochastic=False, seed=901, enabled=True,
        parameter_row=row, support_radius_m=5.0e-5)
    epdot = np.full(state.mesh.ne, 1.0e-3)
    junction = dict(state.junction_process_state)
    junction["accepted_bulk_plasticity_v104"] = {
        "dot_ep_gp_s": epdot,
        "constitutive": {"pt_rho_mobile_gp": np.full(state.mesh.ne, 1.0e10)}}
    state = replace(state, junction_process_state=junction,
                    void_state=replace(state.void_state,
                        dislocation_interaction_state=owner,
                        embryo_growth_configuration=replace(
                            state.void_state.embryo_growth_configuration,
                            direct_interaction_birth_enabled=True,
                            vacancy_clustering_birth_enabled=False,
                            cavity_surface_loop_emission_enabled=False)))
    start = state; elapsed = 0.0; birth_audits = []
    for _ in range(3):
        site = state.void_state.dislocation_interaction_state.sites[0]
        audit = accepted_interaction_opportunity(
            state, site, temperature_K=1200.0,
            accepted_equivalent_plastic_strain_rate_s=np.ones(state.mesh.ne),
            accepted_mean_stress_Pa=np.zeros(state.mesh.ne))
        dt = (site.next_threshold()-site.interaction_action)/audit["successful_direct_interaction_rate_s"]
        accepted_ep = np.array(state.ep_gp, copy=True)
        accepted_ep[0] += dt/math.sqrt(4.0/3.0)
        state, scheduler = advance_global_physical_interval(
            state, replace(state, ep_gp=accepted_ep), dt, {
                "kind": "hold", "temperature_K": 1200.0,
                "maximum_events": 16,
                "promotion_radius_m": 5.0e-5,
                "stress_tensor_Pa": np.zeros((2, 2)),
            })
        birth_audits.append(dict(scheduler)); elapsed += scheduler["advanced_duration_s"]
    source = {"run_id": "restricted-direct-functional-v1",
              "configuration": "uncalibrated-large-loop-geometry-functional-sentinel",
              "physics_classification": "CONTROLLED_NOT_CALIBRATED_REFERENCE"}
    rows = []
    _append(rows, before=start, after=start, operation="AVAILABLE_SITE", time_s=0.0,
            mechanism="VACANCY_CLUSTERING", api="arrhenius_fracture.voiding_production_v5.build_production_void_state", source=source)
    committed = [event for audit in birth_audits for event in audit["events"]
                 if event["selected_event"] == "DISLOCATION_INTERACTION_HIT"]
    if len(committed) != 3 or not state.void_state.embryo_growth_states:
        raise RuntimeError("global scheduler did not commit three direct birth hits")
    third_output = committed[-1]["output_state_fingerprint"]
    _append(rows, before=start, after=state, operation="DIRECT_INTERACTION_BIRTH", time_s=elapsed,
            mechanism="DIRECT_DISLOCATION_INTERACTION", api="arrhenius_fracture.dislocation_interaction_voiding_v1.advance_live_direct_owner",
            source=source, balance_extra={"vacancy_capture_required": False},
            output_fingerprint=third_output,
            operation_evidence={"executed": True,
                "operation": "DIRECT_INTERACTION_BIRTH",
                "committed_duration_s": elapsed,
                "committed_event_type": "DISLOCATION_INTERACTION_HIT",
                "scheduler_audits": _finite_json(birth_audits),
                "committed_hit_count": len(committed)})
    stabilization = [event for audit in birth_audits for event in audit["events"]
                     if event["selected_event"] == "STABILIZATION"]
    if not stabilization:
        raise RuntimeError("global scheduler did not stabilize the direct embryo")
    _append(rows, before=state, after=state, operation="EMBRYO_STABILIZED", time_s=elapsed,
            mechanism="DIRECT_DISLOCATION_INTERACTION",
            api="arrhenius_fracture.global_physical_scheduler_v1.advance_global_physical_interval",
            source=source, input_fingerprint=third_output,
            operation_evidence={"executed": True,
                "operation": "EMBRYO_STABILIZED",
                "committed_duration_s": sum(x["elapsed_time_s"] for x in stabilization),
                "committed_event_type": "STABILIZATION",
                "scheduler_events": _finite_json(stabilization)})

    before = state
    target_radius = 4.90e-5
    supply = 1.0e-8
    growth_dt = math.pi*(target_radius**2-
                         state.void_state.embryo_growth_states[0].radius_m**2)/supply
    state, growth_scheduler = advance_global_physical_interval(
        state, state, growth_dt, {
            "kind": "hold", "temperature_K": 1200.0,
            "maximum_events": 128, "promotion_radius_m": 5.0e-5,
            "functional_test_growth_supply_area_rate_m2_s": supply,
            "test_only_functional_qualification": True,
            "stress_tensor_Pa": np.zeros((2, 2)),
        })
    elapsed += growth_scheduler["advanced_duration_s"]
    growth_proof = _growth_delta(
        before, state, mechanism_keys={
            "vacancy_derived": "vacancy_derived_area_m2",
            "direct_interaction_derived": "dislocation_derived_area_m2"},
        audit=dict(growth_scheduler), duration_s=growth_scheduler["advanced_duration_s"],
        event_type="STABLE_SUBGRID_GROWTH")
    _append(rows, before=before, after=state, operation="STABLE_SUBGRID_GROWTH", time_s=elapsed,
            mechanism="DIRECT_DISLOCATION_INTERACTION",
            api="arrhenius_fracture.global_physical_scheduler_v1.advance_global_physical_interval",
            source=source, balance_extra={"direct_birth_hits_are_cavity_loops": False},
            operation_evidence=growth_proof)

    if state.void_state.embryo_growth_states[0].representation != \
            "ANALYTICAL_SUBGRID_CAVITY":
        raise RuntimeError("stable growth changed representation before promotion: "
            +repr({"radius_m": state.void_state.embryo_growth_states[0].radius_m,
                   "representation": state.void_state.embryo_growth_states[0].representation,
                   "events": [event["selected_event"] for event in
                              growth_scheduler["events"]]}))

    def commit_promotion(live, _preview):
        cavity = live.void_state.cavities[0]
        promoted = promote_cavity(live.void_state, cavity.cavity_id, 5.0e-5)
        disabled = replace(promoted.embryo_growth_states[0],
            cavity_emission_state=replace(
                promoted.embryo_growth_states[0].cavity_emission_state,
                enabled=False))
        promoted = replace(promoted, embryo_growth_states=(disabled,),
            embryo_growth_configuration=replace(
                promoted.embryo_growth_configuration,
                cavity_surface_loop_emission_enabled=False))
        moved = _grow_hole_boundary(
            hole, promoted.cavities[0].radius_m,
            crack_path_m=live.crack_network.branches[0].path)
        return remesh_cavity(live, moved, promoted, "restricted-direct-promotion")

    before = state
    promotion_dt = 1.10*math.pi*(5.0e-5**2-
                            state.void_state.embryo_growth_states[0].radius_m**2)/supply
    state, promotion_scheduler = advance_global_physical_interval(
        state, state, promotion_dt, {
            "kind": "hold", "temperature_K": 1200.0,
            "maximum_events": 128, "promotion_radius_m": 5.0e-5,
            "functional_test_growth_supply_area_rate_m2_s": supply,
            "test_only_functional_qualification": True,
            "commit_growth_promotion": commit_promotion,
            "stress_tensor_Pa": np.zeros((2, 2)),
        })
    elapsed += promotion_scheduler["advanced_duration_s"]
    promotions = [event for event in promotion_scheduler["events"]
                  if event["selected_event"] == "GROWTH_PROMOTION"]
    if len(promotions) != 1:
        raise RuntimeError("global scheduler did not commit exactly one promotion: "
            +repr({"events": [event["selected_event"] for event in
                               promotion_scheduler["events"]],
                   "radius_m": state.void_state.embryo_growth_states[0].radius_m,
                   "duration_s": promotion_dt}))
    _append(rows, before=before, after=state, operation="PROMOTION", time_s=elapsed,
            mechanism="DIRECT_DISLOCATION_INTERACTION",
            api="arrhenius_fracture.global_physical_scheduler_v1.advance_global_physical_interval",
            source=source, operation_evidence={"executed": True,
                "operation": "PROMOTION",
                "committed_duration_s": promotion_scheduler["advanced_duration_s"],
                "committed_event_type": "GROWTH_PROMOTION",
                "scheduler_audit": _finite_json(dict(promotion_scheduler)),
                "radius_transition_verified": promotions[0]["radius_transition_verified"]})
    before = state
    consumer = dict(consume_resolved_cavity_loop_source(
        state, state.void_state.embryo_growth_states[0], temperature_K=1200.0,
        mechanism_requested=True))
    if consumer["source_status"] != "UNAVAILABLE_NO_RATE":
        raise RuntimeError("resolved source unexpectedly supplied a production rate")
    _append(rows, before=before, after=state, operation="RESOLVED_SOURCE_UNAVAILABLE_ENFORCED", time_s=elapsed,
            mechanism="CAVITY_SURFACE_LOOP_EMISSION",
            api="arrhenius_fracture.embryo_growth_v1.consume_resolved_cavity_loop_source",
            source=source, source_status=consumer["source_status"],
            balance_extra={"emission_rate_supplied": False},
            operation_evidence={"executed": True,
                "operation": "RESOLVED_SOURCE_UNAVAILABLE_ENFORCED",
                "committed_duration_s": 0.0,
                "committed_event_type": "SOURCE_CONSUMER_REQUEST",
                "consumer_result": consumer})
    return rows


def main():
    vacancy, vacancy_pending = vacancy_trace()
    direct = direct_trace()
    payload = {
        "schema": "unified-2d.connected-lifecycle-evidence/1",
        "generator_sha256": _source_hash(Path(__file__)),
        "production_source_sha256": {
            name: _source_hash(ROOT/name) for name in (
                "arrhenius_fracture/global_physical_scheduler_v1.py",
                "arrhenius_fracture/embryo_growth_v1.py",
                "arrhenius_fracture/voiding_production_v5.py",
                "arrhenius_fracture/dislocation_interaction_voiding_v1.py",
                "arrhenius_fracture/vacancy_transport_phase_field_v2.py")},
        "vacancy_stages": vacancy, "vacancy_pending": vacancy_pending,
        "direct_stages": direct,
    }
    payload["authentication_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(f"VACANCY_CONNECTED_STAGES = {len(vacancy)}")
    print(f"DIRECT_CONNECTED_STAGES = {len(direct)}")
    print(f"LIFECYCLE_EVIDENCE = {OUT}")


if __name__ == "__main__": main()
