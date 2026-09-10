"""Prospective scientific replay comparator, distinct from V1 bitwise replay."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

SCHEMA = "v5.natural-future-physical-replay/2"
ROUND_OFF_FACTOR = 64.0
RESIDUAL_FACTOR = 4.0
DECISION_MARGIN_FRACTION = 0.01


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_canonical(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _maximum_ulp_difference(first: np.ndarray, second: np.ndarray) -> int:
    """Return the largest representable-float step between finite arrays."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("ULP comparison requires equal-shape finite arrays")
    def ordered(value):
        bits = value.view(np.int64)
        return np.where(bits < 0, np.iinfo(np.int64).min - bits, bits)
    delta = np.abs(ordered(left).astype(object) - ordered(right).astype(object))
    return int(max(delta.flat, default=0))


def _branch_exact(branch) -> dict:
    return {
        "branch_id": branch.branch_id,
        "parent_branch_id": branch.parent_branch_id,
        "generation": branch.generation,
        "initiation_event": branch.initiation_event,
        "path": _canonical(branch.path),
        "orientation_history_rad": _canonical(branch.orientation_history_rad),
        "status": branch.status,
        "local_state_noncontinuous": {
            str(key): _canonical(value) for key, value in branch.local_state.items()
            if key != "r_tip_m" and not isinstance(value, (float, np.floating))
        },
    }


def exact_projection(state) -> dict:
    """Return only quantities whose V2 contract requires exact equality."""
    competition = state.competition
    hazards = []
    for hazard in competition.hazard_states:
        hazards.append({
            "candidate_id": hazard.candidate_id,
            "completed_event_count": hazard.completed_event_count,
            "current_threshold_action": hazard.current_threshold_action,
            "threshold_process": hazard.threshold_process,
            "threshold_seed": hazard.threshold_seed,
            "pending_event_identities": [
                {"event_id": event.event_id, "candidate_id": event.candidate_id,
                 "event_ordinal": event.event_ordinal}
                for event in hazard.pending_events
            ],
        })
    cavity_topology = []
    if state.void_state is not None:
        for cavity in state.void_state.cavities:
            cavity_topology.append({
                "cavity_id": cavity.cavity_id,
                "parent_site_id": cavity.parent_site_id,
                "phase": cavity.phase.value,
                "lineage": _canonical(cavity.lineage),
                "center_m": _canonical(cavity.center_m),
                "connection_entry_m": _canonical(cavity.connection_entry_m),
                "connection_exit_m": _canonical(cavity.connection_exit_m),
                "connection_direction_xy": _canonical(cavity.connection_direction_xy),
            })
        site_phases = [{"site_id": site.site_id, "phase": site.phase.value}
                       for site in state.void_state.sites]
        event_order = [
            {str(key): _canonical(value) for key, value in event.items()
             if not isinstance(value, (float, np.floating))}
            for event in state.void_state.event_history
        ]
    else:
        site_phases, event_order = [], []
    support = state.v12_support_state
    support_exact = None if support is None else {
        "mesh_connectivity_fingerprint": support.mesh_connectivity_fingerprint,
        "mesh_generation": support.mesh_generation,
        "complete_crack_graph_fingerprint": support.complete_crack_graph_fingerprint,
        "certification_arc_fingerprint": support.certification_arc_fingerprint,
        "selected_support_elements": list(support.selected_support_elements),
        "accepted_p0_damage_fingerprint": support.accepted_p0_damage_fingerprint,
        "active_tip_identities": list(support.active_tip_identities),
        "transaction_identity": support.transaction_identity,
        "previous_accepted_transaction": support.previous_accepted_transaction,
        "checkpoint_generation": support.checkpoint_generation,
        "branch_vertex_lineage_fingerprint": support.branch_vertex_lineage_fingerprint,
        "model_id": support.model_id,
        "schema_version": support.schema_version,
    }
    return {
        "mesh_connectivity_sha256": _digest(np.asarray(state.mesh.elems, dtype=np.int64)),
        "boundary_index_identity": _digest(state.boundary),
        "crack_topology": [_branch_exact(branch) for branch in state.crack_network.branches],
        "active_front_ownership": list(state.crack_network.active_tip_ids),
        "crack_geometry_generation": state.crack_network.geometry_generation,
        "cavity_topology": cavity_topology,
        "site_phases": site_phases,
        "event_order": event_order,
        "candidate_inventory": [_canonical(candidate.to_dict()) for candidate in competition.candidates],
        "threshold_and_event_identity": hazards,
        "competition_event_index": competition.competition_event_index,
        "global_hazard_seed": competition.global_hazard_seed,
        "rng_state": _canonical(state.rng_state),
        "event_counters": _canonical(state.event_counters),
        "checkpoint_generation": state.checkpoint_generation,
        "support_topology": support_exact,
        "sharp_wake_model_id": state.sharp_wake_model_id,
    }


def _numeric_arrays(state) -> dict[str, np.ndarray]:
    arrays = {
        "mesh_coordinates_m": np.asarray(state.mesh.nodes, dtype=float),
        "damage": np.asarray(state.damage, dtype=float),
        "displacement_m": np.asarray(state.displacement, dtype=float),
        "plastic_strain": np.asarray(state.ep_gp, dtype=float),
        "density_m-2": np.asarray(state.rho_gp, dtype=float),
        "elasticity_Pa": np.asarray(state.elasticity_D, dtype=float),
        "stored_energy_J_per_m": np.asarray([state.stored_energy_J_per_m], dtype=float),
        "hazard_action": np.asarray([hazard.action for hazard in state.competition.hazard_states], dtype=float),
        "hazard_residual_action": np.asarray([hazard.residual_action for hazard in state.competition.hazard_states], dtype=float),
        "hazard_previous_rate_s-1": np.asarray([
            0.0 if hazard.previous_rate_per_s is None else hazard.previous_rate_per_s
            for hazard in state.competition.hazard_states
        ], dtype=float),
    }
    if state.void_state is not None:
        arrays.update({
            "cavity_radius_m": np.asarray([cavity.radius_m for cavity in state.void_state.cavities], dtype=float),
            "cavity_area_m2": np.asarray([cavity.area_m2 for cavity in state.void_state.cavities], dtype=float),
            "cavity_inventory_m2": np.asarray([cavity.inventory_area_m2 for cavity in state.void_state.cavities], dtype=float),
            "available_inventory_m2": np.asarray([state.void_state.available_defect_inventory_area_m2], dtype=float),
            "consumed_inventory_m2": np.asarray([state.void_state.consumed_defect_inventory_area_m2], dtype=float),
        })
    numeric_ledgers = {key: value for key, value in state.energy_ledgers.items()
                       if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)}
    for key, value in sorted(numeric_ledgers.items()):
        arrays["energy_ledger." + str(key)] = np.asarray([value], dtype=float)
    return arrays


def _stress_and_reaction(state) -> tuple[np.ndarray, np.ndarray]:
    from .fem import assemble_mechanics
    _, residual, stress, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    top = np.asarray(state.boundary.top_nodes, dtype=int)
    bottom = np.asarray(state.boundary.bot_nodes, dtype=int)
    reaction = np.asarray([
        np.sum(residual[2 * top + 1]), np.sum(residual[2 * bottom + 1]), np.linalg.norm(residual),
    ], dtype=float)
    return np.asarray(stress, dtype=float), reaction


def compare_states(reference, replay, *, case_identity: str, seed: int,
                   solver_condition_number: float, free_residual_relative: float,
                   subsequent_crossing: Mapping[str, Any] | None) -> dict:
    condition = float(solver_condition_number)
    residual = float(free_residual_relative)
    if not math.isfinite(condition) or condition < 1.0:
        raise ValueError("finite solver condition number >= 1 is required")
    if not math.isfinite(residual) or residual < 0.0:
        raise ValueError("finite nonnegative free residual is required")
    exact_a, exact_b = exact_projection(reference), exact_projection(replay)
    exact_checks = {
        "case_and_seed_identity": bool(str(case_identity) and isinstance(seed, int)),
        "complete_exact_projection": exact_a == exact_b,
    }
    arrays_a, arrays_b = _numeric_arrays(reference), _numeric_arrays(replay)
    stress_a, reaction_a = _stress_and_reaction(reference)
    stress_b, reaction_b = _stress_and_reaction(replay)
    arrays_a.update({"stress_Pa": stress_a, "reaction_and_residual_N_per_m": reaction_a})
    arrays_b.update({"stress_Pa": stress_b, "reaction_and_residual_N_per_m": reaction_b})
    if set(arrays_a) != set(arrays_b):
        raise ValueError("continuous state inventory differs")
    relative_limit = max(
        ROUND_OFF_FACTOR * np.finfo(float).eps * condition,
        RESIDUAL_FACTOR * residual,
    )
    comparisons = []
    for name in sorted(arrays_a):
        a, b = arrays_a[name], arrays_b[name]
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            comparisons.append({"quantity": name, "passed": False, "reason": "shape_or_finiteness"})
            continue
        scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), np.finfo(float).tiny)
        absolute_error = float(np.linalg.norm(a - b))
        normalized_error = absolute_error / scale
        limit = relative_limit * scale
        comparisons.append({
            "quantity": name, "units_from_quantity_name": name.split("_", 1)[-1],
            "physical_scale": scale, "absolute_error": absolute_error,
            "normalized_error": normalized_error, "relative_limit": relative_limit,
            "absolute_limit": limit,
            "maximum_ulp_difference": _maximum_ulp_difference(a, b),
            "passed": absolute_error <= limit,
        })
    threshold_margins = [
        abs(hazard.current_threshold_action - hazard.action)
        for hazard in reference.competition.hazard_states
    ]
    action_errors = [
        abs(first.action - second.action)
        for first, second in zip(reference.competition.hazard_states, replay.competition.hazard_states)
    ]
    minimum_threshold_margin = min(threshold_margins, default=math.inf)
    maximum_action_error = max(action_errors, default=0.0)
    threshold_safe = (
        math.isinf(minimum_threshold_margin)
        or maximum_action_error < DECISION_MARGIN_FRACTION * minimum_threshold_margin
    )
    crossing = dict(subsequent_crossing or {})
    selection_margin = crossing.get("minimum_event_selection_margin_action")
    selection_error = crossing.get("maximum_event_selection_perturbation_action")
    selection_safe = bool(
        isinstance(selection_margin, (int, float))
        and isinstance(selection_error, (int, float))
        and math.isfinite(float(selection_margin))
        and math.isfinite(float(selection_error))
        and float(selection_margin) > 0.0
        and 0.0 <= float(selection_error) < DECISION_MARGIN_FRACTION * float(selection_margin)
    )
    crossing_passed = bool(
        crossing.get("real_crossing_executed")
        and crossing.get("selected_event_identity_exact")
        and crossing.get("accepted_topology_exact")
        and crossing.get("categorical_terminal_exact")
        and selection_safe
    )
    passed = bool(
        all(exact_checks.values())
        and all(row["passed"] for row in comparisons)
        and threshold_safe
        and crossing_passed
    )
    return {
        "schema": SCHEMA,
        "case_identity": str(case_identity),
        "seed": seed,
        "comparison_kind": "FUTURE_PHYSICAL_REPLAY_NOT_BITWISE_EQUALITY",
        "solver_condition_number": condition,
        "free_residual_relative": residual,
        "round_off_factor": ROUND_OFF_FACTOR,
        "residual_factor": RESIDUAL_FACTOR,
        "decision_margin_fraction": DECISION_MARGIN_FRACTION,
        "exact_checks": exact_checks,
        "exact_projection_sha256": {"reference": _digest(exact_a), "replay": _digest(exact_b)},
        "continuous_comparisons": comparisons,
        "threshold_safety": {
            "minimum_threshold_margin_action": minimum_threshold_margin,
            "maximum_replay_action_error": maximum_action_error,
            "passed": threshold_safe,
        },
        "subsequent_real_crossing": crossing,
        "event_selection_safety": {
            "minimum_margin_action": selection_margin,
            "maximum_perturbation_action": selection_error,
            "passed": selection_safe,
        },
        "passed": passed,
    }


__all__ = ["SCHEMA", "compare_states", "exact_projection"]
