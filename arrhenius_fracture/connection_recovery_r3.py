"""R3 recovery of an admitted crack/void connection and downstream front.

This module deliberately uses the retained V5 production transactions.  It
does not modify a barrier, surface cost, clock, or admission decision.  The
only difference between the veto and accepted fixtures is the documented
geometry/loading history supplied to those transactions.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .checkpoint_v11 import restore_checkpoint, write_checkpoint
from .topology_transaction_v11 import complete_accepted_state_fingerprint
from .unified_fracture_material_v5 import material_bundle, require_bound_identity
from . import voiding_production_v5 as production


HISTORICAL_ACCEPTED_SOURCE_COMMIT = "f6ae6a86"
RETAINED_CRACK_PATH_M = ((0.0, 0.0), (0.0005725993004046688, 0.0))
ACCEPTED_CAVITY_CENTER_M = (7.0e-4, 1.0e-5)
VETO_CAVITY_CENTER_M = (7.0e-4, 0.0)
ACCEPTED_OPENING_SCALE = 2.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0.0 else "-infinity"
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _head() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=Path(__file__).resolve().parents[1], text=True,
    ).strip()


def _equilibrium(state) -> dict[str, Any]:
    value = state.equilibrium_observables
    return {} if value is None else _jsonable(value.to_dict())


def _state_receipt(state) -> dict[str, Any]:
    cavity = state.void_state.cavities[0]
    return {
        "fingerprint": complete_accepted_state_fingerprint(state),
        "geometry_generation": int(state.crack_network.geometry_generation),
        "mesh_generation": int(state.event_counters.get("mesh_generation", 0)),
        "topology_actions": int(state.event_counters.get("topology_actions", 0)),
        "active_tip_ids": list(state.crack_network.active_tip_ids),
        "void_phase": cavity.phase.value,
        "cavity": _jsonable(cavity),
        "length_ledgers": dict(state.void_state.length_ledgers),
        "vacancy_ledgers": dict(state.vacancy_ledgers),
        "available_defect_inventory_area_m2": float(
            state.void_state.available_defect_inventory_area_m2
        ),
        "consumed_defect_inventory_area_m2": float(
            state.void_state.consumed_defect_inventory_area_m2
        ),
        "equilibrium": _equilibrium(state),
        "connection_certificate": _jsonable(
            state.junction_process_state.get("latest_crack_void_connection_certificate")
        ),
        "topology_certificate_stage": state.junction_process_state.get(
            "latest_topology_certificate_stage"
        ),
    }


def _captured_ligament_trial(state, *, operation_log: list[str]):
    """Call the public transaction while retaining its veto result for audit."""
    captured = []
    original = production.execute_topology_trial

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    production.execute_topology_trial = observe
    try:
        try:
            accepted_state, public_result = production.ligament_transaction(
                state, operation_log=operation_log,
            )
            failure = None
        except RuntimeError as exc:
            accepted_state, public_result, failure = state, None, str(exc)
    finally:
        production.execute_topology_trial = original
    if len(captured) != 1:
        raise RuntimeError("ligament transaction did not execute exactly one energy trial")
    return accepted_state, public_result, captured[0], failure


def _trial_receipt(result) -> dict[str, Any]:
    return {
        "accepted": bool(result.accepted),
        "action_id": result.action_id,
        "energy_release_J_per_m": float(result.energy_release_J_per_m),
        "hazard_dissipation_J_per_m": float(result.hazard_dissipation_J_per_m),
        "energy_margin_J_per_m": float(result.energy_margin_J_per_m),
        "rejection_reason": result.rejection_reason,
        "trial_copy_bytes": int(result.trial_copy_bytes),
        "trial_copy_wall_time_s": float(result.trial_copy_wall_time_s),
    }


def run_connection_recovery(output_directory: str | Path) -> dict[str, Any]:
    """Execute veto, accepted connection, restart, child birth and continuation."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    bundle = material_bundle("DBTT")

    veto_pre, _ = production.deterministic_trajectory(
        bundle=bundle, stop_before_ligament=True,
        cavity_center_m=VETO_CAVITY_CENTER_M,
    )
    veto_before = _state_receipt(veto_pre)
    veto_operations: list[str] = []
    veto_after, _, veto_trial, veto_failure = _captured_ligament_trial(
        veto_pre, operation_log=veto_operations,
    )
    veto_after_receipt = _state_receipt(veto_after)
    veto = {
        "fixture": "CENTERED_NATIVE_LOAD_VETO",
        "input": {
            "cavity_center_m": list(VETO_CAVITY_CENTER_M),
            "opening_scale": 1.0,
            "crack_path_policy": "production_default",
        },
        "trial": _trial_receipt(veto_trial),
        "public_failure": veto_failure,
        "operations": veto_operations,
        "before": veto_before,
        "after": veto_after_receipt,
        "rollback_exact": veto_before["fingerprint"] == veto_after_receipt["fingerprint"],
    }

    pre, _ = production.deterministic_trajectory(
        bundle=bundle, stop_before_ligament=True,
        cavity_center_m=ACCEPTED_CAVITY_CENTER_M,
        crack_path_m=RETAINED_CRACK_PATH_M,
    )
    pre = production.equilibrate_fixed_load_with_production_fem(
        replace(pre, displacement=ACCEPTED_OPENING_SCALE * pre.displacement)
    )
    pre_receipt = _state_receipt(pre)
    start = pre.crack_network.branch(production.ROOT_BRANCH_ID).tip
    accepted_operations: list[str] = []
    connected, public_ligament, accepted_trial, accepted_failure = _captured_ligament_trial(
        pre, operation_log=accepted_operations,
    )
    if accepted_failure is not None or public_ligament is None or not public_ligament.accepted:
        raise RuntimeError("documented R3 accepted ligament fixture was vetoed")
    connected_receipt = _state_receipt(connected)
    cavity = connected.void_state.cavities[0]
    entry = tuple(map(float, cavity.connection_entry_m))
    exit_point = tuple(map(float, cavity.connection_exit_m))

    connected_checkpoint = output / "connected.checkpoint.json"
    checkpoint_manifest = write_checkpoint(connected, connected_checkpoint)
    restored_connected = restore_checkpoint(connected_checkpoint)
    connected_restart_exact = (
        complete_accepted_state_fingerprint(restored_connected)
        == complete_accepted_state_fingerprint(connected)
    )

    direct_birth_operations: list[str] = []
    direct_child, direct_birth, _, direct_birth_causal = production.downstream_front_transaction(
        connected, operation_log=direct_birth_operations,
    )
    restart_birth_operations: list[str] = []
    restart_child, restart_birth, _, restart_birth_causal = production.downstream_front_transaction(
        restored_connected, operation_log=restart_birth_operations,
    )
    if direct_birth is None or restart_birth is None:
        raise RuntimeError("documented downstream child first passage did not occur")
    birth_restart_exact = (
        complete_accepted_state_fingerprint(direct_child)
        == complete_accepted_state_fingerprint(restart_child)
    )

    child_checkpoint = output / "downstream-child.checkpoint.json"
    child_manifest = write_checkpoint(restart_child, child_checkpoint)
    restored_child = restore_checkpoint(child_checkpoint)
    child_restart_exact = (
        complete_accepted_state_fingerprint(restored_child)
        == complete_accepted_state_fingerprint(restart_child)
    )

    direct_cont_operations: list[str] = []
    direct_terminal, direct_cont, _, direct_cont_causal = production.downstream_front_transaction(
        direct_child, continuation=True, operation_log=direct_cont_operations,
    )
    restart_cont_operations: list[str] = []
    restart_terminal, restart_cont, _, restart_cont_causal = production.downstream_front_transaction(
        restored_child, continuation=True, operation_log=restart_cont_operations,
    )
    if direct_cont is None or restart_cont is None:
        raise RuntimeError("documented downstream child continuation did not occur")
    continuation_restart_exact = (
        complete_accepted_state_fingerprint(direct_terminal)
        == complete_accepted_state_fingerprint(restart_terminal)
    )

    accepted = {
        "fixture": "OFFSET_RETAINED_PATH_DOUBLE_OPENING_ACCEPTED",
        "input": {
            "material_class": "DBTT",
            "material_identity": dict(require_bound_identity(pre)),
            "cavity_center_m": list(ACCEPTED_CAVITY_CENTER_M),
            "crack_path_m": _jsonable(RETAINED_CRACK_PATH_M),
            "opening_scale": ACCEPTED_OPENING_SCALE,
            "loading_control": "fixed_symmetric_dirichlet_opening",
        },
        "geometry_eligibility": {
            "crack_tip_m": list(start),
            "connection_entry_m": list(entry),
            "connection_exit_m": list(exit_point),
            "remaining_ligament_m": math.dist(start, entry),
            "preexisting_cavity_chord_m": math.dist(entry, exit_point),
            "candidate_ray_first_intersection_is_entry": True,
        },
        "trial": _trial_receipt(accepted_trial),
        "operations": accepted_operations,
        "before": pre_receipt,
        "connected": connected_receipt,
        "vacancy_debit_during_ligament": {
            key: float(connected.vacancy_ledgers[key] - pre.vacancy_ledgers[key])
            for key in pre.vacancy_ledgers
        },
        "checkpoint": {
            "manifest_sha256": _digest(checkpoint_manifest),
            "restart_exact": connected_restart_exact,
        },
        "downstream_child": {
            "trial": _trial_receipt(direct_birth),
            "operations": direct_birth_operations,
            "causal_status": _jsonable(direct_birth_causal),
            "state": _state_receipt(direct_child),
            "restored_execution_operations": restart_birth_operations,
            "restored_causal_status": _jsonable(restart_birth_causal),
            "restart_exact": birth_restart_exact,
            "checkpoint_manifest_sha256": _digest(child_manifest),
            "checkpoint_restart_exact": child_restart_exact,
        },
        "downstream_continuation": {
            "trial": _trial_receipt(direct_cont),
            "operations": direct_cont_operations,
            "causal_status": _jsonable(direct_cont_causal),
            "state": _state_receipt(direct_terminal),
            "restored_execution_operations": restart_cont_operations,
            "restored_causal_status": _jsonable(restart_cont_causal),
            "restart_exact": continuation_restart_exact,
        },
    }

    result = {
        "schema": "r3.connection-recovery/1",
        "implementation_head": _head(),
        "historical_accepted_source_commit": HISTORICAL_ACCEPTED_SOURCE_COMMIT,
        "decision_policy": "NATIVE_TOPOLOGY_ENERGY_GATE_NO_PARAMETER_OR_COST_CHANGE",
        "veto": veto,
        "accepted": accepted,
        "passed": bool(
            not veto_trial.accepted
            and veto["rollback_exact"]
            and accepted_trial.accepted
            and connected_restart_exact
            and direct_birth.accepted
            and birth_restart_exact
            and child_restart_exact
            and direct_cont.accepted
            and continuation_restart_exact
        ),
    }
    (output / "connection_recovery.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
    )
    return result

