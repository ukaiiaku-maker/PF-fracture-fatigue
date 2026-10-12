"""Public bounded adapter for the established branch--cavity compatibility path.

This adapter composes existing V5/V11 transactions. It does not add a branch
criterion, alter a kinetic barrier, or claim calibrated recursive branching.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_v11 import restore_checkpoint, write_checkpoint
from .topology_transaction_v11 import complete_accepted_state_fingerprint
from .unified_fracture_material_v5 import material_bundle
from .voiding_production_v5 import (
    deterministic_trajectory,
    downstream_front_transaction,
    equilibrate_fixed_load_with_production_fem,
    ligament_transaction,
    observables,
)


MODEL_ID = "shared.branch-cavity-native-compatibility/1"
FIXED_PATH_M = ((0.0, 0.0), (0.0005725993004046688, 0.0))


def _hash_array(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _shared_fields(state) -> dict[str, Any]:
    return {
        "displacement_sha256": _hash_array(state.displacement),
        "plastic_strain_sha256": _hash_array(state.ep_gp),
        "density_sha256": _hash_array(state.rho_gp),
        "damage_sha256": _hash_array(state.damage),
        "vacancy_fraction_sha256": _hash_array(state.vacancy_fraction_gp),
        "mesh_nodes_sha256": _hash_array(state.mesh.nodes),
        "mesh_elements_sha256": _hash_array(state.mesh.elems),
        "vacancy_ledgers": dict(state.vacancy_ledgers),
        "field_transfer_audit": state.junction_process_state.get("latest_void_remesh_audit"),
    }


def _owner_summary(state) -> dict[str, Any]:
    cavity_ids = [] if state.void_state is None else [
        cavity.cavity_id for cavity in state.void_state.cavities
    ]
    return {
        "branch_ids": [branch.branch_id for branch in state.crack_network.branches],
        "branch_status": {
            branch.branch_id: branch.status for branch in state.crack_network.branches
        },
        "active_tip_ids": list(state.crack_network.active_tip_ids),
        "cavity_ids": cavity_ids,
        "tip_process_owner_ids": sorted(state.tip_process_state.get("by_branch", {}).keys()),
        "competition_candidate_ids": [
            candidate.candidate_id for candidate in state.competition.candidates
        ],
    }


def _construct_current_combined_state():
    """Use the retained current-row, source-conforming V5/V11 sequence."""
    precursor, _ = deterministic_trajectory(
        bundle=material_bundle("DBTT"),
        stop_before_ligament=True,
        boundary_segments=64,
        radial_layers=24,
        crack_path_m=FIXED_PATH_M,
        source_conforming_geometry=True,
    )
    loaded = equilibrate_fixed_load_with_production_fem(
        replace(precursor, displacement=precursor.displacement * 2.0)
    )
    connected, ligament = ligament_transaction(loaded)
    child, child_result, child_operations, child_audit = downstream_front_transaction(connected)
    if child_result is None or not child_result.accepted:
        raise RuntimeError("established downstream child transaction did not accept")
    terminal, continued_result, continued_operations, continued_audit = (
        downstream_front_transaction(child, continuation=True)
    )
    if continued_result is None or not continued_result.accepted:
        raise RuntimeError("established child continuation did not accept")
    return child, terminal, {
        "mode": "uninterrupted_construction_and_continuation",
        "event_sequence": [
            "source_conforming_resolved_cavity",
            "documented_positive_opening_continuation",
            "crack_to_cavity_ligament",
            "cavity_surface_downstream_child",
            "sharp_front_child_continuation",
        ],
        "ligament_energy_margin_J_per_m": ligament.energy_margin_J_per_m,
        "child_operations": child_operations,
        "child_causal_first_passage": child_audit,
        "continued_operations": continued_operations,
        "continued_causal_first_passage": continued_audit,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cavity-enabled", action="store_true")
    parser.add_argument("--cavity-disabled", action="store_true")
    parser.add_argument("--restart-checkpoint")
    parser.add_argument("--reference-terminal-checkpoint")
    parser.add_argument("--save-snapshots")
    parser.add_argument("--no-plots", action="store_true")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        raise SystemExit("unsupported branch-cavity adapter arguments: " + " ".join(unknown))
    if not args.cavity_enabled or args.cavity_disabled:
        raise SystemExit("combined branch-cavity adapter requires the enabled cavity switch")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.restart_checkpoint:
        source = restore_checkpoint(args.restart_checkpoint)
        initial_fingerprint = complete_accepted_state_fingerprint(source)
        terminal, result, operations, causal = downstream_front_transaction(
            source, continuation=True
        )
        if result is None or not result.accepted:
            raise RuntimeError("restarted combined continuation did not accept")
        execution = {
            "mode": "restarted_continuation",
            "restart_checkpoint": str(Path(args.restart_checkpoint).resolve()),
            "initial_fingerprint": initial_fingerprint,
            "event_sequence": ["restore_combined_owner_state", "sharp_front_child_continuation"],
            "continued_operations": operations,
            "continued_causal_first_passage": causal,
        }
        source_checkpoint = None
    else:
        source, terminal, execution = _construct_current_combined_state()
        source_checkpoint = out / "combined_source_checkpoint.json"
        write_checkpoint(source, source_checkpoint, compression="gzip")

    terminal_checkpoint = out / "combined_terminal_checkpoint.json"
    write_checkpoint(terminal, terminal_checkpoint, compression="gzip")
    restored_terminal = restore_checkpoint(terminal_checkpoint)
    terminal_fingerprint = complete_accepted_state_fingerprint(terminal)
    round_trip_exact = (
        complete_accepted_state_fingerprint(restored_terminal) == terminal_fingerprint
    )
    reference_exact = None
    if args.reference_terminal_checkpoint:
        reference = restore_checkpoint(args.reference_terminal_checkpoint)
        reference_exact = complete_accepted_state_fingerprint(reference) == terminal_fingerprint
    owners = _owner_summary(terminal)
    connection = terminal.junction_process_state.get(
        "latest_crack_void_connection_certificate", {}
    )
    passed = (
        round_trip_exact
        and len(owners["branch_ids"]) >= 2
        and len(owners["cavity_ids"]) >= 1
        and bool(connection.get("passed"))
        and (reference_exact is not False)
    )
    payload = {
        "schema": MODEL_ID,
        "status": "PASS" if passed else "FAIL",
        "scope": "bounded native compatibility fixture",
        "production_eligible": False,
        "recursive_branching_calibrated": False,
        "spontaneous_joint_birth_required": False,
        "cavity_feedback_switch": "enabled",
        "cavity_toggle_changes_branch_permission": False,
        "material_row": "DBTT current unified row",
        "loading": "retained 0.8 micrometre positive-opening continuation",
        "source_conforming_mesh": {"boundary_segments": 64, "radial_layers": 24},
        "execution": execution,
        "owners": owners,
        "terminal": observables(terminal, "combined_terminal"),
        "shared_fields": _shared_fields(terminal),
        "crack_void_connection_certificate": connection,
        "terminal_fingerprint": terminal_fingerprint,
        "source_checkpoint": None if source_checkpoint is None else str(source_checkpoint),
        "terminal_checkpoint": str(terminal_checkpoint),
        "native_checkpoint_round_trip_exact": round_trip_exact,
        "reference_terminal_exact": reference_exact,
        "limitation": (
            "The calculation qualifies state composition, event ordering, field transfer, "
            "and restart using existing laws. It does not calibrate recursive branch birth."
        ),
    }
    (out / "shared_branch_cavity_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    if not passed:
        raise RuntimeError("combined branch-cavity compatibility qualification failed")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MODEL_ID", "main"]
