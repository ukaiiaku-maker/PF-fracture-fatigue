#!/usr/bin/env python3
"""Execute the selected local-incremental P7 drive on native spatial state."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.global_physical_scheduler_v1 import (
    commit_ligament_event, preview_ligament_first_passage,
)
from arrhenius_fracture.prospective_campaign_v3 import REGISTRY
from arrhenius_fracture.shared_v5_cavity_adapter import _build_pair, _finite
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
)
from arrhenius_fracture.voiding_production_v5 import (
    _bundle_from_state, directional_sharp_front_rates,
    dispatch_cavity_intersecting_candidate_drives,
    fresh_sharp_front_engine,
    local_incremental_cavity_candidate_drives,
    qualified_boundary_fitted_source_observation,
    sharp_front_load_provider,
)


DEFAULT_OPTION = "v913_paper_dbtt01_0202500_persistent_sites__taylor_margin_0p10eV"


def qualify(*, registry: Path, option: str, temperature_K: float,
            total_opening_m: float, checkpoint_path: Path | None = None) -> dict:
    args = SimpleNamespace(
        material_registry=str(registry.resolve()), parameter_option=option,
        historical_v5_material=False, material_class="DBTT",
        temperature_K=float(temperature_K), vacancy_energy_variant="reference",
        seed=1733, preexisting_cavity_radius_pz=0.25,
        preexisting_cavity_separation_pz=0.1, process_zone_length_um=50.0,
        initial_crack_length_um=500.0,
        boundary_segments=32, radial_layers=12,
        shape_regular_local_level=None,
        source_conforming_geometry=True,
    )
    _disabled, state, construction = _build_pair(args)
    displacement = np.asarray(state.displacement, dtype=float).copy()
    displacement[2*np.asarray(state.boundary.top_nodes, dtype=int)+1] = total_opening_m/2
    displacement[2*np.asarray(state.boundary.bot_nodes, dtype=int)+1] = -total_opening_m/2
    displacement[2*int(state.boundary.left_bot)+1] = -total_opening_m/2
    state = equilibrate_fixed_load_with_production_fem(
        replace(state, displacement=displacement)
    )
    before = complete_accepted_state_fingerprint(state)
    ordinary = sharp_front_load_provider(
        state, branch_id=ROOT_BRANCH_ID, matched_discretization=True,
        marginal_delta_a_m=(3.0e-5, 2.0e-5), marginal_mesh_levels=(4,),
        virtual_mechanical_trial_transport="deferred_read_only",
    )
    event_advance_m = float(
        fresh_sharp_front_engine(state.material, _bundle_from_state(state)).f.da
    )
    local = local_incremental_cavity_candidate_drives(
        state, branch_id=ROOT_BRANCH_ID, maximum_event_advance_m=event_advance_m,
        refinement_levels=(3, 4),
    )
    dispatched = dispatch_cavity_intersecting_candidate_drives(
        state, branch_id=ROOT_BRANCH_ID,
        directional_rows=ordinary["directional"],
        maximum_event_advance_m=event_advance_m, refinement_levels=(3, 4),
        local_incremental_rows=local,
    )
    rates = directional_sharp_front_rates(
        state, dispatched, branch_id=ROOT_BRANCH_ID,
        temperature_K=float(temperature_K),
    )
    after = complete_accepted_state_fingerprint(state)
    selected = [
        row for row in dispatched
        if row.get("ligament_kinetic_interpretation") == "local_incremental_front"
    ]
    preview = preview_ligament_first_passage(
        state, temperature_K=float(temperature_K), maximum_duration_s=1.0e-4,
        front_load_rows=dispatched,
    )
    connected = None
    event_audit = None
    if preview.event_type == "LIGAMENT_RUPTURE":
        connected, event_audit = commit_ligament_event(state, preview)
    source_audit = None
    if connected is not None:
        source_audit = qualified_boundary_fitted_source_observation(connected)
    checkpoint_manifest = None
    checkpoint_round_trip = False
    if connected is not None and checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_manifest = write_checkpoint(
            connected, checkpoint_path, compression="gzip"
        )
        restored = restore_checkpoint(checkpoint_path)
        checkpoint_round_trip = (
            complete_accepted_state_fingerprint(restored)
            == complete_accepted_state_fingerprint(connected)
        )
    checks = {
        "native_preexisting_cavity_constructed": construction[
            "conditional_preexisting_geometry"
        ],
        "at_least_one_contact_candidate": bool(selected),
        "all_contact_drives_available": bool(selected) and all(
            row["directional_drive_status"] == "AVAILABLE" for row in selected
        ),
        "all_contact_drives_are_local_incremental": bool(selected) and all(
            row["drive_source"] == "EXACT_FIXED_VOID_FINITE_INCREMENT_G"
            and row["complete_ligament_energy_used_as_rate_argument"] is False
            and row["cavity_free_span_charged_as_solid"] is False
            and row["ordinary_front_owner_preserved"] is True
            for row in selected
        ),
        "positive_solid_increment_only": bool(selected) and all(
            0.0 < row["virtual_solid_increment_m"] <= row["new_solid_length_m"]
            <= event_advance_m for row in selected
        ),
        "rate_consumer_executed": any(
            row["candidate_id"] in {item["candidate_id"] for item in selected}
            and row["directional_drive_status"] == "AVAILABLE"
            for row in rates
        ),
        "accepted_state_preserved": before == after,
        "source_observation_did_not_advance_clocks": bool(
            source_audit is not None
            and source_audit.get("physical_time_advanced") is False
            and source_audit.get("transport_solve_performed") is False
        ),
        "native_ligament_event_selected": preview.event_type == "LIGAMENT_RUPTURE",
        "native_ligament_transaction_accepted": bool(
            event_audit is not None and event_audit.accepted
        ),
        "post_event_equilibrium": bool(
            connected is not None
            and connected.equilibrium_observables.free_dof_residual_relative <= 1.0e-8
        ),
        "ordinary_front_became_connected_dormant": bool(
            connected is not None
            and not connected.crack_network.active_tip_ids
            and connected.void_state.cavities[0].phase.value == "CONNECTED_VOID"
        ),
        "downstream_source_qualified": bool(
            source_audit is not None
            and source_audit.get("qualification_status") == "AVAILABLE"
        ),
        "post_event_checkpoint_round_trip_exact": checkpoint_round_trip,
    }
    return {
        "schema": "r10.current-local-incremental-native-qualification/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True,
        ).strip(),
        "production_execution_authorized": False,
        "material_registry": str(registry.resolve()),
        "parameter_option": option,
        "temperature_K": float(temperature_K),
        "qualification_total_opening_m": float(total_opening_m),
        "qualification_input_scope": "CONSTRUCTED_NATIVE_SPECIMEN_STATE_NOT_PRODUCTION_CALIBRATION",
        "construction": construction,
        "checks": checks,
        "accepted_state_fingerprint": before,
        "selected_contact_rows": selected,
        "consumed_rate_rows": [
            row for row in rates
            if row["candidate_id"] in {item["candidate_id"] for item in selected}
        ],
        "event_preview": {
            "event_type": preview.event_type,
            "crossing_time_s": preview.crossing_time_s,
            "candidate_ids": preview.candidate_ids,
            "event_ids": preview.event_ids,
        },
        "event_transaction": None if event_audit is None else {
            "accepted": event_audit.accepted,
            "rejection_reason": event_audit.rejection_reason,
            "energy_release_J_per_m": event_audit.energy_release_J_per_m,
            "hazard_dissipation_J_per_m": event_audit.hazard_dissipation_J_per_m,
            "energy_margin_J_per_m": event_audit.energy_margin_J_per_m,
            "terminal_fingerprint": complete_accepted_state_fingerprint(connected),
            "terminal_active_tip_ids": connected.crack_network.active_tip_ids,
            "terminal_cavity_phase": connected.void_state.cavities[0].phase.value,
            "terminal_equilibrium": connected.equilibrium_observables.to_dict(),
        },
        "post_event_checkpoint": None if checkpoint_manifest is None else {
            "manifest": str(checkpoint_path.resolve()),
            "payload": checkpoint_manifest,
        },
        "downstream_source_audit": source_audit,
        "collective_complete_ligament_role": "RETAINED_DIAGNOSTIC_EXPLICIT_ALTERNATIVE",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / REGISTRY)
    parser.add_argument("--parameter-option", default=DEFAULT_OPTION)
    parser.add_argument("--temperature-K", type=float, default=1000.0)
    parser.add_argument("--total-opening-m", type=float, default=8.0e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    checkpoint = args.output.parent / "local_incremental_connected_checkpoint.json"
    result = _finite(qualify(
        registry=args.registry, option=args.parameter_option,
        temperature_K=args.temperature_K, total_opening_m=args.total_opening_m,
        checkpoint_path=checkpoint,
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
