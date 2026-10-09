#!/usr/bin/env python3
"""Run the single retained central DBTT state at the three frozen V4 levels."""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cavity_source_conforming_geometry_v4 import (
    CavitySourceGeometryV4Unavailable,
    ETA_N_MAX,
    ETA_T_MAX,
    GEOMETRY_ID,
    MINIMUM_MESH_QUALITY,
    TENSOR_RELATIVE_TOLERANCE,
    TRACTION_RESIDUAL_TOLERANCE,
)
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.unified_fracture_material_v5 import (
    identity_record, material_bundle, require_bound_identity,
)
from arrhenius_fracture.voiding_production_v5 import (
    _cavity_resolution_binding, build_production_void_state,
    cavity_source_resolution_metrics, deterministic_trajectory,
    equilibrate_fixed_load_with_production_fem, ligament_transaction,
)
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID


OUTPUT = ROOT / "artifacts/v5_cavity_source_recovery_v4/central_dbtt_v4_readiness.json"
LEVELS = ((32, 12), (64, 24), (128, 48))


def _failure_classes(row):
    mapping = (
        ("tensor_convergence", "SOURCE_TENSOR_CONVERGENCE"),
        ("cavity_traction", "CAVITY_TRACTION_RESIDUAL"),
        ("normal_direction_resolution", "NORMAL_DIRECTION_RESOLUTION"),
        ("tangential_direction_resolution", "TANGENTIAL_DIRECTION_RESOLUTION"),
        ("minimum_mesh_quality", "MESH_QUALITY"),
        ("patch_conditioning", "PATCH_CONDITIONING"),
        ("physical_window_identity", "PHYSICAL_WINDOW_IDENTITY"),
        ("source_geometry", "SOURCE_GEOMETRY_IDENTITY"),
    )
    return [classification for key, classification in mapping if not row["predicates"][key]]


def build_record():
    bundle = material_bundle("DBTT")
    retained_initial, _ = build_production_void_state(bundle=bundle, enabled=True)
    retained_path = tuple(retained_initial.crack_network.branch(ROOT_BRANCH_ID).path)
    reference_competition = retained_initial.competition
    reference_rng = retained_initial.rng_state
    rows = []
    previous_tensor = None
    first_window = None
    accepted_reference = None
    for sectors, radial_layers in LEVELS:
        preconnection, _ = deterministic_trajectory(
            bundle=bundle, stop_before_ligament=True, crack_path_m=retained_path,
            boundary_segments=sectors, radial_layers=radial_layers,
            source_conforming_geometry=True,
        )
        loaded = equilibrate_fixed_load_with_production_fem(
            replace(preconnection, displacement=preconnection.displacement * 2.0)
        )
        connected, ligament = ligament_transaction(loaded)
        before = complete_accepted_state_fingerprint(connected)
        binding = _cavity_resolution_binding(connected)
        connected_identity = dict(require_bound_identity(connected))
        connected_cavity = connected.void_state.cavities[0]
        if accepted_reference is None:
            accepted_reference = {
                "material_identity": connected_identity,
                "competition": connected.competition,
                "rng_state": connected.rng_state,
                "cavity_geometry": (
                    connected_cavity.center_m, connected_cavity.radius_m,
                    connected_cavity.connection_entry_m, connected_cavity.connection_exit_m,
                    connected_cavity.connection_direction_xy,
                ),
                "cavity": connected_cavity,
            }
        material_matches = connected_identity == accepted_reference["material_identity"]
        geometry_matches = (
            connected_cavity.center_m, connected_cavity.radius_m,
            connected_cavity.connection_entry_m, connected_cavity.connection_exit_m,
            connected_cavity.connection_direction_xy,
        ) == accepted_reference["cavity_geometry"]
        clocks_match = (
            connected.competition == accepted_reference["competition"]
            and connected.rng_state == accepted_reference["rng_state"]
        )
        try:
            metrics = cavity_source_resolution_metrics(
                connected, geometry_contract=GEOMETRY_ID
            )
            unavailable = None
        except CavitySourceGeometryV4Unavailable as exc:
            metrics = None
            unavailable = str(exc)
        after = complete_accepted_state_fingerprint(connected)
        if metrics is None:
            rows.append({
                "N_theta": sectors, "radial_layers": radial_layers,
                "scientific_unavailability": unavailable,
                "accepted_input_state_fingerprint": before,
                "accepted_input_state_unchanged": before == after,
                "mechanical_binding": binding,
                "material_identity_matches_reference": material_matches,
                "physical_geometry_matches_reference": geometry_matches,
                "thresholds_and_rng_match_reference": clocks_match,
                "predicates": {
                    "tensor_convergence": False, "cavity_traction": False,
                    "normal_direction_resolution": False,
                    "tangential_direction_resolution": False,
                    "minimum_mesh_quality": False, "patch_conditioning": False,
                    "physical_window_identity": False, "source_geometry": False,
                },
            })
            break
        tensor = np.asarray(metrics["tensor_Pa"], dtype=float)
        change = (None if previous_tensor is None else float(
            np.linalg.norm(tensor - previous_tensor) / max(np.linalg.norm(tensor), 1.0e-300)
        ))
        recovery = metrics["recovery_record"]
        window = recovery["physical_arc_identity"]["sha256"]
        if first_window is None:
            first_window = window
        cavity = connected.void_state.cavities[0]
        source_error = abs(math.dist(cavity.connection_exit_m, cavity.center_m) - cavity.radius_m)
        angular_error = cavity.radius_m / math.cos(math.pi / sectors) - cavity.radius_m
        predicates = {
            "tensor_convergence": change is None or change <= TENSOR_RELATIVE_TOLERANCE,
            "cavity_traction": metrics["normalized_traction"] <= TRACTION_RESIDUAL_TOLERANCE,
            "normal_direction_resolution": metrics["eta_n_max"] <= ETA_N_MAX,
            "tangential_direction_resolution": metrics["eta_t_max"] <= ETA_T_MAX,
            "minimum_mesh_quality": metrics["minimum_quality"] >= MINIMUM_MESH_QUALITY,
            "patch_conditioning": recovery["condition"]["maximum"] <= recovery["condition"]["limit"],
            "physical_window_identity": window == first_window,
            "source_geometry": (
                source_error <= max(1.0e-12, 1.0e-8 * cavity.radius_m)
                and recovery["source_node_certificate"]["degree"] == 2
                and recovery["source_node_certificate"]["unique_tangent_and_outward_normal"]
            ),
        }
        rows.append({
            "N_theta": sectors,
            "radial_layers": radial_layers,
            "angular_geometry_error_m": angular_error,
            "radial_mesh_error_eta_n": float(metrics["eta_n_max"]),
            "source_tensor_Pa": metrics["tensor_Pa"],
            "tensor_relative_change_from_previous_level": change,
            "normalized_cavity_traction": float(metrics["normalized_traction"]),
            "eta_n": float(metrics["eta_n_max"]),
            "eta_t": float(metrics["eta_t_max"]),
            "minimum_quality": float(metrics["minimum_quality"]),
            "patch_condition": recovery["condition"],
            "physical_window_identity": window,
            "source_node_certificate": recovery["source_node_certificate"],
            "accepted_input_state_fingerprint": before,
            "accepted_input_state_unchanged": before == after,
            "mechanical_binding": binding,
            "material_identity_matches_reference": material_matches,
            "physical_geometry_matches_reference": geometry_matches,
            "thresholds_and_rng_match_reference": clocks_match,
            "ligament_energy_gate": {
                "accepted": bool(ligament.accepted),
                "energy_release_J_per_m": float(ligament.energy_release_J_per_m),
                "hazard_dissipation_J_per_m": float(ligament.hazard_dissipation_J_per_m),
                "energy_margin_J_per_m": float(ligament.energy_margin_J_per_m),
            },
            "predicates": predicates,
        })
        previous_tensor = tensor

    complete_levels = len(rows) == len(LEVELS) and all(
        row.get("scientific_unavailability") is None for row in rows
    )
    final_failures = _failure_classes(rows[-1]) if rows else ["NO_LEVEL_RESULT"]
    # A tensor convergence gate is meaningful only after a successive-level comparison.
    if complete_levels and rows[-1]["tensor_relative_change_from_previous_level"] is None:
        final_failures.append("SOURCE_TENSOR_CONVERGENCE")
    passed = complete_levels and not final_failures and all(
        row["accepted_input_state_unchanged"] for row in rows
    )
    status = (
        "PASS_SOURCE_CONFORMING_CAVITY_GEOMETRY_V4"
        if passed else "BLOCKED_WITH_EXACT_V4_FAILURE_CLASS"
    )
    central = rows[0] if rows else {}
    cavity_ref = accepted_reference["cavity"] if accepted_reference else None
    return {
        "schema": "v5.central-dbtt-source-conforming-geometry-v4-readiness/1",
        "geometry_contract": GEOMETRY_ID,
        "material_class": "DBTT",
        "fracture_material_row_id": bundle.fracture_material_row_id,
        "material_identity": dict(identity_record(bundle, retained_initial.material)),
        "retained_physical_crack_path_m": [list(map(float, point)) for point in retained_path],
        "opening_scale_from_4e-7_m": 2.0,
        "geometry": None if cavity_ref is None else {
            "cavity_center_m": list(cavity_ref.center_m),
            "nominal_cavity_radius_m": float(cavity_ref.radius_m),
            "connection_entry_m": list(cavity_ref.connection_entry_m),
            "connection_exit_m": list(cavity_ref.connection_exit_m),
        },
        "levels_requested": [{"N_theta": n, "radial_layers": r} for n, r in LEVELS],
        "levels_run": len(rows),
        "level_records": rows,
        "exact_v4_failure_class": final_failures,
        "DBTT_SOURCE_READINESS": status,
        "accepted_material_identity_exact_across_levels": bool(rows) and all(
            row["material_identity_matches_reference"] for row in rows
        ),
        "physical_geometry_exact_across_levels": bool(rows) and all(
            row["physical_geometry_matches_reference"] for row in rows
        ),
        "thresholds_and_rng_exact_across_levels": bool(rows) and all(
            row["thresholds_and_rng_match_reference"] for row in rows
        ),
        "retained_initial_thresholds_and_rng_captured_unchanged_before_run": (
            retained_initial.competition == reference_competition
            and retained_initial.rng_state == reference_rng
        ),
        "accepted_input_state_unchanged_on_noncertification": all(
            row["accepted_input_state_unchanged"] for row in rows
        ),
        "unexpected_programming_exceptions_caught": False,
        "oracle_states_accepted": 0,
        "oracle_generated_in_this_record": False,
        "paired_trajectories_run": 0,
        "fatigue_started": False,
        "missing_fields_inferred_or_synthesized": False,
        "preserved_v1_v2_v3": {
            "CAVITY_SOURCE_RECOVERY_V1_INCIDENT_CST_MAX_PRINCIPAL": "FAIL_NONCONVERGENT",
            "CAVITY_FIXED_ARC_PATCH_RECOVERY_V2_MANUFACTURED_AND_KIRSCH": "PASS",
            "V3_OPERATOR_MANUFACTURED_AND_KIRSCH": "PASS",
            "V3_CENTRAL_DBTT_GEOMETRY_REGISTRATION": "FAIL_POLYGON_VERTEX_VS_NOMINAL_CIRCLE",
            "V3_CENTRAL_DBTT_TENSOR_CONVERGENCE": "NOT_RUN",
        },
        "next_bounded_step": (
            "GENERATE_18_FIXED_STATE_ORACLE_ROWS_IN_SEPARATE_COMMIT"
            if passed else "STOP_POINT_SOURCE_DEVELOPMENT_AND_SEPARATELY_FORMULATE_ACTIVATION_ZONE_OBSERVABLE"
        ),
        "scope": {
            "scientific_tolerances_changed": False,
            "r_tip_law_changed": False,
            "r_tip_equals_R_void": False,
            "finite_activation_zone_observable_derived": False,
            "paired_trajectories_run": False,
            "fatigue_started": False,
        },
    }


def main():
    payload = build_record()
    destination = Path(os.environ.get("V4_READINESS_OUTPUT", str(OUTPUT)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("DBTT_SOURCE_READINESS=" + payload["DBTT_SOURCE_READINESS"])
    print("V4_FAILURE_CLASS=" + "+".join(payload["exact_v4_failure_class"]))
    print("LEVELS_RUN=" + str(payload["levels_run"]) + "/3")
    print("ORACLE_STATES_ACCEPTED=" + str(payload["oracle_states_accepted"]) + "/18")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
