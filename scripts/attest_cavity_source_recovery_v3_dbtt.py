#!/usr/bin/env python3
"""Run and record exactly one central DBTT V3 source-readiness case."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.closure_static_evidence import RESOLUTION_SCREEN, resolution_screen
from arrhenius_fracture.finalization_v3_schema import SCIENTIFIC_ACCEPTANCE_TOLERANCES
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.unified_fracture_material_v5 import (
    identity_record,
    material_bundle,
    require_bound_identity,
)
from arrhenius_fracture.voiding_production_v5 import (
    _cavity_resolution_binding,
    deterministic_trajectory,
    equilibrate_fixed_load_with_production_fem,
    ligament_transaction,
    refine_downstream_source,
)


OUTPUT = Path("artifacts/v5_cavity_source_recovery_v3/central_dbtt_v3_readiness.json")
PRIOR_V2_LEVEL_BINDINGS = (
    "732349ea489fcab48a717a04739896772394849ab98a5f12558289e277e32cf1",
    "26f2539f263d09f598fe5807316e3904fca680b35e48c980c6a7af7585eb2023",
    "258be304822f14e78dc734dcefd44aa8cf0ee7f5087633913d486abf348da94f",
)


def build_record() -> dict[str, object]:
    bundle = material_bundle("DBTT")
    preconnection, _ = deterministic_trajectory(bundle=bundle, stop_before_ligament=True)
    loaded = equilibrate_fixed_load_with_production_fem(
        replace(preconnection, displacement=preconnection.displacement * 2.0)
    )
    connected, ligament = ligament_transaction(loaded)
    accepted_fingerprint = complete_accepted_state_fingerprint(connected)
    accepted_binding = _cavity_resolution_binding(connected)
    accepted_clocks = (connected.competition, connected.rng_state)
    qualified, audit = refine_downstream_source(
        connected,
        max_refinement_levels=3,
        refinement_region="complete_cavity_ring",
        quality_improvement="constrained_v1",
    )
    if audit["status"] not in ("SOURCE_TENSOR_QUALIFIED", "SOURCE_TENSOR_UNQUALIFIED"):
        raise RuntimeError("central DBTT V3 returned an unregistered source status")

    attempts = []
    for item in audit.get("attempts", []):
        proof = item["proof"]
        previous = proof["previous_metrics"]
        current = proof["current_metrics"]
        before = np.asarray(previous["tensor_Pa"], dtype=float)
        after = np.asarray(current["tensor_Pa"], dtype=float)
        change = float(
            np.linalg.norm(after - before) / max(np.linalg.norm(after), 1.0e-300)
        )
        recovery = current["recovery_record"]
        predicates = {
            "fixed_arc_tensor_convergence": (
                change <= SCIENTIFIC_ACCEPTANCE_TOLERANCES["tensor_probe_relative"]
            ),
            "cavity_traction": (
                current["normalized_traction"]
                <= SCIENTIFIC_ACCEPTANCE_TOLERANCES["cavity_traction_normalized"]
            ),
            "normal_direction_resolution": (
                current["eta_n_max"] <= RESOLUTION_SCREEN["eta_n_max"]
            ),
            "tangential_direction_resolution": (
                current["eta_t_max"] <= RESOLUTION_SCREEN["eta_t_max"]
            ),
            "local_aspect_ratio": (
                current["local_aspect_ratio_max"]
                <= RESOLUTION_SCREEN["local_aspect_ratio_max"]
            ),
            "minimum_mesh_quality": (
                current["minimum_quality"] >= RESOLUTION_SCREEN["minimum_quality"]
            ),
            "patch_conditioning": (
                recovery["condition"]["maximum"] <= recovery["condition"]["limit"]
            ),
            "fixed_physical_window_identity": (
                recovery["physical_arc_identity"]["sha256"]
                == previous["recovery_record"]["physical_arc_identity"]["sha256"]
            ),
            "complete_resolution_screen": resolution_screen(current),
        }
        attempts.append(
            {
                "refinement_level": int(item["level"]),
                "qualified": bool(item["qualified"]),
                "state_binding": proof["current_binding"],
                "matches_retained_v2_mechanical_state_binding": (
                    proof["current_binding"]
                    == PRIOR_V2_LEVEL_BINDINGS[int(item["level"]) - 1]
                ),
                "fixed_arc_tensor_relative_change": change,
                "normalized_cavity_traction": float(current["normalized_traction"]),
                "eta_n_max": float(current["eta_n_max"]),
                "eta_t_max": float(current["eta_t_max"]),
                "local_aspect_ratio_max": float(current["local_aspect_ratio_max"]),
                "minimum_mesh_quality": float(current["minimum_quality"]),
                "tensor_Pa": current["tensor_Pa"],
                "sigma_tt_Pa": float(current["sigma_tt_Pa"]),
                "recovery": recovery,
                "source_transfer_budget": proof.get("source_transfer_budget"),
                "predicates": predicates,
            }
        )

    failure_classes = []
    if attempts:
        final = attempts[-1]["predicates"]
        mapping = (
            ("fixed_arc_tensor_convergence", "TANGENTIAL_STRESS_CONVERGENCE"),
            ("cavity_traction", "CAVITY_TRACTION_RESIDUAL"),
            ("normal_direction_resolution", "NORMAL_DIRECTION_RESOLUTION"),
            ("tangential_direction_resolution", "TANGENTIAL_DIRECTION_RESOLUTION"),
            ("local_aspect_ratio", "LOCAL_ASPECT_RATIO"),
            ("minimum_mesh_quality", "MESH_QUALITY"),
            ("patch_conditioning", "PATCH_CONDITIONING"),
            ("fixed_physical_window_identity", "PHYSICAL_WINDOW_IDENTITY"),
        )
        failure_classes = [classification for key, classification in mapping if not final[key]]
    elif audit.get("scientific_unavailability"):
        unavailable = str(audit["scientific_unavailability"])
        if "source coordinate" in unavailable or "owned cavity boundary" in unavailable:
            failure_classes = ["SOURCE_GEOMETRY_IDENTITY"]
        elif "poor-quality support" in unavailable:
            failure_classes = ["MESH_QUALITY"]
        elif "support" in unavailable:
            failure_classes = ["INSUFFICIENT_SAMPLING"]
        elif "rank deficient" in unavailable:
            failure_classes = ["PATCH_RANK_DEFICIENCY"]
        elif "conditioned" in unavailable:
            failure_classes = ["PATCH_CONDITIONING"]
        elif "ownership" in unavailable:
            failure_classes = ["INCOMPLETE_STATE_OWNERSHIP"]
        else:
            failure_classes = ["V3_SCIENTIFIC_UNAVAILABILITY"]
    status = (
        "PASS_CAVITY_FIXED_PHYSICAL_ARC_PATCH_RECOVERY_V3"
        if audit["status"] == "SOURCE_TENSOR_QUALIFIED"
        else "BLOCKED_WITH_EXACT_V3_FAILURE_CLASS"
    )
    cavity = connected.void_state.cavities[0]
    return {
        "schema": "v5.central-dbtt-cavity-source-recovery-v3-readiness/1",
        "operator": "CAVITY_FIXED_PHYSICAL_ARC_PATCH_RECOVERY_V3",
        "source_implementation_commit": "80bbcabd7e910e8da09d6765116b2a97b06f2300",
        "source_exception_boundary_commit": "bb046019f3c5c518f031b98a677993b70cf81f6b",
        "material_class": "DBTT",
        "fracture_material_row_id": bundle.fracture_material_row_id,
        "material_identity": dict(identity_record(bundle, connected.material)),
        "accepted_material_identity_complete": dict(require_bound_identity(connected)),
        "geometry": {
            "cavity_center_m": list(cavity.center_m),
            "cavity_radius_m": float(cavity.radius_m),
            "connection_entry_m": list(cavity.connection_entry_m),
            "connection_exit_m": list(cavity.connection_exit_m),
            "opening_scale_from_4e-7_m": 2.0,
        },
        "source_geometry_diagnostic": {
            "owned_source_radial_distance_m": float(
                np.linalg.norm(
                    np.asarray(cavity.connection_exit_m) - np.asarray(cavity.center_m)
                )
            ),
            "nominal_cavity_radius_m": float(cavity.radius_m),
            "absolute_radial_mismatch_m": abs(
                float(
                    np.linalg.norm(
                        np.asarray(cavity.connection_exit_m)
                        - np.asarray(cavity.center_m)
                    )
                )
                - float(cavity.radius_m)
            ),
            "classification": "OWNED_POLYGON_BOUNDARY_DIFFERS_FROM_NOMINAL_CIRCLE",
        },
        "ligament_energy_gate": {
            "accepted": bool(ligament.accepted),
            "energy_release_J_per_m": float(ligament.energy_release_J_per_m),
            "hazard_dissipation_J_per_m": float(ligament.hazard_dissipation_J_per_m),
            "energy_margin_J_per_m": float(ligament.energy_margin_J_per_m),
        },
        "accepted_pre_source_state_fingerprint": accepted_fingerprint,
        "accepted_pre_source_mechanical_binding": accepted_binding,
        "requested_refinement_levels": 3,
        "refinement_levels_run": len(attempts),
        "attempts": attempts,
        "scientific_unavailability": audit.get("scientific_unavailability"),
        "source_qualification_status": audit["status"],
        "DBTT_SOURCE_READINESS": status,
        "exact_v3_failure_class": failure_classes,
        "accepted_pre_source_state_unchanged_on_noncertification": (
            qualified is connected
            and complete_accepted_state_fingerprint(connected) == accepted_fingerprint
            and _cavity_resolution_binding(connected) == accepted_binding
        ),
        "thresholds_and_rng_unchanged_on_noncertification": (
            (connected.competition, connected.rng_state) == accepted_clocks
        ),
        "unexpected_programming_exceptions_caught": False,
        "oracle_states_accepted": 0 if status.startswith("BLOCKED") else None,
        "paired_trajectories_run": 0,
        "fatigue_started": False,
        "missing_fields_inferred_or_synthesized": False,
        "next_bounded_step": (
            "DERIVE_FINITE_ACTIVATION_ZONE_WORK_OBSERVABLE"
            if status.startswith("BLOCKED")
            else "GENERATE_18_FIXED_STATE_ORACLE_ROWS_IN_SEPARATE_COMMIT"
        ),
        "preserved_v1_v2": {
            "CAVITY_SOURCE_RECOVERY_V1_INCIDENT_CST_MAX_PRINCIPAL": "FAIL_NONCONVERGENT",
            "CAVITY_FIXED_ARC_PATCH_RECOVERY_V2_MANUFACTURED_AND_KIRSCH": "PASS",
            "DBTT_V2_CAVITY_TRACTION": "PASS",
            "DBTT_V2_FIXED_ARC_TENSOR_CONVERGENCE": "FAIL",
        },
        "scope": {
            "core_physics_changed": False,
            "scientific_tolerances_changed": False,
            "r_tip_law_changed": False,
            "r_tip_equals_R_void": False,
            "new_broad_campaign_run": False,
        },
    }


def main() -> int:
    payload = build_record()
    destination = ROOT / OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["DBTT_SOURCE_READINESS"])
    print("FAILURE_CLASS=" + "+".join(payload["exact_v3_failure_class"]))
    print("ORACLE_STATES_ACCEPTED=" + str(payload["oracle_states_accepted"]) + "/18")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
