#!/usr/bin/env python3
"""Run only the frozen D/E central DBTT V6 traction-closure family."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cavity_source_shape_regular_mesh_v5 import (
    ACCEPTANCE_MINIMUM_QUALITY, MESH_CONTRACT_ID, V6_LOCAL_LEVELS,
)
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.finalization_v3_schema import SCIENTIFIC_ACCEPTANCE_TOLERANCES
from arrhenius_fracture.unified_fracture_material_v5 import (
    identity_record, material_bundle, require_bound_identity,
)
from arrhenius_fracture.voiding_production_v5 import build_production_void_state
from scripts.attest_cavity_source_shape_regular_v5_dbtt import (
    _relative, _run_level, _threshold_identity,
)
from scripts.build_cavity_source_raw_traction_closure_v6_contract import V6_CONTRACT_ID

OUTPUT = ROOT / "artifacts/v6_cavity_source_raw_traction_closure/central_dbtt_v6_readiness.json"
V5_OUTPUT = ROOT / "artifacts/v5_cavity_source_recovery_v5/central_dbtt_v5_readiness.json"
CONTRACT_OUTPUT = ROOT / "artifacts/v6_cavity_source_raw_traction_closure/v6_contract.json"
LOCAL_LEVELS_RUN = ("D", "E")
FIXED_POLYGON_SECTORS = 128
TENSOR_LIMIT = 0.05
TRACTION_LIMIT = 0.05
ETA_N_LIMIT = 0.03
ETA_T_LIMIT = 0.025
WEAK_RESIDUAL_LIMIT = SCIENTIFIC_ACCEPTANCE_TOLERANCES["free_residual_relative"]
REACTION_LIMIT = SCIENTIFIC_ACCEPTANCE_TOLERANCES["static_mesh_reaction_relative"]
ENERGY_LIMIT = SCIENTIFIC_ACCEPTANCE_TOLERANCES["static_mesh_energy_relative"]


def _physical_observable_predicates(row):
    obs = row["observables"]
    finite = all(np.isfinite(obs[key]) for key in (
        "top_reaction_N_per_m", "bottom_reaction_N_per_m", "reaction_N_per_m",
        "applied_opening_m", "compliance_m2_per_N", "external_work_J_per_m",
        "stored_recoverable_energy_J_per_m", "energy_identity_reference_J_per_m",
        "energy_J_per_m", "full_residual_including_reactions_N_per_m",
        "free_dof_residual_l2_N_per_m", "constrained_reaction_l2_N_per_m",
        "top_bottom_reaction_balance", "energy_reaction_identity",
    ))
    return {
        "finite": bool(finite),
        "nonzero_top_reaction": abs(obs["top_reaction_N_per_m"]) > 1.0e-12,
        "nonzero_bottom_reaction": abs(obs["bottom_reaction_N_per_m"]) > 1.0e-12,
        "finite_positive_compliance": bool(
            np.isfinite(obs["compliance_m2_per_N"]) and obs["compliance_m2_per_N"] > 0.0
        ),
        "finite_positive_energy": bool(
            np.isfinite(obs["stored_recoverable_energy_J_per_m"])
            and obs["stored_recoverable_energy_J_per_m"] > 0.0
        ),
        "reaction_balance": (
            obs["top_bottom_reaction_balance"]
            <= SCIENTIFIC_ACCEPTANCE_TOLERANCES["reaction_balance_relative"]
        ),
        "energy_identity": (
            obs["energy_reaction_identity"]
            <= SCIENTIFIC_ACCEPTANCE_TOLERANCES["energy_reaction_identity_relative"]
        ),
    }


def _failure_classes(predicates):
    names = {
        "raw_traction_E": "RAW_ADJACENT_ELEMENT_TRACTION_E",
        "strict_raw_traction_order_C_D_E": "RAW_ADJACENT_ELEMENT_TRACTION_NONMONOTONIC",
        "source_tensor_D_to_E": "SOURCE_TENSOR_D_TO_E_CONVERGENCE",
        "normal_direction_resolution": "NORMAL_DIRECTION_RESOLUTION",
        "tangential_direction_resolution": "TANGENTIAL_DIRECTION_RESOLUTION",
        "minimum_mesh_quality": "MESH_QUALITY",
        "weak_cavity_boundary_residual": "WEAK_CAVITY_BOUNDARY_RESIDUAL",
        "patch_rank_and_condition": "PATCH_RANK_OR_CONDITIONING",
        "fixed_geometry_identity": "FIXED_GEOMETRY_IDENTITY",
        "accepted_input_state_unchanged": "ACCEPTED_INPUT_STATE_MUTATION",
        "exact_state_identity": "STATE_IDENTITY",
        "ligament_event_accepted": "LIGAMENT_EVENT_ACCEPTANCE",
        "physical_equilibrium_observables": "EQUILIBRIUM_OBSERVABLES",
        "reaction_convergence": "REACTION_CONVERGENCE",
        "compliance_convergence": "COMPLIANCE_CONVERGENCE",
        "energy_convergence": "ENERGY_CONVERGENCE",
        "ligament_energy_release_convergence": "LIGAMENT_ENERGY_RELEASE_CONVERGENCE",
    }
    return [names[key] for key, passed in predicates.items() if not passed]


def build_record():
    contract = json.loads(CONTRACT_OUTPUT.read_text())
    if contract["contract"] != V6_CONTRACT_ID:
        raise RuntimeError("the prospective V6 contract is unavailable")
    v5 = json.loads(V5_OUTPUT.read_text())
    retained_rows = v5["fixed_geometry_local_family"]["rows"]
    retained = {row["local_level"]: row for row in retained_rows}
    if [retained[key]["raw_adjacent_element_traction_normalized"] for key in ("A", "B", "C")] != [
        0.14906976345973477, 0.09144336014017264, 0.06715193304848284,
    ]:
        raise RuntimeError("retained V5 A/B/C evidence changed")

    bundle = material_bundle("DBTT")
    retained_initial, _ = build_production_void_state(bundle=bundle, enabled=True)
    retained_path = tuple(retained_initial.crack_network.branch(ROOT_BRANCH_ID).path)
    reference = {
        "material_identity": dict(require_bound_identity(retained_initial)),
        "threshold_identity": None,
        "rng_state": retained_initial.rng_state,
        "source_coordinate_m": [7.55e-4, 0.0],
        "nominal_cavity_geometry": ((7.0e-4, 0.0), 5.5e-5),
    }
    rows = [
        _run_level(
            bundle, retained_path, sectors=FIXED_POLYGON_SECTORS,
            local_level=level, reference=reference,
        )
        for level in LOCAL_LEVELS_RUN
    ]
    connected_thresholds = rows[0]["threshold_identity"]
    for row in rows:
        row["thresholds_match_reference"] = row["threshold_identity"] == connected_thresholds
        row["physical_equilibrium_observable_predicates"] = _physical_observable_predicates(row)

    d, e = rows
    d_equilibrated = d["equilibrated_boundary_traction_recovery_v1"].get(
        "normalized_boundary_traction"
    )
    e_equilibrated = e["equilibrated_boundary_traction_recovery_v1"].get(
        "normalized_boundary_traction"
    )
    comparisons = {
        "source_tensor_relative_D_to_E": _relative(
            d["constrained_boundary_limit_tensor_global_xy_Pa"],
            e["constrained_boundary_limit_tensor_global_xy_Pa"],
        ),
        "reaction_relative_D_to_E": _relative(
            d["observables"]["reaction_N_per_m"], e["observables"]["reaction_N_per_m"],
        ),
        "compliance_relative_D_to_E": _relative(
            d["observables"]["compliance_m2_per_N"], e["observables"]["compliance_m2_per_N"],
        ),
        "potential_energy_relative_D_to_E": _relative(
            d["observables"]["energy_J_per_m"], e["observables"]["energy_J_per_m"],
        ),
        "ligament_energy_release_relative_D_to_E": _relative(
            d["ligament_energy_gate"]["energy_release_J_per_m"],
            e["ligament_energy_gate"]["energy_release_J_per_m"],
        ),
        "equilibrated_traction_absolute_D_to_E": (
            abs(float(d_equilibrated) - float(e_equilibrated))
            if d_equilibrated is not None and e_equilibrated is not None else None
        ),
    }
    c_raw = retained["C"]["raw_adjacent_element_traction_normalized"]
    d_raw = d["raw_adjacent_element_traction_normalized"]
    e_raw = e["raw_adjacent_element_traction_normalized"]
    fixed_geometry = (
        len({
            retained["C"]["discrete_cavity_boundary_fingerprints"]["complete_boundary"],
            *(row["discrete_cavity_boundary_fingerprints"]["complete_boundary"] for row in rows),
        }) == 1
        and len({
            retained["C"]["discrete_cavity_boundary_fingerprints"]["fixed_v3_source_window"],
            *(row["discrete_cavity_boundary_fingerprints"]["fixed_v3_source_window"] for row in rows),
        }) == 1
        and len({retained["C"]["physical_window_identity"], *(row["physical_window_identity"] for row in rows)}) == 1
    )
    exact_state_keys = (
        "material_identity_matches_reference", "thresholds_match_reference",
        "rng_state_matches_reference", "load_history_matches_reference",
        "source_coordinate_matches_reference", "nominal_cavity_geometry_matches_reference",
    )
    predicates = {
        "raw_traction_E": e_raw <= TRACTION_LIMIT,
        "strict_raw_traction_order_C_D_E": c_raw > d_raw > e_raw,
        "source_tensor_D_to_E": comparisons["source_tensor_relative_D_to_E"] <= TENSOR_LIMIT,
        "normal_direction_resolution": max(row["eta_n"] for row in rows) <= ETA_N_LIMIT,
        "tangential_direction_resolution": max(row["eta_t"] for row in rows) <= ETA_T_LIMIT,
        "minimum_mesh_quality": min(row["global_minimum_quality"] for row in rows) >= ACCEPTANCE_MINIMUM_QUALITY,
        "weak_cavity_boundary_residual": max(
            row["assembled_weak_cavity_boundary_residual_normalized"] for row in rows
        ) <= WEAK_RESIDUAL_LIMIT,
        "patch_rank_and_condition": all(
            row["patch_rank"] == {"sigma_nn": 3, "sigma_nt": 3, "sigma_tt": 6}
            and row["patch_condition"]["maximum"] <= row["patch_condition"]["limit"]
            for row in rows
        ),
        "fixed_geometry_identity": fixed_geometry,
        "accepted_input_state_unchanged": all(row["accepted_input_state_unchanged"] for row in rows),
        "exact_state_identity": all(row[key] for row in rows for key in exact_state_keys),
        "ligament_event_accepted": all(row["ligament_energy_gate"]["accepted"] for row in rows),
        "physical_equilibrium_observables": all(
            all(row["physical_equilibrium_observable_predicates"].values()) for row in rows
        ),
        "reaction_convergence": comparisons["reaction_relative_D_to_E"] <= REACTION_LIMIT,
        "compliance_convergence": comparisons["compliance_relative_D_to_E"] <= REACTION_LIMIT,
        "energy_convergence": comparisons["potential_energy_relative_D_to_E"] <= ENERGY_LIMIT,
        "ligament_energy_release_convergence": (
            comparisons["ligament_energy_release_relative_D_to_E"] <= ENERGY_LIMIT
        ),
    }
    fallback_rows = [row["equilibrated_boundary_traction_recovery_v1"] for row in rows]
    fallback_predicates = {
        "available": all(row.get("available") is True for row in fallback_rows),
        "normalized_recovered_boundary_traction": all(
            row.get("normalized_boundary_traction", float("inf")) <= TRACTION_LIMIT
            for row in fallback_rows
        ),
        "mesh_convergence": (
            comparisons["equilibrated_traction_absolute_D_to_E"] is not None
            and comparisons["equilibrated_traction_absolute_D_to_E"] <= 0.05
        ),
        "fit_residual": all(
            row.get("fit_residual", float("inf")) <= row.get("fit_residual_limit", -float("inf"))
            for row in fallback_rows
        ),
        "rank_and_condition": all(
            row.get("rank", -1) >= row.get("required_rank", 10**9)
            and row.get("condition", float("inf")) <= row.get("condition_limit", -float("inf"))
            for row in fallback_rows
        ),
        "weak_fem_residual": predicates["weak_cavity_boundary_residual"],
        "physical_reactions_balance": predicates["physical_equilibrium_observables"],
    }
    primary_passed = all(predicates.values())
    fallback_passed = all(fallback_predicates.values())
    shared_failures = _failure_classes({
        key: value for key, value in predicates.items()
        if key not in {"raw_traction_E", "strict_raw_traction_order_C_D_E"}
    })
    failures = shared_failures + ([] if primary_passed or fallback_passed else [
        "TRACTION_FREE_BOUNDARY_VERIFICATION",
    ])
    passed = not failures
    return {
        "schema": "v6.central-dbtt-raw-traction-closure-readiness/1",
        "executed_code_sha": os.environ.get("V6_SOURCE_COMMIT") or subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip(),
        "contract": V6_CONTRACT_ID,
        "retained_v5_mesh_contract": MESH_CONTRACT_ID,
        "material_class": "DBTT",
        "fracture_material_row_id": bundle.fracture_material_row_id,
        "material_identity": dict(identity_record(bundle, retained_initial.material)),
        "retained_physical_crack_path_m": [list(map(float, point)) for point in retained_path],
        "retained_initial_threshold_identity": _threshold_identity(retained_initial),
        "fixed_geometry_local_family": {
            "polygon_sectors": FIXED_POLYGON_SECTORS,
            "retained_A_B_C_raw_traction": [
                retained[key]["raw_adjacent_element_traction_normalized"] for key in ("A", "B", "C")
            ],
            "new_rows": rows,
            "D_to_E_comparisons": comparisons,
            "predicates": predicates,
            "fallback_predicates": fallback_predicates,
        },
        "RAW_CST_TRACTION": "PASS" if primary_passed else "DIAGNOSTIC_FAIL",
        "TRACTION_FREE_BOUNDARY_VERIFICATION": (
            "PASS_RAW_CST" if primary_passed else
            "PASS_EQUILIBRATED_RECOVERY" if fallback_passed else "FAIL"
        ),
        "exact_v6_failure_class": failures,
        "DBTT_SOURCE_READINESS": (
            "PASS_PRODUCTION_TRACTION_CLOSURE" if passed
            else "BLOCKED_WITH_EXACT_V6_FAILURE_CLASS"
        ),
        "oracle_states_accepted": 0,
        "oracle_generated_in_this_record": False,
        "paired_trajectories_run": 0,
        "fatigue_started": False,
        "unexpected_programming_exceptions_caught": False,
        "preserved_v5": {
            "V5_SOURCE_CONFORMING_GEOMETRY": "PASS",
            "V5_SHAPE_REGULAR_LOCAL_PATCH": "PASS",
            "V5_SOURCE_TENSOR_CONVERGENCE": "PASS",
            "V5_WEAK_TRACTION_FREE_BOUNDARY": "PASS",
            "V5_RAW_ADJACENT_ELEMENT_TRACTION": "FAIL_0.0671519_GT_0.05",
            "V5_REACTION_COMPLIANCE_OBSERVATION": "INVALID_DEFAULT_OR_STALE_LEDGER",
            "DBTT_SOURCE_READINESS": "BLOCKED",
            "FINITE_ACTIVATION_ZONE_REQUIRED": "NOT_ESTABLISHED",
            "conditional_N256_raw_traction": 0.038672813574278736,
            "conditional_N256_role": "LOW_QUALITY_DIAGNOSTIC_ONLY_NOT_QUALIFICATION",
        },
        "scope": {
            "v1_through_v5_reinterpreted": False,
            "material_or_core_law_changed": False,
            "scientific_tolerance_changed": False,
            "raw_traction_normalization_changed": False,
            "recovery_operator_changed": False,
            "r_tip_law_changed": False,
            "r_tip_equals_R_void": False,
            "finite_activation_zone_observable_derived": False,
            "paired_trajectories_run": False,
            "fatigue_started": False,
            "missing_fields_inferred_or_synthesized": False,
        },
        "next_bounded_step": (
            "GENERATE_FROZEN_18_STATE_ORACLE_WITHOUT_MAP_FITTING"
            if passed else "EQUILIBRATED_BOUNDARY_STRESS_RECONSTRUCTION"
        ),
    }


def main():
    payload = build_record()
    destination = Path(os.environ.get("V6_READINESS_OUTPUT", str(OUTPUT)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("DBTT_SOURCE_READINESS=" + payload["DBTT_SOURCE_READINESS"])
    print("V6_FAILURE_CLASS=" + "+".join(payload["exact_v6_failure_class"]))
    rows = payload["fixed_geometry_local_family"]["new_rows"]
    print("RAW_TRACTION_D=" + str(rows[0]["raw_adjacent_element_traction_normalized"]))
    print("RAW_TRACTION_E=" + str(rows[1]["raw_adjacent_element_traction_normalized"]))
    print("ORACLE_STATES_ACCEPTED=" + str(payload["oracle_states_accepted"]) + "/18")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
