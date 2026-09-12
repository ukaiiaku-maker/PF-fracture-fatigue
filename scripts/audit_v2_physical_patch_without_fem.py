#!/usr/bin/env python3
"""Reconstruct the retained V2 patch geometry without a mechanics solve."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.adaptive_multitip_mesh_v11 import refine_accepted_state
from arrhenius_fracture.cavity_source_recovery_v2 import recover_fixed_arc_patch_v2
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.quality_constrained_mesh_v1 import (
    constrained_quality_mesh,
    production_geometry_constraints,
)
from arrhenius_fracture.voiding_production_v5 import _actual_cavity_boundary_edges


SOURCE_CHECKPOINT = Path(
    "artifacts/voiding_v5_finalization_v2/checkpoints/connected_before_downstream.json"
)
OUTPUT = Path(
    "artifacts/v5_cavity_source_recovery_v3/v2_readiness_taxonomy_and_patch_audit.json"
)
V2_READINESS = (
    {
        "level": 1,
        "sigma_tt_Pa": 748567743.0182768,
        "tensor_relative_change": 0.10636255970669334,
        "traction_residual": 0.039290171156949674,
        "eta_n_max": 0.2125103829911522,
        "eta_t_max": 0.09849140335716547,
    },
    {
        "level": 2,
        "sigma_tt_Pa": 769175348.4353346,
        "tensor_relative_change": 0.026791817313149703,
        "traction_residual": 0.024058321815199565,
        "eta_n_max": 0.10625519149557615,
        "eta_t_max": 0.04924570167858326,
    },
    {
        "level": 3,
        "sigma_tt_Pa": 814505323.3733457,
        "tensor_relative_change": 0.055653380815576595,
        "traction_residual": 0.020367426159639445,
        "eta_n_max": 0.05312759574778982,
        "eta_t_max": 0.02462285083929259,
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quality_mesh(state):
    fixed, protected = production_geometry_constraints(state)
    mesh, audit = constrained_quality_mesh(
        state.mesh,
        fixed_nodes=fixed,
        protected_edges=protected,
        strategy="flips",
    )
    return replace(state, mesh=mesh), audit


def _geometry_record(state, level: int) -> dict[str, object]:
    cavity = state.void_state.cavities[0]
    position = np.asarray(cavity.connection_exit_m, dtype=float)
    boundary_node = int(
        np.argmin(np.linalg.norm(np.asarray(state.mesh.nodes) - position, axis=1))
    )
    solid = np.asarray(state.mesh.element_damage_gp, dtype=float) < 0.5
    # A zero tensor is intentional: only geometry, stencil, and WLS design
    # metadata are reconstructed. Scientific stress results come exclusively
    # from the retained published readiness record below.
    recovery = recover_fixed_arc_patch_v2(
        nodes=state.mesh.nodes,
        elements=state.mesh.elems,
        stress_Pa=np.zeros((3, len(state.mesh.elems))),
        boundary_node=boundary_node,
        cavity_id=cavity.cavity_id,
        cavity_center_m=cavity.center_m,
        cavity_radius_m=cavity.radius_m,
        owned_boundary_edges=_actual_cavity_boundary_edges(state),
        solid_element_mask=solid,
    )
    ids = np.asarray(recovery["stencil_element_ids"], dtype=int)
    centroids = np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)[ids]].mean(axis=1)
    offsets = centroids - position
    s = offsets @ np.asarray(recovery["tangent_xy"], dtype=float)
    n = offsets @ np.asarray(recovery["normal_xy"], dtype=float)
    weights = np.asarray(recovery["weights"], dtype=float)
    retained = V2_READINESS[level - 1]
    return {
        "refinement_level": level,
        "mesh_node_count": int(state.mesh.nn),
        "mesh_element_count": int(state.mesh.ne),
        "patch_element_ids": ids.tolist(),
        "sample_count": int(len(ids)),
        "physical_tangential_extent_m": [float(np.min(s)), float(np.max(s))],
        "physical_normal_extent_m": [float(np.min(n)), float(np.max(n))],
        "centroid_distribution_local_s_n_m": np.column_stack((s, n)).tolist(),
        "weight_distribution": weights.tolist(),
        "weighted_moments": {
            "sum_w": float(np.sum(weights)),
            "sum_w_s_m": float(weights @ s),
            "sum_w_n_m": float(weights @ n),
            "sum_w_s2_m2": float(weights @ (s * s)),
            "sum_w_sn_m2": float(weights @ (s * n)),
            "sum_w_n2_m2": float(weights @ (n * n)),
        },
        "polynomial_design_rank": int(recovery["design_rank"]),
        "polynomial_design_condition": float(recovery["design_condition"]),
        "patch_scale_m": float(recovery["patch_scale_m"]),
        "recovered_sigma_tt_Pa": retained["sigma_tt_Pa"],
        "fit_residual": {
            "status": "NOT_RETAINED_IN_V2_EVIDENCE_NO_NEW_FEM_SOLVE_PERMITTED",
            "value": None,
        },
        "traction_residual": retained["traction_residual"],
        "tensor_relative_change": retained["tensor_relative_change"],
        "eta_n_max": retained["eta_n_max"],
        "eta_t_max": retained["eta_t_max"],
        "source_arc_coordinate_identity": {
            "cavity_id": cavity.cavity_id,
            "cavity_center_m": list(cavity.center_m),
            "cavity_radius_m": float(cavity.radius_m),
            "boundary_position_m": list(cavity.connection_exit_m),
        },
        "v2_mesh_bound_physical_arc_identity": recovery["physical_arc_identity"],
    }


def build() -> dict[str, object]:
    checkpoint = ROOT / SOURCE_CHECKPOINT
    state = restore_checkpoint(checkpoint)
    state, preparation = _quality_mesh(state)
    cavity = state.void_state.cavities[0]
    rows = []
    for level in range(1, 4):
        centroids = np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)].mean(axis=1)
        distance = np.abs(
            np.linalg.norm(centroids - np.asarray(cavity.center_m), axis=1)
            - cavity.radius_m
        )
        marked = tuple(
            np.flatnonzero(
                distance <= 0.5 * cavity.radius_m + 2.0 * np.sqrt(state.mesh.area_e)
            )
        )
        state, _ = refine_accepted_state(
            state,
            marked_parent_elements=marked,
            active_tip_ids=(),
            generation=int(state.event_counters.get("mesh_generation", 0)) + level,
            operation_index=int(
                state.event_counters.get("refinement_operation_index", 0)
            )
            + level,
        )
        state, _ = _quality_mesh(state)
        rows.append(_geometry_record(state, level))

    tangential_widths = [
        row["physical_tangential_extent_m"][1]
        - row["physical_tangential_extent_m"][0]
        for row in rows
    ]
    normal_depths = [row["physical_normal_extent_m"][1] for row in rows]
    coordinate_identities = [row["source_arc_coordinate_identity"] for row in rows]
    return {
        "schema": "v5.cavity-source-recovery-v2-taxonomy-and-patch-audit/1",
        "audit_kind": "DIAGNOSTIC_ONLY_NO_NEW_FEM_SOLVE",
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "source_readiness_record": {
            "repository": "ukaiiaku-maker/Arrhenius_FEM_CZM_MPZ",
            "branch": "codex/oneD-v3-stateful-voiding-reduction",
            "published_head": "225a3e244010cc7143d6268aeb331a22f4b18cdb",
            "path": "analysis_outputs/oneD_v3_aligned_temperature_transfer/aligned_oracle_readiness.json",
        },
        "v2_result_preservation": {
            "CAVITY_SOURCE_RECOVERY_V1_INCIDENT_CST_MAX_PRINCIPAL": "FAIL_NONCONVERGENT",
            "CAVITY_FIXED_ARC_PATCH_RECOVERY_V2_MANUFACTURED_AND_KIRSCH": "PASS",
            "DBTT_V2_CAVITY_TRACTION": "PASS",
            "DBTT_V2_FIXED_ARC_TENSOR_CONVERGENCE": "FAIL",
        },
        "readiness_gate_taxonomy": {
            "decision": "B_STILL_MANDATORY_V2_GATES",
            "evidence": [
                "arrhenius_fracture/closure_static_evidence.py:RESOLUTION_SCREEN",
                "arrhenius_fracture/voiding_production_v5.py:_qualified_cavity_source",
            ],
            "eta_n": {
                "classification": "MANDATORY_V2_GATE",
                "limit": 0.03,
                "final_value": V2_READINESS[-1]["eta_n_max"],
                "passed": False,
            },
            "eta_t": {
                "classification": "MANDATORY_V2_GATE",
                "limit": 0.025,
                "final_value": V2_READINESS[-1]["eta_t_max"],
                "passed": True,
            },
            "failure_classification": [
                "TANGENTIAL_STRESS_CONVERGENCE",
                "NORMAL_DIRECTION_RESOLUTION",
            ],
        },
        "quality_preparation_geometry": preparation,
        "levels": rows,
        "physical_footprint_decision": {
            "two_ring_rule_changes_physical_footprint_across_levels": True,
            "tangential_full_width_m_by_level": tangential_widths,
            "normal_depth_m_by_level": normal_depths,
            "successive_tangential_width_ratios": [
                tangential_widths[index + 1] / tangential_widths[index]
                for index in range(2)
            ],
            "successive_normal_depth_ratios": [
                normal_depths[index + 1] / normal_depths[index]
                for index in range(2)
            ],
            "fixed_source_arc_coordinate_across_levels": all(
                value == coordinate_identities[0] for value in coordinate_identities[1:]
            ),
            "v2_identity_hash_changes_with_mesh_boundary_edge_ids": len(
                {row["v2_mesh_bound_physical_arc_identity"]["sha256"] for row in rows}
            )
            > 1,
        },
    }


def main() -> int:
    payload = build()
    destination = ROOT / OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(destination.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
