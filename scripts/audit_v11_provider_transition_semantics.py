#!/usr/bin/env python3
"""Read-only semantic audit of the v10-to-v11 single-front handoff."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.directional_competition_v11 import tungsten_cleavage_candidates
from arrhenius_fracture.fem import assemble_mechanics, elastic_energy_densities
from arrhenius_fracture.hazard_energy_event_gate_v10230 import _infer_boundary_opening
from arrhenius_fracture.j_integral import compute_J_integral
from arrhenius_fracture.kernel_configuration_v10227 import load_configuration
from arrhenius_fracture.live_topology_kernel_registry_v11 import validate_single_front_transition
from arrhenius_fracture.live_topology_kernel_v11 import LiveTopologyRequest, evaluate_exact_topology
from arrhenius_fracture.prescribed_geometry_kernel_v10228 import prescribed_crack_direction


def relative(left, right):
    return abs(float(left) - float(right)) / max(abs(float(left)), abs(float(right)), 1e-300)


def array_relative(left, right):
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)) / max(np.linalg.norm(right), 1e-300))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mechanical-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.checkpoint.read_text())
    checkpoint = pickle.loads(args.checkpoint.with_name(manifest["state_file"]).read_bytes())
    state = checkpoint.state
    configuration = load_configuration(args.mechanical_config)
    candidates = tungsten_cleavage_candidates(theta_deg=configuration.theta_deg)
    branch = state.crack_network.branch(state.crack_network.active_tip_ids[0])
    physical_tangent = np.array(
        [math.cos(branch.current_orientation_rad), math.sin(branch.current_orientation_rad)]
    )
    prescribed = prescribed_crack_direction(configuration)
    segments = [
        (np.asarray(a), np.asarray(b))
        for item in state.crack_network.branches
        for a, b in zip(item.path, item.path[1:])
    ]
    K, residual, sigma, _, _, psi = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    opening_top, opening_bottom = _infer_boundary_opening(state.boundary, state.displacement)
    request_radius = 1.0e-6
    exclude = max(float(state.mesh.hbar_tip), 1e-12)
    request = LiveTopologyRequest(
        mesh=state.mesh, boundary=state.boundary, displacement=state.displacement,
        ep_gp=state.ep_gp, rho_gp=state.rho_gp, damage=state.damage,
        elasticity_D=state.elasticity_D, material=state.material,
        cohesive_network=state.cohesive_network, crack_network=state.crack_network,
        candidates_by_tip={branch.branch_id: candidates},
        mechanical_configuration_fingerprint=configuration.fingerprint(),
        specimen_geometry={
            "Lx": configuration.specimen_length_x_m,
            "Ly": configuration.specimen_length_y_m,
            "a0": configuration.initial_crack_length_m,
            "notch_half_thickness": configuration.notch_half_thickness_m,
        },
        boundary_condition_identity="preserved_checkpoint_read_only",
        elastic_constants={
            "E_Pa": state.material.E, "nu": state.material.nu,
            "Eprime_Pa": state.material.Eprime,
        },
        cluster_frame={"mode": "single_front"},
        mpz_station_coordinates_m=(), wake_station_coordinates_m=(),
        contour_radius_m=request_radius, exclude_radius_m=exclude,
        provider_contract_contour_radius_m=configuration.interaction_length_m,
    )
    live = evaluate_exact_topology(request)
    legacy_directional = []
    for candidate in candidates:
        rows = {}
        for radius in (request_radius, configuration.interaction_length_m):
            _, _, info = compute_J_integral(
                state.mesh, state.displacement, sigma, psi, state.damage,
                np.asarray(branch.tip), np.asarray(candidate.direction_xy), state.material,
                ell=radius, crack_segments=segments, exclude_radius=exclude,
            )
            rows[radius] = info
        nominal = float(rows[request_radius]["J_signed"])
        contract = float(rows[configuration.interaction_length_m]["J_signed"])
        legacy_directional.append({
            "candidate_id": candidate.candidate_id,
            "signed_J_J_per_m2": nominal,
            "positive_J_J_per_m2": max(nominal, 0.0),
            "K_directional_Pa_sqrt_m": math.sqrt(state.material.Eprime * max(nominal, 0.0)),
            "J_provider_contract_signed_J_per_m2": contract,
            "J_provider_contract_positive_J_per_m2": max(contract, 0.0),
            "K_provider_contract_Pa_sqrt_m": math.sqrt(state.material.Eprime * max(contract, 0.0)),
        })
    stored, _ = elastic_energy_densities(
        state.mesh, state.displacement, state.ep_gp, sigma, state.elasticity_D
    )
    legacy = {
        "reaction_force": live["base_equilibrium"]["reaction_force"],
        "recoverable_potential_energy_J_per_m": float(np.sum(stored * state.mesh.area_e)),
        "directional": legacy_directional,
    }
    parity = validate_single_front_transition(legacy, live)
    radii = sorted({
        request_radius, configuration.interaction_length_m,
        *(
            float(row["radius_m"])
            for row in live["tips"][0]["directional"][0]["nested_contour_diagnostics"]
        ),
    })
    directions = {
        "physical_incoming_notch_tangent": physical_tangent,
        "v10_prescribed_crystallographic_direction": prescribed,
        **{f"candidate:{item.plane_variant}": np.asarray(item.direction_xy) for item in candidates},
    }
    matrix = []
    for radius in radii:
        for label, direction in directions.items():
            _, _, info = compute_J_integral(
                state.mesh, live["base_equilibrium"]["displacement"],
                # Recompute stresses below by using the matched live result's
                # equilibrium displacement on the same state arrays.
                sigma, psi, state.damage, np.asarray(branch.tip), direction,
                state.material, ell=radius, crack_segments=segments,
                exclude_radius=exclude,
            )
            matrix.append({
                "auxiliary_direction": label, "direction_xy": direction.tolist(),
                "contour_radius_m": radius, "signed_J_J_per_m2": float(info["J_signed"]),
                "active_elements": int(info["n_active_elements"]),
                "support_valid": bool(info["n_active_elements"] > 0),
            })
    live_u = np.asarray(live["base_equilibrium"]["displacement"])
    element_damage = np.asarray(getattr(state.mesh, "element_damage_gp", np.empty(0)))
    nodes = np.asarray(state.mesh.nodes)
    centroids = np.mean(nodes[np.asarray(state.mesh.elems)], axis=1)
    ahead = (
        (centroids - np.asarray(branch.tip)) @ physical_tangent > 0.0
        if element_damage.size else np.empty(0, dtype=bool)
    )
    intact_ahead = np.linalg.norm(
        centroids[ahead & (element_damage < 0.5)] - np.asarray(branch.tip), axis=1
    ) if element_damage.size and np.any(ahead & (element_damage < 0.5)) else np.empty(0)
    payload = {
        "schema": "v11.provider-transition-semantic-audit/1",
        "inputs": {
            "checkpoint": str(args.checkpoint.resolve()),
            "mechanical_configuration": str(args.mechanical_config.resolve()),
            "hazard_advanced": False, "plasticity_advanced": False,
            "topology_event_committed": False,
        },
        "directions": {
            "v10_physical_incoming_notch_tangent": physical_tangent.tolist(),
            "v10_declared_crack_direction": prescribed.tolist(),
            "v11_physical_incoming_notch_tangent": physical_tangent.tolist(),
            "v11_declared_network_direction": physical_tangent.tolist(),
        },
        "parity_inputs": {
            "specimen": request.specimen_geometry,
            "tip_xy_m": list(branch.tip), "physical_crack_path_vertices_m": list(branch.path),
            "theta_deg": configuration.theta_deg,
            "candidates": [item.to_dict() for item in candidates],
            "mesh": {
                "configured_nx": configuration.mesh_nx, "configured_ny": configuration.mesh_ny,
                "actual_nodes": state.mesh.nn, "actual_elements": state.mesh.ne,
                "tip_h_fine_m": configuration.tip_h_fine_m,
                "tip_ratio": configuration.tip_ratio, "actual_hbar_tip_m": state.mesh.hbar_tip,
            },
            "damage": {
                "representation": "sharp_wake_causal_v11_element_local_P0",
                "killed_nodal_count": int(np.count_nonzero(state.damage >= 0.5)),
                "killed_element_count": int(np.count_nonzero(element_damage >= 0.5)),
                "minimum_intact_centroid_distance_ahead_m": (
                    float(np.min(intact_ahead)) if intact_ahead.size else None
                ),
            },
            "interaction": {
                "v10_provider_contract_radius_m": configuration.interaction_length_m,
                "v11_nominal_radius_m": request_radius, "exclude_radius_m": exclude,
                "coordinate_transformation": "global Cartesian projected onto supplied unit auxiliary direction",
            },
            "elastic_tensor": np.asarray(state.elasticity_D).tolist(),
            "plane_strain": {"E_Pa": state.material.E, "nu": state.material.nu, "Eprime_Pa": state.material.Eprime},
            "applied_displacement_m": opening_top - opening_bottom,
        },
        "global_equilibrium": {
            "reaction_N_per_m": live["base_equilibrium"]["reaction_force"],
            "recoverable_energy_J_per_m": live["base_equilibrium"]["recoverable_potential_energy_J_per_m"],
            "accepted_vs_live_displacement_relative_l2": array_relative(state.displacement, live_u),
            "accepted_vs_live_displacement_max_abs_m": float(np.max(np.abs(state.displacement - live_u))),
            "accepted_vs_live_energy_relative": relative(legacy["recoverable_potential_energy_J_per_m"], live["base_equilibrium"]["recoverable_potential_energy_J_per_m"]),
        },
        "provider_contract_parity": parity,
        "legacy_directional": legacy_directional,
        "live_directional": live["tips"][0]["directional"],
        "contour_direction_matrix": matrix,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False,
        default=lambda value: value.tolist() if isinstance(value, np.ndarray) else float(value),
    ) + "\n")
    print(json.dumps({
        "out": str(args.out), "parity_passed": parity["passed"],
        "maximum_provider_contract_residual": max(
            value for key, value in parity.items() if ":" in key
        ),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
