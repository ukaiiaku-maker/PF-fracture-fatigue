#!/usr/bin/env python3
"""Deterministic whole-body FEM compliance audit for close crack pairs."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.causal_sharp_wake_v11 import causal_segment_support
from arrhenius_fracture.config import ElasticProperties, GeometryConfig, MeshConfig
from arrhenius_fracture.crack_network_v11 import CrackBranchState, CrackNetworkState
from arrhenius_fracture.directional_competition_v11 import tungsten_cleavage_candidates
from arrhenius_fracture.fem import plane_strain_D
from arrhenius_fracture.live_topology_kernel_v11 import LiveTopologyRequest, evaluate_exact_topology
from arrhenius_fracture.mesh import make_boundary_data, make_tri_mesh


SOURCE_PAIR = {
    "checkpoint": "runs/v11_refinement_scalability/corrected_gate500_v3/checkpoint/latest.json",
    "front_ids": ["b437422fcb9474eb", "b5f2bd5610a01132"],
    "tip_separation_um": 22.36067977499797,
}


def branch(branch_id: str, parent: str, path, *, status="active"):
    points = tuple(tuple(map(float, point)) for point in path)
    angles = tuple(math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(points, points[1:]))
    return CrackBranchState(branch_id, parent, 1, 1, points, angles, status=status)


def translated(path, offset):
    return tuple((x + offset[0], y + offset[1]) for x, y in path)


def extended(path, distance=5.0e-6):
    angle = math.atan2(path[-1][1] - path[-2][1], path[-1][0] - path[-2][0])
    return tuple(path) + ((path[-1][0] + distance * math.cos(angle), path[-1][1] + distance * math.sin(angle)),)


def make_network(paths):
    junction = paths[0][0]
    root = CrackBranchState(
        "b00000000", None, 0, 0, ((0.0, 0.0), junction),
        (math.atan2(junction[1], junction[0]),), status="terminated",
    )
    children = tuple(branch(f"tip-{index + 1}", root.branch_id, path) for index, path in enumerate(paths))
    return CrackNetworkState((root,) + children, geometry_generation=1, branching_enabled=True)


def request_for(mesh0, boundary, network, material, opening=2.0e-7):
    visual = np.zeros(mesh0.nn)
    visual[boundary.notch_nodes] = 1.0
    p0 = np.mean(visual[mesh0.elems], axis=1)
    for item in network.branches:
        for start, end in zip(item.path, item.path[1:]):
            selected, _ = causal_segment_support(mesh0, np.asarray(start), np.asarray(end))
            p0[selected] = 1.0
            if selected.size:
                visual[np.unique(mesh0.elems[selected])] = 1.0
    mesh = replace(mesh0, element_damage_gp=p0)
    displacement = np.zeros(mesh.ndof)
    displacement[2 * boundary.top_nodes + 1] = 0.5 * opening
    displacement[2 * boundary.bot_nodes + 1] = -0.5 * opening
    candidates = tungsten_cleavage_candidates(theta_deg=45.0, include_110=True)
    return LiveTopologyRequest(
        mesh=mesh, boundary=boundary, displacement=displacement,
        ep_gp=np.zeros((3, mesh.ne)), rho_gp=np.zeros(mesh.ne), damage=visual,
        elasticity_D=plane_strain_D(material), material=material,
        cohesive_network=None, crack_network=network,
        candidates_by_tip={tip: candidates for tip in network.active_tip_ids},
        mechanical_configuration_fingerprint="v11-close-pair-compliance-audit-v1",
        specimen_geometry={"Lx": 1.0e-3, "Ly": 1.0e-3, "a0": 0.25e-3},
        boundary_condition_identity="symmetric_fixed_opening",
        elastic_constants={"E": material.E, "nu": material.nu},
        cluster_frame={"source_pair": SOURCE_PAIR},
        mpz_station_coordinates_m=(), wake_station_coordinates_m=(),
        contour_radius_m=30.0e-6, exclude_radius_m=3.0e-6,
    )


def summarize(live):
    equilibrium = live["base_equilibrium"]
    tips = {}
    for index, tip in enumerate(live["tips"], 1):
        best = max(tip["directional"], key=lambda row: row["signed_J_J_per_m2"])
        tips[f"tip_{index}"] = {
            "maximum_signed_J_J_per_m2": best["signed_J_J_per_m2"],
            "J_local_signed_J_per_m2": best["J_local_signed_J_per_m2"],
            "local_J_valid": best["local_J_valid"],
            "local_J_invalid_reason": best["local_J_invalid_reason"],
            "J_contour_radius_m": best["J_contour_radius_m"],
            "nearest_other_crack_distance_m": best["nearest_other_crack_distance_m"],
            "nearest_junction_distance_m": best["nearest_junction_distance_m"],
            "nearest_specimen_boundary_distance_m": best["nearest_specimen_boundary_distance_m"],
            "candidate_id": best["candidate_id"],
            "local_contour_valid": best["local_contour_valid"],
        }
    return {
        "reaction_force_N_per_m": equilibrium["reaction_force"],
        "applied_displacement_m": equilibrium["applied_displacement"],
        "apparent_compliance_m2_per_N": equilibrium["apparent_compliance"],
        "recoverable_elastic_energy_J_per_m": equilibrium["recoverable_potential_energy_J_per_m"],
        "topology_fingerprint": live["topology_fingerprint"],
        "tips": tips,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    args = parser.parse_args()
    geometry = GeometryConfig(Lx=1.0e-3, Ly=1.0e-3, a0=0.25e-3, notch_half_thickness=20e-6)
    mesh = make_tri_mesh(geometry, MeshConfig(nx=args.nx, ny=args.ny, jitter=0.0), seed=1729)
    boundary = make_boundary_data(mesh, geometry)
    material = ElasticProperties()
    path1 = (
        (0.0005176776695296637, 1.0606601717798251e-05),
        (0.0005212132034355965, 7.0710678118655e-06),
        (0.0005247487373415292, 3.5355339059327496e-06),
        (0.000528284271247462, 7.0710678118655e-06),
        (0.0005318198051533947, 1.0606601717798251e-05),
    )
    path2 = (
        (0.0005176776695296637, 1.0606601717798251e-05),
        (0.0005212132034355965, 1.4142135623731002e-05),
        (0.0005247487373415292, 1.7677669529663753e-05),
        (0.000528284271247462, 2.1213203435596502e-05),
        (0.0005318198051533947, 2.474873734152925e-05),
        (0.0005353553390593274, 2.8284271247462e-05),
        (0.0005388908729652602, 3.181980515339475e-05),
    )
    normal = np.array((-math.sqrt(0.5), math.sqrt(0.5)))
    wider = translated(path2, normal * (90.0e-6 - SOURCE_PAIR["tip_separation_um"] * 1.0e-6))
    definitions = {
        "A_isolated": (path2,),
        "B_close_parallel": (path1, path2),
        "C_close_tip2_5um_ahead": (path1, extended(path2)),
        "D_wider_parallel": (path1, wider),
    }
    states = {}
    for name, paths in definitions.items():
        network = make_network(paths)
        base = evaluate_exact_topology(request_for(mesh, boundary, network, material))
        row = summarize(base)
        releases = {}
        for index in range(len(paths)):
            advanced = list(paths); advanced[index] = extended(advanced[index])
            trial = evaluate_exact_topology(request_for(mesh, boundary, make_network(tuple(advanced)), material))
            releases[f"advance_tip_{index + 1}_5um_J_per_m"] = (
                row["recoverable_elastic_energy_J_per_m"]
                - trial["base_equilibrium"]["recoverable_potential_energy_J_per_m"]
            )
        if len(paths) == 2:
            both = tuple(extended(path) for path in paths)
            trial = evaluate_exact_topology(request_for(mesh, boundary, make_network(both), material))
            releases["advance_both_5um_J_per_m"] = (
                row["recoverable_elastic_energy_J_per_m"]
                - trial["base_equilibrium"]["recoverable_potential_energy_J_per_m"]
            )
        row["topology_trial_energy_releases"] = releases
        for index, tip in enumerate(row["tips"].values(), 1):
            marginal = releases[f"advance_tip_{index}_5um_J_per_m"] / 5.0e-6
            tip["G_marginal_J_per_m2"] = marginal
            tip["J_kin_used_J_per_m2"] = (
                max(tip["J_local_signed_J_per_m2"], 0.0)
                if tip["local_J_valid"] else max(marginal, 0.0)
            )
        states[name] = row
    payload = {
        "schema": "v11.close-crack-compliance-audit/1",
        "source_geometry": SOURCE_PAIR,
        "mesh": {"nodes": mesh.nn, "elements": mesh.ne, "nx": args.nx, "ny": args.ny},
        "J_contour_radius_um": 30.0,
        "close_pair_contour_intersects_neighbor": True,
        "states": states,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
