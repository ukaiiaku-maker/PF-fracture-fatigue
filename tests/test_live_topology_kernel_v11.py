from dataclasses import replace
import math

import numpy as np
import pytest

from arrhenius_fracture.branch_cluster_v11 import create_unresolved_branch_cluster
from arrhenius_fracture.config import GeometryConfig, MeshConfig
from arrhenius_fracture.crack_network_v11 import CrackNetworkState, ROOT_BRANCH_ID
from arrhenius_fracture.live_topology_kernel_cache_v11 import ExactTopologyCache
from arrhenius_fracture.live_topology_kernel_v11 import topology_fingerprint
from arrhenius_fracture.live_topology_kernel_v11 import LiveTopologyRequest, evaluate_exact_topology
from arrhenius_fracture.live_topology_kernel_registry_v11 import validate_single_front_transition
from arrhenius_fracture.directional_competition_v11 import tungsten_cleavage_candidates
from arrhenius_fracture.fem import elastic_energy_densities, plane_strain_D
from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.j_integral import compute_J_integral
from arrhenius_fracture.unit_slip_perturbation_v1026 import solve_fixed_crack_state
from arrhenius_fracture.topology_transaction_v11 import TopologyArm, extend_network_arm, mark_coalesced
from arrhenius_fracture.mesh import make_boundary_data, make_tri_mesh


def branched(*, asymmetric=False, coalesced=False):
    network = CrackNetworkState.one_tip(((0.0, 0.0), (1.0, 0.0)))
    network, cluster = create_unresolved_branch_cluster(
        network, parent_branch_id=ROOT_BRANCH_ID, candidate_ids=("ca", "cb"),
        event_index=2, shared_process_state={},
        conserved_ledgers={name: 0.0 for name in (
            "retained", "mobile", "escaped", "recovered", "stored_energy",
            "emission_work", "unconsumed_action",
        )},
    )
    lengths = (0.4, 0.7 if asymmetric else 0.4)
    for branch_id, candidate, length, sign in zip(cluster.arm_branch_ids, ("ca", "cb"), lengths, (-1, 1)):
        network = extend_network_arm(network, TopologyArm(
            candidate, branch_id, (1.0, 0.0), (1.0 + length, sign * length),
            math.sqrt(2.0) * length, 0.0,
        ))
    if coalesced:
        network = mark_coalesced(network, cluster.arm_branch_ids[1], cluster.arm_branch_ids[0])
    return network


def fingerprint(network):
    geometry = GeometryConfig(Lx=2.0, Ly=2.0, a0=0.5, notch_half_thickness=0.05)
    mesh = make_tri_mesh(geometry, MeshConfig(nx=8, ny=8, jitter=0.0), seed=1)
    boundary = make_boundary_data(mesh, geometry)
    damage = np.zeros(mesh.nn)
    damage[boundary.notch_nodes] = 1.0
    return topology_fingerprint(
        network=network, mesh=mesh, damage=damage,
        mechanical_configuration_fingerprint="mechanics",
        specimen_geometry={"Lx": 2.0, "Ly": 2.0, "a0": 0.5},
        boundary_condition_identity="symmetric_displacement",
        elastic_constants={"E": 410e9, "nu": 0.28}, cluster_frame={"junction": [1.0, 0.0]},
        mpz_station_coordinates_m=((1.1, 0.0),), wake_station_coordinates_m=((0.9, 0.0),),
        contour_definitions={"radius_m": 0.2},
    )


def relabel(network):
    mapping = {branch.branch_id: f"arbitrary-{index}" for index, branch in enumerate(reversed(network.branches))}
    branches = tuple(
        replace(
            branch, branch_id=mapping[branch.branch_id],
            parent_branch_id=None if branch.parent_branch_id is None else mapping[branch.parent_branch_id],
        )
        for branch in reversed(network.branches)
    )
    return replace(network, branches=branches, primary_branch_id=mapping[network.primary_branch_id])


def test_exact_fingerprint_is_enumeration_edge_order_and_daughter_id_invariant():
    network = branched()
    assert fingerprint(network) == fingerprint(relabel(network))


def test_straight_symmetric_asymmetric_one_arm_and_coalesced_are_distinct():
    straight = CrackNetworkState.one_tip(((0.0, 0.0), (1.0, 0.0)))
    symmetric = branched()
    asymmetric = branched(asymmetric=True)
    one_arm = extend_network_arm(symmetric, TopologyArm(
        "ca", symmetric.active_tip_ids[0], symmetric.branch(symmetric.active_tip_ids[0]).tip,
        (1.8, -0.4), math.hypot(0.4, 0.0), 0.0,
    ))
    coalesced = branched(coalesced=True)
    values = {fingerprint(item) for item in (straight, symmetric, asymmetric, one_arm, coalesced)}
    assert len(values) == 5


def test_cache_persists_only_explicitly_accepted_exact_state(tmp_path):
    cache = ExactTopologyCache(tmp_path)
    topology = fingerprint(branched())
    calls = []
    result, cached = cache.get_or_evaluate_accepted(
        "mechanics", topology,
        lambda: calls.append(1) or {"topology_fingerprint": topology, "value": 7},
    )
    assert not cached and result["value"] == 7
    again, cached = cache.get_or_evaluate_accepted(
        "mechanics", topology, lambda: (_ for _ in ()).throw(AssertionError()),
    )
    assert cached and again == result and calls == [1]
    assert not list(tmp_path.rglob("*rejected*"))


def live_straight_request(extension_m):
    geometry = GeometryConfig(Lx=1.0e-3, Ly=1.0e-3, a0=0.25e-3, notch_half_thickness=20e-6)
    tip = np.array([geometry.a0 + extension_m, 0.0])
    mesh = make_tri_mesh(
        geometry, MeshConfig(nx=24, ny=32, jitter=0.0), seed=1729,
        tip_center=tip,
    )
    boundary = make_boundary_data(mesh, geometry)
    damage = np.zeros(mesh.nn)
    damage[boundary.notch_nodes] = 1.0
    if extension_m > 0.0:
        from arrhenius_fracture.crack_backend import SharpWakeBackend
        damage = SharpWakeBackend().advance(
            mesh=mesh, boundary=boundary, damage=damage,
            displacement=np.zeros(mesh.ndof), p0=np.array([geometry.a0, 0.0]),
            p1=tip, kill_r=0.5 * mesh.hbar_tip,
        ).damage
    path = ((0.0, 0.0), (geometry.a0, 0.0))
    if extension_m > 0.0:
        path += (tuple(tip),)
    network = CrackNetworkState.one_tip(path)
    material = ElasticProperties()
    D = plane_strain_D(material)
    ep = np.zeros((3, mesh.ne))
    rho = np.zeros(mesh.ne)
    initial = np.zeros(mesh.ndof)
    initial[2 * boundary.top_nodes + 1] = 1.0e-7
    initial[2 * boundary.bot_nodes + 1] = -1.0e-7
    candidates = tungsten_cleavage_candidates(theta_deg=30.0, include_110=True)
    request = LiveTopologyRequest(
        mesh=mesh, boundary=boundary, displacement=initial, ep_gp=ep,
        rho_gp=rho, damage=damage, elasticity_D=D, material=material,
        cohesive_network=None, crack_network=network,
        candidates_by_tip={network.active_tip_ids[0]: candidates},
        mechanical_configuration_fingerprint="straight-parity",
        specimen_geometry={"Lx": geometry.Lx, "Ly": geometry.Ly, "a0": geometry.a0},
        boundary_condition_identity="symmetric_fixed_opening",
        elastic_constants={"E": material.E, "nu": material.nu}, cluster_frame={},
        mpz_station_coordinates_m=(), wake_station_coordinates_m=(),
        contour_radius_m=3.0 * mesh.hbar_tip,
        exclude_radius_m=0.5 * mesh.hbar_tip,
    )
    return request


@pytest.mark.parametrize("extension_um", [0.0, 5.0, 10.0, 25.0])
def test_live_provider_matches_direct_single_front_FEM_anchors(extension_um):
    request = live_straight_request(extension_um * 1.0e-6)
    live = evaluate_exact_topology(request)
    Uy_top = 1.0e-7
    base = solve_fixed_crack_state(
        mesh=request.mesh, boundary=request.boundary, u=request.displacement,
        ep_gp=request.ep_gp, rho_gp=request.rho_gp, d=request.damage,
        D=request.elasticity_D, mat=request.material,
        Uy_top=Uy_top, Uy_bot=-Uy_top,
    )
    branch = request.crack_network.branch(request.crack_network.active_tip_ids[0])
    segments = [
        (np.asarray(a), np.asarray(b))
        for item in request.crack_network.branches
        for a, b in zip(item.path, item.path[1:])
    ]
    directional = []
    for candidate in request.candidates_by_tip[branch.branch_id]:
        _, _, info = compute_J_integral(
            request.mesh, base["u"], base["sigma_gp"], base["psi_e_gp"],
            request.damage, np.asarray(branch.tip), np.asarray(candidate.direction_xy),
            request.material, ell=request.contour_radius_m,
            crack_segments=segments, exclude_radius=request.exclude_radius_m,
        )
        signed = float(info["J_signed"])
        positive = max(signed, 0.0)
        directional.append({
            "candidate_id": candidate.candidate_id,
            "signed_J_J_per_m2": signed, "positive_J_J_per_m2": positive,
            "K_directional_Pa_sqrt_m": math.sqrt(request.material.Eprime * positive),
        })
    stored, _ = elastic_energy_densities(
        request.mesh, base["u"], request.ep_gp, base["sigma_gp"], request.elasticity_D
    )
    legacy = {
        "reaction_force": base["reaction_top"],
        "recoverable_potential_energy_J_per_m": float(np.sum(stored * request.mesh.area_e)),
        "directional": directional,
    }
    parity = validate_single_front_transition(legacy, live)
    assert parity["passed"] and parity["sign_agreement"]
