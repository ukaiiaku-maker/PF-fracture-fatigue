"""Exact-crack-network live FEM mechanics provider for v11 branching."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .crack_network_v11 import CrackNetworkState
from .directional_competition_v11 import CleavageCandidate
from .fem import elastic_energy_densities
from .hazard_energy_event_gate_v10230 import _infer_boundary_opening
from .interaction_integral_v1026 import compute_signed_interaction_integral
from .j_integral import compute_J_integral
from .slip_ribbon_overlap_v10214 import overlap_weighted_slip_ribbon_increment
from .unit_slip_perturbation_v10212 import (
    _mask_killed_ribbon_elements,
    equilibrated_base_state,
)
from .unit_slip_perturbation_v1026 import SlipRibbonPerturbation, solve_fixed_crack_state


SCHEMA = "v11_exact_topology_live_fem_provider_v2"
PROVIDER_ID = "v11_exact_crack_network_live_fem_v1"
PROVIDER_SEMANTICS_ID = "provider_contract_j_separated_from_nested_local_j_v1"
MAXIMUM_FRONTS_SUPPORTED = 16


def _canonical_float(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("topology fingerprint values must be finite")
    if number == 0.0:
        number = 0.0
    return number.hex()


def _point(point) -> tuple[str, str]:
    return tuple(_canonical_float(value) for value in point)  # type: ignore[return-value]


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_topology_payload(
    *,
    network: CrackNetworkState,
    mesh,
    damage: np.ndarray,
    mechanical_configuration_fingerprint: str,
    specimen_geometry: Mapping[str, Any],
    boundary_condition_identity: str,
    elastic_constants: Mapping[str, float],
    cluster_frame: Mapping[str, Any],
    mpz_station_coordinates_m: Sequence[Sequence[float]],
    wake_station_coordinates_m: Sequence[Sequence[float]],
    contour_definitions: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize the physical graph without retaining arbitrary branch IDs."""
    records = []
    for branch in network.branches:
        path = tuple(_point(point) for point in branch.path)
        parent_path = (
            None if branch.parent_branch_id is None
            else tuple(_point(point) for point in network.branch(branch.parent_branch_id).path)
        )
        records.append({
            "path": path,
            "parent_path": parent_path,
            "status": branch.status,
            "generation": branch.generation,
            "initiation_event": branch.initiation_event,
            "active_tip": _point(branch.tip) if branch.status == "active" else None,
            "active_direction": (
                (_canonical_float(math.cos(branch.current_orientation_rad)),
                 _canonical_float(math.sin(branch.current_orientation_rad)))
                if branch.status == "active" else None
            ),
        })
    records.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    edges = sorted(
        (path[index], path[index + 1])
        for item in records
        for path in (item["path"],)
        for index in range(len(path) - 1)
    )
    return {
        "mechanical_configuration_fingerprint": str(mechanical_configuration_fingerprint),
        "specimen_geometry": json.loads(json.dumps(dict(specimen_geometry), sort_keys=True, allow_nan=False)),
        "boundary_condition_identity": str(boundary_condition_identity),
        "elastic_constants": {key: _canonical_float(value) for key, value in sorted(elastic_constants.items())},
        "mesh": {
            "nodes_sha256": _array_hash(np.asarray(mesh.nodes, dtype=float)),
            "elements_sha256": _array_hash(np.asarray(mesh.elems, dtype=np.int64)),
            "node_count": int(mesh.nn), "element_count": int(mesh.ne),
        },
        "crack_graph": records,
        "crack_edges": edges,
        "branch_junction_coordinates": sorted(
            path[0] for item in records for path in (item["path"],)
            if item["generation"] > 0
        ),
        "sharp_wake_damage": {
            "representation": (
                "nested_parent_inherited_element_stiffness_kill"
                if getattr(mesh, "element_damage_gp", None) is not None
                else "nodal_stiffness_kill"
            ),
            "sha256": _array_hash(np.asarray(damage, dtype=float)),
            "element_sha256": (
                _array_hash(np.asarray(mesh.element_damage_gp, dtype=float))
                if getattr(mesh, "element_damage_gp", None) is not None else None
            ),
        },
        "shared_cluster_frame": json.loads(json.dumps(dict(cluster_frame), sort_keys=True, allow_nan=False)),
        "mpz_station_coordinates_m": sorted(_point(value) for value in mpz_station_coordinates_m),
        "wake_station_coordinates_m": sorted(_point(value) for value in wake_station_coordinates_m),
        "interaction_integral_contours": json.loads(json.dumps(dict(contour_definitions), sort_keys=True, allow_nan=False)),
    }


def topology_fingerprint(**kwargs) -> str:
    payload = canonical_topology_payload(**kwargs)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LiveTopologyRequest:
    mesh: Any
    boundary: Any
    displacement: np.ndarray
    ep_gp: np.ndarray
    rho_gp: np.ndarray
    damage: np.ndarray
    elasticity_D: np.ndarray
    material: Any
    cohesive_network: Any
    crack_network: CrackNetworkState
    candidates_by_tip: Mapping[str, tuple[CleavageCandidate, ...]]
    mechanical_configuration_fingerprint: str
    specimen_geometry: Mapping[str, Any]
    boundary_condition_identity: str
    elastic_constants: Mapping[str, float]
    cluster_frame: Mapping[str, Any]
    mpz_station_coordinates_m: tuple[tuple[float, float], ...]
    wake_station_coordinates_m: tuple[tuple[float, float], ...]
    contour_radius_m: float
    exclude_radius_m: float
    provider_contract_contour_radius_m: float | None = None
    shared_perturbations: tuple[SlipRibbonPerturbation, ...] = ()


def _segments(network: CrackNetworkState):
    return [
        (np.asarray(a, dtype=float), np.asarray(b, dtype=float))
        for branch in network.branches
        for a, b in zip(branch.path, branch.path[1:])
    ]


def _tip_direction(branch) -> np.ndarray:
    return np.array([math.cos(branch.current_orientation_rad), math.sin(branch.current_orientation_rad)])


def request_contour_definitions(request: LiveTopologyRequest) -> dict[str, Any]:
    return {
        "radius_m": request.contour_radius_m,
        "exclude_radius_m": request.exclude_radius_m,
        "provider_contract_radius_m": request.provider_contract_contour_radius_m,
        "provider_semantics_id": PROVIDER_SEMANTICS_ID,
        "directional_candidates_by_physical_tip": sorted(
            ({
                "tip": _point(request.crack_network.branch(branch_id).tip),
                "directions": sorted(
                    (_point(candidate.direction_xy), candidate.candidate_id)
                    for candidate in request.candidates_by_tip[branch_id]
                ),
            } for branch_id in request.crack_network.active_tip_ids),
            key=lambda item: json.dumps(item, sort_keys=True),
        ),
    }


def _shared_unit_response(request: LiveTopologyRequest, base: Mapping[str, Any], perturbation):
    raw, _, support = overlap_weighted_slip_ribbon_increment(request.mesh, perturbation)
    increment, audit = _mask_killed_ribbon_elements(
        request.mesh, request.damage, raw, support,
        minimum_residual_stiffness_fraction=1.0e-3, stiffness_kappa=1.0e-6,
    )
    Uy_top, Uy_bot = _infer_boundary_opening(request.boundary, np.asarray(base["u"]))
    perturbed = solve_fixed_crack_state(
        mesh=request.mesh, boundary=request.boundary, u=np.asarray(base["u"]),
        ep_gp=np.asarray(request.ep_gp) + increment, rho_gp=request.rho_gp,
        d=request.damage, D=request.elasticity_D, mat=request.material,
        Uy_top=Uy_top, Uy_bot=Uy_bot, cohesive_network=request.cohesive_network,
    )
    rows = []
    segments = _segments(request.crack_network)
    content = float(perturbation.signed_line_content)
    for branch_id in request.crack_network.active_tip_ids:
        branch = request.crack_network.branch(branch_id)
        common = dict(
            mesh=request.mesh, d=request.damage, crack_tip=np.asarray(branch.tip),
            crack_direction=_tip_direction(branch), mat=request.material,
            ell=request.contour_radius_m, crack_segments=segments,
            exclude_radius=request.exclude_radius_m, D=request.elasticity_D,
        )
        before = compute_signed_interaction_integral(
            u=np.asarray(base["u"]), sigma_gp=np.asarray(base["sigma_gp"]), **common
        )
        after = compute_signed_interaction_integral(
            u=np.asarray(perturbed["u"]), sigma_gp=np.asarray(perturbed["sigma_gp"]), **common
        )
        rows.append({
            "tip_physical_key": [_point(branch.tip), _point(_tip_direction(branch))],
            "H_I_Pa_sqrt_m_per_signed_line": float((before.K_I_Pa_sqrt_m - after.K_I_Pa_sqrt_m) / content),
            "H_II_Pa_sqrt_m_per_signed_line": float((before.K_II_Pa_sqrt_m - after.K_II_Pa_sqrt_m) / content),
        })
    rows.sort(key=lambda row: json.dumps(row["tip_physical_key"]))
    return {"perturbation": perturbation.audit_payload(), "rows": rows, "support": audit}


def evaluate_exact_topology(request: LiveTopologyRequest) -> dict[str, Any]:
    """Equilibrate and measure one exact accepted or ephemeral trial topology."""
    Uy_top, Uy_bot = _infer_boundary_opening(request.boundary, request.displacement)
    base = equilibrated_base_state(
        mesh=request.mesh, boundary=request.boundary, baseline_u=request.displacement,
        baseline_ep_gp=request.ep_gp, rho_gp=request.rho_gp, d=request.damage,
        D=request.elasticity_D, mat=request.material, Uy_top=Uy_top, Uy_bot=Uy_bot,
        cohesive_network=request.cohesive_network,
    )
    stored, _ = elastic_energy_densities(
        request.mesh, base["u"], request.ep_gp, base["sigma_gp"], request.elasticity_D
    )
    energy = float(np.sum(stored * request.mesh.area_e))
    segments = _segments(request.crack_network)
    branch_segments = {
        item.branch_id: tuple(
            (np.asarray(a, dtype=float), np.asarray(b, dtype=float))
            for a, b in zip(item.path, item.path[1:])
        ) for item in request.crack_network.branches
    }
    junctions = tuple(
        np.asarray(item.root, dtype=float)
        for item in request.crack_network.branches
        if item.parent_branch_id is not None
    )
    domain_min = np.min(request.mesh.nodes, axis=0)
    domain_max = np.max(request.mesh.nodes, axis=0)

    def point_segment_distance(point, start, end):
        delta = end - start
        scale = float(delta @ delta)
        if scale <= np.finfo(float).tiny:
            return float(np.linalg.norm(point - start))
        fraction = min(1.0, max(0.0, float((point - start) @ delta) / scale))
        return float(np.linalg.norm(point - (start + fraction * delta)))

    tips = []
    for branch_id in request.crack_network.active_tip_ids:
        branch = request.crack_network.branch(branch_id)
        tip_xy = np.asarray(branch.tip, dtype=float)
        other_segments = tuple(
            segment
            for other_id, values in branch_segments.items()
            if other_id != branch_id
            for segment in values
        )
        nearest_other = min(
            (point_segment_distance(tip_xy, *segment) for segment in other_segments),
            default=math.inf,
        )
        nearest_junction = min(
            (float(np.linalg.norm(tip_xy - point)) for point in junctions),
            default=math.inf,
        )
        nearest_boundary = float(np.min(np.concatenate((tip_xy - domain_min, domain_max - tip_xy))))
        local_h = max(float(getattr(request.mesh, "hbar_tip", 0.0) or request.mesh.hbar), 1.0e-15)
        radii = tuple(sorted({
            float(request.contour_radius_m),
            max(3.0 * local_h, 0.75 * float(request.contour_radius_m)),
            max(3.0 * local_h, 0.50 * float(request.contour_radius_m)),
            max(3.0 * local_h, 0.25 * float(request.contour_radius_m)),
        }))
        directional = []
        for candidate in sorted(request.candidates_by_tip[branch_id], key=lambda item: item.candidate_id):
            contour_rows = []
            for radius in radii:
                _, _, info = compute_J_integral(
                    request.mesh, base["u"], base["sigma_gp"], base["psi_e_gp"],
                    request.damage, tip_xy, np.asarray(candidate.direction_xy),
                    request.material, ell=radius,
                    crack_segments=segments, exclude_radius=request.exclude_radius_m,
                )
                signed_radius = float(info.get("J_signed", info.get("J", 0.0)))
                another_crack = nearest_other <= radius
                contains_junction = nearest_junction <= radius
                boundary_intersection = nearest_boundary <= radius
                adequate_support = bool(info.get("n_active_elements", 0) > 0 and math.isfinite(signed_radius))
                reasons = []
                if another_crack: reasons.append("another_committed_crack_in_contour")
                if contains_junction: reasons.append("junction_in_contour")
                if boundary_intersection: reasons.append("specimen_boundary_in_contour")
                if not adequate_support: reasons.append("inadequate_finite_element_support")
                contour_rows.append({
                    "radius_m": radius, "signed_J_J_per_m2": signed_radius,
                    "another_committed_crack_intersects": another_crack,
                    "another_wake_intersects": another_crack,
                    "junction_intersects": contains_junction,
                    "specimen_boundary_intersects": boundary_intersection,
                    "adequate_finite_element_support": adequate_support,
                    "geometrically_valid": not reasons,
                    "invalid_reasons": reasons,
                    "integration": info,
                })
            valid_rows = [row for row in contour_rows if row["geometrically_valid"]]
            plateau = None
            for first, second in zip(valid_rows, valid_rows[1:]):
                scale = max(abs(first["signed_J_J_per_m2"]), abs(second["signed_J_J_per_m2"]), 1.0e-12)
                if abs(first["signed_J_J_per_m2"] - second["signed_J_J_per_m2"]) / scale <= 0.15:
                    plateau = (first, second)
                    break
            local_valid = plateau is not None
            selected_row = plateau[-1] if plateau else contour_rows[-1]
            # Keep the provider's established directional observable evaluated
            # at the requested contour radius.  The irreversible v10 -> v11
            # handoff parity contract is defined on this value.  The nested,
            # independently-valid local value is a separate kinetic
            # diagnostic and must not silently redefine transition parity.
            nominal_row = next(
                row for row in contour_rows
                if row["radius_m"] == float(request.contour_radius_m)
            )
            nominal_signed = float(nominal_row["signed_J_J_per_m2"])
            nominal_positive = max(nominal_signed, 0.0)
            local_signed = float(selected_row["signed_J_J_per_m2"])
            contract_radius = float(
                request.provider_contract_contour_radius_m
                if request.provider_contract_contour_radius_m is not None
                else request.contour_radius_m
            )
            if contract_radius == float(request.contour_radius_m):
                contract_row = nominal_row
            else:
                _, _, contract_info = compute_J_integral(
                    request.mesh, base["u"], base["sigma_gp"], base["psi_e_gp"],
                    request.damage, tip_xy, np.asarray(candidate.direction_xy),
                    request.material, ell=contract_radius,
                    crack_segments=segments, exclude_radius=request.exclude_radius_m,
                )
                contract_row = {
                    "signed_J_J_per_m2": float(
                        contract_info.get("J_signed", contract_info.get("J", 0.0))
                    ),
                    "integration": contract_info,
                }
            contract_signed = float(contract_row["signed_J_J_per_m2"])
            contract_positive = max(contract_signed, 0.0)
            Eprime = float(request.material.Eprime)
            directional.append({
                "candidate_id": candidate.candidate_id,
                "signed_J_J_per_m2": nominal_signed,
                "positive_J_J_per_m2": nominal_positive,
                "K_directional_Pa_sqrt_m": math.sqrt(Eprime * nominal_positive),
                "J_provider_contract_signed_J_per_m2": contract_signed,
                "J_provider_contract_positive_J_per_m2": contract_positive,
                "K_provider_contract_Pa_sqrt_m": math.sqrt(Eprime * contract_positive),
                "provider_contract_contour_radius_m": contract_radius,
                "provider_contract_auxiliary_direction_xy": list(candidate.direction_xy),
                "provider_contract_integration": contract_row["integration"],
                "J_local_signed_J_per_m2": local_signed,
                "local_contour_valid": local_valid,
                "local_J_valid": local_valid,
                "local_J_invalid_reason": None if local_valid else (
                    "no_numerically_converged_independent_contour"
                    if valid_rows else ";".join(sorted({reason for row in contour_rows for reason in row["invalid_reasons"]}))
                ),
                "J_contour_radius_m": selected_row["radius_m"],
                "nearest_other_crack_distance_m": None if not math.isfinite(nearest_other) else nearest_other,
                "nearest_junction_distance_m": None if not math.isfinite(nearest_junction) else nearest_junction,
                "nearest_specimen_boundary_distance_m": nearest_boundary,
                "another_committed_crack_intersects_J_domain": selected_row["another_committed_crack_intersects"],
                "another_wake_intersects_J_domain": selected_row["another_wake_intersects"],
                "local_contour_active_elements": int(selected_row["integration"].get("n_active_elements", 0)),
                "contour_diagnostics": selected_row["integration"],
                "nested_contour_diagnostics": contour_rows,
            })
        tips.append({
            "physical_tip_key": [_point(branch.tip), _point(_tip_direction(branch))],
            "tip_xy_m": list(branch.tip), "tip_direction": _tip_direction(branch).tolist(),
            "status": branch.status, "directional": directional,
        })
    tips.sort(key=lambda item: json.dumps(item["physical_tip_key"]))
    fingerprint_args = dict(
        network=request.crack_network, mesh=request.mesh, damage=request.damage,
        mechanical_configuration_fingerprint=request.mechanical_configuration_fingerprint,
        specimen_geometry=request.specimen_geometry,
        boundary_condition_identity=request.boundary_condition_identity,
        elastic_constants=request.elastic_constants, cluster_frame=request.cluster_frame,
        mpz_station_coordinates_m=request.mpz_station_coordinates_m,
        wake_station_coordinates_m=request.wake_station_coordinates_m,
        contour_definitions=request_contour_definitions(request),
    )
    fingerprint = topology_fingerprint(**fingerprint_args)
    responses = [_shared_unit_response(request, base, item) for item in request.shared_perturbations]
    applied_opening = float(Uy_top - Uy_bot)
    reaction_force = float(base["reaction_top"])
    apparent_compliance = (
        applied_opening / abs(reaction_force)
        if abs(reaction_force) > 1.0e-300 else None
    )
    return {
        "schema": SCHEMA, "kernel_provider_id": PROVIDER_ID,
        "branching_mode": "direct_fem",
        "maximum_fronts_supported": MAXIMUM_FRONTS_SUPPORTED,
        "coverage_kind": "exact_topology", "topology_fingerprint": fingerprint,
        "interpolation_permitted": False, "prior_kernel_family_required": False,
        "stochastic_trajectory_required": False,
        "material_parameter_option_required": False, "hazard_seed_required": False,
        "production_physics_modified": False,
        "base_equilibrium": {
            "displacement": np.asarray(base["u"]),
            "applied_displacement": applied_opening,
            "reaction_force": reaction_force,
            "apparent_compliance": apparent_compliance,
            "recoverable_potential_energy_J_per_m": energy,
        },
        "tips": tips, "signed_shared_cluster_response": responses,
        "shared_perturbation_solve_count": len(request.shared_perturbations),
    }


__all__ = [
    "LiveTopologyRequest", "MAXIMUM_FRONTS_SUPPORTED", "PROVIDER_ID", "PROVIDER_SEMANTICS_ID", "SCHEMA", "canonical_topology_payload",
    "evaluate_exact_topology", "request_contour_definitions", "topology_fingerprint",
]
