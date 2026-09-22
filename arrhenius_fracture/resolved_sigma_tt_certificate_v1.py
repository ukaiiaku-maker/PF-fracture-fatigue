"""Authenticated two-level certificate for resolved cavity-loop sources.

Validity is decided before scientific availability. Controlled records may
exercise the validator; only the builder below measures FEM data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from .embryo_growth_v1 import cavity_surface_emission_rates
from .piecewise_equilibrated_boundary_stress_v2 import (
    PiecewiseEquilibratedStressUnavailable,
    recover_piecewise_equilibrated_boundary_stress_v2,
)
from .topology_transaction_v11 import (
    _canonical_observation_fingerprint,
    complete_accepted_state_fingerprint,
    require_equilibrium_observables,
)

SCHEMA = "unified-2d.resolved-sigma-tt-two-level-certificate/2"
PASS = "PASS"
INVALID_COMPARISON = "INVALID_COMPARISON"
SCIENTIFICALLY_UNAVAILABLE = "SCIENTIFICALLY_UNAVAILABLE"
NOT_EVALUATED = "NOT_EVALUATED"
RELATIVE_CONVERGENCE_LIMIT = 0.05


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if is_dataclass(value):
        return {name: _plain(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, (tuple, list)): return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    return {"type": type(value).__qualname__, "state": _plain(getattr(value, "__dict__", str(value)))}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def authenticated_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(_plain(payload))
    result["authentication_sha256"] = _sha(result)
    return result


def _authentic(record: Mapping[str, Any]) -> bool:
    supplied = record.get("authentication_sha256")
    body = {k: v for k, v in record.items() if k != "authentication_sha256"}
    return isinstance(supplied, str) and supplied == _sha(body)


def _relative(first: float, second: float) -> float:
    return abs(float(second)-float(first))/max(abs(float(first)), abs(float(second)), 1e-300)


@dataclass(frozen=True)
class ResolvedSigmaTTTwoLevelCertificateV1:
    coarse_state_fingerprint: str | None
    fine_state_fingerprint: str | None
    coarse_mesh_fingerprint: str | None
    fine_mesh_fingerprint: str | None
    recovery_window_fingerprint: str | None
    source_coordinate_fingerprint: str | None
    coarse_sources: tuple[dict[str, Any], ...]
    fine_sources: tuple[dict[str, Any], ...]
    sigma_tt_relative_change: tuple[float, ...]
    activation_barrier_relative_change: tuple[float, ...]
    loop_rate_relative_change: tuple[float, ...]
    comparison_prerequisites: dict[str, bool]
    scientific_predicates: dict[str, bool]
    comparison_evidence: dict[str, Any]
    terminal_scientific_availability: str
    schema: str = SCHEMA

    @property
    def predicates(self) -> dict[str, bool]:
        return {**self.comparison_prerequisites, **self.scientific_predicates}

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["predicates"] = self.predicates
        return result


def _missing_certificate(reason: str) -> ResolvedSigmaTTTwoLevelCertificateV1:
    return ResolvedSigmaTTTwoLevelCertificateV1(
        None, None, None, None, None, None, (), (), (), (), (),
        {"evidence_present": False}, {}, {"pending_reason": reason}, NOT_EVALUATED,
    )


def _finite_pair(value: Any) -> bool:
    try:
        array = np.asarray(value, dtype=float)
        return array.shape == (2,) and bool(np.all(np.isfinite(array)))
    except (TypeError, ValueError):
        return False


def _source_shape(row: Mapping[str, Any]) -> bool:
    required = {
        "source_id", "orientation_rad", "nominal_coordinate_m",
        "realized_coordinate_m", "local_normal", "local_tangent",
        "physical_window", "mesh_support_identity", "available", "predicates",
        "sigma_tt_Pa", "activation_barrier_J", "loop_emission_rate_s",
        "authentication_sha256",
    }
    if not required.issubset(row) or not isinstance(row.get("source_id"), str): return False
    if not all(_finite_pair(row.get(k)) for k in (
        "nominal_coordinate_m", "realized_coordinate_m", "local_normal", "local_tangent"
    )): return False
    try:
        if not math.isfinite(float(row["orientation_rad"])): return False
        for key in ("sigma_tt_Pa", "activation_barrier_J", "loop_emission_rate_s"):
            if row[key] is not None and not math.isfinite(float(row[key])): return False
    except (TypeError, ValueError): return False
    return isinstance(row["physical_window"], Mapping) and isinstance(row["predicates"], Mapping)


def evaluate_resolved_sigma_tt_two_level_records(
    comparison_evidence: Mapping[str, Any] | None,
    coarse_sources: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None,
    fine_sources: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]] | None,
    *, convergence_limit: float = RELATIVE_CONVERGENCE_LIMIT,
) -> ResolvedSigmaTTTwoLevelCertificateV1:
    """Validate authenticated records, then and only then decide science."""
    if comparison_evidence is None or coarse_sources is None or fine_sources is None:
        return _missing_certificate("comparison evidence is missing")
    evidence = dict(comparison_evidence)
    coarse = tuple(dict(row) for row in coarse_sources)
    fine = tuple(dict(row) for row in fine_sources)
    coarse_ids = [row.get("source_id") for row in coarse]
    fine_ids = [row.get("source_id") for row in fine]
    coarse_by_id = {row.get("source_id"): row for row in coarse}
    fine_by_id = {row.get("source_id"): row for row in fine}
    required_evidence = {
        "coarse_state_fingerprint", "fine_state_fingerprint",
        "coarse_mesh_fingerprint", "fine_mesh_fingerprint",
        "coarse_mesh_observation_fingerprint", "fine_mesh_observation_fingerprint",
        "coarse_material_observation_fingerprint", "fine_material_observation_fingerprint",
        "coarse_equilibrium", "fine_equilibrium", "coarse_physical_identity",
        "fine_physical_identity", "common_parent_state_fingerprint",
        "refinement_lineage", "transfer_conservation", "boundary_representation_fingerprint",
        "authentication_sha256",
    }
    lineage = evidence.get("refinement_lineage", {})
    coarse_eq, fine_eq = evidence.get("coarse_equilibrium", {}), evidence.get("fine_equilibrium", {})
    transfer = evidence.get("transfer_conservation", {})
    record_shape = bool(coarse and fine) and all(_source_shape(row) for row in coarse+fine)
    unique_sets = (bool(coarse_ids) and len(coarse_ids) == len(set(coarse_ids))
                   and len(fine_ids) == len(set(fine_ids)) and set(coarse_ids) == set(fine_ids))

    def matched_sources(predicate) -> bool:
        return bool(unique_sets and all(predicate(coarse_by_id[key], fine_by_id[key]) for key in coarse_ids))

    coordinate_contract = matched_sources(lambda a, b:
        np.allclose(a["nominal_coordinate_m"], b["nominal_coordinate_m"], rtol=0.0, atol=1e-12)
        and np.allclose(a["realized_coordinate_m"], b["realized_coordinate_m"], rtol=0.0, atol=1e-12)
        and np.linalg.norm(np.asarray(a["nominal_coordinate_m"])-np.asarray(a["realized_coordinate_m"])) <= 1e-12
        and np.linalg.norm(np.asarray(b["nominal_coordinate_m"])-np.asarray(b["realized_coordinate_m"])) <= 1e-12
    ) if record_shape else False
    frame_contract = matched_sources(lambda a, b:
        math.isclose(float(a["orientation_rad"]), float(b["orientation_rad"]), rel_tol=0.0, abs_tol=1e-12)
        and np.allclose(a["local_normal"], b["local_normal"], rtol=0.0, atol=1e-12)
        and np.allclose(a["local_tangent"], b["local_tangent"], rtol=0.0, atol=1e-12)
        and abs(float(np.dot(a["local_normal"], a["local_tangent"]))) <= 1e-12
        and abs(float(np.linalg.norm(a["local_normal"]))-1.0) <= 1e-12
        and abs(float(np.linalg.norm(a["local_tangent"]))-1.0) <= 1e-12
    ) if record_shape else False
    window_contract = matched_sources(lambda a, b:
        _sha(a["physical_window"]) == _sha(b["physical_window"])
    ) if record_shape else False
    physical_match = (evidence.get("coarse_physical_identity") == evidence.get("fine_physical_identity")
                      and evidence.get("coarse_physical_identity") not in (None, {}))
    independent_meshes = (evidence.get("coarse_mesh_fingerprint") != evidence.get("fine_mesh_fingerprint")
                          and bool(evidence.get("coarse_mesh_fingerprint"))
                          and bool(evidence.get("fine_mesh_fingerprint")))
    equilibria_bound = (
        isinstance(coarse_eq, Mapping) and isinstance(fine_eq, Mapping)
        and coarse_eq.get("mesh_fingerprint") == evidence.get("coarse_mesh_observation_fingerprint")
        and fine_eq.get("mesh_fingerprint") == evidence.get("fine_mesh_observation_fingerprint")
        and coarse_eq.get("material_fingerprint") == evidence.get("coarse_material_observation_fingerprint")
        and fine_eq.get("material_fingerprint") == evidence.get("fine_material_observation_fingerprint")
        and coarse_eq.get("source_state_fingerprint") != fine_eq.get("source_state_fingerprint")
        and all(bool(coarse_eq.get(k)) and bool(fine_eq.get(k)) for k in
                ("source_state_fingerprint", "mesh_fingerprint", "material_fingerprint"))
    )
    lineage_bound = (isinstance(lineage, Mapping)
        and lineage.get("parent_mesh_fingerprint") == evidence.get("coarse_mesh_fingerprint")
        and lineage.get("current_mesh_fingerprint") == evidence.get("fine_mesh_fingerprint")
        and evidence.get("common_parent_state_fingerprint") == evidence.get("coarse_state_fingerprint")
        and bool(lineage.get("refined_parent_element_ids")))
    conservation = isinstance(transfer, Mapping) and bool(transfer) and all(v is True for v in transfer.values())
    prerequisites = {
        "evidence_present": required_evidence.issubset(evidence),
        "authenticated_evidence": _authentic(evidence) if required_evidence.issubset(evidence) else False,
        "authenticated_source_records": bool(coarse and fine) and all(_authentic(r) for r in coarse+fine),
        "nonempty_well_formed_sources": record_shape,
        "unique_matched_source_set": unique_sets,
        "matched_physical_problem": physical_match,
        "independent_meshes": independent_meshes,
        "independent_state_bound_equilibria": equilibria_bound,
        "common_parent_refinement_lineage": lineage_bound,
        "qualified_transfer_conservation": conservation,
        "registered_source_coordinate_contract": coordinate_contract,
        "registered_source_frame_contract": frame_contract,
        "matched_physical_recovery_windows": window_contract,
        "unchanged_boundary_representation": bool(evidence.get("boundary_representation_fingerprint")),
    }
    comparable = all(prerequisites.values())
    ordered_coarse = tuple(coarse_by_id[k] for k in coarse_ids) if unique_sets else coarse
    ordered_fine = tuple(fine_by_id[k] for k in coarse_ids) if unique_sets else fine
    measurement_keys = ("sigma_tt_Pa", "activation_barrier_J", "loop_emission_rate_s")
    measurements_available = comparable and all(
        a["available"] and b["available"] and all(a[k] is not None and b[k] is not None for k in measurement_keys)
        for a, b in zip(ordered_coarse, ordered_fine))
    sigma_change = tuple(_relative(a["sigma_tt_Pa"], b["sigma_tt_Pa"])
                         for a, b in zip(ordered_coarse, ordered_fine)) if measurements_available else ()
    barrier_change = tuple(_relative(a["activation_barrier_J"], b["activation_barrier_J"])
                           for a, b in zip(ordered_coarse, ordered_fine)) if measurements_available else ()
    rate_change = tuple(_relative(a["loop_emission_rate_s"], b["loop_emission_rate_s"])
                        for a, b in zip(ordered_coarse, ordered_fine)) if measurements_available else ()
    scientific = {} if not comparable else {
        "measured_at_both_levels": measurements_available,
        "all_level_recovery_predicates": measurements_available and all(
            bool(row["predicates"]) and all(bool(v) for v in row["predicates"].values())
            for row in ordered_coarse+ordered_fine),
        "sigma_tt_convergence": measurements_available and max(sigma_change, default=math.inf) <= convergence_limit,
        "activation_barrier_convergence": measurements_available and max(barrier_change, default=math.inf) <= convergence_limit,
        "loop_rate_convergence": measurements_available and max(rate_change, default=math.inf) <= convergence_limit,
    }
    terminal = INVALID_COMPARISON if not comparable else (PASS if all(scientific.values()) else SCIENTIFICALLY_UNAVAILABLE)
    window_fp = _sha([r["physical_window"] for r in ordered_coarse]) if record_shape else None
    coordinate_fp = _sha([{"source_id": r["source_id"], "nominal": r["nominal_coordinate_m"],
                           "realized": r["realized_coordinate_m"], "normal": r["local_normal"],
                           "tangent": r["local_tangent"]} for r in ordered_coarse]) if record_shape else None
    return ResolvedSigmaTTTwoLevelCertificateV1(
        evidence.get("coarse_state_fingerprint"), evidence.get("fine_state_fingerprint"),
        evidence.get("coarse_mesh_fingerprint"), evidence.get("fine_mesh_fingerprint"),
        window_fp, coordinate_fp, tuple(coarse), tuple(fine), sigma_change, barrier_change,
        rate_change, prerequisites, scientific, evidence, terminal)


def revalidate_resolved_sigma_tt_certificate_payload(
    payload: Mapping[str, Any],
) -> ResolvedSigmaTTTwoLevelCertificateV1:
    """Recompute every decision from a retained certificate's raw records."""
    if payload.get("schema") != SCHEMA:
        raise ValueError("resolved sigma-tt certificate schema mismatch")
    rebuilt = evaluate_resolved_sigma_tt_two_level_records(
        payload.get("comparison_evidence"), payload.get("coarse_sources"),
        payload.get("fine_sources"))
    # ``predicates`` is a redundant convenience projection.  It is included
    # in ``to_dict`` and therefore revalidated along with the terminal result.
    if _plain(rebuilt.to_dict()) != _plain(dict(payload)):
        raise ValueError("retained resolved sigma-tt certificate failed revalidation")
    return rebuilt


def _ordered_boundary_polygon(state: Any, edges: np.ndarray) -> list[list[float]]:
    adjacency: dict[int, list[int]] = {}
    for first, second in np.asarray(edges, dtype=int):
        adjacency.setdefault(int(first), []).append(int(second)); adjacency.setdefault(int(second), []).append(int(first))
    if not adjacency or any(len(v) != 2 for v in adjacency.values()):
        raise ValueError("cavity boundary is not a simple closed polygon")
    start = min(adjacency); order = [start]; previous = None; current = start
    while True:
        following = sorted(node for node in adjacency[current] if node != previous)[0]
        if following == start: break
        if following in order: raise ValueError("cavity boundary traversal is not unique")
        order.append(following); previous, current = current, following
    points = np.asarray(state.mesh.nodes, dtype=float)[order]
    changed = True
    while changed and len(points) > 3:
        changed = False; kept = []
        for index in range(len(points)):
            before, point, after = points[index-1], points[index], points[(index+1) % len(points)]
            first, second = point-before, after-point
            cross = abs(float(first[0]*second[1]-first[1]*second[0]))
            scale = max(float(np.linalg.norm(point-before)*np.linalg.norm(after-point)), 1e-300)
            if cross/scale <= 1e-12 and float(np.dot(point-before, after-point)) >= 0.0: changed = True
            else: kept.append(point)
        points = np.asarray(kept)
    return points.tolist()


def _physical_identity(state: Any, growth: Any, temperature_K: float) -> dict[str, Any]:
    from .unified_fracture_material_v5 import require_bound_identity
    equilibrium = require_equilibrium_observables(state)
    eq = equilibrium.to_dict() if hasattr(equilibrium, "to_dict") else dict(equilibrium)
    population = []
    for source in growth.cavity_emission_state.sources:
        values = asdict(source); values.pop("boundary_element_id", None); population.append(values)
    nodes = np.asarray(state.mesh.nodes); top = nodes[np.asarray(state.boundary.top_nodes, int)]; bottom = nodes[np.asarray(state.boundary.bot_nodes, int)]
    cavity = next(item for item in state.void_state.cavities if item.cavity_id == growth.cavity_id)
    canonical = lambda value: float(f"{float(value):.14g}")
    return _plain({
        "material_bundle_identity": require_bound_identity(state),
        "material_constitutive_identity": {"material": state.material, "elasticity_D": state.elasticity_D},
        "boundary_condition_geometry": {"top_y_m": canonical(np.mean(top[:, 1])), "bottom_y_m": canonical(np.mean(bottom[:, 1])),
            "x_extent_m": (canonical(nodes[:, 0].min()), canonical(nodes[:, 0].max())),
            "top_displacement_m": canonical(eq.get("applied_top_displacement_m")),
            "bottom_displacement_m": canonical(eq.get("applied_bottom_displacement_m"))},
        "temperature_K": float(temperature_K), "physical_time_s": state.junction_process_state.get("physical_time_s", 0.0),
        "crack_network": state.crack_network,
        "cavity_geometry": {"cavity_id": cavity.cavity_id, "parent_site_id": cavity.parent_site_id,
            "center_m": cavity.center_m, "radius_m": cavity.radius_m, "area_m2": cavity.area_m2},
        "process_zone": {"sharp_wake_model_id": state.sharp_wake_model_id, "tip_process_state": state.tip_process_state},
        "kinetic_owners": {"competition": state.competition, "rng_state": state.rng_state,
            "event_counters": state.event_counters, "void_sites": state.void_state.sites},
        "emission_parameter_row": growth.cavity_emission_state.parameter_row,
        "emission_population_state": population,
    })


def _level_record(state: Any, growth: Any, temperature_K: float) -> tuple[dict[str, Any], ...]:
    from .fem import assemble_mechanics
    from .voiding_production_v5 import _actual_cavity_boundary_edges, _bundle_from_state, fresh_sharp_front_engine
    from .vacancy_transport_phase_field_v2 import accepted_elastoplastic_stress_3d
    cavity = next(item for item in state.void_state.cavities if item.cavity_id == growth.cavity_id)
    edges = np.asarray(_actual_cavity_boundary_edges(state), int); boundary_nodes = np.unique(edges)
    _, _, sigma2d, *_ = assemble_mechanics(state.mesh, state.displacement, state.ep_gp, state.rho_gp,
        state.damage, state.elasticity_D, state.material, cohesive_network=state.cohesive_network)
    equilibrium = require_equilibrium_observables(state); width = float(np.ptp(np.asarray(state.mesh.nodes)[:, 0]))
    remote = 0.5*(abs(float(equilibrium["latest_top_reaction_N_per_m"]))+abs(float(equilibrium["latest_bottom_reaction_N_per_m"])))/width
    engine = fresh_sharp_front_engine(state.material, _bundle_from_state(state))
    damage = getattr(state.mesh, "element_damage_gp", None); solid = None if damage is None else np.asarray(damage, float) < 0.5
    stress3d = accepted_elastoplastic_stress_3d(state); records = []
    for source_index, source in enumerate(growth.cavity_emission_state.sources):
        if source.boundary_position_m is None or source.local_normal is None or source.local_tangent is None:
            raise ValueError(f"source {source.source_id!r} lacks a registered physical coordinate/frame")
        registered = np.asarray(source.boundary_position_m, float)
        distances = np.linalg.norm(np.asarray(state.mesh.nodes)[boundary_nodes]-registered, axis=1)
        node = int(boundary_nodes[int(np.argmin(distances))])
        if float(np.min(distances)) > 1e-12: raise ValueError(f"source {source.source_id!r} is absent from nested boundary")
        physical_window = {"source_coordinate_m": registered.tolist(), "local_normal": list(source.local_normal),
            "local_tangent": list(source.local_tangent), "process_length_m": float(engine.f.L_pz),
            "cavity_center_m": list(cavity.center_m), "cavity_radius_m": float(cavity.radius_m),
            "domain_relation": "ACTIVE_CAVITY_BOUNDARY_INTERIOR_SPECIMEN"}
        common = {"source_id": source.source_id, "orientation_rad": float(source.orientation_rad),
            "nominal_coordinate_m": registered.tolist(), "realized_coordinate_m": np.asarray(state.mesh.nodes)[node].tolist(),
            "local_normal": list(source.local_normal), "local_tangent": list(source.local_tangent), "physical_window": physical_window}
        try:
            recovered = recover_piecewise_equilibrated_boundary_stress_v2(
                nodes=state.mesh.nodes, elements=state.mesh.elems, stress_Pa=sigma2d, boundary_node=node,
                cavity_center_m=cavity.center_m, cavity_radius_m=cavity.radius_m, process_length_m=engine.f.L_pz,
                owned_boundary_edges=edges, remote_traction_scale_Pa=remote,
                state_fingerprint=complete_accepted_state_fingerprint(state), solid_element_mask=solid)
        except PiecewiseEquilibratedStressUnavailable as exc:
            records.append(authenticated_payload({**common, "available": False, "unavailable_reason": str(exc),
                "sigma_tt_Pa": None, "activation_barrier_J": None, "loop_emission_rate_s": None,
                "mesh_support_identity": {}, "predicates": {"recovery_available": False}})); continue
        limits = recovered["constraint_limits"]
        predicates = {
            "boundary_traction": recovered["normalized_boundary_traction"] <= limits["normalized_boundary_traction"],
            "fit_residual": recovered["stress_fit_residual"] <= recovered["stress_fit_residual_limit"],
            "design_condition": recovered["condition"] <= recovered["condition_limit"],
            "interface_continuity": recovered["interface_traction_jump_residual"] <= limits["interface_traction_jump"],
            "resultant_balance": recovered["resultant_residual"] <= limits["resultant"],
            "moment_balance": recovered["moment_residual"] <= limits["moment"],
            "mesh_quality": recovered["minimum_support_quality"] >= 0.05,
            "source_window_ownership": bool(recovered["window_identity"]["boundary_edge_ids"])}
        tensor = np.asarray(recovered["source_tensor_Pa"], float); eid = int(recovered["source_tensor_element_id"])
        rate, audit = cavity_surface_emission_rates(growth, stress_tensor_Pa=tensor,
            sigma_zz_Pa=float(stress3d["sigma_zz_Pa"][eid]), temperature_K=temperature_K)
        records.append(authenticated_payload({**common, "available": True,
            "mesh_support_identity": recovered["window_identity"], "sigma_tt_Pa": float(recovered["source_sigma_tt_Pa"]),
            "activation_barrier_J": float(audit["source_drives"][source_index][3]), "loop_emission_rate_s": float(rate[source_index]),
            "normalized_boundary_traction": recovered["normalized_boundary_traction"], "stress_fit_residual": recovered["stress_fit_residual"],
            "design_condition": recovered["condition"], "interface_traction_jump_residual": recovered["interface_traction_jump_residual"],
            "resultant_residual": recovered["resultant_residual"], "moment_residual": recovered["moment_residual"],
            "minimum_support_quality": recovered["minimum_support_quality"], "predicates": predicates}))
    return tuple(records)


def build_resolved_sigma_tt_two_level_certificate(
    coarse_state: Any, coarse_growth: Any, fine_state: Any, fine_growth: Any, *,
    temperature_K: float, refinement_provenance: Mapping[str, Any] | None = None,
) -> ResolvedSigmaTTTwoLevelCertificateV1:
    if refinement_provenance is None:
        evidence = authenticated_payload({"coarse_state_fingerprint": complete_accepted_state_fingerprint(coarse_state),
            "fine_state_fingerprint": complete_accepted_state_fingerprint(fine_state),
            "invalid_reason": "missing common-parent refinement provenance"})
        return evaluate_resolved_sigma_tt_two_level_records(evidence, (), ())
    from .adaptive_multitip_mesh_v11 import mesh_fingerprint
    from .voiding_production_v5 import _actual_cavity_boundary_edges
    coarse_eq = require_equilibrium_observables(coarse_state); fine_eq = require_equilibrium_observables(fine_state)
    coarse_eq_dict = coarse_eq.to_dict() if hasattr(coarse_eq, "to_dict") else dict(coarse_eq)
    fine_eq_dict = fine_eq.to_dict() if hasattr(fine_eq, "to_dict") else dict(fine_eq)
    coarse_polygon = _ordered_boundary_polygon(coarse_state, _actual_cavity_boundary_edges(coarse_state))
    fine_polygon = _ordered_boundary_polygon(fine_state, _actual_cavity_boundary_edges(fine_state))
    boundary_fp = _sha(coarse_polygon) if _sha(coarse_polygon) == _sha(fine_polygon) else None
    evidence = {
        "coarse_state_fingerprint": complete_accepted_state_fingerprint(coarse_state),
        "fine_state_fingerprint": complete_accepted_state_fingerprint(fine_state),
        "coarse_mesh_fingerprint": mesh_fingerprint(coarse_state.mesh), "fine_mesh_fingerprint": mesh_fingerprint(fine_state.mesh),
        "coarse_mesh_observation_fingerprint": _canonical_observation_fingerprint({"nodes": coarse_state.mesh.nodes, "elements": coarse_state.mesh.elems}),
        "fine_mesh_observation_fingerprint": _canonical_observation_fingerprint({"nodes": fine_state.mesh.nodes, "elements": fine_state.mesh.elems}),
        "coarse_material_observation_fingerprint": _canonical_observation_fingerprint({"material": coarse_state.material, "elasticity_D": coarse_state.elasticity_D}),
        "fine_material_observation_fingerprint": _canonical_observation_fingerprint({"material": fine_state.material, "elasticity_D": fine_state.elasticity_D}),
        "coarse_equilibrium": coarse_eq_dict, "fine_equilibrium": fine_eq_dict,
        "coarse_physical_identity": _physical_identity(coarse_state, coarse_growth, temperature_K),
        "fine_physical_identity": _physical_identity(fine_state, fine_growth, temperature_K),
        "common_parent_state_fingerprint": refinement_provenance.get("common_parent_state_fingerprint"),
        "refinement_lineage": refinement_provenance.get("refinement_lineage", {}),
        "transfer_conservation": refinement_provenance.get("transfer_conservation", {}),
        "boundary_representation_fingerprint": boundary_fp}
    return evaluate_resolved_sigma_tt_two_level_records(authenticated_payload(evidence),
        _level_record(coarse_state, coarse_growth, temperature_K), _level_record(fine_state, fine_growth, temperature_K))


__all__ = ["ResolvedSigmaTTTwoLevelCertificateV1", "SCHEMA", "PASS", "INVALID_COMPARISON",
    "SCIENTIFICALLY_UNAVAILABLE", "NOT_EVALUATED", "authenticated_payload",
    "evaluate_resolved_sigma_tt_two_level_records",
    "revalidate_resolved_sigma_tt_certificate_payload",
    "build_resolved_sigma_tt_two_level_certificate"]
