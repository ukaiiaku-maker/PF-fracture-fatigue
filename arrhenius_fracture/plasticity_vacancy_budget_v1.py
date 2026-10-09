"""Geometry-owned spatial and vacancy-economy audits for bounded bulk holds."""
from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .bulk_plasticity_v104 import vacancy_source_diagnostics
from .unified_fracture_material_v5 import front_config
from .voiding_production_v5 import _bundle_from_state

MECHANISM_INTERPRETATION = (
    "COARSE_GRAINED_JOG_INTERSECTION_AND_NONCONSERVATIVE_PLASTIC_SOURCE"
)
REGIONS = ("TIP_MPZ", "NEAR_TIP_ANNULUS", "FORWARD_LIGAMENT", "REMOTE_BULK")


def geometry_region_masks(state: Any) -> Mapping[str, np.ndarray]:
    centers = np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)].mean(axis=1)
    branch = state.crack_network.branch(state.crack_network.active_tip_ids[0])
    tip = np.asarray(branch.tip, float)
    direction = tip-np.asarray(branch.path[-2], float)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-300)
    normal = np.array((-direction[1], direction[0]))
    relative = centers-tip
    distance = np.linalg.norm(relative, axis=1)
    axial = relative@direction
    lateral = np.abs(relative@normal)
    length = float(front_config(_bundle_from_state(state)).L_pz)
    tip_mask = distance <= length
    annulus = (distance > length) & (distance <= 3.0*length)
    forward = (~tip_mask) & (~annulus) & (axial > 0.0) & (lateral <= 3.0*length)
    remote = ~(tip_mask | annulus | forward)
    masks = {"TIP_MPZ": tip_mask, "NEAR_TIP_ANNULUS": annulus,
             "FORWARD_LIGAMENT": forward, "REMOTE_BULK": remote}
    if not np.all(sum(mask.astype(int) for mask in masks.values()) == 1):
        raise RuntimeError("spatial region ownership is not an exact partition")
    return MappingProxyType(masks)


def interval_spatial_budget(
    previous: Any, mechanical: Any, transported: Any, bulk_audit: Mapping[str, Any],
    duration_s: float,
) -> Mapping[str, Any]:
    masks = geometry_region_masks(mechanical)
    area = np.asarray(mechanical.mesh.area_e)
    info = bulk_audit["constitutive"]
    epdot = np.asarray(bulk_audit["dot_ep_gp_s"])
    dep = epdot*float(duration_s)
    work = np.asarray(bulk_audit["accepted_endpoint_path_plastic_work_gp_J_m3"])
    rho = np.asarray(mechanical.rho_gp)
    transport0 = previous.vacancy_transport_state_v2
    transport1 = transported.vacancy_transport_state_v2
    parent = np.asarray(transport1.parent_mechanics_element_ids, int)
    transport_area = np.asarray(transport1.transport_mesh.area_e)
    ratio = float(transport1.parameter_row.inventory_ratio)
    source = vacancy_source_diagnostics(previous, mechanical, duration_s)
    source_total = np.asarray(source["total_rate_gp_s"])
    source_base = np.asarray(source["baseline_rate_gp_s"])
    source_rho = np.asarray(source["dislocation_density_enhancement_rate_gp_s"])
    climb_net = np.asarray(transport1.last_solver_audit[
        "climb_net_fraction_area_by_element"
    ])*ratio
    rows = {}
    for name, mask in masks.items():
        tm = mask[parent]
        rows[name] = {
            "area_m2": float(np.sum(area[mask])),
            "area_integrated_epdot_m2_s": float(np.sum(area[mask]*epdot[mask])),
            "area_integrated_accepted_dep_m2": float(np.sum(area[mask]*dep[mask])),
            "plastic_work_J_per_m": float(np.sum(area[mask]*work[mask])),
            "area_integrated_rho": float(np.sum(area[mask]*rho[mask])),
            "rho_storage_rate_integral_s-1": float(np.sum(area[mask]*np.asarray(
                info["rho_storage_rate_gp_m2_s"])[mask])),
            "rho_dynamic_annihilation_rate_integral_s-1": float(np.sum(area[mask]*np.asarray(
                info["rho_dynamic_annihilation_rate_gp_m2_s"])[mask])),
            "rho_static_recovery_rate_integral_s-1": float(np.sum(area[mask]*np.asarray(
                info["rho_static_recovery_rate_gp_m2_s"])[mask])),
            "vacancy_source_fraction_area_per_s": float(np.sum(
                transport_area[tm]*source_total[tm])),
            "vacancy_source_baseline_fraction_area_per_s": float(np.sum(
                transport_area[tm]*source_base[tm])),
            "vacancy_source_rho_enhancement_fraction_area_per_s": float(np.sum(
                transport_area[tm]*source_rho[tm])),
            "vacancy_inventory_area_m2": float(np.sum(
                transport_area[tm]*np.asarray(transport1.vacancy_fraction_gp)[tm])*ratio),
            "climb_removal_area_m2": float(np.sum(np.maximum(climb_net[tm], 0.0))),
            "crack_surface_loss_area_m2": 0.0,
            "external_surface_loss_area_m2": 0.0,
        }
    flux = transport1.last_solver_audit["face_flux_fraction_area_per_s"]
    from .vacancy_transport_phase_field_v2 import SurfaceKind
    for face in transport1.surface_registry.faces:
        if face.kind not in {SurfaceKind.CRACK_FREE_SURFACE,
                             SurfaceKind.MERGED_CAVITY_CRACK_WAKE,
                             SurfaceKind.EXTERNAL_FREE_SURFACE}:
            continue
        owner = int(parent[int(face.owner_elements[0])])
        region = next(name for name, mask in masks.items() if mask[owner])
        loss = max(float(flux[face.face_id]), 0.0)*float(duration_s)*ratio
        key = ("external_surface_loss_area_m2"
               if face.kind == SurfaceKind.EXTERNAL_FREE_SURFACE
               else "crack_surface_loss_area_m2")
        rows[region][key] += loss
    total_source = math.fsum(row["vacancy_source_fraction_area_per_s"] for row in rows.values())
    outside = total_source-rows["TIP_MPZ"]["vacancy_source_fraction_area_per_s"]
    return MappingProxyType({
        "mechanism_interpretation": MECHANISM_INTERPRETATION,
        "explicit_jog_density_source": "NOT_RESOLVED",
        "independent_jog_source_added": False,
        "regions": rows,
        "bulk_source_fraction_outside_MPZ": 0.0 if total_source <= 0.0 else outside/total_source,
        "remote_source_fraction": 0.0 if total_source <= 0.0 else
            rows["REMOTE_BULK"]["vacancy_source_fraction_area_per_s"]/total_source,
        "source_partition_residual": total_source-source["integrated_source_fraction_area_per_s"],
        "transport_time_before_s": transport0.transport_time_s,
        "transport_time_after_s": transport1.transport_time_s,
    })


def vacancy_economy(initial: Any, final: Any) -> Mapping[str, float]:
    first = initial.vacancy_transport_state_v2
    last = final.vacancy_transport_state_v2
    delta = {key: float(last.ledgers[key]-first.ledgers[key]) for key in last.ledgers}
    area = np.asarray(last.transport_mesh.area_e)
    ratio = float(last.parameter_row.inventory_ratio)
    field = float(np.sum(area*np.asarray(last.vacancy_fraction_gp))*ratio)
    field0 = float(np.sum(np.asarray(first.transport_mesh.area_e)
                          *np.asarray(first.vacancy_fraction_gp))*ratio)
    generated = delta["plastic_production_area_m2"]
    retained_owned = max(field-field0, 0.0)+delta["reserved_embryo_inventory_area_m2"] \
        +delta["subgrid_cavity_inventory_area_m2"]+delta["active_cavity_inventory_area_m2"]
    lhs = (field+float(last.ledgers["reserved_embryo_inventory_area_m2"])
           +float(last.ledgers["subgrid_cavity_inventory_area_m2"])
           +float(last.ledgers["active_cavity_inventory_area_m2"])
           +float(last.ledgers["crack_surface_absorption_area_m2"])
           +float(last.ledgers["external_surface_absorption_area_m2"])
           +float(last.ledgers["dislocation_climb_annihilation_area_m2"])
           -float(last.ledgers["dislocation_climb_emission_area_m2"])
           +float(last.ledgers["embryo_healing_annihilation_area_m2"]))
    rhs = float(last.ledgers["initial_owned_inventory_area_m2"])+float(
        last.ledgers["plastic_production_area_m2"])
    return MappingProxyType({
        "A_plastic_generated": generated,
        "A_climb_removed": delta["dislocation_climb_annihilation_area_m2"],
        "A_crack_surface_removed": max(delta["crack_surface_absorption_area_m2"], 0.0),
        "A_external_surface_removed": max(delta["external_surface_absorption_area_m2"], 0.0),
        "A_crack_surface_emitted": max(-delta["crack_surface_absorption_area_m2"], 0.0),
        "A_external_surface_emitted": max(-delta["external_surface_absorption_area_m2"], 0.0),
        "A_field_owned": field,
        "A_field_retained_above_initial": max(field-field0, 0.0),
        "A_site_captured": delta["subgrid_capture_area_m2"],
        "A_embryo_reserved": delta["reserved_embryo_inventory_area_m2"],
        "A_cavity_absorbed": delta["active_cavity_absorption_area_m2"],
        "conservation_residual": lhs-rhs,
        "vacancy_retention_efficiency": (0.0 if generated <= 0.0
                                          else retained_owned/generated),
    })


__all__ = ["MECHANISM_INTERPRETATION", "REGIONS", "geometry_region_masks",
           "interval_spatial_budget", "vacancy_economy"]
