"""Executable one-void trajectory owned by the V12 production FEM state."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from .crack_network_v11 import CrackBranchState, CrackNetworkState, ROOT_BRANCH_ID
from .directional_competition_v11 import (
    CleavageCandidate, DirectionalCompetitionState, commit_directional_interval,
    construct_action_proposals, preview_directional_interval,
    select_temporal_or_degenerate_proposal, tungsten_cleavage_candidates,
)
from .explicit_cavity_v5 import (
    build_explicit_hole_mesh, build_source_conforming_hole_mesh,
    fill_explicit_hole_mesh, triangle_intersects_open_disk,
)
from .cavity_source_shape_regular_mesh_v5 import (
    MESH_CONTRACT_ID as SHAPE_REGULAR_MESH_V5_ID,
    build_shape_regular_source_patch_hole_mesh,
)
from .mesh import BoundaryData, rebuild_tri_mesh
from .fem import assemble_mechanics, plane_strain_D
from .unified_fracture_material_v5 import (
    UnifiedFractureMaterialBundle, bind_identity, canonical_front_engine,
    elastic_material, require_bound_identity,
)
from .hazard_energy_event_gate_v10230 import hazard_resistance_J_per_m2
from .causal_sharp_wake_v11 import element_damage
from .mechanically_separating_sharp_wake_v12 import (
    certification_arcs, independent_intact_path_certificate,
)
from .sharp_wake_backend_v12 import V12_MODEL_ID
from .topology_transaction_v11 import (
    LiveFEMTopologyState, TopologyArm, apply_v12_production_trial_geometry,
    complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
    execute_topology_trial, initialize_mechanically_separating_v12,
    remesh_mechanically_separating_v12,
)
from .voiding_v5 import (
    Cavity2D, HazardClock, ProductionVoidState, VoidPhase, VoidSite, VoidingConfig,
    advance_site, arrhenius_rates, create_subgrid_cavity, grow_cavity_2d,
    grow_cavity_from_rate, growth_time_to_radius_exact,
    promote_cavity, replace_cavity, update_cavity_growth,
)

SCHEMA = "v12.production-one-void-trajectory/5"
FRONT_ENGINE_STATE_SCHEMA = "v5.downstream-child-front-engine-state/2"
_FRONT_ENGINE_STATE_FIELDS = ("N_em", "B", "a_adv", "n_adv", "W_emit", "t", "K_prev")


def _bundle_from_state(state) -> UnifiedFractureMaterialBundle:
    require_bound_identity(state)
    return UnifiedFractureMaterialBundle(
        **dict(state.junction_process_state["unified_material_bundle"])
    )


def vacancy_coupled_void_rates(
    state, config, *, temperature_K, stress_tensor_Pa, center_m,
    equivalent_plastic_strain_rate_gp, radius_m=None,
):
    """Evaluate vacancy-sensitive birth/growth from one accepted FEM state."""
    from .vacancy_coupling_v1 import (
        REFERENCE_VACANCY_ROW, deterministic_capture_patch, local_vacancy_fraction,
        vacancy_attachment_frequency_s, vacancy_growth_rates_m_s,
    )
    bundle = _bundle_from_state(state)
    if bundle.vacancy_row_id != REFERENCE_VACANCY_ROW.row_id:
        raise RuntimeError("accepted state does not own the common vacancy row")
    process_length = float(fresh_sharp_front_engine(state.material, bundle).f.L_pz)
    capture_radius = process_length + (0.0 if radius_m is None else float(radius_m))
    patch = deterministic_capture_patch(state, center_m, capture_radius)
    local_c = local_vacancy_fraction(state, patch)
    plastic = np.asarray(equivalent_plastic_strain_rate_gp, dtype=float)
    if plastic.shape != (state.mesh.ne,) or np.any(~np.isfinite(plastic)) or np.any(plastic < 0.0):
        raise ValueError("accepted equivalent plastic rate must be finite P0 data")
    area = np.asarray(state.mesh.area_e)[patch]
    local_plastic = float(np.sum(area * plastic[patch]) / np.sum(area))
    attachment = vacancy_attachment_frequency_s(
        temperature_K, local_c, state.material.b, REFERENCE_VACANCY_ROW,
    )
    growth = None
    if radius_m is not None:
        growth = vacancy_growth_rates_m_s(
            temperature_K=temperature_K, radius_m=radius_m,
            vacancy_fraction_site=local_c,
            equivalent_plastic_rate_s=local_plastic,
            row=REFERENCE_VACANCY_ROW,
        )
    rates = arrhenius_rates(
        config, temperature_K=temperature_K, stress_tensor_Pa=stress_tensor_Pa,
        vacancy_attachment_frequency_s=attachment, vacancy_growth=growth,
    )
    return rates, {
        "vacancy_row_id": REFERENCE_VACANCY_ROW.row_id,
        "vacancy_row_fingerprint": REFERENCE_VACANCY_ROW.fingerprint,
        "capture_patch_element_ids": patch.tolist(),
        "c_v_site": local_c,
        "equivalent_plastic_strain_rate_site_s": local_plastic,
        "vacancy_attachment_frequency_s": attachment,
        **({} if growth is None else growth),
    }


def advance_site_on_common_vacancy_clock(
    previous_accepted_state, accepted_mechanical_state, config, *, bundle,
    duration_s, temperature_K, stress_tensor_Pa,
):
    """Advance vacancy and the earliest site clock on one physical interval.

    Mechanics is supplied by the ordinary accepted caller.  This routine does
    not invoke plasticity; it derives its fixed interval rate from the two
    accepted mechanical states and advances every owned site clock by the same
    elapsed time.
    """
    from .vacancy_coupling_v1 import (
        advance_vacancy_with_fixed_rates, deterministic_capture_patch,
        equilibrium_vacancy_fraction, equivalent_plastic_strain_rate_s,
        integrated_capture_vacancy_time, local_vacancy_fraction,
        vacancy_annihilation_rate_s,
        vacancy_attachment_frequency_s, vacancy_production_rate_s,
        vacancy_weighted_first_passage_time,
    )
    from .unified_fracture_material_v5 import front_config

    duration = float(duration_s)
    if duration <= 0.0 or not math.isfinite(duration):
        raise ValueError("common accepted interval must be finite and positive")
    owned_bundle = _bundle_from_state(accepted_mechanical_state)
    if owned_bundle.bundle_hash != bundle.bundle_hash:
        raise RuntimeError("accepted mechanics and vacancy scheduler material identities differ")
    if _bundle_from_state(previous_accepted_state).bundle_hash != bundle.bundle_hash:
        raise RuntimeError("previous accepted mechanics has a different material identity")
    site = accepted_mechanical_state.void_state.sites[0]
    plastic_rate = equivalent_plastic_strain_rate_s(
        previous_accepted_state.ep_gp, accepted_mechanical_state.ep_gp, duration,
    )
    rho = np.asarray(accepted_mechanical_state.rho_gp, dtype=float)
    rho_reference = float(bundle.fracture_row["rho_forest_floor_m2"])
    process_length = float(front_config(bundle).L_pz)
    production = vacancy_production_rate_s(
        plastic_rate, rho, rho_reference_m2=rho_reference,
    )
    sink = vacancy_annihilation_rate_s(
        temperature_K, rho, process_zone_length_m=process_length,
    )
    equilibrium = equilibrium_vacancy_fraction(temperature_K)
    patch = deterministic_capture_patch(
        previous_accepted_state, site.center_m, process_length,
    )
    unit_attachment = vacancy_attachment_frequency_s(
        temperature_K, 1.0, accepted_mechanical_state.material.b,
    )
    unit_rates = arrhenius_rates(
        config, temperature_K=temperature_K, stress_tensor_Pa=stress_tensor_Pa,
        normal_xy=site.normal_xy, vacancy_attachment_frequency_s=unit_attachment,
    )
    event_time = None
    event_kind = None
    if site.phase == VoidPhase.AVAILABLE_SITE:
        remaining_action = max(site.birth.threshold - float(site.birth.action_exact()), 0.0)
        event_time = vacancy_weighted_first_passage_time(
            previous_accepted_state, patch, production_rate_s=production,
            annihilation_rate_s=sink, equilibrium_fraction=equilibrium,
            rate_per_vacancy_s=unit_rates["birth_s"] * site.candidate_weight,
            action_remaining=remaining_action, maximum_duration_s=duration,
        )
        event_kind = "BIRTH_HIT" if event_time is not None else None
    elif site.phase == VoidPhase.EMBRYO:
        stabilization = site.stabilization.crossing_time(unit_rates["stabilization_s"])
        healing = site.healing.crossing_time(unit_rates["healing_s"])
        crossing = min(stabilization, healing)
        if math.isfinite(crossing) and crossing <= duration:
            event_time = crossing
            event_kind = "HEALED" if healing <= stabilization else "STABILIZED"
    elapsed = duration if event_time is None else float(event_time)
    evolved, vacancy_audit = advance_vacancy_with_fixed_rates(
        previous_accepted_state, accepted_mechanical_state,
        duration_s=elapsed, temperature_K=temperature_K,
        equivalent_plastic_rate_s=plastic_rate, production_rate_s=production,
        annihilation_rate_s=sink,
    )
    integrated_site_c = integrated_capture_vacancy_time(
        previous_accepted_state, patch, production_rate_s=production,
        annihilation_rate_s=sink, equilibrium_fraction=equilibrium,
        duration_s=elapsed,
    )
    average_site_c = (
        integrated_site_c / elapsed if elapsed > 0.0
        else local_vacancy_fraction(previous_accepted_state, patch)
    )
    rates = dict(unit_rates)
    rates["birth_s"] = unit_rates["birth_s"] * average_site_c
    void_state, events = advance_site(
        evolved.void_state, site.site_id, elapsed, rates=rates,
    )
    result = _synchronize_legacy_vacancy_observable(
        replace(evolved, void_state=void_state)
    )
    return result, {
        "material_bundle_id": owned_bundle.bundle_id,
        "material_bundle_sha256": owned_bundle.bundle_hash,
        "plasticity_identity": dict(require_bound_identity(accepted_mechanical_state))[
            "plasticity_fingerprint"
        ],
        "elapsed_time_s": elapsed,
        "scheduled_event": event_kind,
        "emitted_events": events,
        "capture_patch_element_ids": patch.tolist(),
        "integrated_site_vacancy_time_s": integrated_site_c,
        "average_site_vacancy_fraction": average_site_c,
        "accepted_equivalent_plastic_rate_s": plastic_rate,
        "vacancy_production_rate_s": production,
        "vacancy_annihilation_rate_s": sink,
        "rates": rates,
        "vacancy": vacancy_audit,
    }


def advance_production_vacancy_transport_interval(
    previous_accepted_state, accepted_mechanical_state, *, duration_s,
    external_event_times_s=None, directional_rate_by_candidate_s=None,
    failure_injector=None, transport_linear_solver_provider=None,
):
    """Advance the state-owned V2 field on one accepted global event clock.

    Mechanics is never invoked here.  The previous/current accepted plastic
    fields supply the only production increment; ordinary scheduler crossings
    are previewed by their existing owners and supplied in
    ``external_event_times_s``.
    """
    from .vacancy_coupling_v1 import equivalent_plastic_strain_rate_s
    from .vacancy_transport_phase_field_v2 import (
        accepted_elastoplastic_stress_3d, compete_on_common_transport_clock,
    )
    transport = previous_accepted_state.vacancy_transport_state_v2
    if transport is None or transport.parameter_row is None or transport.transport_mesh is None:
        raise RuntimeError("production accepted state does not own complete V2 vacancy transport")
    if accepted_mechanical_state.vacancy_transport_state_v2 is not transport:
        if (accepted_mechanical_state.vacancy_transport_state_v2 is None
                or accepted_mechanical_state.vacancy_transport_state_v2.parameter_row_fingerprint
                != transport.parameter_row_fingerprint):
            raise RuntimeError("accepted mechanical trial changed vacancy transport ownership")
    duration = float(duration_s)
    mechanics_rate = equivalent_plastic_strain_rate_s(
        previous_accepted_state.ep_gp, accepted_mechanical_state.ep_gp, duration,
    )
    parent = np.asarray(transport.parent_mechanics_element_ids, int)
    stress3d = accepted_elastoplastic_stress_3d(accepted_mechanical_state)
    mean_stress = np.asarray(stress3d["mean_stress_Pa"])[parent]
    rho = np.asarray(accepted_mechanical_state.rho_gp)[parent]
    epdot = mechanics_rate[parent]
    row = transport.parameter_row
    diffusivity = row.vacancy_diffusivity_prefactor_m2_s*math.exp(
        -row.vacancy_migration_energy_eV*1.602176634e-19
        /(1.380649e-23*transport.vacancy_temperature_K)
    )
    centers = np.asarray(accepted_mechanical_state.mesh.nodes)[
        np.asarray(accepted_mechanical_state.mesh.elems)
    ].mean(axis=1)
    config = VoidingConfig(enabled=True)
    updated_sites = []
    for site in transport.site_states:
        eid = int(np.argmin(np.linalg.norm(centers-np.asarray(site.center_xy_m), axis=1)))
        void_sites = (() if accepted_mechanical_state.void_state is None
                      else accepted_mechanical_state.void_state.sites)
        void_site = next((item for item in void_sites if item.site_id == site.site_id), None)
        normal = np.asarray((1.0, 0.0) if void_site is None else void_site.normal_xy, float)
        normal /= np.linalg.norm(normal)
        tangent = np.array((-normal[1], normal[0]))
        tensor = np.array(((stress3d["sigma_xx_Pa"][eid], stress3d["sigma_xy_Pa"][eid]),
                           (stress3d["sigma_xy_Pa"][eid], stress3d["sigma_yy_Pa"][eid])))
        hydro = stress3d["mean_stress_Pa"][eid]
        opening = float(normal@tensor@normal)
        shear = float(tangent@tensor@normal)
        work = (
            config.hydrostatic_work_coefficient*hydro*config.birth_hydrostatic_volume_m3
            + config.normal_opening_work_coefficient*max(opening, 0.0)
              * config.birth_normal_area_m2*config.birth_normal_separation_m
            + config.signed_shear_work_coefficient*shear*config.birth_shear_volume_m3
        )
        updated_sites.append(replace(
            site,
            capture_rate_area_per_s=row.site_capture_factor*diffusivity,
            birth_barrier_J=config.birth_barrier_J,
            birth_stress_work_J=work,
        ))
    prepared = replace(transport, site_states=tuple(updated_sites))
    competition = compete_on_common_transport_clock(
        prepared, transport.transport_mesh, transport.surface_registry,
        row=row, maximum_duration_s=duration,
        accepted_equivalent_plastic_strain_rate_s=epdot,
        accepted_dislocation_density_m2=rho,
        accepted_mean_stress_Pa=mean_stress,
        external_event_times_s=dict(external_event_times_s or {}),
        failure_injector=failure_injector,
        linear_solver_provider=transport_linear_solver_provider,
    )
    if failure_injector is not None:
        failure_injector("topology_commit", previous_accepted_state)
    directional_rates = dict(directional_rate_by_candidate_s or {})
    directional = accepted_mechanical_state.competition
    if directional_rates:
        from .directional_competition_v11 import (
            commit_directional_interval, preview_directional_interval,
        )
        expected = {item.candidate_id for item in directional.hazard_states}
        if set(directional_rates) != expected:
            raise RuntimeError("directional losing-clock rates do not match accepted ownership")
        elapsed = float(competition["elapsed_time_s"])
        hazards = tuple(commit_directional_interval(
            hazard,
            preview_directional_interval(
                hazard, lambda_per_s=float(directional_rates[hazard.candidate_id]),
                start_time_s=transport.transport_time_s, duration_s=elapsed,
                maximum_completed_events=1,
            ),
        ) for hazard in directional.hazard_states)
        directional = replace(directional, hazard_states=hazards)
    committed = replace(
        accepted_mechanical_state, competition=directional,
        vacancy_transport_state_v2=competition["transport_state"],
    )
    return committed, {
        "elapsed_time_s": competition["elapsed_time_s"],
        "event": competition["event"],
        "tie_group": competition["tie_group"],
        "accepted_plastic_increment_rate_max_s": float(np.max(epdot)),
        "mean_stress_source": stress3d["source"],
        "all_losing_clocks_advance_s": competition["all_losing_clocks_advance_s"],
        "directional_losing_clocks_advanced": bool(directional_rates),
    }


def couple_active_cavity_flux_to_geometry(
    previous_accepted_state, transported_state, hole, *, cavity_id,
    duration_s, accepted_equivalent_plastic_strain_rate_s,
    plastic_accommodation_coefficient=1.0,
    direct_free_volume_rate_area_m2_s=0.0, failure_injector=None,
):
    """Commit one-way cavity absorption, boundary motion, remesh and mechanics atomically.

    The V2 surface integral is the sole vacancy supply.  The existing
    series-limited transport/accommodation law determines boundary speed; any
    absorbed inventory not yet accommodated remains owned by the cavity.
    """
    # This compatibility entry point is now a strict delegate.  It may not
    # independently update radius, inventories, plasticity, or remesh.
    from .embryo_growth_v1 import (
        EmbryoGrowthProcessOwner, source_local_fem_tensors,
        transport_flux_certificate,
    )
    growth = next((x for x in transported_state.void_state.embryo_growth_states
                   if x.cavity_id == cavity_id), None)
    if growth is None:
        raise RuntimeError("active cavity growth requires the authoritative embryo-growth owner")
    if float(direct_free_volume_rate_area_m2_s) != 0.0:
        raise RuntimeError("parallel direct-growth scalar is forbidden; commit through the growth owner")
    if failure_injector is not None:
        failure_injector("cavity_update", previous_accepted_state)
    certificate = transport_flux_certificate(
        previous_accepted_state.vacancy_transport_state_v2,
        transported_state.vacancy_transport_state_v2,
        owner_id=cavity_id, owner_kind="ACTIVE_CAVITY")
    site = next(x for x in transported_state.void_state.sites if x.site_id == growth.site_id)
    centers = np.asarray(transported_state.mesh.nodes)[np.asarray(transported_state.mesh.elems)].mean(axis=1)
    eid = int(np.argmin(np.linalg.norm(centers-np.asarray(site.center_m), axis=1)))
    local_h = math.sqrt(float(transported_state.mesh.area_e[eid]))
    coefficient = float(plastic_accommodation_coefficient)
    epdot = float(np.asarray(accepted_equivalent_plastic_strain_rate_s).reshape(-1)[0])
    bulk_rate = 2.0*math.pi*growth.radius_m**2*coefficient*max(epdot, 0.0)

    def remesh_delegate(mechanical, before_growth, after_growth):
        crack_path = mechanical.crack_network.branch(ROOT_BRANCH_ID).path
        moved_hole = _grow_hole_boundary(hole, after_growth.radius_m, crack_path_m=crack_path)
        return remesh_cavity(
            mechanical, moved_hole, mechanical.void_state,
            "authoritative-embryo-growth-cavity-remesh",
            failure_stage=("transport_cavity_remesh" if failure_injector is not None else None))

    tensors, zz = source_local_fem_tensors(transported_state, growth)
    def solve_delegated_transport_packet(packet_live, packet_dt):
        from .vacancy_transport_phase_field_v2 import (
            accepted_elastoplastic_stress_3d, solve_transport_interval,
        )
        packet_before = packet_live.vacancy_transport_state_v2
        parent = np.asarray(packet_before.parent_mechanics_element_ids, int)
        stress3d = accepted_elastoplastic_stress_3d(packet_live)
        packet_after = solve_transport_interval(
            packet_before, packet_before.transport_mesh,
            packet_before.surface_registry, row=packet_before.parameter_row,
            duration_s=packet_dt,
            accepted_equivalent_plastic_strain_rate_s=np.full(
                len(parent), max(epdot, 0.0)),
            accepted_dislocation_density_m2=np.asarray(packet_live.rho_gp)[parent],
            accepted_mean_stress_Pa=np.asarray(stress3d["mean_stress_Pa"])[parent])
        return (replace(packet_live, vacancy_transport_state_v2=packet_after),
                packet_before, packet_after)
    owner = EmbryoGrowthProcessOwner()
    committed, audit = owner.advance_continuous_state(
        transported_state, site_id=growth.site_id, duration_s=float(duration_s),
        transport_certificate=certificate,
        transport_before_state=previous_accepted_state.vacancy_transport_state_v2,
        transport_after_state=transported_state.vacancy_transport_state_v2,
        bulk_accommodation_area_rate_m2_s=bulk_rate,
        stress_tensor_Pa=tensors, sigma_zz_Pa=zz,
        temperature_K=transported_state.vacancy_temperature_K,
        support_radius_m=max(5.0*local_h, local_h), local_mesh_size_m=local_h,
        remesh_callback=remesh_delegate, source_tensor_observer=source_local_fem_tensors,
        transport_packet_solver=solve_delegated_transport_packet,
        loop_emission_enabled=growth.cavity_emission_state.enabled,
        failure_injector=failure_injector)
    growth_after = next(x for x in committed.void_state.embryo_growth_states
                        if x.site_id == growth.site_id)
    return committed, {**dict(audit),
                       "absorbed_inventory_area_m2": certificate.raw_absorbed_area_m2,
                       "radius_before_m": growth.radius_m,
                       "radius_after_m": growth_after.radius_m,
                       "unaccommodated_cavity_inventory_area_m2":
                           growth_after.captured_unaccommodated_vacancy_area_m2,
                       "growth_transaction_owner": "EmbryoGrowthProcessOwner",
                       "legacy_entry_point_strict_delegate": True}

def _synchronize_legacy_vacancy_observable(state):
    """Legacy global inventory is derived from the authoritative P0 field."""
    if state.void_state is None:
        return state
    from .vacancy_coupling_v1 import vacancy_conservation
    conservation = vacancy_conservation(state)
    voids = replace(
        state.void_state,
        available_defect_inventory_area_m2=conservation["vacancy_field_inventory_area_m2"],
        consumed_defect_inventory_area_m2=float(
            state.vacancy_ledgers["vacancy_consumed_by_void_area_m2"]
        ),
    )
    return replace(state, void_state=voids)


def create_subgrid_cavity_from_vacancies(state, site_id, radius_m, *, process_zone_length_m):
    """Create one cavity only after an exact local field debit."""
    from .vacancy_coupling_v1 import debit_local_vacancy_inventory, deterministic_capture_patch
    site = next(item for item in state.void_state.sites if item.site_id == site_id)
    requested = math.pi * float(radius_m)**2
    patch = deterministic_capture_patch(state, site.center_m, process_zone_length_m)
    debited, audit = debit_local_vacancy_inventory(
        state, requested_area_m2=requested, element_ids=patch,
    )
    if audit["growth_inventory_limited"]:
        raise RuntimeError("local vacancy capture patch cannot supply the cavity seed")
    # The legacy void-state scalar is not consulted as a source.  It is reset
    # from the field after the topology-neutral cavity update.
    generous = replace(
        debited.void_state,
        available_defect_inventory_area_m2=max(
            requested, debited.void_state.available_defect_inventory_area_m2,
        ),
    )
    voids = create_subgrid_cavity(generous, site_id, radius_m)
    return _synchronize_legacy_vacancy_observable(replace(debited, void_state=voids)), audit


def grow_cavity_from_vacancies(
    state, cavity_id, *, rates, dt_s, process_zone_length_m,
):
    """Advance radius at the vacancy/plastic series rate and debit its local patch."""
    from .vacancy_coupling_v1 import debit_local_vacancy_inventory, deterministic_capture_patch
    cavity = next(item for item in state.void_state.cavities if item.cavity_id == cavity_id)
    velocity = max(float(rates.get("Rdot_m_s", 0.0)), 0.0)
    requested_radius = float(cavity.radius_m) + velocity * max(float(dt_s), 0.0)
    requested_area = max(math.pi * requested_radius**2 - cavity.area_m2, 0.0)
    patch = deterministic_capture_patch(
        state, cavity.center_m, float(cavity.radius_m) + float(process_zone_length_m),
    )
    debited, audit = debit_local_vacancy_inventory(
        state, requested_area_m2=requested_area, element_ids=patch,
    )
    actual_area = cavity.area_m2 + float(audit["consumed_vacancy_area_m2"])
    actual_radius = math.sqrt(actual_area / math.pi)
    grown = replace(
        cavity, radius_m=actual_radius, area_m2=actual_area,
        inventory_area_m2=cavity.inventory_area_m2 + float(audit["consumed_vacancy_area_m2"]),
        geometry_generation=cavity.geometry_generation + (actual_radius > cavity.radius_m),
    )
    result = replace(debited, void_state=replace_cavity(debited.void_state, grown))
    return _synchronize_legacy_vacancy_observable(result), {
        **audit, "Rdot_diff_m_s": rates.get("Rdot_diff_m_s", 0.0),
        "Rdot_plast_m_s": rates.get("Rdot_plast_m_s", 0.0),
        "Rdot_m_s": velocity, "controlling_branch": rates.get("controlling_branch", "UNAVAILABLE"),
        "radius_before_m": cavity.radius_m, "radius_after_m": actual_radius,
    }


def integrate_cavity_growth_on_common_clock(
    state, cavity_id, *, target_radius_m, maximum_duration_s, temperature_K,
    equivalent_plastic_strain_rate_gp, production_rate_s,
    annihilation_rate_s, process_zone_length_m,
):
    """Integrate R, c_v, sinks, and cavity consumption on one bounded clock."""
    from scipy.integrate import solve_ivp
    from .vacancy_coupling_v1 import (
        conservation_roundoff_budget_m2, deterministic_capture_patch,
        equilibrium_vacancy_fraction, vacancy_conservation,
        vacancy_growth_rates_m_s,
    )
    cavity = next(item for item in state.void_state.cavities if item.cavity_id == cavity_id)
    target = float(target_radius_m)
    maximum = float(maximum_duration_s)
    if target <= cavity.radius_m or maximum <= 0.0:
        raise ValueError("growth target and clock horizon must be positive")
    patch = deterministic_capture_patch(
        state, cavity.center_m, cavity.radius_m + float(process_zone_length_m),
    )
    area = np.asarray(state.mesh.area_e, dtype=float)
    patch_area = area[patch]
    plastic = np.asarray(equivalent_plastic_strain_rate_gp, dtype=float)
    production = np.asarray(production_rate_s, dtype=float)
    sink = np.asarray(annihilation_rate_s, dtype=float)
    if any(values.shape != (state.mesh.ne,) for values in (plastic, production, sink)):
        raise ValueError("growth fields must own one value per element")
    local_plastic = float(np.sum(patch_area * plastic[patch]) / np.sum(patch_area))
    equilibrium = equilibrium_vacancy_fraction(temperature_K)
    ne = state.mesh.ne

    def rhs(_time, owned):
        concentration = np.maximum(owned[:ne], 0.0)
        radius = max(float(owned[ne]), 1.0e-30)
        patch_inventory = float(np.sum(patch_area * concentration[patch]))
        local_c = patch_inventory / float(np.sum(patch_area))
        growth = vacancy_growth_rates_m_s(
            temperature_K=temperature_K, radius_m=radius,
            vacancy_fraction_site=local_c,
            equivalent_plastic_rate_s=local_plastic,
        )
        rdot = float(growth["Rdot_m_s"])
        derivative = production - sink * (concentration - equilibrium)
        area_rate = 2.0 * math.pi * radius * rdot
        if area_rate > 0.0 and patch_inventory > 0.0:
            derivative[patch] -= area_rate * concentration[patch] / patch_inventory
        else:
            rdot = 0.0
        annihilation_area_rate = float(np.sum(area * sink * concentration))
        return np.concatenate((derivative, (rdot, annihilation_area_rate)))

    def target_event(_time, owned):
        return float(owned[ne]) - target
    target_event.terminal = True
    target_event.direction = 1
    initial = np.concatenate((
        np.asarray(state.vacancy_fraction_gp, dtype=float),
        (float(cavity.radius_m), 0.0),
    ))
    solution = solve_ivp(
        rhs, (0.0, maximum), initial, method="BDF", rtol=1.0e-9,
        atol=np.concatenate((np.full(ne, 1.0e-18), (1.0e-15, 1.0e-24))),
        events=target_event,
    )
    if not solution.success:
        raise RuntimeError("coupled vacancy/growth integration failed: " + solution.message)
    final = solution.y[:, -1]
    elapsed = float(solution.t[-1])
    field = np.maximum(final[:ne], 0.0)
    radius = float(final[ne])
    consumed = math.pi * (radius**2 - cavity.radius_m**2)
    ledgers = dict(state.vacancy_ledgers)
    ledgers["vacancy_plastic_production_area_m2"] += float(np.sum(area * production)) * elapsed
    ledgers["vacancy_thermal_exchange_area_m2"] += float(np.sum(area * sink * equilibrium)) * elapsed
    ledgers["vacancy_annihilation_area_m2"] += float(final[ne + 1])
    ledgers["vacancy_consumed_by_void_area_m2"] += consumed
    grown = replace(
        cavity, radius_m=radius, area_m2=math.pi * radius**2,
        inventory_area_m2=cavity.inventory_area_m2 + consumed,
        geometry_generation=cavity.geometry_generation + int(radius > cavity.radius_m),
    )
    updated = replace(
        state, vacancy_fraction_gp=field, vacancy_ledgers=ledgers,
        vacancy_temperature_K=float(temperature_K),
        void_state=replace_cavity(state.void_state, grown),
    )
    residual = vacancy_conservation(updated)["vacancy_conservation_residual_m2"]
    budget = conservation_roundoff_budget_m2(
        *ledgers.values(), vacancy_conservation(updated)["vacancy_field_inventory_area_m2"],
    )
    if abs(residual) > budget:
        raise RuntimeError("coupled growth conservation exceeds numerical budget")
    if residual:
        ledgers = dict(updated.vacancy_ledgers)
        ledgers["vacancy_annihilation_area_m2"] += residual
        updated = replace(updated, vacancy_ledgers=ledgers)
    patch_inventory = float(np.sum(patch_area * field[patch]))
    final_growth = vacancy_growth_rates_m_s(
        temperature_K=temperature_K, radius_m=radius,
        vacancy_fraction_site=patch_inventory / float(np.sum(patch_area)),
        equivalent_plastic_rate_s=local_plastic,
    )
    return _synchronize_legacy_vacancy_observable(updated), {
        "elapsed_time_s": elapsed,
        "target_reached": bool(solution.t_events[0].size),
        "radius_before_m": cavity.radius_m,
        "radius_after_m": radius,
        "cavity_consumption_area_m2": consumed,
        "capture_patch_element_ids": patch.tolist(),
        "accepted_ledger_roundoff_correction_m2": residual,
        "conservation_roundoff_budget_m2": budget,
        **final_growth,
    }


def evolve_accepted_plastic_vacancy_interval(
    state, *, bundle, temperature_K, duration_s, stress_scale=1.0,
    plasticity_parameter_overrides=None, test_only=False,
):
    """TEST-ONLY Peierls/Taylor perturbation helper.

    Scientific execution must supply ordinary previous/accepted mechanical
    states to ``advance_site_on_common_vacancy_clock``. This helper deliberately
    constructs a generic ablation configuration and may only be used by causal
    perturbation and extreme-reservoir plumbing tests.
    """
    from .config import SimulationConfig
    from .materials import PlasticityModel
    from .plasticity import update_plasticity
    from .vacancy_coupling_v1 import (
        advance_accepted_vacancy_state, equivalent_plastic_strain_rate_s,
    )
    from .unified_fracture_material_v5 import front_config

    if not test_only:
        raise RuntimeError(
            "fresh default plasticity configuration is forbidden in scientific vacancy execution"
        )

    _, _, sigma_gp, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    config = SimulationConfig()
    # This repository's common bulk updater retains its additive
    # Peierls--Taylor implementation.  The accepted ep/rho output—not these
    # parameters—is the sole input to the vacancy closure.
    config.dislocations.bulk_kinetics_model = "legacy_additive_flow_stress"
    config.dislocations.plastic_update_mode = "flow_stress"
    config.dislocations.thermo_consistency_mode = "off"
    for name, value in dict(plasticity_parameter_overrides or {}).items():
        if name not in {
            "peierls_H0_eV", "peierls_S_kB", "peierls_v0_b3",
            "taylor_athermal_alpha",
        }:
            raise ValueError("unsupported causal Peierls/Taylor test override: " + name)
        setattr(config.dislocations, name, float(value))
    ep_gp, rho_gp, _, plasticity_audit = update_plasticity(
        np.asarray(state.ep_gp, dtype=float).copy(),
        np.asarray(state.rho_gp, dtype=float).copy(),
        float(stress_scale) * sigma_gp, state.material, float(temperature_K),
        float(duration_s), PlasticityModel(config.plasticity_barrier, state.material),
        config.dislocations, return_info=True,
    )
    mechanical = equilibrate_fixed_load_with_production_fem(
        replace(state, ep_gp=ep_gp, rho_gp=rho_gp)
    )
    accepted_rate = equivalent_plastic_strain_rate_s(
        state.ep_gp, mechanical.ep_gp, duration_s,
    )
    evolved, vacancy_audit = advance_accepted_vacancy_state(
        state, mechanical, duration_s=duration_s, temperature_K=temperature_K,
        process_zone_length_m=front_config(bundle).L_pz,
        rho_reference_m2=float(bundle.fracture_row["rho_forest_floor_m2"]),
    )
    return evolved, accepted_rate, {
        "plasticity_material_row_id": bundle.fracture_material_row_id,
        "plasticity_kinetics_model": config.dislocations.bulk_kinetics_model,
        "peierls_H0_eV": config.dislocations.peierls_H0_eV,
        "peierls_activation_entropy_kB": config.dislocations.peierls_S_kB,
        "peierls_v0_b3": config.dislocations.peierls_v0_b3,
        "taylor_athermal_alpha": config.dislocations.taylor_athermal_alpha,
        "accepted_equivalent_plastic_rate_max_s": float(np.max(accepted_rate)),
        "accepted_equivalent_plastic_rate_integral_m2_s": float(np.sum(
            state.mesh.area_e * accepted_rate
        )),
        "plasticity_update": plasticity_audit,
        "vacancy_update": vacancy_audit,
    }


def fresh_sharp_front_engine(material, bundle: UnifiedFractureMaterialBundle):
    """Construct every root, branch and downstream engine through one factory."""
    return canonical_front_engine(bundle, material)


def capture_sharp_front_engine(engine) -> dict[str, Any]:
    """Checkpoint the qualified engine and complete spatial MPZ ownership."""
    from .sharp_front_v11_branching import _capture_shared_engine
    return {
        "schema": FRONT_ENGINE_STATE_SCHEMA,
        "initialization_policy": "canonical_front_engine->reset",
        "material_bundle_id": getattr(engine, "_unified_bundle_id", None),
        "material_bundle_sha256": getattr(engine, "_unified_bundle_hash", None),
        "core_model_id": getattr(engine, "_unified_core_model_id", None),
        "shared_engine_state": _capture_shared_engine(engine),
    }


def sharp_front_engine_fingerprint(engine) -> str:
    """Hash every serializable engine and spatial-MPZ field by value."""
    from dataclasses import is_dataclass, fields
    from .sharp_front_v11_branching import _capture_shared_engine
    def normalize(value):
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            return {"dtype": array.dtype.str, "shape": list(array.shape),
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        if isinstance(value, dict) or hasattr(value, "items"):
            return {str(key): normalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        if is_dataclass(value):
            return {item.name: normalize(getattr(value, item.name)) for item in fields(value)}
        if hasattr(value, "__dict__"):
            return normalize(vars(value))
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite": str(value)}
        return value
    encoded = json.dumps(normalize(_capture_shared_engine(engine)), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def restore_sharp_front_engine(material, bundle, payload):
    engine = fresh_sharp_front_engine(material, bundle)
    if payload.get("schema") != FRONT_ENGINE_STATE_SCHEMA:
        raise ValueError("unsupported downstream child front-engine state")
    expected = {
        "material_bundle_id": bundle.bundle_id,
        "material_bundle_sha256": bundle.bundle_hash,
        "core_model_id": getattr(engine, "_unified_core_model_id", None),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("downstream child engine material/core identity differs from accepted state")
    from .sharp_front_v11_branching import _restore_shared_engine
    import copy
    def thaw(value):
        if isinstance(value, np.ndarray):
            return np.asarray(value).copy()
        if isinstance(value, Mapping):
            return {key: thaw(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(thaw(item) for item in value)
        if isinstance(value, list):
            return [thaw(item) for item in value]
        return copy.deepcopy(value)
    return _restore_shared_engine(engine, thaw(payload.get("shared_engine_state", {})))

def sharp_front_constitutive_response(engine, *, K_Pa_sqrt_m, temperature_K, dt_s):
    """Read-only response through the pre-void K -> FrontEngine interface."""
    K = max(float(K_Pa_sqrt_m), 0.0)
    T = float(temperature_K)
    dt = max(float(dt_s), 0.0)
    sigma = engine.sigma_tip(K)
    cleavage_rate, cleavage_raw_rate, cleavage_barrier = engine.lambda_cleave(sigma, T)
    emission_rate, emission_stress, emission_barrier = engine.lambda_emit(sigma, T)
    return {
        "r_eff_m": float(engine.r_eff()),
        "sigma_tip_Pa": float(sigma),
        "cleavage_barrier_J": float(cleavage_barrier),
        "cleavage_rate_s": float(cleavage_rate),
        "cleavage_raw_rate_s": float(cleavage_raw_rate),
        "emission_barrier_J": float(emission_barrier),
        "emission_rate_s": float(emission_rate),
        "emission_effective_stress_Pa": float(emission_stress),
        "clock_increment": float(engine.predict_clock_increment(K, T, dt)),
    }


def advance_disabled_v5_stage2(state, end_m, *, transaction_identity):
    """Default-off entry delegates unchanged to the actual Stage-II V12 driver.

    It cannot accept a V11 comparator or silently discard existing void state.
    This is a mechanics-neutrality screen, not a void-nucleation qualification.
    """
    from .v12_production_driver import execute_event
    if state.sharp_wake_model_id != V12_MODEL_ID or state.void_state is not None:
        raise ValueError("disabled V5 requires the unchanged void-free Stage-II V12 state")
    return execute_event(
        state,end_m,transaction_identity=transaction_identity,
        event_classification="software_forced_geometry")


def _head():
    declared = os.environ.get("VOIDING_V5_SOURCE_COMMIT")
    if declared:
        return declared
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=Path(__file__).resolve().parents[1], text=True).strip()


def _geometry(radius_m=5.0e-5, center_m=(7.0e-4, 0.0), *, boundary_segments=32,
              radial_layers=12, source_conforming_geometry=False,
              source_direction_xy=(1.0, 0.0), shape_regular_local_level=None,
              specimen_width_m=1.0e-3, specimen_height_m=1.0e-3,
              far_h_m=5.0e-5):
    if center_m[1] < 0.0:
        positive_hole, positive_filled = _geometry(
            radius_m, (center_m[0], -center_m[1]),
            boundary_segments=boundary_segments, radial_layers=radial_layers,
            source_conforming_geometry=source_conforming_geometry,
            source_direction_xy=(source_direction_xy[0], -source_direction_xy[1]),
            shape_regular_local_level=shape_regular_local_level,
            specimen_width_m=specimen_width_m,
            specimen_height_m=specimen_height_m, far_h_m=far_h_m,
        )
        def mirrored(value):
            nodes = np.asarray(value.mesh.nodes).copy(); nodes[:, 1] *= -1.0
            elems = np.asarray(value.mesh.elems)[:, [0, 2, 1]]
            mesh = rebuild_tri_mesh(nodes, elems, tip_centers=np.asarray(center_m))
            bottom = np.asarray(value.boundary.top_nodes)
            boundary = BoundaryData(
                np.asarray(value.boundary.bot_nodes), bottom,
                int(bottom[np.argmin(nodes[bottom, 0])]), int(bottom[np.argmax(nodes[bottom, 0])]),
                np.asarray(value.boundary.notch_nodes),
            )
            return replace(value, mesh=mesh, boundary=boundary,
                           center_m=(float(center_m[0]), float(center_m[1])))
        return mirrored(positive_hole), mirrored(positive_filled)
    if shape_regular_local_level is not None:
        if source_conforming_geometry:
            raise ValueError("V4 and V5 cavity geometry contracts are mutually exclusive")
        hole = build_shape_regular_source_patch_hole_mesh(
            specimen_width_m, specimen_height_m, center_m, radius_m, far_h_m,
            local_level=shape_regular_local_level,
            polygon_sectors=boundary_segments, base_radial_layers=radial_layers,
            source_direction_xy=source_direction_xy,
        )
    else:
        builder = (build_source_conforming_hole_mesh if source_conforming_geometry
                   else build_explicit_hole_mesh)
        builder_kwargs = ({"source_direction_xy": source_direction_xy}
                          if source_conforming_geometry else {})
        hole = builder(specimen_width_m, specimen_height_m, center_m, radius_m, far_h_m,
                       boundary_segments, radial_layers_override=radial_layers,
                       **builder_kwargs)
    return hole, fill_explicit_hole_mesh(hole)


def _insert_point_in_mesh(mesh, point):
    point = np.asarray(point, dtype=float)
    nodes = np.asarray(mesh.nodes)
    if float(np.min(np.linalg.norm(nodes - point, axis=1))) <= 1.0e-12:
        return mesh
    owners = []
    for index, triangle_ids in enumerate(np.asarray(mesh.elems, dtype=int)):
        triangle = nodes[triangle_ids]
        matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
        if abs(np.linalg.det(matrix)) <= 1.0e-24: continue
        uv = np.linalg.solve(matrix, point - triangle[0])
        if uv[0] >= -1.0e-12 and uv[1] >= -1.0e-12 and uv.sum() <= 1.0 + 1.0e-12:
            owners.append((index, uv))
    if not owners:
        raise ValueError("fixed crack tip is outside the specimen mesh")
    new = len(nodes)
    source = np.asarray(mesh.elems, dtype=int)
    strict = [(index, uv) for index, uv in owners
              if uv[0] > 1.0e-10 and uv[1] > 1.0e-10 and uv.sum() < 1.0 - 1.0e-10]
    replacements = []
    removed = set()
    if strict:
        owner = strict[0][0]
        a, b, c = map(int, source[owner])
        removed.add(owner)
        replacements.extend(((a, b, new), (b, c, new), (c, a, new)))
    else:
        # A point on an existing edge must split both incident triangles.  A
        # one-sided split creates a hanging node and makes mirrored fixed-crack
        # meshes depend on which owner happens to be encountered first.
        candidate_edges = []
        for owner, _ in owners:
            triangle_ids = source[owner]
            for u, v in ((triangle_ids[0], triangle_ids[1]),
                         (triangle_ids[1], triangle_ids[2]),
                         (triangle_ids[2], triangle_ids[0])):
                edge = nodes[v] - nodes[u]
                scale = max(float(np.dot(edge, edge)), 1.0e-300)
                fraction = float(np.dot(point - nodes[u], edge) / scale)
                offset = point - nodes[u]
                distance = abs(float(edge[0] * offset[1] - edge[1] * offset[0])) / math.sqrt(scale)
                if -1.0e-12 <= fraction <= 1.0 + 1.0e-12 and distance <= 1.0e-12:
                    candidate_edges.append(tuple(sorted((int(u), int(v)))))
        edge = min(candidate_edges)
        for owner, triangle_ids in enumerate(source):
            if not set(edge).issubset(map(int, triangle_ids)):
                continue
            removed.add(owner)
            u, v = edge
            w = next(int(value) for value in triangle_ids if int(value) not in edge)
            original = nodes[triangle_ids]
            oa, ob = original[1] - original[0], original[2] - original[0]
            original_sign = oa[0] * ob[1] - oa[1] * ob[0]
            for tri in ((u, new, w), (new, v, w)):
                coordinates = np.vstack((nodes[tri[0]], point, nodes[tri[2]])) if tri[0] == u else np.vstack((point, nodes[tri[1]], nodes[tri[2]]))
                ca, cb = coordinates[1] - coordinates[0], coordinates[2] - coordinates[0]
                sign = ca[0] * cb[1] - ca[1] * cb[0]
                replacements.append(tri if sign * original_sign > 0.0 else (tri[0], tri[2], tri[1]))
    elems = np.delete(source, sorted(removed), axis=0)
    elems = np.vstack((elems, replacements))
    return rebuild_tri_mesh(np.vstack((nodes, point)), elems, tip_centers=point)


def _external_free_root_context(mesh, boundary, root):
    counts = {}
    for triangle in np.asarray(mesh.elems, dtype=int):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = tuple(sorted((int(a), int(b))))
            counts[edge] = counts.get(edge, 0) + 1
    prescribed_nodes = set(map(int, np.asarray(boundary.top_nodes))) | set(
        map(int, np.asarray(boundary.bot_nodes))
    )
    point = np.asarray(root, dtype=float)
    free_edges = [edge for edge, count in counts.items()
                  if count == 1 and not set(edge).issubset(prescribed_nodes)]
    incident = []
    for edge in free_edges:
        if set(edge).issubset(prescribed_nodes):
            continue
        a, b = np.asarray(mesh.nodes)[list(edge)]
        delta = b - a
        fraction = float((point - a) @ delta / max(delta @ delta, 1.0e-300))
        distance = float(np.linalg.norm(point - (a + np.clip(fraction, 0.0, 1.0) * delta)))
        if distance <= 1.0e-12 and -1.0e-12 <= fraction <= 1.0 + 1.0e-12:
            incident.append(edge)
    if not incident:
        return {}
    component = set(incident)
    frontier = set(node for edge in incident for node in edge)
    changed = True
    while changed:
        changed = False
        for edge in free_edges:
            if edge in component or not frontier.intersection(edge):
                continue
            component.add(edge); frontier.update(edge); changed = True
    return {"b00000000:arc0": ({
        "endpoint": "start", "endpoint_role": "physical_root",
        "endpoint_coordinate_m": tuple(map(float, root)),
        "boundary_kind": "external_free_surface",
        "boundary_component_id": "external-free-component-at-root",
        "boundary_edge_ids": tuple(sorted(component)), "cavity_id": None,
        "tangent_enters_or_approaches_solid": True,
    },)}


def _owned_transport_boundary_tags(mesh, boundary, *, cavity_node_ids=()):
    """Persist boundary semantics supplied by this geometry constructor."""
    counts = {}
    for triangle in np.asarray(mesh.elems, dtype=int):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]),
                     (triangle[2], triangle[0])):
            edge = tuple(sorted((int(a), int(b))))
            counts[edge] = counts.get(edge, 0) + 1
    grips = set(map(int, np.asarray(boundary.top_nodes))) | set(
        map(int, np.asarray(boundary.bot_nodes))
    )
    cavity = set(map(int, cavity_node_ids))
    rows = []
    for edge, count in sorted(counts.items()):
        if count != 1:
            continue
        if set(edge) <= cavity:
            kind = "ACTIVE_CAVITY_SURFACE"
        elif set(edge) <= grips:
            kind = "CONTACT_OR_GRIP_NO_FLUX"
        else:
            # This classification is made once by the known uncoated specimen
            # geometry owner and persisted; the transport solver never infers it.
            kind = "EXTERNAL_FREE_SURFACE"
        rows.append({"node_ids": edge, "kind": kind})
    return tuple(rows)


def _refine_state_around_graph(state, levels):
    from .adaptive_multitip_mesh_v11 import refine_accepted_state
    result = state
    for level in range(int(levels)):
        centroids = np.asarray(result.mesh.nodes)[np.asarray(result.mesh.elems)].mean(axis=1)
        local_h = np.sqrt(np.maximum(np.asarray(result.mesh.area_e), 1.0e-300))
        marked = np.zeros(result.mesh.ne, dtype=bool)
        for branch in result.crack_network.branches:
            for first, second in zip(branch.path, branch.path[1:]):
                a, b = np.asarray(first), np.asarray(second)
                delta = b - a
                fraction = np.clip(((centroids - a) @ delta) /
                                   max(float(delta @ delta), 1.0e-300), 0.0, 1.0)
                distance = np.linalg.norm(centroids - (a + fraction[:, None] * delta), axis=1)
                marked |= distance <= 2.0 * local_h
        result, _ = refine_accepted_state(
            result, marked_parent_elements=tuple(np.flatnonzero(marked)),
            active_tip_ids=result.crack_network.active_tip_ids,
            generation=level + 1, operation_index=level + 1,
        )
    return result


def build_production_void_state(*, bundle: UnifiedFractureMaterialBundle, enabled=True, stochastic=False, seed=3621,
                                cavity_center_m=(7.0e-4, 0.0), crack_path_m=None,
                                void_site_centers_m=None,
                                cleavage_theta_deg=0.0, boundary_segments=32, radial_layers=12,
                                source_conforming_geometry=False,
                                shape_regular_local_level=None,
                                temperature_K=900.0,
                                initial_vacancy_fraction_gp=None,
                                vacancy_energy_variant="reference",
                                direct_interaction_birth_enabled=False,
                                direct_interaction_parameter_row=None,
                                cavity_surface_loop_emission_enabled=False,
                                cavity_emission_parameter_row=None,
                                interaction_population_partition=None,
                                bulk_initial_density_m2=None,
                                initial_plastic_strain_gp=None,
                                specimen_width_m=1.0e-3,
                                specimen_height_m=1.0e-3,
                                far_h_m=5.0e-5,
                                test_only_extreme_reservoir=False,
                                cavity_geometry_radius_m=5.0e-5):
    if (boundary_segments, radial_layers) != (32, 12) and crack_path_m is None:
        raise ValueError("production resolution transfer requires an explicit fixed crack path")
    source_direction = (math.cos(math.radians(cleavage_theta_deg)),
                        math.sin(math.radians(cleavage_theta_deg)))
    hole, filled = _geometry(
        radius_m=float(cavity_geometry_radius_m),
        center_m=cavity_center_m, boundary_segments=boundary_segments,
        radial_layers=radial_layers, source_conforming_geometry=source_conforming_geometry,
        source_direction_xy=source_direction,
        shape_regular_local_level=shape_regular_local_level,
        specimen_width_m=specimen_width_m,
        specimen_height_m=specimen_height_m, far_h_m=far_h_m,
    )
    mesh = filled.mesh
    ray = 16
    if crack_path_m is None:
        path_mesh = mesh
        if source_conforming_geometry or shape_regular_local_level is not None:
            _, retained_filled = _geometry(
                radius_m=float(cavity_geometry_radius_m),
                center_m=cavity_center_m, boundary_segments=32, radial_layers=12,
                source_conforming_geometry=False,
                specimen_width_m=specimen_width_m,
                specimen_height_m=specimen_height_m, far_h_m=far_h_m,
            )
            path_mesh = retained_filled.mesh
        start = tuple(map(float, path_mesh.nodes[12 * 32 + ray]))
        tip = tuple(map(float, path_mesh.nodes[3 * 32 + ray]))
        crack_path = (start, tip)
        if source_conforming_geometry or shape_regular_local_level is not None:
            for point in crack_path:
                mesh = _insert_point_in_mesh(mesh, point)
                hole = replace(hole, mesh=_insert_point_in_mesh(hole.mesh, point))
            filled = replace(filled, mesh=mesh)
    else:
        crack_path = tuple(tuple(map(float, point)) for point in crack_path_m)
        if len(crack_path) < 2:
            raise ValueError("fixed crack path requires at least two points")
        start, tip = crack_path[0], crack_path[-1]
        for point in crack_path:
            mesh = _insert_point_in_mesh(mesh, point)
            hole = replace(hole, mesh=_insert_point_in_mesh(hole.mesh, point))
        filled = replace(filled, mesh=mesh)
    material = elastic_material(bundle)
    from .vacancy_transport_phase_field_v2 import (
        equilibrium_vacancy_fraction as transport_equilibrium_vacancy_fraction,
        vacancy_energy_study_row,
    )
    transport_row = vacancy_energy_study_row(material, vacancy_energy_variant)
    if interaction_population_partition is not None:
        mobile, direct, benign = map(float, interaction_population_partition)
        if min(mobile, direct, benign) < 0.0 or not math.isclose(
            mobile+direct+benign, 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("interaction population partition must be nonnegative and sum to one")
        transport_row = replace(
            transport_row,
            row_id=transport_row.row_id+":interaction-partition-v1",
            interaction_mobile_vacancy_fraction=mobile,
            interaction_direct_cavitation_fraction=direct,
            interaction_benign_fraction=benign,
        )
    element_x = np.asarray(mesh.nodes)[np.asarray(mesh.elems)].mean(axis=1)[:, 0]
    u = np.zeros(mesh.ndof)
    remote_half_opening = 2.0e-7*float(specimen_height_m)/1.0e-3
    u[2 * np.asarray(filled.boundary.top_nodes) + 1] = remote_half_opening
    u[2 * np.asarray(filled.boundary.bot_nodes) + 1] = -remote_half_opening
    if initial_vacancy_fraction_gp is None:
        vacancy = np.full(
            mesh.ne,
            transport_equilibrium_vacancy_fraction(temperature_K, transport_row),
            dtype=float,
        )
    else:
        supplied = np.asarray(initial_vacancy_fraction_gp, dtype=float)
        vacancy = (
            np.full(mesh.ne, float(supplied), dtype=float)
            if supplied.ndim == 0 else supplied.copy()
        )
        if vacancy.shape != (mesh.ne,) or np.any(~np.isfinite(vacancy)) or np.any(vacancy < 0.0):
            raise ValueError("explicit initial vacancy field must be finite, nonnegative P0 data")
        from .vacancy_coupling_v1 import DILUTE_VACANCY_FRACTION_LIMIT
        if np.max(vacancy) > DILUTE_VACANCY_FRACTION_LIMIT and not test_only_extreme_reservoir:
            raise ValueError(
                "non-test vacancy fraction exceeds the declared dilute-model validity range"
            )
    vacancy_inventory = float(np.sum(vacancy * np.asarray(mesh.area_e)))
    void_state = None
    if enabled:
        rng = np.random.default_rng(seed)
        centers = (cavity_center_m,) if void_site_centers_m is None else tuple(void_site_centers_m)
        if not centers:
            raise ValueError("enabled voiding requires at least one predefined site")
        sites = []
        for index, center in enumerate(centers, start=1):
            xy = tuple(map(float, center))
            if len(xy) != 2 or not all(math.isfinite(value) for value in xy):
                raise ValueError("void site centers must be finite 2-vectors")
            thresholds = (tuple(float(rng.exponential()) for _ in range(3))
                          if stochastic else (0.25, 0.25, 0.35))
            sites.append(VoidSite(
                f"site-{index}", xy, VoidPhase.AVAILABLE_SITE, 0, 2, 0.8,
                HazardClock(0.0, thresholds[0]), HazardClock(0.0, thresholds[1]),
                HazardClock(0.0, thresholds[2])))
        void_state = ProductionVoidState(
            tuple(sites), rng_state=rng.bit_generator.state,
            available_defect_inventory_area_m2=vacancy_inventory,
        )
        from .dislocation_interaction_voiding_v1 import initialize_dislocation_interaction_state
        from .embryo_growth_v1 import EmbryoGrowthConfiguration
        void_state = replace(
            void_state,
            embryo_growth_configuration=EmbryoGrowthConfiguration(
                vacancy_clustering_birth_enabled=True,
                direct_interaction_birth_enabled=bool(direct_interaction_birth_enabled),
                cavity_surface_loop_emission_enabled=bool(cavity_surface_loop_emission_enabled),
                cavity_emission_parameter_row=cavity_emission_parameter_row),
            dislocation_interaction_state=initialize_dislocation_interaction_state(
                void_state.sites, stochastic=stochastic, seed=seed+45536,
                enabled=bool(direct_interaction_birth_enabled), support_radius_m=5.0e-5,
                parameter_row=direct_interaction_parameter_row,
            ),
        )
    if initial_plastic_strain_gp is None:
        initial_ep = np.vstack((
            np.full(mesh.ne, 1.0e-5), np.full(mesh.ne, -0.5e-5),
            np.full(mesh.ne, 0.25e-5 * (1.0 if cavity_center_m[1] >= 0.0 else -1.0)),
        ))
    else:
        supplied_ep = np.asarray(initial_plastic_strain_gp, dtype=float)
        initial_ep = np.zeros((3, mesh.ne), dtype=float) if supplied_ep.ndim == 0 and float(supplied_ep) == 0.0 else supplied_ep.copy()
        if initial_ep.shape != (3, mesh.ne) or np.any(~np.isfinite(initial_ep)):
            raise ValueError("initial plastic strain must be finite (3, ne) data or scalar zero")
    initial_rho = (
        float(bundle.fracture_row["rho_source0_m2"])
        if bulk_initial_density_m2 is None else float(bulk_initial_density_m2)
    )
    if not math.isfinite(initial_rho) or initial_rho <= 0.0:
        raise ValueError("bulk initial density must be finite and positive")
    state = LiveFEMTopologyState(
        mesh, filled.boundary, np.zeros(mesh.nn), u,
        initial_ep,
        np.full(mesh.ne, initial_rho),
        plane_strain_D(material), material, None,
        CrackNetworkState.one_tip(crack_path, local_state={"front_engine_state_owner": ROOT_BRANCH_ID}), DirectionalCompetitionState.initialize(
            tungsten_cleavage_candidates(theta_deg=cleavage_theta_deg), global_hazard_seed=seed,
        ),
        {"active_branch_id": ROOT_BRANCH_ID,
         "by_branch": {ROOT_BRANCH_ID: capture_sharp_front_engine(fresh_sharp_front_engine(material, bundle))},
         "owner_by_front": {ROOT_BRANCH_ID: ROOT_BRANCH_ID}}, {
            "source_state": {"density": 3.0, "clock": 0.125},
            "production_mesh_resolution": (int(boundary_segments), int(radial_layers)),
            "boundary_terminal_context": _external_free_root_context(mesh, filled.boundary, start),
            "transport_boundary_tags_v2": _owned_transport_boundary_tags(mesh, filled.boundary),
        },
        {"emission_work": 1.0}, np.random.default_rng(seed).bit_generator.state,
        {"accepted_steps": 0, "topology_actions": 0, "mesh_generation": 0}, 0.0,
        void_state=void_state,
        vacancy_fraction_gp=vacancy,
        vacancy_temperature_K=float(temperature_K),
    )
    if abs(float(cleavage_theta_deg)) > 1.0e-12:
        hole_fields = _project_fields(state, hole.mesh)
        hole_state = replace(
            state, mesh=hole.mesh, boundary=hole.boundary,
            damage=hole_fields["damage"], displacement=hole_fields["displacement"],
            ep_gp=hole_fields["ep_gp"], rho_gp=hole_fields["rho_gp"],
            vacancy_fraction_gp=hole_fields["vacancy_fraction_gp"],
            vacancy_ledgers=hole_fields["vacancy_ledgers"],
            vacancy_temperature_K=hole_fields["vacancy_temperature_K"],
        )
        hole_state = _refine_state_around_graph(hole_state, 2)
        hole = replace(hole, mesh=hole_state.mesh, boundary=hole_state.boundary)
        state = _refine_state_around_graph(state, 2)
        junction = dict(state.junction_process_state)
        junction["boundary_terminal_context"] = _external_free_root_context(
            state.mesh, state.boundary, start,
        )
        state = replace(state, junction_process_state=junction)
    state = bind_identity(state, bundle)
    state = initialize_mechanically_separating_v12(
        state, source_commit=_head(), configuration={"voiding_enabled": enabled, "model": V12_MODEL_ID},
    )
    state = equilibrate_fixed_load_with_production_fem(state)
    from .vacancy_transport_phase_field_v2 import (
        LocalizedCaptureSite,
        build_exact_cut_transport_mesh, initial_transport_state,
    )
    transport_row = replace(
        transport_row,
        row_id=transport_row.row_id + ":rho-ref=" + bundle.fracture_row["rho_forest_floor_m2"],
        rho_reference_m2=float(bundle.fracture_row["rho_forest_floor_m2"]),
    )
    transport_sites = () if state.void_state is None else tuple(
        LocalizedCaptureSite(
            site_id=item.site_id, center_xy_m=item.center_m,
            support_radius_m=5.0e-5,
            capture_rate_area_per_s=0.0,
            required_embryo_inventory_area_m2=VoidingConfig().birth_normal_area_m2,
            hit_count=item.hits, required_hit_count=item.required_hits,
            birth_action=item.birth.accumulated, threshold=item.birth.threshold,
            rng_fingerprint=hashlib.sha256(json.dumps(
                state.void_state.rng_state, sort_keys=True, default=str,
            ).encode()).hexdigest(),
        ) for item in state.void_state.sites
    )
    # Extreme-reservoir V1 plumbing remains explicitly legacy and cannot be
    # mislabeled as a bounded V2 scientific state.
    if not test_only_extreme_reservoir:
        branch = state.crack_network.branch(ROOT_BRANCH_ID)
        exact_cut = build_exact_cut_transport_mesh(
            state.mesh, p0=branch.root, p1=branch.tip,
            topology_boundary_tags=state.junction_process_state["transport_boundary_tags_v2"],
            segment_id=f"{ROOT_BRANCH_ID}:0",
        )
        transport_state = initial_transport_state(
            exact_cut.mesh, temperature_K=temperature_K, row=transport_row,
            registry=exact_cut.registry,
            vacancy_fraction_gp=state.vacancy_fraction_gp[exact_cut.parent_mechanics_element_ids],
            parent_mechanics_element_ids=exact_cut.parent_mechanics_element_ids,
            site_states=transport_sites,
        )
        state = replace(state, vacancy_transport_state_v2=transport_state)
    return state, hole


def observables(state, operation):
    from .topology_transaction_v11 import require_equilibrium_observables
    from .vacancy_coupling_v1 import vacancy_conservation

    cavity = None if state.void_state is None or not state.void_state.cavities else state.void_state.cavities[0]
    site = None if state.void_state is None else state.void_state.sites[0]
    tri = np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)]
    side = np.linalg.norm(tri[:, [1, 2, 0]] - tri[:, [0, 1, 2]], axis=2)
    avec, bvec = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = np.abs(avec[:, 0] * bvec[:, 1] - avec[:, 1] * bvec[:, 0]) / 2.0
    quality = 4.0 * np.sqrt(3.0) * area / np.maximum(np.sum(side**2, axis=1), 1.0e-300)
    equilibrium = require_equilibrium_observables(state)
    reaction = equilibrium["latest_reaction_N_per_m"]
    active_branches = [branch.branch_id for branch in state.crack_network.branches
                       if branch.status == "active"]
    support_active = [] if state.v12_support_state is None else list(state.v12_support_state.active_tip_identities)
    vacancy = vacancy_conservation(state)
    return {
        "operation": operation, "fingerprint": complete_accepted_state_fingerprint(state),
        "mesh_nodes": int(state.mesh.nn), "mesh_elements": int(state.mesh.ne),
        "graph_length_m": float(state.crack_network.total_physical_crack_length_m),
        "reaction_N_per_m": reaction,
        "top_reaction_N_per_m": equilibrium["latest_top_reaction_N_per_m"],
        "bottom_reaction_N_per_m": equilibrium["latest_bottom_reaction_N_per_m"],
        "applied_opening_m": equilibrium["latest_applied_opening_m"],
        "compliance_m2_per_N": equilibrium["latest_compliance_m2_per_N"],
        "energy_J_per_m": float(state.stored_energy_J_per_m),
        "external_work_J_per_m": equilibrium["latest_external_work_J_per_m"],
        "stored_recoverable_energy_J_per_m": equilibrium["latest_stored_recoverable_energy_J_per_m"],
        "plastic_eigenstrain_half_work_J_per_m": equilibrium["latest_plastic_eigenstrain_half_work_J_per_m"],
        "energy_identity_reference_J_per_m": equilibrium["latest_energy_identity_reference_J_per_m"],
        "full_residual_including_reactions_N_per_m": equilibrium["latest_residual_l2_N_per_m"],
        "free_dof_residual_l2_N_per_m": equilibrium["latest_free_dof_residual_l2_N_per_m"],
        "constrained_reaction_l2_N_per_m": equilibrium["latest_constrained_reaction_l2_N_per_m"],
        "top_bottom_reaction_balance": equilibrium["latest_top_bottom_reaction_balance"],
        "energy_reaction_identity": equilibrium["latest_energy_reaction_identity"],
        "void_phase": None if cavity is None else cavity.phase.value,
        "site_phase": None if site is None else site.phase.value,
        "cavity_radius_m": None if cavity is None else cavity.radius_m,
        "cavity_area_m2": None if cavity is None else cavity.area_m2,
        "inventory_area_m2": None if cavity is None else cavity.inventory_area_m2,
        "available_defect_inventory_area_m2": None if state.void_state is None else state.void_state.available_defect_inventory_area_m2,
        "consumed_defect_inventory_area_m2": None if state.void_state is None else state.void_state.consumed_defect_inventory_area_m2,
        "vacancy_fraction_max": float(np.max(state.vacancy_fraction_gp)),
        "vacancy_ledgers": dict(state.vacancy_ledgers),
        **vacancy,
        "void_event_history": [] if state.void_state is None else list(state.void_state.event_history),
        "length_ledgers": {} if state.void_state is None else dict(state.void_state.length_ledgers),
        "event_counters": dict(state.event_counters),
        "active_crack_branch_ids": active_branches,
        "support_active_tip_ids": support_active,
        "mesh_minimum_quality": float(np.min(quality)),
        "mesh_maximum_aspect_ratio": float(np.max(side, axis=1).max() / max(np.min(side), 1.0e-300)),
        "field_transfer_audit": state.junction_process_state.get("latest_void_remesh_audit"),
        "closed_cavity_boundary_cycle_certificate": state.junction_process_state.get("latest_closed_cavity_boundary_cycle_certificate"),
        "crack_void_connection_certificate": state.junction_process_state.get("latest_crack_void_connection_certificate"),
    }


def local_site_tensor(state):
    _, _, sigma, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    center = np.asarray(state.void_state.sites[0].center_m)
    centroids = state.mesh.nodes[state.mesh.elems].mean(axis=1)
    element = int(np.argmin(np.linalg.norm(centroids - center, axis=1)))
    return np.array([[sigma[0, element], sigma[2, element]], [sigma[2, element], sigma[1, element]]])


def crack_tip_tensor(state, *, branch_id):
    """Return a geometry-weighted tensor recovery at an aligned crack-tip node."""
    _, _, sigma, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    tip = np.asarray(state.crack_network.branch(branch_id).tip)
    nodes = np.asarray(state.mesh.nodes)
    node = int(np.argmin(np.linalg.norm(nodes - tip, axis=1)))
    if np.linalg.norm(nodes[node] - tip) > 1.0e-12:
        raise RuntimeError("crack-tip tensor recovery requires an aligned mesh node")
    elements = np.flatnonzero(np.any(np.asarray(state.mesh.elems) == node, axis=1))
    weights = np.asarray(state.mesh.area_e)[elements]
    weights = weights / np.sum(weights)
    recovered = np.array([
        [np.sum(weights * sigma[0, elements]), np.sum(weights * sigma[2, elements])],
        [np.sum(weights * sigma[2, elements]), np.sum(weights * sigma[1, elements])],
    ])
    return recovered, tuple(map(int, elements))


def _first_ray_cavity_intersection(state, start, direction):
    """Intersect a candidate ray with the actual polygonal cavity boundary."""
    nodes = np.asarray(state.mesh.nodes)
    origin = np.asarray(start, dtype=float)
    ray = np.asarray(direction, dtype=float)
    intersections = []
    for edge_ids in _actual_cavity_boundary_edges(state):
        first, second = nodes[edge_ids]
        edge = second - first
        matrix = np.column_stack((ray, -edge))
        if abs(np.linalg.det(matrix)) <= 1.0e-18:
            continue
        distance, fraction = np.linalg.solve(matrix, first - origin)
        if distance > 1.0e-14 and -1.0e-12 <= fraction <= 1.0 + 1.0e-12:
            intersections.append((float(distance), origin + distance * ray))
    if not intersections:
        return None
    return tuple(map(float, min(intersections, key=lambda item: item[0])[1]))


def _conform_cavity_intersection(state, intersection, *, failure_injector=None):
    """Insert an interior ray/cavity-edge intersection as an explicit node."""
    point = np.asarray(intersection, dtype=float)
    nodes = np.asarray(state.mesh.nodes)
    nearest = int(np.argmin(np.linalg.norm(nodes - point, axis=1)))
    if np.linalg.norm(nodes[nearest] - point) <= 1.0e-12:
        return state, nearest, False
    counts: dict[tuple[int, int], int] = {}
    owners: dict[tuple[int, int], list[int]] = {}
    for element, triangle in enumerate(np.asarray(state.mesh.elems, dtype=int)):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = tuple(sorted((int(a), int(b))))
            counts[edge] = counts.get(edge, 0) + 1
            owners.setdefault(edge, []).append(element)
    cavity = state.void_state.cavities[0]
    center = np.asarray(cavity.center_m)
    candidates = []
    for edge, count in counts.items():
        if count != 1:
            continue
        a, b = nodes[list(edge)]
        if max(abs(np.linalg.norm(a - center) - cavity.radius_m),
               abs(np.linalg.norm(b - center) - cavity.radius_m)) > cavity.radius_m * 0.02:
            continue
        delta = b - a
        fraction = float((point - a) @ delta / max(delta @ delta, 1.0e-300))
        distance = float(np.linalg.norm(point - (a + np.clip(fraction, 0.0, 1.0) * delta)))
        if 1.0e-12 < fraction < 1.0 - 1.0e-12:
            candidates.append((distance, edge, owners[edge][0]))
    if not candidates or min(candidates)[0] > 1.0e-11:
        raise RuntimeError("ray intersection is not on a splittable cavity boundary edge")
    _, edge, owner = min(candidates)
    triangle = list(map(int, state.mesh.elems[owner]))
    a, b = edge
    third = next(node for node in triangle if node not in edge)
    new_node = len(nodes)
    elems = np.delete(np.asarray(state.mesh.elems, dtype=int), owner, axis=0)
    elems = np.vstack((elems, (a, new_node, third), (new_node, b, third)))
    mesh = rebuild_tri_mesh(np.vstack((nodes, point)), elems, tip_centers=np.asarray(point))
    fields = _project_fields(state, mesh)
    rebuilt = remesh_mechanically_separating_v12(
        state, mesh=mesh, boundary=state.boundary, transferred_fields=fields,
        source_commit=_head(), configuration={"event": "CAVITY_EDGE_INTERSECTION_INSERTION"},
        transaction_identity="cavity-edge-intersection", failure_injector=failure_injector,
    )
    return rebuilt, new_node, True


def _conform_bulk_point(
    state, point, *, identity, failure_injector=None,
    transfer_v2_transport=True,
):
    mesh = _insert_point_in_mesh(state.mesh, point)
    if mesh is state.mesh:
        return state
    fields = _project_fields(
        state, mesh, transfer_v2_transport=bool(transfer_v2_transport),
    )
    if not transfer_v2_transport and state.vacancy_transport_state_v2 is not None:
        junction = dict(state.junction_process_state)
        junction["virtual_mechanics_transport_transfer"] = {
            "mode": "DEFERRED_ACCEPTED_OWNER_READ_ONLY",
            "accepted_surface_registry_fingerprint": (
                state.vacancy_transport_state_v2.surface_registry_fingerprint
            ),
            "physical_time_advanced": False,
            "transport_solve_performed": False,
        }
        state = replace(state, junction_process_state=junction)
    return remesh_mechanically_separating_v12(
        state, mesh=mesh, boundary=state.boundary, transferred_fields=fields,
        source_commit=_head(), configuration={"event": identity},
        transaction_identity=identity.lower().replace("_", "-"),
        failure_injector=failure_injector,
    )


def _actual_cavity_boundary_edges(state):
    """Radius identifies a component; only one-owner edges define its surface.

    A fine first solid layer can lie inside the 2% identification band. Its
    nodes must never become an invented polygon or a cavity-surface source.
    """
    cavity=state.void_state.cavities[0]
    nodes=np.asarray(state.mesh.nodes); elements=np.asarray(state.mesh.elems)
    edges=np.sort(np.concatenate((elements[:,[0,1]],elements[:,[1,2]],elements[:,[2,0]])),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    boundary=unique[counts==1]
    radii=np.linalg.norm(nodes-np.asarray(cavity.center_m),axis=1)
    selected=boundary[np.all(radii[boundary]<=cavity.radius_m*1.02,axis=1)]
    if not len(selected): raise ValueError("no actual cavity boundary edges")
    return selected


def cavity_boundary_tensor(state, *, boundary_node: int | None = None,
                           boundary_element: int | None = None):
    """Return the most tensile resolved tensor on the explicit cavity boundary.

    First passage is local, so circumferential averaging (which can cancel the
    tensile hot spot) is not a valid kinetic input.
    """
    _, _, sigma, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    boundary_nodes = np.unique(_actual_cavity_boundary_edges(state))
    selected_nodes = boundary_nodes if boundary_node is None else np.asarray((int(boundary_node),))
    if boundary_node is not None and int(boundary_node) not in set(map(int, boundary_nodes)):
        raise ValueError("requested tensor node is not on the cavity boundary")
    elements = np.flatnonzero(np.any(np.isin(state.mesh.elems, selected_nodes), axis=1))
    if boundary_element is not None:
        if boundary_node is None:
            raise ValueError("fixed boundary element requires a fixed boundary node")
        if int(boundary_element) not in set(map(int, elements)):
            raise ValueError("requested tensor element is not incident to the fixed boundary node")
        elements = np.asarray((int(boundary_element),))
    tensors = np.asarray([
        [[sigma[0, element], sigma[2, element]],
         [sigma[2, element], sigma[1, element]]]
        for element in elements
    ])
    selected = int(np.argmax(np.linalg.eigvalsh(tensors)[:, -1]))
    return tensors[selected], (int(elements[selected]),)


def cavity_boundary_recovery_operator(state, boundary_node: int) -> dict[str, Any]:
    """Freeze a geometry-only, single-element boundary stress sampler."""
    elements = np.flatnonzero(np.any(np.asarray(state.mesh.elems) == int(boundary_node), axis=1))
    if not len(elements):
        raise ValueError("boundary node has no incident element")
    selected = int(np.min(elements))
    return {"boundary_node_id": int(boundary_node), "selected_element_id": selected,
            "recovery_operator_id": f"incident-element:{selected}", "recovery_weights": [1.0]}


def cavity_fixed_arc_patch_recovery_v2(state, *, boundary_node: int | None = None):
    """Apply the frozen V2 solid-side patch operator at the owned exit arc."""
    from .cavity_source_recovery_v2 import recover_fixed_arc_patch_v2
    cavity = state.void_state.cavities[0]
    position = np.asarray(cavity.connection_exit_m, dtype=float)
    if boundary_node is None:
        boundary_node = int(np.argmin(np.linalg.norm(np.asarray(state.mesh.nodes) - position, axis=1)))
    node = int(boundary_node)
    if np.linalg.norm(np.asarray(state.mesh.nodes)[node] - position) > 1.0e-12:
        raise ValueError("stored connection exit is not an aligned cavity-boundary node")
    _, _, sigma, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    damage = getattr(state.mesh, "element_damage_gp", None)
    solid = None if damage is None else np.asarray(damage, dtype=float) < 0.5
    return recover_fixed_arc_patch_v2(
        nodes=state.mesh.nodes,
        elements=state.mesh.elems,
        stress_Pa=sigma,
        boundary_node=node,
        cavity_id=cavity.cavity_id,
        cavity_center_m=cavity.center_m,
        cavity_radius_m=cavity.radius_m,
        owned_boundary_edges=_actual_cavity_boundary_edges(state),
        solid_element_mask=solid,
    )


def cavity_fixed_physical_arc_patch_recovery_v3(
    state, *, boundary_node: int | None = None
):
    """Apply the frozen V3 physical-window operator at the owned exit arc."""
    from .cavity_source_recovery_v3 import recover_fixed_physical_arc_patch_v3

    cavity = state.void_state.cavities[0]
    position = np.asarray(cavity.connection_exit_m, dtype=float)
    if boundary_node is None:
        boundary_node = int(
            np.argmin(np.linalg.norm(np.asarray(state.mesh.nodes) - position, axis=1))
        )
    node = int(boundary_node)
    if np.linalg.norm(np.asarray(state.mesh.nodes)[node] - position) > 1.0e-12:
        raise ValueError("stored connection exit is not an aligned cavity-boundary node")
    identity = dict(require_bound_identity(state))
    engine = fresh_sharp_front_engine(state.material, _bundle_from_state(state))
    _, _, sigma, *_ = assemble_mechanics(
        state.mesh,
        state.displacement,
        state.ep_gp,
        state.rho_gp,
        state.damage,
        state.elasticity_D,
        state.material,
        cohesive_network=state.cohesive_network,
    )
    damage = getattr(state.mesh, "element_damage_gp", None)
    solid = None if damage is None else np.asarray(damage, dtype=float) < 0.5
    return recover_fixed_physical_arc_patch_v3(
        nodes=state.mesh.nodes,
        elements=state.mesh.elems,
        stress_Pa=sigma,
        boundary_node=node,
        cavity_id=cavity.cavity_id,
        cavity_center_m=cavity.center_m,
        cavity_radius_m=cavity.radius_m,
        process_length_m=engine.f.L_pz,
        poisson_ratio=state.material.nu,
        owned_boundary_edges=_actual_cavity_boundary_edges(state),
        material_fingerprints=identity,
        state_fingerprint=_cavity_resolution_binding(state),
        solid_element_mask=solid,
    )


def cavity_source_conforming_recovery_v4(state, *, boundary_node: int | None = None):
    """Apply V3 WLS only after the prospective V4 geometry certificate passes."""
    from .cavity_source_conforming_geometry_v4 import recover_source_conforming_geometry_v4

    cavity = state.void_state.cavities[0]
    position = np.asarray(cavity.connection_exit_m, dtype=float)
    if boundary_node is None:
        boundary_node = int(
            np.argmin(np.linalg.norm(np.asarray(state.mesh.nodes) - position, axis=1))
        )
    node = int(boundary_node)
    if np.linalg.norm(np.asarray(state.mesh.nodes)[node] - position) > 1.0e-12:
        raise ValueError("stored connection exit is not an aligned cavity-boundary node")
    identity = dict(require_bound_identity(state))
    engine = fresh_sharp_front_engine(state.material, _bundle_from_state(state))
    _, _, sigma, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    damage = getattr(state.mesh, "element_damage_gp", None)
    solid = None if damage is None else np.asarray(damage, dtype=float) < 0.5
    return recover_source_conforming_geometry_v4(
        nodes=state.mesh.nodes, elements=state.mesh.elems, stress_Pa=sigma,
        boundary_node=node, cavity_id=cavity.cavity_id,
        cavity_center_m=cavity.center_m, cavity_radius_m=cavity.radius_m,
        process_length_m=engine.f.L_pz, poisson_ratio=state.material.nu,
        owned_boundary_edges=_actual_cavity_boundary_edges(state),
        material_fingerprints=identity, state_fingerprint=_cavity_resolution_binding(state),
        solid_element_mask=solid,
    )


def cavity_boundary_fitted_recovery_v1(state, *, boundary_node: int | None = None):
    """Recover on the exact accepted polygon without a circle projection."""
    from .cavity_boundary_fitted_source_v1 import (
        recover_boundary_fitted_physical_patch_v1,
    )
    cavity = state.void_state.cavities[0]
    position = np.asarray(cavity.connection_exit_m, dtype=float)
    if boundary_node is None:
        boundary_node = int(np.argmin(
            np.linalg.norm(np.asarray(state.mesh.nodes)-position, axis=1)
        ))
    node = int(boundary_node)
    if np.linalg.norm(np.asarray(state.mesh.nodes)[node]-position) > 1.0e-12:
        raise ValueError("stored connection exit is not an aligned cavity-boundary node")
    identity = dict(require_bound_identity(state))
    engine = fresh_sharp_front_engine(state.material, _bundle_from_state(state))
    _, _, sigma, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    damage = getattr(state.mesh, "element_damage_gp", None)
    solid = None if damage is None else np.asarray(damage, dtype=float) < 0.5
    return recover_boundary_fitted_physical_patch_v1(
        nodes=state.mesh.nodes, elements=state.mesh.elems, stress_Pa=sigma,
        boundary_node=node, cavity_id=cavity.cavity_id,
        cavity_center_m=cavity.center_m, cavity_radius_m=cavity.radius_m,
        process_length_m=engine.f.L_pz, poisson_ratio=state.material.nu,
        owned_boundary_edges=_actual_cavity_boundary_edges(state),
        material_fingerprints=identity,
        state_fingerprint=_cavity_resolution_binding(state),
        solid_element_mask=solid,
    )


def qualified_boundary_fitted_source_observation(state, *, refinement_levels=(1, 2)):
    """Qualify the polygon-fitted source on isolated nested local refinements."""
    from .adaptive_multitip_mesh_v11 import refine_accepted_state
    from .causal_sharp_wake_v11 import rerasterize_refined_topology_damage
    from .cavity_source_recovery_v3 import (
        MINIMUM_ELEMENT_QUALITY, TENSOR_RELATIVE_TOLERANCE,
        TRACTION_RESIDUAL_TOLERANCE,
    )
    from .topology_transaction_v11 import (
        complete_accepted_state_fingerprint,
        equilibrate_fixed_load_with_production_fem,
    )
    levels = tuple(sorted(set(map(int, refinement_levels))))
    if len(levels) < 2 or levels[0] < 1:
        raise ValueError("boundary-fitted source qualification requires two refined levels")
    accepted_before = complete_accepted_state_fingerprint(state)
    trial = state.isolated_copy()
    source = np.asarray(trial.void_state.cavities[0].connection_exit_m, float)
    records = []
    completed = 0
    for requested_level in levels:
        while completed < requested_level:
            centroids = np.asarray(trial.mesh.nodes)[np.asarray(trial.mesh.elems)].mean(axis=1)
            marked = np.flatnonzero(np.linalg.norm(
                centroids-source[None, :], axis=1,
            ) <= 6.0e-5)
            if not len(marked):
                raise RuntimeError("boundary-fitted source refinement has no local support")
            refined, lineage = refine_accepted_state(
                trial, marked_parent_elements=tuple(map(int, marked)),
                active_tip_ids=trial.crack_network.active_tip_ids,
                generation=int(trial.event_counters.get("mesh_generation", 0))+1,
                operation_index=int(trial.event_counters.get(
                    "refinement_operation_index", 0))+1,
                rebuild_vacancy_transport=False,
            )
            trial, _ = rerasterize_refined_topology_damage(
                trial, refined, lineage.parent_to_child_element_map,
            )
            trial = equilibrate_fixed_load_with_production_fem(trial)
            completed += 1
        recovery = cavity_boundary_fitted_recovery_v1(trial)
        records.append({
            "refinement_level": requested_level,
            "mesh_nodes": int(trial.mesh.nn), "mesh_elements": int(trial.mesh.ne),
            "tensor_Pa": recovery["tensor_Pa"],
            "sigma_tt_Pa": recovery["sigma_tt_Pa"],
            "minimum_element_quality": recovery["minimum_element_quality"],
            "traction_residual": recovery["traction_residual"],
            "sample_count": recovery["sample_count"],
            "recovery": recovery,
        })
    first = np.asarray(records[-2]["tensor_Pa"], float)
    finest = np.asarray(records[-1]["tensor_Pa"], float)
    tensor_error = float(np.linalg.norm(finest-first)/max(
        np.linalg.norm(first), np.linalg.norm(finest), 1.0e-300,
    ))
    quality_passed = all(
        row["minimum_element_quality"] >= MINIMUM_ELEMENT_QUALITY for row in records[-2:]
    )
    traction_passed = all(
        row["traction_residual"] <= TRACTION_RESIDUAL_TOLERANCE for row in records[-2:]
    )
    accepted_preserved = complete_accepted_state_fingerprint(state) == accepted_before
    passed = bool(
        tensor_error <= TENSOR_RELATIVE_TOLERANCE and quality_passed
        and traction_passed and accepted_preserved
    )
    return {
        **records[-1]["recovery"],
        "qualification_schema": "v5.boundary-fitted-source-qualification/1",
        "qualification_status": "AVAILABLE" if passed else "UNAVAILABLE",
        "tensor_relative_error": tensor_error,
        "tensor_relative_limit": TENSOR_RELATIVE_TOLERANCE,
        "quality_passed": quality_passed, "traction_passed": traction_passed,
        "accepted_state_preserved": accepted_preserved,
        "physical_time_advanced": False, "transport_solve_performed": False,
        "refinement_records": tuple(records),
    }


def _boundary_fitted_source_probe(state, recovery, *, boundary_node: int):
    """Create a state-bound proof record from the measured nested observation."""
    from .finalization_v3_schema import canonical_hash
    from .topology_transaction_v11 import complete_accepted_state_fingerprint
    proof = {
        "schema": "v5.boundary-fitted-source-qualification-proof/1",
        "accepted_state_fingerprint": complete_accepted_state_fingerprint(state),
        "qualification_status": recovery["qualification_status"],
        "tensor_Pa": recovery["tensor_Pa"],
        "tensor_relative_error": recovery["tensor_relative_error"],
        "tensor_relative_limit": recovery["tensor_relative_limit"],
        "quality_passed": recovery["quality_passed"],
        "traction_passed": recovery["traction_passed"],
        "accepted_state_preserved": recovery["accepted_state_preserved"],
        "physical_arc_identity_sha256": recovery["physical_arc_identity"]["sha256"],
        "refinement_levels": [row["refinement_level"]
                              for row in recovery["refinement_records"]],
    }
    proof["sha256"] = canonical_hash(proof)
    return {
        "kind": "CAVITY_BOUNDARY_FITTED_PHYSICAL_PATCH_V1",
        "boundary_node_id": int(boundary_node),
        "element_ids": list(map(int, recovery["stencil_element_ids"])),
        "physical_arc_identity": recovery["physical_arc_identity"],
        "qualification_proof": proof,
    }


def _boundary_fitted_probe_is_qualified(state, tensor, probe) -> bool:
    """Verify the internally generated nested proof against this exact state/load."""
    from .finalization_v3_schema import canonical_hash
    from .topology_transaction_v11 import complete_accepted_state_fingerprint
    if not isinstance(probe, Mapping) or probe.get("kind") != (
            "CAVITY_BOUNDARY_FITTED_PHYSICAL_PATCH_V1"):
        return False
    proof = probe.get("qualification_proof")
    if not isinstance(proof, Mapping):
        return False
    unsigned = {key: value for key, value in proof.items() if key != "sha256"}
    checks = (
        proof.get("schema") == "v5.boundary-fitted-source-qualification-proof/1",
        proof.get("sha256") == canonical_hash(unsigned),
        proof.get("accepted_state_fingerprint") == complete_accepted_state_fingerprint(state),
        proof.get("qualification_status") == "AVAILABLE",
        bool(proof.get("quality_passed")), bool(proof.get("traction_passed")),
        bool(proof.get("accepted_state_preserved")),
        float(proof.get("tensor_relative_error", math.inf))
            <= float(proof.get("tensor_relative_limit", -math.inf)),
        tuple(proof.get("refinement_levels", ())) == (1, 2),
        np.array_equal(np.asarray(proof.get("tensor_Pa"), float), np.asarray(tensor, float)),
        proof.get("physical_arc_identity_sha256")
            == probe.get("physical_arc_identity", {}).get("sha256"),
    )
    return bool(all(checks))


def cavity_free_surface_certificate(state) -> dict[str, Any]:
    """Certify one connected, closed mesh-boundary cycle at the cavity."""
    cavity = state.void_state.cavities[0]
    nodes = np.asarray(state.mesh.nodes)
    radii = np.linalg.norm(nodes - np.asarray(cavity.center_m), axis=1)
    cavity_nodes = set(map(int, np.flatnonzero(radii <= cavity.radius_m * 1.02)))
    counts: dict[tuple[int, int], int] = {}
    for triangle in np.asarray(state.mesh.elems, dtype=int):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = tuple(sorted((int(a), int(b))))
            counts[edge] = counts.get(edge, 0) + 1
    edges = tuple(edge for edge, count in counts.items() if count == 1 and set(edge).issubset(cavity_nodes))
    adjacency = {node: set() for edge in edges for node in edge}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    visited = set()
    stack = [next(iter(adjacency))] if adjacency else []
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(adjacency[node] - visited)
    passed = len(edges) >= 16 and visited == set(adjacency) and all(len(peers) == 2 for peers in adjacency.values())
    return {"passed": passed, "boundary_edge_count": len(edges),
            "boundary_node_ids": sorted(adjacency), "boundary_edge_ids": [list(edge) for edge in edges],
            "boundary_node_count": len(adjacency),
            "connected_component_count": 1 if passed else None,
            "topology": "single_closed_cavity_boundary_cycle"}


def _polygon_boundary_arc_length(state, first_point, second_point) -> float:
    cycle = cavity_free_surface_certificate(state)
    nodes = np.asarray(state.mesh.nodes)
    first = min(cycle["boundary_node_ids"], key=lambda node: np.linalg.norm(nodes[node] - first_point))
    second = min(cycle["boundary_node_ids"], key=lambda node: np.linalg.norm(nodes[node] - second_point))
    adjacency = {node: [] for node in cycle["boundary_node_ids"]}
    for a, b in cycle["boundary_edge_ids"]:
        length = float(np.linalg.norm(nodes[a] - nodes[b]))
        adjacency[a].append((b, length)); adjacency[b].append((a, length))
    lengths = []
    for initial in adjacency[first]:
        previous, current, total = first, initial[0], initial[1]
        while current != second:
            following = next(item for item in adjacency[current] if item[0] != previous)
            previous, current, total = current, following[0], total + following[1]
        lengths.append(total)
    return min(lengths)


def _ordered_cavity_polygon(state, cycle):
    adjacency = {int(node): [] for node in cycle["boundary_node_ids"]}
    for a, b in cycle["boundary_edge_ids"]:
        adjacency[int(a)].append(int(b)); adjacency[int(b)].append(int(a))
    first = min(adjacency)
    ordered = [first]
    previous, current = None, first
    while True:
        choices = sorted(node for node in adjacency[current] if node != previous)
        following = choices[0]
        if following == first:
            break
        ordered.append(following)
        previous, current = current, following
    return np.asarray(state.mesh.nodes, dtype=float)[ordered]


def _convex_polygon_interior_overlaps_triangle(polygon, triangle, tolerance=1.0e-14):
    """Exact edge-axis test; boundary contact alone is not interior overlap."""
    polygon = np.asarray(polygon, dtype=float)
    triangle = np.asarray(triangle, dtype=float)
    axes = []
    for vertices in (polygon, triangle):
        for first, second in zip(vertices, np.roll(vertices, -1, axis=0)):
            edge = second - first
            axes.append(np.asarray((-edge[1], edge[0]), dtype=float))
    for axis in axes:
        norm = float(np.linalg.norm(axis))
        if norm <= tolerance:
            continue
        axis /= norm
        poly_projection = polygon @ axis
        tri_projection = triangle @ axis
        overlap = min(float(np.max(poly_projection)), float(np.max(tri_projection))) - max(
            float(np.min(poly_projection)), float(np.min(tri_projection))
        )
        if overlap <= tolerance:
            return False
    return True


def crack_void_connection_certificate(state, *, branch_id: str, cavity_id: str,
                                      intended_intersection) -> dict[str, Any]:
    """Certify combined graph/cavity incidence, clearance, and free boundary."""
    cavities = [item for item in state.void_state.cavities if item.cavity_id == cavity_id]
    if len(cavities) != 1:
        raise ValueError("combined topology certificate requires one exact cavity identity")
    cavity = cavities[0]
    branch = state.crack_network.branch(branch_id)
    endpoint = np.asarray(branch.tip)
    intended = np.asarray(intended_intersection, dtype=float)
    center = np.asarray(cavity.center_m)
    cycle = cavity_free_surface_certificate(state)
    endpoint_matches = bool(np.linalg.norm(endpoint - intended) <= 1.0e-12)
    endpoint_on_boundary = bool(abs(np.linalg.norm(endpoint - center) - cavity.radius_m) <= cavity.radius_m * 0.02)
    def segment_open_disk_distance(first, second):
        a = np.asarray(first); b = np.asarray(second); delta = b - a
        fraction = float(np.clip((center - a) @ delta / max(delta @ delta, 1.0e-300), 0.0, 1.0))
        return float(np.linalg.norm(a + fraction * delta - center))
    segment_rows = [{"branch_id": item.branch_id, "segment_index": index,
                     "minimum_center_distance_m": segment_open_disk_distance(first, second),
                     "intersects_cavity_open_disk": segment_open_disk_distance(first, second) < cavity.radius_m * (1.0 - 1.0e-10)}
                    for item in state.crack_network.branches
                    for index, (first, second) in enumerate(zip(item.path, item.path[1:]))]
    graph_outside = not any(row["intersects_cavity_open_disk"] for row in segment_rows)
    support = state.v12_support_state
    support_ids = () if support is None else support.selected_support_elements
    centroids = np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)].mean(axis=1)
    open_disk_overlap_ids = [int(element) for element in support_ids if triangle_intersects_open_disk(
        np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)[int(element)]], center, cavity.radius_m,
    )]
    cavity_polygon = _ordered_cavity_polygon(state, cycle)
    polygon_overlap_ids = [int(element) for element in support_ids
                           if _convex_polygon_interior_overlaps_triangle(
                               cavity_polygon,
                               np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)[int(element)]],
                           )]
    support_outside = not polygon_overlap_ids
    # Retain sampled centerline coverage as a supplemental diagnostic, while
    # the pass/fail decision uses the independent V12 intact path certificate.
    ligament_start = np.asarray(branch.path[-2])
    samples = np.linspace(ligament_start, endpoint, 65)[1:-1]
    support_triangles = np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)[list(support_ids)]] if support_ids else np.empty((0, 3, 2))
    def point_in_triangle(point, triangle):
        matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
        if abs(np.linalg.det(matrix)) <= 1.0e-24: return False
        uv = np.linalg.solve(matrix, point - triangle[0])
        return uv[0] >= -1.0e-10 and uv[1] >= -1.0e-10 and uv.sum() <= 1.0 + 1.0e-10
    covered = [any(point_in_triangle(sample, triangle) for triangle in support_triangles) for sample in samples]
    uncovered_indices = [index for index, value in enumerate(covered) if not value]
    no_solid_bridge = not uncovered_indices
    intact_certificate = (independent_intact_path_certificate(
        state.mesh, state.crack_network, support_ids,
        boundary_terminal_context=state.junction_process_state.get("boundary_terminal_context"),
    ) if support_ids else {
        "intact_cross_graph_path_exists": True,
        "insufficient_seed_segment_ids": ("all:no-support",),
        "bridge_node_ids": (), "bridge_element_ids": (),
        "edge_cut_certificates": (), "certificate_fingerprint": "NO_SUPPORT",
    })
    intact_certificate = {
        key: ([asdict(item) for item in value] if key == "edge_cut_certificates" else value)
        for key, value in intact_certificate.items()
    }
    exact_no_bridge = bool(
        not intact_certificate["intact_cross_graph_path_exists"]
        and not intact_certificate["insufficient_seed_segment_ids"]
    )
    # Build and traverse the actual branch/cavity incidence graph.
    component_graph = {"cavity:" + cavity_id: set()}
    incidence_edges = []
    boundary_nodes = np.asarray(cycle["boundary_node_ids"], dtype=int)
    boundary_points = np.asarray(state.mesh.nodes)[boundary_nodes]
    intersected_edge_id = None
    for item in state.crack_network.branches:
        key = "branch:" + item.branch_id; component_graph.setdefault(key, set())
        incident_points = (item.path[0], item.tip)
        incident = any(len(boundary_points) and float(np.min(
            np.linalg.norm(boundary_points - np.asarray(point), axis=1)
        )) <= 1.0e-12 for point in incident_points)
        if incident:
            cavity_key = "cavity:" + cavity_id
            component_graph[key].add(cavity_key); component_graph[cavity_key].add(key)
            incidence_edges.append([key, cavity_key])
    unseen = set(component_graph); components = []
    while unseen:
        todo = [min(unseen)]; component = []
        while todo:
            key = todo.pop()
            if key not in unseen: continue
            unseen.remove(key); component.append(key)
            todo.extend(sorted(component_graph[key], reverse=True))
        components.append(sorted(component))
    combined_count = len(components)
    for edge in cycle["boundary_edge_ids"]:
        a, b = np.asarray(state.mesh.nodes)[edge]
        if np.linalg.norm(endpoint - a) <= 1.0e-12 or np.linalg.norm(endpoint - b) <= 1.0e-12:
            intersected_edge_id = edge; break
    passed = bool(cycle["passed"] and endpoint_matches and endpoint_on_boundary
                  and graph_outside and support_outside and no_solid_bridge
                  and exact_no_bridge and combined_count == 1)
    return {
        "passed": passed, "cavity_id": cavity_id, "branch_id": branch_id,
        "branch_endpoint_m": endpoint.tolist(), "intended_intersection_m": intended.tolist(),
        "endpoint_matches_intersection": endpoint_matches, "endpoint_on_cavity_boundary": endpoint_on_boundary,
        "no_surviving_solid_ligament_bridge": no_solid_bridge,
        "exact_no_intact_node_or_element_path": exact_no_bridge,
        "independent_intact_path_certificate": intact_certificate,
        "bridge_search_sample_count": len(samples), "bridge_search_uncovered_sample_indices": uncovered_indices,
        "intersected_cavity_edge_id": intersected_edge_id,
        "crack_graph_outside_cavity": graph_outside, "wake_support_outside_cavity": support_outside,
        "crack_segment_cavity_intersections": segment_rows,
        "support_triangle_cavity_overlap_element_ids": polygon_overlap_ids,
        "supplemental_open_disk_overlap_element_ids": open_disk_overlap_ids,
        "exact_triangle_polygon_interior_overlap_absent": not polygon_overlap_ids,
        "combined_incidence_edges": incidence_edges, "combined_components": components,
        "active_tip_ids": list(state.crack_network.active_tip_ids),
        "combined_incidence_component_count": combined_count,
        "closed_cavity_boundary_cycle": cycle,
        "topology": "connected_crack_graph_and_traction_free_cavity_cycle",
    }


def _polygon_area(vertices):
    polygon = np.asarray(vertices, dtype=float)
    if len(polygon) < 3:
        return 0.0
    return 0.5 * abs(float(np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1)
        - polygon[:, 1] * np.roll(polygon[:, 0], -1)
    )))


def _triangle_overlap_area(first, second):
    """Convex polygon clipping area for two consistently unordered triangles."""
    def cross2(first_vector, second_vector):
        return float(first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0])
    subject = [np.asarray(point, dtype=float) for point in first]
    clip = np.asarray(second, dtype=float)
    orientation = np.sign(cross2(clip[1] - clip[0], clip[2] - clip[0])) or 1.0
    for edge_index in range(3):
        start = clip[edge_index]; end = clip[(edge_index + 1) % 3]
        edge = end - start
        def signed(point):
            return orientation * cross2(edge, point - start)
        output = []
        if not subject:
            break
        previous = subject[-1]; previous_value = signed(previous)
        for current in subject:
            current_value = signed(current)
            previous_inside = previous_value >= -1.0e-18
            current_inside = current_value >= -1.0e-18
            if current_inside != previous_inside:
                denominator = previous_value - current_value
                fraction = 0.0 if denominator == 0.0 else previous_value / denominator
                output.append(previous + fraction * (current - previous))
            if current_inside:
                output.append(current)
            previous = current; previous_value = current_value
        subject = output
    return _polygon_area(subject)


def _overlap_conservative_p0(state, mesh, values):
    """Element-overlap P0 transfer, with bounded local redistribution on domain loss."""
    old_triangles = np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)]
    new_triangles = np.asarray(mesh.nodes)[np.asarray(mesh.elems)]
    old_values = np.asarray(values, dtype=float)
    old_centers = old_triangles.mean(axis=1)
    old_radii = np.max(np.linalg.norm(old_triangles - old_centers[:, None, :], axis=2), axis=1)
    tree = cKDTree(old_centers)
    maximum_old_radius = float(np.max(old_radii))
    projected = np.zeros(mesh.ne)
    covered = np.zeros(mesh.ne)
    for new_index, triangle in enumerate(new_triangles):
        center = triangle.mean(axis=0)
        radius = float(np.max(np.linalg.norm(triangle - center, axis=1)))
        candidates = np.asarray(tree.query_ball_point(
            center, radius + maximum_old_radius + 1.0e-15), dtype=int)
        if len(candidates):
            offsets = old_centers[candidates]-center
            bounds = radius+old_radii[candidates]+1.0e-15
            candidates = candidates[np.einsum(
                "ij,ij->i", offsets, offsets) <= bounds*bounds]
        integral = 0.0; coverage = 0.0
        for old_index in candidates:
            overlap = _triangle_overlap_area(triangle, old_triangles[old_index])
            if overlap > 0.0:
                coverage += overlap
                integral += overlap * old_values[old_index]
        if coverage <= 0.0:
            _, nearest = tree.query(center)
            projected[new_index] = old_values[int(nearest)]
        else:
            projected[new_index] = integral / coverage
        covered[new_index] = coverage
    old_inventory = math.fsum(
        float(value * area) for value, area in zip(old_values, state.mesh.area_e)
    )
    new_inventory = math.fsum(
        float(value * area) for value, area in zip(projected, mesh.area_e)
    )
    difference = old_inventory - new_inventory
    # A growing hole can remove old solid elements. Their already-owned
    # vacancies are redistributed only within the cavity capture shell, never
    # by a whole-domain multiplier.
    recipients = np.arange(mesh.ne, dtype=int)
    if difference > 0.0 and state.void_state is not None and state.void_state.cavities:
        cavity = state.void_state.cavities[0]
        center = np.asarray(cavity.center_m)
        centroids = new_triangles.mean(axis=1)
        distance = np.linalg.norm(centroids - center, axis=1)
        process_length = fresh_sharp_front_engine(
            state.material, _bundle_from_state(state),
        ).f.L_pz
        recipients = np.flatnonzero(distance <= cavity.radius_m + process_length)
        if not len(recipients):
            recipients = np.asarray((int(np.argmin(abs(distance - cavity.radius_m))),))
        recipient_area = float(np.sum(np.asarray(mesh.area_e)[recipients]))
        projected[recipients] += difference / recipient_area
    elif abs(difference) > 1.0e-24:
        # Same-domain overlap should close without a global shape-changing
        # correction. Fail if the geometric projection cannot prove it.
        from .vacancy_coupling_v1 import conservation_roundoff_budget_m2
        if abs(difference) > conservation_roundoff_budget_m2(old_inventory, new_inventory):
            raise RuntimeError("overlap-conservative P0 transfer lost domain inventory")
    transferred = math.fsum(
        float(value * area) for value, area in zip(projected, mesh.area_e)
    )
    residual = old_inventory - transferred
    from .vacancy_coupling_v1 import conservation_roundoff_budget_m2
    budget = conservation_roundoff_budget_m2(old_inventory, transferred)
    if abs(residual) > budget:
        raise RuntimeError("overlap-conservative P0 transfer exceeds numerical budget")
    if residual != 0.0:
        target = int(recipients[np.argmax(np.asarray(mesh.area_e)[recipients])])
        projected[target] += residual / float(mesh.area_e[target])
    return projected, {
        "method": "ELEMENT_OVERLAP_CONSERVATIVE_P0",
        "source_inventory_area_m2": old_inventory,
        "pre_redistribution_inventory_area_m2": new_inventory,
        "locally_redistributed_area_m2": max(difference, 0.0),
        "roundoff_correction_m2": residual,
        "roundoff_budget_m2": budget,
        "minimum_new_element_overlap_fraction": float(np.min(
            covered / np.asarray(mesh.area_e)
        )),
    }


def _project_fields(state, mesh, *, transfer_v2_transport=True):
    old_nodes = np.asarray(state.mesh.nodes); new_nodes = np.asarray(mesh.nodes)
    _, node_parent = cKDTree(old_nodes).query(new_nodes)
    old_centers = old_nodes[state.mesh.elems].mean(axis=1)
    new_centers = new_nodes[mesh.elems].mean(axis=1)
    _, element_parent = cKDTree(old_centers).query(new_centers)
    def p1(values):
        source = np.asarray(values, dtype=float)
        projected = np.asarray(LinearNDInterpolator(old_nodes, source)(new_nodes))
        missing = ~np.all(np.isfinite(projected), axis=1) if projected.ndim == 2 else ~np.isfinite(projected)
        projected[missing] = source[node_parent[missing]]
        return projected

    # Transfer non-topological P0 damage, but never turn the old mesh width of
    # an owned V12 crack representation into permanent material damage.  The
    # authoritative graph is rebuilt on the destination mesh by the V12
    # remesh transaction.  Clearing precisely the owned support here preserves
    # unrelated damage while preventing a coarse node-star halo ahead of a tip
    # from swallowing a later short event.
    source_damage_gp = element_damage(state.mesh, state.damage)
    support = state.v12_support_state
    if support is not None:
        from .sharp_wake_backend_v12 import array_fingerprint
        if (
            support.mesh_geometry_fingerprint == array_fingerprint(state.mesh.nodes)
            and support.mesh_connectivity_fingerprint == array_fingerprint(state.mesh.elems)
        ):
            source_damage_gp = source_damage_gp.copy()
            owned = np.asarray(support.selected_support_elements, dtype=int)
            if np.any(owned < 0) or np.any(owned >= len(source_damage_gp)):
                raise RuntimeError("V12 remesh ownership contains an invalid source element")
            source_damage_gp[owned] = 0.0
    projected_damage_gp = np.asarray(source_damage_gp, dtype=float)[element_parent]
    ep = np.asarray(state.ep_gp, dtype=float)[:, element_parent]
    rho = np.asarray(state.rho_gp, dtype=float)[element_parent]
    vacancy, vacancy_transfer_audit = _overlap_conservative_p0(
        state, mesh, state.vacancy_fraction_gp,
    )
    new_area = np.asarray(mesh.area_e, dtype=float)
    old_area = np.asarray(state.mesh.area_e, dtype=float)
    for component in range(ep.shape[0]):
        source_integral = float(np.sum(np.asarray(state.ep_gp)[component] * old_area))
        projected_integral = float(np.sum(ep[component] * new_area))
        if abs(projected_integral) > 1.0e-300:
            ep[component] *= source_integral / projected_integral
    source_rho = float(np.sum(np.asarray(state.rho_gp) * old_area))
    projected_rho = float(np.sum(rho * new_area))
    if abs(projected_rho) > 1.0e-300:
        rho *= source_rho / projected_rho
    if np.any(vacancy < 0.0):
        raise RuntimeError("P0 vacancy remesh transfer created a negative fraction")
    result = {
        "damage": np.clip(p1(np.asarray(state.damage)), 0.0, 1.0),
        "element_damage_gp": np.clip(projected_damage_gp, 0.0, 1.0),
        "displacement": p1(np.asarray(state.displacement).reshape(-1, 2)).reshape(-1),
        "ep_gp": ep,
        "rho_gp": rho,
        "vacancy_fraction_gp": vacancy,
        "vacancy_ledgers": state.vacancy_ledgers,
        "vacancy_temperature_K": state.vacancy_temperature_K,
        "vacancy_transfer_audit": vacancy_transfer_audit,
        "tip_process_state": state.tip_process_state,
        "source_state": state.junction_process_state.get("source_state", {}),
    }
    if transfer_v2_transport and state.vacancy_transport_state_v2 is not None:
        from .vacancy_transport_phase_field_v2 import (
            build_exact_cut_transport_mesh_for_network, inherit_topology_boundary_tags,
            transfer_transport_state_overlap,
        )
        inherited_tags = inherit_topology_boundary_tags(state, mesh)
        junction = dict(state.junction_process_state)
        junction["transport_boundary_tags_v2"] = inherited_tags
        segments = tuple(
            (first, second, f"{branch.branch_id}:{index}")
            for branch in state.crack_network.branches
            for index, (first, second) in enumerate(zip(branch.path, branch.path[1:]))
        )
        cut = build_exact_cut_transport_mesh_for_network(
            mesh, crack_segments=segments,
            topology_boundary_tags=inherited_tags,
        )
        transport, transport_audit = transfer_transport_state_overlap(
            state.vacancy_transport_state_v2,
            state.vacancy_transport_state_v2.transport_mesh, cut.mesh, cut.registry,
            new_parent_mechanics_element_ids=cut.parent_mechanics_element_ids,
        )
        result.update({
            "vacancy_transport_state_v2": transport,
            "transport_boundary_tags_v2": inherited_tags,
            "vacancy_transport_v2_transfer_audit": transport_audit,
        })
    return result


def _grow_hole_boundary(hole, radius_m, *, crack_path_m):
    """Rebuild all radial layers, then conform the unchanged physical crack.

    Moving only the boundary can overtake the first solid layer and invert
    triangles on a fine mesh. No boundary/topology tolerance can repair that.
    """
    source_conforming = (
        hole.validation.get("geometry_contract")
        == "CAVITY_SOURCE_CONFORMING_GEOMETRY_V4"
    )
    shape_regular_level = (
        hole.validation.get("local_level")
        if hole.validation.get("geometry_contract") == SHAPE_REGULAR_MESH_V5_ID
        else None
    )
    grown, _ = _geometry(
        radius_m, hole.center_m,
        boundary_segments=len(hole.prescribed_polygon_nodes),
        radial_layers=int(hole.validation["radial_layers"]),
        source_conforming_geometry=source_conforming,
        source_direction_xy=hole.validation.get("source_direction_xy", (1.0, 0.0)),
        shape_regular_local_level=shape_regular_level,
    )
    mesh = grown.mesh
    for point in crack_path_m:
        mesh = _insert_point_in_mesh(mesh, point)
    return replace(grown, mesh=mesh)


def remesh_cavity(state, hole, void_state, identity, operation_log=None, failure_stage=None):
    operations = operation_log if operation_log is not None else []
    trial = replace(state, void_state=void_state)
    fields = _project_fields(trial, hole.mesh, transfer_v2_transport=False)
    remesh_junction = dict(trial.junction_process_state)
    remesh_junction["transport_boundary_tags_v2"] = _owned_transport_boundary_tags(
        hole.mesh, hole.boundary, cavity_node_ids=hole.prescribed_polygon_nodes,
    )
    transport_trial = replace(
        trial, mesh=hole.mesh, boundary=hole.boundary,
        junction_process_state=remesh_junction,
        vacancy_fraction_gp=fields["vacancy_fraction_gp"],
        vacancy_transport_state_v2=None,
    )
    if trial.vacancy_transport_state_v2 is not None:
        from .vacancy_transport_phase_field_v2 import (
            build_exact_cut_transport_mesh_for_network, transfer_transport_state_overlap,
        )
        segments = tuple(
            (first, second, f"{branch.branch_id}:{index}")
            for branch in trial.crack_network.branches
            for index, (first, second) in enumerate(zip(branch.path, branch.path[1:]))
        )
        target_cut = build_exact_cut_transport_mesh_for_network(
            hole.mesh, crack_segments=segments,
            topology_boundary_tags=remesh_junction["transport_boundary_tags_v2"],
        )
        promoted_site_ids = {item.parent_site_id for item in trial.void_state.cavities}
        transport_source = replace(
            trial.vacancy_transport_state_v2,
            site_states=tuple(item for item in trial.vacancy_transport_state_v2.site_states
                              if item.site_id not in promoted_site_ids),
        )
        transferred_transport, transport_transfer_audit = transfer_transport_state_overlap(
            transport_source,
            transport_source.transport_mesh,
            target_cut.mesh, target_cut.registry,
            domain_change_policy="ACTIVE_CAVITY_ABSORPTION_ONE_WAY",
            new_parent_mechanics_element_ids=target_cut.parent_mechanics_element_ids,
        )
        fields["vacancy_transport_state_v2"] = transferred_transport
    else:
        transport_transfer_audit = None
    vacancy_local_audits = []
    from .vacancy_coupling_v1 import deterministic_capture_patch
    capture_length = fresh_sharp_front_engine(
        trial.material, _bundle_from_state(trial),
    ).f.L_pz
    capture_owners = [
        (site.site_id, site.center_m, capture_length) for site in trial.void_state.sites
    ] + [
        (cavity.cavity_id, cavity.center_m, cavity.radius_m + capture_length)
        for cavity in trial.void_state.cavities
    ]
    projected_state = replace(
        trial, mesh=hole.mesh, vacancy_fraction_gp=fields["vacancy_fraction_gp"],
    )
    for owner_id, center, radius in capture_owners:
        old_patch = deterministic_capture_patch(trial, center, radius)
        new_patch = deterministic_capture_patch(projected_state, center, radius)
        old_local = math.fsum(float(trial.mesh.area_e[index] * trial.vacancy_fraction_gp[index]) for index in old_patch)
        new_local = math.fsum(float(hole.mesh.area_e[index] * fields["vacancy_fraction_gp"][index]) for index in new_patch)
        vacancy_local_audits.append({
            "owner_id": owner_id, "capture_radius_m": radius,
            "source_capture_inventory_area_m2": old_local,
            "projected_capture_inventory_area_m2": new_local,
            "capture_inventory_change_area_m2": new_local - old_local,
        })
    transfer_audit = {
        "source_displacement_l2_m": float(np.linalg.norm(state.displacement)),
        "projected_displacement_l2_m": float(np.linalg.norm(fields["displacement"])),
        "source_plastic_history_l2": float(np.linalg.norm(state.ep_gp)),
        "projected_plastic_history_l2": float(np.linalg.norm(fields["ep_gp"])),
        "source_density_l2_m2": float(np.linalg.norm(state.rho_gp)),
        "projected_density_l2_m2": float(np.linalg.norm(fields["rho_gp"])),
        "source_vacancy_inventory_area_m2": float(np.sum(
            np.asarray(state.vacancy_fraction_gp) * np.asarray(state.mesh.area_e)
        )),
        "projected_vacancy_inventory_area_m2": float(np.sum(
            np.asarray(fields["vacancy_fraction_gp"]) * np.asarray(hole.mesh.area_e)
        )),
        "plastic_integral_error": float(np.max(np.abs(
            np.sum(fields["ep_gp"] * hole.mesh.area_e[None, :], axis=1)
            - np.sum(state.ep_gp * state.mesh.area_e[None, :], axis=1)
        ))),
        "density_integral_error": float(abs(
            np.sum(fields["rho_gp"] * hole.mesh.area_e)
            - np.sum(state.rho_gp * state.mesh.area_e)
        )),
        "projected_fields_nonzero": bool(
            np.linalg.norm(fields["displacement"]) > 0.0 and np.linalg.norm(fields["ep_gp"]) > 0.0
            and np.linalg.norm(fields["rho_gp"]) > 0.0
        ),
        "vacancy_transfer": fields["vacancy_transfer_audit"],
        "active_capture_region_inventory": vacancy_local_audits,
        "vacancy_transport_v2_transfer": transport_transfer_audit,
    }
    def inject(stage, current):
        operations.append(stage)
        if stage == failure_stage: raise RuntimeError("injected:" + stage)
    if identity == "promotion": inject("explicit_cavity_creation", trial)
    remesh_junction["boundary_terminal_context"] = _external_free_root_context(
        hole.mesh, hole.boundary, trial.crack_network.branch(ROOT_BRANCH_ID).root,
    )
    trial = replace(trial, junction_process_state=remesh_junction)
    rebuilt = remesh_mechanically_separating_v12(
        trial, mesh=hole.mesh, boundary=hole.boundary, transferred_fields=fields,
        source_commit=_head(), configuration={"voiding_enabled": True},
        transaction_identity=identity, failure_injector=inject,
    )
    junction = dict(rebuilt.junction_process_state)
    junction["latest_void_remesh_audit"] = transfer_audit
    rebuilt = replace(rebuilt, junction_process_state=junction)
    solved = equilibrate_fixed_load_with_production_fem(rebuilt)
    operations.append("equilibrium")
    if failure_stage == "equilibrium": raise RuntimeError("injected:equilibrium")
    return solved


def _tensor_fingerprint(tensor) -> str:
    return hashlib.sha256(np.asarray(tensor, dtype=np.float64).tobytes()).hexdigest()


def _source_identity(state, tensor, *, source_kind, source_front_id=None,
                     source_cavity_id=None, source_boundary_site_id=None,
                     source_position_m=None, source_probe_identity=None,
                     source_load_payload=None):
    identity = {
        "source_kind": source_kind,
        "source_front_id": source_front_id,
        "source_cavity_id": source_cavity_id,
        "source_boundary_site_id": source_boundary_site_id,
        "source_position_m": None if source_position_m is None else list(map(float, source_position_m)),
        "source_geometry_generation": int(state.crack_network.geometry_generation),
        "source_tensor_fingerprint": None if tensor is None else _tensor_fingerprint(tensor),
        "source_probe_identity": source_probe_identity,
    }
    if source_load_payload is not None:
        identity["source_load_fingerprint"] = hashlib.sha256(
            json.dumps(source_load_payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
    return identity


def _transition_competition_source(state, source, *, candidates=None):
    """Archive completed events and start source-owned fresh clocks."""
    junction = dict(state.junction_process_state)
    provenance = dict(junction.get("directional_event_provenance", {}))
    stale = list(junction.get("geometrically_stale_directional_events", ()))
    for event in state.competition.pending_events:
        record = dict(provenance[event.event_id])
        record.update({"status": "COMPLETED_BUT_GEOMETRICALLY_STALE",
                       "stale_at_geometry_generation": int(state.crack_network.geometry_generation)})
        provenance[event.event_id] = record
        stale.append(record)
    seed_source = {
        key: source.get(key) for key in (
            "source_kind", "source_front_id", "source_cavity_id",
            "source_boundary_site_id", "source_geometry_generation",
        )
    }
    seed_source["candidate_ids"] = sorted(
        candidate.candidate_id
        for candidate in (state.competition.candidates if candidates is None else candidates)
    )
    seed_payload = json.dumps({"source": seed_source, "seed": state.competition.global_hazard_seed},
                              sort_keys=True, separators=(",", ":")).encode()
    seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big") & ((1 << 63) - 1)
    competition = DirectionalCompetitionState.initialize(
        state.competition.candidates if candidates is None else candidates,
        global_hazard_seed=seed,
    )
    source = dict(source)
    declared = {row["candidate_id"]: dict(row)
                for row in source.get("candidate_source_states", ())}
    source["candidate_source_states"] = tuple({
        **declared.get(candidate.candidate_id, {}),
        "candidate_id": candidate.candidate_id,
        "direction_xy": list(candidate.direction_xy),
        "normal_xy": list(candidate.normal_xy),
        "threshold_identity": {
            "threshold_process": hazard.threshold_process,
            "threshold_seed": hazard.threshold_seed,
            "threshold_action": hazard.current_threshold_action,
        },
        "rng_provenance": {"source_seed": seed,
                           "global_hazard_seed": competition.global_hazard_seed},
    } for candidate, hazard in zip(competition.candidates, competition.hazard_states))
    junction.update({"directional_event_provenance": provenance,
                     "geometrically_stale_directional_events": tuple(stale),
                     "active_event_source": source})
    return replace(state, competition=competition, junction_process_state=junction)


def _mark_consumed_event_provenance(state, event_ids):
    junction = dict(state.junction_process_state)
    provenance = dict(junction.get("directional_event_provenance", {}))
    for event_id in event_ids:
        if event_id in provenance:
            provenance[event_id] = {**provenance[event_id], "status": "CONSUMED_AT_OWNED_SOURCE"}
    junction["directional_event_provenance"] = provenance
    return replace(state, junction_process_state=junction)


def _mesh_aligned_principal_candidate(state, start, tensor, *, variant, target_distance_m,
                                      allow_kinetically_dormant=False):
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(tensor, dtype=float))
    normal = eigenvectors[:, int(np.argmax(eigenvalues))]
    ideal = np.asarray((normal[1], -normal[0]))
    if ideal[0] < 0.0:
        ideal *= -1.0
    delta = np.asarray(state.mesh.nodes) - np.asarray(start)
    distance = np.linalg.norm(delta, axis=1)
    unit = delta / np.maximum(distance[:, None], 1.0e-300)
    forward = unit @ ideal
    normals = np.column_stack((-unit[:, 1], unit[:, 0]))
    resolved_opening = np.einsum("ni,ij,nj->n", normals, np.asarray(tensor, dtype=float), normals)
    geometric = ((distance >= 0.5 * target_distance_m)
                 & (distance <= 1.5 * target_distance_m)
                 & (forward > 0.8)
                 & (np.asarray(state.mesh.nodes)[:, 0] > 1.0e-12)
                 & (np.asarray(state.mesh.nodes)[:, 0] < 1.0e-3 - 1.0e-12)
                 & (np.abs(np.asarray(state.mesh.nodes)[:, 1]) < 5.0e-4 - 1.0e-12))
    tensile = np.flatnonzero(geometric & (resolved_opening > 0.0))
    eligible = tensile if len(tensile) or not allow_kinetically_dormant else np.flatnonzero(geometric)
    if not len(eligible):
        raise RuntimeError("no mesh-aligned forward cleavage endpoint")
    opening_scale = max(float(np.max(np.maximum(resolved_opening[eligible], 0.0))), 1.0)
    score = (1.0 - forward[eligible]) + 0.1 * np.abs(
        distance[eligible] - target_distance_m
    ) / target_distance_m + 0.1 * (1.0 - resolved_opening[eligible] / opening_scale)
    endpoint = int(eligible[int(np.argmin(score))])
    direction = unit[endpoint]
    aligned_normal = np.asarray((-direction[1], direction[0]))
    candidate = CleavageCandidate.create(
        plane_family="cleavage", plane_variant=variant,
        direction_xy=direction, normal_xy=aligned_normal, gamma_rel=1.0,
        orientation_convention="V5 mesh-aligned principal-opening renewal",
    )
    return candidate, tuple(map(float, state.mesh.nodes[endpoint]))


def _mesh_endpoint_on_direction(state, start, direction, target_distance_m):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    delta = np.asarray(state.mesh.nodes) - np.asarray(start)
    axial = delta @ direction
    normal = np.abs(delta[:, 0] * direction[1] - delta[:, 1] * direction[0])
    eligible = np.flatnonzero((axial > 0.5 * target_distance_m)
                              & (axial < 1.5 * target_distance_m)
                              & (normal <= 1.0e-10)
                              & (np.asarray(state.mesh.nodes)[:, 0] < 1.0e-3 - 1.0e-12)
                              & (np.abs(np.asarray(state.mesh.nodes)[:, 1]) < 5.0e-4 - 1.0e-12))
    if not len(eligible):
        return None
    node = int(eligible[np.argmin(np.abs(axial[eligible] - target_distance_m))])
    return tuple(map(float, state.mesh.nodes[node]))


def directional_clock_rates(state, stress_tensor_Pa, *, temperature_K=900.0):
    """Read-only rates from the existing cleavage law and preserved clocks."""
    material = state.material
    engine = fresh_sharp_front_engine(material, _bundle_from_state(state))
    stress = np.asarray(stress_tensor_Pa, dtype=float).reshape(2, 2)
    rates = []
    for candidate, hazard in zip(state.competition.candidates, state.competition.hazard_states):
        normal = np.asarray(candidate.normal_xy, dtype=float)
        resolved_opening = max(float(normal @ stress @ normal), 0.0)
        raw_rate, _, barrier = engine.lambda_cleave(resolved_opening, temperature_K)
        zero_tolerance = 1.0e-12 * max(float(np.linalg.norm(stress, ord=2)), 1.0)
        rate = 0.0 if resolved_opening <= zero_tolerance else max(float(raw_rate), 0.0)
        remaining = max(hazard.current_threshold_action - hazard.action, 0.0)
        crossing = math.inf if rate <= 0.0 else remaining / rate
        rates.append({"candidate_id": candidate.candidate_id, "raw_rate_s": float(raw_rate),
                      "effective_rate_s": rate, "rate_s": rate,
                      "resolved_opening_stress_Pa": resolved_opening,
                      "hazard_barrier_J": barrier, "crossing_time_s": crossing})
    return rates


def _canonical_front_process_owner(state, branch_id):
    """Read one front's process owner from accepted branch/runtime provenance."""
    branch_id = str(branch_id)
    if branch_id not in state.crack_network.active_tip_ids:
        raise ValueError("hybrid directional drive requires an active branch")
    owner = state.crack_network.branch(branch_id).local_state.get(
        "front_engine_state_owner"
    )
    if owner is None:
        raise RuntimeError("active front has no canonical process-owner identity")
    owner = str(owner)
    registry = state.tip_process_state.get("by_branch", {})
    if owner not in registry:
        raise RuntimeError("active front process owner is absent from accepted state")
    checkpoint_mapping = state.tip_process_state.get("owner_by_front")
    if checkpoint_mapping is not None and str(checkpoint_mapping.get(branch_id)) != owner:
        raise RuntimeError("active front process owner disagrees with checkpoint registry")
    return owner


def sharp_front_load_provider(
    state, *, branch_id, candidates=None, evaluate_overlap_marginals=False,
    marginal_delta_a_m=(3.0e-5, 2.0e-5), marginal_mesh_levels=(2, 3, 4),
    finite_increment_mode=False,
    matched_discretization=False,
    virtual_mechanical_trial_transport="full",
    mechanical_refinement_policy="legacy_accumulated_generations",
    batched_state_validation=False,
):
    """Use valid nested local J, else an exact fixed-void marginal G trial."""
    from .hybrid_directional_drive_v5 import hybrid_directional_drive_provider
    process_owner_by_tip = {
        tip_id: _canonical_front_process_owner(state, tip_id)
        for tip_id in state.crack_network.active_tip_ids
    }
    process_owner_id = process_owner_by_tip[str(branch_id)]
    payload = state.tip_process_state.get("by_branch", {}).get(process_owner_id)
    if payload is None:
        raise ValueError("active child has no canonical sharp-front engine state")
    engine = restore_sharp_front_engine(state.material, _bundle_from_state(state), payload)
    inventory = state.competition.candidates if candidates is None else tuple(candidates)
    candidates_by_tip = {
        tip_id: inventory if tip_id == branch_id else ()
        for tip_id in state.crack_network.active_tip_ids
    }
    def refresh_external_root_context(trial):
        root = trial.crack_network.branches[0].path[0]
        return replace(trial, junction_process_state={
            **trial.junction_process_state,
            "boundary_terminal_context": _external_free_root_context(
                trial.mesh, trial.boundary, root,
            ),
        })

    if virtual_mechanical_trial_transport not in ("full", "deferred_read_only"):
        raise ValueError("unknown virtual mechanical trial transport mode")
    return hybrid_directional_drive_provider(
        state, candidates_by_tip,
        process_owner_by_tip=process_owner_by_tip,
        contour_radius_m=max(float(engine.f.L_pz), 1.0e-6),
        provider_contract_contour_radius_m=max(float(engine.f.L_pz), 1.0e-6),
        marginal_delta_a_m=marginal_delta_a_m,
        marginal_mesh_levels=marginal_mesh_levels,
        evaluate_overlap_marginals=bool(evaluate_overlap_marginals),
        source_commit=_head(),
        prepare_support_state=(
            _refresh_downstream_boundary_context
            if state.void_state is not None and any(
                cavity.phase in (
                    VoidPhase.CONNECTED_VOID,
                    VoidPhase.DOWNSTREAM_FRONT_ACTIVE,
                )
                for cavity in state.void_state.cavities
            )
            else refresh_external_root_context
        ),
        conform_trial_endpoint=lambda trial, endpoint: _conform_bulk_point(
            trial, endpoint, identity="HYBRID_MARGINAL_ENDPOINT_CONFORMING",
            transfer_v2_transport=(virtual_mechanical_trial_transport == "full"),
        ),
        finite_increment_mode=bool(finite_increment_mode),
        matched_discretization=bool(matched_discretization),
        virtual_mechanical_trial_transport=str(virtual_mechanical_trial_transport),
        mechanical_refinement_policy=str(mechanical_refinement_policy),
        batched_state_validation=bool(batched_state_validation),
    )


def directional_sharp_front_rates(state, load_rows, *, branch_id, temperature_K=900.0):
    """Preview child rates through the unchanged old engine and owned ledgers."""
    from .directional_competition_v11 import preview_production_cleavage_rate
    from .hybrid_directional_drive_v5 import K_INTERPRETATION
    process_owner_id = _canonical_front_process_owner(state, branch_id)
    payload = state.tip_process_state.get("by_branch", {}).get(process_owner_id)
    if payload is None:
        raise ValueError("active child has no canonical sharp-front engine state")
    engine = restore_sharp_front_engine(state.material, _bundle_from_state(state), payload)
    by_candidate = {row["candidate_id"]: row for row in load_rows}
    expected_candidates = {item.candidate_id for item in state.competition.candidates}
    if set(by_candidate) != expected_candidates:
        raise RuntimeError("hybrid load rows do not exactly match the owned candidate inventory")
    identity_tuples = {
        (row.get("accepted_state_id"), row.get("stress_field_state_id"),
         row.get("topology_fingerprint"))
        for row in load_rows
    }
    if len(identity_tuples) != 1:
        raise RuntimeError("hybrid load rows mix accepted mechanics identities")
    rates = []
    for candidate, hazard in zip(state.competition.candidates, state.competition.hazard_states):
        load = by_candidate[candidate.candidate_id]
        if str(load.get("tip_id", branch_id)) != str(branch_id):
            raise RuntimeError("directional scalar drive is detached from its owning front")
        if str(load.get("process_owner_id")) != process_owner_id:
            raise RuntimeError("directional scalar drive is detached from its process owner")
        if str(load.get("controlling_scalar_tip_id", branch_id)) != str(
            load.get("tensor_probe_tip_id", branch_id)
        ):
            raise RuntimeError("hybrid scalar and tensor observations have different tip owners")
        G_kinetic = max(float(load.get(
            "G_kinetic_used_J_per_m2", load["positive_J_J_per_m2"]
        )), 0.0)
        available = load.get("directional_drive_status", "AVAILABLE") == "AVAILABLE"
        directional_K = math.sqrt(float(state.material.Eprime) * G_kinetic)
        effective_K = directional_K / math.sqrt(float(candidate.gamma_rel))
        response = sharp_front_constitutive_response(
            engine, K_Pa_sqrt_m=effective_K, temperature_K=temperature_K, dt_s=1.0,
        )
        adapted = preview_production_cleavage_rate(
            engine, candidate, signed_J_J_per_m2=G_kinetic,
            Eprime_Pa=float(state.material.Eprime), temperature_K=temperature_K,
        )
        rate = 0.0 if (not available or G_kinetic <= 0.0) else max(adapted.lambda_per_s, 0.0)
        remaining = max(hazard.current_threshold_action - hazard.action, 0.0)
        rates.append({
            "candidate_id": candidate.candidate_id,
            "raw_rate_s": response["cleavage_raw_rate_s"],
            "effective_rate_s": rate,
            "rate_s": rate,
            "rate": rate,
            "resolved_opening_stress_Pa": response["sigma_tip_Pa"],
            "hazard_barrier_J": response["cleavage_barrier_J"],
            "crossing_time_s": math.inf if rate <= 0.0 else remaining / rate,
            "K_directional_Pa_sqrt_m": directional_K,
            "K_energy_equivalent_Pa_sqrt_m": directional_K,
            "K_effective_Pa_sqrt_m": effective_K,
            "K_interpretation": load.get("K_interpretation", K_INTERPRETATION),
            "signed_J_J_per_m2": float(load.get("G_local_J_per_m2", load["signed_J_J_per_m2"])),
            "positive_J_J_per_m2": G_kinetic,
            "G_local_J_per_m2": float(load.get("G_local_J_per_m2", load["signed_J_J_per_m2"])),
            "G_local": float(load.get("G_local_J_per_m2", load["signed_J_J_per_m2"])),
            "local_contour_valid": bool(load.get("local_contour_valid", True)),
            "G_marginal_J_per_m2": load.get("G_marginal_J_per_m2"),
            "G_marginal": load.get("G_marginal_J_per_m2"),
            "G_kinetic_used_J_per_m2": G_kinetic,
            "G_kinetic_used": G_kinetic,
            "drive_source": load.get("drive_source", "LEGACY_DIRECTIONAL_J"),
            "tip_id": str(load.get("tip_id", branch_id)),
            "process_owner_id": process_owner_id,
            "accepted_state_id": load.get("accepted_state_id"),
            "stress_field_state_id": load.get("stress_field_state_id"),
            "topology_fingerprint": load.get("topology_fingerprint"),
            "directional_drive_status": load.get("directional_drive_status", "AVAILABLE"),
            "r_eff_m": response["r_eff_m"],
            "sigma_tip_Pa": response["sigma_tip_Pa"],
            "cleavage_barrier_J": response["cleavage_barrier_J"],
            "cleavage_rate_s": response["cleavage_rate_s"],
            "emission_barrier_J": response["emission_barrier_J"],
            "emission_rate_s": response["emission_rate_s"],
            "clock_increment_per_second": response["clock_increment"],
            "constitutive_engine": "FrontEngine",
            "production_rate_adapter": "preview_production_cleavage_rate",
        })
    return rates


def cavity_source_resolution_metrics(
    state, *, geometry_contract=None, final_traction_verification=False,
    source_position_m=None,
):
    """Source-native tensor and geometrical boundary-recovery diagnostics."""
    cavity = state.void_state.cavities[0]
    position = np.asarray(
        cavity.connection_exit_m if source_position_m is None else source_position_m,
        dtype=float,
    )
    if position.shape != (2,) or not np.isfinite(position).all():
        raise ValueError("cavity source position must be an explicit finite point")
    node = int(np.argmin(np.linalg.norm(state.mesh.nodes-position, axis=1)))
    if geometry_contract is None:
        recovery = cavity_fixed_physical_arc_patch_recovery_v3(state, boundary_node=node)
    elif geometry_contract in (
        "CAVITY_SOURCE_CONFORMING_GEOMETRY_V4", SHAPE_REGULAR_MESH_V5_ID,
    ):
        recovery = cavity_source_conforming_recovery_v4(state, boundary_node=node)
    else:
        raise ValueError("unregistered cavity source geometry contract")
    tensor = np.asarray(recovery["tensor_Pa"], dtype=float)
    selected = tuple(map(int, recovery["stencil_element_ids"]))
    nodes = np.asarray(state.mesh.nodes); elems = np.asarray(state.mesh.elems)
    edges = _actual_cavity_boundary_edges(state)
    owned = {}
    wanted = {tuple(edge) for edge in edges}
    for eid, triangle in enumerate(elems):
        for a, b in ((triangle[0],triangle[1]),(triangle[1],triangle[2]),(triangle[2],triangle[0])):
            edge = tuple(sorted((int(a),int(b))))
            if edge in wanted: owned.setdefault(edge, []).append(eid)
    _, assembled_residual, sigma, *_ = assemble_mechanics(state.mesh, state.displacement,
        state.ep_gp, state.rho_gp, state.damage, state.elasticity_D, state.material,
        cohesive_network=state.cohesive_network)
    local_rows = []; all_rows = []; traction_sum = 0.; boundary_length = 0.
    for a, b in edges:
        owners = owned[tuple((a,b))]
        if len(owners) != 1: raise ValueError("cavity source edge has nonunique owner")
        eid = owners[0]; delta = nodes[b]-nodes[a]; length = float(np.linalg.norm(delta))
        tangent = delta/length; normal = np.array((-tangent[1],tangent[0]))
        height = float(2*state.mesh.area_e[eid]/length)
        tri = nodes[elems[eid]]; sides = np.linalg.norm(tri-tri[[1,2,0]],axis=1)
        quality = float(4*np.sqrt(3)*state.mesh.area_e[eid]/(sides@sides))
        stress = np.array(((sigma[0,eid],sigma[2,eid]),(sigma[2,eid],sigma[1,eid])))
        traction_sum += float(np.linalg.norm(stress@normal)**2)*length
        boundary_length += length
        all_rows.append((height/cavity.radius_m,length/cavity.radius_m,
                         max(height/length,length/height),quality))
        if np.linalg.norm((nodes[a]+nodes[b])/2-position) <= .25*cavity.radius_m:
            local_rows.append((height/cavity.radius_m,length/cavity.radius_m,
                               max(height/length,length/height),quality))
    if not local_rows: raise ValueError("fixed cavity source has no local boundary neighborhood")
    local = np.asarray(local_rows)
    all_boundary = np.asarray(all_rows)
    from .topology_transaction_v11 import require_equilibrium_observables
    equilibrium = require_equilibrium_observables(state)
    specimen_width = float(np.ptp(nodes[:, 0]))
    if not np.isfinite(specimen_width) or specimen_width <= 0.0:
        raise ValueError("specimen width must be finite and positive")
    reaction_magnitude = 0.5 * (
        abs(float(equilibrium["latest_top_reaction_N_per_m"]))
        + abs(float(equilibrium["latest_bottom_reaction_N_per_m"]))
    )
    remote = reaction_magnitude / specimen_width
    triangles = nodes[elems]
    side2 = np.sum((triangles-triangles[:,[1,2,0]])**2,axis=(1,2))
    global_quality = 4*np.sqrt(3)*np.asarray(state.mesh.area_e)/side2
    radial = (position-np.asarray(cavity.center_m)); normal = radial/np.linalg.norm(radial)
    tangent = np.array((-normal[1],normal[0]))
    basis = np.column_stack((normal, tangent))
    local_tensor = basis.T @ tensor @ basis
    cavity_nodes = np.unique(edges)
    weak_force = np.asarray(assembled_residual).reshape(-1, 2)[cavity_nodes]
    if not np.isfinite(boundary_length) or boundary_length <= 0.0:
        raise ValueError("cavity boundary length must be finite and positive")
    weak_residual = float(np.linalg.norm(weak_force) / (remote * boundary_length))
    raw_traction = math.sqrt(traction_sum/boundary_length)/remote
    from .equilibrated_boundary_traction_recovery_v1 import (
        EquilibratedBoundaryTractionUnavailable,
        recover_equilibrated_boundary_traction_v1,
    )
    damage = getattr(state.mesh, "element_damage_gp", None)
    solid = None if damage is None else np.asarray(damage, dtype=float) < 0.5
    engine = fresh_sharp_front_engine(state.material, _bundle_from_state(state))
    try:
        equilibrated = recover_equilibrated_boundary_traction_v1(
            nodes=nodes, elements=elems, stress_Pa=sigma, boundary_node=node,
            cavity_center_m=cavity.center_m, cavity_radius_m=cavity.radius_m,
            process_length_m=engine.f.L_pz, owned_boundary_edges=edges,
            remote_traction_scale_Pa=remote,
            state_fingerprint=_cavity_resolution_binding(state), solid_element_mask=solid,
        )
    except EquilibratedBoundaryTractionUnavailable as exc:
        equilibrated = {
            "operator": "EQUILIBRATED_BOUNDARY_TRACTION_RECOVERY_V1",
            "available": False, "scientific_unavailability": str(exc),
        }
    final_records = {}
    if final_traction_verification:
        from .arc_resolved_cavity_traction_v1 import diagnose_arc_resolved_cavity_traction_v1
        arc_diagnosis = diagnose_arc_resolved_cavity_traction_v1(
            nodes=nodes, elements=elems, stress_Pa=sigma, owned_boundary_edges=edges,
            cavity_center_m=cavity.center_m, cavity_radius_m=cavity.radius_m,
            source_position_m=cavity.connection_exit_m,
            junction_position_m=(cavity.connection_entry_m or cavity.connection_exit_m),
            process_length_m=engine.f.L_pz, remote_traction_scale_Pa=remote,
            plastic_strain=state.ep_gp, density_m2=state.rho_gp,
        )
        if not math.isclose(
            arc_diagnosis["recomputed_raw_adjacent_element_traction_normalized"],
            raw_traction, rel_tol=0.0, abs_tol=2.0e-15,
        ):
            raise RuntimeError("ARC_TRACTION_ROWS_DO_NOT_RECOMPUTE_RETAINED_SCALAR")
        from .piecewise_equilibrated_boundary_stress_v2 import (
            PiecewiseEquilibratedStressUnavailable,
            recover_piecewise_equilibrated_boundary_stress_v2,
        )
        try:
            piecewise = recover_piecewise_equilibrated_boundary_stress_v2(
                nodes=nodes, elements=elems, stress_Pa=sigma, boundary_node=node,
                cavity_center_m=cavity.center_m, cavity_radius_m=cavity.radius_m,
                process_length_m=engine.f.L_pz, owned_boundary_edges=edges,
                remote_traction_scale_Pa=remote,
                state_fingerprint=_cavity_resolution_binding(state), solid_element_mask=solid,
            )
        except PiecewiseEquilibratedStressUnavailable as exc:
            piecewise = {
                "operator": "PIECEWISE_EQUILIBRATED_BOUNDARY_STRESS_V2",
                "available": False, "scientific_unavailability": str(exc),
            }
        final_records = {
            "arc_resolved_cavity_traction_v1": arc_diagnosis,
            "piecewise_equilibrated_boundary_stress_v2": piecewise,
        }
    plastic_active = bool(np.any(np.abs(np.asarray(state.ep_gp, dtype=float)) > 0.0))
    return {"tensor_Pa": tensor.tolist(), "probe_element_ids": list(selected),
        "boundary_node_id": node, "boundary_position_m": position.tolist(),
        "normal_xy": normal.tolist(), "tangent_xy": tangent.tolist(),
        "sigma_tt_Pa": float(tangent@tensor@tangent), "sigma_nn_Pa": float(normal@tensor@normal),
        "sigma_nt_Pa": float(normal@tensor@tangent), "principal_stresses_Pa": np.linalg.eigvalsh(tensor).tolist(),
        "eta_n_max": float(all_boundary[:,0].max()), "eta_n_median": float(np.median(all_boundary[:,0])),
        "eta_t_max": float(all_boundary[:,1].max()), "eta_t_median": float(np.median(all_boundary[:,1])),
        "local_aspect_ratio_max": float(all_boundary[:,2].max()), "local_aspect_ratio_median": float(np.median(all_boundary[:,2])),
        "source_neighborhood_eta_n_max":float(local[:,0].max()),
        "source_neighborhood_eta_t_max":float(local[:,1].max()),
        "minimum_quality": float(global_quality.min()),
        "local_minimum_quality": float(local[:,3].min()),
        "normalized_traction": float(recovery["traction_residual"]),
        "constrained_boundary_limit_traction_residual": float(recovery["traction_residual"]),
        "raw_adjacent_element_traction_normalized": raw_traction,
        "incident_cst_boundary_traction_v1": raw_traction,
        "RAW_CST_BOUNDARY_TRACTION_DIAGNOSTIC": {
            "classification": "NON_AUTHORITATIVE_DIAGNOSTIC",
            "normalized_traction": raw_traction,
        },
        "equilibrated_boundary_traction_recovery_v1": equilibrated,
        **final_records,
        **({"PIECEWISE_EQUILIBRATED_TRACTION_DIAGNOSTIC": {
            "classification": "NON_AUTHORITATIVE_DIAGNOSTIC",
            "result": final_records["piecewise_equilibrated_boundary_stress_v2"],
        }} if final_traction_verification else {}),
        "assembled_weak_cavity_boundary_residual_normalized": weak_residual,
        "source_tensor_local_nt_Pa": local_tensor.tolist(),
        "source_tensor_global_xy_Pa": tensor.tolist(),
        "constrained_fit_residual": float(recovery["constrained_fit_residual"]),
        "patch_rank": recovery["rank"],
        "patch_condition": recovery["condition"],
        "sigma_zz_and_mean_stress_audit": {
            "plastic_strain_active": plastic_active,
            "mean_stress_consumed_by_cavity_source_kinetics": False,
            "classification": "DIAGNOSTIC_ONLY",
            "reported_plane_strain_sigma_zz_Pa": float(recovery["plane_strain_sigma_zz_Pa"]),
            "reported_mean_stress_Pa": float(recovery["mean_stress_Pa"]),
            "interpretation": (
                "elastic projection of the constrained in-plane boundary-limit tensor; "
                "not a constitutive plastic sigma_zz recovery and not consumed by source kinetics"
            ),
        },
        "recovery_operator": recovery["recovery_operator"],
        "recovery_record": recovery}


def _cavity_resolution_binding(state):
    digest = hashlib.sha256()
    for value in (state.mesh.nodes, state.mesh.elems, state.displacement, state.ep_gp,
                  state.rho_gp, state.damage, state.elasticity_D):
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode()); digest.update(array.tobytes())
    p0 = getattr(state.mesh,"element_damage_gp",None)
    if p0 is None:
        digest.update(b"element_damage_gp:None")
    else:
        array = np.ascontiguousarray(p0)
        digest.update(b"element_damage_gp:")
        digest.update(str((array.shape,array.dtype.str)).encode());digest.update(array.tobytes())
    cavity = state.void_state.cavities[0]
    digest.update(repr((cavity.cavity_id,cavity.center_m,cavity.radius_m,cavity.connection_exit_m,
        state.crack_network.geometry_generation,tuple((c.candidate_id,c.direction_xy,c.normal_xy)
        for c in state.competition.candidates))).encode())
    digest.update(repr(state.material).encode())
    return digest.hexdigest()


def _qualified_cavity_source(state, tensor, *, temperature_K=900.0, return_reason=False):
    def answer(passed, reason):
        return (bool(passed), reason) if return_reason else bool(passed)
    from .closure_static_evidence import resolution_screen
    from .finalization_v3_schema import SCIENTIFIC_ACCEPTANCE_TOLERANCES as limits
    from .finalization_v3_schema import canonical_hash
    proof = state.junction_process_state.get("cavity_source_resolution_proof", {})
    from .source_quality_transaction_v1 import PROOF_SCHEMA, source_transfer_budget
    if (proof.get("schema") not in ("v12.cavity-source-local-refinement/1", PROOF_SCHEMA)
        or proof.get("current_binding") != _cavity_resolution_binding(state)
        or not isinstance(proof.get("refinement_count"), int)
        or proof["refinement_count"] < 1
        or not all(key in proof for key in ("current_metrics","previous_metrics","previous_binding"))):
        return answer(False, "INCOMPLETE_OR_STALE_REFINEMENT_PROOF")
    current = proof["current_metrics"]; reference = proof["previous_metrics"]
    if (current.get("recovery_operator") != "CAVITY_FIXED_PHYSICAL_ARC_PATCH_RECOVERY_V3"
        or reference.get("recovery_operator") != "CAVITY_FIXED_PHYSICAL_ARC_PATCH_RECOVERY_V3"
        or not current.get("recovery_record", {}).get("available")
        or not reference.get("recovery_record", {}).get("available")):
        return answer(False, "RECOVERY_OPERATOR_UNAVAILABLE_OR_WRONG")
    capture = proof.get("previous_source_capture")
    if not capture or "element_damage_gp" not in capture:
        return answer(False, "PREVIOUS_SOURCE_CAPTURE_MISSING")
    previous_mesh = replace(rebuild_tri_mesh(capture["nodes"], capture["elements"]),
        element_damage_gp=capture["element_damage_gp"])
    from .topology_transaction_v11 import EquilibriumObservables
    previous_observables = capture.get("equilibrium_observables")
    if previous_observables is not None and not isinstance(previous_observables, EquilibriumObservables):
        previous_observables = EquilibriumObservables(**dict(previous_observables))
    previous_state = replace(state, mesh=previous_mesh,
        boundary=capture["boundary"],
        displacement=capture["displacement"], damage=capture["damage"],
        ep_gp=capture["ep_gp"], rho_gp=capture["rho_gp"],
        vacancy_fraction_gp=capture["vacancy_fraction_gp"],
        vacancy_ledgers=capture["vacancy_ledgers"],
        vacancy_temperature_K=capture["vacancy_temperature_K"],
        energy_ledgers=capture["energy_ledgers"],
        equilibrium_observables=previous_observables)
    if (_cavity_resolution_binding(previous_state) != proof["previous_binding"]
        or canonical_hash(cavity_source_resolution_metrics(state)) != canonical_hash(current)
        or canonical_hash(cavity_source_resolution_metrics(previous_state)) != canonical_hash(reference)):
        return answer(False, "CURRENT_OR_PREVIOUS_METRICS_DO_NOT_RECOMPUTE")
    a = np.asarray(current["tensor_Pa"]); b = np.asarray(reference["tensor_Pa"])
    error = float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-300))
    if proof.get("schema") == PROOF_SCHEMA:
        # Bind the stored transfer diagnostic to its original owned clocks.
        # A legitimate partial hazard advance must not invalidate the proof
        # merely because a relative waiting-time subtraction rounds differently.
        anchor = proof.get("budget_competition")
        if anchor is None or anchor.candidates != state.competition.candidates:
            return answer(False, "SOURCE_TRANSFER_COMPETITION_ANCHOR_CHANGED")
        if any((old.candidate_id,old.current_threshold_action,old.threshold_seed) !=
               (new.candidate_id,new.current_threshold_action,new.threshold_seed)
               for old,new in zip(anchor.hazard_states,state.competition.hazard_states)):
            return answer(False, "SOURCE_TRANSFER_THRESHOLD_IDENTITY_CHANGED")
        anchored_current = replace(state,competition=anchor)
        anchored_previous = replace(previous_state,competition=anchor)
        budget = source_transfer_budget(anchored_current, anchored_previous, a, b)
        if (reference["minimum_quality"] < limits["mesh_minimum_quality"]
            or canonical_hash(budget) != canonical_hash(proof.get("source_transfer_budget"))
            or not source_transfer_budget(state, previous_state, a, b, temperature_K=temperature_K)["passed"]):
            return answer(False, "SOURCE_TRANSFER_BUDGET_FAILED")
    from .cavity_free_boundary_verification import verify_cavity_free_boundary
    weak_boundary = verify_cavity_free_boundary(state)
    recorded_boundary = proof.get("weak_boundary_verification", {})
    if (recorded_boundary.get("geometry_fingerprint") != weak_boundary.geometry_fingerprint
        or recorded_boundary.get("predicates") != dict(weak_boundary.predicates)):
        return answer(False, "WEAK_BOUNDARY_PROOF_STALE")
    recovery = current["recovery_record"]
    ranks = recovery.get("rank", {})
    condition = recovery.get("condition", {})
    checks = {
        "tensor_identity": np.array_equal(a, np.asarray(tensor)),
        "resolution_screen": resolution_screen(current),
        "bounded_fit_rank": set(ranks) == {"sigma_tt", "sigma_nn", "sigma_nt"}
            and ranks == {"sigma_tt": 6, "sigma_nn": 3, "sigma_nt": 3},
        "bounded_fit_condition": condition.get("maximum", math.inf) <= condition.get("limit", 1.0e6),
        "weak_boundary": weak_boundary.passed,
        "tensor_convergence": error <= limits["tensor_probe_relative"],
    }
    passed = bool(all(checks.values()))
    return answer(passed, "PASS" if passed else "FAILED:" + ",".join(
        key for key, value in checks.items() if not value
    ))


def _bounded_quality_edge_flips(state):
    """One nonoverlapping convex-quad sweep, preserving every graph edge.

    No node, physical boundary or crack coordinate moves. A flip is accepted
    only when it strictly improves both triangles' minimum quality. There is
    no scientific tolerance override: the downstream global-quality gate is
    evaluated independently after transfer, support rebuild and equilibrium.
    """
    from .finalization_v3_schema import SCIENTIFIC_ACCEPTANCE_TOLERANCES as limits
    nodes = np.asarray(state.mesh.nodes); triangles = np.asarray(state.mesh.elems).copy()
    def cross(a,b): return float(a[0]*b[1]-a[1]*b[0])
    def quality(ids):
        p = nodes[list(ids)]; area2 = abs(cross(p[1]-p[0],p[2]-p[0]))
        return 2*np.sqrt(3)*area2/float(np.sum((p-p[[1,2,0]])**2))
    graph_segments = [(np.asarray(a),np.asarray(b)) for branch in state.crack_network.branches
        for a,b in zip(branch.path,branch.path[1:])]
    def on_graph_edge(a,b):
        for p,q in graph_segments:
            delta = q-p; length = np.linalg.norm(delta)
            if (abs(cross(nodes[a]-p,delta))/length <= 1e-12
                and abs(cross(nodes[b]-p,delta))/length <= 1e-12
                and max((nodes[a]-p)@delta,(nodes[b]-p)@delta) >= 0
                and min((nodes[a]-p)@delta,(nodes[b]-p)@delta) <= delta@delta): return True
        return False
    owners = {}
    coordinates = nodes[triangles]
    side2 = np.sum((coordinates-coordinates[:,[1,2,0]])**2,axis=(1,2))
    initial_quality = 4*np.sqrt(3)*np.asarray(state.mesh.area_e)/side2
    for eid,tri in enumerate(triangles):
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            owners.setdefault(tuple(sorted((int(a),int(b)))),[]).append(eid)
    touched = set(); flips = []
    for (a,b), incident in sorted(owners.items()):
        if len(incident) != 2 or touched.intersection(incident): continue
        i,j = incident; original = min(initial_quality[i],initial_quality[j])
        if original >= limits["mesh_minimum_quality"]: continue
        if on_graph_edge(a,b): continue
        c = next(int(n) for n in triangles[i] if n not in (a,b))
        d = next(int(n) for n in triangles[j] if n not in (a,b))
        # Diagonals must cross strictly inside a convex quadrilateral.
        if (cross(nodes[b]-nodes[a],nodes[c]-nodes[a])*cross(nodes[b]-nodes[a],nodes[d]-nodes[a]) >= 0
            or cross(nodes[d]-nodes[c],nodes[a]-nodes[c])*cross(nodes[d]-nodes[c],nodes[b]-nodes[c]) >= 0): continue
        if tuple(sorted((c,d))) in owners: continue
        if any(cross(nodes[d]-nodes[c],p-nodes[c])*cross(nodes[d]-nodes[c],q-nodes[c]) < 0
            and cross(q-p,nodes[c]-p)*cross(q-p,nodes[d]-p) < 0 for p,q in graph_segments): continue
        replacement = [(c,d,a),(d,c,b)]
        if min(map(quality,replacement)) <= original: continue
        for eid,tri in zip((i,j),replacement):
            p = nodes[list(tri)]
            if cross(p[1]-p[0],p[2]-p[0]) < 0: tri = (tri[0],tri[2],tri[1])
            triangles[eid] = tri
        touched.update(incident); flips.append((a,b,c,d))
    mesh = rebuild_tri_mesh(nodes,triangles)
    return mesh,tuple(flips)


def refine_downstream_source(state, *, max_refinement_levels=3,
                             failure_stage=None, operation_log=None,
                             refinement_region="source_neighborhood", quality_improvement=False):
    """Bounded, transactional source refinement; preserves every owned clock.

    Success requires measured convergence on successive meshes of this same
    accepted material/history problem. An inadequate attempt returns the
    original state; cached metrics never authorize a changed mechanical state.
    """
    from .adaptive_multitip_mesh_v11 import refine_accepted_state
    if max_refinement_levels not in range(0,9): raise ValueError("source refinement budget must be 0..8")
    if refinement_region not in ("source_neighborhood","complete_cavity_ring"):
        raise ValueError("unregistered source refinement region")
    if quality_improvement not in (False, True, "constrained_v1"):
        raise ValueError("unregistered source quality strategy")
    cavity = state.void_state.cavities[0]
    if cavity.phase != VoidPhase.CONNECTED_VOID or state.crack_network.active_tip_ids:
        raise ValueError("source refinement requires a connected dormant cavity")
    operations = operation_log if operation_log is not None else []
    def inject(stage, current):
        operations.append(stage)
        if stage == failure_stage: raise RuntimeError("injected:" + stage)
    current = state.isolated_copy(); rows = []; quality_preparation = None
    from .cavity_source_recovery_v3 import CavitySourceRecoveryV3Unavailable
    if quality_improvement == "constrained_v1":
        from .source_quality_transaction_v1 import repair_connected_quality
        try:
            current, quality_preparation = repair_connected_quality(current)
        except CavitySourceRecoveryV3Unavailable as exc:
            return state, {"status":"SOURCE_TENSOR_UNQUALIFIED", "attempts":[],
                           "scientific_unavailability":str(exc), "operations":operations}
        if not quality_preparation["accepted"]:
            return state, {"status":"SOURCE_TENSOR_UNQUALIFIED", "attempts":[],
                           "quality_preparation":quality_preparation,"operations":operations}
    try:
        previous = cavity_source_resolution_metrics(current)
    except CavitySourceRecoveryV3Unavailable as exc:
        return state, {"status":"SOURCE_TENSOR_UNQUALIFIED", "attempts":[],
                       "scientific_unavailability":str(exc), "operations":operations}
    original_clock = (state.competition, state.rng_state)
    for level in range(1,max_refinement_levels+1):
        position = np.asarray(cavity.connection_exit_m)
        centroids = current.mesh.nodes[current.mesh.elems].mean(axis=1)
        distance = (np.linalg.norm(centroids-position,axis=1) if refinement_region == "source_neighborhood"
            else np.abs(np.linalg.norm(centroids-np.asarray(cavity.center_m),axis=1)-cavity.radius_m))
        marked = tuple(np.flatnonzero(distance <= .5*cavity.radius_m + 2*np.sqrt(current.mesh.area_e)))
        refined, _ = refine_accepted_state(current, marked_parent_elements=marked,
            active_tip_ids=(), generation=int(current.event_counters.get("mesh_generation",0))+1,
            operation_index=int(current.event_counters.get("refinement_operation_index",0))+1)
        flips = ()
        if quality_improvement is True:
            quality_mesh, flips = _bounded_quality_edge_flips(refined)
            if flips:
                fields = _project_fields(refined,quality_mesh)
                refined = replace(refined,mesh=quality_mesh,damage=fields["damage"],
                    displacement=fields["displacement"],ep_gp=fields["ep_gp"],rho_gp=fields["rho_gp"],
                    vacancy_fraction_gp=fields["vacancy_fraction_gp"],
                    vacancy_ledgers=fields["vacancy_ledgers"],
                    vacancy_temperature_K=fields["vacancy_temperature_K"],
                    vacancy_transport_state_v2=fields.get(
                        "vacancy_transport_state_v2", refined.vacancy_transport_state_v2,
                    ),
                    junction_process_state={**dict(refined.junction_process_state),
                        **({"transport_boundary_tags_v2": fields["transport_boundary_tags_v2"]}
                           if "transport_boundary_tags_v2" in fields else {})})
        constrained_quality = None
        if quality_improvement == "constrained_v1":
            from .quality_constrained_mesh_v1 import constrained_quality_mesh, production_geometry_constraints
            fixed, protected = production_geometry_constraints(refined)
            quality_mesh, constrained_quality = constrained_quality_mesh(refined.mesh,
                fixed_nodes=fixed, protected_edges=protected, strategy="flips")
            fields = _project_fields(refined, quality_mesh)
            refined = replace(refined,mesh=quality_mesh,damage=fields["damage"],
                displacement=fields["displacement"],ep_gp=fields["ep_gp"],rho_gp=fields["rho_gp"],
                vacancy_fraction_gp=fields["vacancy_fraction_gp"],
                vacancy_ledgers=fields["vacancy_ledgers"],
                vacancy_temperature_K=fields["vacancy_temperature_K"],
                vacancy_transport_state_v2=fields.get(
                    "vacancy_transport_state_v2", refined.vacancy_transport_state_v2,
                ),
                junction_process_state={**dict(refined.junction_process_state),
                    **({"transport_boundary_tags_v2": fields["transport_boundary_tags_v2"]}
                       if "transport_boundary_tags_v2" in fields else {})})
        inject("downstream_source_refinement", refined)
        refined = _prepare_connected_ligament_support(refined, entry=cavity.connection_entry_m,
            exit_point=cavity.connection_exit_m, direction=cavity.connection_direction_xy,
            new_connection=False)
        fields = {name: getattr(refined,name) for name in (
            "damage","displacement","ep_gp","rho_gp","vacancy_fraction_gp",
            "vacancy_ledgers","vacancy_temperature_K","tip_process_state",
        )}
        fields["source_state"] = refined.junction_process_state.get("source_state",{})
        rebuilt = remesh_mechanically_separating_v12(refined, mesh=refined.mesh,
            boundary=refined.boundary, transferred_fields=fields, source_commit=_head(),
            configuration={"event":"DOWNSTREAM_SOURCE_REFINEMENT","level":level,"region":refinement_region},
            transaction_identity="source-refinement:"+str(level))
        inject("source_support_rebuild", rebuilt)
        trial = equilibrate_fixed_load_with_production_fem(rebuilt)
        inject("source_equilibrium", trial)
        if (trial.competition,trial.rng_state) != original_clock:
            raise RuntimeError("source refinement changed an owned threshold/hazard/RNG")
        try:
            metrics = cavity_source_resolution_metrics(trial)
        except CavitySourceRecoveryV3Unavailable as exc:
            return state, {"status":"SOURCE_TENSOR_UNQUALIFIED", "attempts":rows,
                           "scientific_unavailability":str(exc), "operations":operations}
        proof = {"schema":"v12.cavity-source-local-refinement/1", "refinement_count":level,
            "refinement_region":refinement_region,
            "quality_edge_flips":flips,
            "previous_metrics":previous,"current_metrics":metrics,
            "previous_binding":_cavity_resolution_binding(current),"current_binding":_cavity_resolution_binding(trial)}
        from .cavity_free_boundary_verification import verify_cavity_free_boundary
        weak_boundary = verify_cavity_free_boundary(trial)
        proof["weak_boundary_verification"] = {
            "schema": weak_boundary.schema,
            "geometry_fingerprint": weak_boundary.geometry_fingerprint,
            "accepted_state_fingerprint": weak_boundary.accepted_state_fingerprint,
            "predicates": dict(weak_boundary.predicates),
            "passed": weak_boundary.passed,
            "assembled_boundary_residual_relative": weak_boundary.assembled_boundary_residual_relative,
            "normalized_virtual_work": dict(weak_boundary.normalized_virtual_work),
            "total_boundary_resultant_relative": weak_boundary.total_boundary_resultant_relative,
            "total_boundary_moment_relative": weak_boundary.total_boundary_moment_relative,
            "global_top_bottom_reaction_balance": weak_boundary.global_top_bottom_reaction_balance,
            "free_dof_residual_relative": weak_boundary.free_dof_residual_relative,
        }
        proof["previous_source_capture"] = {"nodes":current.mesh.nodes,"elements":current.mesh.elems,
            "boundary":current.boundary,
            "element_damage_gp":getattr(current.mesh,"element_damage_gp",None),
            "displacement":current.displacement,"damage":current.damage,"ep_gp":current.ep_gp,
            "rho_gp":current.rho_gp,"energy_ledgers":dict(current.energy_ledgers),
            "vacancy_fraction_gp":current.vacancy_fraction_gp,
            "vacancy_ledgers":dict(current.vacancy_ledgers),
            "vacancy_temperature_K":current.vacancy_temperature_K,
            "equilibrium_observables":current.equilibrium_observables.to_dict()}
        if quality_improvement == "constrained_v1":
            from .source_quality_transaction_v1 import PROOF_SCHEMA, source_transfer_budget
            proof.update({"schema":PROOF_SCHEMA, "quality_preparation":quality_preparation,
                "constrained_quality":constrained_quality,
                "budget_competition":current.competition,
                "source_transfer_budget":source_transfer_budget(trial,current,metrics["tensor_Pa"],previous["tensor_Pa"])})
        trial = replace(trial,junction_process_state={**trial.junction_process_state,
            "cavity_source_resolution_proof":proof})
        passed, qualification_reason = _qualified_cavity_source(
            trial, metrics["tensor_Pa"], return_reason=True,
        )
        rows.append({"level":level,"proof":{key:value for key,value in proof.items()
            if key != "previous_source_capture"},"qualified":passed,
            "qualification_reason": qualification_reason})
        if passed:
            inject("source_qualification",trial)
            rates = {row["candidate_id"]:row for row in directional_clock_rates(trial,metrics["tensor_Pa"])}
            source_candidates = trial.junction_process_state.get("active_event_source",{}).get("candidate_source_states",())
            candidates = {item.candidate_id:item for item in trial.competition.candidates}
            hazards = {item.candidate_id:item for item in trial.competition.hazard_states}
            if len(source_candidates) != len(candidates) or {row["candidate_id"] for row in source_candidates} != set(candidates):
                raise RuntimeError("source refinement changed candidate inventory")
            for row in source_candidates:
                owned = candidates[row["candidate_id"]]; clock = hazards[row["candidate_id"]]
                if (tuple(row["direction_xy"]) != tuple(owned.direction_xy)
                    or tuple(row["normal_xy"]) != tuple(owned.normal_xy)
                    or tuple(row["tangent_xy"]) != tuple(owned.direction_xy)
                    or row["threshold_identity"]["threshold_action"] != clock.current_threshold_action
                    or row["threshold_identity"]["threshold_seed"] != clock.threshold_seed
                    or row["rng_provenance"]["global_hazard_seed"] != trial.competition.global_hazard_seed):
                    raise RuntimeError("source refinement candidate geometry/threshold/RNG provenance mismatch")
                if (row.get("source_cavity_id",cavity.cavity_id) != cavity.cavity_id
                    or row.get("source_boundary_site_id","connection_exit") != "connection_exit"
                    or math.dist(row.get("source_position_m",cavity.connection_exit_m),cavity.connection_exit_m)>1e-12):
                    raise RuntimeError("source refinement candidate boundary-site identity mismatch")
            candidate_probe = {"kind":"CAVITY_FIXED_PHYSICAL_ARC_PATCH_RECOVERY_V3",
                "boundary_node_id":metrics["boundary_node_id"],"element_ids":metrics["probe_element_ids"],
                "physical_arc_identity":metrics["recovery_record"]["physical_arc_identity"]}
            candidate_rows = tuple({**row,
                "source_kind":"cavity_surface","source_cavity_id":cavity.cavity_id,
                "source_boundary_site_id":"connection_exit","source_position_m":list(cavity.connection_exit_m),
                "source_geometry_generation":int(trial.crack_network.geometry_generation),
                "source_mesh_generation":int(trial.event_counters.get("mesh_generation",0)),
                "source_tensor_fingerprint":_tensor_fingerprint(metrics["tensor_Pa"]),
                "source_probe_identity":candidate_probe,
                "geometry_status":"GEOMETRICALLY_VALID_KINETICALLY_ACTIVE" if rates[row["candidate_id"]]["effective_rate_s"] > 0
                    else "GEOMETRICALLY_VALID_KINETICALLY_DORMANT",
                "instantaneous_status":"POSITIVE_DOWNSTREAM_DRIVE" if rates[row["candidate_id"]]["effective_rate_s"] > 0
                    else "ZERO_DOWNSTREAM_DRIVE",
                "effective_rate_s":rates[row["candidate_id"]]["effective_rate_s"],
                "crossing_time_s":rates[row["candidate_id"]]["crossing_time_s"],
                "resolved_opening_stress_Pa":rates[row["candidate_id"]]["resolved_opening_stress_Pa"]}
                for row in source_candidates)
            source = {**trial.junction_process_state.get("active_event_source",{}),
                **_source_identity(trial,metrics["tensor_Pa"],source_kind="cavity_surface",
                    source_cavity_id=cavity.cavity_id,source_boundary_site_id="connection_exit",
                    source_position_m=cavity.connection_exit_m,
                    source_probe_identity={"kind":"CAVITY_FIXED_PHYSICAL_ARC_PATCH_RECOVERY_V3",
                        "boundary_node_id":metrics["boundary_node_id"],
                        "element_ids":metrics["probe_element_ids"],
                        "physical_arc_identity":metrics["recovery_record"]["physical_arc_identity"]}),
                "source_mesh_generation":trial.event_counters.get("mesh_generation",0),
                "candidate_source_states":candidate_rows}
            trial = replace(trial,junction_process_state={**trial.junction_process_state,
                "active_event_source":source})
            return trial,{"status":"SOURCE_TENSOR_QUALIFIED","attempts":rows,"operations":operations}
        current = trial; previous = metrics
    return state,{"status":"SOURCE_TENSOR_UNQUALIFIED","attempts":rows,"operations":operations}


def _complete_next_clock_fused(state, stress_tensor_Pa=None, *, source_kind="sharp_front",
                         source_front_id=None, source_cavity_id=None,
                         source_boundary_site_id=None, source_position_m=None,
                         source_probe_identity=None, temperature_K=900.0,
                         maximum_advance_duration_s=None, front_load_rows=None,
                         functional_test_rate_scale=1.0,
                         test_only_functional_qualification=False):
    """Advance source-owned clocks, never an unqualified cavity first passage."""
    if maximum_advance_duration_s is not None and (
        not math.isfinite(float(maximum_advance_duration_s)) or maximum_advance_duration_s < 0):
        raise ValueError("maximum_advance_duration_s must be finite and nonnegative")
    if front_load_rows is None:
        stress = np.asarray(stress_tensor_Pa, dtype=float).reshape(2, 2)
        rates = directional_clock_rates(state, stress, temperature_K=temperature_K)
    else:
        if source_kind != "sharp_front" or source_front_id is None:
            raise ValueError("directional J/K loads require an explicit sharp-front owner")
        stress = None
        rates = directional_sharp_front_rates(
            state, front_load_rows, branch_id=source_front_id,
            temperature_K=temperature_K,
        )
    rate_scale = float(functional_test_rate_scale)
    if not math.isfinite(rate_scale) or rate_scale <= 0.0:
        raise ValueError("directional functional-test rate scale must be finite and positive")
    if rate_scale != 1.0 and not bool(test_only_functional_qualification):
        raise RuntimeError(
            "directional rate scaling is restricted to labeled functional qualification")
    if rate_scale != 1.0:
        for rate in rates:
            original_rate = float(rate.get("rate_s", rate["effective_rate_s"]))
            rate["unscaled_effective_rate_s"] = float(rate["effective_rate_s"])
            rate["unscaled_crossing_time_s"] = float(rate["crossing_time_s"])
            rate["effective_rate_s"] = float(rate["effective_rate_s"])*rate_scale
            rate["rate_s"] = original_rate*rate_scale
            rate["crossing_time_s"] = (
                float(rate["crossing_time_s"])/rate_scale
                if math.isfinite(float(rate["crossing_time_s"])) else math.inf)
            rate["functional_test_rate_scale"] = rate_scale
            rate["rate_scale_classification"] = "CONTROLLED_NOT_CALIBRATED_REFERENCE"
    junction = dict(state.junction_process_state)
    start_time = float(junction.get("production_time_s", 0.0))
    from .canonical_kinetic_time_v1 import AcceptedTime
    accepted_clock = junction.get('canonical_accepted_time_v1', AcceptedTime.from_seconds(start_time))
    if accepted_clock.seconds() != start_time: accepted_clock = AcceptedTime.from_seconds(start_time)
    source = _source_identity(
        state, stress, source_kind=source_kind, source_front_id=source_front_id,
        source_cavity_id=source_cavity_id, source_boundary_site_id=source_boundary_site_id,
        source_position_m=source_position_m, source_probe_identity=source_probe_identity,
        source_load_payload=front_load_rows,
    )
    from .finalization_v3_schema import canonical_hash
    source_signature = canonical_hash(source)
    # Policy C: activity requires the actual owned production-source transfer
    # qualification. Neither a caller-supplied PASS flag nor a static traction
    # screen grants authority; the versioned fine-source transaction may qualify.
    # The accepted connection and all clock/RNG/graph state are returned intact.
    # Check owned phase/source as well as the argument to prevent source spoofing.
    owned_cavity = (state.junction_process_state.get("active_event_source", {}).get("source_kind") == "cavity_surface"
        or (state.void_state is not None and any(c.phase == VoidPhase.CONNECTED_VOID for c in state.void_state.cavities)))
    source_qualified = (
        _boundary_fitted_probe_is_qualified(state, stress, source_probe_identity)
        or _qualified_cavity_source(state, stress, temperature_K=temperature_K)
    ) if source_kind == "cavity_surface" or owned_cavity else True
    if ((source_kind == "cavity_surface" or owned_cavity)
        and any(r["effective_rate_s"] > 0 for r in rates)
        and not source_qualified):
        for rate in rates:
            rate.update({"unqualified_effective_rate_s": rate["effective_rate_s"],
                         "unqualified_crossing_time_s": rate["crossing_time_s"],
                         "effective_rate_s": 0., "rate_s": 0., "crossing_time_s": math.inf,
                         "common_advance_duration_s": 0., "emitted_event_ids": [], "winner": False,
                         "instantaneous_status": "UNQUALIFIED_CAVITY_SOURCE_TENSOR",
                         "source_resolution_policy": "C_FIRST_PASSAGE_UNAVAILABLE_UNTIL_QUALIFIED"})
        return state, rates
    if owned_cavity:
        # Retain complete candidate geometry/threshold/probe provenance during
        # dormancy; a bare _source_identity would erase the candidate inventory.
        source = {**state.junction_process_state.get("active_event_source", {}),
                  **(source if any(r["effective_rate_s"] > 0 for r in rates) else {})}
    elif source_kind=='sharp_front':
        # A partial accepted clock interval does not renew the front or erase
        # its geometry-owned candidate endpoints. Only retain them for the
        # same front, position and graph generation; tensor/probe metadata is
        # refreshed above and the eventual transaction still certifies geometry.
        previous_source=state.junction_process_state.get('active_event_source',{})
        same_owner=all(previous_source.get(key)==source.get(key) for key in (
            'source_kind','source_front_id','source_geometry_generation')) and tuple(
                previous_source.get('source_position_m') or ())==tuple(source.get('source_position_m') or ())
        if same_owner:
            source={**previous_source,**source}
    hazards = list(state.competition.hazard_states)
    crossing_times = [rate["crossing_time_s"] for rate in rates]
    duration = min(crossing_times, default=math.inf)
    if not math.isfinite(duration):
        duration = 0.0 if maximum_advance_duration_s is None else float(maximum_advance_duration_s)
        if duration < 0.0 or not math.isfinite(duration):
            raise ValueError("maximum_advance_duration_s must be finite and nonnegative")
        for rate in rates:
            rate.update({"common_advance_duration_s": duration, "emitted_event_ids": [],
                         "winner": False,
                         "instantaneous_status": "ZERO_DOWNSTREAM_DRIVE"})
        junction.update({
            "production_time_s": accepted_clock.advance(duration).seconds(),
            "canonical_accepted_time_v1": accepted_clock.advance(duration),
            "active_event_source": source,
            "latest_directional_clock_status": "NO_KINETICALLY_ACTIVE_CANDIDATE",
        })
        return replace(state, junction_process_state=junction), rates
    if maximum_advance_duration_s is not None:
        bounded = float(maximum_advance_duration_s)
        if bounded < 0.0 or not math.isfinite(bounded):
            raise ValueError("maximum_advance_duration_s must be finite and nonnegative")
        duration = min(duration, bounded)
    from .canonical_directional_interval_v1 import advance as advance_canonical
    hazards, completed, anchors, exact_duration = advance_canonical(tuple(hazards), rates,
        clock=accepted_clock, anchors=junction.get('directional_integration_anchors_v1', {}),
        source_signature=source_signature, maximum_duration_s=maximum_advance_duration_s)
    duration = float(exact_duration)
    emitted_winner_ids = []
    for rate, events in zip(rates, completed):
        rate["common_advance_duration_s"] = duration
        rate["emitted_event_ids"] = [event.event_id for event in events]
        rate["winner"] = bool(events)
        if events:
            emitted_winner_ids.append(rate["candidate_id"])
    pending_new = {
        event.candidate_id for before, after in zip(state.competition.hazard_states, hazards)
        for event in after.pending_events[len(before.pending_events):]
    }
    if pending_new != set(emitted_winner_ids):
        raise AssertionError("pending events do not match emitted first-passage winners")
    provenance = dict(junction.get("directional_event_provenance", {}))
    threshold_by_candidate = {hazard.candidate_id: {
        "threshold_process": hazard.threshold_process,
        "threshold_seed": hazard.threshold_seed,
        "threshold_action": hazard.current_threshold_action,
    } for hazard in state.competition.hazard_states}
    for hazard in hazards:
        before_ids = {event.event_id for old in state.competition.hazard_states
                      for event in old.pending_events}
        for event in hazard.pending_events:
            if event.event_id in before_ids:
                continue
            provenance[event.event_id] = {
                **source, "completion_time_s": event.completion_time_s,
                "candidate_id": event.candidate_id,
                "threshold_identity": threshold_by_candidate[event.candidate_id],
                "status": "PENDING_SOURCE_VALIDATION",
            }
    junction.update({"production_time_s": accepted_clock.advance(exact_duration).seconds(),
                     "canonical_accepted_time_v1": accepted_clock.advance(exact_duration),
                     "directional_integration_anchors_v1": anchors,
                     "directional_event_provenance": provenance,
                     "active_event_source": source,
                     "latest_directional_clock_status": (
                         "FIRST_PASSAGE_COMPLETED" if emitted_winner_ids
                         else "KINETICALLY_ACTIVE_NO_COMPLETION"
                     )})
    return replace(state, competition=replace(state.competition, hazard_states=tuple(hazards)),
                   junction_process_state=junction), rates


def preview_directional_first_passage(
    state, stress_tensor_Pa=None, *, source_kind="sharp_front",
    source_front_id=None, source_cavity_id=None, source_boundary_site_id=None,
    source_position_m=None, source_probe_identity=None, temperature_K=900.0,
    maximum_advance_duration_s=None, front_load_rows=None, failure_injector=None,
    functional_test_rate_scale=1.0,
    test_only_functional_qualification=False,
):
    """Pure preview of the established directional first-passage transaction.

    The old routine already returns a new immutable accepted-state graph.  This
    adapter evaluates that exact routine on an isolated state and freezes its
    result as the commit payload.  No threshold, pending event, RNG state, or
    accepted time in ``state`` is changed by preview.
    """
    from .global_physical_scheduler_v1 import make_directional_preview
    return make_directional_preview(
        state, stress_tensor_Pa=stress_tensor_Pa, source_kind=source_kind,
        source_front_id=source_front_id, source_cavity_id=source_cavity_id,
        source_boundary_site_id=source_boundary_site_id,
        source_position_m=source_position_m,
        source_probe_identity=source_probe_identity,
        temperature_K=temperature_K,
        maximum_advance_duration_s=maximum_advance_duration_s,
        front_load_rows=front_load_rows, failure_injector=failure_injector,
        functional_test_rate_scale=functional_test_rate_scale,
        test_only_functional_qualification=test_only_functional_qualification,
    )


def advance_directional_clocks(state, preview, *, failure_injector=None):
    """Advance all directional owners by the frozen preview interval."""
    from .global_physical_scheduler_v1 import advance_directional_preview
    return advance_directional_preview(state, preview, failure_injector=failure_injector)


def commit_directional_first_passage(state, preview, *, failure_injector=None):
    """Validate and expose the event already localized by ``preview``."""
    from .global_physical_scheduler_v1 import commit_directional_preview
    return commit_directional_preview(state, preview, failure_injector=failure_injector)


def _complete_next_clock(state, stress_tensor_Pa=None, *, source_kind="sharp_front",
                         source_front_id=None, source_cavity_id=None,
                         source_boundary_site_id=None, source_position_m=None,
                         source_probe_identity=None, temperature_K=900.0,
                         maximum_advance_duration_s=None, front_load_rows=None,
                         functional_test_rate_scale=1.0,
                         test_only_functional_qualification=False):
    """Compatibility wrapper over preview/advance/commit.

    Its accepted state and audit are the exact objects produced by the former
    fused implementation, so existing event times and ownership are unchanged.
    """
    preview = preview_directional_first_passage(
        state, stress_tensor_Pa, source_kind=source_kind,
        source_front_id=source_front_id, source_cavity_id=source_cavity_id,
        source_boundary_site_id=source_boundary_site_id,
        source_position_m=source_position_m,
        source_probe_identity=source_probe_identity,
        temperature_K=temperature_K,
        maximum_advance_duration_s=maximum_advance_duration_s,
        front_load_rows=front_load_rows,
        functional_test_rate_scale=functional_test_rate_scale,
        test_only_functional_qualification=test_only_functional_qualification,
    )
    advanced, audit = advance_directional_clocks(state, preview)
    return commit_directional_first_passage(advanced, preview), audit


def _select_emitted_proposal(state, audit, *, eligible_candidate_ids=None):
    emitted_ids = {event_id for row in audit for event_id in row["emitted_event_ids"]}
    eligible = None if eligible_candidate_ids is None else set(eligible_candidate_ids)
    proposals = tuple(
        proposal for proposal in construct_action_proposals(
            state.competition.hazard_states, correlation_interval_s=0.0,
        )
        if set(proposal.member_event_ids).issubset(emitted_ids)
        and proposal.action_type == "one_arm"
        and (eligible is None or set(proposal.member_candidate_ids).issubset(eligible))
    )
    proposal = select_temporal_or_degenerate_proposal(
        proposals, global_hazard_seed=state.competition.global_hazard_seed,
        competition_event_index=state.competition.competition_event_index,
    )
    winners = {row["candidate_id"] for row in audit if row["winner"]}
    if not set(proposal.member_candidate_ids).issubset(winners):
        raise AssertionError("selected proposal is not owned by emitted winners")
    return proposal


def _prepare_connected_ligament_support(state, *, entry, exit_point, direction,
                                        failure_injector=None, new_connection=True):
    """Stage exact inactive cavity contact before any new support certificate.

    Called only on the refined isolated topology trial. No certificate is
    waived, and the enclosing energy/topology transaction still owns commit.
    Boundary IDs are freshly derived from this mesh, never transferred IDs.
    """
    cavity = state.void_state.cavities[0]
    root = state.crack_network.branch(ROOT_BRANCH_ID)
    if math.dist(root.tip, entry) > 1.0e-12:
        raise ValueError("connected terminal must equal the realized ligament endpoint")
    connected = replace(cavity, phase=VoidPhase.CONNECTED_VOID,
        lineage=cavity.lineage + (("CRACK_TO_VOID_LIGAMENT",) if new_connection else ()),
        connection_entry_m=entry, connection_exit_m=exit_point,
        connection_direction_xy=direction)
    state = replace(state, void_state=replace_cavity(state.void_state, connected))
    if failure_injector is not None: failure_injector("cavity_phase_update", state)
    network = replace(state.crack_network, branches=tuple(
        replace(branch, status="arrested", local_state={**branch.local_state,
            "terminal_boundary_kind": "traction_free_cavity"})
        if branch.branch_id == ROOT_BRANCH_ID else branch
        for branch in state.crack_network.branches))
    state = replace(state, crack_network=network)
    if failure_injector is not None: failure_injector("root_status_change", state)
    cycle = cavity_free_surface_certificate(state)
    if not cycle["passed"]:
        raise RuntimeError("connected terminal requires a certified closed cavity cycle")
    contexts = _external_free_root_context(state.mesh, state.boundary, root.path[0])
    found = False
    for arc_start, arc_end, arc_id in certification_arcs(network):
        if math.dist(arc_end, entry) <= 1.0e-12: endpoint = "end"
        elif math.dist(arc_start, entry) <= 1.0e-12: endpoint = "start"
        else: continue
        found = True
        context = {"endpoint": endpoint, "endpoint_role": "inactive_terminal",
            "endpoint_coordinate_m": tuple(map(float, entry)),
            "boundary_kind": "cavity_free_surface",
            "boundary_component_id": "cavity-cycle:" + cavity.cavity_id,
            "boundary_edge_ids": tuple(tuple(map(int, edge)) for edge in cycle["boundary_edge_ids"]),
            "cavity_id": cavity.cavity_id, "certified_cavity_id": cavity.cavity_id,
            "cavity_cycle_certified": True, "tangent_enters_or_approaches_solid": True}
        contexts[arc_id] = tuple(contexts.get(arc_id, ())) + (context,)
    if not found:
        raise RuntimeError("connected terminal has no exact certification arc")
    return replace(state, junction_process_state={**state.junction_process_state,
        "boundary_terminal_context": contexts})


def _refresh_downstream_boundary_context(state):
    """Re-derive endpoint incidence from the actual child-trial mesh.

    Refinement invalidates stored boundary edge IDs. This does not change any
    role or waive any certificate: active tips remain prohibited, and every
    new context is independently checked by the existing support verifier.
    """
    from .mechanically_separating_sharp_wake_v12 import classify_graph_vertices
    connected = () if state.void_state is None else tuple(
        cavity for cavity in state.void_state.cavities
        if cavity.phase in (
            VoidPhase.CONNECTED_VOID,
            VoidPhase.DOWNSTREAM_FRONT_ACTIVE,
        )
    )
    if not connected:
        root = state.crack_network.branch(ROOT_BRANCH_ID)
        return replace(state, junction_process_state={
            **state.junction_process_state,
            'boundary_terminal_context': _external_free_root_context(
                state.mesh, state.boundary, root.path[0],
            ),
        })
    if len(connected) != 1:
        raise RuntimeError('downstream support requires exactly one connected cavity')
    cavity = connected[0]
    root = state.crack_network.branch(ROOT_BRANCH_ID)
    cycle = cavity_free_surface_certificate(state)
    if not cycle['passed']: raise RuntimeError('downstream trial cavity cycle is not certified')
    contexts = _external_free_root_context(state.mesh, state.boundary, root.path[0])
    roles = classify_graph_vertices(state.crack_network)
    for first, last, arc_id in certification_arcs(state.crack_network):
        for endpoint, point in (('start', first), ('end', last)):
            role = roles.get(tuple(point), frozenset())
            if 'inactive_terminal' not in role or 'active_tip' in role: continue
            if not any(anchor is not None and math.dist(point, anchor) <= 1e-12
                       for anchor in (cavity.connection_entry_m, cavity.connection_exit_m)): continue
            context = {'endpoint': endpoint, 'endpoint_role': 'inactive_terminal',
                'endpoint_coordinate_m': tuple(map(float, point)), 'boundary_kind': 'cavity_free_surface',
                'boundary_component_id': 'cavity-cycle:'+cavity.cavity_id,
                'boundary_edge_ids': tuple(tuple(map(int, edge)) for edge in cycle['boundary_edge_ids']),
                'cavity_id': cavity.cavity_id, 'certified_cavity_id': cavity.cavity_id,
                'cavity_cycle_certified': True}
            contexts[arc_id] = tuple(contexts.get(arc_id, ()))+(context,)
    return replace(state, junction_process_state={**state.junction_process_state,
        'boundary_terminal_context': contexts})


def matched_ligament_energy_observation(
    state, *, branch_id, candidate, refinement_levels=(3, 4),
):
    """Observe a finite cavity connection on matched isolated mechanics meshes.

    This is a read-only mechanics observation.  It neither completes a clock
    nor commits the connection.  Each level constructs the exact polygonal
    entry/exit geometry, compares unextended and connected states on the same
    mesh, and retains the production support/cavity certificates.
    """
    from .hybrid_directional_drive_v5 import MARGINAL_G_MESH_RELATIVE_LIMIT
    from .topology_transaction_v11 import (
        TopologyArm, apply_v12_production_trial_geometry,
        complete_accepted_state_fingerprint,
        equilibrate_fixed_load_with_production_fem,
        remesh_mechanically_separating_v12,
    )

    levels = tuple(sorted(set(map(int, refinement_levels))))
    if len(levels) < 2 or any(level < 1 for level in levels):
        raise ValueError("ligament observation requires two positive refinement levels")
    before = complete_accepted_state_fingerprint(state)
    start = state.crack_network.branch(str(branch_id)).tip
    entry = _first_ray_cavity_intersection(state, start, candidate.direction_xy)
    if entry is None:
        raise ValueError("candidate does not intersect the active cavity")
    ligament_length = math.dist(start, entry)
    if ligament_length <= 0.0:
        raise ValueError("ligament observation requires positive solid length")
    rows = []
    for level in levels:
        aligned = state.isolated_copy()
        aligned, _, _ = _conform_cavity_intersection(aligned, entry)
        exit_point = _first_ray_cavity_intersection(
            aligned, entry, candidate.direction_xy,
        )
        if exit_point is None:
            raise RuntimeError("connected cavity has no far-side chord intersection")
        aligned, exit_node, _ = _conform_cavity_intersection(aligned, exit_point)
        exit_point = tuple(map(float, aligned.mesh.nodes[exit_node]))
        arm = TopologyArm(
            candidate.candidate_id, str(branch_id), start, entry,
            ligament_length, 0.0, candidate_direction_xy=candidate.direction_xy,
            first_intersection_xy_m=entry,
        )
        extended = apply_v12_production_trial_geometry(
            aligned, (arm,), source_commit=_head(),
            configuration={"event": "VIRTUAL_MATCHED_LIGAMENT_ENERGY"},
            transaction_identity=f"virtual-matched-ligament-L{level}",
            refinement_levels=level, preserve_cavity_boundary_edges=True,
            prepare_support_state=lambda refined: _prepare_connected_ligament_support(
                refined, entry=entry, exit_point=exit_point,
                direction=candidate.direction_xy,
            ),
        )
        fields = _project_fields(aligned, extended.mesh)
        base = remesh_mechanically_separating_v12(
            aligned, mesh=extended.mesh, boundary=extended.boundary,
            transferred_fields=fields, source_commit=_head(),
            configuration={"event": "VIRTUAL_MATCHED_LIGAMENT_UNEXTENDED"},
            transaction_identity=f"virtual-matched-ligament-base-L{level}",
        )
        base = equilibrate_fixed_load_with_production_fem(base)
        extended = equilibrate_fixed_load_with_production_fem(extended)
        certificate = crack_void_connection_certificate(
            extended, branch_id=str(branch_id),
            cavity_id=extended.void_state.cavities[0].cavity_id,
            intended_intersection=entry,
        )
        if not certificate["passed"]:
            raise RuntimeError("virtual connected ligament certificate failed")
        rows.append({
            "refinement_level": level,
            "mesh_nodes": int(base.mesh.nn),
            "mesh_elements": int(base.mesh.ne),
            "Pi_unextended_J_per_m": float(base.stored_energy_J_per_m),
            "Pi_connected_J_per_m": float(extended.stored_energy_J_per_m),
            "G_ligament_J_per_m2": (
                float(base.stored_energy_J_per_m-extended.stored_energy_J_per_m)
                / ligament_length
            ),
            "connection_certificate": certificate,
        })
    first, finest = rows[-2], rows[-1]
    error = abs(
        finest["G_ligament_J_per_m2"]-first["G_ligament_J_per_m2"]
    )/max(abs(finest["G_ligament_J_per_m2"]),
          abs(first["G_ligament_J_per_m2"]), 1.0e-300)
    if complete_accepted_state_fingerprint(state) != before:
        raise RuntimeError("ligament energy observation mutated the accepted state")
    return {
        "schema": "v5.matched-finite-ligament-energy-observation/1",
        "status": ("AVAILABLE" if error <= MARGINAL_G_MESH_RELATIVE_LIMIT
                   else "DIRECTIONAL_DRIVE_UNAVAILABLE"),
        "candidate_id": candidate.candidate_id,
        "branch_id": str(branch_id),
        "entry_m": tuple(map(float, entry)),
        "ligament_length_m": ligament_length,
        "G_ligament_J_per_m2": float(finest["G_ligament_J_per_m2"]),
        "mesh_relative_error": error,
        "mesh_relative_limit": MARGINAL_G_MESH_RELATIVE_LIMIT,
        "levels": tuple(rows),
        "accepted_state_preserved": True,
        "physical_time_advanced": False,
        "transport_solve_performed": False,
    }


def ligament_transaction(state, *, failure_stage=None, operation_log=None,
                         _precompleted_directional=None):
    cavity = state.void_state.cavities[0]
    start = state.crack_network.branch(ROOT_BRANCH_ID).tip
    tensor, source_element = crack_tip_tensor(state, branch_id=ROOT_BRANCH_ID)
    if _precompleted_directional is None:
        state, cleavage_audit = _complete_next_clock(
            state, tensor, source_kind="sharp_front", source_front_id=ROOT_BRANCH_ID,
            source_position_m=start, source_probe_identity={"kind": "crack_tip_tensor",
                                                            "element_ids": list(source_element)},
        )
    else:
        state, cleavage_audit = _precompleted_directional
    intersections = {candidate.candidate_id: _first_ray_cavity_intersection(state, start, candidate.direction_xy)
                     for candidate in state.competition.candidates}
    eligible_ids = {candidate_id for candidate_id, intersection in intersections.items() if intersection is not None}
    proposal = _select_emitted_proposal(state, cleavage_audit, eligible_candidate_ids=eligible_ids)
    candidate = next(item for item in state.competition.candidates if item.candidate_id == proposal.member_candidate_ids[0])
    end = intersections[candidate.candidate_id]
    if end is None:
        raise RuntimeError("selected candidate ray does not intersect the cavity polygon")
    engine = fresh_sharp_front_engine(state.material, _bundle_from_state(state))
    winner = next(item for item in cleavage_audit if item["candidate_id"] == candidate.candidate_id)
    barrier = winner["hazard_barrier_J"]
    resistance = hazard_resistance_J_per_m2(
        barrier_J=barrier, cooperative_hits=engine.f.m_hits,
        burgers_vector_m=engine.b, gamma_relative=candidate.gamma_rel,
    )
    arm = TopologyArm(
        proposal.member_candidate_ids[0], ROOT_BRANCH_ID, start, end, math.dist(start, end),
        resistance * math.dist(start, end), event_classification="physical_cleavage",
        candidate_direction_xy=candidate.direction_xy, first_intersection_xy_m=end,
    )
    operations = operation_log if operation_log is not None else []
    def inject(stage, current):
        operations.append(stage)
        if stage == failure_stage: raise RuntimeError("injected:" + stage)
    inject("ligament_hazard_completion", state)
    def geometry(trial, arms):
        trial, aligned_node, inserted = _conform_cavity_intersection(
            trial, end, failure_injector=inject,
        )
        inject("intersection_alignment", trial)
        exit_point = _first_ray_cavity_intersection(trial, end, candidate.direction_xy)
        if exit_point is None:
            raise RuntimeError("connected cavity has no far-side chord intersection")
        trial, exit_node, exit_inserted = _conform_cavity_intersection(
            trial, exit_point, failure_injector=inject,
        )
        exit_point = tuple(map(float, trial.mesh.nodes[exit_node]))
        inject("exit_alignment", trial)
        realized = apply_v12_production_trial_geometry(
            trial, arms, source_commit=_head(), configuration={"event": "CRACK_TO_VOID_LIGAMENT"},
            transaction_identity="ligament", failure_injector=inject,
            preserve_cavity_boundary_edges=True,
            prepare_support_state=lambda refined: _prepare_connected_ligament_support(
                refined, entry=end, exit_point=exit_point, direction=candidate.direction_xy,
                failure_injector=inject),
        )
        void_state = realized.void_state
        ledgers = dict(void_state.length_ledgers)
        ligament_length = math.dist(start, end)
        physical_span = math.dist(end, exit_point)
        projected_span = float(np.asarray(exit_point)[0] - np.asarray(end)[0])
        increments = {
            "fractured_ligament_length_m": ligament_length,
            "active_front_coordinate_advance_m": ligament_length,
            "physical_active_front_travel_m": ligament_length,
            "projected_fractured_length_m": end[0] - start[0],
            "projected_front_advance_m": end[0] - start[0],
            "preexisting_void_free_span_m": physical_span,
            "connected_void_free_span_m": physical_span,
            "projected_connected_void_free_span_m": projected_span,
            "connected_free_surface_extent_m": _polygon_boundary_arc_length(realized, end, exit_point),
        }
        for name, increment in increments.items():
            ledgers[name] += increment
            void_state = replace(void_state, length_ledgers=dict(ledgers))
            inject("length_ledger_update:" + name, replace(realized, void_state=void_state))
        void_state = replace(
            void_state,
            event_history=void_state.event_history + ({"event": "CRACK_TO_VOID_LIGAMENT",
                                                        "candidate_id": candidate.candidate_id,
                                                        "source_element": source_element},),
        )
        realized = replace(realized, void_state=void_state)
        inject("dormant_support_rebuild", realized)
        cycle = cavity_free_surface_certificate(realized)
        certificate = crack_void_connection_certificate(
            realized, branch_id=ROOT_BRANCH_ID, cavity_id=cavity.cavity_id,
            intended_intersection=end,
        )
        if not certificate["passed"]:
            raise RuntimeError("connected free surface is not certified: " + repr(certificate))
        junction = dict(realized.junction_process_state)
        junction["latest_intersection_alignment"] = {
            "intersection_m": list(end), "aligned_node_id_before_refinement": aligned_node,
            "interior_edge_split_performed": inserted,
            "exit_intersection_m": list(exit_point), "exit_aligned_node_id": exit_node,
            "exit_interior_edge_split_performed": exit_inserted,
            "accepted_mesh_has_aligned_node": bool(np.min(np.linalg.norm(np.asarray(realized.mesh.nodes) - np.asarray(end), axis=1)) <= 1.0e-12),
        }
        junction["latest_closed_cavity_boundary_cycle_certificate"] = cycle
        junction["latest_crack_void_connection_certificate"] = certificate
        realized = replace(realized, junction_process_state=junction)
        inject("connected_surface_certification", realized)
        return realized
    result = execute_topology_trial(
        state, proposal, (arm,), apply_trial_geometry=geometry,
        equilibrate_fixed_load=equilibrate_fixed_load_with_production_fem,
        network_geometry_already_realized=True, failure_injector=inject,
    )
    if not result.accepted: raise RuntimeError("ligament event rejected")
    result_state = _mark_consumed_event_provenance(result.state, proposal.member_event_ids)
    cavity = result_state.void_state.cavities[0]
    exit_nodes = np.asarray(result_state.mesh.nodes)
    exit_node = int(np.argmin(np.linalg.norm(exit_nodes - np.asarray(cavity.connection_exit_m), axis=1)))
    from .cavity_source_recovery_v3 import CavitySourceRecoveryV3Unavailable
    try:
        surface_recovery = cavity_fixed_physical_arc_patch_recovery_v3(
            result_state, boundary_node=exit_node
        )
        surface_tensor = np.asarray(surface_recovery["tensor_Pa"], dtype=float)
        boundary_elements = tuple(map(int, surface_recovery["stencil_element_ids"]))
    except CavitySourceRecoveryV3Unavailable as exc:
        surface_tensor = np.zeros((2, 2), dtype=float)
        boundary_elements = ()
        surface_recovery = {
            "recovery_operator":"CAVITY_FIXED_PHYSICAL_ARC_PATCH_RECOVERY_V3",
            "available":False,
            "scientific_unavailability":str(exc),
            "physical_arc_identity":{
                "cavity_id":cavity.cavity_id,
                "boundary_position_m":list(map(float, cavity.connection_exit_m)),
            },
        }
    origin = np.asarray(cavity.connection_exit_m, dtype=float)
    candidate_inventory = tuple(result_state.competition.candidates)
    candidate_rows = []
    endpoints = {}
    for surface_candidate in candidate_inventory:
        direction = np.asarray(surface_candidate.direction_xy, dtype=float)
        direction /= np.linalg.norm(direction)
        endpoint = origin + 7.5e-5 * direction
        if not (1.0e-12 < endpoint[0] < 1.0e-3 - 1.0e-12
                and abs(endpoint[1]) < 5.0e-4 - 1.0e-12):
            continue
        normal = np.asarray(surface_candidate.normal_xy, dtype=float)
        opening = max(float(normal @ surface_tensor @ normal), 0.0)
        endpoints[surface_candidate.candidate_id] = list(map(float, endpoint))
        candidate_rows.append({
            "candidate_id": surface_candidate.candidate_id,
            "source_kind":"cavity_surface","source_cavity_id":cavity.cavity_id,
            "source_boundary_site_id":"connection_exit","source_position_m":list(cavity.connection_exit_m),
            "source_geometry_generation":int(result_state.crack_network.geometry_generation),
            "source_mesh_generation":int(result_state.event_counters.get("mesh_generation",0)),
            "source_tensor_fingerprint":_tensor_fingerprint(surface_tensor),
            "source_probe_identity":{"kind":"CAVITY_FIXED_PHYSICAL_ARC_PATCH_RECOVERY_V3",
                "boundary_node_id":exit_node,"element_ids":list(boundary_elements),
                "physical_arc_identity":surface_recovery["physical_arc_identity"]},
            "direction_xy": list(surface_candidate.direction_xy),
            "normal_xy": list(surface_candidate.normal_xy),
            "tangent_xy": list(surface_candidate.direction_xy),
            "planned_endpoint_m": endpoints[surface_candidate.candidate_id],
            "geometry_status": "GEOMETRICALLY_VALID_KINETICALLY_DORMANT"
                               if opening <= 0.0 else "GEOMETRICALLY_VALID_KINETICALLY_UNAVAILABLE",
            "instantaneous_status": "ZERO_DOWNSTREAM_DRIVE"
                                    if opening <= 0.0 else "SOURCE_TENSOR_UNQUALIFIED",
            "resolved_opening_stress_Pa": opening,
            "effective_rate_s": 0.0,
            "crossing_time_s": "infinity",
        })
    candidate_inventory = tuple(candidate for candidate in candidate_inventory
                                if candidate.candidate_id in endpoints)
    if not candidate_inventory:
        raise RuntimeError("connected cavity has no geometrically admissible downstream direction")
    surface_source = _source_identity(
        result_state, surface_tensor, source_kind="cavity_surface",
        source_cavity_id=cavity.cavity_id, source_boundary_site_id="connection_exit",
        source_position_m=cavity.connection_exit_m,
        source_probe_identity={"kind": "CAVITY_FIXED_PHYSICAL_ARC_PATCH_RECOVERY_V3",
                               "boundary_node_id": exit_node,
                               "element_ids": list(boundary_elements),
                               "physical_arc_identity":surface_recovery["physical_arc_identity"]},
    )
    surface_source["candidate_source_states"] = tuple(candidate_rows)
    surface_source["source_mesh_generation"] = int(result_state.event_counters.get("mesh_generation",0))
    surface_source["next_candidate_endpoints_m"] = endpoints
    transitioned = _transition_competition_source(
        result_state, surface_source, candidates=candidate_inventory,
    )
    return transitioned, replace(result, state=transitioned)


def downstream_source_observation(state, *, continuation=False):
    """Pure source/endpoint observation shared by preview and transaction."""
    child_id = "void-front-1"
    cavity = state.void_state.cavities[0]
    active_source = state.junction_process_state.get("active_event_source", {})
    if continuation:
        if state.crack_network.active_tip_ids != (child_id,):
            raise RuntimeError("continued propagation requires the sole active downstream child")
        start = state.crack_network.branch(child_id).tip
        rows = sharp_front_load_provider(state, branch_id=child_id)["directional"]
        return {
            "owner_id": child_id, "start_m": start, "tensor_Pa": None,
            "source_kind": "sharp_front", "source_front_id": child_id,
            "source_cavity_id": None, "source_boundary_site_id": None,
            "probe_identity": {
                "kind": "established_directional_J_K_provider", "branch_id": child_id,
                "candidate_ids": [row["candidate_id"] for row in rows],
            },
            "front_load_rows": rows,
            "endpoints": dict(active_source.get("next_candidate_endpoints_m", {})),
        }
    if cavity.connection_exit_m is None:
        raise RuntimeError("downstream nucleation requires the stored connection exit")
    start = tuple(map(float, cavity.connection_exit_m))
    nodes = np.asarray(state.mesh.nodes)
    exit_node = int(np.argmin(np.linalg.norm(nodes-np.asarray(start), axis=1)))
    if np.linalg.norm(nodes[exit_node]-np.asarray(start)) > 1.0e-12:
        raise RuntimeError("stored connection exit is not an aligned cavity-boundary node")
    recovery = qualified_boundary_fitted_source_observation(state)
    if recovery["qualification_status"] != "AVAILABLE":
        raise RuntimeError("boundary-fitted downstream source did not qualify")
    probe = _boundary_fitted_source_probe(state, recovery, boundary_node=exit_node)
    endpoints = dict(active_source.get("next_candidate_endpoints_m", {}))
    fallback = active_source.get("next_mesh_aligned_endpoint_m")
    if fallback is not None and len(state.competition.candidates) == 1:
        endpoints.setdefault(state.competition.candidates[0].candidate_id, fallback)
    return {
        "owner_id": cavity.cavity_id, "start_m": start,
        "tensor_Pa": np.asarray(recovery["tensor_Pa"], dtype=float),
        "source_kind": "cavity_surface", "source_front_id": None,
        "source_cavity_id": cavity.cavity_id,
        "source_boundary_site_id": "connection_exit",
        "probe_identity": probe,
        "front_load_rows": None, "endpoints": endpoints,
    }


def downstream_front_transaction(state, *, continuation=False, failure_stage=None,
                                 operation_log=None, _precompleted_directional=None,
                                 _precompleted_front_process=None):
    child_id = "void-front-1"
    operations = operation_log if operation_log is not None else []
    def early_inject(stage):
        operations.append(stage)
        if stage == failure_stage:
            raise RuntimeError("injected:" + stage)
    cavity = state.void_state.cavities[0]
    nodes = np.asarray(state.mesh.nodes)
    active_source = state.junction_process_state.get("active_event_source", {})
    front_load_rows = None
    if continuation:
        if state.crack_network.active_tip_ids != (child_id,):
            raise RuntimeError("continued propagation requires the sole active downstream child")
        start = state.crack_network.branch(child_id).tip
        front_load_rows = sharp_front_load_provider(
            state, branch_id=child_id,
        )["directional"]
        tensor, boundary_elements = None, ()
        source_kind = "sharp_front"
        probe_identity = {
            "kind": "established_directional_J_K_provider",
            "branch_id": child_id,
            "candidate_ids": [row["candidate_id"] for row in front_load_rows],
        }
    else:
        if cavity.connection_exit_m is None:
            raise RuntimeError("downstream nucleation requires the stored connection exit")
        start = tuple(map(float, cavity.connection_exit_m))
        exit_node = int(np.argmin(np.linalg.norm(nodes - np.asarray(start), axis=1)))
        if np.linalg.norm(nodes[exit_node] - np.asarray(start)) > 1.0e-12:
            raise RuntimeError("stored connection exit is not an aligned cavity-boundary node")
        try:
            recovery = qualified_boundary_fitted_source_observation(state)
            if recovery["qualification_status"] != "AVAILABLE":
                raise RuntimeError("boundary-fitted downstream source did not qualify")
        except RuntimeError as exc:
            return state, None, operation_log if operation_log is not None else [], {
                "status": "UNQUALIFIED_CAVITY_SOURCE_TENSOR",
                "scientific_unavailability": str(exc),
                "recovery_operator": "CAVITY_BOUNDARY_FITTED_PHYSICAL_PATCH_V1",
                "source_kind": "cavity_surface",
                "source_position_m": list(start),
            }
        tensor = np.asarray(recovery["tensor_Pa"], dtype=float)
        boundary_elements = tuple(map(int, recovery["stencil_element_ids"]))
        source_kind = "cavity_surface"
        probe_identity = _boundary_fitted_source_probe(
            state, recovery, boundary_node=exit_node,
        )
    early_inject("source_observation")
    pre_clock_state = state
    if _precompleted_directional is None:
        state, cleavage_audit = _complete_next_clock(
            state, tensor, source_kind=source_kind,
            source_front_id=child_id if continuation else None,
            source_cavity_id=None if continuation else cavity.cavity_id,
            source_boundary_site_id=None if continuation else "connection_exit",
            source_position_m=start, source_probe_identity=probe_identity,
            front_load_rows=front_load_rows,
        )
    else:
        state, cleavage_audit = _precompleted_directional
    early_inject("source_clock_update")
    if not any(row["winner"] for row in cleavage_audit):
        return state, None, operation_log if operation_log is not None else [], {
            "status": ("UNQUALIFIED_CAVITY_SOURCE_TENSOR" if any(
                row.get("instantaneous_status") == "UNQUALIFIED_CAVITY_SOURCE_TENSOR" for row in cleavage_audit)
                else "NO_KINETICALLY_ACTIVE_CANDIDATE"),
            "tensor_Pa": None if tensor is None else tensor.tolist(),
            "front_load_rows": front_load_rows,
            "boundary_element_ids": boundary_elements,
            "source_kind": source_kind, "source_front_id": child_id if continuation else None,
            "source_position_m": list(start), "source_probe_identity": probe_identity,
            "cleavage": cleavage_audit,
        }
    if operation_log is not None: operation_log.append('downstream_threshold_completion')
    if failure_stage == 'downstream_threshold_completion':
        raise RuntimeError('injected:downstream_threshold_completion')
    proposal = _select_emitted_proposal(state, cleavage_audit)
    candidate = next(item for item in state.competition.candidates
                     if item.candidate_id == proposal.member_candidate_ids[0])
    planned_endpoint = active_source.get("next_candidate_endpoints_m", {}).get(candidate.candidate_id)
    if planned_endpoint is None:
        planned_endpoint = active_source.get("next_mesh_aligned_endpoint_m")
    end = tuple(map(float, planned_endpoint or ()))
    if len(end) != 2:
        raise RuntimeError("event source lacks its mesh-aligned endpoint")
    if continuation:
        base_network = state.crack_network
    else:
        child = CrackBranchState(child_id, ROOT_BRANCH_ID, 1,
                                 int(state.event_counters.get("topology_actions", 0)) + 1,
                                 (start,), (candidate.angle_rad,), local_state={
                                     "nucleation_source": "CAVITY_BOUNDARY_FITTED_PHYSICAL_PATCH_V1",
                                     "upstream_lineage_branch_id": ROOT_BRANCH_ID,
                                     "active_source": "established_directional_J_K_provider",
                                     "front_engine_state_owner": child_id,
                                     "front_engine_initialization_policy": "FrontEngine.__init__->reset",
                                 })
        base_network = replace(state.crack_network, branches=state.crack_network.branches + (child,),
                               geometry_generation=state.crack_network.geometry_generation + 1,
                               branching_enabled=True)
    engine = fresh_sharp_front_engine(state.material, _bundle_from_state(state))
    winner = next(item for item in cleavage_audit if item["candidate_id"] == candidate.candidate_id)
    front_process_advance = _precompleted_front_process
    if continuation and front_process_advance is None:
        from .legacy_front_process_adapter import LegacyFrontProcessAdapter
        process_owner_id = _canonical_front_process_owner(pre_clock_state, child_id)
        payload = pre_clock_state.tip_process_state.get("by_branch", {}).get(process_owner_id)
        if payload is None:
            raise RuntimeError("continued child has no canonical legacy process owner")
        pre_progress = max(
            (float(item.residual_action) for item in pre_clock_state.competition.hazard_states),
            default=0.0,
        )
        post_progress = max(
            (float(item.residual_action) for item in state.competition.hazard_states),
            default=pre_progress,
        )
        front_process_advance = LegacyFrontProcessAdapter(
            state.material, _bundle_from_state(state),
        ).advance_front_process(
            payload, float(winner["K_effective_Pa_sqrt_m"]), 900.0,
            float(winner["common_advance_duration_s"]),
            topology_owned_cleavage_action_before=pre_progress,
            topology_owned_cleavage_action_after=post_progress,
        )
        tip_state = dict(state.tip_process_state)
        by_branch = dict(tip_state.get("by_branch", {}))
        by_branch[process_owner_id] = front_process_advance.advanced_front_state
        tip_state["by_branch"] = by_branch
        state = replace(state, tip_process_state=tip_state)
    barrier = winner["hazard_barrier_J"]
    resistance = hazard_resistance_J_per_m2(
        barrier_J=barrier, cooperative_hits=engine.f.m_hits,
        burgers_vector_m=engine.b, gamma_relative=candidate.gamma_rel,
    )
    arm = TopologyArm(
        proposal.member_candidate_ids[0], child_id, start, end, math.dist(start, end),
        resistance * math.dist(start, end), event_classification="physical_cleavage",
        candidate_direction_xy=candidate.direction_xy, first_intersection_xy_m=end,
    )
    def inject(stage, current):
        operations.append(stage)
        if stage == failure_stage:
            raise RuntimeError("injected:" + stage)
    def geometry(trial, arms):
        trial = _conform_bulk_point(
            trial, end,
            identity="CONTINUED_FRONT_ENDPOINT_CONFORMING" if continuation
                     else "DOWNSTREAM_FIRST_PASSAGE_ENDPOINT_CONFORMING",
            failure_injector=inject,
        )
        if not continuation:
            inject("downstream_child_activation", replace(trial, crack_network=base_network))
        realized = apply_v12_production_trial_geometry(
            replace(trial, crack_network=base_network), arms,
            source_commit=_head(), configuration={"event": "DOWNSTREAM_FRONT" if not continuation else "CONTINUED_FRONT"},
            transaction_identity="downstream-continued" if continuation else "downstream-first-passage",
            failure_injector=inject, refinement_levels=1,
            prepare_support_state=_refresh_downstream_boundary_context,
        )
        inject('downstream_child_support_rebuild', realized)
        cavity = realized.void_state.cavities[0]
        updated = replace(cavity, phase=VoidPhase.DOWNSTREAM_FRONT_ACTIVE,
                          lineage=cavity.lineage + (("CONTINUED_EVENT" if continuation else "DOWNSTREAM_FIRST_PASSAGE"),))
        void_state = replace_cavity(realized.void_state, updated)
        inject("downstream_phase_update", replace(realized, void_state=void_state))
        ledgers = dict(void_state.length_ledgers)
        projected = end[0] - start[0]
        physical_span = cavity.connection_entry_m is not None and cavity.connection_exit_m is not None
        chord_length = math.dist(cavity.connection_entry_m, cavity.connection_exit_m) if physical_span else 0.0
        chord_projected = (cavity.connection_exit_m[0] - cavity.connection_entry_m[0]) if physical_span else 0.0
        increments = {
            "ordinary_crack_fractured_length_m": math.dist(start, end),
            "active_front_coordinate_advance_m": math.dist(start, end) + (chord_length if not continuation else 0.0),
            "physical_active_front_travel_m": math.dist(start, end) + (chord_length if not continuation else 0.0),
            "projected_fractured_length_m": projected,
            "projected_front_advance_m": projected + (chord_projected if not continuation else 0.0),
            "projected_free_span_m": chord_projected if not continuation else 0.0,
            "traversed_void_free_span_m": chord_length if not continuation else 0.0,
        }
        for name, increment in increments.items():
            ledgers[name] += increment
            void_state = replace(void_state, length_ledgers=dict(ledgers))
            inject("downstream_length_ledger_update:" + name, replace(realized, void_state=void_state))
        void_state = replace(void_state, event_history=void_state.event_history + ({
            "event": "CONTINUED_ACCEPTED_EVENT" if continuation else "DOWNSTREAM_FIRST_PASSAGE",
            "candidate_id": candidate.candidate_id,
            "source_kind": source_kind, "source_front_id": child_id if continuation else None,
            "source_cavity_id": None if continuation else cavity.cavity_id,
        },))
        inject("downstream_event_history_update", replace(realized, void_state=void_state))
        tip_state = dict(realized.tip_process_state)
        if not continuation:
            tip_state.update({
                "active_branch_id": child_id,
                "by_branch": {child_id: capture_sharp_front_engine(
                    fresh_sharp_front_engine(realized.material, _bundle_from_state(realized))
                )},
                "owner_by_front": {child_id: child_id},
            })
        realized = replace(realized, void_state=void_state, tip_process_state=tip_state)
        certificate = crack_void_connection_certificate(
            realized, branch_id=ROOT_BRANCH_ID, cavity_id=cavity.cavity_id,
            intended_intersection=cavity.connection_entry_m,
        )
        if not certificate["passed"]:
            raise RuntimeError("post-downstream combined topology is not certified: " + repr(certificate))
        junction = dict(realized.junction_process_state)
        junction["latest_crack_void_connection_certificate"] = certificate
        junction["latest_topology_certificate_stage"] = (
            "POST_CONTINUATION" if continuation else "ROOT_CAVITY_DOWNSTREAM_CHILD"
        )
        return replace(realized, junction_process_state=junction)
    result = execute_topology_trial(
        state, proposal, (arm,), apply_trial_geometry=geometry,
        equilibrate_fixed_load=equilibrate_fixed_load_with_production_fem,
        network_geometry_already_realized=True, failure_injector=inject,
    )
    if not result.accepted:
        raise RuntimeError(
            "downstream front event rejected: "
            f"release={result.energy_release_J_per_m!r} "
            f"dissipation={result.hazard_dissipation_J_per_m!r} "
            f"margin={result.energy_margin_J_per_m!r}"
        )
    accepted = _mark_consumed_event_provenance(result.state, proposal.member_event_ids)
    if continuation:
        process_owner_id = _canonical_front_process_owner(accepted, child_id)
        payload = accepted.tip_process_state.get("by_branch", {}).get(process_owner_id)
        child_engine = restore_sharp_front_engine(accepted.material, _bundle_from_state(accepted), payload)
        # The canonical process interval was already evolved exactly once.
        # Apply only the topology-owned moving-frame renewal after acceptance.
        from .tip_directional_observation_v11 import apply_post_interval_event_renewal
        step_info = dict(front_process_advance.audit["step_info"])
        step_info["interval_evolved"] = True
        step_info["da"] = 0.0
        apply_post_interval_event_renewal(
            child_engine.mpz, step_info, event_selected=True,
            event_distance_m=math.dist(start, end),
        )
        tip_state = dict(accepted.tip_process_state)
        by_branch = dict(tip_state.get("by_branch", {}))
        by_branch[process_owner_id] = {
            **capture_sharp_front_engine(child_engine),
            "last_event_identity": tuple(proposal.member_event_ids),
            "last_step_audit": step_info,
        }
        tip_state["by_branch"] = by_branch
        accepted = replace(accepted, tip_process_state=tip_state)
    # Every accepted child advance changes the source position and geometry
    # generation.  Rebuild the source-owned candidate endpoints after both
    # first passage *and* continuation; otherwise a second continuation sees
    # the already-consumed endpoint at the previous tip.
    if child_id in accepted.crack_network.active_tip_ids:
        child_load_rows = sharp_front_load_provider(
            accepted, branch_id=child_id,
        )["directional"]
        child_load_by_id = {row["candidate_id"]: row for row in child_load_rows}
        child_rates = {row["candidate_id"]: row for row in directional_sharp_front_rates(
            accepted, child_load_rows, branch_id=child_id,
        )}
        child_tip = np.asarray(accepted.crack_network.branch(child_id).tip, dtype=float)
        child_candidates = tuple(accepted.competition.candidates)
        child_endpoints = {}
        child_rows = []
        for child_candidate in child_candidates:
            direction = np.asarray(child_candidate.direction_xy, dtype=float)
            endpoint = child_tip + 7.5e-5 * direction
            if not (1.0e-12 < endpoint[0] < 1.0e-3 - 1.0e-12
                    and abs(endpoint[1]) < 5.0e-4 - 1.0e-12):
                continue
            load = child_load_by_id[child_candidate.candidate_id]
            response = child_rates[child_candidate.candidate_id]
            opening = float(response["sigma_tip_Pa"])
            child_endpoints[child_candidate.candidate_id] = list(map(float, endpoint))
            child_rows.append({
                "candidate_id": child_candidate.candidate_id,
                "direction_xy": list(child_candidate.direction_xy),
                "normal_xy": list(child_candidate.normal_xy),
                "tangent_xy": list(child_candidate.direction_xy),
                "planned_endpoint_m": child_endpoints[child_candidate.candidate_id],
                "geometry_status": "GEOMETRICALLY_VALID_KINETICALLY_DORMANT"
                                   if opening <= 0.0 else "GEOMETRICALLY_VALID_KINETICALLY_ACTIVE",
                "instantaneous_status": "ZERO_DOWNSTREAM_DRIVE"
                                        if opening <= 0.0 else "POSITIVE_DOWNSTREAM_DRIVE",
                "resolved_opening_stress_Pa": opening,
                "K_directional_Pa_sqrt_m": float(load["K_directional_Pa_sqrt_m"]),
                "signed_J_J_per_m2": float(load["signed_J_J_per_m2"]),
                "effective_rate_s": response["effective_rate_s"],
                "crossing_time_s": (response["crossing_time_s"] if math.isfinite(
                    response["crossing_time_s"]
                ) else "infinity"),
            })
        child_candidates = tuple(item for item in child_candidates
                                 if item.candidate_id in child_endpoints)
        child_source = _source_identity(
            accepted, None, source_kind="sharp_front", source_front_id=child_id,
            source_position_m=accepted.crack_network.branch(child_id).tip,
            source_probe_identity={"kind": "established_directional_J_K_provider", "branch_id": child_id},
            source_load_payload=child_load_rows,
        )
        child_source["candidate_source_states"] = tuple(child_rows)
        child_source["next_candidate_endpoints_m"] = child_endpoints
        if not child_candidates:
            child_source["terminal_geometry_status"] = (
                "NO_GEOMETRICALLY_ADMISSIBLE_FIXED_INCREMENT")
        accepted = _transition_competition_source(
            accepted, child_source, candidates=child_candidates,
        )
        if not child_candidates:
            accepted = replace(accepted, junction_process_state={
                **accepted.junction_process_state,
                "downstream_front_terminal_status": {
                    "reason": "SPECIMEN_BOUNDARY_FIXED_INCREMENT_EXHAUSTED",
                    "front_id": child_id,
                    "tip_m": tuple(map(float, child_tip)),
                    "attempted_increment_m": 7.5e-5,
                },
            })
        result = replace(result, state=accepted)
    cavity = accepted.void_state.cavities[0]
    child_process_owner = (
        _canonical_front_process_owner(accepted, child_id)
        if child_id in accepted.crack_network.active_tip_ids else None
    )
    return accepted, result, operations, {
        "tensor_Pa": None if tensor is None else tensor.tolist(),
        "front_load_rows": front_load_rows,
        "boundary_element_ids": boundary_elements,
        "source_kind": source_kind, "source_front_id": child_id if continuation else None,
        "source_position_m": list(start), "source_probe_identity": probe_identity,
        "r_eff_m": (restore_sharp_front_engine(
            accepted.material, _bundle_from_state(accepted),
            accepted.tip_process_state.get("by_branch", {}).get(child_process_owner),
        ).r_eff() if child_process_owner is not None else None),
        "candidate_id": proposal.member_candidate_ids[0],
        "selected_proposal_candidate_ids": list(proposal.member_candidate_ids),
        "emitted_winner_candidate_ids": [row["candidate_id"] for row in cleavage_audit if row["winner"]],
        "barrier_candidate_id": winner["candidate_id"],
        "cavity_id": cavity.cavity_id, "cleavage": cleavage_audit,
    }


def deterministic_trajectory(*, bundle: UnifiedFractureMaterialBundle, stop_before_ligament=False, cavity_center_m=(7.0e-4, 0.0),
                             crack_path_m=None, cleavage_theta_deg=0.0,
                             state_trace=None, boundary_segments=32, radial_layers=12,
                             qualify_source=False, common_restart_protocol=False,
                             source_conforming_geometry=False,
                             shape_regular_local_level=None,
                             resolved_radius_m=5.5e-5,
                             sentinel_initial_vacancy_fraction_gp=1.0,
                             vacancy_coupled=False,
                             promotion_radius_m=5.0e-5,
                             subgrid_radius_m=2.5e-5,
                             bulk_initial_density_m2=None,
                             initial_plastic_strain_gp=None,
                             specimen_width_m=1.0e-3,
                             specimen_height_m=1.0e-3,
                             far_h_m=5.0e-5):
    resolved_radius_m = float(resolved_radius_m)
    if not math.isfinite(resolved_radius_m) or resolved_radius_m < float(promotion_radius_m):
        raise ValueError("resolved cavity radius must be finite and at least the promotion radius")
    state, hole = build_production_void_state(bundle=bundle, enabled=True, cavity_center_m=cavity_center_m,
                                              crack_path_m=crack_path_m,
                                              cleavage_theta_deg=cleavage_theta_deg,
                                              boundary_segments=boundary_segments, radial_layers=radial_layers,
                                              source_conforming_geometry=source_conforming_geometry,
                                              shape_regular_local_level=shape_regular_local_level,
                                              temperature_K=900.0,
                                              initial_vacancy_fraction_gp=sentinel_initial_vacancy_fraction_gp,
                                              bulk_initial_density_m2=bulk_initial_density_m2,
                                              initial_plastic_strain_gp=initial_plastic_strain_gp,
                                              specimen_width_m=specimen_width_m,
                                              specimen_height_m=specimen_height_m,
                                              far_h_m=far_h_m,
                                              test_only_extreme_reservoir=(
                                                  sentinel_initial_vacancy_fraction_gp is not None
                                                  and sentinel_initial_vacancy_fraction_gp > 1.0e-2),
                                              cavity_geometry_radius_m=promotion_radius_m)
    cfg = VoidingConfig(enabled=True, promotion_radius_m=float(promotion_radius_m))
    rows = [observables(state, "available_site")]
    def capture(label):
        if state_trace is not None: state_trace.append((label,state))
    capture("available_site")
    accepted_plastic_rate = None
    if vacancy_coupled:
        state, accepted_plastic_rate, plastic_vacancy_audit = evolve_accepted_plastic_vacancy_interval(
            state, bundle=bundle, temperature_K=900.0, duration_s=0.1,
            test_only=True,
        )
        rows.append({
            **observables(state, "plastic_vacancy_production"),
            "plastic_vacancy_audit": plastic_vacancy_audit,
        })
        capture("plastic_vacancy_production")
    for label in ("multi_hit_1", "multi_hit_2"):
        tensor = local_site_tensor(state)
        if vacancy_coupled:
            rates, rate_audit = vacancy_coupled_void_rates(
                state, cfg, temperature_K=900.0, stress_tensor_Pa=tensor,
                center_m=state.void_state.sites[0].center_m,
                equivalent_plastic_strain_rate_gp=accepted_plastic_rate,
            )
        else:
            rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
            rate_audit = {}
        site = state.void_state.sites[0]
        dt = site.birth.crossing_time_exact(rates["birth_s"] * site.candidate_weight)
        if dt is None: raise RuntimeError('NO_KINETICALLY_ACTIVE_BIRTH_CANDIDATE')
        void_state, events = advance_site(state.void_state, site.site_id, dt, rates=rates)
        state = equilibrate_fixed_load_with_production_fem(replace(state, void_state=void_state))
        rows.append({**observables(state, label), "local_tensor_Pa": tensor.tolist(), "rates": rates,
                     "vacancy_rate_audit": rate_audit, "events": events})
        capture(label)
    tensor = local_site_tensor(state)
    if vacancy_coupled:
        rates, rate_audit = vacancy_coupled_void_rates(
            state, cfg, temperature_K=900.0, stress_tensor_Pa=tensor,
            center_m=state.void_state.sites[0].center_m,
            equivalent_plastic_strain_rate_gp=accepted_plastic_rate,
        )
    else:
        rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
        rate_audit = {}
    site = state.void_state.sites[0]
    dt = site.stabilization.crossing_time_exact(rates["stabilization_s"])
    if dt is None: raise RuntimeError('NO_KINETICALLY_ACTIVE_STABILIZATION_CANDIDATE')
    void_state, events = advance_site(state.void_state, site.site_id, dt, rates=rates)
    state = equilibrate_fixed_load_with_production_fem(replace(state, void_state=void_state))
    rows.append({**observables(state, "stabilization"), "local_tensor_Pa": tensor.tolist(), "rates": rates,
                 "vacancy_rate_audit": rate_audit, "events": events})
    capture("stabilization")
    if vacancy_coupled:
        state, debit_audit = create_subgrid_cavity_from_vacancies(
            state, "site-1", subgrid_radius_m, process_zone_length_m=5.0e-5,
        )
    else:
        state = replace(state, void_state=create_subgrid_cavity(
            state.void_state, "site-1", subgrid_radius_m,
        ))
        debit_audit = {}
    state = equilibrate_fixed_load_with_production_fem(state)
    rows.append({**observables(state, "subgrid_void"), "vacancy_debit_audit": debit_audit})
    capture("subgrid_void")
    tensor = local_site_tensor(state)
    cavity = state.void_state.cavities[0]
    if vacancy_coupled:
        rates, rate_audit = vacancy_coupled_void_rates(
            state, cfg, temperature_K=900.0, stress_tensor_Pa=tensor,
            center_m=cavity.center_m, equivalent_plastic_strain_rate_gp=accepted_plastic_rate,
            radius_m=cavity.radius_m,
        )
        promotion_target_m = cfg.promotion_radius_m * (1.0 + 1.0e-12)
        growth_dt = (promotion_target_m - cavity.radius_m) / rates["Rdot_m_s"] if rates["Rdot_m_s"] > 0.0 else None
    else:
        rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
        rate_audit = {}
        growth_dt = growth_time_to_radius_exact(
            state.void_state, cavity.cavity_id, promotion_radius_m, rates=rates,
            radial_growth_scale_m=cfg.radial_growth_scale_m,
        )
    if growth_dt is None:raise RuntimeError('NO_KINETICALLY_ACTIVE_GROWTH_CHANNEL')
    if vacancy_coupled:
        state, growth_audit = grow_cavity_from_vacancies(
            state, cavity.cavity_id, rates=rates, dt_s=growth_dt,
            process_zone_length_m=5.0e-5,
        )
    else:
        state = replace(state, void_state=update_cavity_growth(
            state.void_state, cavity.cavity_id, rates=rates, dt_s=growth_dt,
            radial_growth_scale_m=cfg.radial_growth_scale_m,
        ))
        growth_audit = {}
    grown = state.void_state.cavities[0]
    state = equilibrate_fixed_load_with_production_fem(state)
    rows.append({**observables(state, "subgrid_growth"), "rates": rates, "growth_dt_s": float(growth_dt),
                 "vacancy_rate_audit": rate_audit, "growth_audit": growth_audit})
    capture("subgrid_growth")
    promoted = promote_cavity(state.void_state, grown.cavity_id, cfg.promotion_radius_m)
    operations = []
    state = remesh_cavity(state, hole, promoted, "promotion", operations)
    rows.append({**observables(state, "geometric_promotion"), "executed_operations": operations})
    capture("geometric_promotion")
    if vacancy_coupled:
        state, accepted_plastic_rate, resolved_plastic_vacancy_audit = evolve_accepted_plastic_vacancy_interval(
            state, bundle=bundle, temperature_K=900.0, duration_s=0.1,
            test_only=True,
        )
        rows.append({**observables(state, "resolved_plastic_vacancy_production"),
                     "plastic_vacancy_audit": resolved_plastic_vacancy_audit})
        capture("resolved_plastic_vacancy_production")
    grown_hole = _grow_hole_boundary(hole, resolved_radius_m,
                                    crack_path_m=state.crack_network.branch(ROOT_BRANCH_ID).path)
    tensor = cavity_boundary_tensor(state)[0]
    cavity = state.void_state.cavities[0]
    if vacancy_coupled:
        rates, rate_audit = vacancy_coupled_void_rates(
            state, cfg, temperature_K=900.0, stress_tensor_Pa=tensor,
            center_m=cavity.center_m, equivalent_plastic_strain_rate_gp=accepted_plastic_rate,
            radius_m=cavity.radius_m,
        )
        growth_dt = (resolved_radius_m - cavity.radius_m) / rates["Rdot_m_s"] if rates["Rdot_m_s"] > 0.0 else None
    else:
        rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
        rate_audit = {}
        growth_dt = growth_time_to_radius_exact(
            state.void_state, cavity.cavity_id, resolved_radius_m, rates=rates,
            radial_growth_scale_m=cfg.radial_growth_scale_m,
        )
    if growth_dt is None:raise RuntimeError('NO_KINETICALLY_ACTIVE_GROWTH_CHANNEL')
    if vacancy_coupled:
        state, growth_audit = grow_cavity_from_vacancies(
            state, cavity.cavity_id, rates=rates, dt_s=growth_dt,
            process_zone_length_m=5.0e-5,
        )
    else:
        state = replace(state, void_state=update_cavity_growth(
            state.void_state, cavity.cavity_id, rates=rates, dt_s=growth_dt,
            radial_growth_scale_m=cfg.radial_growth_scale_m,
        ))
        growth_audit = {}
    state = remesh_cavity(state, grown_hole, state.void_state, "resolved-growth")
    rows.append({**observables(state, "resolved_growth"), "rates": rates, "growth_dt_s": float(growth_dt),
                 "vacancy_rate_audit": rate_audit, "growth_audit": growth_audit})
    capture("resolved_growth")
    if stop_before_ligament:
        return state, rows
    state, result = ligament_transaction(state)
    rows.append({
        **observables(state, "ligament_rupture"),
        "energy_release_J_per_m": result.energy_release_J_per_m,
        "hazard_dissipation_J_per_m": result.hazard_dissipation_J_per_m,
        "energy_margin_J_per_m": result.energy_margin_J_per_m,
        "event_classification": "physical_cleavage",
    })
    capture("ligament_rupture")
    if common_restart_protocol:
        from .closure_lifecycle_evidence import prepare_common_restart_reload
        load_operations=[]
        state=prepare_common_restart_reload(state,load_operations,state_trace,
            protocol_version=2 if common_restart_protocol=='v2' else 1)
        rows.append({**observables(state,'common_restart_reload'), 'executed_load_operations':load_operations})
    if qualify_source:
        state, source_audit = refine_downstream_source(state, max_refinement_levels=2 if common_restart_protocol in (True,'v1') else 1,
            refinement_region='complete_cavity_ring', quality_improvement='constrained_v1')
        rows.append({**observables(state, 'source_resolution_attempt'), 'source_resolution_audit':source_audit})
        capture('source_resolution_attempt')
    rows.append(observables(state, "connected_topology"))
    source_recovery = cavity_fixed_physical_arc_patch_recovery_v3(state)
    tensor = np.asarray(source_recovery["tensor_Pa"], dtype=float)
    boundary_elements = tuple(map(int, source_recovery["stencil_element_ids"]))
    rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
    rows.append({**observables(state, "downstream_surface_probe"), "cavity_fixed_physical_arc_patch_recovery_v3_tensor_Pa": tensor.tolist(),
                 "cavity_boundary_element_ids": boundary_elements,
                 "cavity_source_recovery": source_recovery,
                 "void_birth_rate_s": rates["birth_s"],
                 "classification": "PRE_CLEAVAGE_SURFACE_PROBE"})
    state, result, operations, causal = downstream_front_transaction(state)
    if result is None:
        rows.append({**observables(state, "downstream_first_passage_unavailable"),
                     "executed_operations": operations, "causal_first_passage": causal})
        capture("downstream_first_passage_unavailable")
        return state, rows
    rows.append({**observables(state, "new_graph_front"), "executed_operations": operations,
                 "energy_release_J_per_m": result.energy_release_J_per_m,
                 "causal_first_passage": causal})
    capture("new_graph_front")
    state, result, operations, causal = downstream_front_transaction(state, continuation=True)
    rows.append({**observables(state, "continued_accepted_event"), "executed_operations": operations,
                 "energy_release_J_per_m": result.energy_release_J_per_m,
                 "causal_first_passage": causal})
    capture("continued_accepted_event")
    return state, rows


def natural_trajectory(*, bundle: UnifiedFractureMaterialBundle, seed=3621, steps=6):
    state, _ = build_production_void_state(bundle=bundle, enabled=True, stochastic=True, seed=seed)
    cfg = VoidingConfig(enabled=True)
    rows = []
    for step in range(steps):
        tensor = local_site_tensor(state); rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
        state = equilibrate_fixed_load_with_production_fem(state)
        void_state, events = advance_site(state.void_state, "site-1", 1.0e-12, rates=rates)
        state = replace(state, void_state=void_state)
        rows.append({**observables(state, f"natural_step_{step+1}"), "local_tensor_Pa": tensor.tolist(), "rates": rates, "events": events})
    return state, rows
