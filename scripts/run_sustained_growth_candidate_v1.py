#!/usr/bin/env python3
"""Run the single prospective sustained-growth candidate to its first promotion.

The accepted global scheduler remains the time owner.  This wrapper only
chooses a mechanics macro step with a full-step/two-half-step local error test;
all event roots and losing-clock advances are performed by the production
scheduler.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.bulk_plasticity_v104 import PlasticityScope, accepted_bulk_load_step
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.dislocation_interaction_voiding_v1 import accepted_interaction_opportunity
from arrhenius_fracture.embryo_growth_v1 import (
    advance_cavity_emission, cavity_surface_emission_rates,
    source_local_fem_tensors,
)
from arrhenius_fracture.global_physical_scheduler_v1 import (
    AcceptedIntervalContinuationV1, advance_global_physical_interval,
)
from arrhenius_fracture.topology_transaction_v11 import (
    EquilibriumObservablesUnavailable, complete_accepted_state_fingerprint,
)
from arrhenius_fracture.vacancy_coupling_v1 import equivalent_plastic_strain_rate_s
from arrhenius_fracture.voiding_production_v5 import (
    _geometry, _grow_hole_boundary, promote_cavity, remesh_cavity,
)
from scripts.run_active_two_mechanism_natural_void_study import (
    canonical_drive_check, site_source_exposure,
)
from scripts.run_natural_void_hypothesis_v1 import (
    accepted_time, atomic_dump, phase_summary, pickle_safe_metadata,
)


TEMPERATURE_K = 1200.0
TOTAL_TIME_S = 10000.0
PROMOTION_RADIUS_M = 8.0e-6
ROW_ID = "hypothesis.sustained-growth-v1:krec=0.01:depletion=0.01"
OUT_DEFAULT = ROOT / "runs/managed_sustained_growth_v1_20260920/output"
DT_LEVELS = (0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
             10.0, 20.0, 50.0, 100.0)
# The established mechanics-transfer requirements are 2% for energy and
# compliance.  This driver uses a tighter 0.5% local gate for every accepted
# full-step/two-half-step comparison.
LOCAL_ERROR_LIMIT = 5.0e-3
RADIUS_MILESTONES_M = (0.1e-6, 0.5e-6, 1.0e-6, 2.0e-6, 4.0e-6, 8.0e-6)


def _observables(state) -> dict[str, float]:
    eq = state.equilibrium_observables
    if eq is None:
        raise RuntimeError("accepted mechanics has no equilibrium observables")
    return {
        "reaction_N_per_m": float(eq.mean_reaction_magnitude_N_per_m),
        "compliance_m2_per_N": float(eq.compliance_m2_per_N),
        "recoverable_energy_J_per_m": float(eq.recoverable_elastic_energy_J_per_m),
        "total_internal_energy_J_per_m": float(eq.total_stored_internal_energy_J_per_m),
    }


def _field_error_pair(full, refined, base, name: str) -> tuple[float, float]:
    a = np.asarray(getattr(full, name), dtype=float)
    b = np.asarray(getattr(refined, name), dtype=float)
    c = np.asarray(getattr(base, name), dtype=float)
    difference = float(np.linalg.norm(a-b))
    increment_scale = max(
        float(np.linalg.norm(b-c)), 1.0e-14*max(float(np.linalg.norm(b)), 1.0))
    endpoint_scale = max(float(np.linalg.norm(b)), 1.0e-30)
    return difference/increment_scale, difference/endpoint_scale


def _mechanics_pair(state, duration_s: float):
    pair_wall_start = time.perf_counter()
    kwargs = dict(
        scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
        load_factor=1.0, temperature_K=TEMPERATURE_K,
        maximum_stagger_iterations=80,
    )
    full, full_audit = accepted_bulk_load_step(state, duration_s=duration_s, **kwargs)
    first, first_audit = accepted_bulk_load_step(state, duration_s=0.5*duration_s, **kwargs)
    refined, second_audit = accepted_bulk_load_step(
        first, duration_s=0.5*duration_s, **kwargs)
    ep_increment, ep_endpoint = _field_error_pair(full, refined, state, "ep_gp")
    rho_increment, rho_endpoint = _field_error_pair(full, refined, state, "rho_gp")
    field_errors = {
        "plastic_strain_increment_relative_l2_diagnostic": ep_increment,
        "forest_density_increment_relative_l2_diagnostic": rho_increment,
        "plastic_strain_endpoint_relative_l2_diagnostic": ep_endpoint,
        "forest_density_endpoint_relative_l2_diagnostic": rho_endpoint,
    }
    full_obs = _observables(full)
    refined_obs = _observables(refined)
    observable_errors = {
        key+"_relative": abs(full_obs[key]-value)/max(abs(value), 1.0e-30)
        for key, value in refined_obs.items()
    }
    directional_wall_start = time.perf_counter()
    full_tensor, full_K, _, _ = canonical_drive_check(full)
    refined_tensor, refined_K, refined_canonical, refined_load_rows = (
        canonical_drive_check(refined))
    directional_wall_s = time.perf_counter()-directional_wall_start
    drive_errors = {
        "canonical_source_tensor_relative_l2": float(np.linalg.norm(
            np.asarray(full_tensor)-np.asarray(refined_tensor))/max(
                float(np.linalg.norm(refined_tensor)), 1.0e-30)),
        "canonical_K_relative": abs(float(full_K)-float(refined_K))/max(
            abs(float(refined_K)), 1.0e-30),
    }
    source_wall_start = time.perf_counter()
    full_exposure, full_source = site_source_exposure(state, full, duration_s)
    refined_exposure, refined_source = site_source_exposure(state, refined, duration_s)
    source_observation_wall_s = time.perf_counter()-source_wall_start
    source_functionals = {
        "integrated_plastic_vacancy_source": {
            "full_fraction_area": float(
                full_source["integrated_source_fraction_area_per_s"]*duration_s),
            "two_half_fraction_area": float(
                refined_source["integrated_source_fraction_area_per_s"]*duration_s),
        },
        "site_plastic_vacancy_exposure": {},
        "direct_interaction": {},
        "cavity_emission": {},
    }
    source_errors = []
    for site_id, refined_value in refined_exposure.items():
        full_value = float(full_exposure[site_id]); refined_value = float(refined_value)
        absolute = abs(full_value-refined_value)
        relative = absolute/max(abs(refined_value), 1.0e-22)
        production = next(item for item in state.void_state.sites
                          if item.site_id == site_id)
        # A raw production increment is an active event-control error only
        # while this site's vacancy-birth owner remains eligible.  Once the
        # site owns a stable cavity, its effect is certified by the coupled
        # transport inventory/ledger replay; the raw near-zero increment is
        # retained here but does not masquerade as a missed-event error.
        controls_active_birth = production.phase.value == "AVAILABLE_SITE"
        if controls_active_birth:
            source_errors.append(relative)
        source_functionals["site_plastic_vacancy_exposure"][site_id] = {
            "full_fraction_area": full_value,
            "two_half_fraction_area": refined_value,
            "absolute_error_fraction_area": absolute,
            "mixed_relative_error": relative,
            "absolute_scale_fraction_area": 1.0e-22,
            "controls_active_birth_action": controls_active_birth,
            "acceptance_role": ("ACTIVE_EVENT_CONTROL" if controls_active_birth
                                else "DIAGNOSTIC_COUPLED_TRANSPORT_REPLAY_CONTROLS"),
        }
    for site in state.void_state.dislocation_interaction_state.sites:
        production = next(item for item in state.void_state.sites
                          if item.site_id == site.site_id)
        remaining = max(site.next_threshold()-site.interaction_action, 0.0)
        if production.phase.value != "AVAILABLE_SITE" or not math.isfinite(remaining):
            source_functionals["direct_interaction"][site.site_id] = {
                "status": "INACTIVE_LIFECYCLE_OWNER", "remaining_action": remaining}
            continue
        values = []
        for endpoint in (full, refined):
            epdot = equivalent_plastic_strain_rate_s(
                state.ep_gp, endpoint.ep_gp, duration_s)
            values.append(dict(accepted_interaction_opportunity(
                endpoint, site, temperature_K=TEMPERATURE_K,
                accepted_equivalent_plastic_strain_rate_s=epdot)))
        full_action = values[0]["successful_direct_interaction_rate_s"]*duration_s
        refined_action = values[1]["successful_direct_interaction_rate_s"]*duration_s
        absolute = abs(full_action-refined_action)
        relative = absolute/max(remaining, abs(refined_action), 1.0e-12)
        source_errors.append(relative)
        source_functionals["direct_interaction"][site.site_id] = {
            "status": "ACTIVE", "remaining_action": remaining,
            "full_action_increment": full_action,
            "two_half_action_increment": refined_action,
            "absolute_error": absolute,
            "error_relative_to_remaining_or_increment": relative,
            "absolute_action_scale": 1.0e-12,
        }
    for growth in state.void_state.embryo_growth_states:
        predictions = []
        for endpoint in (full, refined):
            tensors, zz = source_local_fem_tensors(endpoint, growth)
            rates, _ = cavity_surface_emission_rates(
                growth, stress_tensor_Pa=tensors, sigma_zz_Pa=zz,
                temperature_K=TEMPERATURE_K)
            emission, emission_audit = advance_cavity_emission(
                growth.cavity_emission_state, radius_m=growth.radius_m,
                rates_s=rates, duration_s=duration_s)
            count = (emission.cumulative_loop_count_effective
                     -growth.cavity_emission_state.cumulative_loop_count_effective)
            area = growth.area_m2+float(emission_audit["area_increment_m2"])
            predictions.append({
                "effective_loop_increment": count,
                "source_availability": [item.available_fraction
                                         for item in emission.sources],
                "radius_m": math.sqrt(max(area, 0.0)/math.pi),
            })
        full_prediction, refined_prediction = predictions
        loop_error = abs(full_prediction["effective_loop_increment"]
                         -refined_prediction["effective_loop_increment"])/max(
                             abs(refined_prediction["effective_loop_increment"]), 1.0e-8)
        availability_error = max(
            (abs(a-b)/max(abs(b), 1.0e-10)
             for a, b in zip(full_prediction["source_availability"],
                             refined_prediction["source_availability"])), default=0.0)
        radius_error = abs(full_prediction["radius_m"]-refined_prediction["radius_m"])/max(
            abs(refined_prediction["radius_m"]-growth.radius_m), 1.0e-12)
        source_errors.extend((loop_error, availability_error, radius_error))
        source_functionals["cavity_emission"][growth.site_id] = {
            "full": full_prediction, "two_half": refined_prediction,
            "effective_loop_increment_relative_error": loop_error,
            "source_availability_mixed_relative_error": availability_error,
            "radius_increment_mixed_relative_error": radius_error,
            "loop_absolute_scale": 1.0e-8,
            "availability_absolute_scale": 1.0e-10,
            "radius_increment_absolute_scale_m": 1.0e-12,
            "canonical_random_measure_reused": True,
        }
    error = max(*observable_errors.values(), *drive_errors.values(), *source_errors)
    audit = {
        "scheme": "FULL_STEP_VS_TWO_HALF_STEPS_ACCEPT_HALF_PATH",
        "duration_s": duration_s,
        "field_errors": field_errors,
        "observable_errors": observable_errors,
        "consumed_drive_errors": drive_errors,
        "consumed_source_functionals": source_functionals,
        "maximum_consumed_source_functional_error": max(source_errors, default=0.0),
        "maximum_acceptance_relative_error": error,
        "maximum_relative_error": error,
        "acceptance_error_definition": (
            "REACTION_COMPLIANCE_ENERGIES_CANONICAL_K_AND_SOURCE_TENSOR; "
            "FIELD_ENDPOINT_AND_INCREMENT_ERRORS_RETAINED_AS_DIAGNOSTICS"),
        "limit": LOCAL_ERROR_LIMIT,
        "full_step_plastic_work_J_per_m": full_audit[
            "accepted_endpoint_path_plastic_work_J_per_m"],
        "accepted_half_path_plastic_work_J_per_m": (
            first_audit["accepted_endpoint_path_plastic_work_J_per_m"]
            + second_audit["accepted_endpoint_path_plastic_work_J_per_m"]),
        "full_observables": full_obs,
        "accepted_observables": refined_obs,
        "wall_profile_s": {
            "constitutive_evolution": sum(
                item["wall_profile_s"]["constitutive_evolution"]
                for item in (full_audit, first_audit, second_audit)),
            "equilibrium": sum(item["wall_profile_s"]["equilibrium"]
                               for item in (full_audit, first_audit, second_audit)),
            "constitutive_observation_assembly": sum(
                item["wall_profile_s"]["constitutive_observation_assembly"]
                for item in (full_audit, first_audit, second_audit)),
            "directional_observation": directional_wall_s,
            "source_observation": source_observation_wall_s,
            "mechanics_pair_total": time.perf_counter()-pair_wall_start,
        },
    }
    dependency_valid_observations = {
        "accepted_state_fingerprint": complete_accepted_state_fingerprint(refined),
        "site_source_exposure_fraction_area": refined_exposure,
        "integrated_plastic_source_fraction_area": float(
            refined_source["integrated_source_fraction_area_per_s"]*duration_s),
        "canonical_source_tensor_Pa": refined_tensor,
        "canonical_K_effective_Pa_sqrt_m": refined_K,
        "canonical_drive": refined_canonical,
        "directional_load_rows": refined_load_rows,
    }
    return refined, audit, dependency_valid_observations


def _growth_radius(state, site_id: str = "site-1") -> float | None:
    growth = next((item for item in state.void_state.embryo_growth_states
                   if item.site_id == site_id), None)
    return None if growth is None else float(growth.radius_m)


def _promotion_commit(live, preview):
    """Execute the ordinary 8-um subgrid-to-resolved topology handoff."""
    cavity_id = preview.owner_id
    cavity = next(item for item in live.void_state.cavities
                  if item.cavity_id == cavity_id)
    promoted = promote_cavity(live.void_state, cavity_id, PROMOTION_RADIUS_M)
    resolution = live.junction_process_state.get("production_mesh_resolution", (32, 12))
    width = float(np.ptp(np.asarray(live.mesh.nodes)[:, 0]))
    height = float(np.ptp(np.asarray(live.mesh.nodes)[:, 1]))
    hole, _ = _geometry(
        radius_m=PROMOTION_RADIUS_M, center_m=cavity.center_m,
        boundary_segments=int(resolution[0]), radial_layers=int(resolution[1]),
        specimen_width_m=width, specimen_height_m=height, far_h_m=5.0e-5,
    )
    hole = _grow_hole_boundary(
        hole, PROMOTION_RADIUS_M,
        crack_path_m=live.crack_network.branch(ROOT_BRANCH_ID).path)
    return remesh_cavity(live, hole, promoted, "sustained-growth-v1-promotion")


def _next_dt(current: float, error: float, state, selected: tuple,
             *, ceiling: float | None, allow_reprobe: bool) -> float:
    # Event-rich low-count startup stays resolved at the legacy 0.1-s scale.
    growth = tuple(state.void_state.embryo_growth_states)
    depleted = bool(growth) and max(
        source.available_fraction
        for item in growth for source in item.cavity_emission_state.sources
    ) < 0.05
    if selected or not depleted:
        target = min(current, 0.1)
    elif error < 0.70*LOCAL_ERROR_LIMIT:
        target = next((value for value in DT_LEVELS if value > current+1.0e-15), current)
    else:
        target = current
    if ceiling is not None and not allow_reprobe:
        target = min(target, ceiling)
    return float(target)


def _checkpoint(out: Path, state, continuation, progress, name: str) -> None:
    path = out/name
    checkpoint_wall_start = time.perf_counter()
    manifest = write_checkpoint(
        state, path, compression="gzip", provider_runtime=continuation)
    write_wall_s = time.perf_counter()-checkpoint_wall_start
    progress["latest_restart"] = path.name
    progress["latest_restart_state_file"] = manifest["state_file"]
    progress["accepted_time_s"] = accepted_time(state)
    restore_wall_start = time.perf_counter()
    restored, restored_continuation = restore_checkpoint(
        path, with_provider_runtime=True)
    restore_wall_s = time.perf_counter()-restore_wall_start
    progress["latest_restart_direct_restored_equal"] = (
        complete_accepted_state_fingerprint(state)
        == complete_accepted_state_fingerprint(restored)
        and ((continuation is None and restored_continuation is None)
             or (continuation is not None and restored_continuation is not None
                 and continuation.audit_payload()
                 == restored_continuation.audit_payload())))
    if not progress["latest_restart_direct_restored_equal"]:
        raise RuntimeError("checkpoint direct/restored equality failed")
    progress["latest_restart_accepted_state_fingerprint"] = (
        complete_accepted_state_fingerprint(state))
    progress["last_checkpoint_wall_profile_s"] = {
        "write": write_wall_s,
        "restore_and_verify": restore_wall_s,
        "checkpoint_total_before_progress_write": time.perf_counter()-checkpoint_wall_start,
    }
    atomic_dump(out/"progress.json", progress)


def main() -> None:
    out = Path(os.environ.get("SUSTAINED_GROWTH_RUN_OUT", OUT_DEFAULT)).resolve()
    progress_path = out/"progress.json"
    if not progress_path.exists():
        raise RuntimeError("run prepare_sustained_growth_candidate_v1.py first")
    progress = json.loads(progress_path.read_text())
    state, continuation = restore_checkpoint(
        out/progress["latest_restart"], with_provider_runtime=True)
    if continuation is not None and not isinstance(continuation, AcceptedIntervalContinuationV1):
        raise RuntimeError("unknown in-flight scheduler continuation")
    row = state.void_state.embryo_growth_configuration.cavity_emission_parameter_row
    if (row.row_id != ROW_ID or row.recovery_rate_s != 0.01
            or row.depletion_per_loop != 0.01):
        raise RuntimeError("candidate parameter identity changed")
    history = [item for item in progress.get("history", [])
               if float(item["end_time_s"]) <= accepted_time(state)+1.0e-12]
    dt_next = float(progress.get("adaptive_step_s", 0.1))
    adaptive_ceiling = progress.get("adaptive_ceiling_step_s")
    adaptive_ceiling = None if adaptive_ceiling is None else float(adaptive_ceiling)
    reprobe_after = float(progress.get("adaptive_reprobe_after_time_s", -math.inf))
    if (os.environ.get("RESET_OBSERVATION_ONLY_ADAPTIVITY") == "1"
            and progress.get("stop_reason")
            == "MINIMUM_STEP_EQUILIBRIUM_CERTIFICATE_UNAVAILABLE"):
        previous_adaptivity = {
            "adaptive_step_s": dt_next,
            "adaptive_ceiling_step_s": adaptive_ceiling,
            "adaptive_reprobe_after_time_s": reprobe_after,
        }
        dt_next = 0.5
        adaptive_ceiling = None
        reprobe_after = -math.inf
        progress["equilibrium_accounting_adaptivity_reset"] = {
            "reason": "OBSERVATION_ONLY_TOP_BOTTOM_REACTION_GATE_REPLACED_BY_COMPLETE_SUPPORT_ACCOUNTING",
            "accepted_state_unchanged": True,
            "previous": previous_adaptivity,
            "new_adaptive_step_s": dt_next,
        }
        progress["adaptive_step_s"] = dt_next
        progress["adaptive_ceiling_step_s"] = adaptive_ceiling
        progress["adaptive_reprobe_after_time_s"] = reprobe_after
        atomic_dump(progress_path, progress)
    reached = set(progress.get("radius_milestones_reached_m", []))
    wall_limit = float(os.environ.get("MAX_WALL_SECONDS", "1800"))
    wall_start = time.monotonic()
    stop_reason = None
    atomic_dump(out/"active_worker.json", {
        "pid": os.getpid(), "started_wall_time_s": time.time(),
        "wall_time_chunk_limit_s": wall_limit,
        "accepted_time_s": accepted_time(state),
    })

    while accepted_time(state) < TOTAL_TIME_S-1.0e-12:
        if time.monotonic()-wall_start >= wall_limit:
            stop_reason = "WALL_TIME_CHUNK_LIMIT"
            break
        now = accepted_time(state)
        if continuation is None:
            dt = min(dt_next, TOTAL_TIME_S-now)
            previous = state
            try:
                mechanical, local, cached_observations = _mechanics_pair(previous, dt)
            except EquilibriumObservablesUnavailable as exc:
                if dt <= DT_LEVELS[0]+1e-15:
                    stop_reason = "MINIMUM_STEP_EQUILIBRIUM_CERTIFICATE_UNAVAILABLE"
                    progress["numerical_limitation"] = {
                        "start_time_s": now, "duration_s": dt,
                        "exception_type": type(exc).__name__, "message": str(exc)}
                    break
                smaller = max(value for value in DT_LEVELS if value < dt-1e-15)
                adaptive_ceiling = smaller
                reprobe_after = now+max(5.0, 10.0*smaller)
                progress["last_rejected_step"] = {
                    "start_time_s": now, "duration_s": dt,
                    "reason": "EQUILIBRIUM_CERTIFICATE_UNAVAILABLE",
                    "exception_type": type(exc).__name__, "message": str(exc),
                    "replacement_duration_s": smaller,
                    "accepted_state_unchanged": True,
                }
                dt_next = smaller
                progress.update({
                    "adaptive_step_s": dt_next,
                    "adaptive_ceiling_step_s": adaptive_ceiling,
                    "adaptive_reprobe_after_time_s": reprobe_after,
                })
                atomic_dump(progress_path, progress)
                print("SUSTAINED_GROWTH_STEP_REJECT_CERTIFICATE", now, dt,
                      smaller, str(exc), flush=True)
                continue
            if local["maximum_relative_error"] > LOCAL_ERROR_LIMIT and dt > DT_LEVELS[0]+1e-15:
                smaller = max(value for value in DT_LEVELS if value < dt-1e-15)
                adaptive_ceiling = smaller
                reprobe_after = now+max(5.0, 10.0*smaller)
                progress["last_rejected_step"] = {
                    "start_time_s": now, "duration_s": dt,
                    "error": local, "replacement_duration_s": smaller,
                }
                dt_next = smaller
                progress["adaptive_step_s"] = dt_next
                progress["adaptive_ceiling_step_s"] = adaptive_ceiling
                progress["adaptive_reprobe_after_time_s"] = reprobe_after
                atomic_dump(progress_path, progress)
                print("SUSTAINED_GROWTH_STEP_REJECT", now, dt, smaller,
                      local["maximum_relative_error"], flush=True)
                continue
            epdot = equivalent_plastic_strain_rate_s(previous.ep_gp, mechanical.ep_gp, dt)
            if (cached_observations["accepted_state_fingerprint"]
                    != complete_accepted_state_fingerprint(mechanical)):
                raise RuntimeError("accepted mechanics observation cache dependency changed")
            exposure = cached_observations["site_source_exposure_fraction_area"]
            tensor = cached_observations["canonical_source_tensor_Pa"]
            K = cached_observations["canonical_K_effective_Pa_sqrt_m"]
            canonical = cached_observations["canonical_drive"]
            load_rows = cached_observations["directional_load_rows"]
            direct = [accepted_interaction_opportunity(
                mechanical, site, temperature_K=TEMPERATURE_K,
                accepted_equivalent_plastic_strain_rate_s=epdot)
                for site in mechanical.void_state.dislocation_interaction_state.sites]
            driver_context = {
                "adaptive_local_error": local,
                "site_source_exposure_fraction_area": exposure,
                "integrated_plastic_source_fraction_area": cached_observations[
                    "integrated_plastic_source_fraction_area"],
                "canonical_drive": canonical, "direct_opportunity": direct,
                "wall_profile_s": dict(local["wall_profile_s"]),
                "accepted_observation_cache": (
                    "REUSED_ONLY_FOR_IDENTICAL_ACCEPTED_STATE_FINGERPRINT"),
            }
            load_history = {
                "kind": "hold", "temperature_K": TEMPERATURE_K,
                "stress_tensor_Pa": tensor,
                "K_effective_by_front": {ROOT_BRANCH_ID: K},
                "directional_load_rows_by_front": {ROOT_BRANCH_ID: load_rows},
                "promotion_radius_m": PROMOTION_RADIUS_M,
                "maximum_events": int(os.environ.get(
                    "MAX_SCHEDULER_EVENTS_PER_INTERVAL", "4096")),
                "maximum_transactions_per_call": int(os.environ.get(
                    "MAX_SCHEDULER_TRANSACTIONS_PER_CHECKPOINT", "32")),
            }
            macro_start = now
            selected_before = ()
            transactions_before = 0
            all_owners_before = True
        else:
            dt = continuation.remaining_duration_s
            mechanical = continuation.mechanical_endpoint_state
            load_history = dict(continuation.load_history)
            # Checkpoint batching is numerical I/O policy, not physical state;
            # it may be tightened or relaxed safely between resumable calls.
            load_history["maximum_transactions_per_call"] = int(os.environ.get(
                "MAX_SCHEDULER_TRANSACTIONS_PER_CHECKPOINT",
                str(load_history["maximum_transactions_per_call"])))
            driver_context = dict(continuation.driver_context)
            macro_start = continuation.macro_start_time_s
            selected_before = continuation.selected_events
            transactions_before = continuation.completed_transaction_count
            all_owners_before = continuation.all_owners_previewed

        call_history = dict(load_history)
        call_history["commit_growth_promotion"] = _promotion_commit
        remaining_budget = int(load_history["maximum_events"])-transactions_before
        if remaining_budget <= 0:
            raise RuntimeError("scheduler exhausted the macro transaction budget")
        call_history["maximum_events"] = remaining_budget
        scheduler_wall_start = time.perf_counter()
        advanced, scheduler = advance_global_physical_interval(
            state, mechanical, dt, call_history)
        scheduler_call_wall_s = time.perf_counter()-scheduler_wall_start
        driver_context["wall_profile_s"] = {
            **dict(driver_context.get("wall_profile_s", {})),
            "transport": float(scheduler["wall_profile_s"]["transport"]),
            "scheduler_total": float(scheduler["wall_profile_s"]["scheduler_total"]),
            "scheduler_call": scheduler_call_wall_s,
        }
        selected_chunk = tuple(item["selected_event"] for item in scheduler["events"]
                               if item["selected_event"] is not None)
        selected = selected_before+selected_chunk
        transactions = transactions_before+len(scheduler["events"])
        all_owners = all_owners_before and scheduler["all_owners_previewed"]
        remaining = max(dt-scheduler["advanced_duration_s"], 0.0)
        requested = continuation.requested_duration_s if continuation is not None else dt
        tolerance = max(1e-12, 1e-8*requested)
        if remaining > tolerance:
            continuation = AcceptedIntervalContinuationV1(
                mechanical_endpoint_state=mechanical,
                remaining_duration_s=remaining, requested_duration_s=requested,
                macro_start_time_s=macro_start,
                load_history=pickle_safe_metadata(load_history),
                driver_context=pickle_safe_metadata(driver_context),
                selected_events=selected,
                completed_transaction_count=transactions,
                all_owners_previewed=all_owners)
            state = advanced
            progress.update({
                "history": history, "adaptive_step_s": dt_next,
                "inflight_interval": dict(continuation.audit_payload()),
                "phase": phase_summary(state), "stop_reason": None,
            })
            _checkpoint(out, state, continuation, progress, "inflight_restart.json")
            print("SUSTAINED_GROWTH_INFLIGHT", accepted_time(state), transactions,
                  remaining, flush=True)
            continue

        continuation = None
        before_radius = _growth_radius(state)
        state = advanced
        after_radius = _growth_radius(state)
        newly_reached = []
        for milestone in RADIUS_MILESTONES_M:
            if milestone not in reached and after_radius is not None and after_radius >= milestone:
                reached.add(milestone); newly_reached.append(milestone)
        history.append({
            **driver_context, "start_time_s": macro_start,
            "end_time_s": accepted_time(state),
            "accepted_duration_s": accepted_time(state)-macro_start,
            "selected_events": selected,
            "scheduler_transaction_count": transactions,
            "all_owners_previewed": all_owners,
            "radius_before_m": before_radius, "radius_after_m": after_radius,
            "new_radius_milestones_m": newly_reached,
            "phase": phase_summary(state),
            "accepted_state_fingerprint": complete_accepted_state_fingerprint(state),
        })
        allow_reprobe = accepted_time(state) >= reprobe_after-1e-12
        dt_next = _next_dt(
            float(driver_context["adaptive_local_error"]["duration_s"]),
            float(driver_context["adaptive_local_error"]["maximum_relative_error"]),
            state, selected, ceiling=adaptive_ceiling,
            allow_reprobe=allow_reprobe)
        if allow_reprobe and adaptive_ceiling is not None and dt_next > adaptive_ceiling:
            adaptive_ceiling = None
        progress.update({
            "history": history, "adaptive_step_s": dt_next,
            "adaptive_ceiling_step_s": adaptive_ceiling,
            "adaptive_reprobe_after_time_s": reprobe_after,
            "radius_milestones_reached_m": sorted(reached),
            "phase": phase_summary(state), "last_accepted_events": selected,
            "stop_reason": None,
        })
        label = int(round(1000.0*accepted_time(state)))
        checkpoint_name = f"restart_{label:09d}ms.json"
        _checkpoint(out, state, None, progress, checkpoint_name)
        counts = {name: selected.count(name) for name in sorted(set(selected))}
        print("SUSTAINED_GROWTH_PROGRESS", accepted_time(state), after_radius,
              dt_next, counts, flush=True)
        if "GROWTH_PROMOTION" in selected:
            stop_reason = "FIRST_8UM_PROMOTION_COMMITTED"
            break

    if stop_reason is None:
        stop_reason = ("PHYSICAL_HORIZON_REACHED" if accepted_time(state) >= TOTAL_TIME_S-1e-12
                       else "CHUNK_STOP")
    if continuation is None:
        label = int(round(1000.0*accepted_time(state)))
        name = f"restart_{label:09d}ms.json"
    else:
        name = "inflight_restart.json"
    progress.update({"history": history, "adaptive_step_s": dt_next,
                     "stop_reason": stop_reason, "phase": phase_summary(state)})
    _checkpoint(out, state, continuation, progress, name)
    atomic_dump(out/"result.json", {
        "schema": "unified-2d.sustained-growth-candidate-result/1",
        "classification": "PROSPECTIVE_UNCALIBRATED_PARAMETER_HYPOTHESIS",
        "outcome": stop_reason, "accepted_time_s": accepted_time(state),
        "latest_restart": progress["latest_restart"],
        "phase": phase_summary(state), "history": history,
    })
    atomic_dump(out/"active_worker.json", {
        "pid": os.getpid(), "worker_live": False,
        "stopped_wall_time_s": time.time(), "stop_reason": stop_reason,
        "accepted_time_s": accepted_time(state),
    })
    print("SUSTAINED_GROWTH_CANDIDATE_CHUNK", stop_reason,
          accepted_time(state), flush=True)


if __name__ == "__main__":
    main()
