#!/usr/bin/env python3
"""One bounded full-step/two-half-step coupled replay at the 7.8-s state."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.bulk_plasticity_v104 import PlasticityScope, accepted_bulk_load_step
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.dislocation_interaction_voiding_v1 import accepted_interaction_opportunity
from arrhenius_fracture.global_physical_scheduler_v1 import advance_global_physical_interval
from arrhenius_fracture.vacancy_coupling_v1 import equivalent_plastic_strain_rate_s
from scripts.run_active_two_mechanism_natural_void_study import (
    canonical_drive_check, site_source_exposure,
)
from scripts.run_natural_void_hypothesis_v1 import accepted_time, finite, phase_summary


CHECKPOINT = Path(os.environ.get(
    "QUALIFICATION_CHECKPOINT",
    ROOT/"runs/managed_sustained_growth_v1_20260920/output/restart_000007800ms.json"))
OUT = Path(os.environ.get(
    "QUALIFICATION_OUT",
    ROOT/"runs/managed_sustained_growth_v1_20260920/adaptive_functional_qualification.json"))
DT = float(os.environ.get("QUALIFICATION_DT_S", "0.5"))
TEMPERATURE_K = 1200.0


def mechanics(state, dt):
    return accepted_bulk_load_step(
        state, scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
        load_factor=1.0, duration_s=dt, temperature_K=TEMPERATURE_K,
        maximum_stagger_iterations=80)[0]


def history(previous, endpoint, dt):
    tensor, K, _, rows = canonical_drive_check(endpoint)
    epdot = equivalent_plastic_strain_rate_s(previous.ep_gp, endpoint.ep_gp, dt)
    direct = [accepted_interaction_opportunity(
        endpoint, site, temperature_K=TEMPERATURE_K,
        accepted_equivalent_plastic_strain_rate_s=epdot)
        for site in endpoint.void_state.dislocation_interaction_state.sites]
    return {
        "kind": "hold", "temperature_K": TEMPERATURE_K,
        "stress_tensor_Pa": tensor,
        "K_effective_by_front": {ROOT_BRANCH_ID: K},
        "directional_load_rows_by_front": {ROOT_BRANCH_ID: rows},
        "maximum_events": 256, "maximum_transactions_per_call": 256,
    }, direct


def coupled(state, endpoint, dt):
    load, direct = history(state, endpoint, dt)
    advanced, audit = advance_global_physical_interval(state, endpoint, dt, load)
    return advanced, audit, direct


def rel(a, b, floor):
    return abs(float(a)-float(b))/max(abs(float(b)), float(floor))


def main() -> None:
    initial = restore_checkpoint(CHECKPOINT)
    start_time = float(os.environ.get("QUALIFICATION_START_TIME_S", "7.79999"))
    if not math.isclose(accepted_time(initial), start_time, abs_tol=1e-12):
        raise RuntimeError("qualification checkpoint time changed")

    full_mechanics = mechanics(initial, DT)
    full, full_audit, full_direct = coupled(initial, full_mechanics, DT)
    first_mechanics = mechanics(initial, DT/2.0)
    first, first_audit, first_direct = coupled(initial, first_mechanics, DT/2.0)
    second_mechanics = mechanics(first, DT/2.0)
    halves, second_audit, second_direct = coupled(first, second_mechanics, DT/2.0)

    eq_full = full.equilibrium_observables
    eq_half = halves.equilibrium_observables
    mechanical = {
        "reaction_relative": rel(eq_full.mean_reaction_magnitude_N_per_m,
                                 eq_half.mean_reaction_magnitude_N_per_m, 1.0),
        "compliance_relative": rel(eq_full.compliance_m2_per_N,
                                   eq_half.compliance_m2_per_N, 1e-30),
        "recoverable_energy_relative": rel(eq_full.recoverable_elastic_energy_J_per_m,
                                           eq_half.recoverable_elastic_energy_J_per_m, 1e-30),
    }
    full_source, full_source_audit = site_source_exposure(initial, full_mechanics, DT)
    first_source, first_source_audit = site_source_exposure(initial, first_mechanics, DT/2)
    second_source, second_source_audit = site_source_exposure(first, second_mechanics, DT/2)
    source = {}
    for site_id in full_source:
        full_value = full_source[site_id]
        halves_value = first_source[site_id]+second_source[site_id]
        source[site_id] = {
            "full_fraction_area": full_value,
            "two_half_fraction_area": halves_value,
            "absolute_error_fraction_area": abs(full_value-halves_value),
            "mixed_relative_error": rel(full_value, halves_value, 1e-22),
            "absolute_scale_fraction_area": 1e-22,
        }

    tf = full.vacancy_transport_state_v2
    th = halves.vacancy_transport_state_v2
    inventory_full = float(np.dot(tf.transport_mesh.area_e, tf.vacancy_fraction_gp))
    inventory_half = float(np.dot(th.transport_mesh.area_e, th.vacancy_fraction_gp))
    ledger_errors = {}
    for key in sorted(set(tf.ledgers)&set(th.ledgers)):
        if isinstance(tf.ledgers[key], (int, float)) and isinstance(th.ledgers[key], (int, float)):
            ledger_errors[key] = {
                "full": tf.ledgers[key], "two_half": th.ledgers[key],
                "absolute_error": abs(tf.ledgers[key]-th.ledgers[key]),
                "mixed_relative_error": rel(tf.ledgers[key], th.ledgers[key], 1e-24),
            }

    growth = {}
    for gf in full.void_state.embryo_growth_states:
        gh = next(item for item in halves.void_state.embryo_growth_states
                  if item.site_id == gf.site_id)
        growth[gf.site_id] = {
            "radius": {"full_m": gf.radius_m, "two_half_m": gh.radius_m,
                       "relative_error": rel(gf.radius_m, gh.radius_m, 1e-12)},
            "effective_loop_count": {
                "full": gf.emitted_loop_count_effective,
                "two_half": gh.emitted_loop_count_effective,
                "increment_relative_error": rel(
                    gf.emitted_loop_count_effective-next(
                        item for item in initial.void_state.embryo_growth_states
                        if item.site_id == gf.site_id).emitted_loop_count_effective,
                    gh.emitted_loop_count_effective-next(
                        item for item in initial.void_state.embryo_growth_states
                        if item.site_id == gf.site_id).emitted_loop_count_effective,
                    1e-8)},
            "source_availability": {
                "full": [item.available_fraction for item in gf.cavity_emission_state.sources],
                "two_half": [item.available_fraction for item in gh.cavity_emission_state.sources],
            },
            "random_measure_cursor_equal": all(
                a.random_measure_cursor_count == b.random_measure_cursor_count
                and a.random_measure_seed == b.random_measure_seed
                for a, b in zip(gf.cavity_emission_state.sources,
                                gh.cavity_emission_state.sources)),
        }

    selected_full = [item["selected_event"] for item in full_audit["events"]
                     if item["selected_event"]]
    selected_halves = [item["selected_event"] for audit in (first_audit, second_audit)
                       for item in audit["events"] if item["selected_event"]]
    maximum_source_error = max(item["mixed_relative_error"] for item in source.values())
    maximum_ledger_error = max(
        (item["mixed_relative_error"] for item in ledger_errors.values()), default=0.0)
    maximum_growth_error = max(
        item["effective_loop_count"]["increment_relative_error"]
        for item in growth.values())
    maximum_radius_error = max(
        item["radius"]["relative_error"] for item in growth.values())
    maximum_availability_error = max(
        rel(full_value, half_value, 1e-12)
        for item in growth.values()
        for full_value, half_value in zip(
            item["source_availability"]["full"],
            item["source_availability"]["two_half"]))
    active_birth_site_ids = [
        site.site_id for site in initial.void_state.sites
        if site.phase.value in {"AVAILABLE_SITE", "EMBRYO"}]
    raw_source_gate_applies = bool(active_birth_site_ids)
    inventory_error = rel(inventory_full, inventory_half, 1e-24)
    preserve = (
        max(mechanical.values()) <= 0.005
        and maximum_ledger_error <= 0.005
        and inventory_error <= 0.005
        and maximum_growth_error <= 0.005
        and maximum_radius_error <= 0.005
        and maximum_availability_error <= 0.005
        and (not raw_source_gate_applies or maximum_source_error <= 0.005)
        and selected_full == selected_halves)
    result = finite({
        "schema": "unified-2d.sustained-growth-adaptive-functional-qualification/1",
        "checkpoint": str(CHECKPOINT), "start_time_s": accepted_time(initial),
        "duration_s": DT, "full_end_time_s": accepted_time(full),
        "two_half_end_time_s": accepted_time(halves),
        "mechanical_errors": mechanical,
        "plastic_vacancy_source_exposure": source,
        "full_integrated_source_fraction_area": (
            full_source_audit["integrated_source_fraction_area_per_s"]*DT),
        "two_half_integrated_source_fraction_area": (
            first_source_audit["integrated_source_fraction_area_per_s"]*DT/2
            +second_source_audit["integrated_source_fraction_area_per_s"]*DT/2),
        "vacancy_inventory": {
            "full_fraction_area": inventory_full,
            "two_half_fraction_area": inventory_half,
            "absolute_error_fraction_area": abs(inventory_full-inventory_half),
            "mixed_relative_error": rel(inventory_full, inventory_half, 1e-24)},
        "transport_ledger_errors": ledger_errors,
        "growth": growth,
        "direct_interaction_full": full_direct,
        "direct_interaction_two_half": second_direct,
        "selected_events_full": selected_full,
        "selected_events_two_half": selected_halves,
        "canonical_random_measure_reused": True,
        "event_history_match": selected_full == selected_halves,
        "active_birth_or_embryo_source_owner_site_ids": active_birth_site_ids,
        "raw_plastic_source_gate_applies": raw_source_gate_applies,
        "source_gate_policy": (
            "Raw support source exposure gates AVAILABLE_SITE/EMBRYO first passages. "
            "After all sites are stable it remains a diagnostic; conservative field, "
            "transport-ledger, cavity-growth, source-availability, radius, and event "
            "errors gate acceptance."),
        "maximum_errors": {
            "mechanical": max(mechanical.values()),
            "plastic_vacancy_source": maximum_source_error,
            "transport_ledger": maximum_ledger_error,
            "growth_increment": maximum_growth_error,
            "radius": maximum_radius_error,
            "source_availability": maximum_availability_error,
            "field_vacancy_inventory": inventory_error,
        },
        "accepted_history_disposition": "PRESERVE" if preserve else "REQUIRES_BRANCH",
        "full_phase": phase_summary(full), "two_half_phase": phase_summary(halves),
    })
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(OUT)
    print(json.dumps(result["maximum_errors"], sort_keys=True))
    print(result["accepted_history_disposition"])


if __name__ == "__main__":
    main()
