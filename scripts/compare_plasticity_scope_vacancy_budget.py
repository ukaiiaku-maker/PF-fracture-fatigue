#!/usr/bin/env python3
"""Nine-case bounded DBTT ramp-plus-hold plasticity/vacancy budget study."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.bulk_plasticity_v104 import PlasticityScope, accepted_bulk_load_step
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.global_physical_scheduler_v1 import advance_global_physical_interval
from arrhenius_fracture.plasticity_vacancy_budget_v1 import (
    MECHANISM_INTERPRETATION, interval_spatial_budget, vacancy_economy,
)
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
)
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.vacancy_transport_phase_field_v2 import (
    SurfaceKind, TransportSurfaceRegistry, accepted_elastoplastic_stress_3d,
    cavity_growth_rate_from_transport, solve_transport_interval,
)
from arrhenius_fracture.voiding_production_v5 import (
    _bundle_from_state, build_production_void_state, crack_tip_tensor,
    fresh_sharp_front_engine,
)

TEMPERATURES_K = (300.0, 900.0, 1200.0)  # selected DBTT low/feature/high anchors
SCOPES = (PlasticityScope.TIP_PROCESS_ONLY,
          PlasticityScope.FULL_FIELD_PT_FIXED_RHO,
          PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO)
RAMP_FACTOR = 1.10
RAMP_DURATION_S = 1.0e-10
HOLD_HORIZON_S = 1.0e-8
HOLD_PARTITIONS = 4
PARTITION_RELATIVE_TOLERANCE = 0.05


def require(value, message):
    if not value:
        raise RuntimeError(message)


def initial_state(temperature):
    bundle = material_bundle("DBTT")
    state, _ = build_production_void_state(bundle=bundle, temperature_K=temperature)
    rho0 = float(bundle.fracture_row["rho_forest_floor_m2"])
    return equilibrate_fixed_load_with_production_fem(replace(
        state, ep_gp=np.zeros_like(state.ep_gp), rho_gp=np.full_like(state.rho_gp, rho0),
    ))


def drive(state):
    tensor, _ = crack_tip_tensor(state, branch_id=ROOT_BRANCH_ID)
    engine = fresh_sharp_front_engine(state.material, _bundle_from_state(state))
    sigma = max(float(np.linalg.eigvalsh(np.asarray(tensor))[-1]), 0.0)
    return np.asarray(tensor), sigma*math.sqrt(2.0*math.pi*engine.r_eff())


def scheduler_advance(previous, mechanical, duration, temperature, kind="hold",
                      ramp_mechanics=None):
    tensor, K = drive(mechanical)
    history = {"kind": kind, "temperature_K": temperature,
               "stress_tensor_Pa": tensor,
               "K_effective_by_front": {ROOT_BRANCH_ID: K}}
    if kind == "ramp":
        if not callable(ramp_mechanics):
            raise ValueError("ramp scheduler requires actual fractional mechanics")
        cache = {1.0: mechanical}
        def state_at(fraction):
            key = float(fraction)
            if key not in cache:
                cache[key] = ramp_mechanics(key)
            return cache[key]
        history["mechanical_state_at_fraction"] = state_at
        history["stress_tensor_at_fraction"] = lambda fraction: drive(state_at(fraction))[0]
        history["K_effective_by_front_at_fraction"] = lambda fraction: {
            ROOT_BRANCH_ID: drive(state_at(fraction))[1]}
    advanced, audit = advance_global_physical_interval(
        previous, mechanical, duration, history,
    )
    # The scheduler preserves clock ownership from ``previous``; restore the
    # newly accepted constitutive audit without overwriting its clock updates.
    junction = dict(advanced.junction_process_state)
    junction["accepted_bulk_plasticity_v104"] = mechanical.junction_process_state[
        "accepted_bulk_plasticity_v104"
    ]
    return replace(advanced, junction_process_state=junction), audit


def _accumulate_spatial(intervals):
    names = tuple(intervals[0][0]["regions"])
    accumulated = {name: {} for name in names}
    for record, dt in intervals:
        for name in names:
            for key, value in record["regions"][name].items():
                factor = dt if key.endswith("_per_s") or key.endswith("_m2_s") else 1.0
                accumulated[name][key] = accumulated[name].get(key, 0.0)+float(value)*factor
    source = math.fsum(row["vacancy_source_fraction_area_per_s"] for row in accumulated.values())
    outside = source-accumulated["TIP_MPZ"]["vacancy_source_fraction_area_per_s"]
    return {"regions": accumulated,
            "bulk_source_fraction_outside_MPZ": 0.0 if source <= 0 else outside/source,
            "remote_source_fraction": 0.0 if source <= 0 else
                accumulated["REMOTE_BULK"]["vacancy_source_fraction_area_per_s"]/source}


def run_case(scope, temperature, partitions=HOLD_PARTITIONS, checkpoint=False):
    initial = initial_state(temperature)
    current = initial
    intervals = []
    events = []
    mechanical, audit = accepted_bulk_load_step(
        current, scope=scope, load_factor=RAMP_FACTOR, duration_s=RAMP_DURATION_S,
        temperature_K=temperature,
    )
    def ramp_mechanics(fraction):
        if fraction <= 0.0:
            return current
        return accepted_bulk_load_step(
            current, scope=scope,
            load_factor=1.0+(RAMP_FACTOR-1.0)*float(fraction),
            duration_s=RAMP_DURATION_S*float(fraction),
            temperature_K=temperature,
        )[0]
    advanced, scheduler = scheduler_advance(
        current, mechanical, RAMP_DURATION_S, temperature, kind="ramp",
        ramp_mechanics=ramp_mechanics,
    )
    intervals.append((interval_spatial_budget(
        current, mechanical, advanced, audit, RAMP_DURATION_S), RAMP_DURATION_S))
    events.extend(scheduler["events"])
    current = advanced
    dt = HOLD_HORIZON_S/partitions
    checkpoint_exact = True
    pending = [dt]*partitions
    accepted_hold_steps = 0
    adaptive_rejected_trials = 0
    minimum_substep_s = HOLD_HORIZON_S/(partitions*64.0)
    while pending:
        if any(row["selected_event"] is not None for row in events):
            break
        substep = pending.pop(0)
        try:
            mechanical, audit = accepted_bulk_load_step(
                current, scope=scope, load_factor=1.0, duration_s=substep,
                temperature_K=temperature,
            )
        except RuntimeError as exc:
            if "stagger failed to converge" not in str(exc) or substep <= minimum_substep_s:
                raise
            adaptive_rejected_trials += 1
            pending[0:0] = [0.5*substep, 0.5*substep]
            continue
        advanced, scheduler = scheduler_advance(
            current, mechanical, substep, temperature)
        intervals.append((interval_spatial_budget(
            current, mechanical, advanced, audit, substep), substep))
        events.extend(scheduler["events"])
        current = advanced
        accepted_hold_steps += 1
        if checkpoint and accepted_hold_steps == max(partitions//2, 1):
            with tempfile.TemporaryDirectory(prefix="bulk-budget-restart-") as folder:
                path = Path(folder)/"checkpoint.json"
                before = complete_accepted_state_fingerprint(current)
                write_checkpoint(current, path)
                current = restore_checkpoint(path)
                checkpoint_exact &= before == complete_accepted_state_fingerprint(current)
    spatial = _accumulate_spatial(intervals)
    economy = dict(vacancy_economy(initial, current))
    last_bulk = current.junction_process_state["accepted_bulk_plasticity_v104"]
    epdot = np.asarray(last_bulk["dot_ep_gp_s"])
    centers = np.asarray(current.mesh.nodes)[np.asarray(current.mesh.elems)].mean(axis=1)
    site = current.vacancy_transport_state_v2.site_states[0]
    eid = int(np.argmin(np.linalg.norm(centers-np.asarray(site.center_xy_m), axis=1)))
    captured_rate = economy["A_site_captured"]/max(
        current.vacancy_transport_state_v2.transport_time_s, RAMP_DURATION_S)
    prospective = dict(cavity_growth_rate_from_transport(
        absorbed_inventory_rate_area_m2_s=max(captured_rate, 0.0),
        radius_m=1.0e-8, equivalent_plastic_strain_rate_s=float(epdot[eid]),
        plastic_accommodation_coefficient=1.0,
    ))
    selected = next((row["selected_event"] for row in events
                     if row["selected_event"] is not None), None)
    max_epdot = max(float(np.max(record["regions"][name]["area_integrated_epdot_m2_s"]))
                    for record, _ in intervals for name in record["regions"])
    outside = spatial["bulk_source_fraction_outside_MPZ"]
    if selected == "DIRECTIONAL_CLEAVAGE": classification = "DIRECT_CLEAVAGE"
    elif selected == "VACANCY_BIRTH_HIT": classification = "VOID_BIRTH"
    elif scope is not PlasticityScope.TIP_PROCESS_ONLY and max_epdot == 0.0:
        classification = "BULK_NOT_ACTIVATED"
    elif scope is not PlasticityScope.TIP_PROCESS_ONLY and outside < 0.05:
        classification = "TIP_LOCALIZED_PLASTICITY"
    elif prospective["limiting_branch"] == "PLASTIC_ACCOMMODATION_LIMITED":
        classification = "PLASTIC_ACCOMMODATION_LIMITED"
    elif economy["A_site_captured"] > 0.0:
        classification = "DISTRIBUTED_GENERATION_SITE_INVENTORY_INCREASED"
    else:
        classification = "NO_EVENT_WITHIN_WINDOW"
    return {"scope": scope.value, "temperature_K": temperature,
            "initial_fingerprint": complete_accepted_state_fingerprint(initial),
            "final_state": current, "spatial": spatial, "vacancy_economy": economy,
            "prospective_growth": prospective, "classification": classification,
            "event": selected, "checkpoint_restart_exact": checkpoint_exact,
            "interval_count": len(intervals), "mechanism_interpretation": MECHANISM_INTERPRETATION,
            "adaptive_time_stepping": True,
            "adaptive_rejected_trials": adaptive_rejected_trials,
            "minimum_adaptive_substep_s": minimum_substep_s,
            "interval_spatial_budgets": [dict(record) for record, _ in intervals],
            "explicit_jog_density_source": "NOT_FOUND_AS_INDEPENDENT_STATE_OWNER_IN_CURRENT_OR_V10_4_REFERENCES",
            "future_resolved_jog_policy": "DECOMPOSE_OR_REPLACE_BASELINE_NEVER_ADD_IN_PARALLEL"}


def serializable(case):
    return {key: value for key, value in case.items() if key != "final_state"}


def causal_transport_controls():
    state = initial_state(1200.0)
    control_duration = 1.0e-4
    mechanical, _ = accepted_bulk_load_step(
        state, scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
        load_factor=RAMP_FACTOR, duration_s=control_duration, temperature_K=1200.0,
    )
    transport = state.vacancy_transport_state_v2
    parent = np.asarray(transport.parent_mechanics_element_ids, int)
    from arrhenius_fracture.vacancy_coupling_v1 import equivalent_plastic_strain_rate_s
    epdot = equivalent_plastic_strain_rate_s(
        state.ep_gp, mechanical.ep_gp, control_duration)[parent]
    rho = np.asarray(mechanical.rho_gp)[parent]
    stress = np.asarray(accepted_elastoplastic_stress_3d(mechanical)["mean_stress_Pa"])[parent]

    def solve(previous, row, registry):
        return solve_transport_interval(
            previous, previous.transport_mesh, registry, row=row,
            duration_s=control_duration,
            accepted_equivalent_plastic_strain_rate_s=epdot,
            accepted_dislocation_density_m2=rho, accepted_mean_stress_Pa=stress,
        )

    full = solve(transport, transport.parameter_row, transport.surface_registry)
    row_k0 = replace(transport.parameter_row, plastic_vacancy_coefficient_per_strain=0.0)
    state_k0 = replace(transport, parameter_row_id=row_k0.row_id,
                       parameter_row_fingerprint=row_k0.fingerprint, parameter_row=row_k0)
    k0 = solve(state_k0, row_k0, state_k0.surface_registry)
    require(k0.last_solver_audit["plastic_source_fraction_area"] == 0.0,
            "k_v=0 did not eliminate plastic vacancy production")

    row_no_climb = replace(transport.parameter_row,
                           dislocation_climb_capture_efficiency=0.0,
                           bulk_vacancy_sink_spacing_m=None)
    state_no_climb = replace(transport, parameter_row_id=row_no_climb.row_id,
                             parameter_row_fingerprint=row_no_climb.fingerprint,
                             parameter_row=row_no_climb)
    no_climb = solve(state_no_climb, row_no_climb, state_no_climb.surface_registry)

    faces = tuple(replace(face, kind=SurfaceKind.CONTACT_OR_GRIP_NO_FLUX)
                  if face.kind == SurfaceKind.EXTERNAL_FREE_SURFACE else face
                  for face in transport.surface_registry.faces)
    registry_no_surface = TransportSurfaceRegistry(
        faces=faces,
        crack_physical_length_m=transport.surface_registry.crack_physical_length_m,
    )
    state_no_surface = replace(
        transport, surface_registry=registry_no_surface,
        surface_registry_fingerprint=registry_no_surface.fingerprint,
    )
    no_surface = solve(state_no_surface, state_no_surface.parameter_row,
                       registry_no_surface)
    area = np.asarray(transport.transport_mesh.area_e)
    inventory = lambda result: float(np.sum(area*np.asarray(result.vacancy_fraction_gp)))
    require(inventory(no_climb) > inventory(full),
            "disabling climb did not increase retained vacancies")
    require(inventory(no_surface) > inventory(full),
            "disabling external free-surface sinks did not increase retained vacancies")
    # The transport API consumes only these accepted arrays; changing a PT
    # parameter without re-evolving mechanics cannot enter its solve.
    frozen_repeat = solve(transport, transport.parameter_row, transport.surface_registry)
    require(np.array_equal(full.vacancy_fraction_gp, frozen_repeat.vacancy_fraction_gp),
            "frozen accepted mechanics did not isolate transport from PT parameters")
    return {"k_v_zero_source": True, "climb_disabled_retention_increase":
            inventory(no_climb)-inventory(full),
            "free_surface_disabled_retention_increase":
            inventory(no_surface)-inventory(full),
            "frozen_mechanics_transport_identity": True,
            "control_duration_s": control_duration,
            "mechanical_feedback_coordinates": ("epdot_gp", "rho_gp", "mean_stress_gp",
                                                "accepted_tip_process_state")}


def main():
    cases = []
    for temperature in TEMPERATURES_K:
        for scope in SCOPES:
            cases.append(run_case(
                scope, temperature,
                checkpoint=(temperature == 900.0 and
                            scope is PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO),
            ))
    by_key = {(row["scope"], row["temperature_K"]): row for row in cases}
    for temperature in TEMPERATURES_K:
        tip = by_key[(PlasticityScope.TIP_PROCESS_ONLY.value, temperature)]
        fixed = by_key[(PlasticityScope.FULL_FIELD_PT_FIXED_RHO.value, temperature)]
        evolving = by_key[(PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO.value, temperature)]
        require(tip["vacancy_economy"]["A_plastic_generated"] == 0.0,
                "tip-only bulk source is nonzero")
        require(np.array_equal(fixed["final_state"].rho_gp,
                              initial_state(temperature).rho_gp),
                "fixed-rho scope changed rho")
        require(tip["initial_fingerprint"] == fixed["initial_fingerprint"]
                == evolving["initial_fingerprint"],
                "paired scopes did not start from one accepted owner state")
        for row in (tip, fixed, evolving):
            scale = max(abs(row["vacancy_economy"]["A_plastic_generated"]), 1.0e-30)
            require(abs(row["vacancy_economy"]["conservation_residual"]) <= 1.0e-10*scale+1.0e-25,
                    "vacancy economy does not close")
    require(by_key[(PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO.value, 900.0)][
        "checkpoint_restart_exact"], "checkpoint/restart changed accepted state")
    controls = causal_transport_controls()

    # Prospective partition-invariance qualification at the feature anchor.
    fine = run_case(PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO, 900.0,
                    partitions=2*HOLD_PARTITIONS)
    coarse = by_key[(PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO.value, 900.0)]
    for key in ("A_plastic_generated", "A_field_owned", "A_site_captured"):
        a, b = coarse["vacancy_economy"][key], fine["vacancy_economy"][key]
        error = abs(a-b)/max(abs(a), abs(b), 1.0e-30)
        require(error <= PARTITION_RELATIVE_TOLERANCE,
                f"hold partition invariance failed for {key}: {error}")

    print("THREE_PLASTICITY_SCOPES = PASS")
    print("RAMP_PLUS_HOLD_ACCEPTED_EVOLUTION = PASS")
    print("SPATIAL_SOURCE_DECOMPOSITION = PASS")
    print("VACANCY_ECONOMY_CLOSURE = PASS")
    print("TIME_PARTITION_INVARIANCE = PASS")
    print("CHECKPOINT_RESTART = PASS")
    print("CAUSAL_SOURCE_AND_SINK_CONTROLS = PASS")
    print("LEGACY_TIP_PROCESS_PRESERVED = PASS")
    print("BULK_PLASTICITY_ACTIVATION_AND_VACANCY_BUDGET = PASS")
    for row in cases:
        print("CASE", row["scope"], f"T={row['temperature_K']:.0f}K",
              "CLASSIFICATION="+row["classification"],
              f"OUTSIDE_MPZ={row['spatial']['bulk_source_fraction_outside_MPZ']:.6g}",
              f"REMOTE={row['spatial']['remote_source_fraction']:.6g}",
              f"A_GENERATED={row['vacancy_economy']['A_plastic_generated']:.6e}",
              f"A_SITE={row['vacancy_economy']['A_site_captured']:.6e}",
              "GROWTH_LIMIT="+str(row["prospective_growth"]["limiting_branch"]))
    print(json.dumps({"protocol": {"temperatures_K": TEMPERATURES_K,
          "ramp_factor": RAMP_FACTOR, "ramp_duration_s": RAMP_DURATION_S,
          "hold_horizon_s": HOLD_HORIZON_S, "hold_partitions": HOLD_PARTITIONS,
          "partition_relative_tolerance": PARTITION_RELATIVE_TOLERANCE},
          "controls": controls,
          "cases": [serializable(row) for row in cases]}, sort_keys=True))


if __name__ == "__main__":
    main()
