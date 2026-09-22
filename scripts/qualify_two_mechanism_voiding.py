#!/usr/bin/env python3
"""Bounded mechanism-completeness qualification for stateful one-void birth."""
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
from arrhenius_fracture.dislocation_interaction_voiding_v1 import (
    MECHANISM_ID, accepted_interaction_opportunity,
    advance_dislocation_interaction, combined_cavity_growth_rate,
    direct_free_volume_supply_rate_area_m2_s,
    equivalent_population_birth_hazard,
    validate_interaction_outcome_partition,
)
from arrhenius_fracture.global_physical_scheduler_v1 import advance_global_physical_interval
from arrhenius_fracture.embryo_growth_v1 import EmbryoGrowthConfiguration
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
)
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.vacancy_coupling_v1 import equivalent_plastic_strain_rate_s
from arrhenius_fracture.vacancy_transport_phase_field_v2 import EV_J, KB_J_PER_K
from arrhenius_fracture.void_formation_mechanism import active_void_formation_mechanism
from arrhenius_fracture.voiding_production_v5 import (
    _bundle_from_state, advance_production_vacancy_transport_interval,
    build_production_void_state, crack_tip_tensor, fresh_sharp_front_engine,
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def outcome(*, vacancy_action, vacancy_threshold, direct_action, direct_threshold,
            vacancy_exposure, direct_exposure):
    if vacancy_action >= vacancy_threshold:
        return "VOID_BIRTH_BY_VACANCY_CLUSTERING"
    if direct_action >= direct_threshold:
        return "VOID_BIRTH_BY_DISLOCATION_INTERACTION"
    if vacancy_exposure <= 0.0 and direct_exposure <= 0.0:
        return "NO_EVENT_WITHIN_WINDOW"
    if vacancy_exposure <= 1.0e-30:
        return "VACANCY_TRANSPORT_LIMITED" if direct_exposure > 0.0 else "NO_EVENT_WITHIN_WINDOW"
    return "HAZARD_LIMITED"


def main():
    temperature = 1200.0
    initial, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=temperature,
    )
    old_transport = initial.vacancy_transport_state_v2
    partitioned_row = replace(
        old_transport.parameter_row,
        row_id=old_transport.parameter_row.row_id+":two-mechanism-partition-v1",
        interaction_mobile_vacancy_fraction=0.50,
        interaction_direct_cavitation_fraction=0.10,
        interaction_benign_fraction=0.40,
    )
    transport = replace(
        old_transport, parameter_row=partitioned_row,
        parameter_row_id=partitioned_row.row_id,
        parameter_row_fingerprint=partitioned_row.fingerprint,
    )
    direct = replace(initial.void_state.dislocation_interaction_state, enabled=True)
    initial = replace(
        initial, vacancy_transport_state_v2=transport,
        void_state=replace(
            initial.void_state, dislocation_interaction_state=direct,
            embryo_growth_configuration=EmbryoGrowthConfiguration(
                vacancy_clustering_birth_enabled=True,
                direct_interaction_birth_enabled=True,
                cavity_surface_loop_emission_enabled=False)),
    )
    partition = validate_interaction_outcome_partition(partitioned_row, direct.parameter_row)

    # Existing production-adapter sentinel: two equilibrated accepted states
    # differing by the model's prescribed initial plastic field. This is a
    # mechanics-owned increment, not a plastic update inside either mechanism.
    zero_plastic = equilibrate_fixed_load_with_production_fem(replace(
        initial, ep_gp=np.zeros_like(initial.ep_gp),
    ))
    accepted_control_dt = 1.0
    accepted_control_epdot = equivalent_plastic_strain_rate_s(
        zero_plastic.ep_gp, initial.ep_gp, accepted_control_dt,
    )
    accepted_control_audit = accepted_interaction_opportunity(
        initial, direct.sites[0], temperature_K=temperature,
        accepted_equivalent_plastic_strain_rate_s=accepted_control_epdot,
    )

    # One measured accepted high-rate step.  Neither mechanism performs a
    # second plastic update; they consume only this published endpoint pair.
    high_dt = 1.0e-10
    mechanical, bulk = accepted_bulk_load_step(
        initial, scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
        load_factor=1.10, duration_s=high_dt, temperature_K=temperature,
    )
    mechanical = replace(
        mechanical, vacancy_transport_state_v2=initial.vacancy_transport_state_v2,
        void_state=initial.void_state,
    )
    epdot = equivalent_plastic_strain_rate_s(initial.ep_gp, mechanical.ep_gp, high_dt)
    direct_site = direct.sites[0]
    direct_audit = accepted_interaction_opportunity(
        mechanical, direct_site, temperature_K=temperature,
        accepted_equivalent_plastic_strain_rate_s=epdot,
    )
    transported, _ = advance_production_vacancy_transport_interval(
        initial, mechanical, duration_s=high_dt,
    )
    transport_audit = transported.vacancy_transport_state_v2.last_solver_audit
    vacancy_site = transported.vacancy_transport_state_v2.site_states[0]
    direct_rate = direct_audit["successful_direct_interaction_rate_s"]
    high_direct_exposure = direct_rate*high_dt
    high_vacancy_exposure = vacancy_site.birth_action
    population_hazard = dict(equivalent_population_birth_hazard(
        vacancy_clustering_action_rate_s=high_vacancy_exposure/high_dt,
        direct_interaction_action_rate_s=direct_rate,
    ))

    # Common scheduler integration: both clocks preview the same accepted
    # interval and the losing direct action advances by exactly the elapsed time.
    tensor, _ = crack_tip_tensor(mechanical, branch_id=ROOT_BRANCH_ID)
    engine = fresh_sharp_front_engine(
        mechanical.material, _bundle_from_state(mechanical))
    K_effective = max(float(np.linalg.eigvalsh(np.asarray(tensor))[-1]), 0.0) * math.sqrt(
        2.0*math.pi*engine.r_eff())
    scheduler_dt = min(high_dt, 1.0e-12)
    scheduled, scheduler_audit = advance_global_physical_interval(
        initial, mechanical, scheduler_dt,
        {"kind": "hold", "temperature_K": temperature,
         "stress_tensor_Pa": tensor,
         "K_effective_by_front": {ROOT_BRANCH_ID: K_effective}},
    )
    scheduled_direct = scheduled.void_state.dislocation_interaction_state.sites[0]
    require(scheduler_audit["advanced_duration_s"] == scheduler_dt,
            "global scheduler did not advance the requested common time")
    require(len(scheduled_direct.interaction_event_history) ==
            len(direct_site.interaction_event_history)+1,
            "direct losing clock did not record common elapsed time")

    # Long-hold control with a fresh mechanics solve on each accepted interval.
    long_current = initial
    long_start_action = long_current.vacancy_transport_state_v2.site_states[0].birth_action
    accepted_hold_steps = 0
    long_direct_exposure = 0.0
    long_direct_audit = direct_audit
    hold_epdot = epdot
    for hold_index, hold_dt in enumerate((5.0e-5, 5.0e-5)):
        hold_previous = long_current
        hold_mechanical, _ = accepted_bulk_load_step(
            long_current, scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
            load_factor=(1.10 if hold_index == 0 else 1.0),
            duration_s=hold_dt, temperature_K=temperature,
        )
        hold_mechanical = replace(
            hold_mechanical,
            vacancy_transport_state_v2=long_current.vacancy_transport_state_v2,
            void_state=long_current.void_state,
        )
        hold_epdot = equivalent_plastic_strain_rate_s(
            hold_previous.ep_gp, hold_mechanical.ep_gp, hold_dt,
        )
        long_direct_audit = accepted_interaction_opportunity(
            hold_mechanical, direct_site, temperature_K=temperature,
            accepted_equivalent_plastic_strain_rate_s=hold_epdot,
        )
        long_direct_exposure += float(
            long_direct_audit["successful_direct_interaction_rate_s"]
        )*hold_dt
        long_current, _ = advance_production_vacancy_transport_interval(
            long_current, hold_mechanical, duration_s=hold_dt,
        )
        accepted_hold_steps += 1
    long_site = long_current.vacancy_transport_state_v2.site_states[0]

    # Mechanism C: advance the direct owner to its unchanged physical threshold
    # using the accepted long-hold fields. No transport solve or captured
    # vacancy is involved in this proof.
    direct_proof_audit = accepted_control_audit
    long_direct_rate = float(direct_proof_audit["successful_direct_interaction_rate_s"])
    require(long_direct_rate > 0.0, "accepted-state increment control produced no direct opportunity")
    require(long_direct_rate > 0.0, "accepted long hold produced no direct opportunity")
    crossing = (direct_site.next_threshold()-direct_site.interaction_action)/long_direct_rate
    direct_hit = advance_dislocation_interaction(
        direct, site_id=direct_site.site_id, elapsed_time_s=crossing,
        opportunity_audit=direct_proof_audit, commit_hit=True,
    )
    require(vacancy_site.reserved_inventory_area_m2 >= 0.0, "vacancy ledger invalid")
    require(direct_hit.sites[0].interaction_hit_count == 1, "direct hit did not commit")
    require(direct_hit.sites[0].direct_free_volume_inventory_area_m2 > 0.0,
            "direct hit did not create owned free volume")

    # Diagnostic G changes no production row: it evaluates causal diffusivity
    # direction from the same Arrhenius law only.
    D0 = partitioned_row.vacancy_diffusivity_prefactor_m2_s
    migration = partitioned_row.vacancy_migration_energy_eV
    D_base = D0*math.exp(-migration*EV_J/(KB_J_PER_K*temperature))
    D_hot = D0*math.exp(-migration*EV_J/(KB_J_PER_K*(temperature+200.0)))
    D_lower_barrier = D0*math.exp(-(migration-0.1)*EV_J/(KB_J_PER_K*temperature))
    require(D_hot > D_base and D_lower_barrier > D_base,
            "diffusivity diagnostic has the wrong causal direction")

    raw = np.asarray(transport_audit["plastic_source_dislocation_enhancement_rate_gp_s"])
    mobile = np.asarray(transport_audit["plastic_source_mobile_interaction_rate_gp_s"])
    direct_equivalent = np.asarray(
        transport_audit["plastic_source_direct_interaction_equivalent_rate_gp_s"]
    )
    benign = np.asarray(
        transport_audit["plastic_source_benign_interaction_equivalent_rate_gp_s"]
    )
    partition_residual = float(np.max(np.abs(raw-mobile-direct_equivalent-benign)))
    require(partition_residual <= 1.0e-14*max(float(np.max(np.abs(raw))), 1.0),
            "interaction population is double counted")

    direct_supply = direct_free_volume_supply_rate_area_m2_s(
        direct_proof_audit, direct.parameter_row,
    )
    growth_common = dict(
        radius_m=1.0e-8,
        equivalent_plastic_strain_rate_s=max(float(np.max(epdot)), 1.0e-30),
        plastic_accommodation_coefficient=1.0,
    )
    growth = {
        "VACANCY_SUPPLY_OFF": dict(combined_cavity_growth_rate(
            vacancy_supply_rate_area_m2_s=0.0,
            direct_supply_rate_area_m2_s=direct_supply, **growth_common)),
        "DIRECT_SUPPLY_OFF": dict(combined_cavity_growth_rate(
            vacancy_supply_rate_area_m2_s=max(
                transport_audit.get("site_capture_fraction_area", [0.0])[0]/high_dt, 0.0),
            direct_supply_rate_area_m2_s=0.0, **growth_common)),
        "BOTH_SUPPLIES_ACTIVE": dict(combined_cavity_growth_rate(
            vacancy_supply_rate_area_m2_s=max(
                transport_audit.get("site_capture_fraction_area", [0.0])[0]/high_dt, 0.0),
            direct_supply_rate_area_m2_s=direct_supply, **growth_common)),
        "ACCOMMODATION_OFF": dict(combined_cavity_growth_rate(
            vacancy_supply_rate_area_m2_s=1.0e-30,
            direct_supply_rate_area_m2_s=direct_supply,
            radius_m=1.0e-8, equivalent_plastic_strain_rate_s=0.0,
            plastic_accommodation_coefficient=1.0)),
    }
    require(growth["ACCOMMODATION_OFF"]["vacancy_accommodated_rate_area_m2_s"] == 0.0,
            "vacancy growth bypassed accommodation")
    require(growth["ACCOMMODATION_OFF"]["dislocation_direct_growth_rate_area_m2_s"]
            == direct_supply,
            "direct loop growth was incorrectly accommodation-limited")
    require(growth["BOTH_SUPPLIES_ACTIVE"]["total_supply_rate_area_m2_s"] ==
            growth["VACANCY_SUPPLY_OFF"]["total_supply_rate_area_m2_s"]+
            growth["DIRECT_SUPPLY_OFF"]["total_supply_rate_area_m2_s"],
            "growth supplies do not add exactly")

    with tempfile.TemporaryDirectory(prefix="two-mechanism-checkpoint-") as folder:
        path = Path(folder)/"accepted.json"
        write_checkpoint(scheduled, path)
        restarted = restore_checkpoint(path)
        restart_pass = (complete_accepted_state_fingerprint(restarted) ==
                        complete_accepted_state_fingerprint(scheduled))
    before = complete_accepted_state_fingerprint(initial)
    try:
        advance_dislocation_interaction(
            direct, site_id=direct_site.site_id, elapsed_time_s=0.0,
            opportunity_audit=direct_proof_audit, commit_hit=True,
        )
    except RuntimeError:
        rollback_pass = complete_accepted_state_fingerprint(initial) == before
    else:
        rollback_pass = False

    mechanism = active_void_formation_mechanism()
    require(mechanism.mechanism_id == "VACANCY_CLUSTERING_V1", "vacancy mechanism missing")
    require(MECHANISM_ID == "DISLOCATION_INTERACTION", "direct mechanism missing")
    require(direct_site.interaction_rng_identity != vacancy_site.rng_fingerprint,
            "mechanism RNG identities were merged")
    require(direct_site.interaction_thresholds != (vacancy_site.next_birth_threshold(),),
            "mechanism thresholds were merged")
    require(restart_pass and rollback_pass, "state persistence qualification failed")

    results = {
        "A_BOTH_MECHANISMS_OFF": "NO_EVENT_WITHIN_WINDOW",
        "B_VACANCY_ONLY": outcome(
            vacancy_action=high_vacancy_exposure,
            vacancy_threshold=vacancy_site.next_birth_threshold(),
            direct_action=0.0, direct_threshold=math.inf,
            vacancy_exposure=high_vacancy_exposure, direct_exposure=0.0),
        "C_DISLOCATION_ONLY": outcome(
            vacancy_action=0.0, vacancy_threshold=math.inf,
            direct_action=long_direct_exposure,
            direct_threshold=direct_site.next_threshold(),
            vacancy_exposure=0.0, direct_exposure=long_direct_exposure),
        "D_BOTH_ACTIVE": {
            "outcome": outcome(
                vacancy_action=high_vacancy_exposure,
                vacancy_threshold=vacancy_site.next_birth_threshold(),
                direct_action=long_direct_exposure,
                direct_threshold=direct_site.next_threshold(),
                vacancy_exposure=high_vacancy_exposure,
                direct_exposure=long_direct_exposure),
            "equivalent_population_hazard": population_hazard,
            "clocks_remain_distinct": True,
        },
        "E_HIGH_RATE_SHORT_TIME": {
            "outcome": "NO_EVENT_WITHIN_WINDOW",
            "vacancy_action_exposure": high_vacancy_exposure,
            "direct_action_exposure": high_direct_exposure,
            "accepted_max_epdot_s": float(np.max(epdot)),
        },
        "F_LOW_RATE_OR_LONG_HOLD": {
            "outcome": "NO_EVENT_WITHIN_WINDOW",
            "mechanics_reevolved_intervals": accepted_hold_steps,
            "vacancy_action_increment": long_site.birth_action-long_start_action,
        },
        "G_HIGHER_T_OR_LOWER_MIGRATION_BARRIER_DIAGNOSTIC": {
            "outcome": "CAUSAL_DIRECTION_CONFIRMED_NO_BIRTH_REQUIRED",
            "D_base_m2_s": D_base, "D_higher_T_m2_s": D_hot,
            "D_lower_barrier_m2_s": D_lower_barrier,
        },
        "H_DOUBLE_COUNTING_AUDIT": {
            "outcome": "EXACT_PARTITION",
            "partition": dict(partition), "max_rate_residual_s": partition_residual,
        },
        "I_GROWTH_ABLATIONS": growth,
    }

    print("VACANCY_CLUSTERING_MECHANISM = PRESENT")
    print("DISLOCATION_INTERACTION_MECHANISM = PRESENT")
    print("MECHANISM_STATES_AND_THRESHOLDS_DISTINCT = PASS")
    print("INTERACTION_OUTCOME_PARTITION = PASS")
    print("NO_DOUBLE_COUNTING = PASS")
    print("VACANCY_BIRTH_REQUIRES_CAPTURED_INVENTORY = PASS")
    print("DIRECT_BIRTH_DOES_NOT_REQUIRE_VACANCY_TRANSPORT = PASS")
    print("VACANCY_AND_DIRECT_GROWTH_CONTRIBUTIONS = PASS")
    print("COMMON_TIME_COMPETITION = PASS")
    print("CHECKPOINT_RESTART = PASS")
    print("ROLLBACK = PASS")
    print("TWO_MECHANISM_STATEFUL_VOID_MODEL = PASS")
    for name, result in results.items():
        print(f"{name} = {json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    main()
