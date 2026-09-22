from dataclasses import replace
import math
import subprocess
import sys

import numpy as np
import pytest

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.embryo_growth_v1 import CavityEmissionParameterRowV1
from arrhenius_fracture.dislocation_interaction_voiding_v1 import (
    DislocationInteractionParameterRowV1,
    accepted_interaction_opportunity, advance_dislocation_interaction,
    advance_live_direct_owner,
    combined_cavity_growth_rate, validate_interaction_outcome_partition,
    equivalent_population_birth_hazard, transfer_direct_birth_inventory,
)
from arrhenius_fracture.global_physical_scheduler_v1 import (
    PhysicalEventPreview, select_physical_event,
)
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.vacancy_transport_phase_field_v2 import (
    VacancyTransportParameterRowV2, solve_transport_interval,
)
from arrhenius_fracture.voiding_production_v5 import build_production_void_state


@pytest.fixture(scope="module")
def accepted_state():
    state, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=900.0,
    )
    epdot = np.full(state.mesh.ne, 1.0e-3)
    junction = dict(state.junction_process_state)
    junction["accepted_bulk_plasticity_v104"] = {
        "dot_ep_gp_s": epdot,
        "constitutive": {"pt_rho_mobile_gp": np.full(state.mesh.ne, 1.0e10)},
    }
    direct = replace(state.void_state.dislocation_interaction_state, enabled=True)
    return replace(state, junction_process_state=junction, void_state=replace(
        state.void_state, dislocation_interaction_state=direct,
    ))


def test_distinct_state_owned_mechanisms_and_checkpoint(accepted_state, tmp_path):
    direct = accepted_state.void_state.dislocation_interaction_state
    vacancy = accepted_state.vacancy_transport_state_v2.site_states[0]
    assert direct.sites[0].interaction_rng_identity != vacancy.rng_fingerprint
    assert direct.sites[0].interaction_thresholds != (vacancy.next_birth_threshold(),)
    path = tmp_path/"two-mechanism.json"
    write_checkpoint(accepted_state, path)
    restored = restore_checkpoint(path)
    assert complete_accepted_state_fingerprint(restored) == complete_accepted_state_fingerprint(accepted_state)
    assert restored.void_state.dislocation_interaction_state.fingerprint == direct.fingerprint


def test_builder_explicitly_activates_established_partition_and_virgin_bulk_state():
    bundle = material_bundle("DBTT")
    state, _ = build_production_void_state(
        bundle=bundle, temperature_K=1200.0,
        direct_interaction_birth_enabled=True,
        cavity_surface_loop_emission_enabled=True,
        interaction_population_partition=(0.50, 0.10, 0.40),
        bulk_initial_density_m2=float(bundle.fracture_row["rho_forest_floor_m2"]),
        initial_plastic_strain_gp=0.0,
    )
    cfg = state.void_state.embryo_growth_configuration
    row = state.vacancy_transport_state_v2.parameter_row
    assert cfg.vacancy_clustering_birth_enabled
    assert cfg.direct_interaction_birth_enabled
    assert cfg.cavity_surface_loop_emission_enabled
    assert state.void_state.dislocation_interaction_state.enabled
    assert row.interaction_mobile_vacancy_fraction == pytest.approx(0.50)
    assert row.interaction_direct_cavitation_fraction == pytest.approx(0.10)
    assert row.interaction_benign_fraction == pytest.approx(0.40)
    assert np.all(state.ep_gp == 0.0)
    assert np.all(state.rho_gp == float(bundle.fracture_row["rho_forest_floor_m2"]))


def test_builder_owns_declared_direct_and_emission_hypothesis_rows(tmp_path):
    direct = replace(
        DislocationInteractionParameterRowV1(),
        row_id="test.direct-hypothesis", interaction_geometry_factor_C_dd=1.0e-8,
        interaction_barrier_eV=0.25,
    )
    emission = replace(
        CavityEmissionParameterRowV1(),
        row_id="test.emission-hypothesis", zero_stress_barrier_J=1.8e-19,
        chi_loop=direct.chi_loop,
    )
    state, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=1200.0,
        direct_interaction_parameter_row=direct,
        cavity_emission_parameter_row=emission,
    )

    assert state.void_state.dislocation_interaction_state.parameter_row == direct
    assert state.void_state.embryo_growth_configuration.cavity_emission_parameter_row == emission
    checkpoint = tmp_path / "declared-hypothesis.json"
    write_checkpoint(state, checkpoint)
    restored = restore_checkpoint(checkpoint)
    assert restored.void_state.dislocation_interaction_state.parameter_row == direct
    assert restored.void_state.embryo_growth_configuration.cavity_emission_parameter_row == emission
    assert complete_accepted_state_fingerprint(restored) == complete_accepted_state_fingerprint(state)


def test_direct_opportunity_uses_accepted_fields_and_never_updates_plasticity(accepted_state):
    site = accepted_state.void_state.dislocation_interaction_state.sites[0]
    before_ep = np.asarray(accepted_state.ep_gp).copy()
    audit = accepted_interaction_opportunity(accepted_state, site, temperature_K=900.0)
    assert audit["support_integrated_interaction_rate_s"] > 0.0
    assert audit["successful_direct_interaction_rate_s"] > 0.0
    assert audit["plasticity_update_inside_mechanism"] is False
    assert np.array_equal(accepted_state.ep_gp, before_ep)
    zero = accepted_interaction_opportunity(
        accepted_state, site, temperature_K=900.0,
        accepted_equivalent_plastic_strain_rate_s=np.zeros(accepted_state.mesh.ne),
    )
    assert zero["support_integrated_interaction_rate_s"] == 0.0


def test_direct_hit_has_own_inventory_and_requires_no_vacancy_capture(accepted_state):
    owner = accepted_state.void_state.dislocation_interaction_state
    site = owner.sites[0]
    audit = accepted_interaction_opportunity(accepted_state, site, temperature_K=900.0)
    rate = audit["successful_direct_interaction_rate_s"]
    elapsed = (site.next_threshold()-site.interaction_action)/rate
    advanced = advance_dislocation_interaction(
        owner, site_id=site.site_id, elapsed_time_s=elapsed,
        opportunity_audit=audit, commit_hit=True,
    )
    after = advanced.sites[0]
    assert accepted_state.vacancy_transport_state_v2.site_states[0].reserved_inventory_area_m2 == 0.0
    assert after.interaction_hit_count == 1
    assert after.direct_free_volume_inventory_area_m2 == after.direct_free_volume_per_hit_area_m2

    transferred, amount, count = transfer_direct_birth_inventory(
        advanced, site.site_id)
    assert amount == after.direct_free_volume_per_hit_area_m2
    assert count == 1
    assert transferred.sites[0].direct_free_volume_inventory_area_m2 == 0.0
    retransferred, second_amount, second_count = transfer_direct_birth_inventory(
        transferred, site.site_id)
    assert second_amount == 0.0 and second_count == count
    assert retransferred.fingerprint == transferred.fingerprint


def test_direct_multihit_owner_creates_exactly_one_embryo(accepted_state):
    state = accepted_state
    epdot = state.junction_process_state["accepted_bulk_plasticity_v104"]["dot_ep_gp_s"]
    mean_stress = np.zeros(state.mesh.ne)
    for expected_hit in (1, 2, 3):
        site = state.void_state.dislocation_interaction_state.sites[0]
        audit = accepted_interaction_opportunity(
            state, site, temperature_K=900.0,
            accepted_equivalent_plastic_strain_rate_s=epdot,
            accepted_mean_stress_Pa=mean_stress,
        )
        elapsed = (site.next_threshold()-site.interaction_action) / audit[
            "successful_direct_interaction_rate_s"
        ]
        state = advance_live_direct_owner(
            accepted_state, state, elapsed_time_s=elapsed, temperature_K=900.0,
            accepted_equivalent_plastic_strain_rate_s=epdot,
            accepted_mean_stress_Pa=mean_stress, winning_site_id=site.site_id,
        )
        assert state.void_state.dislocation_interaction_state.sites[0].interaction_hit_count == expected_hit
    from arrhenius_fracture.voiding_v5 import VoidPhase
    assert state.void_state.sites[0].phase == VoidPhase.EMBRYO
    assert sum(row["event"] == "EMBRYO" for row in state.void_state.event_history) == 1
    unchanged = advance_live_direct_owner(
        accepted_state, state, elapsed_time_s=1.0, temperature_K=900.0,
        accepted_equivalent_plastic_strain_rate_s=epdot,
        accepted_mean_stress_Pa=mean_stress, winning_site_id=None,
    )
    assert unchanged.void_state.dislocation_interaction_state.fingerprint == \
        state.void_state.dislocation_interaction_state.fingerprint


def test_direct_winner_atomically_merges_preexisting_vacancy_inventory(accepted_state):
    amount = 2.0e-18
    transport = accepted_state.vacancy_transport_state_v2
    transport_site = replace(transport.site_states[0], reserved_inventory_area_m2=amount)
    ledgers = dict(transport.ledgers)
    ledgers["reserved_embryo_inventory_area_m2"] += amount
    ledgers["initial_owned_inventory_area_m2"] += amount
    state = replace(
        accepted_state,
        vacancy_transport_state_v2=replace(
            transport, site_states=(transport_site,), ledgers=ledgers))
    epdot = state.junction_process_state["accepted_bulk_plasticity_v104"]["dot_ep_gp_s"]
    for _ in range(3):
        direct_site = state.void_state.dislocation_interaction_state.sites[0]
        audit = accepted_interaction_opportunity(
            state, direct_site, temperature_K=900.0,
            accepted_equivalent_plastic_strain_rate_s=epdot,
            accepted_mean_stress_Pa=np.zeros(state.mesh.ne))
        elapsed = (direct_site.next_threshold()-direct_site.interaction_action) / audit[
            "successful_direct_interaction_rate_s"]
        state = advance_live_direct_owner(
            accepted_state, state, elapsed_time_s=elapsed, temperature_K=900.0,
            accepted_equivalent_plastic_strain_rate_s=epdot,
            accepted_mean_stress_Pa=np.zeros(state.mesh.ne),
            winning_site_id=direct_site.site_id)
    growth = state.void_state.embryo_growth_states[0]
    direct_site = state.void_state.dislocation_interaction_state.sites[0]
    assert growth.vacancy_derived_area_m2 == amount
    assert growth.direct_interaction_birth_hits == 3
    assert growth.emitted_loop_count_effective == 0.0
    assert growth.dislocation_derived_area_m2 > 0.0
    assert direct_site.direct_free_volume_inventory_area_m2 == 0.0
    assert state.vacancy_transport_state_v2.site_states[0].reserved_inventory_area_m2 == 0.0


def test_partition_removes_direct_share_from_transport_source(accepted_state):
    old = accepted_state.vacancy_transport_state_v2
    direct_row = accepted_state.void_state.dislocation_interaction_state.parameter_row
    row = replace(
        old.parameter_row, row_id=old.parameter_row.row_id+":two-mechanism",
        interaction_mobile_vacancy_fraction=0.50,
        interaction_direct_cavitation_fraction=0.10,
        interaction_benign_fraction=0.40,
    )
    partition = validate_interaction_outcome_partition(row, direct_row)
    assert partition["sum"] == pytest.approx(1.0)
    transport = replace(old, parameter_row=row, parameter_row_id=row.row_id,
                        parameter_row_fingerprint=row.fingerprint)
    result = solve_transport_interval(
        transport, transport.transport_mesh, transport.surface_registry,
        row=row, duration_s=1.0e-6,
        accepted_equivalent_plastic_strain_rate_s=np.full(transport.transport_mesh.ne, 1.0e-3),
        accepted_dislocation_density_m2=np.full(transport.transport_mesh.ne, 1.0e14),
        accepted_mean_stress_Pa=np.zeros(transport.transport_mesh.ne),
    )
    audit = result.last_solver_audit
    raw = np.asarray(audit["plastic_source_dislocation_enhancement_rate_gp_s"])
    mobile = np.asarray(audit["plastic_source_mobile_interaction_rate_gp_s"])
    direct = np.asarray(audit["plastic_source_direct_interaction_equivalent_rate_gp_s"])
    benign = np.asarray(audit["plastic_source_benign_interaction_equivalent_rate_gp_s"])
    assert np.allclose(mobile+direct+benign, raw)
    assert np.allclose(mobile, 0.5*raw)


def test_independent_competing_risks_and_losing_clock_advance(accepted_state):
    common = complete_accepted_state_fingerprint(accepted_state)
    def preview(kind, owner, time):
        return PhysicalEventPreview(
            event_type=kind, owner_kind=owner, owner_id=owner,
            crossing_time_s=time, accepted_state_fingerprint=common,
            source_fingerprint="source", topology_fingerprint="topology",
            event_ids=(kind,), candidate_ids=(owner,),
            threshold_identity=((kind, time),), rng_fingerprint="rng",
            commit_payload={},
        )
    vacancy = preview("VACANCY_BIRTH_HIT", "VacancyTransportStateV2", 2.0)
    direct = preview("DISLOCATION_INTERACTION_HIT", "DislocationInteractionStateV1", 1.0)
    winner, tied = select_physical_event((vacancy, direct))
    assert winner is direct and tied == (direct,)
    owner = accepted_state.void_state.dislocation_interaction_state
    site = owner.sites[0]
    audit = accepted_interaction_opportunity(accepted_state, site, temperature_K=900.0)
    losing = advance_dislocation_interaction(
        owner, site_id=site.site_id, elapsed_time_s=0.5,
        opportunity_audit=audit, commit_hit=False,
    )
    assert losing.sites[0].interaction_action > site.interaction_action
    assert losing.sites[0].interaction_hit_count == site.interaction_hit_count
    population = equivalent_population_birth_hazard(
        vacancy_clustering_action_rate_s=0.25,
        direct_interaction_action_rate_s=0.75,
    )
    assert population["h_birth_total_s"] == 1.0


def test_growth_ablations_are_mechanism_resolved():
    kwargs = dict(radius_m=1.0e-6, equivalent_plastic_strain_rate_s=1.0e-3,
                  plastic_accommodation_coefficient=10.0)
    vacancy = combined_cavity_growth_rate(
        vacancy_supply_rate_area_m2_s=1.0e-12,
        direct_supply_rate_area_m2_s=0.0, **kwargs,
    )
    direct = combined_cavity_growth_rate(
        vacancy_supply_rate_area_m2_s=0.0,
        direct_supply_rate_area_m2_s=2.0e-12, **kwargs,
    )
    both = combined_cavity_growth_rate(
        vacancy_supply_rate_area_m2_s=1.0e-12,
        direct_supply_rate_area_m2_s=2.0e-12, **kwargs,
    )
    no_accommodation = combined_cavity_growth_rate(
        vacancy_supply_rate_area_m2_s=1.0e-12,
        direct_supply_rate_area_m2_s=2.0e-12,
        radius_m=kwargs["radius_m"], equivalent_plastic_strain_rate_s=0.0,
        plastic_accommodation_coefficient=kwargs["plastic_accommodation_coefficient"],
    )
    assert both["Rdot_m_s"] > vacancy["Rdot_m_s"]
    assert both["Rdot_m_s"] > direct["Rdot_m_s"]
    assert no_accommodation["vacancy_accommodated_rate_area_m2_s"] == 0.0
    assert no_accommodation["dislocation_direct_growth_rate_area_m2_s"] == 2.0e-12
    assert no_accommodation["Rdot_m_s"] > 0.0


def test_direct_state_is_immutable_under_failed_preview(accepted_state):
    owner = accepted_state.void_state.dislocation_interaction_state
    before = owner.fingerprint
    site = owner.sites[0]
    audit = accepted_interaction_opportunity(accepted_state, site, temperature_K=900.0)
    with pytest.raises(RuntimeError):
        advance_dislocation_interaction(
            owner, site_id=site.site_id, elapsed_time_s=0.0,
            opportunity_audit=audit, commit_hit=True,
        )
    assert owner.fingerprint == before


def test_two_mechanism_terminal_contract():
    completed = subprocess.run(
        [sys.executable, "scripts/qualify_two_mechanism_voiding.py"],
        check=True, capture_output=True, text=True,
    )
    required = (
        "VACANCY_CLUSTERING_MECHANISM = PRESENT",
        "DISLOCATION_INTERACTION_MECHANISM = PRESENT",
        "MECHANISM_STATES_AND_THRESHOLDS_DISTINCT = PASS",
        "INTERACTION_OUTCOME_PARTITION = PASS", "NO_DOUBLE_COUNTING = PASS",
        "VACANCY_BIRTH_REQUIRES_CAPTURED_INVENTORY = PASS",
        "DIRECT_BIRTH_DOES_NOT_REQUIRE_VACANCY_TRANSPORT = PASS",
        "VACANCY_AND_DIRECT_GROWTH_CONTRIBUTIONS = PASS",
        "COMMON_TIME_COMPETITION = PASS", "CHECKPOINT_RESTART = PASS",
        "ROLLBACK = PASS", "TWO_MECHANISM_STATEFUL_VOID_MODEL = PASS",
    )
    assert all(line in completed.stdout for line in required)
