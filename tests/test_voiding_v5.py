from dataclasses import replace
import math

import numpy as np
import pytest

from arrhenius_fracture.voiding_production_v5 import (
    _complete_next_clock, build_production_void_state,
    deterministic_trajectory, downstream_front_transaction, ligament_transaction, natural_trajectory,
)
from arrhenius_fracture.directional_competition_v11 import (
    CleavageCandidate, DirectionalCompetitionState, DirectionalHazardState, tungsten_cleavage_candidates,
)
from arrhenius_fracture.voiding_v5 import (
    Cavity2D, VoidPhase, VoidingConfig, advance_site, arrhenius_rates,
    create_subgrid_cavity, grow_cavity_2d, grow_cavity_from_rate, update_cavity_growth,
)


def test_plane_strain_cavity_inventory_is_area_not_spherical_inventory():
    radius = 5.0e-5
    cavity = Cavity2D("v", "s", (0.0, 0.0), radius, math.pi * radius**2,
                      math.pi * radius**2, VoidPhase.RESOLVED_VOID)
    grown = grow_cavity_2d(cavity, 1.0e-5)
    assert grown.area_m2 == math.pi * grown.radius_m**2
    assert grown.inventory_area_m2 - cavity.inventory_area_m2 == grown.area_m2 - cavity.area_m2


def test_deterministic_driver_reaches_real_continued_graph_event():
    final, rows = deterministic_trajectory()
    assert rows[-1]["operation"] == "continued_accepted_event"
    assert rows[-1]["event_counters"]["topology_actions"] >= 3
    assert final.void_state.cavities[0].phase == VoidPhase.DOWNSTREAM_FRONT_ACTIVE


def test_natural_driver_integrates_actual_stress_history():
    final, rows = natural_trajectory()
    assert len(rows) == 6
    assert all(row["rates"]["local_max_principal_stress_Pa"] >= 0.0 for row in rows)
    assert final.void_state.sites[0].birth.accumulated >= 0.0


def test_multihit_renews_stochastic_threshold_and_rng_state():
    state, _ = build_production_void_state(stochastic=True)
    site = state.void_state.sites[0]
    rates = {"birth_s": 1.0, "stabilization_s": 0.0, "healing_s": 0.0}
    advanced, events = advance_site(
        state.void_state, site.site_id, site.birth.threshold / site.candidate_weight,
        rates=rates,
    )
    renewed = advanced.sites[0]
    assert events == ("BIRTH_HIT",)
    assert renewed.birth.threshold != site.birth.threshold
    assert advanced.rng_state != state.void_state.rng_state


def test_series_limiter_and_duplicate_cavity_guard():
    cfg = VoidingConfig(enabled=True)
    rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=np.eye(2) * 1.0e9)
    assert rates["series_limited_growth_s"] < min(
        rates["surface_reaction_s"], rates["vacancy_transport_s"]
    )
    cavity = Cavity2D("v", "s", (0.0, 0.0), 1.0e-6, math.pi * 1.0e-12,
                      math.pi * 1.0e-12, VoidPhase.STABLE_SUBGRID_VOID)
    grown = grow_cavity_from_rate(
        cavity, rates=rates, dt_s=1.0e-12,
        radial_growth_scale_m=cfg.radial_growth_scale_m,
    )
    assert grown.radius_m > cavity.radius_m

    state, _ = build_production_void_state()
    site = replace(state.void_state.sites[0], phase=VoidPhase.STABLE_SUBGRID_VOID)
    voids = replace(state.void_state, sites=(site,))
    voids = create_subgrid_cavity(voids, site.site_id, 1.0e-6)
    with pytest.raises(ValueError, match="only one cavity"):
        create_subgrid_cavity(voids, site.site_id, 1.0e-6)


def test_no_zero_drive_or_unit_rate_bypass_for_cleavage_first_passage():
    state, _ = build_production_void_state()
    with pytest.raises(RuntimeError, match="cannot reach first passage"):
        _complete_next_clock(state, np.zeros((2, 2)))


def test_activation_work_has_energy_units_and_reference_is_not_saturated():
    cfg = VoidingConfig(enabled=True)
    tensor = np.array([[1.0e9, 1.0e8], [1.0e8, 0.6e9]])
    base = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
    hydro = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor + np.eye(2) * 2.0e8)
    reversed_shear = arrhenius_rates(
        cfg, temperature_K=900.0,
        stress_tensor_Pa=np.array([[1.0e9, -1.0e8], [-1.0e8, 0.6e9]]),
    )
    barrier = arrhenius_rates(
        replace(cfg, birth_barrier_J=1.1 * cfg.birth_barrier_J),
        temperature_K=900.0, stress_tensor_Pa=tensor,
    )
    assert 0.0 < base["birth_activation_work_J"] < cfg.birth_barrier_J
    assert base["birth_s"] < cfg.attempt_frequency_s
    assert hydro["birth_s"] > base["birth_s"]
    assert reversed_shear["birth_s"] < base["birth_s"]
    assert barrier["birth_s"] < base["birth_s"]


def test_directional_candidates_advance_over_one_common_earliest_interval():
    state, _ = build_production_void_state()
    candidates = tungsten_cleavage_candidates(theta_deg=45.0)
    competition = DirectionalCompetitionState(
        candidates=candidates,
        hazard_states=tuple(DirectionalHazardState(item.candidate_id) for item in candidates),
        global_hazard_seed=3621,
    )
    state = replace(state, competition=competition)
    advanced, audit = _complete_next_clock(
        state, np.array([[2.0e9, 4.0e8], [4.0e8, 0.5e9]])
    )
    durations = {item["common_advance_duration_s"] for item in audit}
    assert len(durations) == 1
    assert sum(item["winner"] for item in audit) == 1
    assert len(advanced.competition.pending_events) == 1
    assert all(item.action > 0.0 for item in advanced.competition.hazard_states)


def test_directional_exact_tie_fires_only_prospectively_equal_candidates():
    state, _ = build_production_void_state()
    candidates = tungsten_cleavage_candidates(theta_deg=45.0)
    competition = DirectionalCompetitionState(
        candidates=candidates,
        hazard_states=tuple(DirectionalHazardState(item.candidate_id) for item in candidates),
        global_hazard_seed=3621,
    )
    state = replace(state, competition=competition)
    advanced, audit = _complete_next_clock(state, np.eye(2) * 1.0e9)
    assert sum(item["winner"] for item in audit) == 2
    assert len(advanced.competition.pending_events) == 2


def test_zero_rate_candidate_keeps_residual_while_peer_wins():
    state, _ = build_production_void_state()
    candidates = tungsten_cleavage_candidates(theta_deg=45.0)
    competition = DirectionalCompetitionState(
        candidates=candidates,
        hazard_states=tuple(DirectionalHazardState(item.candidate_id) for item in candidates),
        global_hazard_seed=3621,
    )
    state = replace(state, competition=competition)
    # Rank-one tension opens one candidate plane and closes the other.
    tensor = np.array([[1.0e9, 1.0e9], [1.0e9, 1.0e9]])
    advanced, audit = _complete_next_clock(state, tensor)
    assert sum(item["winner"] for item in audit) == 1
    loser = next(index for index, item in enumerate(audit) if not item["winner"])
    assert audit[loser]["rate_s"] == 0.0
    assert advanced.competition.hazard_states[loser].action == 0.0


@pytest.mark.parametrize("stage", [
    "cavity_phase_update",
    "length_ledger_update:fractured_ligament_length_m",
    "length_ledger_update:active_front_coordinate_advance_m",
    "connected_surface_certification",
])
def test_connection_phase_surface_and_ledgers_rollback_atomically(stage):
    accepted, _ = deterministic_trajectory(stop_before_ligament=True)
    from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
    before = complete_accepted_state_fingerprint(accepted)
    with pytest.raises(RuntimeError, match="injected:" + stage):
        ligament_transaction(accepted, failure_stage=stage)
    assert complete_accepted_state_fingerprint(accepted) == before


def test_growth_uses_drive_magnitude_inventory_and_configured_shrinkage():
    cfg = VoidingConfig(enabled=True)
    cavity = Cavity2D("v", "s", (0.0, 0.0), 1.0e-6, math.pi * 1.0e-12,
                      math.pi * 1.0e-12, VoidPhase.STABLE_SUBGRID_VOID)
    rates = {"series_limited_growth_s": 1.0}
    low = grow_cavity_from_rate(cavity, rates=rates, dt_s=1.0,
                                radial_growth_scale_m=1.0e-8,
                                chemical_potential_drive_J=1.0e-21)
    high = grow_cavity_from_rate(cavity, rates=rates, dt_s=1.0,
                                 radial_growth_scale_m=1.0e-8,
                                 chemical_potential_drive_J=2.0e-21)
    exhausted = grow_cavity_from_rate(cavity, rates=rates, dt_s=1.0,
                                      radial_growth_scale_m=1.0e-8,
                                      chemical_potential_drive_J=2.0e-21,
                                      available_inventory_area_m2=0.0)
    shrink = grow_cavity_from_rate(cavity, rates=rates, dt_s=1.0,
                                   radial_growth_scale_m=1.0e-8,
                                   chemical_potential_drive_J=-1.0e-20,
                                   shrinkage_mobility_m_per_J_s=2.0e8)
    assert high.radius_m - cavity.radius_m == pytest.approx(2.0 * (low.radius_m - cavity.radius_m))
    assert exhausted.radius_m == cavity.radius_m
    assert shrink.radius_m < cavity.radius_m


def test_state_owned_growth_and_shrinkage_conserve_inventory_area():
    # Use a minimal standalone state because site lifecycle is irrelevant here.
    cavity = Cavity2D("v", "s", (0.0, 0.0), 1.0e-6, math.pi * 1.0e-12,
                      math.pi * 1.0e-12, VoidPhase.STABLE_SUBGRID_VOID)
    from arrhenius_fracture.voiding_v5 import ProductionVoidState
    inventory = ProductionVoidState((), (cavity,), available_defect_inventory_area_m2=1.0e-9)
    total = inventory.available_defect_inventory_area_m2 + inventory.consumed_defect_inventory_area_m2
    grown = update_cavity_growth(inventory, "v", rates={"series_limited_growth_s": 1.0},
                                 dt_s=1.0, radial_growth_scale_m=1.0e-8)
    delta = grown.cavities[0].area_m2 - cavity.area_m2
    assert inventory.available_defect_inventory_area_m2 - grown.available_defect_inventory_area_m2 == pytest.approx(delta)
    assert grown.consumed_defect_inventory_area_m2 == pytest.approx(delta)
    assert grown.available_defect_inventory_area_m2 + grown.consumed_defect_inventory_area_m2 == pytest.approx(total)
    shrunk = update_cavity_growth(grown, "v", rates={"series_limited_growth_s": 1.0},
                                  dt_s=1.0, radial_growth_scale_m=1.0e-8,
                                  chemical_potential_drive_J=-1.0e-20)
    assert shrunk.available_defect_inventory_area_m2 + shrunk.consumed_defect_inventory_area_m2 == pytest.approx(total)
    assert shrunk.consumed_defect_inventory_area_m2 < grown.consumed_defect_inventory_area_m2


def test_initial_cavity_seed_debits_finite_inventory_atomically():
    from arrhenius_fracture.voiding_v5 import HazardClock, ProductionVoidState, VoidSite
    site = VoidSite("s", (0.0, 0.0), VoidPhase.STABLE_SUBGRID_VOID, 2, 2, 1.0,
                    HazardClock(1.0, 1.0), HazardClock(1.0, 1.0), HazardClock(0.0, 1.0))
    state = ProductionVoidState((site,), available_defect_inventory_area_m2=1.0e-8)
    seeded = create_subgrid_cavity(state, "s", 2.0e-6)
    area = math.pi * (2.0e-6) ** 2
    assert seeded.available_defect_inventory_area_m2 == pytest.approx(1.0e-8 - area)
    assert seeded.consumed_defect_inventory_area_m2 == pytest.approx(area)
    assert seeded.event_history[-1]["event"] == "INITIAL_CAVITY_SEED_INVENTORY_DEBIT"
    with pytest.raises(ValueError, match="exceeds available"):
        create_subgrid_cavity(replace(state, available_defect_inventory_area_m2=area / 2), "s", 2.0e-6)


def test_full_deterministic_trajectory_preserves_total_defect_inventory():
    _, rows = deterministic_trajectory()
    totals = [row["available_defect_inventory_area_m2"] + row["consumed_defect_inventory_area_m2"]
              for row in rows if row["available_defect_inventory_area_m2"] is not None]
    assert all(total == pytest.approx(totals[0], abs=1.0e-20) for total in totals)


@pytest.mark.parametrize("stage", [
    "downstream_phase_update",
    "downstream_length_ledger_update:ordinary_crack_fractured_length_m",
    "downstream_length_ledger_update:projected_front_advance_m",
    "downstream_event_history_update",
])
def test_downstream_state_updates_rollback_atomically(stage):
    accepted, _ = deterministic_trajectory(stop_before_ligament=True)
    accepted, _ = ligament_transaction(accepted)
    from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
    before = complete_accepted_state_fingerprint(accepted)
    with pytest.raises(RuntimeError, match="injected:" + stage):
        downstream_front_transaction(accepted, failure_stage=stage)
    assert complete_accepted_state_fingerprint(accepted) == before


@pytest.mark.parametrize("threshold_offset,expected_winners", [(5.0e-14, 2), (2.0e-13, 1)])
def test_first_passage_near_tie_uses_frozen_action_tolerance(threshold_offset, expected_winners):
    state, _ = build_production_void_state()
    candidates = tungsten_cleavage_candidates(theta_deg=45.0)
    hazards = [DirectionalHazardState(item.candidate_id) for item in candidates]
    hazards[1] = replace(hazards[1], current_threshold_action=1.0 + threshold_offset)
    state = replace(state, competition=DirectionalCompetitionState(
        candidates=candidates, hazard_states=tuple(hazards), global_hazard_seed=3621,
    ))
    advanced, audit = _complete_next_clock(state, np.eye(2) * 1.0e9)
    assert sum(row["winner"] for row in audit) == expected_winners
    emitted = {row["candidate_id"] for row in audit if row["emitted_event_ids"]}
    assert {event.candidate_id for event in advanced.competition.pending_events} == emitted


def _two_horizontal_candidate_competition():
    candidates = tuple(CleavageCandidate.create(
        plane_family="cleavage", plane_variant=variant,
        direction_xy=(1.0, 0.0), normal_xy=(0.0, 1.0), gamma_rel=1.0,
        orientation_convention="V5 exact-tie single-arm test",
    ) for variant in ("tie-a", "tie-b"))
    return DirectionalCompetitionState(
        candidates=candidates,
        hazard_states=tuple(DirectionalHazardState(candidate.candidate_id) for candidate in candidates),
        global_hazard_seed=3621,
    )


def test_exact_tie_ligament_transaction_selects_one_arm_and_preserves_peer():
    accepted, _ = deterministic_trajectory(stop_before_ligament=True)
    accepted = replace(accepted, competition=_two_horizontal_candidate_competition())
    connected, _ = ligament_transaction(accepted)
    assert len(connected.competition.pending_events) == 1
    assert len(connected.competition.consumed_event_ids) == 1
    assert connected.void_state.cavities[0].phase == VoidPhase.CONNECTED_VOID


def test_exact_tie_downstream_transaction_selects_one_arm_and_preserves_peer():
    accepted, _ = deterministic_trajectory(stop_before_ligament=True)
    accepted, _ = ligament_transaction(accepted)
    accepted = replace(accepted, competition=_two_horizontal_candidate_competition())
    downstream, _, _, causal = downstream_front_transaction(accepted)
    assert len(causal["emitted_winner_candidate_ids"]) == 2
    assert len(causal["selected_proposal_candidate_ids"]) == 1
    assert len(downstream.competition.pending_events) == 1
    assert len(downstream.competition.consumed_event_ids) == 1


def test_offset_cavity_ligament_transaction_preserves_combined_topology():
    accepted, _ = deterministic_trajectory(
        stop_before_ligament=True, cavity_center_m=(7.0e-4, 2.5e-5)
    )
    connected, result = ligament_transaction(accepted)
    assert result.accepted
    assert connected.junction_process_state["latest_crack_void_connection_certificate"]["passed"]
    assert connected.junction_process_state["latest_intersection_alignment"]["accepted_mesh_has_aligned_node"]


def test_oblique_interior_edge_intersection_is_inserted_and_rollback_safe():
    accepted, _ = deterministic_trajectory(stop_before_ligament=True)
    angle = math.radians(10.0)
    candidate = CleavageCandidate.create(
        plane_family="cleavage", plane_variant="oblique-interior-edge",
        direction_xy=(math.cos(angle), math.sin(angle)),
        normal_xy=(-math.sin(angle), math.cos(angle)), gamma_rel=1.0,
        orientation_convention="V5 semantic-hardening qualification",
    )
    competition = DirectionalCompetitionState(
        candidates=(candidate,), hazard_states=(DirectionalHazardState(candidate.candidate_id),),
        global_hazard_seed=3621,
    )
    accepted = replace(accepted, competition=competition)
    from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
    before = complete_accepted_state_fingerprint(accepted)
    with pytest.raises(RuntimeError, match="injected:intersection_alignment"):
        ligament_transaction(accepted, failure_stage="intersection_alignment")
    assert complete_accepted_state_fingerprint(accepted) == before
    connected, result = ligament_transaction(accepted)
    alignment = connected.junction_process_state["latest_intersection_alignment"]
    assert result.accepted and alignment["interior_edge_split_performed"]
    assert alignment["accepted_mesh_has_aligned_node"]
    assert connected.junction_process_state["latest_crack_void_connection_certificate"]["passed"]
