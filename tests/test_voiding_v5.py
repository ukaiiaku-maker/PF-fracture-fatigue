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


def test_true_fixed_crack_positive_negative_offset_pair_is_mirrored_and_atomic():
    from arrhenius_fracture.voiding_production_v5 import crack_tip_tensor
    from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
    fixed_path = ((0.0, 0.0), (0.0005725993004046688, 0.0))
    results = []
    for offset in (1.0e-5, -1.0e-5):
        accepted, rows = deterministic_trajectory(
            stop_before_ligament=True, cavity_center_m=(7.0e-4, offset),
            crack_path_m=fixed_path,
        )
        before = complete_accepted_state_fingerprint(accepted)
        with pytest.raises(RuntimeError, match="injected:intersection_alignment"):
            ligament_transaction(accepted, failure_stage="intersection_alignment")
        assert complete_accepted_state_fingerprint(accepted) == before
        tensor, _ = crack_tip_tensor(accepted)
        connected, result = ligament_transaction(accepted)
        results.append((rows[-1], tensor, connected, result))
    positive, negative = results
    assert positive[3].accepted and negative[3].accepted
    assert positive[2].junction_process_state["latest_crack_void_connection_certificate"]["passed"]
    assert negative[2].junction_process_state["latest_crack_void_connection_certificate"]["passed"]
    positive_tip = positive[2].crack_network.branch("b00000000").tip
    negative_tip = negative[2].crack_network.branch("b00000000").tip
    assert positive_tip[0] == pytest.approx(negative_tip[0], abs=1.0e-12)
    assert positive_tip[1] == pytest.approx(-negative_tip[1], abs=1.0e-12)
    assert positive[0]["reaction_N_per_m"] == pytest.approx(negative[0]["reaction_N_per_m"], rel=5.0e-3)
    assert positive[0]["compliance_m2_per_N"] == pytest.approx(negative[0]["compliance_m2_per_N"], rel=5.0e-3)
    centered, _ = deterministic_trajectory(
        stop_before_ligament=True, cavity_center_m=(7.0e-4, 0.0), crack_path_m=fixed_path,
    )
    centered_tensor, _ = crack_tip_tensor(centered)
    assert positive[1][0, 0] == pytest.approx(negative[1][0, 0], rel=1.0e-2)
    assert positive[1][1, 1] == pytest.approx(negative[1][1, 1], rel=1.0e-2)
    # The bottom-corner rigid-body pins create a small baseline shear.  The
    # offset-induced shear increment, rather than the biased raw component, is
    # the reflection-odd observable.
    positive_shear_increment = positive[1][0, 1] - centered_tensor[0, 1]
    negative_shear_increment = negative[1][0, 1] - centered_tensor[0, 1]
    assert positive_shear_increment == pytest.approx(-negative_shear_increment, rel=2.0e-2)


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
    cavity = connected.void_state.cavities[0]
    ledger = connected.void_state.length_ledgers
    physical_chord = math.dist(cavity.connection_entry_m, cavity.connection_exit_m)
    projected_chord = cavity.connection_exit_m[0] - cavity.connection_entry_m[0]
    ligament = math.dist(accepted.crack_network.branch("b00000000").tip,
                         cavity.connection_entry_m)
    assert ledger["connected_void_free_span_m"] == pytest.approx(physical_chord)
    assert ledger["projected_connected_void_free_span_m"] == pytest.approx(projected_chord)
    assert ledger["fractured_ligament_length_m"] == pytest.approx(ligament)
    assert ledger["physical_active_front_travel_m"] == pytest.approx(ligament)
    assert ledger["projected_fractured_length_m"] == pytest.approx(
        cavity.connection_entry_m[0] - accepted.crack_network.branch("b00000000").tip[0]
    )
    assert physical_chord > projected_chord
    assert physical_chord != pytest.approx(2.0 * cavity.radius_m)


@pytest.mark.parametrize("stage", ["root_status_change", "dormant_support_rebuild"])
def test_connected_dormant_ownership_rolls_back(stage):
    accepted, _ = deterministic_trajectory(stop_before_ligament=True)
    from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
    before = complete_accepted_state_fingerprint(accepted)
    with pytest.raises(RuntimeError, match="injected:" + stage):
        ligament_transaction(accepted, failure_stage=stage)
    assert complete_accepted_state_fingerprint(accepted) == before


def test_connected_and_downstream_states_own_zero_then_one_active_front(tmp_path):
    from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
    from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
    accepted, _ = deterministic_trajectory(stop_before_ligament=True)
    connected, _ = ligament_transaction(accepted)
    assert connected.crack_network.active_tip_ids == ()
    assert connected.v12_support_state.active_tip_identities == ()
    connected_path = tmp_path / "connected.json"
    write_checkpoint(connected, connected_path)
    assert complete_accepted_state_fingerprint(restore_checkpoint(connected_path)) == complete_accepted_state_fingerprint(connected)

    downstream, _, _, _ = downstream_front_transaction(connected)
    assert downstream.crack_network.branch("b00000000").status == "arrested"
    assert downstream.crack_network.active_tip_ids == ("void-front-1",)
    assert downstream.v12_support_state.active_tip_identities == ("void-front-1",)
    downstream_path = tmp_path / "downstream.json"
    write_checkpoint(downstream, downstream_path)
    assert complete_accepted_state_fingerprint(restore_checkpoint(downstream_path)) == complete_accepted_state_fingerprint(downstream)


def test_downstream_child_activation_rolls_back():
    accepted, _ = deterministic_trajectory(stop_before_ligament=True)
    accepted, _ = ligament_transaction(accepted)
    from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
    before = complete_accepted_state_fingerprint(accepted)
    with pytest.raises(RuntimeError, match="injected:downstream_child_activation"):
        downstream_front_transaction(accepted, failure_stage="downstream_child_activation")
    assert complete_accepted_state_fingerprint(accepted) == before


def test_distinct_direction_tie_selects_intersecting_ligament_and_retains_miss():
    from arrhenius_fracture.voiding_production_v5 import crack_tip_tensor
    accepted, _ = deterministic_trajectory(stop_before_ligament=True)
    candidates = (
        CleavageCandidate.create(plane_family="cleavage", plane_variant="hits-cavity",
                                 direction_xy=(1.0, 0.0), normal_xy=(0.0, 1.0), gamma_rel=1.0,
                                 orientation_convention="V5 distinct-direction tie"),
        CleavageCandidate.create(plane_family="cleavage", plane_variant="misses-cavity",
                                 direction_xy=(0.0, 1.0), normal_xy=(1.0, 0.0), gamma_rel=1.0,
                                 orientation_convention="V5 distinct-direction tie"),
    )
    initial = DirectionalCompetitionState(
        candidates=candidates,
        hazard_states=tuple(DirectionalHazardState(candidate.candidate_id) for candidate in candidates),
        global_hazard_seed=3621,
    )
    probe = replace(accepted, competition=initial)
    _, audit = _complete_next_clock(probe, crack_tip_tensor(probe)[0])
    rates = {row["candidate_id"]: row["rate_s"] for row in audit}
    assert all(rate > 0.0 for rate in rates.values())
    tied = replace(initial, hazard_states=tuple(
        replace(hazard, current_threshold_action=rates[hazard.candidate_id])
        for hazard in initial.hazard_states
    ))
    connected, _ = ligament_transaction(replace(accepted, competition=tied))
    consumed_candidates = {event_id.split("#event:")[0] for event_id in connected.competition.consumed_event_ids}
    pending_candidates = {event.candidate_id for event in connected.competition.pending_events}
    assert consumed_candidates == {candidates[0].candidate_id}
    assert pending_candidates == {candidates[1].candidate_id}


def test_combined_certificate_rejects_wrong_identity_and_stale_support():
    from arrhenius_fracture.voiding_production_v5 import crack_void_connection_certificate
    accepted, _ = deterministic_trajectory(stop_before_ligament=True)
    connected, _ = ligament_transaction(accepted)
    endpoint = connected.crack_network.branch("b00000000").tip
    with pytest.raises(ValueError, match="exact cavity identity"):
        crack_void_connection_certificate(connected, branch_id="b00000000",
                                          cavity_id="wrong-cavity", intended_intersection=endpoint)
    stale = replace(connected, v12_support_state=replace(
        connected.v12_support_state, selected_support_elements=()
    ))
    certificate = crack_void_connection_certificate(
        stale, branch_id="b00000000", cavity_id=connected.void_state.cavities[0].cavity_id,
        intended_intersection=endpoint,
    )
    assert certificate["no_surviving_solid_ligament_bridge"] is False
    assert certificate["passed"] is False


def test_combined_certificate_rejects_segment_through_cavity_and_broken_cycle():
    from arrhenius_fracture.voiding_production_v5 import crack_void_connection_certificate, cavity_free_surface_certificate
    accepted, _ = deterministic_trajectory(stop_before_ligament=True)
    connected, _ = ligament_transaction(accepted)
    root = connected.crack_network.branch("b00000000")
    center = connected.void_state.cavities[0].center_m
    crossing = replace(root, path=root.path + (center,),
                       orientation_history_rad=root.orientation_history_rad + (0.0,))
    crossing_state = replace(connected, crack_network=replace(
        connected.crack_network,
        branches=tuple(crossing if branch.branch_id == root.branch_id else branch
                       for branch in connected.crack_network.branches),
    ))
    certificate = crack_void_connection_certificate(
        crossing_state, branch_id=root.branch_id,
        cavity_id=connected.void_state.cavities[0].cavity_id,
        intended_intersection=center,
    )
    assert any(row["intersects_cavity_open_disk"] for row in certificate["crack_segment_cavity_intersections"])
    assert certificate["passed"] is False
    # Removing one cavity-adjacent boundary triangle breaks the degree-two cycle.
    boundary_element = cavity_free_surface_certificate(connected)["boundary_edge_ids"][0]
    owner = next(index for index, triangle in enumerate(connected.mesh.elems)
                 if set(boundary_element).issubset(set(map(int, triangle))))
    broken_mesh = replace(connected.mesh, elems=np.delete(connected.mesh.elems, owner, axis=0))
    assert cavity_free_surface_certificate(replace(connected, mesh=broken_mesh))["passed"] is False


def test_combined_certificate_detects_support_triangle_overlap_with_centroid_outside():
    from arrhenius_fracture.mesh import rebuild_tri_mesh
    from arrhenius_fracture.voiding_production_v5 import crack_void_connection_certificate
    accepted, _ = deterministic_trajectory(stop_before_ligament=True)
    connected, _ = ligament_transaction(accepted)
    cavity = connected.void_state.cavities[0]
    center = np.asarray(cavity.center_m)
    radius = cavity.radius_m
    adversarial = center + radius * np.asarray(((0.5, 0.0), (2.0, 0.1), (2.0, -0.1)))
    first = connected.mesh.nn
    mesh = rebuild_tri_mesh(
        np.vstack((connected.mesh.nodes, adversarial)),
        np.vstack((connected.mesh.elems, (first, first + 1, first + 2))),
        tip_centers=np.asarray(cavity.center_m),
    )
    support = replace(connected.v12_support_state,
                      selected_support_elements=(mesh.ne - 1,))
    altered = replace(connected, mesh=mesh, v12_support_state=support)
    certificate = crack_void_connection_certificate(
        altered, branch_id="b00000000", cavity_id=cavity.cavity_id,
        intended_intersection=connected.crack_network.branch("b00000000").tip,
    )
    centroid = adversarial.mean(axis=0)
    assert np.linalg.norm(centroid - center) > radius
    assert certificate["support_triangle_cavity_overlap_element_ids"] == [mesh.ne - 1]
    assert certificate["wake_support_outside_cavity"] is False
    assert certificate["passed"] is False
