from dataclasses import fields, replace
import math
import subprocess
import sys

import numpy as np
import pytest

from arrhenius_fracture.dislocation_interaction_voiding_v1 import (
    DislocationInteractionParameterRowV1, combined_cavity_growth_rate,
)
from arrhenius_fracture.embryo_growth_v1 import (
    RESIDUAL, RESOLVED, CavityEmissionParameterRowV1, advance_cavity_emission,
    advance_embryo_growth, heal_embryo, initialize_cavity_emission_state,
    initialize_embryo_growth_state, loop_area_kinematics, promote_representation,
    TransportFluxCertificate, transport_flux_certificate,
    cavity_surface_emission_rates, classify_cavity_emission_sources,
    continuum_packet_rate_upper,
)


def test_old_hit_quantum_is_derived_not_a_parameter():
    row = DislocationInteractionParameterRowV1()
    assert "direct_free_volume_per_hit_area_m2" not in {item.name for item in fields(row)}
    assert row.direct_free_volume_per_hit_area_m2 == (
        row.chi_loop*2*math.pi*row.embryo_reference_radius_m*row.burgers_m
    )


def _flux_certificate_with(*, ledger_before, ledger_after, raw, flux):
    return TransportFluxCertificate(
        owner_id="cavity-1", owner_kind="ACTIVE_CAVITY",
        transport_before_fingerprint="before",
        transport_after_fingerprint="after",
        surface_registry_fingerprint="registry",
        capture_ledger_before_area_m2=ledger_before,
        capture_ledger_after_area_m2=ledger_after,
        raw_absorbed_area_m2=raw,
        field_inventory_before_area_m2=1.0e-12,
        field_inventory_after_area_m2=1.0e-12,
        field_inventory_decrement_area_m2=0.0,
        surface_flux_integral_area_m2=flux,
        owner_inventory_increment_area_m2=flux,
        plastic_and_climb_source_area_m2=flux,
        other_sink_integrals_area_m2=0.0,
        conservation_residual_area_m2=0.0,
        duration_s=1.0,
    )


def test_transport_flux_certificate_uses_cumulative_operand_roundoff_budget():
    before = 1.0e-3
    after = np.nextafter(before, math.inf)
    increment = after-before
    # A direct flux differing by one ulp at the cumulative-ledger scale is
    # admissible even though it is large relative to the tiny increment.
    _flux_certificate_with(
        ledger_before=before, ledger_after=after, raw=increment,
        flux=increment+np.spacing(before))
    # A discrepancy beyond the operand-aware subtraction budget is rejected;
    # it is not hidden by a widened increment-relative tolerance.
    with pytest.raises(ValueError, match="surface flux"):
        _flux_certificate_with(
            ledger_before=before, ledger_after=after, raw=increment,
            flux=increment+256.0*np.spacing(before))


def test_loop_kinematics_records_both_interpretations():
    row = CavityEmissionParameterRowV1()
    audit = loop_area_kinematics(row, 2e-9)
    assert audit["delta_volume_loop_m3"] == row.burgers_m*row.loop_swept_area_m2
    assert audit["full_3d_projected_area_m2"] == (
        audit["delta_volume_loop_m3"]/row.model_thickness_m
    )
    assert audit["selected_projection"] == row.loop_projection


def test_vacancy_and_loop_growth_have_distinct_bottlenecks():
    result = combined_cavity_growth_rate(
        vacancy_supply_rate_area_m2_s=1e-12,
        direct_supply_rate_area_m2_s=2e-12, radius_m=1e-6,
        equivalent_plastic_strain_rate_s=0.0,
        plastic_accommodation_coefficient=1.0,
    )
    assert result["vacancy_accommodated_rate_area_m2_s"] == 0.0
    assert result["dislocation_direct_growth_rate_area_m2_s"] == 2e-12
    assert result["total_growth_rate_area_m2_s"] == 2e-12


def test_exact_emission_action_is_partition_invariant():
    row = replace(CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,))
    initial = initialize_cavity_emission_state(row=row, seed=91)
    one, _ = advance_cavity_emission(
        initial, radius_m=1e-8, rates_s=(0.7,), duration_s=10.0,
        forced_regime="LOW_COUNT_EXACT",
    )
    split = initial
    for _ in range(10):
        split, _ = advance_cavity_emission(
            split, radius_m=1e-8, rates_s=(0.7,), duration_s=1.0,
            forced_regime="LOW_COUNT_EXACT",
        )
    assert one.sources[0].integrated_action_exact == split.sources[0].integrated_action_exact
    assert one.cumulative_loop_count_effective == split.cumulative_loop_count_effective
    assert one.sources[0].next_exact_threshold == split.sources[0].next_exact_threshold


def test_regime_switches_share_one_consumed_threshold_measure():
    row = replace(CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,))
    for regimes in (("LOW_COUNT_EXACT", "TAU_LEAP", "LOW_COUNT_EXACT"),
                    ("LOW_COUNT_EXACT", "CONTINUUM", "LOW_COUNT_EXACT"),
                    ("TAU_LEAP", "CONTINUUM", "TAU_LEAP")):
        state = initialize_cavity_emission_state(row=row, seed=712)
        for regime in regimes:
            state, audit = advance_cavity_emission(
                state, radius_m=1e-8, rates_s=(1.7,), duration_s=1.0,
                forced_regime=regime)
            source = state.sources[0]
            assert source.emitted_loop_count_effective == pytest.approx(
                source.realized_loop_count+source.continuum_moment_loop_count)
            assert audit["one_underlying_counting_process"] is True
        all_exact = initialize_cavity_emission_state(row=row, seed=712)
        all_exact, _ = advance_cavity_emission(
            all_exact, radius_m=1e-8, rates_s=(1.7,), duration_s=3.0,
            forced_regime="LOW_COUNT_EXACT")
        assert state.sources[0].random_measure_cursor_count == \
            all_exact.sources[0].random_measure_cursor_count
        assert state.sources[0].next_exact_threshold == \
            all_exact.sources[0].next_exact_threshold


def test_continuum_mu_30_does_not_enumerate_thresholds_and_has_one_interpretation():
    row = replace(CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,))
    initial = initialize_cavity_emission_state(row=row, seed=123)
    advanced, audit = advance_cavity_emission(
        initial, radius_m=1e-8, rates_s=(30.0,), duration_s=1.0,
        forced_regime="CONTINUUM")
    source = advanced.sources[0]
    assert audit["continuum_enumerated_thresholds"] == 0
    assert source.realized_loop_count == 0
    assert source.continuum_moment_loop_count == pytest.approx(
        audit["expected_counts"][0])
    assert 0.0 < source.emitted_loop_count_effective < 30.0


def test_packet_rate_bound_uses_owned_availability_and_recovery_equilibrium():
    row = replace(
        CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,),
        recovery_rate_s=1.0e-3, depletion_per_loop=0.02)
    state = initialize_cavity_emission_state(row=row, seed=123)
    depleted = replace(
        state, sources=(replace(state.sources[0], available_fraction=0.005),))
    rate = 1.0e6
    equilibrium = row.recovery_rate_s/(row.recovery_rate_s
                                       +row.depletion_per_loop*rate)
    assert continuum_packet_rate_upper(depleted, (rate,), (False,))[0] == \
        pytest.approx(rate*max(0.005, equilibrium))
    exhausted = replace(
        state, sources=(replace(state.sources[0], available_fraction=0.0),))
    assert continuum_packet_rate_upper(exhausted, (rate,), (False,))[0] == \
        pytest.approx(rate*equilibrium)
    assert continuum_packet_rate_upper(depleted, (rate,), (True,))[0] == 0.0


def test_checkpoint_manifest_serializer_tags_legitimate_infinite_diagnostic():
    import json
    from arrhenius_fracture.voiding_v5 import serialize

    emission = initialize_cavity_emission_state(enabled=True)
    observed = replace(emission, last_audit={"crossing_time_s": math.inf})
    payload = json.loads(serialize(observed))
    assert payload["last_audit"]["crossing_time_s"] == {
        "__nonfinite_float__": "positive_infinity"
    }


@pytest.mark.parametrize("regime,rate,duration", [
    ("LOW_COUNT_EXACT", 0.01, 16.0),
    ("CONTINUUM", 30.0, 1.0),
])
def test_one_step_equals_2_4_8_16_substeps(regime, rate, duration):
    row = replace(CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,))
    initial = initialize_cavity_emission_state(row=row, seed=123)
    one, _ = advance_cavity_emission(
        initial, radius_m=1e-8, rates_s=(rate,), duration_s=duration,
        forced_regime=regime)
    for subdivisions in (2, 4, 8, 16):
        split = initial
        for _ in range(subdivisions):
            split, _ = advance_cavity_emission(
                split, radius_m=1e-8, rates_s=(rate,),
                duration_s=duration/subdivisions, forced_regime=regime)
        assert split.fingerprint == one.fingerprint
        assert split.sources[0].random_measure_cursor_count == \
            one.sources[0].random_measure_cursor_count


def test_tau_compatibility_path_is_exact_random_time_change():
    row = replace(CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,))
    initial = initialize_cavity_emission_state(row=row, seed=123)
    exact_state, _ = advance_cavity_emission(
        initial, radius_m=1e-8, rates_s=(1.0,), duration_s=16.0,
        forced_regime="LOW_COUNT_EXACT")
    for subdivisions in (1, 2, 8, 32, 64):
        tau = initial
        for _ in range(subdivisions):
            tau, _ = advance_cavity_emission(
                tau, radius_m=1e-8, rates_s=(1.0,),
                duration_s=16.0/subdivisions, forced_regime="TAU_LEAP")
        assert tau.fingerprint == exact_state.fingerprint


def test_availability_dependent_production_rate_is_partition_invariant():
    row = replace(CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,),
                  recovery_rate_s=0.1)
    growth = initialize_embryo_growth_state(
        site_id="s", radius_m=1e-8, row=row, emission_enabled=True)
    stress = np.array(((8e9, 0.0), (0.0, 2e9)))
    base, _ = cavity_surface_emission_rates(
        growth, stress_tensor_Pa=stress, sigma_zz_Pa=1e9,
        temperature_K=1200.0)
    one, _ = advance_cavity_emission(
        growth.cavity_emission_state, radius_m=growth.radius_m,
        rates_s=base, duration_s=1.0, forced_regime="CONTINUUM")
    half, _ = advance_cavity_emission(
        growth.cavity_emission_state, radius_m=growth.radius_m,
        rates_s=base, duration_s=0.5, forced_regime="CONTINUUM")
    half_growth = replace(growth, cavity_emission_state=half)
    recomputed_base, audit = cavity_surface_emission_rates(
        half_growth, stress_tensor_Pa=stress, sigma_zz_Pa=1e9,
        temperature_K=1200.0)
    assert audit["returned_rate_kind"] == "UNDEPLETED_BASE_INTENSITY_LAMBDA_0"
    assert np.array_equal(recomputed_base, base)
    split, _ = advance_cavity_emission(
        half, radius_m=growth.radius_m, rates_s=recomputed_base,
        duration_s=0.5, forced_regime="CONTINUUM")
    assert split.fingerprint == one.fingerprint


def test_high_rate_exact_epoch_switches_without_enumerating_microscopic_events():
    row = replace(CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,))
    state = initialize_cavity_emission_state(row=row, seed=123)
    source = replace(state.sources[0], integrated_action=10.0,
                     integrated_action_exact=(10, 1), next_exact_threshold=10.1,
                     availability_flow_regime="LOW_COUNT_EXACT")
    state = replace(state, sources=(source,))
    regimes, discrete, _ = classify_cavity_emission_sources(
        state, base_rates_s=(10.0,), duration_s=1.0)
    assert regimes == ("CONTINUUM",)
    assert discrete == (False,)
    advanced, audit = advance_cavity_emission(
        state, radius_m=1e-8, rates_s=(10.0,), duration_s=100.0)
    assert audit["continuum_enumerated_thresholds"] == 0
    assert advanced.sources[0].realized_loop_count == source.realized_loop_count
    assert advanced.sources[0].continuum_moment_loop_count > 0.0


def test_automatic_hybrid_partition_convergence_and_no_recount():
    row = replace(CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,))
    initial = initialize_cavity_emission_state(row=row, seed=123)
    one, _ = advance_cavity_emission(
        initial, radius_m=1e-8, rates_s=(30.0,), duration_s=1.0)
    for subdivisions in (2, 4, 8, 16):
        split = initial
        for _ in range(subdivisions):
            split, _ = advance_cavity_emission(
                split, radius_m=1e-8, rates_s=(30.0,),
                duration_s=1.0/subdivisions)
        assert split.fingerprint == one.fingerprint
        assert split.sources[0].random_measure_cursor_count == \
            one.sources[0].random_measure_cursor_count


def test_default_low_rate_sources_retain_discrete_event_capability():
    row = replace(CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,))
    initial = initialize_cavity_emission_state(row=row, seed=84731)
    regimes, discrete, _ = classify_cavity_emission_sources(
        initial, base_rates_s=(1.0e-5,), duration_s=1.0)
    assert regimes == ("LOW_COUNT_EXACT",)
    assert discrete == (True,)


def test_exact_to_continuum_and_continuum_to_exact_preserve_measure_cursor():
    row = replace(CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,))
    initial = initialize_cavity_emission_state(row=row, seed=84731)
    low, _ = advance_cavity_emission(
        initial, radius_m=1e-8, rates_s=(1.0e-5,), duration_s=1.0)
    assert low.sources[0].availability_flow_regime == "LOW_COUNT_EXACT"
    high, _ = advance_cavity_emission(
        low, radius_m=1e-8, rates_s=(30.0,), duration_s=0.1)
    assert high.sources[0].availability_flow_regime == "CONTINUUM"
    assert high.sources[0].regime_epoch == 1
    assert high.sources[0].integrated_action > low.sources[0].integrated_action
    realized_before = high.sources[0].realized_loop_count
    cursor_before = high.sources[0].random_measure_cursor_count
    returned, _ = advance_cavity_emission(
        high, radius_m=1e-8, rates_s=(1.0e-5,), duration_s=1.0)
    assert returned.sources[0].availability_flow_regime == "LOW_COUNT_EXACT"
    assert returned.sources[0].regime_epoch == 2
    assert returned.sources[0].realized_loop_count == realized_before
    assert returned.sources[0].random_measure_cursor_count >= cursor_before
    assert returned.sources[0].next_exact_threshold > \
        returned.sources[0].integrated_action


def test_regime_switch_1_2_4_8_16_partition_invariance():
    row = replace(CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,))
    initial = initialize_cavity_emission_state(row=row, seed=84731)

    one = initial
    for rate, duration in ((1.0e-5, 1.0), (30.0, 0.1), (1.0e-5, 1.0)):
        one, _ = advance_cavity_emission(
            one, radius_m=1e-8, rates_s=(rate,), duration_s=duration)
    for subdivisions in (2, 4, 8, 16):
        split = initial
        for rate, duration in ((1.0e-5, 1.0), (30.0, 0.1), (1.0e-5, 1.0)):
            for _ in range(subdivisions):
                split, _ = advance_cavity_emission(
                    split, radius_m=1e-8, rates_s=(rate,),
                    duration_s=duration/subdivisions)
        assert split.fingerprint == one.fingerprint


def test_nonzero_exact_history_continuum_exact_partition_closure():
    row = replace(CavityEmissionParameterRowV1(), source_orientations_rad=(0.0,))
    initial = initialize_cavity_emission_state(row=row, seed=1)

    def evolve(subdivisions):
        state = initial
        for rate, duration in ((0.01, 100.0), (30.0, 0.1), (1.0e-5, 1.0)):
            for _ in range(subdivisions):
                state, _ = advance_cavity_emission(
                    state, radius_m=1e-8, rates_s=(rate,),
                    duration_s=duration/subdivisions)
        return state

    reference = evolve(1)
    source = reference.sources[0]
    assert source.realized_loop_count >= 1
    assert source.continuum_moment_loop_count > 0.0
    assert source.availability_flow_regime == "LOW_COUNT_EXACT"
    assert source.emitted_loop_count_effective == (
        float(source.realized_loop_count)+source.continuum_moment_loop_count)
    assert reference.cumulative_loop_count_effective == math.fsum(
        item.emitted_loop_count_effective for item in reference.sources)
    for subdivisions in (2, 4, 8, 16):
        split = evolve(subdivisions)
        split_source = split.sources[0]
        assert split_source.realized_loop_count == source.realized_loop_count
        assert split_source.continuum_moment_loop_count == \
            source.continuum_moment_loop_count
        assert split_source.emitted_loop_count_effective == \
            source.emitted_loop_count_effective
        assert split_source.integrated_action_exact == source.integrated_action_exact
        assert split_source.random_measure_cursor_count == \
            source.random_measure_cursor_count
        assert split.cumulative_area_m2 == reference.cumulative_area_m2
        assert split.cumulative_emitted_burgers_m == \
            reference.cumulative_emitted_burgers_m
        assert split.fingerprint == reference.fingerprint


def test_exact_winner_with_continuum_loser_advances_both_once():
    row = replace(
        CavityEmissionParameterRowV1(), source_orientations_rad=(0.0, math.pi/2))
    state = initialize_cavity_emission_state(row=row, seed=123)
    winner, loser = state.sources
    action = winner.next_exact_threshold-0.1
    winner = replace(
        winner, integrated_action=action, integrated_action_exact=None,
        availability_reference_action_exact=(0, 1),
        availability_elapsed_time_exact=(0, 1),
        availability_flow_regime=None)
    state = replace(state, sources=(winner, loser))
    advanced, _ = advance_cavity_emission(
        state, radius_m=1e-8, rates_s=(1.0, 300.0), duration_s=0.1,
        forced_regimes_by_source={
            winner.source_id: "LOW_COUNT_EXACT",
            loser.source_id: "CONTINUUM",
        })
    assert advanced.sources[0].realized_loop_count-winner.realized_loop_count == 1
    assert advanced.sources[1].realized_loop_count == loser.realized_loop_count
    assert advanced.sources[1].continuum_moment_loop_count > 0.0


def test_stabilization_waiting_state_continues_healing_then_stabilizes():
    from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
    from arrhenius_fracture.voiding_production_v5 import build_production_void_state
    from arrhenius_fracture.voiding_v5 import (
        HazardClock, VoidPhase, advance_site,
    )
    live, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=900.0)
    site = replace(
        live.void_state.sites[0], phase=VoidPhase.EMBRYO,
        stabilization=HazardClock(0.0, 1.0),
        healing=HazardClock(0.0, 10.0))
    growth = initialize_embryo_growth_state(
        site_id=site.site_id, radius_m=1.0e-10, emission_enabled=False)
    voids = replace(
        live.void_state, sites=(site,), embryo_growth_states=(growth,))
    waiting, events = advance_site(
        voids, site.site_id, 1.0,
        rates={"stabilization_s": 1.0, "healing_s": 0.1})
    waiting_growth = waiting.embryo_growth_states[0]
    waiting_site = waiting.sites[0]
    assert events == ("STABILIZATION_RADIUS_NOT_REACHED",)
    assert waiting_growth.stabilization_action_complete_waiting_for_radius
    assert waiting_site.phase == VoidPhase.EMBRYO
    healing_before = waiting_site.healing.accumulated
    continued, events = advance_site(
        waiting, site.site_id, 2.0,
        rates={"stabilization_s": 1.0, "healing_s": 0.1})
    assert not events
    assert continued.sites[0].healing.accumulated > healing_before
    stable_radius = growth.cavity_emission_state.parameter_row.minimum_stable_radius_m
    reached = replace(continued, embryo_growth_states=(replace(
        continued.embryo_growth_states[0], radius_m=stable_radius,
        area_m2=math.pi*stable_radius**2,
        initial_area_m2=math.pi*stable_radius**2),))
    stabilized, events = advance_site(
        reached, site.site_id, 1.0e-9,
        rates={"stabilization_s": 1.0, "healing_s": 0.1})
    assert events == ("STABILIZED",)
    assert stabilized.sites[0].phase == VoidPhase.STABLE_SUBGRID_VOID
    assert not stabilized.embryo_growth_states[0].stabilization_action_complete_waiting_for_radius


def test_resolved_sigma_tt_rejects_failed_boundary_traction():
    from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
    from arrhenius_fracture.voiding_production_v5 import (
        deterministic_trajectory, equilibrate_fixed_load_with_production_fem,
        ligament_transaction,
    )
    from arrhenius_fracture.embryo_growth_v1 import RESOLVED, source_local_fem_tensors
    state, _ = deterministic_trajectory(
        bundle=material_bundle("DBTT"), stop_before_ligament=True)
    state = equilibrate_fixed_load_with_production_fem(
        replace(state, displacement=2.0*state.displacement))
    state, result = ligament_transaction(state)
    assert result.accepted
    cavity, site = state.void_state.cavities[0], state.void_state.sites[0]
    growth = initialize_embryo_growth_state(
        site_id=site.site_id, radius_m=cavity.radius_m,
        initial_area_m2=cavity.area_m2, representation=RESOLVED,
        cavity_id=cavity.cavity_id, emission_enabled=True)
    state = replace(state, void_state=replace(
        state.void_state, embryo_growth_states=(growth,)))
    with pytest.raises(RuntimeError, match="failed qualification"):
        source_local_fem_tensors(state, growth)


def test_two_level_resolved_sigma_tt_certificate_is_typed_and_fail_closed():
    from arrhenius_fracture.resolved_sigma_tt_certificate_v1 import (
        ResolvedSigmaTTTwoLevelCertificateV1,
        build_resolved_sigma_tt_two_level_certificate,
    )
    from arrhenius_fracture.topology_transaction_v11 import (
        complete_accepted_state_fingerprint,
    )
    from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
    from arrhenius_fracture.voiding_production_v5 import deterministic_trajectory

    bundle = material_bundle("DBTT")
    coarse, _ = deterministic_trajectory(
        bundle=bundle, stop_before_ligament=True,
        boundary_segments=32, radial_layers=12)
    crack_path = coarse.crack_network.branches[0].path
    fine, _ = deterministic_trajectory(
        bundle=bundle, stop_before_ligament=True,
        boundary_segments=64, radial_layers=24, crack_path_m=crack_path)

    def bind_growth(state):
        cavity = state.void_state.cavities[0]
        growth = initialize_embryo_growth_state(
            site_id=cavity.parent_site_id, radius_m=cavity.radius_m,
            initial_area_m2=cavity.area_m2, representation=RESOLVED,
            cavity_id=cavity.cavity_id, emission_enabled=True)
        return replace(state, void_state=replace(
            state.void_state, embryo_growth_states=(growth,),
            embryo_growth_configuration=replace(
                state.void_state.embryo_growth_configuration,
                cavity_surface_loop_emission_enabled=True))), growth

    coarse, coarse_growth = bind_growth(coarse)
    fine, fine_growth = bind_growth(fine)
    before = (complete_accepted_state_fingerprint(coarse),
              complete_accepted_state_fingerprint(fine))
    certificate = build_resolved_sigma_tt_two_level_certificate(
        coarse, coarse_growth, fine, fine_growth, temperature_K=900.0)
    assert isinstance(certificate, ResolvedSigmaTTTwoLevelCertificateV1)
    # Independently generated similar meshes are not an authenticated nested
    # comparison. Scientific predicates must not be evaluated on invalid data.
    assert certificate.terminal_scientific_availability == "INVALID_COMPARISON"
    assert not certificate.comparison_prerequisites["evidence_present"]
    assert certificate.scientific_predicates == {}
    assert (complete_accepted_state_fingerprint(coarse),
            complete_accepted_state_fingerprint(fine)) == before


def test_healing_retains_emitted_loop_defect_and_promotion_is_physical_noop():
    growth = initialize_embryo_growth_state(
        site_id="s", radius_m=1e-8, emission_enabled=True,
    )
    emission, emission_audit = advance_cavity_emission(
        growth.cavity_emission_state, radius_m=growth.radius_m,
        rates_s=(3e11, 0.0, 0.0), duration_s=1e-10,
        forced_regime="CONTINUUM")
    emitted = replace(
        growth, cavity_emission_state=emission,
        emitted_loop_count_effective=emission.cumulative_loop_count_effective,
        dislocation_derived_area_m2=emission_audit["area_increment_m2"],
        area_m2=growth.area_m2+emission_audit["area_increment_m2"],
        radius_m=math.sqrt((growth.area_m2+emission_audit["area_increment_m2"])/math.pi))
    residual, audit = heal_embryo(emitted)
    assert residual.representation == RESIDUAL
    assert audit["pristine_available_site_allowed"] is False
    promoted = promote_representation(emitted, "void:s")
    assert promoted.representation == RESOLVED
    assert replace(
        promoted, representation=emitted.representation, cavity_id=None,
        last_remesh_radius_m=emitted.last_remesh_radius_m,
    ) == emitted


def test_terminal_embryo_to_large_void_qualification():
    completed = subprocess.run(
        [sys.executable, "scripts/qualify_embryo_to_large_void_growth.py"],
        check=True, capture_output=True, text=True,
    )
    required = (
        "EMBRYO_GROWTH_STATE = PASS",
        "VACANCY_ABSORPTION_GROWTH = PASS",
        "CAVITY_DISLOCATION_EMISSION_GROWTH = PASS",
        "BURGERS_TO_CAVITY_AREA_KINEMATICS = PASS",
        "EMITTED_DISLOCATION_MECHANICAL_FEEDBACK = PASS",
        "HYBRID_EVENT_CONTINUUM_SCALING = PASS",
        "HEALING_AND_STABILIZATION_CONSERVATION = PASS",
        "SUBGRID_TO_RESOLVED_CONTINUITY = PASS",
        "ADAPTIVE_REMESH_CONSERVATION = PASS",
        "GLOBAL_GROWTH_PACKET_SCHEDULER = PASS",
        "RESOLVED_SOURCE_BOUNDARY_TRACTION = PASS_FAIL_CLOSED",
        "RESOLVED_SIGMA_TT_TWO_LEVEL_CONVERGENCE = BLOCKED",
        "NONLINEAR_RADIUS_OVERSHOOT_REPLAY = PASS",
        "GENERAL_NONLINEAR_RADIUS_ROOT_LOCALIZATION = NOT_YET_QUALIFIED",
        "RAMP_ACCUMULATED_FIRST_PASSAGE_ACTION = IMPLEMENTED_PENDING_ANALYTIC_EVENT_QUALIFICATION",
        "EMBRYO_TO_LARGE_VOID_FRAMEWORK = PARTIAL_PRODUCTION_INTEGRATION",
    )
    assert all(item in completed.stdout for item in required)
