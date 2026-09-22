from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from arrhenius_fracture.adaptive_multitip_mesh_v11 import refine_accepted_state
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.unified_fracture_material_v5 import all_material_bundles, material_bundle
from arrhenius_fracture.vacancy_coupling_v1 import (
    COMMON_VACANCY_ROW_ID, REFERENCE_VACANCY_ROW, advance_accepted_vacancy_state,
    ENGINEERING_SHEAR_VOIGT_CONVENTION, analytic_vacancy_field_and_integral,
    conservation_roundoff_budget_m2,
    debit_local_vacancy_inventory, deterministic_capture_patch,
    equilibrium_vacancy_fraction,
    equivalent_plastic_strain_rate_s, exact_vacancy_interval, initial_vacancy_ledgers,
    vacancy_annihilation_rate_s, vacancy_attachment_frequency_s, vacancy_conservation,
    vacancy_diffusivity_m2_s, vacancy_growth_rates_m_s, vacancy_production_rate_s,
    vacancy_weighted_first_passage_time,
)
from arrhenius_fracture.voiding_production_v5 import (
    _overlap_conservative_p0,
    advance_site_on_common_vacancy_clock, build_production_void_state,
    evolve_accepted_plastic_vacancy_interval,
    integrate_cavity_growth_on_common_clock, vacancy_coupled_void_rates,
)
from arrhenius_fracture.voiding_v5 import VoidPhase, VoidingConfig, advance_site


TEMPERATURE_VALUES = {
    300: (1.7472941508322758e-54, 2.762211631782678e-35),
    875: (3.7051961824506883e-19, 1.6159612939876314e-16),
    950: (1.0565865061173238e-17, 9.581854307003544e-16),
    1100: (2.1818610889580016e-15, 1.6264918646932926e-14),
    1200: (3.6357290122358446e-14, 7.249603215142833e-14),
}


def test_reference_row_is_exact_common_and_uncalibrated():
    assert REFERENCE_VACANCY_ROW.row_id == COMMON_VACANCY_ROW_ID
    assert REFERENCE_VACANCY_ROW.calibration_status == "REFERENCE_UNCALIBRATED"
    assert REFERENCE_VACANCY_ROW.vacancy_formation_energy_eV == 3.2
    assert REFERENCE_VACANCY_ROW.vacancy_migration_energy_eV == 1.7
    assert {bundle.vacancy_row_id for bundle in all_material_bundles().values()} == {
        COMMON_VACANCY_ROW_ID
    }


def test_full_precision_equilibrium_and_diffusivity_temperature_matrix():
    equilibrium = []
    diffusivity = []
    for temperature, expected in TEMPERATURE_VALUES.items():
        c_eq = equilibrium_vacancy_fraction(temperature)
        D_v = vacancy_diffusivity_m2_s(temperature)
        assert c_eq == expected[0]
        assert D_v == expected[1]
        equilibrium.append(c_eq); diffusivity.append(D_v)
    assert np.all(np.diff(equilibrium) > 0.0)
    assert np.all(np.diff(diffusivity) > 0.0)


@pytest.mark.parametrize("initial_factor", (0.1, 10.0))
def test_no_plasticity_exact_relaxation_toward_equilibrium(initial_factor):
    c_eq = equilibrium_vacancy_fraction(950.0)
    result = exact_vacancy_interval(
        np.asarray([initial_factor * c_eq]), element_area_m2=np.asarray([2.0]),
        production_rate_s=0.0, annihilation_rate_s=0.25,
        equilibrium_fraction=c_eq, duration_s=3.0,
    )
    expected = c_eq + (initial_factor * c_eq - c_eq) * math.exp(-0.75)
    assert result["vacancy_fraction_gp"][0] == pytest.approx(expected, rel=2.0e-15)
    assert abs(result["interval_conservation_residual_m2"]) < 1.0e-30
    assert min(initial_factor * c_eq, c_eq) <= result["vacancy_fraction_gp"][0] <= max(initial_factor * c_eq, c_eq)


def test_plastic_pulse_and_dislocation_boost_are_causal():
    previous = np.zeros((3, 2))
    accepted = np.asarray([[2.0e-4, 2.0e-4], [-1.0e-4, -1.0e-4], [5.0e-5, 5.0e-5]])
    plastic = equivalent_plastic_strain_rate_s(previous, accepted, 2.0)
    assert np.all(plastic > 0.0)
    baseline = vacancy_production_rate_s(plastic, np.zeros(2), rho_reference_m2=1.0e12)
    boosted = vacancy_production_rate_s(plastic, np.full(2, 4.0e12), rho_reference_m2=1.0e12)
    assert np.all(boosted > baseline)
    assert np.all(vacancy_production_rate_s(0.0, 4.0e12, rho_reference_m2=1.0e12) == 0.0)
    no_boost = replace(REFERENCE_VACANCY_ROW, dislocation_production_boost=0.0)
    assert vacancy_production_rate_s(2.0, 4.0e12, rho_reference_m2=1.0e12, row=no_boost) == pytest.approx(6.0e-5)


def test_engineering_shear_voigt_convention_is_bound_to_fem_storage():
    assert "gamma_xy=2epsilon_xy" in ENGINEERING_SHEAR_VOIGT_CONVENTION
    previous = np.zeros((3, 1))
    accepted = np.asarray([[0.0], [0.0], [3.0e-4]])
    rate = equivalent_plastic_strain_rate_s(previous, accepted, 1.0)[0]
    assert rate == pytest.approx(3.0e-4 / math.sqrt(3.0), rel=2.0e-15)


def test_transport_birth_and_growth_reciprocal_controls():
    b = 2.74e-10
    low = vacancy_attachment_frequency_s(950.0, 1.0e-8, b)
    high = vacancy_attachment_frequency_s(950.0, 2.0e-8, b)
    assert high == pytest.approx(2.0 * low)
    slow = vacancy_growth_rates_m_s(
        temperature_K=950.0, radius_m=1.0e-8,
        vacancy_fraction_site=1.0e-8, equivalent_plastic_rate_s=1.0e-4,
    )
    rich = vacancy_growth_rates_m_s(
        temperature_K=950.0, radius_m=1.0e-8,
        vacancy_fraction_site=2.0e-8, equivalent_plastic_rate_s=1.0e-4,
    )
    blocked = vacancy_growth_rates_m_s(
        temperature_K=950.0, radius_m=1.0e-8,
        vacancy_fraction_site=2.0e-8, equivalent_plastic_rate_s=0.0,
    )
    assert rich["Rdot_diff_m_s"] > slow["Rdot_diff_m_s"]
    assert rich["Rdot_m_s"] >= slow["Rdot_m_s"]
    assert blocked["Rdot_plast_m_s"] == blocked["Rdot_m_s"] == 0.0
    assert vacancy_annihilation_rate_s(1100.0, 2.0e12, process_zone_length_m=5.0e-5) > vacancy_annihilation_rate_s(950.0, 2.0e12, process_zone_length_m=5.0e-5)


def test_production_default_is_equilibrium_and_explicit_supersaturation_is_declared():
    default, _ = build_production_void_state(bundle=material_bundle("DBTT"), enabled=True, temperature_K=875.0)
    assert np.array_equal(
        default.vacancy_fraction_gp,
        np.full(default.mesh.ne, equilibrium_vacancy_fraction(875.0)),
    )
    supersaturated, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), enabled=True, temperature_K=875.0,
        initial_vacancy_fraction_gp=2.0e-2,
        test_only_extreme_reservoir=True,
    )
    assert np.all(supersaturated.vacancy_fraction_gp == 2.0e-2)
    assert vacancy_conservation(default)["vacancy_conservation_residual_m2"] == 0.0
    assert vacancy_conservation(supersaturated)["vacancy_conservation_residual_m2"] == 0.0
    with pytest.raises(ValueError, match="dilute-model"):
        build_production_void_state(
            bundle=material_bundle("DBTT"), enabled=True,
            initial_vacancy_fraction_gp=2.0e-2,
        )


def test_accepted_state_interval_and_checkpoint_are_exact(tmp_path):
    state, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), enabled=True, temperature_K=950.0,
        initial_vacancy_fraction_gp=1.0e-6,
    )
    ep = np.asarray(state.ep_gp).copy(); ep[0] += 1.0e-5; ep[1] -= 0.5e-5
    accepted = replace(state, ep_gp=ep)
    evolved, audit = advance_accepted_vacancy_state(
        state, accepted, duration_s=0.5, temperature_K=950.0,
        process_zone_length_m=5.0e-5, rho_reference_m2=1.0e12,
    )
    assert audit["vacancy_plastic_production_area_m2"] > 0.0
    assert abs(audit["vacancy_conservation_residual_m2"]) < 1.0e-24
    path = tmp_path / "vacancy.json"
    write_checkpoint(evolved, path)
    restored = restore_checkpoint(path)
    assert np.array_equal(restored.vacancy_fraction_gp, evolved.vacancy_fraction_gp)
    assert dict(restored.vacancy_ledgers) == dict(evolved.vacancy_ledgers)


def test_p0_refinement_and_local_consumption_are_conservative():
    state, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), enabled=True,
        initial_vacancy_fraction_gp=1.0e-2,
    )
    heterogeneous = np.linspace(1.0e-3, 2.0e-2, state.mesh.ne)
    state = replace(
        state, vacancy_fraction_gp=heterogeneous,
        vacancy_ledgers=initial_vacancy_ledgers(heterogeneous, state.mesh.area_e),
    )
    before = vacancy_conservation(state)["vacancy_field_inventory_area_m2"]
    refined, _ = refine_accepted_state(
        state, marked_parent_elements=tuple(range(8)),
        active_tip_ids=state.crack_network.active_tip_ids,
        generation=1, operation_index=1,
    )
    after = vacancy_conservation(refined)["vacancy_field_inventory_area_m2"]
    assert after == pytest.approx(before, abs=1.0e-24)
    consumed, audit = debit_local_vacancy_inventory(
        refined, requested_area_m2=1.0e-12, element_ids=np.arange(20),
    )
    assert audit["consumed_vacancy_area_m2"] > 0.0
    assert abs(vacancy_conservation(consumed)["vacancy_conservation_residual_m2"]) < 1.0e-24


def test_void_rates_have_no_hidden_peierls_taylor_parameter_leakage_at_frozen_state():
    state, _ = build_production_void_state(
        bundle=material_bundle("Peak"), enabled=True,
        initial_vacancy_fraction_gp=1.0e-3,
    )
    stress = np.asarray([[1.0e9, 0.0], [0.0, 2.0e9]])
    plastic = np.full(state.mesh.ne, 1.0e-4)
    first, audit = vacancy_coupled_void_rates(
        state, VoidingConfig(enabled=True), temperature_K=950.0,
        stress_tensor_Pa=stress, center_m=(7.0e-4, 0.0),
        equivalent_plastic_strain_rate_gp=plastic, radius_m=1.0e-8,
    )
    # The coupled closure receives accepted rate/rho/c_v, not Peierls/Taylor
    # parameters. Re-querying the identical accepted state is bitwise stable.
    second, audit2 = vacancy_coupled_void_rates(
        state, VoidingConfig(enabled=True), temperature_K=950.0,
        stress_tensor_Pa=stress, center_m=(7.0e-4, 0.0),
        equivalent_plastic_strain_rate_gp=plastic, radius_m=1.0e-8,
    )
    assert first == second and audit == audit2
    assert first["birth_s"] > 0.0 and first["Rdot_m_s"] > 0.0


def test_re_evolved_peierls_taylor_change_is_mediated_by_accepted_state():
    bundle = material_bundle("DBTT")
    state, _ = build_production_void_state(
        bundle=bundle, enabled=True, initial_vacancy_fraction_gp=1.0e-3,
    )
    soft, soft_rate, soft_audit = evolve_accepted_plastic_vacancy_interval(
        state, bundle=bundle, temperature_K=900.0, duration_s=0.1,
        plasticity_parameter_overrides={"peierls_H0_eV": 1.0, "taylor_athermal_alpha": 0.1},
        test_only=True,
    )
    hard, hard_rate, hard_audit = evolve_accepted_plastic_vacancy_interval(
        state, bundle=bundle, temperature_K=900.0, duration_s=0.1,
        plasticity_parameter_overrides={"peierls_H0_eV": 5.0, "taylor_athermal_alpha": 1.0},
        test_only=True,
    )
    assert not np.array_equal(soft_rate, hard_rate)
    assert not np.array_equal(soft.ep_gp, hard.ep_gp)
    assert not np.array_equal(soft.rho_gp, hard.rho_gp)
    assert not np.array_equal(soft.vacancy_fraction_gp, hard.vacancy_fraction_gp)
    assert (
        soft_audit["vacancy_update"]["vacancy_plastic_production_area_m2"]
        != hard_audit["vacancy_update"]["vacancy_plastic_production_area_m2"]
    )


def test_temperature_matrix_exercises_all_vacancy_branches():
    b = 2.74e-10
    for temperature in TEMPERATURE_VALUES:
        c_eq = equilibrium_vacancy_fraction(temperature)
        diffusivity = vacancy_diffusivity_m2_s(temperature)
        production = vacancy_production_rate_s(1.0e-4, 4.0e12, rho_reference_m2=1.0e12)
        annihilation = vacancy_annihilation_rate_s(
            temperature, 4.0e12, process_zone_length_m=5.0e-5,
        )
        attachment = vacancy_attachment_frequency_s(temperature, 10.0 * c_eq, b)
        growth = vacancy_growth_rates_m_s(
            temperature_K=temperature, radius_m=1.0e-8,
            vacancy_fraction_site=10.0 * c_eq, equivalent_plastic_rate_s=1.0e-4,
        )
        assert diffusivity > 0.0 and production > 0.0 and annihilation > 0.0
        assert attachment > 0.0 and growth["Rdot_diff_m_s"] > 0.0
        assert growth["Rdot_plast_m_s"] > 0.0 and growth["Rdot_m_s"] > 0.0
        assert growth["controlling_branch"] in {"DIFFUSION", "PLASTIC_ACCOMMODATION"}


def test_equilibrium_zero_plasticity_has_no_artificial_void_in_bounded_interval():
    state, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), enabled=True, temperature_K=950.0,
    )
    rates, audit = vacancy_coupled_void_rates(
        state, VoidingConfig(enabled=True), temperature_K=950.0,
        stress_tensor_Pa=np.zeros((2, 2)), center_m=(7.0e-4, 0.0),
        equivalent_plastic_strain_rate_gp=np.zeros(state.mesh.ne), radius_m=1.0e-8,
    )
    assert audit["Rdot_diff_m_s"] == audit["Rdot_plast_m_s"] == audit["Rdot_m_s"] == 0.0
    site = state.void_state.sites[0]
    crossing = site.birth.crossing_time_exact(rates["birth_s"] * site.candidate_weight)
    bounded = 1.0 if crossing is None else min(1.0, 0.5 * crossing)
    advanced, events = advance_site(state.void_state, site.site_id, bounded, rates=rates)
    assert not events and not advanced.cavities


def test_analytic_vacancy_action_and_common_clock_first_passage(monkeypatch):
    bundle = material_bundle("DBTT")
    previous, _ = build_production_void_state(
        bundle=bundle, enabled=True, temperature_K=950.0,
        initial_vacancy_fraction_gp=1.0e-8,
    )
    ep = np.asarray(previous.ep_gp).copy()
    ep[0] += 2.0e-5; ep[1] -= 1.0e-5
    accepted = replace(previous, ep_gp=ep)
    production = np.full(previous.mesh.ne, 2.0e-8)
    sink = np.full(previous.mesh.ne, 0.2)
    patch = np.arange(20)
    root = vacancy_weighted_first_passage_time(
        previous, patch, production_rate_s=production,
        annihilation_rate_s=sink, equilibrium_fraction=1.0e-18,
        rate_per_vacancy_s=1.0e8, action_remaining=0.25,
        maximum_duration_s=10.0,
    )
    assert root is not None and 0.0 < root < 10.0
    _, integral = analytic_vacancy_field_and_integral(
        previous.vacancy_fraction_gp, production_rate_s=production,
        annihilation_rate_s=sink, equilibrium_fraction=1.0e-18,
        duration_s=root,
    )
    average_integral = np.average(integral[patch], weights=previous.mesh.area_e[patch])
    assert 1.0e8 * average_integral == pytest.approx(0.25, rel=1.0e-10)
    monkeypatch.setattr(
        "arrhenius_fracture.plasticity.update_plasticity",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("second plasticity update")),
    )
    evolved, audit = advance_site_on_common_vacancy_clock(
        previous, accepted, VoidingConfig(enabled=True), bundle=bundle,
        duration_s=1.0, temperature_K=950.0,
        stress_tensor_Pa=np.asarray([[1.0e9, 0.0], [0.0, 2.0e9]]),
    )
    assert 0.0 < audit["elapsed_time_s"] <= 1.0
    assert audit["vacancy"]["vacancy_plastic_production_area_m2"] > 0.0
    assert abs(vacancy_conservation(evolved)["vacancy_conservation_residual_m2"]) <= 1.0e-24


def test_coupled_growth_integrates_radius_field_and_inventory_together():
    bundle = material_bundle("DBTT")
    state, _ = build_production_void_state(
        bundle=bundle, enabled=True, initial_vacancy_fraction_gp=1.0e-3,
    )
    site = replace(state.void_state.sites[0], phase=VoidPhase.STABLE_SUBGRID_VOID)
    state = replace(state, void_state=replace(state.void_state, sites=(site,)))
    from arrhenius_fracture.voiding_production_v5 import create_subgrid_cavity_from_vacancies
    state, _ = create_subgrid_cavity_from_vacancies(
        state, site.site_id, 1.0e-9, process_zone_length_m=5.0e-5,
    )
    plastic = np.full(state.mesh.ne, 1.0e-3)
    production = vacancy_production_rate_s(
        plastic, state.rho_gp, rho_reference_m2=5.0e12,
    )
    sink = vacancy_annihilation_rate_s(
        900.0, state.rho_gp, process_zone_length_m=5.0e-5,
    )
    cavity = state.void_state.cavities[0]
    grown, audit = integrate_cavity_growth_on_common_clock(
        state, cavity.cavity_id, target_radius_m=1.01e-9,
        maximum_duration_s=1.0e4, temperature_K=900.0,
        equivalent_plastic_strain_rate_gp=plastic,
        production_rate_s=production, annihilation_rate_s=sink,
        process_zone_length_m=5.0e-5,
    )
    assert audit["target_reached"]
    assert grown.void_state.cavities[0].radius_m == pytest.approx(1.01e-9)
    assert audit["cavity_consumption_area_m2"] > 0.0
    assert abs(vacancy_conservation(grown)["vacancy_conservation_residual_m2"]) <= audit["conservation_roundoff_budget_m2"]
    assert audit["accepted_ledger_roundoff_correction_m2"] <= conservation_roundoff_budget_m2(
        *grown.vacancy_ledgers.values()
    )


def test_overlap_p0_transfer_preserves_active_site_capture_inventory():
    state, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), enabled=True,
        initial_vacancy_fraction_gp=1.0e-3,
    )
    heterogeneous = np.linspace(1.0e-8, 1.0e-3, state.mesh.ne)
    state = replace(
        state, vacancy_fraction_gp=heterogeneous,
        vacancy_ledgers=initial_vacancy_ledgers(heterogeneous, state.mesh.area_e),
    )
    refined, _ = refine_accepted_state(
        state, marked_parent_elements=tuple(range(100)),
        active_tip_ids=state.crack_network.active_tip_ids,
        generation=1, operation_index=1,
    )
    projected, audit = _overlap_conservative_p0(
        state, refined.mesh, state.vacancy_fraction_gp,
    )
    projected_state = replace(refined, vacancy_fraction_gp=projected)
    center = state.void_state.sites[0].center_m
    old_patch = deterministic_capture_patch(state, center, 5.0e-5)
    new_patch = deterministic_capture_patch(projected_state, center, 5.0e-5)
    old_local = np.sum(state.mesh.area_e[old_patch] * state.vacancy_fraction_gp[old_patch])
    new_local = np.sum(refined.mesh.area_e[new_patch] * projected[new_patch])
    assert audit["method"] == "ELEMENT_OVERLAP_CONSERVATIVE_P0"
    assert abs(audit["source_inventory_area_m2"] - np.sum(refined.mesh.area_e * projected)) <= audit["roundoff_budget_m2"]
    assert abs(new_local - old_local) <= 1.0e-24


def test_common_clock_binds_each_exact_material_bundle_identity():
    observed = {}
    for family, bundle in all_material_bundles().items():
        previous, _ = build_production_void_state(bundle=bundle, enabled=True)
        ep = np.asarray(previous.ep_gp).copy()
        ep[0] += 1.0e-7; ep[1] -= 0.5e-7
        accepted = replace(previous, ep_gp=ep)
        _, audit = advance_site_on_common_vacancy_clock(
            previous, accepted, VoidingConfig(enabled=True), bundle=bundle,
            duration_s=0.1, temperature_K=900.0,
            stress_tensor_Pa=np.asarray([[1.0e9, 0.0], [0.0, 2.0e9]]),
        )
        assert audit["material_bundle_id"] == bundle.bundle_id
        assert audit["material_bundle_sha256"] == bundle.bundle_hash
        observed[family] = audit["plasticity_identity"]
    assert set(observed) == {"Peak", "DBTT", "weak-T", "ceramic-like"}
    assert len(set(observed.values())) == 4
