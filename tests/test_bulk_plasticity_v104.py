from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from arrhenius_fracture.bulk_plasticity_v104 import (
    PlasticityScope, accepted_bulk_load_step, production_bulk_configuration,
    vacancy_source_diagnostics,
)
from arrhenius_fracture.emission_derived_plasticity import (
    EmissionDerivedPeierlsTaylorModel, config_from_dislocation_config,
)
from arrhenius_fracture.topology_transaction_v11 import equilibrate_fixed_load_with_production_fem
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.vacancy_transport_phase_field_v2 import cavity_growth_rate_from_transport
from arrhenius_fracture.voiding_production_v5 import build_production_void_state


@pytest.fixture(scope="module")
def initial():
    state, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=900.0,
    )
    rho0 = float(material_bundle("DBTT").fracture_row["rho_forest_floor_m2"])
    return equilibrate_fixed_load_with_production_fem(
        replace(state, ep_gp=np.zeros_like(state.ep_gp),
                rho_gp=np.full_like(state.rho_gp, rho0))
    )


def test_scope_is_explicit_immutable_and_future_mode_is_disabled(initial):
    cfg, manifest = production_bulk_configuration(
        initial, PlasticityScope.FULL_FIELD_BULK_PT
    )
    assert cfg.dislocations.bulk_kinetics_model == "emission_derived_peierls_taylor_multihit"
    assert cfg.dislocations.plastic_update_mode == "explicit_rate"
    assert cfg.dislocations.thermo_consistency_mode == "time_cone"
    assert cfg.dislocations.bulk_mult_frac == 1.0
    assert cfg.dislocations.tip_source_rho_per_emit == 0.0
    assert cfg.dislocations.rho_transport_c == 0.0
    assert cfg.dislocations.freeze_rho is False
    with pytest.raises(FrozenInstanceError):
        manifest.bulk_mult_frac = 0.0
    with pytest.raises(ValueError, match="reserved and disabled"):
        production_bulk_configuration(initial, PlasticityScope.SOURCES_ONLY_TRANSPORTED)


def test_detailed_balance_is_exactly_zero_at_zero_stress(initial):
    cfg, _ = production_bulk_configuration(initial, PlasticityScope.FULL_FIELD_BULK_PT)
    rates = EmissionDerivedPeierlsTaylorModel(
        config_from_dislocation_config(cfg.dislocations)
    ).rates(np.zeros(4), np.full(4, 1.0e14), 900.0, initial.material.b)
    assert np.array_equal(rates["equivalent_plastic_rate_s"], np.zeros(4))
    assert np.array_equal(rates["peierls_rate_s"], np.zeros(4))
    assert np.array_equal(rates["taylor_single_hit_rate_s"], np.zeros(4))


def test_paired_accepted_step_preserves_tip_owner_and_records_bulk_rates(initial):
    tip, tip_audit = accepted_bulk_load_step(
        initial, scope=PlasticityScope.TIP_PROCESS_ONLY, load_factor=1.05,
        duration_s=1.0e-10, temperature_K=900.0,
    )
    bulk, audit = accepted_bulk_load_step(
        initial, scope=PlasticityScope.FULL_FIELD_BULK_PT, load_factor=1.05,
        duration_s=1.0e-10, temperature_K=900.0,
    )
    assert np.array_equal(tip.ep_gp, initial.ep_gp)
    assert np.array_equal(tip.rho_gp, initial.rho_gp)
    assert tip.tip_process_state == bulk.tip_process_state == initial.tip_process_state
    assert np.max(audit["dot_ep_gp_s"]) > 0.0
    info = audit["constitutive"]
    assert np.any(info["rho_storage_rate_gp_m2_s"] > 0.0)
    assert np.any(info["rho_dynamic_annihilation_rate_gp_m2_s"] > 0.0)
    assert np.array_equal(info["rho_previous_gp_m2"], initial.rho_gp)
    assert np.array_equal(info["rho_accepted_gp_m2"], bulk.rho_gp)
    source = vacancy_source_diagnostics(initial, bulk, 1.0e-10)
    assert source["integrated_source_fraction_area_per_s"] > 0.0
    assert source["far_field_source_fraction_area_per_s"] > 0.0
    assert tip_audit["scope"] == PlasticityScope.TIP_PROCESS_ONLY.value


def test_fixed_rho_scope_evolves_plastic_strain_without_density(initial):
    fixed, audit = accepted_bulk_load_step(
        initial, scope=PlasticityScope.FULL_FIELD_PT_FIXED_RHO,
        load_factor=1.05, duration_s=1.0e-10, temperature_K=900.0,
    )
    assert np.max(audit["dot_ep_gp_s"]) > 0.0
    assert not np.array_equal(fixed.ep_gp, initial.ep_gp)
    assert np.array_equal(fixed.rho_gp, initial.rho_gp)
    assert audit["manifest"]["freeze_rho"] is True


def test_series_limited_growth_requires_both_branches():
    both = cavity_growth_rate_from_transport(
        absorbed_inventory_rate_area_m2_s=1.0e-18, radius_m=1.0e-7,
        equivalent_plastic_strain_rate_s=1.0, plastic_accommodation_coefficient=1.0,
    )
    no_transport = cavity_growth_rate_from_transport(
        absorbed_inventory_rate_area_m2_s=0.0, radius_m=1.0e-7,
        equivalent_plastic_strain_rate_s=1.0, plastic_accommodation_coefficient=1.0,
    )
    no_accommodation = cavity_growth_rate_from_transport(
        absorbed_inventory_rate_area_m2_s=1.0e-18, radius_m=1.0e-7,
        equivalent_plastic_strain_rate_s=1.0, plastic_accommodation_coefficient=0.0,
    )
    assert both["Rdot_m_s"] > 0.0
    assert no_transport["Rdot_m_s"] == 0.0
    assert no_accommodation["Rdot_m_s"] == 0.0
    assert both["Rdot_accommodation_m_s"] == both["Rdot_plastic_m_s"]
