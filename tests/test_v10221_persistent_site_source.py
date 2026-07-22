import math
from pathlib import Path

import pytest

from arrhenius_fracture.parameter_registry_v9111 import select_option
from arrhenius_fracture.persistent_site_source_v10221 import (
    PersistentSiteConfig,
    PersistentSiteStateResolvedTipEngine,
    effective_front_width_m,
    persistent_site_multiplicity,
    solve_backstress_limited_activations,
)


def test_reference_width_and_density_scaling():
    w0 = effective_front_width_m(
        5.0e12,
        reference_width_m=10.0e-6,
        reference_density_m2=5.0e12,
        minimum_width_m=0.625e-6,
        maximum_width_m=50.0e-6,
    )
    assert w0 == pytest.approx(10.0e-6)
    w4 = effective_front_width_m(
        2.0e13,
        reference_width_m=10.0e-6,
        reference_density_m2=5.0e12,
        minimum_width_m=0.625e-6,
        maximum_width_m=50.0e-6,
    )
    assert w4 == pytest.approx(5.0e-6)
    wmin = effective_front_width_m(
        1.0e22,
        reference_width_m=10.0e-6,
        reference_density_m2=5.0e12,
        minimum_width_m=0.625e-6,
        maximum_width_m=50.0e-6,
    )
    assert wmin == pytest.approx(0.625e-6)


def test_reference_area_reproduces_top1_multiplicity():
    area = 25.0e-12
    r0 = 1.0e-6
    width = 10.0e-6
    arc = area / (r0 * width)
    multiplicity = persistent_site_multiplicity(
        1.4115242646890916e16,
        r0,
        width,
        arc,
    )
    assert multiplicity == pytest.approx(3.528810661722729e5)


def test_backstress_solver_blocks_without_inventory():
    prefactor = 50.0
    result = solve_backstress_limited_activations(
        multiplicity=3.5e5,
        dt_s=1.0,
        drive_stress_Pa=1.0e9,
        rho_initial_m2=1.0e14,
        rho_increment_per_activation_m2=1.0e11,
        backstress_prefactor_Pa_sqrt_m2=prefactor,
        rate_function=lambda sigma: 1.0e4 * math.exp(sigma / 1.0e9),
    )
    block = (((1.0e9 / prefactor) ** 2) - 1.0e14) / 1.0e11
    assert 0.0 < result < block


def test_backstress_solver_zero_when_already_blocked():
    result = solve_backstress_limited_activations(
        multiplicity=3.5e5,
        dt_s=10.0,
        drive_stress_Pa=1.0e9,
        rho_initial_m2=1.0e14,
        rho_increment_per_activation_m2=1.0e11,
        backstress_prefactor_Pa_sqrt_m2=1.0e3,
        rate_function=lambda sigma: 1.0e8,
    )
    assert result == 0.0


def test_top1_registry_and_no_recovery_contract():
    root = Path(__file__).resolve().parents[1]
    registry = (
        root
        / "arrhenius_fracture"
        / "data"
        / "materials"
        / "v10_2_21_v912_top1_persistent_site_registry.csv"
    )
    selected = select_option(
        "v912_top1_peak_persistent_sites",
        registry,
        canonical_stage3_only=False,
    )
    assert selected.candidate_id == "v912_targeted_local_peak_013476_0368"
    assert float(selected.row["rho_source0_m2"]) == pytest.approx(1.4115242646890916e16)
    assert float(selected.row["retained_recovery_rate_s"]) == 0.0
    assert float(selected.row["recovery_nu0_s"]) == 0.0
    assert float(selected.row["legacy_source_sites_active"]) == 0.0


def test_engine_audit_declares_no_finite_inventory():
    PersistentSiteStateResolvedTipEngine.configure_persistent_sites(
        PersistentSiteConfig(rho_site0_m2=1.4115242646890916e16)
    )
    audit = PersistentSiteStateResolvedTipEngine.audit_payload()[
        "persistent_site_source_v10221"
    ]
    assert audit["finite_source_inventory"] is False
    assert audit["source_depletion_on_emission"] is False
    assert audit["source_refresh_on_crack_advance"] is False
    assert audit["site_multiplicity_in_arrhenius_hazard"] is True
