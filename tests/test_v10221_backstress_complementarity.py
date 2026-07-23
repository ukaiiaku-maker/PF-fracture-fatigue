import math

import pytest

from arrhenius_fracture import persistent_site_source_v10221 as source
from arrhenius_fracture.persistent_site_bracket_fix_v10221 import (
    install_backstress_complementarity_fix,
    solve_backstress_limited_activations,
)


def test_high_hazard_returns_mechanical_blocking_increment():
    drive = 1.0e9
    rho0 = 1.0e12
    rho_per = 1.0e10
    kback = 100.0
    expected = (((drive / kback) ** 2) - rho0) / rho_per
    result = solve_backstress_limited_activations(
        multiplicity=3.5e5,
        dt_s=4.2,
        drive_stress_Pa=drive,
        rho_initial_m2=rho0,
        rho_increment_per_activation_m2=rho_per,
        backstress_prefactor_Pa_sqrt_m2=kback,
        rate_function=lambda sigma: 1.0e11,
    )
    assert result == pytest.approx(expected)


def test_interior_root_is_preserved_when_it_exists():
    result = solve_backstress_limited_activations(
        multiplicity=10.0,
        dt_s=0.1,
        drive_stress_Pa=1.0e9,
        rho_initial_m2=1.0e12,
        rho_increment_per_activation_m2=1.0e10,
        backstress_prefactor_Pa_sqrt_m2=100.0,
        rate_function=lambda sigma: 2.0,
    )
    assert result == pytest.approx(2.0, rel=1.0e-8)


def test_install_replaces_solver_used_by_persistent_emission():
    original = source.solve_backstress_limited_activations
    try:
        install_backstress_complementarity_fix()
        assert source.solve_backstress_limited_activations is (
            solve_backstress_limited_activations
        )
    finally:
        source.solve_backstress_limited_activations = original
