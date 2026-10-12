import pytest

from arrhenius_fracture.candidate_drive_contract import (
    CandidateClass, FiniteSeparationOperation, LigamentKineticInterpretation,
    finite_difference_energy_release,
)


def test_cavity_free_span_is_never_charged_as_new_solid_area():
    op = FiniteSeparationOperation(
        CandidateClass.FINITE_SOLID_LIGAMENT,
        released_energy_J_per_m_thickness=3.0e-3,
        new_solid_length_m=2.0e-6,
        cavity_free_span_m=8.0e-6,
        total_geometric_span_m=10.0e-6,
    )
    assert op.energy_release_J_m2 == pytest.approx(1500.0)


def test_finite_operation_converges_to_derivative_for_smooth_energy():
    # Pi(a)=Pi0-J*a+c*a^2.  The forward finite operation approaches J.
    J = 417.0
    c = 2.5e6
    # Stay on the energy-releasing side of the quadratic landscape so this
    # test exercises convergence rather than the fail-closed sign check.
    a = 40.0e-6
    energy = lambda x: 10.0 - J * x + c * x * x
    exact_release = J - 2.0 * c * a
    for da in (1e-6, 1e-7, 1e-8):
        value = finite_difference_energy_release(energy(a), energy(a + da), da)
        assert value == pytest.approx(exact_release - c * da)


def test_local_and_collective_ligament_interpretations_are_not_silently_identical():
    op = FiniteSeparationOperation(
        CandidateClass.FINITE_SOLID_LIGAMENT, 1e-3, 2e-6,
    )
    local = op.rate_contract(
        interpretation=LigamentKineticInterpretation.LOCAL_INCREMENTAL_FRONT,
        activation_area_m2=4e-18, physical_site_area_m2=1e-19,
    )
    collective = op.rate_contract(
        interpretation=LigamentKineticInterpretation.COLLECTIVE_FINITE_LIGAMENT,
        activation_area_m2=4e-18, physical_site_area_m2=1e-19,
    )
    assert local["energy_release_J_m2"] == collective["energy_release_J_m2"]
    assert local["multiplicity"] == pytest.approx(40.0)
    assert collective["multiplicity"] == 1.0


def test_geometry_accounting_fails_closed():
    with pytest.raises(ValueError, match="must equal"):
        FiniteSeparationOperation(
            CandidateClass.FINITE_SOLID_LIGAMENT,
            released_energy_J_per_m_thickness=1e-3,
            new_solid_length_m=2e-6,
            cavity_free_span_m=3e-6,
            total_geometric_span_m=9e-6,
        )
