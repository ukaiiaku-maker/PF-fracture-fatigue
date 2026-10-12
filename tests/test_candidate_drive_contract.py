import pytest

from arrhenius_fracture.candidate_drive_contract import (
    CandidateClass, FiniteSeparationOperation, LigamentKineticInterpretation,
    LocalIncrementalFrontState, finite_difference_energy_release,
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


def test_option_a_transaction_preserves_owner_mark_and_full_rollback():
    state = LocalIncrementalFrontState("front-7", "site-12", 0.75)
    trial = state.trial(duration_s=0.2, rate_s=2.0, new_solid_advance_m=1e-7)
    assert trial.rollback() is state
    accepted = trial.accept()
    assert accepted.owner_id == state.owner_id
    assert accepted.physical_site_id == state.physical_site_id
    assert accepted.exponential_mark == state.exponential_mark
    assert accepted.physical_time_s == pytest.approx(0.2)
    assert accepted.cumulative_action == pytest.approx(0.4)


def test_refinement_does_not_create_clocks_or_change_action_and_advance():
    state = LocalIncrementalFrontState("front", "physical-site", 1.25)
    coarse = state.trial(duration_s=1.0, rate_s=0.8, new_solid_advance_m=2e-7).accept()
    fine = state
    for _ in range(10):
        fine = fine.trial(duration_s=0.1, rate_s=0.8,
                          new_solid_advance_m=2e-8).accept()
    assert fine.cumulative_action == pytest.approx(coarse.cumulative_action)
    assert fine.accepted_solid_advance_m == pytest.approx(coarse.accepted_solid_advance_m)
    assert fine.exponential_mark == coarse.exponential_mark
    assert fine.physical_site_id == coarse.physical_site_id
