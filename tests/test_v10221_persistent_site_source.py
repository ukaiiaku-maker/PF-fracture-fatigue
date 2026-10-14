import math
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.parameter_registry_v9111 import select_option
from arrhenius_fracture.persistent_site_source_v10221 import (
    PersistentSiteConfig,
    PersistentSiteStateResolvedTipEngine,
    effective_front_width_m,
    persistent_site_multiplicity,
    solve_backstress_limited_activations,
    install_persistent_site_source,
)
from arrhenius_fracture.signed_burgers_shared_v1025 import (
    KERNEL_SCHEMA, SignedShieldingKernel, install_signed_burgers_population,
)
from arrhenius_fracture.unified_mpz import UnifiedMPZState


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


def _coupled_test_state():
    class State:
        _advect_forward = staticmethod(UnifiedMPZState._advect_forward)

        @property
        def mobile_count(self):
            return float(np.sum(self.mobile))

        @property
        def retained_count(self):
            return float(np.sum(self.retained))

        @property
        def wake_mobile_count(self):
            return float(np.sum(self.wake_mobile))

        @property
        def wake_retained_count(self):
            return float(np.sum(self.wake_retained))

        @property
        def available_site_fraction(self):
            return float(np.sum(self.available_sites) / np.sum(self.site_capacity))

        def blunted_radius(self, r0, b):
            return float(r0 + 0.1 * abs(b) * np.sum(self.accumulated_slip))

        def local_stress_profile_Pa(self, stress):
            return float(stress) * np.sqrt(1.0e-6 / (1.0e-6 + self.x))

        def local_forest_density_m2(self, wake=False):
            return 1.0e10 + np.sum(self.retained, axis=0) / (self.dx * 1.0e-6)

        def emission_rate_per_site(self, stress, temperature):
            return 0.02 * math.exp(max(float(stress), 0.0) / 1.0e9)

        def _transport_rates(self, stress, rho, temperature, b):
            n = self.n_bins
            return {
                "peierls": np.full(n, 4.0),
                "taylor": np.full(n, 3.0),
                "encounter": 10.0 + 1.0e-10 * np.asarray(rho),
                "m": np.full(n, 2.0),
                "velocity": np.full(n, 2.0e-6),
            }

    state = State()
    state.n_systems = 1
    state.n_bins = 8
    state.length_m = 8.0e-6
    state.dx = 1.0e-6
    state.x = (np.arange(8) + 0.5) * state.dx
    state.wake_n_bins = 8
    state.wake_dx = state.dx
    state.wake_x = state.x.copy()
    state.cfg = SimpleNamespace(
        source_bin_count=2, mobile_recovery_rate_s=0.0,
        blunting_length_m=1.0e-6, forest_density_floor_m2=1.0e10,
        taylor_stress_fraction=1.0 / math.sqrt(3.0), wake_shielding=False,
        mobile_shield_fraction=0.0,
    )
    state.manifest = SimpleNamespace(
        retained_recovery_rate_s=0.0, source_refresh_length_m=1.0e-6,
    )
    state.site_capacity = np.ones(1)
    state.available_sites = np.ones(1)
    for name in ("mobile", "retained", "accumulated_slip"):
        setattr(state, name, np.zeros((1, 8)))
    for name in ("wake_mobile", "wake_retained", "wake_slip"):
        setattr(state, name, np.zeros((1, 8)))
    for name in (
        "emitted_total", "escaped_total", "recovered_total",
        "net_retained_gain_total", "net_retained_loss_total",
        "gross_trapping_total", "gross_release_total",
        "mobile_distance_population_total_m", "advance_total_m",
        "wake_discarded_mobile_total", "wake_discarded_retained_total",
        "wake_discarded_slip_total", "time_s",
    ):
        setattr(state, name, 0.0)
    kernel = SignedShieldingKernel(
        active_kernel=np.ones((1, 8)), wake_kernel=np.zeros((1, 8)),
        active_x_m=state.x.copy(), wake_x_m=state.x.copy(),
        activation_to_line_content=np.array([1.0]),
        source_capacity_bounds=np.array([[0.1, 10.0]]),
        metadata={"schema": KERNEL_SCHEMA}, source_path="coupled-test",
    )
    install_signed_burgers_population(state, kernel, "validated_scalar")
    state._campaign_G_Pa = 1.0e9
    state._campaign_b = 1.0e-9
    state._campaign_backstress_scale = 1.0
    state._anisotropic_drive_reliable = True
    state._anisotropic_drive_factors = np.array([1.0])
    state._anisotropic_tau_signed_Pa = np.array([1.0])
    install_persistent_site_source(
        state,
        config=PersistentSiteConfig(
            rho_site0_m2=1.0e14, reference_source_area_m2=1.0e-12,
            reference_front_width_m=1.0e-6, reference_density_m2=1.0e10,
            source_zone_length_m=2.0e-6,
        ),
        r0_m=1.0e-6, b_m=1.0e-9,
    )
    return state


def _populate_coupled_test_state(state):
    state.mobile_positive[0, 1] = 0.7
    state.mobile_negative[0, 3] = 0.2
    state.retained_positive[0, 2] = 0.6
    state.retained_negative[0, 4] = 0.4
    state.accumulated_slip_positive[0, 0] = 0.3
    state.accumulated_slip_negative[0, 1] = 0.1
    state.mobile = state.mobile_positive + state.mobile_negative
    state.retained = state.retained_positive + state.retained_negative
    state.accumulated_slip = (
        state.accumulated_slip_positive + state.accumulated_slip_negative
    )
    return state


def test_continuous_source_backstress_update_converges_under_time_partition():
    def evolve(parts):
        state = _coupled_test_state()
        ledgers = np.zeros(5)
        for _ in range(parts):
            out = state._coupled_persistent_source_transport(
                0.5 / parts, 5.0e8, 950.0, 1.0e-9
            )
            result = out["result"]
            ledgers += (
                result.escaped, result.gross_trapping, result.gross_release,
                result.mobile_distance_population_m, out["emitted"],
            )
            assert out["coefficient_residual"] <= 1.0e-4
            assert out["complementarity_residual"] <= 1.0e-4
        vector = np.concatenate((
            state.mobile_positive.ravel(), state.mobile_negative.ravel(),
            state.retained_positive.ravel(), state.retained_negative.ravel(),
            state.accumulated_slip_positive.ravel(),
            state.accumulated_slip_negative.ravel(), ledgers,
        ))
        return vector

    one, two, four, eight, sixteen, thirty_two, sixty_four = (
        evolve(parts) for parts in (1, 2, 4, 8, 16, 32, 64)
    )
    e12 = np.linalg.norm(one - two)
    e24 = np.linalg.norm(two - four)
    e48 = np.linalg.norm(four - eight)
    e816 = np.linalg.norm(eight - sixteen)
    e1632 = np.linalg.norm(sixteen - thirty_two)
    e3264 = np.linalg.norm(thirty_two - sixty_four)
    assert e24 < e12
    assert e48 < e24
    assert e3264 < e1632 < e816 < e48
    assert e3264 / max(np.linalg.norm(sixty_four), 1.0) < 2.0e-3


def test_coupled_acceptance_recomputes_full_coefficients_at_accepted_state():
    state = _coupled_test_state()
    out = state._coupled_persistent_source_transport(1.0, 5.0e8, 950.0, 1.0e-9)
    accepted = state._transport_rates(
        state.local_stress_profile_Pa(5.0e8),
        state.local_forest_density_m2(False),
        950.0,
        1.0e-9,
    )
    for key in ("encounter", "taylor", "velocity"):
        np.testing.assert_allclose(out["rates"][key], accepted[key], rtol=1.0e-4, atol=1.0e-8)
    assert out["coefficient_residual"] <= 1.0e-4


@pytest.mark.parametrize(
    "mobile_recovery,retained_recovery",
    ((0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1)),
)
@pytest.mark.parametrize("populated", (False, True))
def test_coupled_source_uses_one_recovery_generator_for_all_affine_maps(
    mobile_recovery, retained_recovery, populated
):
    state = _coupled_test_state()
    if populated:
        _populate_coupled_test_state(state)
    state.cfg.mobile_recovery_rate_s = mobile_recovery
    state.manifest.retained_recovery_rate_s = retained_recovery
    initial_population = float(np.sum(state.mobile) + np.sum(state.retained))

    out = state._coupled_persistent_source_transport(
        1.0, 5.0e8, 950.0, 1.0e-9
    )
    result = out["result"]
    terminal_population = float(np.sum(result.mobile) + np.sum(result.retained))

    assert out["emitted"] > 0.0
    assert out["activation_rate_s"] > 0.0
    assert out["affine_direct_state_residual"] <= 1.0e-8
    assert out["coefficient_residual"] <= 1.0e-6
    assert initial_population + out["emitted"] == pytest.approx(
        terminal_population + result.escaped + result.recovered, abs=2.0e-10
    )
    if mobile_recovery > 0.0 or retained_recovery > 0.0:
        assert result.recovered > 0.0
    else:
        assert result.recovered == 0.0
        if not populated:
            assert out["emitted"] == pytest.approx(
                3.2917807917488653, rel=2.0e-13, abs=2.0e-13
            )
            assert result.escaped == pytest.approx(
                1.1938952951645873e-06, rel=2.0e-12, abs=2.0e-15
            )


def test_coupled_source_recovery_failure_rolls_back_exactly():
    state = _populate_coupled_test_state(_coupled_test_state())
    state.cfg.mobile_recovery_rate_s = 0.1
    state.manifest.retained_recovery_rate_s = 0.1
    state._persistent_site_cfg = replace(
        state._persistent_site_cfg,
        coupled_tolerance=1.0e-30,
        coupled_max_iterations=8,
    )
    arrays = {
        name: np.asarray(getattr(state, name)).copy()
        for name in (
            "mobile_positive", "mobile_negative", "retained_positive",
            "retained_negative", "accumulated_slip_positive",
            "accumulated_slip_negative",
        )
    }
    ledgers = {
        name: float(getattr(state, name))
        for name in (
            "emitted_total", "escaped_total", "recovered_total",
            "gross_trapping_total", "gross_release_total",
            "mobile_distance_population_total_m", "time_s",
        )
    }

    with pytest.raises(RuntimeError, match="fixed point did not converge"):
        state._coupled_persistent_source_transport(
            1.0, 5.0e8, 950.0, 1.0e-9
        )

    for name, expected in arrays.items():
        np.testing.assert_array_equal(getattr(state, name), expected)
    for name, expected in ledgers.items():
        assert getattr(state, name) == expected


def test_coupled_source_expands_bracket_when_blunting_raises_site_target():
    state = _coupled_test_state()

    def stronger_blunting(self, r0, _b):
        return float(r0) * (
            1.0 + 0.05 * float(np.sum(self.accumulated_slip))
        )

    state.blunted_radius = MethodType(stronger_blunting, state)
    out = state._coupled_persistent_source_transport(
        1.0, 5.0e8, 950.0, 1.0e-9
    )
    assert out["activation_rate_s"] > 0.0
    assert out["complementarity_residual"] <= 1.0e-6
    assert out["source_bracket_relative_width"] <= 1.0e-8
    assert out["state_fixed_point_residual"] <= 1.0e-4
    assert out["affine_direct_state_residual"] <= 1.0e-8


def test_outer_evolve_owns_coupled_flux_ledgers_once_and_conserves_source():
    state = _coupled_test_state()
    out = state.evolve(1.0, 950.0, 5.0e8, 1.0e-9)
    accounted = (
        state.mobile_count + state.retained_count
        + state.wake_mobile_count + state.wake_retained_count
        + state.escaped_total + state.recovered_total
        + state.wake_discarded_mobile_total + state.wake_discarded_retained_total
    )
    assert accounted == pytest.approx(state.emitted_total, abs=2.0e-10)
    assert state.emitted_total == pytest.approx(out["dN_emit"])
    assert state.escaped_total == pytest.approx(out["dN_escaped"])
    assert state.gross_trapping_total == pytest.approx(out["dN_gross_trapping"])
    assert state.gross_release_total == pytest.approx(out["dN_gross_release"])
    assert state.mobile_distance_population_total_m == pytest.approx(
        out["mobile_distance_population_m"])
    assert state.escaped_total > 0.0
    assert state.gross_trapping_total > 0.0
    assert state.gross_release_total > 0.0
    assert state.mobile_distance_population_total_m > 0.0


def test_outer_evolve_owns_nonzero_recovery_ledger_exactly_once():
    state = _populate_coupled_test_state(_coupled_test_state())
    state.cfg.mobile_recovery_rate_s = 0.1
    state.manifest.retained_recovery_rate_s = 0.1
    initial_population = (
        state.mobile_count + state.retained_count
        + state.wake_mobile_count + state.wake_retained_count
    )

    out = state.evolve(1.0, 950.0, 5.0e8, 1.0e-9)
    accounted = (
        state.mobile_count + state.retained_count
        + state.wake_mobile_count + state.wake_retained_count
        + state.escaped_total + state.recovered_total
        + state.wake_discarded_mobile_total + state.wake_discarded_retained_total
    )

    assert accounted == pytest.approx(
        initial_population + state.emitted_total, abs=2.0e-10
    )
    assert state.recovered_total == pytest.approx(out["dN_recovered"])
    assert state.recovered_total > 0.0
    assert state.escaped_total == pytest.approx(out["dN_escaped"])
    assert state.gross_trapping_total == pytest.approx(out["dN_gross_trapping"])
    assert state.gross_release_total == pytest.approx(out["dN_gross_release"])
