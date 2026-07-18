from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.anisotropic_emission_v10174 import AnisotropicEmissionConfig
from arrhenius_fracture.kinetic_tip_cell import KineticTipConfig
from arrhenius_fracture.reduced_shared_state_v1025 import ExactProductionConfig
from arrhenius_fracture.signed_burgers_shared_v1025 import (
    KERNEL_SCHEMA,
    SignedShieldingKernel,
    install_signed_burgers_population,
)
from arrhenius_fracture.unified_mpz import MPZConfig, UnifiedMPZState


class FakeState:
    _advect_forward = staticmethod(UnifiedMPZState._advect_forward)

    def __init__(self):
        self.n_systems = 2
        self.n_bins = 3
        self.dx = 1.0
        self.x = np.array([0.5, 1.5, 2.5])
        self.wake_n_bins = 3
        self.wake_dx = 1.0
        self.wake_x = np.array([0.5, 1.5, 2.5])
        self.cfg = SimpleNamespace(
            mobile_shield_fraction=1.0,
            wake_shielding=True,
            source_bin_count=1,
            mobile_recovery_rate_s=0.0,
            blunting_length_m=1.0,
        )
        self.manifest = SimpleNamespace(source_refresh_length_m=2.0)
        self.site_capacity = np.array([10.0, 10.0])
        self.available_sites = self.site_capacity.copy()
        self.mobile = np.zeros((2, 3))
        self.retained = np.zeros((2, 3))
        self.accumulated_slip = np.zeros((2, 3))
        self.wake_mobile = np.zeros((2, 3))
        self.wake_retained = np.zeros((2, 3))
        self.wake_slip = np.zeros((2, 3))
        self.wake_discarded_mobile_total = 0.0
        self.wake_discarded_retained_total = 0.0
        self.wake_discarded_slip_total = 0.0
        self.advance_total_m = 0.0
        self._campaign_refresh_scale = 1.0
        self.campaign_source_last_refresh_fraction = 0.0
        self.campaign_source_last_refresh_length_m = 0.0
        self.emitted_total = 0.0
        self.escaped_total = 0.0
        self.recovered_total = 0.0

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
        return float(r0)


def kernel(active=None, bounds=None):
    active = np.asarray(
        active if active is not None else [[2.0, 1.0, 0.5], [-3.0, -1.5, -0.75]],
        dtype=float,
    )
    return SignedShieldingKernel(
        active_kernel=active,
        wake_kernel=0.5 * active,
        active_x_m=np.array([0.5, 1.5, 2.5]),
        wake_x_m=np.array([0.5, 1.5, 2.5]),
        activation_to_line_content=np.array([1.0, 1.0]),
        source_capacity_bounds=np.asarray(bounds if bounds is not None else [[1.0, 20.0], [1.0, 20.0]]),
        metadata={"schema": KERNEL_SCHEMA},
        source_path="unit-test.json",
    )


def test_equal_opposite_burgers_content_cancels_shielding_not_density():
    state = FakeState()
    install_signed_burgers_population(state, kernel(), "validated_scalar")
    state.retained_positive[0, 0] = 5.0
    state.retained_negative[0, 0] = 5.0
    state.retained = state.retained_positive + state.retained_negative
    assert state.retained_count == pytest.approx(10.0)
    assert state.active_K_shielding() == pytest.approx(0.0)


def test_reversing_burgers_sign_reverses_K_and_channel_kernel_allows_antishielding():
    state = FakeState()
    install_signed_burgers_population(state, kernel(), "validated_scalar")
    state.retained_positive[0, 0] = 2.0
    state.retained = state.retained_positive + state.retained_negative
    positive = state.active_K_shielding()
    state.retained_positive[:] = 0.0
    state.retained_negative[0, 0] = 2.0
    state.retained = state.retained_positive + state.retained_negative
    negative = state.active_K_shielding()
    assert positive == pytest.approx(-negative)
    state.retained_negative[:] = 0.0
    state.retained_positive[1, 0] = 2.0
    state.retained = state.retained_positive + state.retained_negative
    assert state.active_K_shielding() < 0.0


def test_moving_frame_preserves_positive_and_negative_species_separately():
    state = FakeState()
    install_signed_burgers_population(state, kernel(), "validated_scalar")
    state.mobile_positive[0, 0] = 4.0
    state.mobile_negative[1, 1] = 3.0
    state.mobile = state.mobile_positive + state.mobile_negative
    before_positive = float(np.sum(state.mobile_positive) + np.sum(state.wake_mobile_positive))
    before_negative = float(np.sum(state.mobile_negative) + np.sum(state.wake_mobile_negative))
    state.advance(0.25)
    after_positive = float(np.sum(state.mobile_positive) + np.sum(state.wake_mobile_positive))
    after_negative = float(np.sum(state.mobile_negative) + np.sum(state.wake_mobile_negative))
    assert after_positive == pytest.approx(before_positive)
    assert after_negative == pytest.approx(before_negative)


def test_old_thousands_site_anchor_is_rejected_by_physical_source_bounds():
    state = FakeState()
    state.site_capacity[:] = 4639.7
    state.available_sites = state.site_capacity.copy()
    with pytest.raises(ValueError, match="source-capacity range"):
        install_signed_burgers_population(state, kernel(bounds=[[10, 200], [10, 200]]), "validated_scalar")


def test_exact_trace_configuration_preserves_local_30_GPa_strength_limit():
    payload = {
        "front_config": {
            "r0": 1.0e-6,
            "L_pz": 2.0e-6,
            "da": 5.0e-6,
            "sigma_cap": 30.0e9,
            "m_hits": 3.0,
            "tau_c": 1.0e-6,
        },
        "mpz_config": asdict(MPZConfig()),
        "tip_config": asdict(KineticTipConfig()),
        "anisotropic_config": asdict(AnisotropicEmissionConfig()),
        "G_Pa": 160.0e9,
        "poisson": 0.28,
        "b_m": 2.74e-10,
        "transport_mode": "validated_scalar",
        "campaign_config": {"backstress_scale": 1.25, "refresh_scale": 0.75},
    }
    config = ExactProductionConfig.from_trace(payload, "kernel.json")
    assert config.front_config["sigma_cap"] == pytest.approx(30.0e9)
    assert config.front_config["L_pz"] == pytest.approx(2.0e-6)
    assert config.tip_config["packet_length_m"] == pytest.approx(KineticTipConfig().packet_length_m)
    assert config.campaign_backstress_scale == pytest.approx(1.25)
    assert config.campaign_refresh_scale == pytest.approx(0.75)
