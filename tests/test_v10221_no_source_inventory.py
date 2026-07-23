from types import SimpleNamespace

import numpy as np

import arrhenius_fracture.persistent_site_source_v10221 as model


def test_emission_adds_signed_line_content_without_consuming_sites(monkeypatch):
    state = SimpleNamespace()
    state.n_systems = 1
    state.n_bins = 4
    state.length_m = 4.0e-6
    state.dx = 1.0e-6
    state.x = (np.arange(state.n_bins) + 0.5) * state.dx
    state.cfg = SimpleNamespace(
        source_bin_count=2,
        blunting_length_m=1.0e-6,
        forest_density_floor_m2=5.0e12,
        taylor_stress_fraction=1.0,
    )
    state._persistent_site_cfg = model.PersistentSiteConfig(
        rho_site0_m2=1.0e12,
        reference_source_area_m2=1.0e-12,
        reference_front_width_m=1.0e-6,
        reference_density_m2=5.0e12,
        source_zone_length_m=2.0e-6,
        maximum_front_width_m=4.0e-6,
    )
    state._persistent_r0_m = 1.0e-6
    state._persistent_b = 2.5e-10
    state._persistent_active_arc_factor = 1.0
    state._campaign_G_Pa = 1.0e9
    state._campaign_b = 2.5e-10
    state._campaign_backstress_scale = 1.0
    state._anisotropic_drive_reliable = True
    state._anisotropic_tau_signed_Pa = np.array([1.0])
    state._signed_kernel = SimpleNamespace(
        activation_to_line_content=np.array([1.0])
    )
    shape = (1, state.n_bins)
    state.mobile_positive = np.zeros(shape)
    state.mobile_negative = np.zeros(shape)
    state.retained_positive = np.zeros(shape)
    state.retained_negative = np.zeros(shape)
    state.accumulated_slip_positive = np.zeros(shape)
    state.accumulated_slip_negative = np.zeros(shape)
    state.mobile = np.zeros(shape)
    state.retained = np.zeros(shape)
    state.accumulated_slip = np.zeros(shape)
    state.site_capacity = np.array([17.0])
    state.available_sites = state.site_capacity.copy()
    state.tip_source_activity = np.ones(1)
    state.emitted_total = 0.0
    state.signed_source_activations_total = 0.0
    state.signed_line_content_emitted_total = 0.0
    state.blunted_radius = lambda r0, b: r0
    state.emission_rate_per_site = lambda stress, temperature: 1.0

    monkeypatch.setattr(model, "_drive_factors_for_state", lambda obj: np.array([1.0]))
    monkeypatch.setattr(
        model,
        "_campaign_local_density_m2",
        lambda obj: np.array([0.0]),
    )
    monkeypatch.setattr(
        model,
        "_campaign_backstress",
        lambda obj: (np.array([0.0]), np.array([0.0]), np.array([0.0])),
    )

    def sync(obj):
        obj.mobile = obj.mobile_positive + obj.mobile_negative
        obj.retained = obj.retained_positive + obj.retained_negative
        obj.accumulated_slip = (
            obj.accumulated_slip_positive + obj.accumulated_slip_negative
        )

    monkeypatch.setattr(model, "_sync_active", sync)
    before = state.available_sites.copy()
    emitted = model._persistent_emit(state, 1.0e-3, 1.0e6, 300.0)

    assert emitted > 0.0
    assert np.sum(state.mobile_positive) > 0.0
    assert np.array_equal(state.available_sites, before)
    assert np.all(state.tip_source_activity == 1.0)
