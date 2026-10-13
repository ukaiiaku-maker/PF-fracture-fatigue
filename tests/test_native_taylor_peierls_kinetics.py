import numpy as np

from arrhenius_fracture.coupled_transport_exchange import joint_transport_exchange
from arrhenius_fracture.material_manifest import MaterialManifest, default_manifest_path
from arrhenius_fracture.taylor_release import signed_arrhenius_components
from arrhenius_fracture.unified_mpz import MPZConfig, UnifiedMPZState


def _state(n=8):
    manifest = MaterialManifest.from_csv(default_manifest_path("DBTT"))
    return UnifiedMPZState(manifest, MPZConfig(n_bins=n, length_m=8.0e-6))


def test_native_transport_matches_forward_reverse_and_mean_order_identity():
    state = _state()
    stress = np.linspace(1.0e7, 7.0e8, state.n_bins)
    rho = np.geomspace(5.0e12, 5.0e16, state.n_bins)
    b = 2.74e-10
    rates = state._transport_rates(stress, rho, 950.0, b)
    spacing = 1.0 / (2.0 * np.sqrt(rho))
    p = signed_arrhenius_components(
        state.manifest.peierls.as_surface(state.manifest.emission),
        state.cfg.peierls_stress_fraction * stress, 950.0)
    t = signed_arrhenius_components(
        state.manifest.taylor.as_surface(state.manifest.emission),
        state.cfg.taylor_stress_fraction * stress * spacing / b, 950.0)
    order = 1.0 + state.manifest.taylor_corr_scale * np.sqrt(
        rho / state.manifest.taylor_corr_rho_c_m2)
    np.testing.assert_allclose(rates["peierls"], p["net_rate_s"], rtol=2e-15)
    np.testing.assert_allclose(rates["velocity"], spacing * p["net_rate_s"], rtol=2e-15)
    np.testing.assert_allclose(rates["taylor"], t["net_rate_s"] / order, rtol=2e-15)
    np.testing.assert_allclose(rates["m"], order, rtol=2e-15)


def test_zero_bias_detailed_balance_and_peierls_reversal_are_exact():
    state = _state()
    rho = np.full(state.n_bins, 5.0e12)
    positive = state._transport_rates(np.full(state.n_bins, 4.0e8), rho, 950.0, 2.74e-10)
    zero = state._transport_rates(np.zeros(state.n_bins), rho, 950.0, 2.74e-10)
    negative = state._transport_rates(np.full(state.n_bins, -4.0e8), rho, 950.0, 2.74e-10)
    assert np.array_equal(zero["peierls"], np.zeros(state.n_bins))
    assert np.array_equal(zero["taylor"], np.zeros(state.n_bins))
    np.testing.assert_allclose(negative["velocity"], -positive["velocity"], rtol=2e-15)
    assert np.array_equal(negative["taylor"], np.zeros(state.n_bins))


def test_native_rates_have_no_phi_cutoff_and_are_mesh_independent():
    b = 2.74e-10
    stress = np.full(4, 5.0e8)
    # spacing/b is far beyond the inherited donor's former value 20.
    rho = np.full(4, 1.0e8)
    coarse = _state(4)._transport_rates(stress, rho, 950.0, b)
    fine = _state(16)._transport_rates(
        np.repeat(stress, 4), np.repeat(rho, 4), 950.0, b)
    assert np.all(coarse["jump"] / b > 20.0)
    np.testing.assert_allclose(fine["taylor"][::4], coarse["taylor"], rtol=2e-15)
    np.testing.assert_allclose(fine["velocity"][::4], coarse["velocity"], rtol=2e-15)


def test_joint_generator_consumes_the_spatial_velocity_field():
    mobile = np.zeros((1, 3)); retained = np.zeros_like(mobile)
    mobile[0, 0] = 1.0
    stopped = joint_transport_exchange(
        mobile, retained, 0.0, 0.0, velocity_m_s=np.zeros(3),
        dt_s=1.0, dx_m=1.0)
    graded = joint_transport_exchange(
        mobile, retained, 0.0, 0.0, velocity_m_s=np.array([1.0, 0.0, 0.0]),
        dt_s=1.0, dx_m=1.0)
    np.testing.assert_array_equal(stopped.mobile, mobile)
    assert graded.mobile[0, 0] < mobile[0, 0]
    assert graded.mobile[0, 1] > 0.0
    assert graded.mobile[0, 2] == 0.0
