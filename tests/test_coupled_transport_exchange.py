import numpy as np
from types import SimpleNamespace

from arrhenius_fracture.coupled_transport_exchange import (
    _independent_stiff_action,
    _partitioned_affine_action,
    exchange_gross_flux,
    joint_transport_exchange,
)
from arrhenius_fracture.unified_mpz import UnifiedMPZState
from arrhenius_fracture.signed_burgers_shared_v1025 import _signed_evolve_scalar


KE_1200 = 2.5665736214707075e12
KR_1200 = 9.999999999556313e8
V_1200 = 1751.0613095657714
LENGTH = 50.0e-6


def _run(parts, *, ke=KE_1200, kr=KR_1200, velocity=V_1200, n=50):
    mobile = np.zeros((1, n))
    retained = np.zeros((1, n))
    retained[0, 0] = 1.0
    escaped = gross_trap = gross_release = distance = 0.0
    for _ in range(parts):
        result = joint_transport_exchange(
            mobile,
            retained,
            ke,
            kr,
            velocity_m_s=velocity,
            dt_s=1.0 / parts,
            dx_m=LENGTH / n,
        )
        mobile, retained = result.mobile, result.retained
        escaped += result.escaped
        gross_trap += result.gross_trapping
        gross_release += result.gross_release
        distance += result.mobile_distance_population_m
    return mobile, retained, escaped, gross_trap, gross_release, distance


def test_uniform_1200_benchmark_escapes_within_markov_bound():
    mobile, retained, escaped, trap, release, distance = _run(1)
    remaining = float(np.sum(mobile) + np.sum(retained))
    mean_exit = 1.0 / KR_1200 + LENGTH / V_1200 * (1.0 + KE_1200 / KR_1200)
    assert np.isclose(mean_exit, 7.33158e-5, rtol=2e-6)
    assert remaining <= mean_exit
    assert np.isclose(remaining + escaped, 1.0, atol=2e-8)
    assert trap >= 0.0 and release >= 0.0
    assert distance >= 0.0


def test_stiff_asymptotic_gross_flux_and_distance_are_partition_invariant():
    one = _run(1)
    ten = _run(10)
    for index, (left, right) in enumerate(zip(one, ten)):
        # Gross trapping/release integrate O(1e5) stiff turnovers; their
        # independent augmented-state quadrature is converged to <1e-9
        # relative while populations, escape and distance are much tighter.
        tolerance = 3.0e-9 if index in (3, 4) else 2.0e-12
        assert np.allclose(left, right, rtol=tolerance, atol=2e-12)


def test_legacy_one_exchange_then_unrestricted_advection_fails_benchmark():
    mobile = np.zeros((1, 50)); retained = np.zeros((1, 50)); retained[0, 0] = 1.0
    mobile, retained, _, _ = UnifiedMPZState._exchange(
        mobile, retained, np.full(50, KE_1200), np.full(50, KR_1200), 1.0
    )
    mobile, escaped = UnifiedMPZState._advect_forward(mobile, V_1200, LENGTH / 50)
    remaining = float(np.sum(mobile) + np.sum(retained))
    expected = KE_1200 / (KE_1200 + KR_1200)
    assert np.isclose(remaining, expected, rtol=2e-13)
    assert np.isclose(remaining, 0.9996105272, rtol=2e-10)
    assert remaining > 1.0e4 * 7.33158e-5
    assert np.isclose(remaining + escaped, 1.0, atol=2e-15)


def test_actual_validated_scalar_binding_selects_joint_complete_block(monkeypatch):
    def state():
        n = 80
        s = SimpleNamespace(
            n_systems=1, n_bins=n, dx=LENGTH/n,
            mobile_positive=np.zeros((1,n)), mobile_negative=np.zeros((1,n)),
            retained_positive=np.zeros((1,n)), retained_negative=np.zeros((1,n)),
            accumulated_slip_positive=np.zeros((1,n)), accumulated_slip_negative=np.zeros((1,n)),
            cfg=SimpleNamespace(source_bin_count=1, mobile_recovery_rate_s=0.0),
            manifest=SimpleNamespace(retained_recovery_rate_s=0.0),
            escaped_total=0.0, recovered_total=0.0,
            net_retained_gain_total=0.0, net_retained_loss_total=0.0,
            gross_trapping_total=0.0, gross_release_total=0.0,
            mobile_distance_population_total_m=0.0, time_s=0.0,
            wake_mobile_count=0.0, wake_retained_count=0.0,
            available_site_fraction=1.0, signed_last_source_activations=0.0,
        )
        s.retained_positive[0,0] = 1.0
        s.mobile = s.mobile_positive + s.mobile_negative
        s.retained = s.retained_positive + s.retained_negative
        s.accumulated_slip = s.accumulated_slip_positive + s.accumulated_slip_negative
        s._emit = lambda *args: 0.0
        s.local_stress_profile_Pa = lambda stress: np.zeros(n)
        s.local_forest_density_m2 = lambda wake: np.ones(n)
        s._transport_rates = lambda *args: {
            "peierls": np.ones(n), "taylor": np.full(n, KR_1200),
            "encounter": np.full(n, KE_1200), "m": np.ones(n),
            "velocity": np.full(n, V_1200),
        }
        s._exchange = UnifiedMPZState._exchange
        s._advect_forward = UnifiedMPZState._advect_forward
        return s
    monkeypatch.setenv("MPZ_TRANSPORT_EXCHANGE_MODE", "joint_ctmc")
    joint = state()
    info = _signed_evolve_scalar(joint, 1.0, 1200.0, 0.0, 1.0)
    assert info["exchange_transport_mode"] == "joint_ctmc"
    assert float(np.sum(joint.mobile + joint.retained)) <= 7.33158e-5
    assert np.isclose(joint.escaped_total, 1.0, atol=2e-12)
    monkeypatch.setenv("MPZ_TRANSPORT_EXCHANGE_MODE", "legacy_split")
    legacy = state()
    info = _signed_evolve_scalar(legacy, 1.0, 1200.0, 0.0, 1.0)
    assert info["exchange_transport_mode"] == "legacy_split"
    assert np.isclose(float(np.sum(legacy.mobile + legacy.retained)), 0.9996105272, rtol=2e-10)


def test_joint_operator_is_partition_invariant_and_flux_identity_holds():
    one = _run(1, ke=20.0, kr=3.0, velocity=2.0e-5)
    ten = _run(10, ke=20.0, kr=3.0, velocity=2.0e-5)
    for left, right in zip(one, ten):
        assert np.allclose(left, right, rtol=2e-11, atol=2e-11)
    mobile, retained, _, trap, release, _ = one
    assert np.min(mobile) >= 0.0 and np.min(retained) >= 0.0
    assert np.isclose(trap - release, float(np.sum(retained)) - 1.0, atol=2e-11)


def test_exchange_only_gross_flux_is_positive_partition_invariant_and_exact():
    m0 = np.array([[0.2, 0.9]]); r0 = np.array([[0.8, 0.1]])
    ke = np.array([20.0, 0.5]); kr = np.array([3.0, 8.0]); dt = 0.7
    whole = UnifiedMPZState._exchange(m0, r0, ke, kr, dt)
    qt, qr = exchange_gross_flux(m0, r0, ke, kr, dt)
    assert qt > 0.0 and qr > 0.0
    assert np.isclose(qt - qr, float(np.sum(whole[1] - r0)), atol=2e-14)
    m, r = m0.copy(), r0.copy(); qtp = qrp = 0.0
    for _ in range(7):
        a, b = exchange_gross_flux(m, r, ke, kr, dt / 7)
        qtp += a; qrp += b
        m, r = UnifiedMPZState._exchange(m, r, ke, kr, dt / 7)[:2]
    assert np.isclose(qtp, qt, rtol=2e-14)
    assert np.isclose(qrp, qr, rtol=2e-14)


def test_signed_species_and_feedback_moment_are_partition_invariant():
    n = 30
    m0 = np.zeros((2, n)); r0 = np.zeros((2, n))
    m0[0, 2] = 0.4; r0[0, 4] = 0.7
    m0[1, 1] = 0.2; r0[1, 8] = 0.3
    def evolve(parts):
        m, r = m0.copy(), r0.copy()
        totals = np.zeros(3)
        for _ in range(parts):
            out = joint_transport_exchange(m, r, 7.0, 2.5,
                velocity_m_s=8e-6, dt_s=0.4 / parts, dx_m=LENGTH / n)
            m, r = out.mobile, out.retained
            totals += (out.escaped, out.gross_trapping, out.gross_release)
        x = (np.arange(n) + 0.5) * LENGTH / n
        moment = float(np.sum((r[0] - r[1]) / np.sqrt(x)))
        return m, r, totals, moment
    one, many = evolve(1), evolve(8)
    assert np.allclose(one[0], many[0], rtol=2e-11, atol=2e-11)
    assert np.allclose(one[1], many[1], rtol=2e-11, atol=2e-11)
    assert np.allclose(one[2], many[2], rtol=2e-11, atol=2e-11)
    assert np.isclose(one[3], many[3], rtol=2e-11, atol=2e-11)


def test_continuous_mobile_source_is_conservative_and_partition_invariant():
    n = 24
    source = np.zeros((2, n))
    source[0, :2] = 3.0
    source[1, :2] = 0.75

    def evolve(parts):
        mobile = np.zeros_like(source)
        retained = np.zeros_like(source)
        escaped = trap = release = distance = 0.0
        for _ in range(parts):
            out = joint_transport_exchange(
                mobile,
                retained,
                20.0,
                3.0,
                velocity_m_s=2.0e-5,
                dt_s=0.8 / parts,
                dx_m=LENGTH / n,
                mobile_source_rate=source,
            )
            mobile, retained = out.mobile, out.retained
            escaped += out.escaped
            trap += out.gross_trapping
            release += out.gross_release
            distance += out.mobile_distance_population_m
        return mobile, retained, escaped, trap, release, distance

    one = evolve(1)
    eight = evolve(8)
    for left, right in zip(one, eight):
        np.testing.assert_allclose(left, right, rtol=3e-11, atol=3e-11)
    emitted = float(np.sum(source)) * 0.8
    assert np.isclose(np.sum(one[0]) + np.sum(one[1]) + one[2], emitted, atol=2e-11)
    assert np.isclose(one[3] - one[4], np.sum(one[1]), atol=2e-11)


def test_stiff_continuous_source_is_partition_invariant():
    n = 40
    source = np.zeros((1, n))
    source[0, :2] = 5.0

    def evolve(parts):
        mobile = np.zeros_like(source)
        retained = np.zeros_like(source)
        escaped = distance = 0.0
        methods = set()
        for _ in range(parts):
            out = joint_transport_exchange(
                mobile,
                retained,
                KE_1200,
                KR_1200,
                velocity_m_s=V_1200,
                dt_s=0.2 / parts,
                dx_m=LENGTH / n,
                mobile_source_rate=source,
            )
            mobile, retained = out.mobile, out.retained
            escaped += out.escaped
            distance += out.mobile_distance_population_m
            methods.add(out.stiff_solver_method)
        return mobile, retained, escaped, distance, methods

    one = evolve(1)
    four = evolve(4)
    for left, right in zip(one[:4], four[:4]):
        np.testing.assert_allclose(left, right, rtol=3e-11, atol=3e-11)
    emitted = float(np.sum(source)) * 0.2
    assert np.isclose(np.sum(one[0]) + np.sum(one[1]) + one[2], emitted, atol=2e-8)
    assert "stable_population_expm_flux_identity" in one[4]


def test_stiff_source_projection_preserves_each_species_budget():
    n = 80
    source = np.zeros((2, n))
    source[0, :2] = 1.915
    source[1, :2] = 0.637
    mobile = np.zeros_like(source)
    retained = np.zeros_like(source)
    out = joint_transport_exchange(
        mobile,
        retained,
        np.full(n, 2.0e12),
        np.full(n, 1.0e9),
        velocity_m_s=1700.0,
        dt_s=1.0,
        dx_m=LENGTH / n,
        mobile_source_rate=source,
    )
    budget = np.sum(source, axis=1)
    remaining = np.sum(out.mobile + out.retained, axis=1)
    assert np.all(remaining <= budget * (1.0 + 2.0e-14))
    assert np.isclose(np.sum(remaining) + out.escaped, np.sum(budget), atol=2e-12)


def test_large_transport_partition_count_uses_same_semigroup():
    matrix = np.array([[-2.0, 0.0], [2.0, 0.0]])
    vectors = np.array([[1.0], [0.0]])
    from arrhenius_fracture.coupled_transport_exchange import _partitioned_exponential_action
    actual = _partitioned_exponential_action(
        matrix, vectors, 10.0, activity=1.0e8, limit=1.0e4
    )
    expected = np.array([[np.exp(-20.0)], [1.0 - np.exp(-20.0)]])
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_active_950_scale_large_hop_preserves_small_species_budget():
    n = 80
    mobile = np.zeros((1, n))
    retained = np.zeros((1, n))
    retained[0, 0] = 2.202448e-2
    out = joint_transport_exchange(
        mobile, retained, KE_1200, KR_1200,
        velocity_m_s=V_1200, dt_s=21.0, dx_m=LENGTH / n,
    )
    remaining = float(np.sum(out.mobile) + np.sum(out.retained))
    assert remaining <= retained[0, 0]
    assert np.isclose(remaining + out.escaped, retained[0, 0], atol=2e-10)
    assert out.invariant_projection_correction <= 3e-10


def test_stiff_affine_source_and_diagnostics_match_independent_radau_oracle():
    # [m0,m1,r0,r1,escape,gross_trap,gross_release,distance,constant_source]
    matrix = np.zeros((9, 9))
    ke, kr, hop, source, dt = 3.0e4, 7.0e4, 2.0e3, 2.5, 0.02
    for i in range(2):
        mi, ri = i, 2 + i
        matrix[mi, mi] -= ke + hop
        matrix[ri, mi] += ke
        matrix[mi, ri] += kr
        matrix[ri, ri] -= kr
        matrix[mi + 1 if i == 0 else 4, mi] += hop
        matrix[5, mi] += ke
        matrix[6, ri] += kr
        matrix[7, mi] += 4.0e-7
    matrix[0, 8] = source
    initial = np.zeros((9, 1))
    initial[8, 0] = 1.0
    composed = _partitioned_affine_action(
        matrix, initial, dt, activity=(ke + kr + hop) * dt,
        dynamic_size=8,
        invariant_weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        correction_row=4,
    )
    independent = _independent_stiff_action(matrix, initial, dt)
    np.testing.assert_allclose(composed[:5], independent[:5], rtol=2e-8, atol=2e-11)
    np.testing.assert_allclose(composed[5:8], independent[5:8], rtol=2e-8, atol=2e-11)
    assert np.isclose(np.sum(composed[:5]), source * dt, rtol=0.0, atol=2e-14)
    assert np.isclose(composed[5] - composed[6], np.sum(composed[2:4]), atol=2e-11)


def test_closed_stiff_exchange_does_not_create_escape_or_material_correction():
    mobile = np.array([[2.0e-9, 0.0]])
    retained = np.array([[0.0, 3.0e-9]])
    out = joint_transport_exchange(
        mobile, retained,
        encounter_rate_s=np.array([1.0e-7, 1.0e-6]),
        release_rate_s=np.array([5.69e10, 5.69e10]),
        velocity_m_s=np.zeros(2), dt_s=4.2, dx_m=1.0e-6,
    )
    assert out.escaped == 0.0
    assert out.recovered == 0.0
    np.testing.assert_allclose(
        out.mobile + out.retained, mobile + retained, rtol=2.0e-10, atol=2.0e-18)
    assert out.invariant_projection_correction <= 2.0e-18
    assert out.base_map_invariant_correction <= 2.0e-18
    assert out.positivity_correction <= 2.0e-18


def test_full_attempt_release_scale_uses_error_controlled_same_equation_retry():
    # Same stiffness separation as the failed full-native state: Taylor release
    # O(5.69e10/s) beside O(1e-6/s) encounter/transport, with continuous source.
    matrix = np.zeros((9, 9))
    ke, kr, hop, source, dt = 1.0e-6, 5.69e10, 1.0e-6, 0.02, 4.2
    for i in range(2):
        mi, ri = i, 2 + i
        matrix[mi, mi] -= ke + hop
        matrix[ri, mi] += ke
        matrix[mi, ri] += kr
        matrix[ri, ri] -= kr
        matrix[mi + 1 if i == 0 else 4, mi] += hop
        matrix[5, mi] += ke
        matrix[6, ri] += kr
        matrix[7, mi] += hop
    matrix[0, 8] = source
    initial = np.zeros((9, 1))
    initial[2, 0] = 1.0e-8
    initial[8, 0] = 1.0
    out = joint_transport_exchange(
        np.zeros((1, 2)), np.array([[1.0e-8, 0.0]]),
        encounter_rate_s=np.full(2, ke),
        release_rate_s=np.full(2, kr),
        velocity_m_s=np.full(2, hop), dt_s=dt, dx_m=1.0,
        mobile_source_rate=np.array([[source, 0.0]]),
    )
    independent = _independent_stiff_action(matrix, initial, dt)
    accepted = np.r_[
        out.mobile.ravel(), out.retained.ravel(), out.escaped,
        out.gross_trapping, out.gross_release,
        out.mobile_distance_population_m,
    ]
    np.testing.assert_allclose(accepted[:5], independent[:5, 0], rtol=2e-8, atol=2e-15)
    np.testing.assert_allclose(accepted[5:8], independent[5:8, 0], rtol=2e-8, atol=2e-15)
    expected_budget = 1.0e-8 + source * dt
    assert np.isclose(np.sum(accepted[:5]), expected_budget, atol=2e-14)
    assert out.error_controlled_stiff_solve
    assert out.affine_invariant_defect > 1.0e-8
    assert out.affine_retry_discrepancy > 1.0e-8
    assert out.complete_numerical_correction <= 2.0e-14


def test_explicit_zero_recovery_is_operator_neutral():
    mobile = np.array([[0.3, 0.1, 0.0], [0.0, 0.2, 0.1]])
    retained = np.array([[0.4, 0.0, 0.2], [0.1, 0.3, 0.0]])
    source = np.array([[0.7, 0.0, 0.0], [0.2, 0.0, 0.0]])
    kwargs = dict(
        encounter_rate_s=np.array([4.0, 5.0, 6.0]),
        release_rate_s=np.array([3.0, 2.0, 1.0]),
        velocity_m_s=np.array([2.0e-6, 1.5e-6, 1.0e-6]),
        dt_s=0.4,
        dx_m=1.0e-6,
        mobile_source_rate=source,
    )
    omitted = joint_transport_exchange(mobile, retained, **kwargs)
    explicit = joint_transport_exchange(
        mobile,
        retained,
        mobile_recovery_rate_s=0.0,
        retained_recovery_rate_s=0.0,
        **kwargs,
    )
    np.testing.assert_array_equal(explicit.mobile, omitted.mobile)
    np.testing.assert_array_equal(explicit.retained, omitted.retained)
    for name in (
        "escaped", "recovered", "net_retained_gain", "net_retained_loss",
        "gross_trapping", "gross_release", "mobile_distance_population_m",
        "conservation_error", "invariant_projection_correction",
        "base_map_invariant_correction", "positivity_correction",
        "complete_numerical_correction", "affine_invariant_defect",
        "affine_retry_discrepancy",
    ):
        assert getattr(explicit, name) == getattr(omitted, name)


def test_recovery_source_operator_is_affine_for_populations_and_flux_integrals():
    source_unit = np.zeros((2, 6))
    source_unit[0, :2] = (0.7, 0.3)
    source_unit[1, :2] = (0.2, 0.1)
    initial_states = (
        (np.zeros((2, 6)), np.zeros((2, 6))),
        (
            np.array([
                [0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.3, 0.1, 0.0, 0.0, 0.0],
            ]),
            np.array([
                [0.4, 0.0, 0.2, 0.0, 0.0, 0.0],
                [0.1, 0.2, 0.0, 0.1, 0.0, 0.0],
            ]),
        ),
    )
    scalar_fields = (
        "escaped", "recovered", "gross_trapping", "gross_release",
        "mobile_distance_population_m",
    )
    for mobile_recovery, retained_recovery in (
        (0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1)
    ):
        frozen = dict(
            encounter_rate_s=np.linspace(4.0, 9.0, 6),
            release_rate_s=np.linspace(3.0, 1.0, 6),
            velocity_m_s=np.linspace(2.0e-6, 0.5e-6, 6),
            dt_s=0.6,
            dx_m=1.0e-6,
            mobile_recovery_rate_s=mobile_recovery,
            retained_recovery_rate_s=retained_recovery,
        )
        for mobile, retained in initial_states:
            base = joint_transport_exchange(mobile, retained, **frozen)
            unit = joint_transport_exchange(
                mobile, retained, mobile_source_rate=source_unit, **frozen
            )
            for amplitude in (0.25, 1.0, 2.5):
                direct = joint_transport_exchange(
                    mobile,
                    retained,
                    mobile_source_rate=amplitude * source_unit,
                    **frozen,
                )
                np.testing.assert_allclose(
                    direct.mobile,
                    base.mobile + amplitude * (unit.mobile - base.mobile),
                    rtol=2.0e-12,
                    atol=2.0e-13,
                )
                np.testing.assert_allclose(
                    direct.retained,
                    base.retained + amplitude * (unit.retained - base.retained),
                    rtol=2.0e-12,
                    atol=2.0e-13,
                )
                for name in scalar_fields:
                    predicted = getattr(base, name) + amplitude * (
                        getattr(unit, name) - getattr(base, name)
                    )
                    np.testing.assert_allclose(
                        getattr(direct, name), predicted,
                        rtol=3.0e-12,
                        atol=3.0e-13,
                    )
