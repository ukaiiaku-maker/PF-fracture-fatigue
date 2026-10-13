"""Conservative joint mobile/retained exchange and forward transport.

The production legacy reference applies exact local exchange and then a single
semi-Lagrangian mobile advection.  This module instead exponentiates the frozen-
coefficient finite-volume generator for exchange, recovery, mobile transport,
escape, and diagnostic flux integrals together. Persistent-site installation
selects it explicitly; the legacy split remains an explicitly selected
historical/alternative route.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class CoupledTransportExchangeResult:
    mobile: np.ndarray
    retained: np.ndarray
    escaped: float
    recovered: float
    net_retained_gain: float
    net_retained_loss: float
    gross_trapping: float
    gross_release: float
    mobile_distance_population_m: float
    conservation_error: float
    invariant_projection_correction: float = 0.0


def exchange_gross_flux(mobile, retained, encounter_rate_s, release_rate_s, dt_s):
    """Exact gross trapping/release for a frozen local exchange interval."""
    m0 = np.maximum(np.asarray(mobile, dtype=float), 0.0)
    r0 = np.maximum(np.asarray(retained, dtype=float), 0.0)
    if m0.shape != r0.shape or m0.ndim != 2:
        raise ValueError("mobile and retained must have the same 2-D shape")
    n = m0.shape[1]
    ke = _rates(encounter_rate_s, n, "encounter_rate_s")[None, :]
    kr = _rates(release_rate_s, n, "release_rate_s")[None, :]
    dt = max(float(dt_s), 0.0)
    total = m0 + r0
    rate = ke + kr
    req = np.divide(ke * total, rate, out=np.zeros_like(total), where=rate > 0.0)
    phi = np.divide(
        -np.expm1(-np.minimum(rate * dt, 700.0)),
        rate,
        out=np.full_like(total, dt),
        where=rate > 0.0,
    )
    integral_r = req * dt + (r0 - req) * phi
    release = float(np.sum(kr * integral_r))
    trapping = float(np.sum(ke * (total * dt - integral_r)))
    return max(trapping, 0.0), max(release, 0.0)


def _rates(values, n: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(n, float(array))
    array = np.broadcast_to(array.reshape(-1), (n,)).copy()
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return array


def _partitioned_exponential_action(matrix, vectors, dt, activity, limit=1.0e2):
    """Apply one autonomous semigroup with a numerically bounded exponent."""
    parts = max(1, int(np.ceil(max(float(activity), 0.0) / float(limit))))
    propagator = expm(np.asarray(matrix, dtype=float) * (float(dt) / parts))
    # Binary exponentiation preserves the same partitioned autonomous
    # semigroup while avoiding O(parts) dense products in the difficult
    # high-transport regime.  Direct expm(matrix*dt) is deliberately avoided:
    # the bounded exponent is what protects the stiff population invariants.
    return np.linalg.matrix_power(propagator, parts) @ np.asarray(vectors, dtype=float)


def _partitioned_affine_action(
    matrix, vectors, dt, activity, dynamic_size: int, limit=1.0
):
    """Compose the exact bounded-step affine map without state projection."""
    parts = max(1, int(np.ceil(max(float(activity), 0.0) / float(limit))))
    propagator = expm(np.asarray(matrix, dtype=float) * (float(dt) / parts))
    n = int(dynamic_size)
    # The bounded exponential is evaluated in float64 by LAPACK, then its
    # O(log(parts)) affine compositions use extended precision. This retains
    # tiny source/transport columns beside O(1e11) exchange eigenvalues without
    # modifying the generator or normalizing the resulting population.
    work_dtype = np.longdouble
    base_a = np.asarray(propagator[:n, :n], dtype=work_dtype)
    base_b = np.asarray(propagator[:n, n:], dtype=work_dtype)
    result_a = np.eye(n, dtype=work_dtype)
    result_b = np.zeros_like(base_b)
    remaining = parts
    while remaining:
        if remaining & 1:
            result_b = base_a @ result_b + base_b
            result_a = base_a @ result_a
        base_b = base_a @ base_b + base_b
        base_a = base_a @ base_a
        remaining >>= 1
    initial = np.asarray(vectors, dtype=work_dtype)
    output = np.zeros_like(initial)
    output[:n] = result_a @ initial[:n] + result_b @ initial[n:]
    output[n:] = initial[n:]
    return np.asarray(output, dtype=float)


def _independent_stiff_action(matrix, vectors, dt):
    """Independent error-controlled action of the same augmented generator."""
    operator = np.asarray(matrix, dtype=float)
    initial = np.asarray(vectors, dtype=float)
    output = np.empty_like(initial)
    for column in range(initial.shape[1]):
        solution = solve_ivp(
            lambda _time, state: operator @ state,
            (0.0, float(dt)), initial[:, column], method="BDF", jac=operator,
            rtol=2.0e-11, atol=1.0e-13,
        )
        if not solution.success:
            raise FloatingPointError(
                f"independent stiff augmented solve failed: {solution.message}"
            )
        output[:, column] = solution.y[:, -1]
    return output


def joint_transport_exchange(
    mobile,
    retained,
    encounter_rate_s,
    release_rate_s,
    *,
    velocity_m_s,
    dt_s: float,
    dx_m: float,
    mobile_recovery_rate_s: float = 0.0,
    retained_recovery_rate_s: float = 0.0,
    mobile_source_rate=None,
) -> CoupledTransportExchangeResult:
    """Integrate a frozen-coefficient exchange/transport block exactly in time.

    Mobile content follows a conservative first-order finite-volume forward
    generator with local hop rate ``velocity_m_s / dx_m``. Retained content is local.
    Gross exchange fluxes and mobile distance are integrated as augmented states.
    """
    m0 = np.asarray(mobile, dtype=float)
    r0 = np.asarray(retained, dtype=float)
    if m0.shape != r0.shape or m0.ndim != 2:
        raise ValueError("mobile and retained must have the same 2-D shape")
    if np.any(~np.isfinite(m0)) or np.any(~np.isfinite(r0)):
        raise ValueError("populations must be finite")
    if np.any(m0 < 0.0) or np.any(r0 < 0.0):
        raise ValueError("populations must be nonnegative")
    dt = float(dt_s)
    dx = float(dx_m)
    km = float(mobile_recovery_rate_s)
    kr_rec = float(retained_recovery_rate_s)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt_s must be finite and nonnegative")
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx_m must be positive and finite")
    if not np.isfinite(km) or km < 0.0 or not np.isfinite(kr_rec) or kr_rec < 0.0:
        raise ValueError("recovery rates must be finite and nonnegative")
    species, n = m0.shape
    velocity = _rates(velocity_m_s, n, "velocity_m_s")
    if mobile_source_rate is None:
        source = np.zeros_like(m0)
    else:
        source = np.asarray(mobile_source_rate, dtype=float)
        if source.shape != m0.shape:
            raise ValueError("mobile_source_rate must match the population shape")
        if np.any(~np.isfinite(source)) or np.any(source < 0.0):
            raise ValueError("mobile_source_rate must be finite and nonnegative")
    ke = _rates(encounter_rate_s, n, "encounter_rate_s")
    kr = _rates(release_rate_s, n, "release_rate_s")
    initial = float(np.sum(m0) + np.sum(r0))
    emitted = float(np.sum(source)) * dt
    if dt == 0.0 or (initial == 0.0 and emitted == 0.0):
        return CoupledTransportExchangeResult(
            m0.copy(), r0.copy(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )

    escaped_i = 2 * n
    recovered_i = escaped_i + 1
    gross_trap_i = escaped_i + 2
    gross_release_i = escaped_i + 3
    distance_i = escaped_i + 4
    source_i0 = distance_i + 1
    size = source_i0 + species
    generator = np.zeros((size, size), dtype=float)
    hop = velocity / dx
    flux_scale = max(float(np.max(ke)), float(np.max(kr)), 1.0)
    distance_scale = max(float(np.max(velocity)), 1.0)
    for i in range(n):
        mi = i
        ri = n + i
        generator[mi, mi] -= ke[i] + hop[i] + km
        generator[ri, mi] += ke[i]
        generator[mi, ri] += kr[i]
        generator[ri, ri] -= kr[i] + kr_rec
        if i + 1 < n:
            generator[mi + 1, mi] += hop[i]
        else:
            generator[escaped_i, mi] += hop[i]
        generator[recovered_i, mi] += km
        generator[recovered_i, ri] += kr_rec
        generator[gross_trap_i, mi] += ke[i] / flux_scale
        generator[gross_release_i, ri] += kr[i] / flux_scale
        generator[distance_i, mi] += velocity[i] / distance_scale
    source_coordinate_scale = np.ones(species, dtype=float)
    kinetic_scale = max(float(np.max(ke + kr + hop + km + kr_rec)), 1.0)
    for system in range(species):
        source_max = float(np.max(source[system]))
        if source_max > 0.0:
            source_coordinate_scale[system] = kinetic_scale / source_max
        generator[:n, source_i0 + system] = (
            source[system] * source_coordinate_scale[system]
        )

    initial_vectors = np.zeros((size, species), dtype=float)
    initial_vectors[:n, :] = m0.T
    initial_vectors[n : 2 * n, :] = r0.T
    for system in range(species):
        initial_vectors[source_i0 + system, system] = (
            1.0 / source_coordinate_scale[system]
        )
    # Exact similarity transform from [mobile, retained] to
    # [mobile+retained, retained]. The conserved local-total coordinate avoids
    # subtracting O(kr) release fluxes in the stiff population balance while
    # retaining the complete two-state dynamics and every augmented ledger.
    change = np.eye(size)
    inverse_change = np.eye(size)
    for i in range(n):
        change[i, n + i] = 1.0
        inverse_change[i, n + i] = -1.0
    conditioned_generator = change @ generator @ inverse_change
    conditioned_initial = change @ initial_vectors
    activity = float(np.max(ke + kr + hop + km + kr_rec) * dt)
    final = _partitioned_affine_action(
        conditioned_generator, conditioned_initial, dt, activity,
        dynamic_size=source_i0,
    )
    final = inverse_change @ final
    # Scaling-and-squaring a generator whose rates span ~1e12 s^-1 and whose
    # augmented flux rows grow to ~1e12 loses a few ulps in the conserved
    # population columns.  This tolerance is diagnostic only; it is many
    # orders tighter than the spatial upwind error being measured here.
    tolerance = 2.0e-8 * max(initial + emitted, 1.0)
    physical = final[: 2 * n + 2]
    if float(np.min(physical)) < -tolerance:
        raise FloatingPointError("joint transport/exchange produced a negative population")
    physical = np.maximum(physical, 0.0)
    m1 = physical[:n].T
    r1 = physical[n : 2 * n].T
    escaped_by_species = np.asarray(physical[escaped_i], dtype=float).copy()
    recovered_by_species = np.asarray(physical[recovered_i], dtype=float).copy()
    if km == 0.0 and kr_rec == 0.0:
        recovered_by_species.fill(0.0)
    # Enforce the exact per-species affine CTMC invariant after the exponential.
    # This is a small roundoff correction along the conservation direction, not
    # a local-equilibrium projection: mobile/retained ratios and rates are never
    # replaced by their stiff limit. Overshoots are removed from the computed
    # mobile field, which owns the continuous source; undershoots accrue to the
    # absorbing escape ledger.
    budgets = np.sum(m0 + r0, axis=1) + np.sum(source, axis=1) * dt
    accounted_by_species = (
        np.sum(m1 + r1, axis=1) + escaped_by_species + recovered_by_species
    )
    invariant_defect = budgets - accounted_by_species
    invariant_limit = 1.0e-4 * np.maximum(budgets, 1.0)
    if np.any(np.abs(invariant_defect) > invariant_limit):
        raise FloatingPointError(
            "stiff affine invariant correction exceeds conditioning limit: "
            f"defect={float(np.max(np.abs(invariant_defect))):.6e}, "
            f"limit={float(np.max(invariant_limit)):.6e}"
        )
    for system, defect in enumerate(invariant_defect):
        if defect >= 0.0:
            escaped_by_species[system] += defect
        else:
            excess = -defect
            from_escape = min(float(escaped_by_species[system]), excess)
            escaped_by_species[system] -= from_escape
            excess -= from_escape
            if excess > 0.0:
                removable = float(np.sum(m1[system] + r1[system]))
                if removable < excess:
                    raise FloatingPointError("invariant overshoot exceeds physical content")
                scale = (removable - excess) / removable
                m1[system] *= scale
                r1[system] *= scale
    invariant_correction = float(np.sum(np.abs(invariant_defect)))
    escaped = float(np.sum(escaped_by_species))
    recovered = float(np.sum(recovered_by_species))
    gross_trap = max(float(np.sum(final[gross_trap_i])) * flux_scale, 0.0)
    gross_release = max(float(np.sum(final[gross_release_i])) * flux_scale, 0.0)
    distance_population = max(float(np.sum(final[distance_i])) * distance_scale, 0.0)
    accounted = float(np.sum(m1) + np.sum(r1) + escaped + recovered)
    error = accounted - initial - emitted
    if abs(error) > tolerance:
        raise FloatingPointError(
            f"joint transport/exchange conservation error {error:.6e} exceeds {tolerance:.6e}; "
            f"initial={initial:.9e}, emitted={emitted:.9e}, dt={dt:.9e}, "
            f"ke_max={float(np.max(ke)):.9e}, kr_max={float(np.max(kr)):.9e}, "
            f"hop_max={float(np.max(hop)):.9e}, activity={activity:.9e}, "
            f"population={float(np.sum(m1)+np.sum(r1)):.9e}, "
            f"escaped={escaped:.9e}, recovered={recovered:.9e}"
        )
    return CoupledTransportExchangeResult(
        mobile=m1,
        retained=r1,
        escaped=escaped,
        recovered=recovered,
        net_retained_gain=float(np.sum(np.maximum(r1 - r0, 0.0))),
        net_retained_loss=float(np.sum(np.maximum(r0 - r1, 0.0))),
        gross_trapping=gross_trap,
        gross_release=gross_release,
        mobile_distance_population_m=distance_population,
        conservation_error=error,
        invariant_projection_correction=invariant_correction,
    )


__all__ = [
    "CoupledTransportExchangeResult",
    "exchange_gross_flux",
    "joint_transport_exchange",
]
