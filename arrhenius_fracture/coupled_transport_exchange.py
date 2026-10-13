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


def _partitioned_exponential_action(matrix, vectors, dt, activity, limit=1.0e4):
    """Apply one autonomous semigroup with a numerically bounded exponent."""
    parts = max(1, int(np.ceil(max(float(activity), 0.0) / float(limit))))
    propagator = expm(np.asarray(matrix, dtype=float) * (float(dt) / parts))
    # Binary exponentiation preserves the same partitioned autonomous
    # semigroup while avoiding O(parts) dense products in the difficult
    # high-transport regime.  Direct expm(matrix*dt) is deliberately avoided:
    # the bounded exponent is what protects the stiff population invariants.
    return np.linalg.matrix_power(propagator, parts) @ np.asarray(vectors, dtype=float)


def _partitioned_population_action(
    matrix, vectors, dt, activity, population_size: int, limit=1.0e4
):
    """Stable binary composition of a substochastic population affine map."""
    parts = max(1, int(np.ceil(max(float(activity), 0.0) / float(limit))))
    propagator = expm(np.asarray(matrix, dtype=float) * (float(dt) / parts))
    n = int(population_size)
    source_size = propagator.shape[0] - n
    base_a = np.maximum(propagator[:n, :n], 0.0)
    base_b = np.maximum(propagator[:n, n:], 0.0)

    def project(a):
        result = np.maximum(a, 0.0)
        sums = np.sum(result, axis=0)
        high = sums > 1.0
        if np.any(high):
            result[:, high] /= sums[high]
        return result

    base_a = project(base_a)
    result_a = np.eye(n)
    result_b = np.zeros((n, source_size))
    remaining = parts
    while remaining:
        if remaining & 1:
            result_b = base_a @ result_b + base_b
            result_a = project(base_a @ result_a)
        base_b = base_a @ base_b + base_b
        base_a = project(base_a @ base_a)
        remaining >>= 1
    initial = np.asarray(vectors, dtype=float)
    output = np.zeros_like(initial)
    output[:n] = result_a @ initial[:n]
    if source_size:
        output[:n] += result_b @ initial[n:]
        output[n:] = initial[n:]
    return output


def joint_transport_exchange(
    mobile,
    retained,
    encounter_rate_s,
    release_rate_s,
    *,
    velocity_m_s: float,
    dt_s: float,
    dx_m: float,
    mobile_recovery_rate_s: float = 0.0,
    retained_recovery_rate_s: float = 0.0,
    mobile_source_rate=None,
) -> CoupledTransportExchangeResult:
    """Integrate a frozen-coefficient exchange/transport block exactly in time.

    Mobile content follows a conservative first-order finite-volume forward
    generator with hop rate ``velocity_m_s / dx_m``. Retained content is local.
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
    velocity = float(velocity_m_s)
    km = float(mobile_recovery_rate_s)
    kr_rec = float(retained_recovery_rate_s)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt_s must be finite and nonnegative")
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx_m must be positive and finite")
    if not np.isfinite(velocity) or velocity < 0.0:
        raise ValueError("velocity_m_s must be finite and nonnegative")
    if not np.isfinite(km) or km < 0.0 or not np.isfinite(kr_rec) or kr_rec < 0.0:
        raise ValueError("recovery rates must be finite and nonnegative")
    species, n = m0.shape
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

    # In the singularly stiff regime, eliminate the O((ke+kr)^-1) exchange
    # layer and evolve the local-equilibrium total with its mobile-weighted
    # transport rate.  This is the conservative asymptotic limit of the same
    # CTMC and avoids forming an ill-conditioned 1e12-by-second exponential.
    stiffness = float(np.max((ke + kr) * dt))
    if stiffness > 1.0e6 and km == 0.0 and kr_rec == 0.0:
        rate = ke + kr
        mobile_fraction = np.divide(kr, rate, out=np.ones_like(rate), where=rate > 0.0)
        retained_fraction = 1.0 - mobile_fraction
        hop_eff = mobile_fraction * velocity / dx
        turnover_rate = ke * mobile_fraction
        size_reduced = n + 3 + species
        escape_j, turnover_j, distance_j = n, n + 1, n + 2
        source_j0 = n + 3
        reduced = np.zeros((size_reduced, size_reduced), dtype=float)
        turnover_scale = max(float(np.max(turnover_rate)), 1.0)
        distance_scale = max(velocity, 1.0)
        for i in range(n):
            reduced[i, i] -= hop_eff[i]
            if i + 1 < n:
                reduced[i + 1, i] += hop_eff[i]
            else:
                reduced[escape_j, i] += hop_eff[i]
            reduced[turnover_j, i] += turnover_rate[i] / turnover_scale
            reduced[distance_j, i] += velocity * mobile_fraction[i] / distance_scale
        for system in range(species):
            reduced[:n, source_j0 + system] = source[system]
        vectors = np.zeros((size_reduced, species), dtype=float)
        vectors[:n] = (m0 + r0).T
        for system in range(species):
            vectors[source_j0 + system, system] = 1.0
        max_hop_dt = float(np.max(hop_eff) * dt)
        evolved = _partitioned_exponential_action(
            reduced, vectors, dt, max_hop_dt
        )
        # Re-evaluate the physical state without diagnostic accumulator rows.
        # In very stiff cases those lower-triangular rows can contaminate small
        # population columns during dense scaling and squaring even though
        # they cannot affect the state analytically.
        if not np.any(source):
            # Do not append inactive unit-source states.  Their zero
            # eigenvalues unnecessarily enlarge the spectral span and were
            # the dominant loss of accuracy for small DBTT populations.
            population = reduced[:n, :n]
            population_vectors = vectors[:n]
        else:
            indices = np.r_[:n, source_j0:size_reduced]
            population = reduced[indices][:, indices]
            population_vectors = vectors[indices]
        physical_evolved = _partitioned_population_action(
            population, population_vectors, dt, max_hop_dt,
            population_size=n,
        )
        total1 = np.maximum(physical_evolved[:n].T, 0.0)
        # The augmented source columns can lose several digits when expm
        # scales a generator spanning O(1e12) exchange rates and O(1) source
        # rates.  The reduced CTMC has an exact per-species population budget.
        # Project only small positive roundoff overshoots back onto that
        # invariant; a larger defect remains a hard numerical failure.
        species_budget = (
            np.sum(m0 + r0, axis=1) + np.sum(source, axis=1) * dt
        )
        species_total = np.sum(total1, axis=1)
        overshoot = np.maximum(species_total - species_budget, 0.0)
        projection_tolerance = np.maximum(2.0e-5 * species_budget, 1.0e-4)
        if np.any(overshoot > projection_tolerance):
            relative_overshoot = overshoot / np.maximum(species_budget, 1.0)
            raise FloatingPointError(
                "asymptotic transport/exchange invariant projection exceeds "
                f"tolerance: absolute={float(np.max(overshoot)):.6e}, "
                f"relative={float(np.max(relative_overshoot)):.6e}, "
                f"budget={float(np.max(species_budget)):.6e}, "
                f"max_hop_dt={max_hop_dt:.6e}"
            )
        for system in np.flatnonzero(overshoot > 0.0):
            total1[system] *= species_budget[system] / species_total[system]
        m1 = total1 * mobile_fraction[None, :]
        r1 = total1 * retained_fraction[None, :]
        # With no recovery, escape is the exact conservative complement.  This
        # avoids cancellation in the absorbing accumulator after a very long,
        # stiff interval while leaving the evolved spatial profile unchanged.
        escaped = max(initial + emitted - float(np.sum(total1)), 0.0)
        turnover = max(float(np.sum(evolved[turnover_j])) * turnover_scale, 0.0)
        distance_population = max(float(np.sum(evolved[distance_j])) * distance_scale, 0.0)
        delta_r = float(np.sum(r1) - np.sum(r0))
        gross_trap = turnover + max(delta_r, 0.0)
        gross_release = turnover + max(-delta_r, 0.0)
        error = float(np.sum(m1) + np.sum(r1) + escaped - initial - emitted)
        tolerance = 2.0e-8 * max(initial + emitted, 1.0)
        if abs(error) > tolerance:
            raise FloatingPointError(
                f"asymptotic transport/exchange conservation error {error:.6e} exceeds {tolerance:.6e}"
            )
        return CoupledTransportExchangeResult(
            mobile=m1,
            retained=r1,
            escaped=escaped,
            recovered=0.0,
            net_retained_gain=float(np.sum(np.maximum(r1 - r0, 0.0))),
            net_retained_loss=float(np.sum(np.maximum(r0 - r1, 0.0))),
            gross_trapping=gross_trap,
            gross_release=gross_release,
            mobile_distance_population_m=distance_population,
            conservation_error=error,
            invariant_projection_correction=float(np.sum(overshoot)),
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
    distance_scale = max(velocity, 1.0)
    for i in range(n):
        mi = i
        ri = n + i
        generator[mi, mi] -= ke[i] + hop + km
        generator[ri, mi] += ke[i]
        generator[mi, ri] += kr[i]
        generator[ri, ri] -= kr[i] + kr_rec
        if i + 1 < n:
            generator[mi + 1, mi] += hop
        else:
            generator[escaped_i, mi] += hop
        generator[recovered_i, mi] += km
        generator[recovered_i, ri] += kr_rec
        generator[gross_trap_i, mi] += ke[i] / flux_scale
        generator[gross_release_i, ri] += kr[i] / flux_scale
        generator[distance_i, mi] += velocity / distance_scale
    for system in range(species):
        generator[:n, source_i0 + system] = source[system]

    initial_vectors = np.zeros((size, species), dtype=float)
    initial_vectors[:n, :] = m0.T
    initial_vectors[n : 2 * n, :] = r0.T
    for system in range(species):
        initial_vectors[source_i0 + system, system] = 1.0
    final = expm(generator * dt) @ initial_vectors
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
    escaped = float(np.sum(physical[escaped_i]))
    recovered = float(np.sum(physical[recovered_i]))
    gross_trap = max(float(np.sum(final[gross_trap_i])) * flux_scale, 0.0)
    gross_release = max(float(np.sum(final[gross_release_i])) * flux_scale, 0.0)
    distance_population = max(float(np.sum(final[distance_i])) * distance_scale, 0.0)
    accounted = float(np.sum(m1) + np.sum(r1) + escaped + recovered)
    error = accounted - initial - emitted
    if abs(error) > tolerance:
        raise FloatingPointError(
            f"joint transport/exchange conservation error {error:.6e} exceeds {tolerance:.6e}"
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
    )


__all__ = [
    "CoupledTransportExchangeResult",
    "exchange_gross_flux",
    "joint_transport_exchange",
]
