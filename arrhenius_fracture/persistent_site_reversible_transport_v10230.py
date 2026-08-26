"""Signed mobile transport primitives for the v10.2.30 persistent-site state.

Cleavage and emission remain opening-only. These functions act only on existing
positive/negative Burgers mobile populations and deliberately do not select an
emission/source-drive projection.
"""
from __future__ import annotations

import math
from types import MethodType
from typing import Any

import numpy as np

MODEL_ID = "v10.2.30_intrinsic_signed_mobile_transport_v1"


def _sync_reversible_totals(mpz: Any) -> None:
    mpz.mobile = mpz.mobile_positive + mpz.mobile_negative
    mpz.retained = mpz.retained_positive + mpz.retained_negative
    mpz.accumulated_slip = (
        mpz.accumulated_slip_positive + mpz.accumulated_slip_negative
    )


def _ensure_reversible_state(mpz: Any) -> None:
    shape = (mpz.n_systems, mpz.n_bins)
    for name in ("returned_slip_positive", "returned_slip_negative"):
        if not hasattr(mpz, name):
            setattr(mpz, name, np.zeros(shape, dtype=float))
    ledger_shape = (mpz.n_systems, 2)
    for name in (
        "cumulative_returned_mobile",
        "cumulative_physical_returned_mobile",
        "cumulative_escaped_mobile",
        "cumulative_cancelled_source_slip",
    ):
        if not hasattr(mpz, name):
            setattr(mpz, name, np.zeros(ledger_shape, dtype=float))
    for name in (
        "cumulative_gross_source_activity",
        "cumulative_gross_return_activity",
    ):
        if not hasattr(mpz, name):
            setattr(mpz, name, 0.0)


def net_source_slip(mpz: Any) -> np.ndarray:
    _ensure_reversible_state(mpz)
    positive = np.maximum(
        np.asarray(mpz.accumulated_slip_positive)
        - np.asarray(mpz.returned_slip_positive),
        0.0,
    )
    negative = np.maximum(
        np.asarray(mpz.accumulated_slip_negative)
        - np.asarray(mpz.returned_slip_negative),
        0.0,
    )
    return positive + negative


def _reversible_local_slip_count(mpz: Any) -> float:
    length = max(float(mpz.cfg.blunting_length_m), float(mpz.dx))
    weights = np.exp(-np.asarray(mpz.x, dtype=float) / length)
    return float(np.sum(net_source_slip(mpz) * weights[None, :]))


def _reversible_blunted_radius(mpz: Any, r0: float, b: float) -> float:
    return max(
        float(r0)
        + max(float(mpz.manifest.c_blunt), 0.0)
        * abs(float(b))
        * _reversible_local_slip_count(mpz),
        float(r0),
    )


def _cancel_returned_source_slip(
    mpz: Any, physical_returned: np.ndarray
) -> np.ndarray:
    _ensure_reversible_state(mpz)
    cancelled = np.zeros_like(physical_returned)
    nsrc = max(min(int(mpz.cfg.source_bin_count), mpz.n_bins), 1)
    for system in range(mpz.n_systems):
        for q, suffix in enumerate(("negative", "positive")):
            requested = max(float(physical_returned[system, q]), 0.0)
            if requested <= 0.0:
                continue
            emitted = np.asarray(
                getattr(mpz, f"accumulated_slip_{suffix}"), dtype=float
            )[system, :nsrc]
            returned_field = getattr(mpz, f"returned_slip_{suffix}")
            available = np.maximum(emitted - returned_field[system, :nsrc], 0.0)
            total = float(np.sum(available))
            amount = min(requested, total)
            if amount > 0.0 and total > 0.0:
                returned_field[system, :nsrc] += available * (amount / total)
                cancelled[system, q] = amount
    mpz.cumulative_cancelled_source_slip += cancelled
    return cancelled


def signed_finite_tip_stress_Pa(K_signed_Pa_sqrt_m: float, radius_m: float) -> float:
    """Map signed applied K to finite-tip stress without an opening-only clip."""
    radius = max(float(radius_m), 1.0e-30)
    return float(K_signed_Pa_sqrt_m) / math.sqrt(2.0 * math.pi * radius)


def signed_arrhenius_rate(surface, stress_Pa, temperature_K: float) -> np.ndarray:
    """Forward-minus-reverse rate using one unchanged barrier surface."""
    stress = np.asarray(stress_Pa, dtype=float)
    magnitude = np.abs(stress)
    driven = np.asarray(surface.rate(magnitude, float(temperature_K)), dtype=float)
    zero = float(np.asarray(surface.rate(0.0, float(temperature_K))))
    return np.sign(stress) * np.maximum(driven - zero, 0.0)


def advect_signed_mobile_populations(
    mpz: Any,
    velocity_by_system_m_s,
    dt_s: float,
    source_linked_eligible_by_system_sign,
) -> dict[str, np.ndarray]:
    """Advect actual signed-Burgers fields and classify both boundary fates.

    Positive and negative Burgers populations move in opposite spatial
    directions under the same signed glide rate. Left-boundary outflow is a
    physical surface return only for a sign channel with uncancelled
    source-linked emitted slip.
    """
    velocity = np.asarray(velocity_by_system_m_s, dtype=float).reshape(-1)
    eligible = np.asarray(source_linked_eligible_by_system_sign, dtype=bool)
    if velocity.shape != (mpz.n_systems,):
        raise ValueError("one signed velocity is required per slip system")
    if eligible.shape != (mpz.n_systems, 2):
        raise ValueError("source-linked eligibility must have shape (systems, 2)")
    duration = max(float(dt_s), 0.0)
    returned = np.zeros((mpz.n_systems, 2), dtype=float)
    escaped = np.zeros((mpz.n_systems, 2), dtype=float)
    physical_return = np.zeros((mpz.n_systems, 2), dtype=float)
    for system in range(mpz.n_systems):
        for q, suffix in enumerate(("negative", "positive")):
            field = np.asarray(getattr(mpz, f"mobile_{suffix}"), dtype=float)
            spatial_velocity = (-1.0 if q == 0 else 1.0) * velocity[system]
            distance = abs(spatial_velocity) * duration
            row = field[system : system + 1]
            if spatial_velocity >= 0.0:
                moved, loss = mpz._advect_forward(row, distance, mpz.dx)
                escaped[system, q] = loss
            else:
                flipped, loss = mpz._advect_forward(row[:, ::-1], distance, mpz.dx)
                moved = flipped[:, ::-1]
                returned[system, q] = loss
                if eligible[system, q]:
                    physical_return[system, q] = loss
            field[system : system + 1] = moved
            setattr(mpz, f"mobile_{suffix}", field)
    mpz.mobile = mpz.mobile_positive + mpz.mobile_negative
    return {
        "returned_mobile": returned,
        "escaped_mobile": escaped,
        "physical_returned_mobile": physical_return,
    }


def _reversible_evolve_scalar(
    mpz: Any,
    dt_s: float,
    temperature_K: float,
    opening_stress_Pa: float,
    burgers_vector_m: float,
    system_weights=None,
) -> dict[str, float]:
    """Evolve emission with opening stress and existing mobile state with signed stress."""
    from .signed_burgers_shared_v1025 import (
        _exchange_species,
        _recover_active,
        _signed_evolve_wake,
    )

    _ensure_reversible_state(mpz)
    dt = max(float(dt_s), 0.0)
    before_source = float(np.sum(mpz.accumulated_slip))
    emitted = mpz._emit(dt, max(float(opening_stress_Pa), 0.0), temperature_K, system_weights)
    _sync_reversible_totals(mpz)
    mpz.cumulative_gross_source_activity += max(
        float(np.sum(mpz.accumulated_slip)) - before_source, 0.0
    )

    rho = mpz.local_forest_density_m2(False)
    opening_profile = mpz.local_stress_profile_Pa(max(float(opening_stress_Pa), 0.0))
    opening_rates = mpz._transport_rates(
        opening_profile, rho, temperature_K, burgers_vector_m
    )
    K_signed = float(
        getattr(mpz, "_reversible_transport_K_signed_Pa_sqrt_m", math.nan)
    )
    radius = max(float(getattr(mpz, "_reversible_tip_radius_m", math.nan)), 0.0)
    if not math.isfinite(K_signed) or not math.isfinite(radius) or radius <= 0.0:
        signed_profile = opening_profile
    else:
        signed_tip = signed_finite_tip_stress_Pa(K_signed, radius)
        ref = max(float(mpz.cfg.blunting_length_m), float(mpz.dx), 1.0e-12)
        attenuation = np.sqrt(ref / np.maximum(ref + np.asarray(mpz.x), ref))
        internal = np.asarray(
            getattr(mpz, "_reversible_mobile_internal_stress_Pa", np.zeros(mpz.n_bins)),
            dtype=float,
        )
        if internal.shape not in {(mpz.n_bins,), (mpz.n_systems, mpz.n_bins)}:
            raise ValueError("mobile internal stress must be bin- or system-resolved")
        signed_profile = signed_tip * attenuation + internal

    p_surface = mpz.manifest.peierls.as_surface(mpz.manifest.emission)
    tau = max(float(mpz.cfg.peierls_stress_fraction), 0.0) * signed_profile
    peierls = signed_arrhenius_rate(p_surface, tau, temperature_K)
    spacing = 1.0 / (2.0 * np.sqrt(np.maximum(rho, 1.0)))
    jump = max(float(mpz.cfg.jump_fraction), 0.0) * spacing
    signed_velocity_profile = jump * peierls
    encounter = (
        max(float(mpz.manifest.encounter_efficiency), 0.0)
        * np.abs(signed_velocity_profile)
        * np.sqrt(np.maximum(rho, 0.0))
    )
    trapped, released = _exchange_species(
        mpz, encounter, opening_rates["taylor"], dt
    )
    recovered = _recover_active(mpz, dt)
    mobile_by_bin = np.sum(np.maximum(mpz.mobile, 0.0), axis=0)
    if np.sum(mobile_by_bin) > 0.0:
        mean_velocity = float(
            np.sum(signed_velocity_profile * mobile_by_bin)
            / np.sum(mobile_by_bin)
        )
    else:
        nsrc = max(min(int(mpz.cfg.source_bin_count), mpz.n_bins), 1)
        mean_velocity = float(np.mean(signed_velocity_profile[:nsrc]))
    velocities = np.full(mpz.n_systems, mean_velocity)
    nsrc = max(min(int(mpz.cfg.source_bin_count), mpz.n_bins), 1)
    eligible = np.zeros((mpz.n_systems, 2), dtype=bool)
    for q, suffix in enumerate(("negative", "positive")):
        available = np.maximum(
            np.asarray(getattr(mpz, f"accumulated_slip_{suffix}"))[:, :nsrc]
            - np.asarray(getattr(mpz, f"returned_slip_{suffix}"))[:, :nsrc],
            0.0,
        )
        eligible[:, q] = np.sum(available, axis=1) > 0.0
    fate = advect_signed_mobile_populations(
        mpz, velocities, dt, eligible
    )
    cancelled = _cancel_returned_source_slip(
        mpz, fate["physical_returned_mobile"]
    )
    mpz.cumulative_returned_mobile += fate["returned_mobile"]
    mpz.cumulative_physical_returned_mobile += fate["physical_returned_mobile"]
    mpz.cumulative_escaped_mobile += fate["escaped_mobile"]
    mpz.cumulative_gross_return_activity += float(
        np.sum(fate["physical_returned_mobile"])
    )
    escaped = float(np.sum(fate["escaped_mobile"]))
    mpz.escaped_total += escaped
    mpz.recovered_total += recovered
    mpz.time_s += dt
    wake = _signed_evolve_wake(mpz, dt, temperature_K, burgers_vector_m)
    _sync_reversible_totals(mpz)
    return {
        "dN_emit": emitted,
        "dN_source_activations": float(mpz.signed_last_source_activations),
        "dN_trapped": trapped,
        "dN_released": released,
        "dN_recovered": recovered,
        "dN_escaped": escaped,
        "dN_returned": float(np.sum(fate["returned_mobile"])),
        "dN_physical_returned": float(np.sum(fate["physical_returned_mobile"])),
        "dN_source_slip_cancelled": float(np.sum(cancelled)),
        "peierls_rate_s": float(np.max(np.abs(peierls))),
        "peierls_signed_rate_min_s": float(np.min(peierls)),
        "peierls_signed_rate_max_s": float(np.max(peierls)),
        "taylor_completion_rate_s": float(np.max(opening_rates["taylor"])),
        "encounter_rate_s": float(np.max(encounter)),
        "taylor_m_eff": float(np.max(opening_rates["m"])),
        "available_site_fraction": mpz.available_site_fraction,
        "reversible_transport_active": 1.0,
        "reversible_transport_K_signed_Pa_sqrt_m": K_signed,
        "reversible_mobile_velocity_m_s": mean_velocity,
        "reversible_true_reverse_system_count": float(
            np.count_nonzero(
                (velocities[:, None] * np.array([-1.0, 1.0])[None, :] < 0.0)
                & eligible
            )
        ),
        "net_source_slip_count": _reversible_local_slip_count(mpz),
        **wake,
    }


def install_reversible_transport(mpz: Any) -> None:
    """Install reversible mobile evolution on an initialized v10 signed state."""
    if not all(
        hasattr(mpz, name)
        for name in ("mobile_positive", "mobile_negative", "accumulated_slip_positive")
    ):
        raise RuntimeError("v10 signed-Burgers state must be installed first")
    if str(getattr(mpz, "_signed_transport_mode", "")) != "validated_scalar":
        raise RuntimeError("initial reversible integration requires validated scalar transport")
    _ensure_reversible_state(mpz)
    mpz.evolve = MethodType(_reversible_evolve_scalar, mpz)
    mpz.local_slip_count = MethodType(_reversible_local_slip_count, mpz)
    mpz.blunted_radius = MethodType(_reversible_blunted_radius, mpz)
    mpz._reversible_transport_installed = True
    mpz._reversible_mobile_internal_stress_model = "explicit_provider_or_zero"


__all__ = [
    "MODEL_ID",
    "advect_signed_mobile_populations",
    "install_reversible_transport",
    "net_source_slip",
    "signed_arrhenius_rate",
    "signed_finite_tip_stress_Pa",
]
