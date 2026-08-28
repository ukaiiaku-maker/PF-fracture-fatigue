"""Analysis-only monotonic first-passage reductions for v10.2.30.

The production solver remains authoritative.  This module preserves its
bounded EXP-floor barriers, cooperative gamma renewal, distinct opening,
cleavage and emission stress channels, persistent source semantics, and
pre-first-event ordering.  No crack-advance translation or empirical
toughness law appears here.

F0 integrates opening action only.  F1 adds the transient source-linked
blunting moment.  F2 adds signed mobile/retained population moments and the
source shielding kernel evaluated at the blunting length.  F2B is the same
moment balance split into near-tip and outer-MPZ compartments; it is provided
for a predeclared closure check and is not activated automatically.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.special import gammainc, gammaln
from scipy.linalg import expm

from .material_manifest import ExpFloorBarrier, KB_EV_PER_K, MaterialManifest


MODEL_ID = "v10.2.30_transient_monotonic_fracture_hierarchy_v1"


@dataclass(frozen=True)
class MonotonicControls:
    K0_MPa_sqrt_m: float = 0.0
    Kmax_MPa_sqrt_m: float = 100.0
    dK_MPa_sqrt_m: float = 0.02
    loading_rate_MPa_sqrt_m_s: float = 0.02
    r0_m: float = 1.0e-6
    burgers_m: float = 2.74e-10
    shear_modulus_Pa: float = 160.0e9
    poisson_ratio: float = 0.28
    taylor_stress_fraction: float = 1.0 / math.sqrt(3.0)
    emission_drive_factor: float = 1.0
    forest_floor_m2: float = 5.0e12
    blunting_length_m: float = 0.5e-6
    mpz_length_m: float = 50.0e-6
    source_zone_length_m: float = 2.0e-6
    reference_front_width_m: float = 10.0e-6
    reference_source_area_m2: float = 25.0e-12
    n_systems: int = 2
    cleavage_hits: float = 3.0
    cleavage_tau_s: float = 1.0e-6
    shielding_orientation_factor: float = 1.0
    mobile_shield_fraction: float = 0.0
    retained_recovery_rate_s: float | None = None
    f2b_exchange_rate_s: float = 1.0
    endpoint_localization_min: float = 3.0


def cooperative_rate(raw_rate_s: np.ndarray | float, hits: float, tau_s: float) -> np.ndarray:
    """Exact gamma-renewal completion rate used by production."""
    raw = np.maximum(np.asarray(raw_rate_s, dtype=float), 0.0)
    m = max(float(hits), 1.0)
    if m <= 1.0 + 1.0e-12:
        return raw
    x = np.minimum(raw * max(float(tau_s), 1.0e-300), 1.0e12)
    return gammainc(m, x) / max(float(tau_s), 1.0e-300)


def cooperative_log_sensitivity(raw_rate_s: np.ndarray | float, hits: float,
                                tau_s: float) -> np.ndarray:
    """Return d ln(Lambda)/d ln(lambda_raw), including saturation."""
    raw = np.maximum(np.asarray(raw_rate_s, dtype=float), 0.0)
    m = max(float(hits), 1.0)
    if m <= 1.0 + 1.0e-12:
        return np.ones_like(raw)
    x = np.maximum(raw * max(float(tau_s), 1.0e-300), 1.0e-300)
    probability = np.maximum(gammainc(m, x), 1.0e-300)
    log_numerator = m * np.log(x) - x - gammaln(m)
    return np.exp(np.clip(log_numerator - np.log(probability), -700.0, 700.0))


def stress_channels(Kloc_Pa_sqrt_m: float, radius_m: float,
                    Kshield_Pa_sqrt_m: float, backstress_Pa: float,
                    emission_weight: float = 1.0) -> dict[str, float]:
    denom = math.sqrt(2.0 * math.pi * max(float(radius_m), 1.0e-30))
    opening = max(float(Kloc_Pa_sqrt_m), 0.0) / denom
    cleavage = max(float(Kloc_Pa_sqrt_m) - float(Kshield_Pa_sqrt_m), 0.0) / denom
    emission = max(float(emission_weight) * opening - float(backstress_Pa), 0.0)
    return {
        "sigma_open_Pa": opening,
        "sigma_cleavage_Pa": cleavage,
        "sigma_emission_Pa": emission,
    }


def _row_float(row: Mapping[str, Any], name: str, default: float) -> float:
    try:
        value = float(row.get(name, default))
    except (TypeError, ValueError):
        value = float(default)
    return value if math.isfinite(value) else float(default)


def _source_multiplicity(row: Mapping[str, Any], radius_m: float,
                         controls: MonotonicControls) -> float:
    rho = _row_float(row, "rho_source0_m2", 0.0)
    arc = controls.reference_source_area_m2 / (
        controls.r0_m * controls.reference_front_width_m
    )
    return max(rho * arc * radius_m * controls.reference_front_width_m, 0.0)


def _transport_rates(manifest: MaterialManifest, sigma_open_Pa: float,
                     rho_state_m2: float, temperature_K: float,
                     controls: MonotonicControls) -> dict[str, float]:
    tau = controls.taylor_stress_fraction * max(sigma_open_Pa, 0.0)
    peierls = manifest.peierls.as_surface(manifest.emission)
    taylor = manifest.taylor.as_surface(manifest.emission)
    kp = float(peierls.rate(tau, temperature_K))
    spacing = 1.0 / (2.0 * math.sqrt(max(rho_state_m2, 1.0)))
    phi = spacing / max(controls.burgers_m, 1.0e-30)
    kt_single = float(taylor.rate(tau * phi, temperature_K))
    m_eff = 1.0 + max(manifest.taylor_corr_scale, 0.0) * max(
        math.sqrt(max(rho_state_m2, 1.0) / max(manifest.taylor_corr_rho_c_m2, 1.0)) - 1.0,
        0.0,
    )
    kt = float(gammainc(max(m_eff, 1.0), min(max(kt_single, 0.0), 1.0e12)))
    velocity = spacing * max(kp, 0.0)
    return {
        "peierls_rate_s": max(kp, 0.0),
        "taylor_rate_s": max(kt, 0.0),
        "encounter_rate_s": max(manifest.encounter_efficiency, 0.0)
        * velocity * math.sqrt(max(rho_state_m2, 1.0)),
        "escape_rate_s": velocity / max(controls.mpz_length_m, 1.0e-30),
        "blunting_transport_rate_s": velocity / max(controls.blunting_length_m, 1.0e-30),
        "taylor_m_eff": m_eff,
    }


def _shielding_from_retained(retained_signed: float, controls: MonotonicControls,
                             distance_m: float | None = None) -> float:
    x = controls.blunting_length_m if distance_m is None else max(float(distance_m), 0.0)
    core = max(0.25 * abs(controls.burgers_m), 1.0e-12)
    kernel = (
        controls.shear_modulus_Pa * controls.burgers_m
        / max(1.0 - controls.poisson_ratio, 1.0e-6)
        / math.sqrt(2.0 * math.pi * max(x, core))
    )
    return controls.shielding_orientation_factor * float(retained_signed) * kernel


def _rhs(level: str, K_Pa_sqrt_m: float, state: np.ndarray,
         manifest: MaterialManifest, row: Mapping[str, Any], temperature_K: float,
         controls: MonotonicControls) -> tuple[np.ndarray, dict[str, float]]:
    q = max(float(state[0]), 0.0)
    mobile = max(float(state[1]), 0.0) if state.size > 1 else 0.0
    retained = max(float(state[2]), 0.0) if state.size > 2 else 0.0
    outer_mobile = max(float(state[3]), 0.0) if state.size > 3 else 0.0
    outer_retained = max(float(state[4]), 0.0) if state.size > 4 else 0.0
    radius = controls.r0_m + max(manifest.c_blunt, 0.0) * controls.burgers_m * q
    rho_state = controls.forest_floor_m2 + (retained + outer_retained + q) / max(
        controls.blunting_length_m ** 2, 1.0e-30
    )
    backstress = (
        controls.shear_modulus_Pa * controls.burgers_m
        * math.sqrt(max(rho_state - controls.forest_floor_m2, 0.0))
        / max(abs(controls.taylor_stress_fraction), 1.0e-6)
    )
    shield = 0.0
    if level in {"F2", "F2B"}:
        shield = _shielding_from_retained(retained, controls)
        if level == "F2B":
            shield += _shielding_from_retained(outer_retained, controls, controls.mpz_length_m)
    channels = stress_channels(
        K_Pa_sqrt_m, radius, shield, backstress,
        emission_weight=controls.emission_drive_factor,
    )
    raw_c = float(manifest.cleavage.rate(channels["sigma_cleavage_Pa"], temperature_K))
    lambda_c = float(cooperative_rate(raw_c, controls.cleavage_hits, controls.cleavage_tau_s))
    raw_e = float(manifest.emission.rate(channels["sigma_emission_Pa"], temperature_K))
    emit = controls.n_systems * _source_multiplicity(row, radius, controls) * raw_e
    rates = _transport_rates(manifest, channels["sigma_open_Pa"], rho_state,
                             temperature_K, controls)
    if level == "F0":
        dzdt = np.zeros(1, dtype=float)
    elif level == "F1":
        dzdt = np.array([emit - rates["blunting_transport_rate_s"] * q])
    elif level == "F2":
        recovery = (
            manifest.retained_recovery_rate_s
            if controls.retained_recovery_rate_s is None
            else controls.retained_recovery_rate_s
        )
        dzdt = np.array([
            emit - rates["blunting_transport_rate_s"] * q,
            emit - (rates["encounter_rate_s"] + rates["escape_rate_s"]) * mobile
            + rates["taylor_rate_s"] * retained,
            rates["encounter_rate_s"] * mobile
            - (rates["taylor_rate_s"] + max(recovery, 0.0)) * retained,
        ])
    elif level == "F2B":
        recovery = (
            manifest.retained_recovery_rate_s
            if controls.retained_recovery_rate_s is None
            else controls.retained_recovery_rate_s
        )
        exchange = max(controls.f2b_exchange_rate_s, 0.0)
        dzdt = np.array([
            emit - rates["blunting_transport_rate_s"] * q,
            emit - (rates["encounter_rate_s"] + exchange) * mobile
            + rates["taylor_rate_s"] * retained,
            rates["encounter_rate_s"] * mobile
            - (rates["taylor_rate_s"] + max(recovery, 0.0) + exchange) * retained,
            exchange * mobile - rates["escape_rate_s"] * outer_mobile
            + rates["taylor_rate_s"] * outer_retained,
            exchange * retained + rates["encounter_rate_s"] * outer_mobile
            - (rates["taylor_rate_s"] + max(recovery, 0.0)) * outer_retained,
        ])
    else:
        raise ValueError(f"unknown monotonic level {level!r}")
    diagnostics = {
        **channels,
        **rates,
        "raw_cleavage_rate_s": raw_c,
        "cleavage_rate_s": lambda_c,
        "raw_emission_rate_s": raw_e,
        "emission_rate_s": emit,
        "r_eff_m": radius,
        "q": q,
        "mobile": mobile + outer_mobile,
        "retained": retained + outer_retained,
        "backstress_Pa": backstress,
        "K_shield_Pa_sqrt_m": shield,
        "rho_state_m2": rho_state,
    }
    return dzdt, diagnostics


def _linear_positive_advance(level: str, state: np.ndarray, dt: float,
                             diagnostics: Mapping[str, float],
                             manifest: MaterialManifest,
                             controls: MonotonicControls) -> np.ndarray:
    """Frozen-rate exponential update for the positive compartment balance."""
    if level == "F0":
        return state.copy()
    emit = float(diagnostics["emission_rate_s"])
    kq = float(diagnostics["blunting_transport_rate_s"])
    if kq > 1.0e-30:
        q = state[0] * math.exp(-min(kq * dt, 700.0)) + emit * (-math.expm1(-min(kq * dt, 700.0))) / kq
    else:
        q = state[0] + emit * dt
    if level == "F1":
        return np.array([max(q, 0.0)])
    recovery = (
        manifest.retained_recovery_rate_s
        if controls.retained_recovery_rate_s is None
        else controls.retained_recovery_rate_s
    )
    kenc = float(diagnostics["encounter_rate_s"])
    kesc = float(diagnostics["escape_rate_s"])
    kt = float(diagnostics["taylor_rate_s"])
    if level == "F2":
        matrix = np.array([[-kenc-kesc, kt], [kenc, -kt-max(recovery, 0.0)]])
        source = np.array([emit, 0.0])
        augmented = np.zeros((3, 3))
        augmented[:2, :2] = matrix
        augmented[:2, 2] = source
        transition = expm(augmented * dt)
        populations = transition[:2, :2] @ state[1:3] + transition[:2, 2]
        return np.maximum(np.r_[q, populations], 0.0)
    exchange = max(controls.f2b_exchange_rate_s, 0.0)
    matrix = np.array([
        [-kenc-exchange, kt, 0.0, 0.0],
        [kenc, -kt-max(recovery, 0.0)-exchange, 0.0, 0.0],
        [exchange, 0.0, -kesc-kenc, kt],
        [0.0, exchange, kenc, -kt-max(recovery, 0.0)],
    ])
    augmented = np.zeros((5, 5))
    augmented[:4, :4] = matrix
    augmented[0, 4] = emit
    transition = expm(augmented * dt)
    populations = transition[:4, :4] @ state[1:5] + transition[:4, 4]
    return np.maximum(np.r_[q, populations], 0.0)


def _fast_affine_compartment_advance(
    matrix: np.ndarray, source: np.ndarray, initial: np.ndarray, dt: float
) -> np.ndarray:
    """Small dense affine update without norm-dependent Pade scaling.

    Archived PF steps can span more than 10^12 fastest compartment time
    constants.  Diagonalizing the stable two/four-compartment generator avoids
    thousands of matrix squarings while retaining the exact frozen-rate
    exponential.  Ill-conditioned cases fail back to the established augmented
    matrix exponential.  The compartment generator is Metzler, so only
    round-off-scale negative content is clipped.
    """
    if dt <= 0.0:
        return np.maximum(np.asarray(initial, dtype=float), 0.0)
    matrix = np.asarray(matrix, dtype=float)
    source = np.asarray(source, dtype=float)
    initial = np.asarray(initial, dtype=float)
    try:
        augmented = np.zeros((matrix.shape[0] + 1, matrix.shape[1] + 1))
        augmented[:-1, :-1] = matrix
        augmented[:-1, -1] = source
        eigenvalues, eigenvectors = np.linalg.eig(augmented)
        transition = eigenvectors @ np.diag(
            np.exp(np.clip(eigenvalues * dt, -745.0, 0.0))
        ) @ np.linalg.inv(eigenvectors)
        transition = np.real_if_close(transition, tol=1000).real
        lifted = np.r_[initial, 1.0]
        propagated = transition @ lifted
        scale = float(propagated[-1])
        if not math.isfinite(scale) or abs(scale - 1.0) > 1.0e-7:
            raise np.linalg.LinAlgError("affine coordinate did not close")
        result = propagated[:-1] / scale
        if not np.all(np.isfinite(result)):
            raise np.linalg.LinAlgError("non-finite eigen exponential")
    except np.linalg.LinAlgError:
        # A nearly conservative exchange matrix can contain a Jordan block at
        # zero, for which eigenvector propagation is ill-conditioned.  Scale
        # only the affine coordinate before the established augmented expm;
        # this is an exact similarity transformation and prevents a very large
        # emission source from driving unnecessary Pade scaling/squaring.
        source_scale = max(float(np.max(np.abs(source))), 1.0)
        augmented = np.zeros((matrix.shape[0] + 1, matrix.shape[1] + 1))
        augmented[:-1, :-1] = matrix
        augmented[:-1, -1] = source / source_scale
        transition = expm(augmented * dt)
        result = (
            transition[:-1, :-1] @ initial
            + transition[:-1, -1] * source_scale
        )
    return np.maximum(result, 0.0)


def _linear_positive_advance_history(
    level: str, state: np.ndarray, dt: float,
    diagnostics: Mapping[str, float], manifest: MaterialManifest,
    controls: MonotonicControls,
) -> np.ndarray:
    """Fast equivalent of ``_linear_positive_advance`` for archived histories."""
    if level in {"F0", "F1"}:
        return _linear_positive_advance(level, state, dt, diagnostics, manifest, controls)
    emit = float(diagnostics["emission_rate_s"])
    kq = float(diagnostics["blunting_transport_rate_s"])
    if kq > 1.0e-30:
        q = state[0] * math.exp(-min(kq * dt, 700.0)) + emit * (-math.expm1(-min(kq * dt, 700.0))) / kq
    else:
        q = state[0] + emit * dt
    recovery = (
        manifest.retained_recovery_rate_s
        if controls.retained_recovery_rate_s is None
        else controls.retained_recovery_rate_s
    )
    kenc = float(diagnostics["encounter_rate_s"])
    kesc = float(diagnostics["escape_rate_s"])
    kt = float(diagnostics["taylor_rate_s"])
    if level == "F2":
        matrix = np.array([[-kenc-kesc, kt], [kenc, -kt-max(recovery, 0.0)]])
        populations = _fast_affine_compartment_advance(
            matrix, np.array([emit, 0.0]), state[1:3], dt
        )
        return np.maximum(np.r_[q, populations], 0.0)
    exchange = max(controls.f2b_exchange_rate_s, 0.0)
    matrix = np.array([
        [-kenc-exchange, kt, 0.0, 0.0],
        [kenc, -kt-max(recovery, 0.0)-exchange, 0.0, 0.0],
        [exchange, 0.0, -kesc-kenc, kt],
        [0.0, exchange, kenc, -kt-max(recovery, 0.0)],
    ])
    populations = _fast_affine_compartment_advance(
        matrix, np.array([emit, 0.0, 0.0, 0.0]), state[1:5], dt
    )
    return np.maximum(np.r_[q, populations], 0.0)


def solve_first_passage(manifest: MaterialManifest, row: Mapping[str, Any],
                        temperature_K: float, level: str = "F2",
                        controls: MonotonicControls = MonotonicControls(),
                        *, return_history: bool = False) -> dict[str, Any]:
    """Integrate one virgin monotonic ramp until exact action first passage."""
    level = str(level).upper()
    nstate = {"F0": 1, "F1": 1, "F2": 3, "F2B": 5}[level]
    state = np.zeros(nstate, dtype=float)
    action = 0.0
    emission_action = 0.0
    K = controls.K0_MPa_sqrt_m * 1.0e6
    Kmax = controls.Kmax_MPa_sqrt_m * 1.0e6
    dK = controls.dK_MPa_sqrt_m * 1.0e6
    Kdot = controls.loading_rate_MPa_sqrt_m_s * 1.0e6
    history: list[dict[str, float]] = []
    rate_history: list[tuple[float, float]] = []
    previous: tuple[float, float, np.ndarray, dict[str, float]] | None = None
    terminal: dict[str, float] = {}
    while K <= Kmax + 0.5 * dK:
        rhs0, diag0 = _rhs(level, K, state, manifest, row, temperature_K, controls)
        rate_history.append((K / 1.0e6, diag0["cleavage_rate_s"]))
        if return_history:
            history.append({"K_MPa_sqrt_m": K / 1.0e6, "action": action,
                            "emission_action": emission_action, **diag0})
        if action >= 1.0:
            terminal = diag0
            break
        previous = (K, action, state.copy(), diag0)
        h = min(dK, Kmax - K)
        if h <= 0.0:
            terminal = diag0
            break
        dt = h / Kdot
        # A frozen-rate exponential compartment update is positivity preserving
        # even when transport is much faster than the external K ramp.
        midpoint_state = _linear_positive_advance(
            level, state, 0.5 * dt, diag0, manifest, controls
        )
        rhsm, diagm = _rhs(level, K + 0.5 * h, midpoint_state, manifest, row,
                           temperature_K, controls)
        state = _linear_positive_advance(level, state, dt, diagm, manifest, controls)
        action += dt * diagm["cleavage_rate_s"]
        emission_action += dt * diagm["emission_rate_s"]
        K += h
        terminal = diagm
    reached = action >= 1.0 and previous is not None
    if reached:
        Kprev, Aprev, sprev, dprev = previous
        fraction = min(max((1.0 - Aprev) / max(action - Aprev, 1.0e-300), 0.0), 1.0)
        Kinit = (Kprev + fraction * (K - Kprev)) / 1.0e6
        state = sprev + fraction * (state - sprev)
        _, terminal = _rhs(level, Kinit * 1.0e6, np.maximum(state, 0.0), manifest,
                           row, temperature_K, controls)
        action = 1.0
    else:
        Kinit = math.nan
    endpoint_slope = math.nan
    endpoint_K = math.nan
    endpoint_error = math.nan
    localization = math.nan
    if reached and len(rate_history) >= 3:
        last = rate_history[-min(7, len(rate_history)):]
        x = np.array([h[0] for h in last])
        y = np.log(np.maximum([h[1] for h in last], 1.0e-300))
        endpoint_slope = float(np.polyfit(x, y, 1)[0])
        localization = endpoint_slope * max(Kinit - controls.K0_MPa_sqrt_m, 0.0)
        if endpoint_slope > 0.0:
            endpoint_K = Kinit + math.log(
                max(controls.loading_rate_MPa_sqrt_m_s * endpoint_slope, 1.0e-300)
                / max(terminal["cleavage_rate_s"], 1.0e-300)
            ) / endpoint_slope
            endpoint_error = (endpoint_K - Kinit) / max(abs(Kinit), 1.0e-30)
    return {
        "model_id": MODEL_ID,
        "level": level,
        "temperature_K": float(temperature_K),
        "loading_rate_MPa_sqrt_m_s": controls.loading_rate_MPa_sqrt_m_s,
        "first_passage_reached": reached,
        "right_censored": not reached,
        "K_init_MPa_sqrt_m": Kinit,
        "cleavage_action": action,
        "emission_action": emission_action,
        "endpoint_K_init_MPa_sqrt_m": endpoint_K,
        "endpoint_relative_error": endpoint_error,
        "endpoint_log_hazard_slope_per_MPa_sqrt_m": endpoint_slope,
        "hazard_localization": localization,
        "endpoint_asymptotic_applicable": bool(
            math.isfinite(localization) and localization >= controls.endpoint_localization_min
        ),
        "pre_event_state": terminal,
        "history": history if return_history else None,
        "no_crack_advance_translation_before_first_event": True,
    }


def solve_first_passage_history(
    manifest: MaterialManifest,
    row: Mapping[str, Any],
    temperature_K: float,
    history: Iterable[Mapping[str, Any]],
    level: str = "F2",
    controls: MonotonicControls = MonotonicControls(),
    *,
    threshold_action: float = 1.0,
    return_history: bool = False,
) -> dict[str, Any]:
    """Drive the no-fit hierarchy with an archived piecewise-constant local-K history.

    Each history record must provide ``dt_s`` and ``K_local_MPa_sqrt_m``.  The
    update uses the same positivity-preserving frozen-rate midpoint operator as
    :func:`solve_first_passage`; only the external loading protocol changes.
    No applied-to-local transfer coefficient is inferred here.
    """
    level = str(level).upper()
    nstate = {"F0": 1, "F1": 1, "F2": 3, "F2B": 5}[level]
    threshold = float(threshold_action)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold_action must be finite and positive")
    state = np.zeros(nstate, dtype=float)
    action = 0.0
    emission_action = 0.0
    elapsed = 0.0
    used = 0
    passage_index: int | None = None
    passage_fraction = math.nan
    passage_K = math.nan
    terminal: dict[str, float] = {}
    trace: list[dict[str, float]] = []
    for index, item in enumerate(history):
        dt = float(item["dt_s"])
        K_MPa = float(item["K_local_MPa_sqrt_m"])
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError(f"history interval {index} has invalid dt_s")
        if not math.isfinite(K_MPa) or K_MPa < 0.0:
            raise ValueError(f"history interval {index} has invalid local K")
        K = K_MPa * 1.0e6
        _, diag0 = _rhs(level, K, state, manifest, row, temperature_K, controls)
        midpoint = _linear_positive_advance_history(
            level, state, 0.5 * dt, diag0, manifest, controls
        )
        _, diagm = _rhs(level, K, midpoint, manifest, row, temperature_K, controls)
        advanced = _linear_positive_advance_history(
            level, state, dt, diagm, manifest, controls
        )
        increment = dt * diagm["cleavage_rate_s"]
        emission_increment = dt * diagm["emission_rate_s"]
        if return_history:
            trace.append({
                "interval_index": float(index),
                "time_start_s": elapsed,
                "time_end_s": elapsed + dt,
                "K_local_MPa_sqrt_m": K_MPa,
                "action_start": action,
                "action_end_unlocalized": action + increment,
                **diagm,
            })
        used += 1
        terminal = diagm
        if action + increment >= threshold:
            fraction = min(
                max((threshold - action) / max(increment, 1.0e-300), 0.0),
                1.0,
            )
            state = np.maximum(state + fraction * (advanced - state), 0.0)
            elapsed += fraction * dt
            emission_action += fraction * emission_increment
            action = threshold
            passage_index = index
            passage_fraction = fraction
            passage_K = K_MPa
            _, terminal = _rhs(level, K, state, manifest, row, temperature_K, controls)
            break
        state = advanced
        elapsed += dt
        action += increment
        emission_action += emission_increment
    reached = passage_index is not None
    return {
        "model_id": MODEL_ID,
        "loading_protocol": "ARCHIVED_PIECEWISE_CONSTANT_LOCAL_K",
        "level": level,
        "temperature_K": float(temperature_K),
        "threshold_action": threshold,
        "first_passage_reached": reached,
        "right_censored": not reached,
        "first_passage_time_s": elapsed if reached else math.nan,
        "first_passage_interval_index": passage_index,
        "first_passage_interval_fraction": passage_fraction,
        "K_local_at_first_passage_MPa_sqrt_m": passage_K,
        "cleavage_action": action,
        "emission_action": emission_action,
        "intervals_consumed": used,
        "pre_event_state": terminal,
        "history": trace if return_history else None,
        "no_transfer_coefficient_fitted": True,
        "no_crack_advance_translation_before_first_event": True,
    }


def finite_difference_sensitivity(manifest: MaterialManifest, row: Mapping[str, Any],
                                  temperature_K: float, level: str,
                                  controls: MonotonicControls, variable: str,
                                  relative_step: float = 1.0e-3) -> float:
    """Centered derivative used as an independent tangent-validation oracle."""
    if variable == "temperature_K":
        step = max(abs(float(temperature_K)) * relative_step, 1.0e-3)
        lo = solve_first_passage(manifest, row, temperature_K - step, level, controls)
        hi = solve_first_passage(manifest, row, temperature_K + step, level, controls)
    elif variable == "loading_rate_MPa_sqrt_m_s":
        base = controls.loading_rate_MPa_sqrt_m_s
        step = max(abs(base) * relative_step, 1.0e-8)
        lo = solve_first_passage(manifest, row, temperature_K, level,
                                 replace(controls, loading_rate_MPa_sqrt_m_s=base-step))
        hi = solve_first_passage(manifest, row, temperature_K, level,
                                 replace(controls, loading_rate_MPa_sqrt_m_s=base+step))
    else:
        raise ValueError(f"unsupported sensitivity variable {variable!r}")
    return (float(hi["K_init_MPa_sqrt_m"]) - float(lo["K_init_MPa_sqrt_m"])) / (2.0 * step)


def _barrier_value_and_tangents(barrier: ExpFloorBarrier, stress_Pa: np.ndarray,
                                temperature_K: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Exact active-branch EXP-floor value and parameter/temperature tangents."""
    sigma = np.maximum(np.asarray(stress_Pa, dtype=float), 0.0)
    dtemp = float(temperature_K) - barrier.Tref_K
    unclipped_G0 = barrier.G00_eV + barrier.gT_eV_per_K * dtemp
    G0 = max(unclipped_G0, 1e-12)
    unclipped_sigc = barrier.sigc0_Pa + barrier.sT_Pa_per_K * dtemp
    sigc = max(unclipped_sigc, 1.0)
    product = barrier.floor_fraction * G0
    if product <= barrier.floor_min_eV:
        floor = barrier.floor_min_eV
        floor_G0_factor = 0.0
        floor_fraction_derivative = 0.0
    elif product >= barrier.floor_max_fraction * G0:
        floor = barrier.floor_max_fraction * G0
        floor_G0_factor = barrier.floor_max_fraction
        floor_fraction_derivative = 0.0
    else:
        floor = product
        floor_G0_factor = barrier.floor_fraction
        floor_fraction_derivative = G0
    x = sigma / sigc
    xn = np.power(x, max(barrier.exponent, 1e-9))
    decay = np.exp(-max(barrier.alpha, 0.0) * xn)
    value = floor + (G0-floor)*decay

    def tangent(dG0: float = 0.0, dsigc: float = 0.0, da: float = 0.0,
                dn: float = 0.0, dfloor_extra: float = 0.0) -> np.ndarray:
        dfloor = floor_G0_factor*dG0 + dfloor_extra
        logx = np.log(np.maximum(x, 1e-300))
        dxn = xn*(dn*logx - max(barrier.exponent, 1e-9)*dsigc/sigc)
        ddecay = decay*(-da*xn-max(barrier.alpha,0.0)*dxn)
        return dfloor + (dG0-dfloor)*decay + (G0-floor)*ddecay

    active_G0 = float(unclipped_G0 > 1e-12)
    active_sigc = float(unclipped_sigc > 1.0)
    tangents = {
        "temperature_K": tangent(
            dG0=barrier.gT_eV_per_K*active_G0,
            dsigc=barrier.sT_Pa_per_K*active_sigc,
        ),
        "cleave_G00_eV": tangent(dG0=active_G0),
        "cleave_gT_eV_per_K": tangent(dG0=dtemp*active_G0),
        "cleave_sigc0_GPa": tangent(dsigc=1e9*active_sigc),
        "cleave_sT_GPa_per_K": tangent(dsigc=dtemp*1e9*active_sigc),
        "cleave_exp_a": tangent(da=1.0),
        "cleave_exp_n": tangent(dn=1.0),
        "cleave_floor_frac": tangent(dfloor_extra=floor_fraction_derivative),
    }
    return np.maximum(value, 0.0), tangents


def f0_implicit_sensitivities(manifest: MaterialManifest, row: Mapping[str, Any],
                              temperature_K: float,
                              controls: MonotonicControls = MonotonicControls()) -> dict[str, float]:
    """Implicit first-passage tangents for F0, with exact gamma sensitivity."""
    result = solve_first_passage(manifest, row, temperature_K, "F0", controls)
    Kinit = float(result["K_init_MPa_sqrt_m"])
    if not math.isfinite(Kinit):
        return {name: math.nan for name in (
            "temperature_K", "ln_loading_rate", "cleave_G00_eV",
            "cleave_gT_eV_per_K", "cleave_sigc0_GPa", "cleave_sT_GPa_per_K",
            "cleave_exp_a", "cleave_exp_n", "cleave_floor_frac",
        )}
    count = max(int(math.ceil((Kinit-controls.K0_MPa_sqrt_m)/controls.dK_MPa_sqrt_m))*2, 256)
    K_MPa = np.linspace(controls.K0_MPa_sqrt_m, Kinit, count)
    stress = K_MPa*1e6/math.sqrt(2*math.pi*controls.r0_m)
    G, dG = _barrier_value_and_tangents(manifest.cleavage, stress, temperature_K)
    raw = manifest.cleavage.attempt_frequency_s*np.exp(
        np.clip(-G/(KB_EV_PER_K*temperature_K), -700.0, 0.0)
    )
    Lambda = cooperative_rate(raw, controls.cleavage_hits, controls.cleavage_tau_s)
    Q = cooperative_log_sensitivity(raw, controls.cleavage_hits, controls.cleavage_tau_s)
    endpoint = max(float(Lambda[-1]), 1e-300)
    output: dict[str, float] = {}
    for name, derivative in dG.items():
        if name == "temperature_K":
            dlnraw = G/(KB_EV_PER_K*temperature_K**2) - derivative/(KB_EV_PER_K*temperature_K)
        else:
            dlnraw = -derivative/(KB_EV_PER_K*temperature_K)
        dLambda = Lambda*Q*dlnraw
        # MPa integration divided by endpoint yields MPa per parameter.
        output[name] = -float(np.trapezoid(dLambda, K_MPa))/endpoint
    output["ln_loading_rate"] = controls.loading_rate_MPa_sqrt_m_s/endpoint
    return output


def controls_dict(controls: MonotonicControls) -> dict[str, Any]:
    return asdict(controls)


__all__ = [
    "MODEL_ID", "MonotonicControls", "controls_dict", "cooperative_rate",
    "cooperative_log_sensitivity", "finite_difference_sensitivity",
    "f0_implicit_sensitivities", "solve_first_passage", "stress_channels",
    "solve_first_passage_history",
]
