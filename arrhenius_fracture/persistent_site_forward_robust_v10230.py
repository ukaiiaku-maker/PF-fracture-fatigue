"""Partition-robust bounded forward VHCF marcher for v10.2.30.

This is the authoritative transient integrator used before any stationary-tail
propagation.  It retains the v10.2.30 full-step/two-half-step construction, but
adds error control on the active process-zone state that determines future
emission and cleavage response.  Cumulative emitted/escaped ledgers are never
used as absolute block targets.

The adaptive segment size is carried across outer driver calls so artificial
outer-block boundaries do not restart the step-growth sequence.  A hard maximum
transient segment prevents an unvalidated VHCF-sized explicit step.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np

from . import persistent_site_forward_coupled_hazard_v10230 as _base


MODEL_ID = "v10.2.30_partition_robust_forward_state_coupled_hazard_v2"

_ACTIVE_PROFILE_FIELDS = (
    "mobile_positive",
    "mobile_negative",
    "retained_positive",
    "retained_negative",
    "accumulated_slip_positive",
    "accumulated_slip_negative",
)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return max(value, minimum)


def forward_config(controller) -> dict[str, float | int]:
    config = dict(_base.forward_config(controller))
    config.update(
        {
            "state_profile_relative_tol": _env_float(
                "V10230_FORWARD_STATE_PROFILE_REL_TOL", 1.0e-4, 1.0e-12
            ),
            "mobile_relative_tol": _env_float(
                "V10230_FORWARD_MOBILE_REL_TOL", 1.0e-4, 1.0e-12
            ),
            "retained_relative_tol": _env_float(
                "V10230_FORWARD_RETAINED_REL_TOL", 1.0e-4, 1.0e-12
            ),
            "backstress_relative_tol": _env_float(
                "V10230_FORWARD_BACKSTRESS_REL_TOL", 1.0e-4, 1.0e-12
            ),
            "emission_log_rate_tol_decades": _env_float(
                "V10230_FORWARD_EMISSION_LOG_RATE_TOL_DECADES", 0.01, 1.0e-12
            ),
            "maximum_segment_cycles": _env_float(
                "V10230_FORWARD_MAX_SEGMENT_CYCLES", 1.0e6, 1.0e-12
            ),
        }
    )
    return config


def _profile_vector(engine) -> np.ndarray:
    mpz = getattr(engine, "mpz", None)
    values: list[np.ndarray] = [
        np.asarray(
            [
                float(getattr(mpz, "mobile_count", 0.0)),
                float(getattr(mpz, "retained_count", 0.0)),
            ],
            dtype=float,
        )
    ]
    for name in _ACTIVE_PROFILE_FIELDS:
        raw = getattr(mpz, name, None)
        if raw is None:
            continue
        array = np.asarray(raw, dtype=float).reshape(-1)
        if array.size:
            values.append(array)
    return np.concatenate(values) if values else np.zeros(0, dtype=float)


def _active_state(engine) -> dict[str, float]:
    mpz = getattr(engine, "mpz", None)
    geometry = dict(getattr(mpz, "persistent_site_last_geometry", {}) or {})
    profile = _profile_vector(engine)
    return {
        "mobile_count": float(getattr(mpz, "mobile_count", 0.0)),
        "retained_count": float(getattr(mpz, "retained_count", 0.0)),
        "sigma_back_Pa": float(
            getattr(mpz, "continuum_source_last_sigma_back_Pa", 0.0)
        ),
        "emission_hazard_s": max(
            float(getattr(mpz, "continuum_source_last_aggregate_hazard_s", 0.0)),
            0.0,
        ),
        "front_width_m": max(float(geometry.get("front_width_m", 0.0)), 0.0),
        "source_multiplicity": max(
            float(geometry.get("multiplicity_per_system", 0.0)), 0.0
        ),
        "profile_norm": float(np.linalg.norm(profile)),
    }


def _relative_difference(a: float, b: float, floor: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), floor)


def _log_difference(a: float, b: float) -> float:
    high = max(float(a), float(b), 0.0)
    if high <= 0.0:
        return 0.0
    low = max(min(max(float(a), 0.0), max(float(b), 0.0)), 1.0e-300)
    return abs(math.log10(high / low))


def _state_error_metrics(full, half, config) -> dict[str, float | str]:
    full_state = _active_state(full)
    half_state = _active_state(half)
    full_profile = _profile_vector(full)
    half_profile = _profile_vector(half)
    size = max(full_profile.size, half_profile.size)
    if full_profile.size != size:
        full_profile = np.pad(full_profile, (0, size - full_profile.size))
    if half_profile.size != size:
        half_profile = np.pad(half_profile, (0, size - half_profile.size))
    profile_error = float(np.linalg.norm(full_profile - half_profile)) / max(
        float(np.linalg.norm(full_profile)),
        float(np.linalg.norm(half_profile)),
        1.0,
    )
    metrics: dict[str, float | str] = {
        "state_profile_relative_error": profile_error,
        "mobile_relative_error": _relative_difference(
            full_state["mobile_count"], half_state["mobile_count"], 1.0
        ),
        "retained_relative_error": _relative_difference(
            full_state["retained_count"], half_state["retained_count"], 1.0
        ),
        "backstress_relative_error": _relative_difference(
            full_state["sigma_back_Pa"], half_state["sigma_back_Pa"], 1.0
        ),
        "emission_log_rate_error_decades": _log_difference(
            full_state["emission_hazard_s"], half_state["emission_hazard_s"]
        ),
    }
    ratios = {
        "state_profile": float(metrics["state_profile_relative_error"])
        / float(config["state_profile_relative_tol"]),
        "mobile": float(metrics["mobile_relative_error"])
        / float(config["mobile_relative_tol"]),
        "retained": float(metrics["retained_relative_error"])
        / float(config["retained_relative_tol"]),
        "backstress": float(metrics["backstress_relative_error"])
        / float(config["backstress_relative_tol"]),
        "emission_rate": float(metrics["emission_log_rate_error_decades"])
        / float(config["emission_log_rate_tol_decades"]),
    }
    metrics["state_maximum_error_ratio"] = max(ratios.values(), default=0.0)
    metrics["state_limiting_error"] = max(ratios, key=ratios.get) if ratios else "none"
    return metrics


def integrate_state_coupled_waveform(
    engine,
    controller,
    waveform,
    temperature_K: float,
    cycles_requested: float,
) -> dict[str, Any]:
    requested = max(float(cycles_requested), 0.0)
    frequency = max(float(waveform.frequency_Hz), 0.0)
    period = float(waveform.period_s)
    config = forward_config(controller)
    if requested <= 0.0 or frequency <= 0.0 or period <= 0.0:
        result = _base.integrate_state_coupled_waveform(
            engine, controller, waveform, temperature_K, cycles_requested
        )
        result["coupled_hazard_model_id"] = MODEL_ID
        return result

    totals: dict[str, float] = {}
    wake_totals: dict[str, float] = {}
    consumed_cycles = 0.0
    dB_total = 0.0
    dH_total = 0.0
    da_total = 0.0
    packet_mean = 0.0
    packet_variance = 0.0
    microsteps = 0
    accepted_segments = 0
    rejected_segments = 0
    trial_integrations = 0
    segment_audit: list[dict[str, Any]] = []
    last_result: dict[str, Any] = {}
    fired = False
    work_budget_exhausted = False
    zero_progress = False
    start_wall = time.monotonic()

    initial_endpoint = _base._endpoint(engine, controller, waveform, temperature_K)
    initial_active = _active_state(engine)
    lambda_history = [initial_endpoint["lambda_per_s"]]
    sigma_history = [initial_endpoint["sigma_Pa"]]
    shield_history = [initial_endpoint["K_shield_Pa_sqrt_m"]]

    minimum = float(config["minimum_cycles"])
    maximum_segment = float(config["maximum_segment_cycles"])
    carried = float(
        getattr(
            engine,
            "_v10230_forward_next_segment_cycles",
            config["initial_cycles"],
        )
    )
    segment_target = min(max(carried, minimum), maximum_segment)

    while consumed_cycles < requested and not fired:
        if (
            accepted_segments >= int(config["maximum_accepted_segments"])
            or trial_integrations + 3 > int(config["maximum_trial_integrations"])
        ):
            work_budget_exhausted = True
            break

        remaining = requested - consumed_cycles
        if remaining <= 0.0:
            break
        proposed = min(max(segment_target, minimum), maximum_segment)
        segment = min(proposed, remaining)
        boundary_truncated = segment + 1.0e-15 < proposed

        (
            full,
            full_result,
            full_endpoint,
            half,
            half_result,
            half_endpoint,
            evaluations,
        ) = _base._evaluate_segment(
            engine, controller, waveform, temperature_K, segment
        )
        trial_integrations += evaluations
        errors = dict(
            _base._error_metrics(
                full_endpoint,
                half_endpoint,
                full_result,
                half_result,
                waveform,
                config,
            )
        )
        state_errors = _state_error_metrics(full, half, config)
        errors.update(state_errors)
        if float(state_errors["state_maximum_error_ratio"]) > float(
            errors["maximum_error_ratio"]
        ):
            errors["maximum_error_ratio"] = float(
                state_errors["state_maximum_error_ratio"]
            )
            errors["limiting_error"] = str(state_errors["state_limiting_error"])

        trial_fired = bool(
            full_result.get("fired", False) or half_result.get("fired", False)
        )
        minimum_reached = segment <= minimum
        event_localized = (
            trial_fired
            and segment <= float(config["event_localization_cycles"])
        )
        accept = (
            (not trial_fired and float(errors["maximum_error_ratio"]) <= 1.0)
            or event_localized
            or minimum_reached
        )

        if not accept:
            rejected_segments += 1
            segment_target = max(
                segment * float(config["shrink_factor"]), minimum
            )
            continue

        _base._adopt_state(engine, half, fired=bool(half_result.get("fired", False)))
        actual_cycles = _base._positive(half_result.get("dt_consumed", 0.0)) * frequency
        if actual_cycles <= 0.0:
            zero_progress = True
            work_budget_exhausted = True
            break

        consumed_cycles += actual_cycles
        dB_segment = _base._positive(half_result.get("dB", 0.0))
        dB_total += dB_segment
        dH_total += _base._positive(
            half_result.get("physical_hazard_action_step", 0.0)
        )
        da_total += _base._positive(half_result.get("da", 0.0))
        packet_mean += _base._positive(half_result.get("packet_mean", 0.0))
        packet_variance += _base._positive(
            half_result.get("packet_variance_m2", 0.0)
        )
        microsteps += int(half_result.get("microsteps", 0))
        _base._sum_numeric(totals, dict(half_result.get("plastic", {})))
        _base._sum_numeric(wake_totals, dict(half_result.get("advance", {})))
        accepted_segments += 1
        fired = bool(half_result.get("fired", False))
        last_result = dict(half_result)

        active = _active_state(engine)
        lambda_history.append(half_endpoint["lambda_per_s"])
        sigma_history.append(half_endpoint["sigma_Pa"])
        shield_history.append(half_endpoint["K_shield_Pa_sqrt_m"])
        segment_audit.append(
            {
                "cycles_proposed": float(segment),
                "cycles_consumed": float(actual_cycles),
                "cumulative_cycles": float(consumed_cycles),
                "accepted": True,
                "fired": bool(fired),
                "B_end": float(getattr(engine, "B", 0.0)),
                "dB_segment": float(dB_segment),
                "dB_per_cycle": float(dB_segment / actual_cycles),
                "lambda_end_per_s": float(half_endpoint["lambda_per_s"]),
                "sigma_end_Pa": float(half_endpoint["sigma_Pa"]),
                "shield_end_Pa_sqrt_m": float(
                    half_endpoint["K_shield_Pa_sqrt_m"]
                ),
                "radius_end_m": float(half_endpoint["r_eff_m"]),
                "mobile_count_end": float(active["mobile_count"]),
                "retained_count_end": float(active["retained_count"]),
                "sigma_back_end_Pa": float(active["sigma_back_Pa"]),
                "emission_hazard_end_s": float(active["emission_hazard_s"]),
                "state_profile_norm_end": float(active["profile_norm"]),
                **errors,
            }
        )

        if (
            accepted_segments % int(config["heartbeat_segments"]) == 0
            and consumed_cycles < requested
        ):
            print(
                "  v10.2.30 robust forward VHCF heartbeat: "
                f"consumed={consumed_cycles:.9g}/{requested:.9g} cycles "
                f"accepted={accepted_segments} rejected={rejected_segments} "
                f"trials={trial_integrations} next={segment_target:.9g} "
                f"limit={errors['limiting_error']}",
                flush=True,
            )

        next_target = segment
        if float(errors["maximum_error_ratio"]) <= 0.125:
            next_target = segment * float(config["growth_factor"])
        if boundary_truncated:
            next_target = max(next_target, proposed)
        segment_target = min(max(next_target, minimum), maximum_segment)
        engine._v10230_forward_next_segment_cycles = float(segment_target)

        if fired or actual_cycles + 1.0e-15 < segment:
            break

    if fired:
        engine._v10230_forward_next_segment_cycles = float(config["initial_cycles"])
    else:
        engine._v10230_forward_next_segment_cycles = float(segment_target)

    final_endpoint = _base._endpoint(engine, controller, waveform, temperature_K)
    final_active = _active_state(engine)
    combined = dict(last_result)
    combined.update(
        {
            "fired": bool(fired),
            "n_fire": 1 if fired else 0,
            "v_crack": da_total / (consumed_cycles * period)
            if consumed_cycles > 0.0
            else 0.0,
            "dB": float(dB_total),
            "physical_hazard_action_step": float(dH_total),
            "da": float(da_total),
            "dt_consumed": float(consumed_cycles * period),
            "dt_unused": max((requested - consumed_cycles) * period, 0.0),
            "packet_mean": float(packet_mean),
            "packet_variance_m2": float(packet_variance),
            "lambda_c": float(final_endpoint["lambda_per_s"]),
            "lambda_c_raw": float(final_endpoint["lambda_per_s"]),
            "sigma_tip": float(final_endpoint["sigma_Pa"]),
            "plastic": totals,
            "advance": wake_totals,
            "microsteps": int(microsteps),
            "coupled_hazard_model_id": MODEL_ID,
            "coupled_hazard_phase_resolved": True,
            "coupled_hazard_frozen_within_outer_block": False,
            "coupled_hazard_forward_marcher": True,
            "coupled_hazard_partition_robust_state_control": True,
            "coupled_hazard_recursive_bisection": False,
            "coupled_hazard_two_half_step_state_committed": True,
            "coupled_hazard_third_commit_integration": False,
            "coupled_hazard_accepted_segments": int(accepted_segments),
            "coupled_hazard_rejected_splits": int(rejected_segments),
            "coupled_hazard_trial_integrations": int(trial_integrations),
            "coupled_hazard_work_budget_exhausted": bool(work_budget_exhausted),
            "coupled_hazard_zero_progress": bool(zero_progress),
            "coupled_hazard_partial_return": bool(
                not fired and consumed_cycles + 1.0e-15 < requested
            ),
            "coupled_hazard_cycles_requested": float(requested),
            "coupled_hazard_cycles_consumed": float(consumed_cycles),
            "coupled_hazard_lambda_start_s": float(lambda_history[0]),
            "coupled_hazard_lambda_end_s": float(lambda_history[-1]),
            "coupled_hazard_lambda_start_per_s": float(lambda_history[0]),
            "coupled_hazard_lambda_end_per_s": float(lambda_history[-1]),
            "coupled_hazard_lambda_min_per_s": float(min(lambda_history)),
            "coupled_hazard_lambda_max_per_s": float(max(lambda_history)),
            "coupled_hazard_log_lambda_span_decades": float(
                _base._legacy._log_span(lambda_history)
            ),
            "coupled_hazard_sigma_start_Pa": float(sigma_history[0]),
            "coupled_hazard_sigma_end_Pa": float(sigma_history[-1]),
            "coupled_hazard_shield_start_Pa_sqrt_m": float(shield_history[0]),
            "coupled_hazard_shield_end_Pa_sqrt_m": float(shield_history[-1]),
            "coupled_hazard_mobile_start": float(initial_active["mobile_count"]),
            "coupled_hazard_mobile_end": float(final_active["mobile_count"]),
            "coupled_hazard_retained_start": float(initial_active["retained_count"]),
            "coupled_hazard_retained_end": float(final_active["retained_count"]),
            "coupled_hazard_backstress_start_Pa": float(
                initial_active["sigma_back_Pa"]
            ),
            "coupled_hazard_backstress_end_Pa": float(
                final_active["sigma_back_Pa"]
            ),
            "coupled_hazard_event_localized": bool(
                fired and consumed_cycles + 1.0e-15 < requested
            ),
            "coupled_hazard_wall_seconds": float(time.monotonic() - start_wall),
            "coupled_hazard_next_segment_cycles": float(segment_target),
            "coupled_hazard_config": dict(config),
            "coupled_hazard_segments": segment_audit,
        }
    )
    return combined


__all__ = [
    "MODEL_ID",
    "forward_config",
    "integrate_state_coupled_waveform",
    "_active_state",
    "_profile_vector",
    "_state_error_metrics",
]
