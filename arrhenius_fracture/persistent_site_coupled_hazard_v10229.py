"""Adaptive state-coupled cleavage integration for v10.2.29 VHCF blocks.

The existing persistent-site commit already advances emission, transport, storage,
shielding, blunting, and stochastic first passage. This module adds an outer
cycle-domain quadrature that repeatedly re-evaluates the phase-averaged cleavage
hazard along that evolving state. No independent fatigue law is introduced.
"""
from __future__ import annotations

import copy
import dataclasses
import math
import os
from typing import Any

import numpy as np


MODEL_ID = "v10.2.29_state_coupled_waveform_hazard_v1"


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _positive(value: Any, default: float = 0.0) -> float:
    return max(_finite(value, default), 0.0)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    value = _finite(os.environ.get(name, default), default)
    return max(value, minimum)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = int(default)
    return max(value, minimum)


def coupled_hazard_config(controller) -> dict[str, float | int]:
    """Return numerical controls for the adaptive coupled hazard quadrature."""
    min_cycles = max(
        _positive(getattr(controller.cfg, "min_block_cycles", 1.0e-6), 1.0e-6),
        _env_float("V10229_COUPLED_HAZARD_MIN_CYCLES", 1.0e-6, 1.0e-18),
    )
    return {
        "log_lambda_tol_decades": _env_float(
            "V10229_COUPLED_HAZARD_LOG_TOL_DECADES", 0.05, 1.0e-6
        ),
        "sigma_relative_tol": _env_float(
            "V10229_COUPLED_HAZARD_SIGMA_REL_TOL", 0.02, 1.0e-8
        ),
        "state_target_fraction": _env_float(
            "V10229_COUPLED_HAZARD_STATE_TARGET_FRACTION", 0.25, 1.0e-6
        ),
        "absolute_dB_tol": _env_float(
            "V10229_COUPLED_HAZARD_ABS_DB_TOL", 1.0e-6, 0.0
        ),
        "stationary_log_tol_decades": _env_float(
            "V10229_COUPLED_HAZARD_STATIONARY_LOG_TOL_DECADES", 0.01, 0.0
        ),
        "stationary_state_fraction": _env_float(
            "V10229_COUPLED_HAZARD_STATIONARY_STATE_FRACTION", 1.0e-3, 0.0
        ),
        "minimum_cycles": min_cycles,
        "maximum_depth": _env_int("V10229_COUPLED_HAZARD_MAX_DEPTH", 60, 4),
    }


def _phase_statistics(engine, controller, waveform, temperature_K: float) -> dict[str, float]:
    phases = np.asarray(controller._phases(), dtype=float)
    if phases.size < 1:
        raise ValueError("state-coupled cyclic integration requires waveform phases")
    K_values = np.asarray(waveform.K_phase(phases), dtype=float).reshape(-1)
    if K_values.size != phases.size:
        raise ValueError("waveform K_phase output does not match quadrature")

    # Optional crack-rebonding (v10.2.30, default off): confirmed by direct
    # construction of the real production engine class
    # (CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine) that
    # CoupledPersistentSiteCyclicTipEngine.cycle_step_waveform -- which
    # bypasses BOTH kinetic_tip_cell.py::cycle_step_waveform AND
    # persistent_site_cyclic_v10229.py::cycle_step_waveform entirely,
    # calling integrate_state_coupled_waveform (this module) instead -- is
    # the actual production commit pathway for the real persistent-site
    # engine hierarchy. See docs/v10_2_30_crack_rebonding_equation_lineage.md
    # for the full correction record (two prior injection-point attempts
    # were dead code for this real engine class). K_values (feeding
    # sigma_avg_Pa, hence stress_override -> _plastic_half_step, hence
    # emission) is never touched; only a separate sig_cleave array, used
    # solely for the lambda_cleave call below, is rebond-aware.
    from . import crack_rebonding_v10230 as _rebond

    rebonding_state = getattr(engine, "_rebonding_state", None)
    rebonding_active = rebonding_state is not None and _rebond.rebonding_kinetics_active(
        rebonding_state.cfg
    )
    K_signed_phase = None
    K_rebond_phase = None
    K_shield_now = 0.0
    r_eff_now = 1.0e-30
    if rebonding_active:
        signed_waveform = dataclasses.replace(waveform, closure_clip=False)
        phase_offset_rad = _rebond.chronological_phase_offset_rad(
            rebonding_state.elapsed_time_s, waveform.period_s
        )
        K_signed_phase = np.asarray(
            signed_waveform.K_phase(phases + phase_offset_rad), dtype=float
        )
        Eprime_Pa = _rebond.reduced_modulus_Pa(engine.G, engine.nu)
        r_contact_m = max(engine.r_eff(), rebonding_state.cfg.contact_radius_min_m)
        active_patches = [p for p in rebonding_state.active if not p.retired]
        patch_states_now = {p.patch_id: p.state_vector() for p in active_patches}
        K_rebond_phase = _rebond.representative_cycle_K_rebond(
            active_patches=active_patches,
            patch_states=patch_states_now,
            K_phase=K_signed_phase,
            dt_phase=float(waveform.period_s) / float(phases.size),
            r_contact_m=r_contact_m,
            cfg=rebonding_state.cfg,
            T_K=temperature_K,
            Eprime_Pa=Eprime_Pa,
        )
        K_shield_now = engine.K_shield()
        r_eff_now = engine.r_eff()

    sigma: list[float] = []
    lambdas: list[float] = []
    raw: list[float] = []
    barriers: list[float] = []
    for _idx, value in enumerate(K_values):
        K = max(float(value), 0.0)
        sig = _positive(engine.sigma_tip(K))
        if rebonding_active:
            sig_cleave = _rebond.cleavage_stress_with_rebond(
                float(K_signed_phase[_idx]), K_shield_now, float(K_rebond_phase[_idx]), r_eff_now
            )
        else:
            sig_cleave = sig
        lam, lam_raw, Gc = engine.lambda_cleave(sig_cleave, float(temperature_K))
        sigma.append(sig)
        lambdas.append(_positive(lam))
        raw.append(_positive(lam_raw))
        barriers.append(_finite(Gc))

    engine.sigma_tip(float(waveform.Kmax))
    shield = (
        _finite(engine.K_shield())
        if callable(getattr(engine, "K_shield", None))
        else 0.0
    )
    return {
        "lambda_avg_s": float(np.mean(lambdas)),
        "lambda_min_s": float(np.min(lambdas)),
        "lambda_max_s": float(np.max(lambdas)),
        "lambda_raw_avg_s": float(np.mean(raw)),
        "Gc_avg_J": float(np.mean(barriers)),
        "sigma_avg_Pa": float(np.mean(sigma)),
        "sigma_min_Pa": float(np.min(sigma)),
        "sigma_max_Pa": float(np.max(sigma)),
        "r_eff_m": _positive(engine.r_eff(), 1.0e-30),
        "K_shield_Pa_sqrt_m": shield,
    }


def _state_snapshot(engine) -> dict[str, float]:
    mpz = engine.mpz
    return {
        "emitted_total": _positive(getattr(mpz, "emitted_total", 0.0)),
        "mobile_count": _positive(getattr(mpz, "mobile_count", 0.0)),
        "retained_count": _positive(getattr(mpz, "retained_count", 0.0)),
        "escaped_total": _positive(getattr(mpz, "escaped_total", 0.0)),
    }


def _state_targets(controller) -> dict[str, float]:
    cfg = controller.cfg
    pairs = (
        ("emitted_total", "target_dN_emit"),
        ("mobile_count", "target_dN_mobile"),
        ("retained_count", "target_dN_store"),
        ("escaped_total", "target_dN_escape"),
    )
    targets: dict[str, float] = {}
    for key, attr in pairs:
        value = _finite(getattr(cfg, attr, math.inf), math.inf)
        if math.isfinite(value) and value > 0.0:
            targets[key] = value
    return targets


def _state_target_ratio(
    start: dict[str, float],
    end: dict[str, float],
    targets: dict[str, float],
) -> float:
    ratios = [
        abs(float(end[key]) - float(start[key])) / target
        for key, target in targets.items()
    ]
    return max(ratios, default=0.0)


def _log_span(values: list[float]) -> float:
    positive = [max(float(value), 0.0) for value in values]
    high = max(positive, default=0.0)
    if high <= 0.0:
        return 0.0
    low = max(min(positive, default=0.0), 1.0e-300)
    return math.log10(high / low)


def _sum_numeric(target: dict[str, float], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            target[key] = target.get(key, 0.0) + float(value)


def _commit_constant_segment(
    engine,
    controller,
    waveform,
    temperature_K: float,
    cycles: float,
    sigma_average_Pa: float,
    lambda_average_s: float,
) -> dict[str, Any]:
    engine.sigma_tip(float(waveform.Kmax))

    from . import crack_rebonding_v10230 as _rebond

    rebonding_state = getattr(engine, "_rebonding_state", None)
    rebonding_active = rebonding_state is not None and _rebond.rebonding_kinetics_active(
        rebonding_state.cfg
    )
    dt_segment = max(float(cycles), 0.0) * float(waveform.period_s)
    if rebonding_active:
        phases = np.asarray(controller._phases(), dtype=float)
        dt_phase = float(waveform.period_s) / float(phases.size)
        signed_waveform = dataclasses.replace(waveform, closure_clip=False)
        phase_offset_rad = _rebond.chronological_phase_offset_rad(
            rebonding_state.elapsed_time_s, waveform.period_s
        )
        K_signed_phase = np.asarray(
            signed_waveform.K_phase(phases + phase_offset_rad), dtype=float
        )
        engine._rebonding_block_context = {
            "K_signed_phase": K_signed_phase,
            "dt_phase": dt_phase,
            "n_phase": int(phases.size),
            "r_contact_m": max(engine.r_eff(), rebonding_state.cfg.contact_radius_min_m),
            "Eprime_Pa": _rebond.reduced_modulus_Pa(engine.G, engine.nu),
            "T_K": float(temperature_K),
            "K_shield_Pa_sqrt_m": engine.K_shield(),
            "r_eff_m": engine.r_eff(),
            "B_start": float(engine.B),
            "period_s": float(waveform.period_s),
        }

    result = engine._integrate_coupled(
        float(waveform.Kmax),
        float(temperature_K),
        dt_segment,
        stress_override=max(float(sigma_average_Pa), 0.0),
        lambda_override=max(float(lambda_average_s), 0.0),
    )

    if rebonding_active and not result.get("fired", False):
        # This segment did not fire: its full nominal duration was genuinely
        # consumed, so finalize by committing the exact segment-length wake
        # advance now (mirrors the kinetic_tip_cell.py/persistent_site_
        # cyclic_v10229.py injection points exactly). If it fired, the wake
        # state is left untouched here -- the transactional commit
        # (accepted) or rollback (rejected) happens later in
        # commit_energy_gated_event/restore_geometry_veto, using the
        # rebonding_block_context stashed above via _energy_gate_pending.
        # Uses result["dt_consumed"] (the actual elapsed segment time) in
        # preference to the nominal dt_segment, since the two can differ.
        from .crack_rebonding_kinetics_v10230 import build_phase_factors, propagate

        ctx = engine._rebonding_block_context
        dt_actual = max(float(result.get("dt_consumed", dt_segment)), 0.0)
        active_patches = [p for p in rebonding_state.active if not p.retired]
        end_states = {}
        for p in active_patches:
            Q_list_p = [
                _rebond.patch_Q(float(K), p.s_j_m, ctx["r_contact_m"], rebonding_state.cfg, temperature_K)
                for K in ctx["K_signed_phase"]
            ]
            factors_p = build_phase_factors(Q_list_p, ctx["dt_phase"])
            end_states[p.patch_id] = propagate(
                p.state_vector(), Q_list_p, factors_p, k0=0,
                dt=dt_actual, dt_phase=ctx["dt_phase"],
            )
        rebonding_state.commit_no_event_block(end_states, ctx["Eprime_Pa"])
        rebonding_state.elapsed_time_s = (
            rebonding_state.elapsed_time_s + dt_actual
        ) % waveform.period_s
        engine._rebonding_block_context = None

    return result


def integrate_state_coupled_waveform(
    engine,
    controller,
    waveform,
    temperature_K: float,
    cycles_requested: float,
) -> dict[str, Any]:
    """Advance a block with cleavage hazard coupled to the evolving tip state."""
    requested = max(float(cycles_requested), 0.0)
    period = float(waveform.period_s)
    frequency = max(float(waveform.frequency_Hz), 0.0)
    config = coupled_hazard_config(controller)
    state_targets = _state_targets(controller)
    threshold = max(
        _positive(getattr(engine, "hazard_threshold_action", 1.0), 1.0),
        1.0e-300,
    )

    totals: dict[str, float] = {}
    wake_totals: dict[str, float] = {}
    consumed_s = 0.0
    dB_total = 0.0
    dH_total = 0.0
    da_total = 0.0
    packet_mean = 0.0
    packet_variance = 0.0
    internal_microsteps = 0
    accepted_segments = 0
    rejected_splits = 0
    maximum_depth_used = 0
    transient_cycles = 0.0
    stationary_cycles = 0.0
    lambda_evaluations = 0
    lambda_history: list[float] = []
    sigma_history: list[float] = []
    segment_audit: list[dict[str, Any]] = []
    fired = False
    last_result: dict[str, Any] = {}
    last_stats = _phase_statistics(engine, controller, waveform, temperature_K)
    lambda_evaluations += 1
    lambda_history.append(last_stats["lambda_avg_s"])
    sigma_history.append(last_stats["sigma_avg_Pa"])

    def commit_interval(cycles: float, depth: int) -> None:
        nonlocal consumed_s, dB_total, dH_total, da_total
        nonlocal packet_mean, packet_variance, internal_microsteps
        nonlocal accepted_segments, rejected_splits, maximum_depth_used
        nonlocal transient_cycles, stationary_cycles, lambda_evaluations
        nonlocal fired, last_result, last_stats

        if fired or cycles <= 0.0:
            return
        maximum_depth_used = max(maximum_depth_used, depth)
        start_stats = _phase_statistics(engine, controller, waveform, temperature_K)
        start_state = _state_snapshot(engine)
        lambda_evaluations += 1

        provisional = copy.deepcopy(engine)
        provisional_result = _commit_constant_segment(
            provisional,
            controller,
            waveform,
            temperature_K,
            cycles,
            start_stats["sigma_avg_Pa"],
            start_stats["lambda_avg_s"],
        )
        provisional_fired = bool(provisional_result.get("fired", False))
        end_stats = _phase_statistics(provisional, controller, waveform, temperature_K)
        end_state = _state_snapshot(provisional)
        lambda_evaluations += 1

        half_cycles = 0.5 * cycles
        midpoint = copy.deepcopy(engine)
        midpoint_result = _commit_constant_segment(
            midpoint,
            controller,
            waveform,
            temperature_K,
            half_cycles,
            start_stats["sigma_avg_Pa"],
            start_stats["lambda_avg_s"],
        )
        midpoint_fired = bool(midpoint_result.get("fired", False))
        mid_stats = _phase_statistics(midpoint, controller, waveform, temperature_K)
        lambda_evaluations += 1

        lambda_values = [
            start_stats["lambda_avg_s"],
            mid_stats["lambda_avg_s"],
            end_stats["lambda_avg_s"],
        ]
        sigma_values = [
            start_stats["sigma_avg_Pa"],
            mid_stats["sigma_avg_Pa"],
            end_stats["sigma_avg_Pa"],
        ]
        lambda_simpson = (
            lambda_values[0] + 4.0 * lambda_values[1] + lambda_values[2]
        ) / 6.0
        sigma_simpson = (
            sigma_values[0] + 4.0 * sigma_values[1] + sigma_values[2]
        ) / 6.0
        log_span = _log_span(lambda_values)
        sigma_relative = (
            max(sigma_values) - min(sigma_values)
        ) / max(max(sigma_values), 1.0)
        state_ratio = _state_target_ratio(start_state, end_state, state_targets)
        dB_bound = max(lambda_values) * cycles * period / threshold
        hazard_irrelevant = dB_bound <= float(config["absolute_dB_tol"])
        hazard_accurate = hazard_irrelevant or (
            log_span <= float(config["log_lambda_tol_decades"])
            and sigma_relative <= float(config["sigma_relative_tol"])
        )
        state_accurate = state_ratio <= float(config["state_target_fraction"])
        minimum_reached = cycles <= float(config["minimum_cycles"])
        maximum_reached = depth >= int(config["maximum_depth"])
        accept = (
            (not provisional_fired)
            and (not midpoint_fired)
            and hazard_accurate
            and state_accurate
        ) or minimum_reached or maximum_reached

        if not accept:
            rejected_splits += 1
            commit_interval(half_cycles, depth + 1)
            if not fired:
                commit_interval(cycles - half_cycles, depth + 1)
            return

        result = _commit_constant_segment(
            engine,
            controller,
            waveform,
            temperature_K,
            cycles,
            sigma_simpson,
            lambda_simpson,
        )
        actual_cycles = _positive(result.get("dt_consumed", 0.0)) * frequency
        consumed_s += _positive(result.get("dt_consumed", 0.0))
        dB_total += _positive(result.get("dB", 0.0))
        dH_total += _positive(result.get("physical_hazard_action_step", 0.0))
        da_total += _positive(result.get("da", 0.0))
        packet_mean += _positive(result.get("packet_mean", 0.0))
        packet_variance += _positive(result.get("packet_variance_m2", 0.0))
        internal_microsteps += int(result.get("microsteps", 0))
        _sum_numeric(totals, dict(result.get("plastic", {})))
        _sum_numeric(wake_totals, dict(result.get("advance", {})))
        accepted_segments += 1
        fired = bool(result.get("fired", False))
        last_result = dict(result)
        last_stats = _phase_statistics(engine, controller, waveform, temperature_K)
        lambda_evaluations += 1
        lambda_history.extend(lambda_values)
        lambda_history.append(last_stats["lambda_avg_s"])
        sigma_history.extend(sigma_values)
        sigma_history.append(last_stats["sigma_avg_Pa"])

        stationary = (
            log_span <= float(config["stationary_log_tol_decades"])
            and state_ratio <= float(config["stationary_state_fraction"])
            and sigma_relative <= 0.25 * float(config["sigma_relative_tol"])
        )
        if stationary:
            stationary_cycles += actual_cycles
        else:
            transient_cycles += actual_cycles
        segment_audit.append(
            {
                "cycles_requested": float(cycles),
                "cycles_consumed": float(actual_cycles),
                "depth": int(depth),
                "lambda_start_s": float(lambda_values[0]),
                "lambda_mid_s": float(lambda_values[1]),
                "lambda_end_trial_s": float(lambda_values[2]),
                "lambda_simpson_s": float(lambda_simpson),
                "lambda_end_committed_s": float(last_stats["lambda_avg_s"]),
                "log_lambda_span_decades": float(log_span),
                "sigma_relative_span": float(sigma_relative),
                "state_target_ratio": float(state_ratio),
                "dB_upper_bound": float(dB_bound),
                "hazard_irrelevant": bool(hazard_irrelevant),
                "stationary": bool(stationary),
                "fired": bool(fired),
            }
        )

    commit_interval(requested, 0)

    combined = dict(last_result)
    combined.update(
        {
            "fired": bool(fired),
            "n_fire": 1 if fired else 0,
            "v_crack": da_total / consumed_s if consumed_s > 0.0 else 0.0,
            "dB": float(dB_total),
            "physical_hazard_action_step": float(dH_total),
            "da": float(da_total),
            "dt_consumed": float(consumed_s),
            "dt_unused": max(requested * period - consumed_s, 0.0),
            "packet_mean": float(packet_mean),
            "packet_variance_m2": float(packet_variance),
            "lambda_c": float(last_stats["lambda_avg_s"]),
            "lambda_c_raw": float(last_stats["lambda_raw_avg_s"]),
            "Gc_J": float(last_stats["Gc_avg_J"]),
            "sigma_tip": float(last_stats["sigma_avg_Pa"]),
            "plastic": totals,
            "advance": wake_totals,
            "microsteps": int(internal_microsteps),
            "coupled_hazard_model_id": MODEL_ID,
            "coupled_hazard_phase_resolved": True,
            "coupled_hazard_frozen_within_outer_block": False,
            "coupled_hazard_accepted_segments": int(accepted_segments),
            "coupled_hazard_rejected_splits": int(rejected_splits),
            "coupled_hazard_maximum_depth": int(maximum_depth_used),
            "coupled_hazard_lambda_evaluations": int(lambda_evaluations),
            "coupled_hazard_lambda_start_s": float(
                lambda_history[0] if lambda_history else 0.0
            ),
            "coupled_hazard_lambda_end_s": float(
                lambda_history[-1] if lambda_history else 0.0
            ),
            "coupled_hazard_lambda_min_s": float(min(lambda_history, default=0.0)),
            "coupled_hazard_lambda_max_s": float(max(lambda_history, default=0.0)),
            "coupled_hazard_log_lambda_span_decades": float(
                _log_span(lambda_history)
            ),
            "coupled_hazard_sigma_start_Pa": float(
                sigma_history[0] if sigma_history else 0.0
            ),
            "coupled_hazard_sigma_end_Pa": float(
                sigma_history[-1] if sigma_history else 0.0
            ),
            "coupled_hazard_transient_cycles": float(transient_cycles),
            "coupled_hazard_stationary_tail_cycles": float(stationary_cycles),
            "coupled_hazard_event_localized": bool(
                fired and consumed_s + 1.0e-15 < requested * period
            ),
            "coupled_hazard_config": dict(config),
            "coupled_hazard_segments": segment_audit,
        }
    )
    return combined


__all__ = [
    "MODEL_ID",
    "coupled_hazard_config",
    "integrate_state_coupled_waveform",
]
