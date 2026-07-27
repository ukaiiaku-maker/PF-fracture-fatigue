"""Audit adapter for the v10.2.29 persistent-site cyclic engine."""
from __future__ import annotations

import math

from .persistent_site_cyclic_v10229 import PersistentSiteCyclicTipEngine


def _finite_nonnegative(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number) or number < 0.0:
        return float(default)
    return number


def _add_persistent_fields(engine, result: dict) -> dict:
    sigma_back = float(
        getattr(engine.mpz, "continuum_source_last_sigma_back_Pa", 0.0)
    )
    aggregate_hazard = float(
        getattr(engine.mpz, "continuum_source_last_aggregate_hazard_s", 0.0)
    )
    geometry = dict(getattr(engine.mpz, "persistent_site_last_geometry", {}))
    Kmax = max(float(result.get("Kmax_Pa_sqrt_m", 0.0)), 0.0)
    result["sigma_tip"] = float(result.get("sigma_tip", engine.sigma_tip(Kmax)))
    result["r_eff"] = float(result.get("r_eff", engine.r_eff()))
    result["sigma_back"] = sigma_back
    result["sigma_back_pre_renewal"] = sigma_back
    result["lambda_e"] = aggregate_hazard
    result["persistent_site_aggregate_hazard_s"] = aggregate_hazard
    result["persistent_site_multiplicity_per_system"] = float(
        geometry.get("multiplicity_per_system", 0.0)
    )
    result["persistent_site_front_width_m"] = float(
        geometry.get("front_width_m", 0.0)
    )
    result["persistent_site_source_area_m2"] = float(
        geometry.get("source_area_m2", 0.0)
    )
    result["persistent_tip_radius_m"] = float(
        geometry.get("tip_radius_m", engine.r_eff())
    )
    result["persistent_source_inventory_active"] = False
    result["persistent_source_refresh_active"] = False
    result["explicit_recovery_active"] = False
    return result


def _adaptive_diagnostics(controller, engine, result, targets, predictor):
    """Recover the physical adaptive limiter hidden by driver force_cycles plumbing."""
    cfg = controller.cfg
    mode = str(getattr(cfg, "cycle_block_mode", "unknown") or "unknown").lower()
    max_block = _finite_nonnegative(getattr(cfg, "max_block_cycles", 0.0))
    nominal = _finite_nonnegative(getattr(cfg, "block_cycles", 0.0))
    if mode in ("hazard", "hazard_limited", "rate", "auto"):
        base_name = "max_block_cycles"
        base = max_block
    else:
        base_name = "block_cycles"
        base = min(max_block, nominal)

    effective_targets = dict(targets)
    B_pre = _finite_nonnegative(result.get("B_pre", getattr(engine, "B", 0.0)))
    remaining = max(1.0 - B_pre, 0.0)
    if effective_targets["cleavage_clock"] > 0.0:
        effective_targets["cleavage_clock"] = min(
            effective_targets["cleavage_clock"], remaining
        )

    limits = {base_name: base}
    for name, target in effective_targets.items():
        rate = predictor.get(name, 0.0)
        if target > 0.0 and math.isfinite(target) and rate > 0.0 and math.isfinite(rate):
            limits[name] = target / rate

    limiter = min(limits, key=limits.get) if limits else base_name
    unlimited = _finite_nonnegative(limits.get(limiter, base))
    min_block = _finite_nonnegative(getattr(cfg, "min_block_cycles", 0.0))
    if unlimited < min_block:
        limiter = "min_block_cycles"
    elif max_block > 0.0 and unlimited > max_block:
        limiter = "max_block_cycles"
    return limiter, unlimited, limits, effective_targets


def _cycle_block_audit_fields(controller, engine, result: dict) -> dict:
    """Return JSON-safe diagnostics for adaptive cycle-block selection."""
    cfg = controller.cfg
    targets = {
        "cleavage_clock": _finite_nonnegative(getattr(cfg, "target_dB", 0.0)),
        "stored_pz": _finite_nonnegative(getattr(cfg, "target_dN_store", 0.0)),
        "emitted_pz": _finite_nonnegative(getattr(cfg, "target_dN_emit", 0.0)),
        "mobile_pz": _finite_nonnegative(getattr(cfg, "target_dN_mobile", 0.0)),
        "escape_pz": _finite_nonnegative(getattr(cfg, "target_dN_escape", 0.0)),
        "peierls_clock": _finite_nonnegative(
            getattr(cfg, "target_dN_peierls", 0.0)
        ),
        "taylor_clock": _finite_nonnegative(
            getattr(cfg, "target_dN_taylor", 0.0)
        ),
    }
    predictor = {
        "cleavage_clock": _finite_nonnegative(result.get("mu_cleave_pred", 0.0)),
        "stored_pz": _finite_nonnegative(result.get("store_per_cycle", 0.0)),
        "emitted_pz": _finite_nonnegative(result.get("mu_emit", 0.0)),
        "mobile_pz": _finite_nonnegative(result.get("mobile_per_cycle", 0.0)),
        "escape_pz": _finite_nonnegative(result.get("escape_per_cycle", 0.0)),
        "peierls_clock": _finite_nonnegative(
            result.get("peierls_per_cycle", 0.0)
        ),
        "taylor_clock": _finite_nonnegative(
            result.get("taylor_per_cycle", 0.0)
        ),
    }
    requested = _finite_nonnegative(result.get("cycles_requested", 0.0))
    consumed = _finite_nonnegative(result.get("cycles_consumed", 0.0))
    fraction = consumed / requested if requested > 0.0 else 0.0

    applied_limiter = str(result.get("cycle_limiter", "unknown"))
    reported_candidates = {}
    for key, value in dict(result.get("cycle_candidate_limits", {})).items():
        number = _finite_nonnegative(value, default=-1.0)
        if number >= 0.0:
            reported_candidates[str(key)] = number

    limiter, unlimited, reconstructed, effective_targets = _adaptive_diagnostics(
        controller, engine, result, targets, predictor
    )
    if applied_limiter not in ("global_forced", "unknown", ""):
        limiter = applied_limiter
        unlimited = _finite_nonnegative(result.get("cycle_unlimited", unlimited))
        reconstructed.update(reported_candidates)

    forced_below = requested + 1.0e-12 * max(unlimited, 1.0) < unlimited
    return {
        "cycle_block_mode": str(
            getattr(cfg, "cycle_block_mode", "unknown") or "unknown"
        ),
        "cycle_limiter": limiter,
        "cycle_applied_limiter": applied_limiter,
        "cycle_unlimited": unlimited,
        "cycle_candidate_limits": reconstructed,
        "cycle_reported_candidate_limits": reported_candidates,
        "cycle_target_increments": targets,
        "cycle_effective_target_increments": effective_targets,
        "cycle_predicted_increments_per_cycle": predictor,
        "cycle_forced_below_adaptive_limit": forced_below,
        "cycles_consumed_fraction": fraction,
        "cycle_min_block_cycles": _finite_nonnegative(
            getattr(cfg, "min_block_cycles", 0.0)
        ),
        "cycle_max_block_cycles": _finite_nonnegative(
            getattr(cfg, "max_block_cycles", 0.0)
        ),
        "cycle_nominal_block_cycles": _finite_nonnegative(
            getattr(cfg, "block_cycles", 0.0)
        ),
        "state_N_em": _finite_nonnegative(result.get("N_em", 0.0)),
        "state_mobile_count": _finite_nonnegative(
            getattr(engine.mpz, "mobile_count", 0.0)
        ),
        "state_retained_count": _finite_nonnegative(
            getattr(engine.mpz, "retained_count", 0.0)
        ),
        "state_emitted_total": _finite_nonnegative(
            getattr(engine.mpz, "emitted_total", 0.0)
        ),
        "state_escaped_total": _finite_nonnegative(
            getattr(engine.mpz, "escaped_total", 0.0)
        ),
        "state_micro_advance_total_m": _finite_nonnegative(
            getattr(engine, "micro_advance_total_m", 0.0)
        ),
        "state_active_K_shield_signed_Pa_sqrt_m": float(
            result.get("kinetic_active_K_shield_signed_Pa_sqrt_m", 0.0)
        ),
        "state_wake_K_shield_signed_Pa_sqrt_m": float(
            result.get("kinetic_wake_K_shield_signed_Pa_sqrt_m", 0.0)
        ),
    }


class AuditedPersistentSiteCyclicTipEngine(PersistentSiteCyclicTipEngine):
    """Expose the same persistent-source audit fields for monotonic and cyclic calls."""

    def step(self, K, T, dt):
        return _add_persistent_fields(self, super().step(K, T, dt))

    def cycle_step_waveform(
        self, controller, waveform, T_K: float, requested_cycles=None, force_cycles=None
    ):
        result = _add_persistent_fields(
            self,
            super().cycle_step_waveform(
                controller,
                waveform,
                T_K,
                requested_cycles=requested_cycles,
                force_cycles=force_cycles,
            ),
        )
        type(self)._audit_records.append(
            {
                "engine_id": int(getattr(self, "_engine_id", -1)),
                "loading_mode": "cyclic",
                "time_s": float(result.get("time_s", self.t)),
                "temperature_K": float(T_K),
                "cycles_requested": float(result.get("cycles_requested", 0.0)),
                "cycles_consumed": float(result.get("cycles_consumed", 0.0)),
                "cycles_unused": float(result.get("cycles_unused", 0.0)),
                "event_localized": bool(result.get("cycle_event_localized", False)),
                "fired": bool(result.get("fired", False)),
                "B": float(result.get("B", self.B)),
                "persistent_sigma_back_Pa": float(result["sigma_back"]),
                "persistent_aggregate_emission_hazard_s": float(
                    result["persistent_site_aggregate_hazard_s"]
                ),
                "persistent_site_multiplicity_per_system": float(
                    result["persistent_site_multiplicity_per_system"]
                ),
                "persistent_site_front_width_m": float(
                    result["persistent_site_front_width_m"]
                ),
                "persistent_site_source_area_m2": float(
                    result["persistent_site_source_area_m2"]
                ),
                "persistent_tip_radius_m": float(result["persistent_tip_radius_m"]),
                "persistent_source_inventory_active": False,
                "persistent_source_refresh_active": False,
                "explicit_recovery_active": False,
                "engine_native_cycle_predictor": True,
                **_cycle_block_audit_fields(controller, self, result),
            }
        )
        return result


__all__ = [
    "AuditedPersistentSiteCyclicTipEngine",
    "_add_persistent_fields",
    "_cycle_block_audit_fields",
]
