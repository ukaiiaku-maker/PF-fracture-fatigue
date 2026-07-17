"""Reproducible event-level stochastic cleavage thresholds for the v10.1.7 tip.

The deterministic kinetic moving-tip model is recovered exactly when the event
threshold is one.  In stochastic mode every completed cleavage renewal draws

    Xi = -log(U),   U ~ Uniform(0, 1),

so Xi is an exponential unit-mean integrated-hazard threshold.  The normalized
checkpoint progress B evolves at lambda_c / Xi.  Because each renewal still
advances the fixed checkpoint length, its waiting time is Xi/lambda_c while the
long-time mean event rate remains lambda_c.  No noise is added to K, barriers,
source capacity, shielding, or any reported observable.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .developed_state_diagnostic_tip import DevelopedStateDiagnosticTipEngine


HAZARD_SCHEMA = "v10.1.7.2_stochastic_cleavage_threshold"


@dataclass
class HazardThresholdConfig:
    mode: str = "deterministic"
    seed: int = 0
    minimum_threshold: float = 1.0e-12

    def validate(self) -> "HazardThresholdConfig":
        self.mode = str(self.mode).strip().lower()
        if self.mode not in {"deterministic", "exponential"}:
            raise ValueError("hazard threshold mode must be deterministic or exponential")
        self.seed = int(self.seed)
        self.minimum_threshold = max(float(self.minimum_threshold), 1.0e-300)
        return self


def draw_hazard_threshold(
    mode: str,
    rng: np.random.Generator,
    minimum_threshold: float = 1.0e-12,
) -> float:
    """Draw one positive integrated-hazard threshold."""
    mode = str(mode).strip().lower()
    if mode == "deterministic":
        return 1.0
    if mode != "exponential":
        raise ValueError("hazard threshold mode must be deterministic or exponential")
    return max(float(rng.exponential(1.0)), max(float(minimum_threshold), 1.0e-300))


def normalized_progress_rate(lambda_c_s: float, threshold_action: float) -> float:
    """Convert the physical cleavage rate to normalized checkpoint progress."""
    return max(float(lambda_c_s), 0.0) / max(float(threshold_action), 1.0e-300)


class StochasticHazardDiagnosticTipEngine(DevelopedStateDiagnosticTipEngine):
    """Campaign-calibrated developed-state engine with optional random thresholds."""

    stochastic_hazard_threshold_active = True
    _hazard_config_default = HazardThresholdConfig()

    @classmethod
    def configure_hazard(
        cls,
        mode: str = "deterministic",
        seed: int = 0,
        minimum_threshold: float = 1.0e-12,
    ) -> None:
        cls._hazard_config_default = HazardThresholdConfig(
            mode=mode,
            seed=seed,
            minimum_threshold=minimum_threshold,
        ).validate()

    @classmethod
    def audit_payload(cls) -> dict[str, Any]:
        payload = super().audit_payload()
        payload["stochastic_hazard"] = {
            "schema": HAZARD_SCHEMA,
            **asdict(cls._hazard_config_default),
            "distribution": (
                "delta_at_one"
                if cls._hazard_config_default.mode == "deterministic"
                else "exponential_unit_mean"
            ),
            "noise_added_to_K": False,
            "noise_added_to_barriers": False,
        }
        return payload

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hazard_cfg = copy.deepcopy(type(self)._hazard_config_default).validate()
        seed_sequence = np.random.SeedSequence(
            [int(self.hazard_cfg.seed), int(getattr(self, "_engine_id", 0))]
        )
        self._hazard_rng = np.random.default_rng(seed_sequence)
        self.hazard_threshold_action = self._draw_threshold()
        self.hazard_action_current = 0.0
        self.hazard_event_index = 0
        self.hazard_last_completed_threshold = 0.0
        self.hazard_last_completed_action = 0.0
        self.hazard_last_progress_rate_s = 0.0
        self.hazard_threshold_history: list[float] = []

    def _draw_threshold(self) -> float:
        return draw_hazard_threshold(
            self.hazard_cfg.mode,
            self._hazard_rng,
            self.hazard_cfg.minimum_threshold,
        )

    def clone_split(self, daughter_fraction=0.5):
        child = super().clone_split(daughter_fraction)
        child.hazard_cfg = copy.deepcopy(self.hazard_cfg)
        child_seed = int(self._hazard_rng.integers(0, np.iinfo(np.uint32).max))
        child._hazard_rng = np.random.default_rng(
            np.random.SeedSequence([child_seed, int(getattr(child, "_engine_id", 0))])
        )
        child.hazard_threshold_action = float(self.hazard_threshold_action)
        child.hazard_action_current = float(self.hazard_action_current)
        child.hazard_event_index = int(self.hazard_event_index)
        child.hazard_last_completed_threshold = float(self.hazard_last_completed_threshold)
        child.hazard_last_completed_action = float(self.hazard_last_completed_action)
        child.hazard_last_progress_rate_s = float(self.hazard_last_progress_rate_s)
        child.hazard_threshold_history = list(self.hazard_threshold_history)
        return child

    def predict_clock_increment(self, K, T, dt):
        physical = super().predict_clock_increment(K, T, dt)
        return float(physical) / max(float(self.hazard_threshold_action), 1.0e-300)

    def _integrate_coupled(
        self,
        K: float,
        T: float,
        dt: float,
        stress_override: float | None = None,
        lambda_override: float | None = None,
    ) -> dict[str, Any]:
        """Port of the validated Strang integrator with threshold-normalized B."""
        dt_requested = max(float(dt), 0.0)
        remaining = dt_requested
        consumed = 0.0
        dB_total = 0.0
        dH_total = 0.0
        da_total = 0.0
        packet_mean = 0.0
        packet_variance = 0.0
        totals: dict[str, float] = {}
        wake_totals: dict[str, float] = {}
        fired = False
        microsteps = 0
        last_lam = 0.0
        last_raw = 0.0
        last_Gc = 0.0
        last_sig = (
            self.sigma_tip(K)
            if stress_override is None
            else max(float(stress_override), 0.0)
        )
        completed_threshold = 0.0
        completed_action = 0.0

        while remaining > 0.0:
            microsteps += 1
            if microsteps > self.tip_cfg.max_internal_steps:
                raise RuntimeError(
                    "stochastic kinetic tip-cell exceeded max_internal_steps; reduce "
                    "the outer time increment or increase the internal-step limit"
                )

            threshold = max(float(self.hazard_threshold_action), 1.0e-300)
            sig0 = (
                self.sigma_tip(K)
                if stress_override is None
                else max(float(stress_override), 0.0)
            )
            lam0, raw0, Gc0 = self.lambda_cleave(sig0, T)
            if lambda_override is not None:
                lam0 = max(float(lambda_override), 0.0)
                raw0 = lam0
            lam0 = max(float(lam0), 0.0) if math.isfinite(lam0) else 0.0
            progress0 = normalized_progress_rate(lam0, threshold)
            h = self._substep_limit(remaining, progress0)

            mpz_before = self.mpz.copy()
            W_before = float(self.W_emit)
            first = self._plastic_half_step(0.5 * h, T, sig0)
            sig_mid = (
                self.sigma_tip(K)
                if stress_override is None
                else max(float(stress_override), 0.0)
            )
            lam_mid, raw_mid, Gc_mid = self.lambda_cleave(sig_mid, T)
            if lambda_override is not None:
                lam_mid = max(float(lambda_override), 0.0)
                raw_mid = lam_mid
            lam_mid = max(float(lam_mid), 0.0) if math.isfinite(lam_mid) else 0.0
            progress_mid = normalized_progress_rate(lam_mid, threshold)
            remaining_progress = max(1.0 - float(self.B), 0.0)

            if (
                progress_mid > 0.0
                and progress_mid * h > remaining_progress + 1.0e-12
            ):
                self.mpz = mpz_before
                self.W_emit = W_before
                h = max(
                    remaining_progress / progress_mid,
                    self.tip_cfg.min_substep_s,
                )
                h = min(h, remaining)
                first = self._plastic_half_step(0.5 * h, T, sig0)
                sig_mid = (
                    self.sigma_tip(K)
                    if stress_override is None
                    else max(float(stress_override), 0.0)
                )
                lam_mid, raw_mid, Gc_mid = self.lambda_cleave(sig_mid, T)
                if lambda_override is not None:
                    lam_mid = max(float(lambda_override), 0.0)
                    raw_mid = lam_mid
                lam_mid = (
                    max(float(lam_mid), 0.0)
                    if math.isfinite(lam_mid)
                    else 0.0
                )
                progress_mid = normalized_progress_rate(lam_mid, threshold)

            dB = min(progress_mid * h, max(1.0 - float(self.B), 0.0))
            dH = dB * threshold
            da = float(self.f.da) * self.tip_cfg.velocity_scale * dB
            packet_rate = (
                float(self.f.da)
                * self.tip_cfg.velocity_scale
                / self.tip_cfg.packet_length_m
                * progress_mid
            )
            packet_n = packet_rate * h
            packet_var = self.tip_cfg.packet_length_m ** 2 * packet_n

            advance = self.mpz.advance(da) if da > 0.0 else {}
            sig1 = (
                self.sigma_tip(K)
                if stress_override is None
                else max(float(stress_override), 0.0)
            )
            second = self._plastic_half_step(0.5 * h, T, sig1)

            self._sum_numeric(totals, first)
            self._sum_numeric(totals, second)
            self._sum_numeric(wake_totals, advance)
            self.B += dB
            self.hazard_action_current += dH
            self.micro_advance_total_m += da
            self.packet_count_mean_total += packet_n
            self.packet_variance_total_m2 += packet_var
            dB_total += dB
            dH_total += dH
            da_total += da
            packet_mean += packet_n
            packet_variance += packet_var
            consumed += h
            remaining = max(remaining - h, 0.0)
            self.t += h
            last_lam, last_raw, last_Gc, last_sig = (
                lam_mid,
                raw_mid,
                Gc_mid,
                sig1,
            )
            self.hazard_last_progress_rate_s = progress_mid

            if self.B >= 1.0 - 1.0e-10:
                self.B = max(self.B - 1.0, 0.0)
                self.a_adv += float(self.f.da)
                self.checkpoint_advance_total_m += float(self.f.da)
                self.n_adv += 1
                fired = True
                completed_threshold = threshold
                completed_action = float(self.hazard_action_current)
                self.hazard_last_completed_threshold = completed_threshold
                self.hazard_last_completed_action = completed_action
                self.hazard_threshold_history.append(completed_threshold)
                self.hazard_event_index += 1
                self.hazard_action_current = 0.0
                self.hazard_threshold_action = self._draw_threshold()
                break

            if h <= 0.0:
                break

        return {
            "fired": fired,
            "n_fire": 1 if fired else 0,
            "v_crack": da_total / consumed if consumed > 0.0 else 0.0,
            "dB": dB_total,
            "physical_hazard_action_step": dH_total,
            "da": da_total,
            "dt_consumed": consumed,
            "dt_unused": max(dt_requested - consumed, 0.0),
            "packet_mean": packet_mean,
            "packet_variance_m2": packet_variance,
            "lambda_c": last_lam,
            "lambda_c_raw": last_raw,
            "hazard_progress_rate_s-1": self.hazard_last_progress_rate_s,
            "hazard_threshold_completed_action": completed_threshold,
            "hazard_action_completed": completed_action,
            "hazard_threshold_next_action": float(self.hazard_threshold_action),
            "Gc_J": last_Gc,
            "sigma_tip": last_sig,
            "plastic": totals,
            "advance": wake_totals,
            "microsteps": microsteps,
        }

    def _hazard_diagnostics(self) -> dict[str, Any]:
        return {
            "stochastic_hazard_schema": HAZARD_SCHEMA,
            "stochastic_hazard_enabled": self.hazard_cfg.mode == "exponential",
            "hazard_threshold_mode": self.hazard_cfg.mode,
            "hazard_seed": int(self.hazard_cfg.seed),
            "hazard_event_index": int(self.hazard_event_index),
            "hazard_threshold_current_action": float(self.hazard_threshold_action),
            "hazard_action_current": float(self.hazard_action_current),
            "hazard_last_completed_threshold": float(
                self.hazard_last_completed_threshold
            ),
            "hazard_last_completed_action": float(self.hazard_last_completed_action),
            "hazard_last_progress_rate_s-1": float(
                self.hazard_last_progress_rate_s
            ),
            "hazard_threshold_history_count": len(self.hazard_threshold_history),
            "hazard_noise_added_to_K": False,
            "hazard_noise_added_to_barriers": False,
        }

    def step(self, K, T, dt):
        result = super().step(K, T, dt)
        result.update(self._hazard_diagnostics())
        if type(self)._audit_records:
            type(self)._audit_records[-1].update(self._hazard_diagnostics())
            for key in (
                "physical_hazard_action_step",
                "hazard_progress_rate_s-1",
                "hazard_threshold_completed_action",
                "hazard_action_completed",
                "hazard_threshold_next_action",
            ):
                type(self)._audit_records[-1][key] = float(result.get(key, 0.0))
        return result


__all__ = [
    "HAZARD_SCHEMA",
    "HazardThresholdConfig",
    "StochasticHazardDiagnosticTipEngine",
    "draw_hazard_threshold",
    "normalized_progress_rate",
]
