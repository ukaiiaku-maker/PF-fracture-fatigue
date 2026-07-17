"""Threshold-correlated crack-growth rewards for the stochastic hazard pilot.

The v10.1.7.2 pilot randomizes the integrated cleavage-hazard threshold but keeps
one fixed geometric reward per renewal.  Consequently all realizations visit the
same 5 micrometre geometry sequence.  This module adds an opt-in renewal-reward
pilot in which the threshold also sets the total crack advance of that event.

For an exponential threshold Xi with deterministic reference Xi_det = 1,

    L_event = L0 * clip(Xi, q_min, q_max) / E[clip(Xi, q_min, q_max)].

The normalization preserves the mean checkpoint length.  The same Xi controls
waiting time and event size, so under constant lambda the renewal-reward mean
velocity remains L0*lambda.  No noise is added to K, J, barriers, shielding,
back stress, source capacity, or any material parameter.
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from .stochastic_hazard_tip import StochasticHazardDiagnosticTipEngine


AVALANCHE_SCHEMA = "v10.1.7.3_threshold_correlated_event_length"


@dataclass
class AvalancheLengthConfig:
    mode: str = "fixed"
    minimum_factor: float = 0.2
    maximum_factor: float = 4.0
    geometry_subsegment_fraction: float = 0.1

    def validate(self) -> "AvalancheLengthConfig":
        self.mode = str(self.mode).strip().lower()
        if self.mode not in {"fixed", "threshold_scaled"}:
            raise ValueError("avalanche length mode must be fixed or threshold_scaled")
        self.minimum_factor = max(float(self.minimum_factor), 1.0e-12)
        self.maximum_factor = max(float(self.maximum_factor), self.minimum_factor)
        self.geometry_subsegment_fraction = float(self.geometry_subsegment_fraction)
        if not (0.0 < self.geometry_subsegment_fraction <= 1.0):
            raise ValueError("geometry_subsegment_fraction must lie in (0, 1]")
        return self


def clipped_exponential_mean(minimum_factor: float, maximum_factor: float) -> float:
    """Return E[clip(X, a, b)] for X~Exponential(1)."""
    a = max(float(minimum_factor), 0.0)
    b = max(float(maximum_factor), a)
    return max(a + math.exp(-a) - math.exp(-b), 1.0e-300)


def threshold_event_length_factor(
    threshold_action: float,
    mode: str = "threshold_scaled",
    minimum_factor: float = 0.2,
    maximum_factor: float = 4.0,
    deterministic_threshold: bool = False,
) -> float:
    """Map a stochastic threshold to a bounded mean-preserving length factor."""
    mode = str(mode).strip().lower()
    if mode == "fixed" or deterministic_threshold:
        return 1.0
    if mode != "threshold_scaled":
        raise ValueError("avalanche length mode must be fixed or threshold_scaled")
    a = max(float(minimum_factor), 1.0e-12)
    b = max(float(maximum_factor), a)
    clipped = min(max(float(threshold_action), a), b)
    return clipped / clipped_exponential_mean(a, b)


_PENDING_GEOMETRY_EVENTS: deque[dict[str, Any]] = deque()


def clear_pending_geometry_events() -> None:
    _PENDING_GEOMETRY_EVENTS.clear()


def pop_pending_geometry_event() -> dict[str, Any] | None:
    if not _PENDING_GEOMETRY_EVENTS:
        return None
    return _PENDING_GEOMETRY_EVENTS.popleft()


class StochasticAvalancheDiagnosticTipEngine(StochasticHazardDiagnosticTipEngine):
    """Stochastic first passage with a threshold-correlated crack-growth reward."""

    stochastic_avalanche_length_active = True
    _avalanche_config_default = AvalancheLengthConfig()

    @classmethod
    def configure_avalanche(
        cls,
        mode: str = "fixed",
        minimum_factor: float = 0.2,
        maximum_factor: float = 4.0,
        geometry_subsegment_fraction: float = 0.1,
    ) -> None:
        cls._avalanche_config_default = AvalancheLengthConfig(
            mode=mode,
            minimum_factor=minimum_factor,
            maximum_factor=maximum_factor,
            geometry_subsegment_fraction=geometry_subsegment_fraction,
        ).validate()
        clear_pending_geometry_events()

    @classmethod
    def audit_payload(cls) -> dict[str, Any]:
        payload = super().audit_payload()
        cfg = copy.deepcopy(cls._avalanche_config_default).validate()
        payload["stochastic_avalanche"] = {
            "schema": AVALANCHE_SCHEMA,
            **asdict(cfg),
            "length_reference": "deterministic_integrated_hazard_threshold_one",
            "mean_length_preserved": True,
            "noise_added_to_K": False,
            "noise_added_to_barriers": False,
            "geometry_subsegments_re_equilibrated": False,
        }
        return payload

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avalanche_cfg = copy.deepcopy(
            type(self)._avalanche_config_default
        ).validate()
        self.avalanche_base_checkpoint_m = max(float(self.f.da), 1.0e-30)
        self.avalanche_event_length_factor = 1.0
        self.avalanche_event_advance_m = self.avalanche_base_checkpoint_m
        self.avalanche_last_completed_advance_m = 0.0
        self.avalanche_last_completed_factor = 0.0
        self.avalanche_event_length_history: list[float] = []
        self._set_current_event_length()

    def clone_split(self, daughter_fraction=0.5):
        child = super().clone_split(daughter_fraction)
        child.avalanche_cfg = copy.deepcopy(self.avalanche_cfg)
        child.avalanche_base_checkpoint_m = float(self.avalanche_base_checkpoint_m)
        child.avalanche_event_length_factor = float(self.avalanche_event_length_factor)
        child.avalanche_event_advance_m = float(self.avalanche_event_advance_m)
        child.avalanche_last_completed_advance_m = float(
            self.avalanche_last_completed_advance_m
        )
        child.avalanche_last_completed_factor = float(
            self.avalanche_last_completed_factor
        )
        child.avalanche_event_length_history = list(
            self.avalanche_event_length_history
        )
        return child

    def _set_current_event_length(self) -> None:
        deterministic = self.hazard_cfg.mode == "deterministic"
        factor = threshold_event_length_factor(
            self.hazard_threshold_action,
            mode=self.avalanche_cfg.mode,
            minimum_factor=self.avalanche_cfg.minimum_factor,
            maximum_factor=self.avalanche_cfg.maximum_factor,
            deterministic_threshold=deterministic,
        )
        self.avalanche_event_length_factor = float(factor)
        self.avalanche_event_advance_m = (
            self.avalanche_base_checkpoint_m * self.avalanche_event_length_factor
        )

    def _integrate_coupled(self, *args, **kwargs) -> dict[str, Any]:
        """Use the current event reward as the checkpoint length for one event."""
        event_length = float(self.avalanche_event_advance_m)
        event_factor = float(self.avalanche_event_length_factor)
        base = float(self.f.da)
        self.f.da = event_length
        try:
            result = super()._integrate_coupled(*args, **kwargs)
        finally:
            self.f.da = base

        fired = bool(result.get("fired", False))
        completed_threshold = float(
            result.get("hazard_threshold_completed_action", 0.0)
        )
        if fired:
            self.avalanche_last_completed_advance_m = event_length
            self.avalanche_last_completed_factor = event_factor
            self.avalanche_event_length_history.append(event_length)
            _PENDING_GEOMETRY_EVENTS.append({
                "event_advance_m": event_length,
                "event_length_factor": event_factor,
                "threshold_action": completed_threshold,
                "hazard_seed": int(self.hazard_cfg.seed),
                "hazard_event_index": int(self.hazard_event_index - 1),
                "geometry_subsegment_fraction": float(
                    self.avalanche_cfg.geometry_subsegment_fraction
                ),
            })
            self._set_current_event_length()

        result.update({
            "avalanche_event_advance_m": event_length if fired else 0.0,
            "avalanche_event_length_factor": event_factor if fired else 0.0,
            "avalanche_current_event_advance_m": float(
                self.avalanche_event_advance_m
            ),
            "avalanche_current_event_length_factor": float(
                self.avalanche_event_length_factor
            ),
        })
        return result

    def _avalanche_diagnostics(self) -> dict[str, Any]:
        return {
            "stochastic_avalanche_schema": AVALANCHE_SCHEMA,
            "stochastic_avalanche_length_enabled": (
                self.avalanche_cfg.mode == "threshold_scaled"
            ),
            "avalanche_length_mode": self.avalanche_cfg.mode,
            "avalanche_base_checkpoint_m": float(
                self.avalanche_base_checkpoint_m
            ),
            "avalanche_minimum_factor": float(
                self.avalanche_cfg.minimum_factor
            ),
            "avalanche_maximum_factor": float(
                self.avalanche_cfg.maximum_factor
            ),
            "avalanche_geometry_subsegment_fraction": float(
                self.avalanche_cfg.geometry_subsegment_fraction
            ),
            "avalanche_current_event_advance_m": float(
                self.avalanche_event_advance_m
            ),
            "avalanche_current_event_length_factor": float(
                self.avalanche_event_length_factor
            ),
            "avalanche_last_completed_advance_m": float(
                self.avalanche_last_completed_advance_m
            ),
            "avalanche_last_completed_factor": float(
                self.avalanche_last_completed_factor
            ),
            "avalanche_event_length_history_count": len(
                self.avalanche_event_length_history
            ),
            "avalanche_mean_length_preserved": True,
            "avalanche_noise_added_to_K": False,
            "avalanche_noise_added_to_barriers": False,
        }

    def step(self, K, T, dt):
        result = super().step(K, T, dt)
        result["kinetic_checkpoint_progress_m"] = (
            float(self.B) * float(self.avalanche_event_advance_m)
        )
        diag = self._avalanche_diagnostics()
        result.update(diag)
        if type(self)._audit_records:
            type(self)._audit_records[-1].update(diag)
            for key in (
                "avalanche_event_advance_m",
                "avalanche_event_length_factor",
                "avalanche_current_event_advance_m",
                "avalanche_current_event_length_factor",
            ):
                type(self)._audit_records[-1][key] = float(
                    result.get(key, 0.0)
                )
        return result


__all__ = [
    "AVALANCHE_SCHEMA",
    "AvalancheLengthConfig",
    "StochasticAvalancheDiagnosticTipEngine",
    "clear_pending_geometry_events",
    "clipped_exponential_mean",
    "pop_pending_geometry_event",
    "threshold_event_length_factor",
]
