"""Bounded tip-geometry feedback for the campaign-calibrated source budget.

The promoted campaign source multiplicity and all first-passage kinetics are
preserved exactly. Feedback is armed only after the first crack advance, while
the geometry reference remains the original unblunted radius ``r0``. Subsequent
source exposure follows the irreversible running maximum of normalized
blunting. Newly exposed sites remain subject to the original Arrhenius emission
law and Taylor back stress.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from .developed_state_diagnostic_tip import DevelopedStateDiagnosticTipEngine


GEOMETRY_SCHEMA = "v10.1.9.1_geometry_source_feedback"
SOURCE_MODEL = "campaign_tip_budget_with_geometry_capacity_gain"


class GeometrySourceFeedbackTipEngine(DevelopedStateDiagnosticTipEngine):
    """Campaign tip engine with one bounded post-initiation geometry gain."""

    geometry_source_feedback_active = True
    _geometry_source_gain_default = 0.0

    @classmethod
    def configure_geometry_source_feedback(cls, gain: float = 0.0) -> None:
        cls._geometry_source_gain_default = max(float(gain), 0.0)

    @classmethod
    def audit_payload(cls) -> dict[str, Any]:
        payload = super().audit_payload()
        payload["geometry_source_feedback"] = {
            "schema": GEOMETRY_SCHEMA,
            "gain": cls._geometry_source_gain_default,
            "reference_geometry": "original unblunted tip radius r0",
            "feedback_armed_only_after_first_advance": True,
            "first_passage_feedback_disabled": True,
            "capacity_growth_irreversible": True,
            "running_maximum_blunting": True,
            "maximum_capacity_ratio": 1.0 + cls._geometry_source_gain_default,
            "temperature_dependent_parameter": False,
        }
        return payload

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry_source_gain = max(
            float(type(self)._geometry_source_gain_default), 0.0
        )
        self.geometry_source_base_capacity = np.asarray(
            self.mpz.site_capacity, dtype=float
        ).copy()
        self.geometry_source_reference_radius_m = max(float(self.f.r0), 1.0e-30)
        self.geometry_source_first_advance_radius_m = math.nan
        self.geometry_source_feedback_armed = False
        self.geometry_source_capacity_ratio = 1.0
        self.geometry_source_cumulative_exposed = 0.0
        self.geometry_source_last_exposed = 0.0
        self.geometry_source_current_normalized_blunting = 0.0
        self.geometry_source_running_max_normalized_blunting = 0.0
        self.geometry_source_last_normalized_blunting = 0.0

    def clone_split(self, daughter_fraction=0.5):
        frac = float(np.clip(daughter_fraction, 0.0, 1.0))
        base_before = np.asarray(
            self.geometry_source_base_capacity, dtype=float
        ).copy()
        child = super().clone_split(frac)
        self.geometry_source_base_capacity = base_before * (1.0 - frac)
        child.geometry_source_base_capacity = base_before * frac
        for name in (
            "geometry_source_gain",
            "geometry_source_reference_radius_m",
            "geometry_source_first_advance_radius_m",
            "geometry_source_feedback_armed",
            "geometry_source_capacity_ratio",
            "geometry_source_cumulative_exposed",
            "geometry_source_last_exposed",
            "geometry_source_current_normalized_blunting",
            "geometry_source_running_max_normalized_blunting",
            "geometry_source_last_normalized_blunting",
        ):
            setattr(child, name, copy.deepcopy(getattr(self, name)))
        return child

    def _current_normalized_blunting(self, radius_m: float) -> float:
        ref = max(float(self.geometry_source_reference_radius_m), 1.0e-30)
        return max(float(radius_m) - ref, 0.0) / ref

    def _target_capacity_ratio(self, radius_m: float) -> tuple[float, float]:
        current = self._current_normalized_blunting(radius_m)
        self.geometry_source_current_normalized_blunting = current
        if not self.geometry_source_feedback_armed or self.geometry_source_gain <= 0.0:
            return 1.0, current
        self.geometry_source_running_max_normalized_blunting = max(
            float(self.geometry_source_running_max_normalized_blunting), current
        )
        running = self.geometry_source_running_max_normalized_blunting
        saturation = running / (1.0 + running)
        return 1.0 + self.geometry_source_gain * saturation, current

    def _apply_geometry_capacity_gain(self) -> float:
        self.geometry_source_last_exposed = 0.0
        ratio_target, current = self._target_capacity_ratio(self.r_eff())
        self.geometry_source_last_normalized_blunting = (
            self.geometry_source_running_max_normalized_blunting
            if self.geometry_source_feedback_armed else current
        )
        if not self.geometry_source_feedback_armed:
            return 0.0

        ratio = max(float(self.geometry_source_capacity_ratio), ratio_target)
        ratio = min(ratio, 1.0 + self.geometry_source_gain)
        target = np.maximum(self.geometry_source_base_capacity * ratio, 0.0)
        old_capacity = np.maximum(
            np.asarray(self.mpz.site_capacity, dtype=float), 0.0
        )
        increase = np.maximum(target - old_capacity, 0.0)
        exposed = float(np.sum(increase))
        if exposed > 0.0:
            self.mpz.site_capacity = old_capacity + increase
            self.mpz.available_sites = np.minimum(
                np.maximum(
                    np.asarray(self.mpz.available_sites, dtype=float) + increase,
                    0.0,
                ),
                self.mpz.site_capacity,
            )
            self.mpz.tip_source_activity = np.divide(
                self.mpz.available_sites,
                self.mpz.site_capacity,
                out=np.zeros_like(self.mpz.available_sites),
                where=self.mpz.site_capacity > 0.0,
            )
            self.mpz.campaign_source_budget_remaining_total = float(
                np.sum(self.mpz.available_sites)
            )
            self.mpz.campaign_source_budget_consumed_total = float(
                np.sum(self.mpz.site_capacity - self.mpz.available_sites)
            )
            self.geometry_source_cumulative_exposed += exposed
            self.geometry_source_last_exposed = exposed
        self.geometry_source_capacity_ratio = ratio
        return exposed

    def _arm_after_first_advance(self, result: dict[str, Any]) -> bool:
        if self.geometry_source_feedback_armed or not bool(result.get("fired", False)):
            return False
        self.geometry_source_feedback_armed = True
        self.geometry_source_first_advance_radius_m = max(
            float(result.get("r_eff", self.r_eff())), 1.0e-30
        )
        self.geometry_source_current_normalized_blunting = (
            self._current_normalized_blunting(self.geometry_source_first_advance_radius_m)
        )
        # Do not expose sites in the first fired record. The already-developed
        # blunting is evaluated at the start of the next kinetic interval.
        self.geometry_source_capacity_ratio = 1.0
        self.geometry_source_last_exposed = 0.0
        self.geometry_source_last_normalized_blunting = 0.0
        return True

    def _geometry_source_diagnostics(self) -> dict[str, Any]:
        current = max(float(self.r_eff()), 0.0)
        return {
            "geometry_source_schema": GEOMETRY_SCHEMA,
            "geometry_source_model": SOURCE_MODEL,
            "geometry_source_gain": self.geometry_source_gain,
            "geometry_source_feedback_armed": self.geometry_source_feedback_armed,
            "geometry_source_reference_established": True,
            "geometry_source_reference_radius_m": self.geometry_source_reference_radius_m,
            "geometry_source_r0_m": self.geometry_source_reference_radius_m,
            "geometry_source_first_advance_radius_m": self.geometry_source_first_advance_radius_m,
            "geometry_source_current_radius_m": current,
            "geometry_source_current_radius_over_r0": current / self.geometry_source_reference_radius_m,
            "geometry_source_current_normalized_blunting": self.geometry_source_current_normalized_blunting,
            "geometry_source_running_max_normalized_blunting": self.geometry_source_running_max_normalized_blunting,
            "geometry_source_normalized_blunting": self.geometry_source_last_normalized_blunting,
            "geometry_source_capacity_ratio": self.geometry_source_capacity_ratio,
            "geometry_source_capacity_total": float(np.sum(self.mpz.site_capacity)),
            "geometry_source_available_total": float(np.sum(self.mpz.available_sites)),
            "geometry_source_last_exposed": self.geometry_source_last_exposed,
            "geometry_source_cumulative_exposed": self.geometry_source_cumulative_exposed,
            "geometry_source_first_passage_feedback_disabled": True,
            "geometry_source_maximum_capacity_ratio": 1.0 + self.geometry_source_gain,
        }

    def step(self, K, T, dt):
        was_armed = self.geometry_source_feedback_armed
        if was_armed:
            self._apply_geometry_capacity_gain()
        result = super().step(K, T, dt)

        newly_armed = self._arm_after_first_advance(result)
        if was_armed and not newly_armed:
            self._apply_geometry_capacity_gain()

        campaign = self._campaign_diagnostics()
        geometry = self._geometry_source_diagnostics()
        result.update(campaign)
        result.update(geometry)
        if type(self)._audit_records:
            type(self)._audit_records[-1].update(campaign)
            type(self)._audit_records[-1].update(geometry)
        return result

    def cycle_step_waveform(self, *args, **kwargs):
        was_armed = self.geometry_source_feedback_armed
        if was_armed:
            self._apply_geometry_capacity_gain()
        result = super().cycle_step_waveform(*args, **kwargs)
        newly_armed = self._arm_after_first_advance(result)
        if was_armed and not newly_armed:
            self._apply_geometry_capacity_gain()
        result.update(self._campaign_diagnostics())
        result.update(self._geometry_source_diagnostics())
        return result
