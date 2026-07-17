"""Developed-state diagnostics for the campaign-calibrated moving tip.

This subclass changes no constitutive rates or state updates.  It only integrates
source, transport, storage, recovery, and population-history diagnostics so a
transient first-emission burst can be distinguished from a sustained moving-tip
plastic zone.
"""
from __future__ import annotations

import copy
from typing import Any

from .campaign_calibrated_tip import CampaignCalibratedTipEngine


DIAGNOSTIC_SCHEMA = "v10.1.7_dbtt_developed_state"


class DevelopedStateDiagnosticTipEngine(CampaignCalibratedTipEngine):
    """Campaign engine with cumulative and residence-time diagnostics only."""

    developed_state_diagnostics_active = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostic_cumulative_emitted = 0.0
        self.diagnostic_cumulative_refreshed = 0.0
        self.diagnostic_cumulative_trapped = 0.0
        self.diagnostic_cumulative_released = 0.0
        self.diagnostic_cumulative_recovered = 0.0
        self.diagnostic_cumulative_escaped = 0.0
        self.diagnostic_mobile_residence_count_s = 0.0
        self.diagnostic_retained_residence_count_s = 0.0
        self.diagnostic_active_residence_count_s = 0.0

    def clone_split(self, daughter_fraction=0.5):
        child = super().clone_split(daughter_fraction)
        for name in (
            "diagnostic_cumulative_emitted",
            "diagnostic_cumulative_refreshed",
            "diagnostic_cumulative_trapped",
            "diagnostic_cumulative_released",
            "diagnostic_cumulative_recovered",
            "diagnostic_cumulative_escaped",
            "diagnostic_mobile_residence_count_s",
            "diagnostic_retained_residence_count_s",
            "diagnostic_active_residence_count_s",
        ):
            setattr(child, name, copy.deepcopy(getattr(self, name)))
        return child

    @staticmethod
    def _positive(result: dict[str, Any], *names: str) -> float:
        for name in names:
            if name in result:
                return max(float(result.get(name, 0.0)), 0.0)
        return 0.0

    def _developed_state_diagnostics(self) -> dict[str, Any]:
        mobile = max(float(self.mpz.mobile_count), 0.0)
        retained = max(float(self.mpz.retained_count), 0.0)
        active = mobile + retained
        return {
            "developed_state_schema": DIAGNOSTIC_SCHEMA,
            "developed_state_mobile_count": mobile,
            "developed_state_retained_count": retained,
            "developed_state_active_count": active,
            "developed_state_retained_fraction": retained / active if active > 0.0 else 0.0,
            "developed_state_cumulative_emitted": self.diagnostic_cumulative_emitted,
            "developed_state_cumulative_refreshed": self.diagnostic_cumulative_refreshed,
            "developed_state_cumulative_trapped": self.diagnostic_cumulative_trapped,
            "developed_state_cumulative_released": self.diagnostic_cumulative_released,
            "developed_state_cumulative_recovered": self.diagnostic_cumulative_recovered,
            "developed_state_cumulative_escaped": self.diagnostic_cumulative_escaped,
            "developed_state_mobile_residence_count_s": self.diagnostic_mobile_residence_count_s,
            "developed_state_retained_residence_count_s": self.diagnostic_retained_residence_count_s,
            "developed_state_active_residence_count_s": self.diagnostic_active_residence_count_s,
        }

    def step(self, K, T, dt):
        result = super().step(K, T, dt)
        dt_used = max(float(result.get("kinetic_dt_consumed_s", dt)), 0.0)
        self.diagnostic_cumulative_emitted += self._positive(result, "dN_emit_raw", "dN_emit")
        self.diagnostic_cumulative_refreshed += self._positive(result, "source_sites_refreshed")
        self.diagnostic_cumulative_trapped += self._positive(result, "dN_trapped")
        self.diagnostic_cumulative_released += self._positive(result, "dN_released")
        self.diagnostic_cumulative_recovered += self._positive(result, "dN_recovered")
        self.diagnostic_cumulative_escaped += self._positive(result, "dN_escaped")

        mobile = max(float(self.mpz.mobile_count), 0.0)
        retained = max(float(self.mpz.retained_count), 0.0)
        self.diagnostic_mobile_residence_count_s += mobile * dt_used
        self.diagnostic_retained_residence_count_s += retained * dt_used
        self.diagnostic_active_residence_count_s += (mobile + retained) * dt_used

        diag = self._developed_state_diagnostics()
        result.update(diag)
        if type(self)._audit_records:
            type(self)._audit_records[-1].update(diag)
            for key in (
                "dN_emit_raw",
                "dN_trapped",
                "dN_released",
                "dN_recovered",
                "dN_escaped",
                "source_sites_refreshed",
                "kinetic_dt_consumed_s",
            ):
                type(self)._audit_records[-1][key] = float(result.get(key, 0.0))
        return result

    def cycle_step_waveform(self, *args, **kwargs):
        result = super().cycle_step_waveform(*args, **kwargs)
        result.update(self._developed_state_diagnostics())
        return result
