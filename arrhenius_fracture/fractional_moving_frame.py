"""Conservative fractional moving-frame translation for the v10.1 tip cell."""
from __future__ import annotations

import math
import numpy as np


def _translate_toward_tip(
    field: np.ndarray,
    distance_m: float,
    dx: float,
    wake_bins: int,
    wake_dx: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Shift active finite-volume mass toward xi=0 with fractional-bin weights."""
    source = np.asarray(field, dtype=float)
    active = np.zeros_like(source)
    wake = np.zeros((source.shape[0], wake_bins), dtype=float)
    shift = max(float(distance_m), 0.0) / max(float(dx), 1.0e-30)

    for i in range(source.shape[1]):
        dest = float(i) - shift
        j0 = math.floor(dest)
        frac = dest - j0
        for j, weight in ((j0, 1.0 - frac), (j0 + 1, frac)):
            if weight <= 0.0:
                continue
            mass = source[:, i] * weight
            if j >= 0:
                if j < source.shape[1]:
                    active[:, j] += mass
            else:
                # Negative active indices are mapped by their distance behind
                # the moving tip.  For the production substep (<=0.1 um on a
                # 0.5 um grid), only the near-wake bin is normally populated.
                y = (-float(j) - 0.5) * dx
                k = max(int(y / max(wake_dx, 1.0e-30)), 0)
                if k < wake_bins:
                    wake[:, k] += mass

    total = float(np.sum(source))
    discarded = max(total - float(np.sum(active)) - float(np.sum(wake)), 0.0)
    return active, wake, discarded


def fractional_moving_frame_advance(self, distance_m: float) -> dict[str, float]:
    """Drop-in replacement for ``UnifiedMPZState.advance``.

    Existing wake fields move away from the new tip with fractional finite-volume
    advection.  Active fields move toward the tip; only the crossed fraction is
    deposited in the wake.  Source capacity is exposed continuously with swept
    distance.  The method preserves the original return schema.
    """
    d = max(float(distance_m), 0.0)
    old_wm, lost_old_m = self._advect_forward(self.wake_mobile, d, self.wake_dx)
    old_wr, lost_old_r = self._advect_forward(self.wake_retained, d, self.wake_dx)
    old_ws, lost_old_s = self._advect_forward(self.wake_slip, d, self.wake_dx)

    self.mobile, crossed_m, lost_m = _translate_toward_tip(
        self.mobile, d, self.dx, self.wake_n_bins, self.wake_dx
    )
    self.retained, crossed_r, lost_r = _translate_toward_tip(
        self.retained, d, self.dx, self.wake_n_bins, self.wake_dx
    )
    self.accumulated_slip, crossed_s, lost_s = _translate_toward_tip(
        self.accumulated_slip, d, self.dx, self.wake_n_bins, self.wake_dx
    )

    self.wake_mobile = old_wm + crossed_m
    self.wake_retained = old_wr + crossed_r
    self.wake_slip = old_ws + crossed_s
    self.wake_discarded_mobile_total += lost_old_m + lost_m
    self.wake_discarded_retained_total += lost_old_r + lost_r
    self.wake_discarded_slip_total += lost_old_s + lost_s

    fresh = min(d / max(self.manifest.source_refresh_length_m, self.dx), 1.0)
    refreshed = (self.site_capacity - self.available_sites) * fresh
    self.available_sites += refreshed
    self.advance_total_m += d

    return {
        "wake_mobile": float(np.sum(crossed_m)),
        "wake_retained": float(np.sum(crossed_r)),
        "wake_slip": float(np.sum(crossed_s)),
        "source_sites_refreshed": float(np.sum(refreshed)),
        "active_mobile_postcommit": self.mobile_count,
        "active_retained_postcommit": self.retained_count,
        "wake_mobile_postcommit": self.wake_mobile_count,
        "wake_retained_postcommit": self.wake_retained_count,
        "fractional_moving_frame": 1.0,
    }
