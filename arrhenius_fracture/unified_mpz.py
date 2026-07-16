"""Unified finite-source, Peierls--Taylor, active/wake MPZ state.

This state is independent of crack geometry.  Each sharp front owns one active
process zone and one persistent signed wake.  The same object is used by
monotonic and cyclic loading.
"""
from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import gammainc

from .material_manifest import MaterialManifest


@dataclass
class MPZConfig:
    length_m: float = 100.0e-6
    n_bins: int = 200
    n_systems: int = 2
    source_bin_count: int = 2
    shielding_orientation_factors: tuple[float, ...] = (1.0, 1.0)
    mobile_shield_fraction: float = 0.0
    shielding_core_m: float = 2.5e-10
    blunting_length_m: float = 0.5e-6
    forest_density_floor_m2: float = 5.0e12
    jump_fraction: float = 1.0
    peierls_stress_fraction: float = 1.0 / math.sqrt(3.0)
    taylor_stress_fraction: float = 1.0 / math.sqrt(3.0)
    mobile_recovery_rate_s: float = 0.0
    pair_annihilation_rate_per_count_s: float = 0.0
    wake_length_m: float = 100.0e-6
    wake_n_bins: int = 0
    wake_shielding: bool = True
    wake_shield_projection: float = 1.0


class UnifiedMPZState:
    state_model = "sharp_front_v10_unified_active_persistent_wake"

    def __init__(self, manifest: MaterialManifest, cfg: MPZConfig):
        self.manifest = manifest
        self.cfg = copy.deepcopy(cfg)
        self.n_systems = max(int(cfg.n_systems), 1)
        self.n_bins = max(int(cfg.n_bins), 4)
        self.length_m = max(float(cfg.length_m), 1.0e-12)
        self.dx = self.length_m / self.n_bins
        self.x = (np.arange(self.n_bins, dtype=float) + 0.5) * self.dx
        factors = np.asarray(cfg.shielding_orientation_factors, dtype=float).reshape(-1)
        if factors.size < self.n_systems:
            factors = np.pad(factors, (0, self.n_systems - factors.size), mode="edge")
        self.orientation_factors = factors[: self.n_systems].copy()

        cap = max(float(manifest.source_sites_per_system), 0.0)
        self.site_capacity = np.full(self.n_systems, cap, dtype=float)
        self.available_sites = self.site_capacity.copy()
        shape = (self.n_systems, self.n_bins)
        self.mobile = np.zeros(shape, dtype=float)
        self.retained = np.zeros(shape, dtype=float)
        self.accumulated_slip = np.zeros(shape, dtype=float)

        self.wake_length_m = max(float(cfg.wake_length_m), self.dx)
        self.wake_n_bins = int(cfg.wake_n_bins) if int(cfg.wake_n_bins) > 0 else max(int(round(self.wake_length_m / self.dx)), 1)
        self.wake_dx = self.wake_length_m / self.wake_n_bins
        self.wake_x = (np.arange(self.wake_n_bins, dtype=float) + 0.5) * self.wake_dx
        wshape = (self.n_systems, self.wake_n_bins)
        self.wake_mobile = np.zeros(wshape, dtype=float)
        self.wake_retained = np.zeros(wshape, dtype=float)
        self.wake_slip = np.zeros(wshape, dtype=float)

        self.emitted_total = 0.0
        self.escaped_total = 0.0
        self.recovered_total = 0.0
        self.advance_total_m = 0.0
        self.wake_discarded_mobile_total = 0.0
        self.wake_discarded_retained_total = 0.0
        self.wake_discarded_slip_total = 0.0
        self.time_s = 0.0

    def copy(self) -> "UnifiedMPZState":
        return copy.deepcopy(self)

    @property
    def mobile_count(self) -> float:
        return float(np.sum(self.mobile))

    @property
    def retained_count(self) -> float:
        return float(np.sum(self.retained))

    @property
    def wake_mobile_count(self) -> float:
        return float(np.sum(self.wake_mobile))

    @property
    def wake_retained_count(self) -> float:
        return float(np.sum(self.wake_retained))

    @property
    def total_count(self) -> float:
        return self.mobile_count + self.retained_count + self.wake_mobile_count + self.wake_retained_count

    @property
    def available_site_fraction(self) -> float:
        denom = float(np.sum(self.site_capacity))
        return float(np.sum(self.available_sites) / denom) if denom > 0.0 else 0.0

    def split(self, daughter_fraction: float) -> "UnifiedMPZState":
        """Conservative branch split; no source or wake state is duplicated."""
        frac = float(np.clip(daughter_fraction, 0.0, 1.0))
        child = self.copy()
        for name in (
            "site_capacity", "available_sites", "mobile", "retained",
            "accumulated_slip", "wake_mobile", "wake_retained", "wake_slip",
        ):
            original = np.asarray(getattr(self, name), dtype=float).copy()
            setattr(child, name, original * frac)
            setattr(self, name, original * (1.0 - frac))
        for name in (
            "emitted_total", "escaped_total", "recovered_total",
            "wake_discarded_mobile_total", "wake_discarded_retained_total",
            "wake_discarded_slip_total",
        ):
            value = float(getattr(self, name))
            setattr(child, name, value * frac)
            setattr(self, name, value * (1.0 - frac))
        child.advance_total_m = 0.0
        return child

    def _shielding_raw(self, retained: np.ndarray, mobile: np.ndarray, x: np.ndarray, G: float, nu: float, b: float) -> float:
        core = max(float(self.cfg.shielding_core_m), 0.25 * abs(float(b)), 1.0e-12)
        kernel = float(G) * float(b) / max(1.0 - float(nu), 1.0e-6) / np.sqrt(2.0 * np.pi * np.maximum(x, core))
        signed = retained + float(self.cfg.mobile_shield_fraction) * mobile
        return float(np.sum(self.orientation_factors[:, None] * signed * kernel[None, :]))

    def active_K_shielding(self, G: float, nu: float, b: float) -> float:
        return max(self._shielding_raw(self.retained, self.mobile, self.x, G, nu, b), 0.0)

    def wake_K_shielding(self, G: float, nu: float, b: float) -> float:
        if not bool(self.cfg.wake_shielding):
            return 0.0
        return max(float(self.cfg.wake_shield_projection) * self._shielding_raw(
            self.wake_retained, self.wake_mobile, self.wake_x, G, nu, b
        ), 0.0)

    def shielding_K(self, G: float, nu: float, b: float) -> float:
        return self.active_K_shielding(G, nu, b) + self.wake_K_shielding(G, nu, b)

    def local_slip_count(self) -> float:
        L = max(float(self.cfg.blunting_length_m), self.dx)
        w = np.exp(-self.x / L)
        return float(np.sum(self.accumulated_slip * w[None, :]))

    def blunted_radius(self, r0: float, b: float) -> float:
        return max(float(r0) + max(self.manifest.c_blunt, 0.0) * abs(float(b)) * self.local_slip_count(), float(r0))

    def local_forest_density_m2(self, wake: bool = False) -> np.ndarray:
        field = self.wake_retained if wake else self.retained
        dx = self.wake_dx if wake else self.dx
        width = max(float(self.cfg.blunting_length_m), dx, 1.0e-12)
        count = np.sum(np.maximum(field, 0.0), axis=0)
        return np.maximum(float(self.cfg.forest_density_floor_m2) + count / max(dx * width, 1.0e-30), 1.0)

    def local_stress_profile_Pa(self, tip_stress_Pa: float) -> np.ndarray:
        ref = max(float(self.cfg.blunting_length_m), self.dx, 1.0e-12)
        return max(float(tip_stress_Pa), 0.0) * np.sqrt(ref / np.maximum(ref + self.x, ref))

    def emission_rate_per_site(self, stress_Pa: float, T_K: float) -> float:
        return float(np.asarray(self.manifest.emission.rate(stress_Pa, T_K)))

    def _transport_rates(self, stress_profile: np.ndarray, rho: np.ndarray, T_K: float, b: float) -> dict[str, np.ndarray]:
        p_surface = self.manifest.peierls.as_surface(self.manifest.emission)
        t_surface = self.manifest.taylor.as_surface(self.manifest.emission)
        tau_p = max(float(self.cfg.peierls_stress_fraction), 0.0) * np.maximum(stress_profile, 0.0)
        spacing = 1.0 / (2.0 * np.sqrt(np.maximum(rho, 1.0)))
        phi = spacing / max(abs(float(b)), 1.0e-30)
        tau_t = max(float(self.cfg.taylor_stress_fraction), 0.0) * np.maximum(stress_profile, 0.0) * phi
        p = p_surface.rate(tau_p, T_K)
        t1 = t_surface.rate(tau_t, T_K)
        ratio = np.sqrt(np.maximum(rho, 0.0) / max(self.manifest.taylor_corr_rho_c_m2, 1.0))
        m = 1.0 + max(self.manifest.taylor_corr_scale, 0.0) * np.maximum(ratio - 1.0, 0.0)
        t = gammainc(np.maximum(m, 1.0), np.minimum(np.maximum(t1, 0.0), 1.0e12))
        jump = max(float(self.cfg.jump_fraction), 0.0) * spacing
        velocity = jump * p
        encounter = max(self.manifest.encounter_efficiency, 0.0) * velocity * np.sqrt(np.maximum(rho, 0.0))
        return {"peierls": p, "taylor": t, "taylor_single": t1, "m": m, "jump": jump, "velocity": velocity, "encounter": encounter}

    @staticmethod
    def _exchange(mobile: np.ndarray, retained: np.ndarray, k_enc: np.ndarray, k_rel: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, float, float]:
        ke = np.maximum(k_enc, 0.0)[None, :]
        kr = np.maximum(k_rel, 0.0)[None, :]
        total = np.maximum(mobile, 0.0) + np.maximum(retained, 0.0)
        rate = ke + kr
        req = np.divide(ke, rate, out=np.zeros_like(rate), where=rate > 0.0) * total
        decay = np.exp(-np.minimum(rate * max(float(dt), 0.0), 700.0))
        new_r = np.clip(req + (retained - req) * decay, 0.0, total)
        new_m = total - new_r
        trapped = float(np.sum(np.maximum(new_r - retained, 0.0)))
        released = float(np.sum(np.maximum(retained - new_r, 0.0)))
        return new_m, new_r, trapped, released

    @staticmethod
    def _advect_forward(field: np.ndarray, distance: float, dx: float) -> tuple[np.ndarray, float]:
        if distance <= 0.0:
            return field.copy(), 0.0
        shift = distance / dx
        whole = int(math.floor(shift))
        frac = shift - whole
        out = np.zeros_like(field)
        n = field.shape[1]
        for i in range(n):
            j = i + whole
            if j < n:
                out[:, j] += field[:, i] * (1.0 - frac)
            if frac > 0.0 and j + 1 < n:
                out[:, j + 1] += field[:, i] * frac
        return out, max(float(np.sum(field) - np.sum(out)), 0.0)

    def _emit(self, dt: float, stress_Pa: float, T_K: float, system_weights: np.ndarray | None = None) -> float:
        lam = self.emission_rate_per_site(stress_Pa, T_K)
        probability = 1.0 - math.exp(-min(max(lam * max(dt, 0.0), 0.0), 700.0))
        emitted_system = self.available_sites * probability
        if system_weights is not None:
            weights = np.maximum(np.asarray(system_weights, dtype=float).reshape(-1), 0.0)
            if weights.size < self.n_systems:
                weights = np.pad(weights, (0, self.n_systems - weights.size), mode="edge")
            weights = weights[: self.n_systems]
            if np.max(weights) > 0.0:
                emitted_system *= weights / np.max(weights)
        self.available_sites -= emitted_system
        nsrc = max(min(int(self.cfg.source_bin_count), self.n_bins), 1)
        self.mobile[:, :nsrc] += emitted_system[:, None] / nsrc
        self.accumulated_slip[:, :nsrc] += emitted_system[:, None] / nsrc
        emitted = float(np.sum(emitted_system))
        self.emitted_total += emitted
        return emitted

    def evolve(self, dt_s: float, T_K: float, stress_Pa: float, b: float, system_weights: np.ndarray | None = None) -> dict[str, float]:
        dt = max(float(dt_s), 0.0)
        emitted = self._emit(dt, stress_Pa, T_K, system_weights)
        stress = self.local_stress_profile_Pa(stress_Pa)
        rho = self.local_forest_density_m2(False)
        rates = self._transport_rates(stress, rho, T_K, b)
        self.mobile, self.retained, trapped, released = self._exchange(self.mobile, self.retained, rates["encounter"], rates["taylor"], dt)
        fr = 1.0 - math.exp(-min(max(self.manifest.retained_recovery_rate_s, 0.0) * dt, 700.0))
        fm = 1.0 - math.exp(-min(max(self.cfg.mobile_recovery_rate_s, 0.0) * dt, 700.0))
        rec_r = self.retained * fr
        rec_m = self.mobile * fm
        self.retained -= rec_r
        self.mobile -= rec_m
        recovered = float(np.sum(rec_r) + np.sum(rec_m))
        mobile_by_bin = np.sum(np.maximum(self.mobile, 0.0), axis=0)
        velocity = float(np.sum(rates["velocity"] * mobile_by_bin) / np.sum(mobile_by_bin)) if np.sum(mobile_by_bin) > 0.0 else float(np.mean(rates["velocity"][: max(min(self.cfg.source_bin_count, self.n_bins), 1)]))
        self.mobile, escaped = self._advect_forward(self.mobile, max(velocity, 0.0) * dt, self.dx)
        self.escaped_total += escaped
        self.recovered_total += recovered
        self.time_s += dt
        wake = self._evolve_wake(dt, T_K, b)
        return {
            "dN_emit": emitted,
            "dN_trapped": trapped,
            "dN_released": released,
            "dN_recovered": recovered,
            "dN_escaped": escaped,
            "peierls_rate_s": float(np.max(rates["peierls"])),
            "taylor_completion_rate_s": float(np.max(rates["taylor"])),
            "encounter_rate_s": float(np.max(rates["encounter"])),
            "taylor_m_eff": float(np.max(rates["m"])),
            "available_site_fraction": self.available_site_fraction,
            **wake,
        }

    def _evolve_wake(self, dt: float, T_K: float, b: float) -> dict[str, float]:
        if dt <= 0.0 or self.wake_mobile_count + self.wake_retained_count <= 0.0:
            return {"wake_dN_trapped": 0.0, "wake_dN_released": 0.0, "wake_dN_recovered": 0.0, "wake_dN_mobile_transport_loss": 0.0}
        rho = self.local_forest_density_m2(True)
        stress = np.zeros(self.wake_n_bins)
        rates = self._transport_rates(stress, rho, T_K, b)
        self.wake_mobile, self.wake_retained, trapped, released = self._exchange(self.wake_mobile, self.wake_retained, rates["encounter"], rates["taylor"], dt)
        fr = 1.0 - math.exp(-min(max(self.manifest.retained_recovery_rate_s, 0.0) * dt, 700.0))
        fm = 1.0 - math.exp(-min(max(self.cfg.mobile_recovery_rate_s, 0.0) * dt, 700.0))
        rec = float(np.sum(self.wake_retained * fr) + np.sum(self.wake_mobile * fm))
        self.wake_retained *= (1.0 - fr)
        self.wake_mobile *= (1.0 - fm)
        mobile_by_bin = np.sum(np.maximum(self.wake_mobile, 0.0), axis=0)
        velocity = float(np.sum(rates["velocity"] * mobile_by_bin) / np.sum(mobile_by_bin)) if np.sum(mobile_by_bin) > 0.0 else 0.0
        self.wake_mobile, lost = self._advect_forward(self.wake_mobile, max(velocity, 0.0) * dt, self.wake_dx)
        self.wake_discarded_mobile_total += lost
        return {"wake_dN_trapped": trapped, "wake_dN_released": released, "wake_dN_recovered": rec, "wake_dN_mobile_transport_loss": lost}

    @staticmethod
    def _translate_active(field: np.ndarray, distance: float, dx: float, wake_bins: int, wake_dx: float) -> tuple[np.ndarray, np.ndarray, float]:
        active = np.zeros_like(field)
        wake = np.zeros((field.shape[0], wake_bins), dtype=float)
        total = float(np.sum(field))
        for i in range(field.shape[1]):
            center = (i + 0.5) * dx - distance
            mass = field[:, i]
            if center >= 0.0:
                j = min(int(center / dx), field.shape[1] - 1)
                active[:, j] += mass
            else:
                y = -center
                j = int(y / wake_dx)
                if j < wake_bins:
                    wake[:, j] += mass
        discarded = max(total - float(np.sum(active)) - float(np.sum(wake)), 0.0)
        return active, wake, discarded

    @staticmethod
    def _shift_wake(field: np.ndarray, distance: float, dx: float) -> tuple[np.ndarray, float]:
        out, lost = UnifiedMPZState._advect_forward(field, distance, dx)
        return out, lost

    def advance(self, distance_m: float) -> dict[str, float]:
        d = max(float(distance_m), 0.0)
        old_wm, lost_old_m = self._shift_wake(self.wake_mobile, d, self.wake_dx)
        old_wr, lost_old_r = self._shift_wake(self.wake_retained, d, self.wake_dx)
        old_ws, lost_old_s = self._shift_wake(self.wake_slip, d, self.wake_dx)
        self.mobile, crossed_m, lost_m = self._translate_active(self.mobile, d, self.dx, self.wake_n_bins, self.wake_dx)
        self.retained, crossed_r, lost_r = self._translate_active(self.retained, d, self.dx, self.wake_n_bins, self.wake_dx)
        self.accumulated_slip, crossed_s, lost_s = self._translate_active(self.accumulated_slip, d, self.dx, self.wake_n_bins, self.wake_dx)
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
        }

    def diagnostics(self, G: float, nu: float, b: float, r0: float) -> dict[str, Any]:
        active = self.active_K_shielding(G, nu, b)
        wake = self.wake_K_shielding(G, nu, b)
        return {
            "mpz_state_model": self.state_model,
            "mpz_mobile_count": self.mobile_count,
            "mpz_retained_count": self.retained_count,
            "mpz_wake_mobile_count": self.wake_mobile_count,
            "mpz_wake_retained_count": self.wake_retained_count,
            "mpz_available_site_fraction": self.available_site_fraction,
            "mpz_active_K_shield_Pa_sqrt_m": active,
            "mpz_wake_K_shield_Pa_sqrt_m": wake,
            "mpz_total_K_shield_Pa_sqrt_m": active + wake,
            "mpz_blunted_radius_m": self.blunted_radius(r0, b),
            "mpz_emitted_total": self.emitted_total,
            "mpz_escaped_total": self.escaped_total,
            "mpz_recovered_total": self.recovered_total,
            "mpz_wake_discarded_mobile_total": self.wake_discarded_mobile_total,
            "mpz_wake_discarded_retained_total": self.wake_discarded_retained_total,
        }
