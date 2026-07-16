"""Manifest-driven sharp-front engine with unified active/wake MPZ state."""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.special import gammainc

from .config import EV_TO_J
from .material_manifest import MaterialManifest
from .unified_mpz import MPZConfig, UnifiedMPZState


class UnifiedMPZFrontEngine:
    """Drop-in replacement for the legacy scalar ``FrontEngine``.

    Cleavage and plasticity evolve concurrently while geometry is fixed.  At
    first passage exactly one calibrated crack quantum is accepted; all excess
    hazard remains in ``B`` for the post-remesh equilibrium state.
    """

    unified_mpz_active = True

    def __init__(self, fcfg, cleave_barrier, emit_barrier, G_shear: float, nu: float, b: float,
                 manifest: MaterialManifest, mpz_cfg: MPZConfig):
        self.f = fcfg
        self.cb = cleave_barrier
        self.eb = emit_barrier
        self.G = float(G_shear)
        self.nu = float(nu)
        self.b = float(b)
        self.manifest = manifest
        self.mpz = UnifiedMPZState(manifest, mpz_cfg)
        self.reset()

    def reset(self):
        self.B = 0.0
        self.a_adv = 0.0
        self.n_adv = 0
        self.W_emit = 0.0
        self.t = 0.0
        self.K_prev = None
        self._lambda_c_prev = None

    @property
    def N_em(self) -> float:
        return self.mpz.mobile_count + self.mpz.retained_count

    @N_em.setter
    def N_em(self, value: float) -> None:
        target = max(float(value), 0.0)
        current = self.mpz.mobile_count + self.mpz.retained_count
        if current > 0.0:
            factor = target / current
            self.mpz.mobile *= factor
            self.mpz.retained *= factor
        elif target > 0.0:
            nsrc = max(min(self.mpz.cfg.source_bin_count, self.mpz.n_bins), 1)
            self.mpz.retained[:, :nsrc] = target / (self.mpz.n_systems * nsrc)

    def clone_split(self, daughter_fraction=0.5):
        frac = float(np.clip(daughter_fraction, 0.0, 1.0))
        child = copy.copy(self)
        child.mpz = self.mpz.split(frac)
        child.B = self.B * frac
        child.W_emit = self.W_emit * frac
        child.a_adv = 0.0
        child.n_adv = 0
        child.K_prev = self.K_prev
        child._lambda_c_prev = self._lambda_c_prev
        self.B *= (1.0 - frac)
        self.W_emit *= (1.0 - frac)
        return child

    def r_eff(self):
        return self.mpz.blunted_radius(self.f.r0, self.b)

    def K_shield(self):
        return self.mpz.shielding_K(self.G, self.nu, self.b)

    def sigma_tip(self, K):
        K_eff = max(float(K) - self.K_shield(), 0.0)
        s = K_eff / math.sqrt(2.0 * math.pi * max(self.r_eff(), 1.0e-30))
        if self.f.sigma_cap > 0.0:
            s = min(s, self.f.sigma_cap)
        return float(s)

    def sigma_back(self):
        # Compatibility field only.  No empirical scalar back-stress is used.
        return 0.0

    def e_stored(self):
        return 0.0

    def dG_emb(self):
        return 0.0

    def lambda_emit(self, sig_tip, T):
        G_eV = float(np.asarray(self.manifest.emission.values_eV(sig_tip, T)))
        lam = float(np.asarray(self.manifest.emission.rate(sig_tip, T)))
        return lam, float(sig_tip), G_eV * EV_TO_J

    def lambda_cleave(self, sig_tip, T):
        G_eV = float(np.asarray(self.manifest.cleavage.values_eV(sig_tip, T)))
        raw = float(np.asarray(self.manifest.cleavage.rate(sig_tip, T)))
        m = max(float(self.f.m_hits), 1.0)
        if m > 1.0 + 1.0e-12:
            tau = max(float(self.f.tau_c), 1.0e-30)
            effective = float(gammainc(m, min(raw * tau, 1.0e12)) / tau)
        else:
            effective = raw
        return effective, raw, G_eV * EV_TO_J

    def cleavage_diagnostics(self, sig_tip, T):
        sigma = max(float(sig_tip), 0.0)
        G = float(np.asarray(self.manifest.cleavage.values_eV(sigma, T)))
        ds = max(1.0e5, 1.0e-5 * max(sigma, 1.0e9))
        gp = float(np.asarray(self.manifest.cleavage.values_eV(sigma + ds, T)))
        gm = float(np.asarray(self.manifest.cleavage.values_eV(max(sigma - ds, 0.0), T)))
        dG_dsigma = (gp - gm) / (2.0 * ds)
        dT = max(1.0e-3, 1.0e-5 * max(float(T), 1.0))
        gTp = float(np.asarray(self.manifest.cleavage.values_eV(sigma, T + dT)))
        gTm = float(np.asarray(self.manifest.cleavage.values_eV(sigma, max(T - dT, 1.0e-6))))
        entropy_kB = -(gTp - gTm) / (2.0 * dT) / 8.617333262145e-5
        return {
            "sigma_cleave_eff_Pa": sigma,
            "G_cleave_raw_eV": G,
            "G_cleave_eff_eV": G,
            "S_cleave_kB": entropy_kB,
            "dGcleave_dsigma_eV_per_GPa": dG_dsigma * 1.0e9,
            "vstar_cleave_b3": max(-dG_dsigma * EV_TO_J / max(self.b ** 3, 1.0e-40), 0.0),
            "cleave_barrier_kind_code": 1.0,
        }

    @staticmethod
    def _logmean(a: float, b: float) -> float:
        lo, hi = sorted((max(float(a), 0.0), max(float(b), 0.0)))
        if hi <= 0.0:
            return 0.0
        if lo <= 0.0:
            return 0.5 * hi
        if abs(hi - lo) <= 1.0e-12 * hi:
            return hi
        return (hi - lo) / math.log(hi / lo)

    def predict_clock_increment(self, K, T, dt):
        if dt <= 0.0:
            return 0.0
        lam, _, _ = self.lambda_cleave(self.sigma_tip(K), T)
        previous = self._lambda_c_prev if self._lambda_c_prev is not None else lam
        return self._logmean(previous, lam) * float(dt)

    def _commit_one_renewal(self, dt: float) -> dict[str, Any]:
        if self.B < 1.0 or not math.isfinite(self.B):
            if not math.isfinite(self.B):
                self.B = 0.0
            return {"fired": False, "n_fire": 0, "v_crack": 0.0, "wake": {}}
        self.B -= 1.0
        wake = self.mpz.advance(self.f.da)
        self.a_adv += self.f.da
        self.n_adv += 1
        return {
            "fired": True,
            "n_fire": 1,
            "v_crack": self.f.da / dt if dt > 0.0 else 0.0,
            "wake": wake,
        }

    def step(self, K, T, dt):
        dt = max(float(dt), 0.0)
        K = max(float(K), 0.0)
        sig_pre = self.sigma_tip(K)
        lam_e, sig_e, Ge = self.lambda_emit(sig_pre, T)
        N_pre = self.N_em
        Kshield_pre = self.K_shield()
        r_pre = self.r_eff()
        evolve = self.mpz.evolve(dt, T, sig_pre, self.b)
        self.W_emit += sig_e * self.b * self.f.L_pz * max(evolve["dN_emit"], 0.0)

        sig_post = self.sigma_tip(K)
        lam_c, lam_raw, Gc = self.lambda_cleave(sig_post, T)
        dB = self._logmean(self._lambda_c_prev if self._lambda_c_prev is not None else lam_c, lam_c) * dt
        self.B += dB
        self._lambda_c_prev = lam_c
        self.K_prev = K
        self.t += dt
        renew = self._commit_one_renewal(dt)
        diagnostics = self.mpz.diagnostics(self.G, self.nu, self.b, self.f.r0)
        wake = renew.pop("wake")
        return {
            **renew,
            "sigma_tip": sig_post,
            "sigma_back": 0.0,
            "lambda_e": lam_e,
            "lambda_c": lam_c,
            "lambda_c_raw": lam_raw,
            "B": self.B,
            "N_em": self.N_em,
            "r_eff": self.r_eff(),
            "dG_emb_eV": 0.0,
            "G_cleave_eff_eV": Gc / EV_TO_J,
            **self.cleavage_diagnostics(sig_post, T),
            "G_emit_eV": Ge / EV_TO_J,
            "W_emit": self.W_emit,
            "sigma_tip_uncapped": (max(K - Kshield_pre, 0.0) / math.sqrt(2.0 * math.pi * max(r_pre, 1.0e-30))),
            "sigma_cap_active": bool(self.f.sigma_cap > 0.0 and sig_post >= self.f.sigma_cap),
            "dN_emit_raw": evolve["dN_emit"],
            "dN_cap_active": False,
            "N_sat_factor": 1.0,
            "N_sat_active": False,
            "N_em_pre_renewal": N_pre,
            "N_em_retained": self.N_em,
            "N_em_shed_to_wake": float(wake.get("wake_mobile", 0.0) + wake.get("wake_retained", 0.0)),
            "sigma_back_pre_renewal": 0.0,
            "r_eff_pre_renewal": r_pre,
            "dG_emb_pre_renewal_eV": 0.0,
            "dB_step": dB,
            "one_renewal_transaction": True,
            "material_class": self.manifest.name,
            "candidate_id": self.manifest.candidate_id,
            **evolve,
            **wake,
            **diagnostics,
        }

    def cycle_step_waveform(self, controller, waveform, T_K: float, requested_cycles=None, force_cycles=None):
        phase = controller._phases()
        Kvals = waveform.K_phase(phase)
        dt_phase = waveform.period_s / len(phase)
        sig = np.array([self.sigma_tip(float(k)) for k in Kvals])
        lam_e_site = self.manifest.emission.rate(sig, T_K)
        available = float(np.sum(self.mpz.available_sites))
        mu_emit = float(np.sum(lam_e_site * available) * dt_phase)
        lam_c = np.array([self.lambda_cleave(float(s), T_K)[0] for s in sig])
        mu_c = float(np.sum(lam_c) * dt_phase)
        limits = [float(requested_cycles if requested_cycles is not None else controller.cfg.block_cycles), float(controller.cfg.max_block_cycles)]
        if controller.cfg.adaptive_cycles:
            if mu_c > 0.0 and math.isfinite(controller.cfg.target_dB):
                limits.append(controller.cfg.target_dB / mu_c)
            if mu_emit > 0.0 and math.isfinite(controller.cfg.target_dN_emit):
                limits.append(controller.cfg.target_dN_emit / mu_emit)
        cycles = max(float(force_cycles) if force_cycles is not None else min(limits), float(controller.cfg.min_block_cycles))
        cycles = min(cycles, float(controller.cfg.max_block_cycles))
        dt_block = cycles * waveform.period_s
        weights = np.maximum(lam_e_site, 0.0)
        avg_sig = float(np.sum(weights * sig) / np.sum(weights)) if np.sum(weights) > 0.0 else float(np.mean(sig))
        N_pre = self.N_em
        evolve = self.mpz.evolve(dt_block, T_K, avg_sig, self.b)
        self.W_emit += avg_sig * self.b * self.f.L_pz * evolve["dN_emit"]
        dB = 0.0
        for k in Kvals:
            dB += self.lambda_cleave(self.sigma_tip(float(k)), T_K)[0] * cycles * dt_phase
        self.B += dB
        self.t += dt_block
        self.K_prev = waveform.Kmax
        renew = self._commit_one_renewal(dt_block)
        wake = renew.pop("wake")
        diag = self.mpz.diagnostics(self.G, self.nu, self.b, self.f.r0)
        return {
            "cycles": cycles,
            "cycle_limiter": "unified_hazard_state",
            "cycle_unlimited": cycles,
            "time_s": self.t,
            "Kmax_Pa_sqrt_m": waveform.Kmax,
            "DeltaK_Pa_sqrt_m": waveform.DeltaK,
            "R": waveform.R,
            "frequency_Hz": waveform.frequency_Hz,
            "T_K": T_K,
            "mu_emit": mu_emit,
            "mu_cleave_pred": mu_c,
            "lambda_e": mu_emit * waveform.frequency_Hz,
            "lambda_c": mu_c * waveform.frequency_Hz,
            "lambda_c_raw": mu_c * waveform.frequency_Hz,
            "dN_emit_block": evolve["dN_emit"],
            "dB_block": dB,
            "B": self.B,
            "N_em": self.N_em,
            "r_eff": self.r_eff(),
            "r_eff_m": self.r_eff(),
            "sigma_tip": float(np.max(sig)),
            "sigma_back": 0.0,
            "sigma_back_Pa": 0.0,
            "dG_emb_eV": 0.0,
            "a_adv_m": self.a_adv,
            "n_adv": self.n_adv,
            "N_em_pre_renewal": N_pre,
            "N_em_retained": self.N_em,
            "N_em_shed_to_wake": float(wake.get("wake_mobile", 0.0) + wake.get("wake_retained", 0.0)),
            "G_emit_eV": float(np.asarray(self.manifest.emission.values_eV(avg_sig, T_K))),
            "G_cleave_eff_eV": float(np.asarray(self.manifest.cleavage.values_eV(float(np.max(sig)), T_K))),
            **renew,
            **evolve,
            **wake,
            **diag,
            **self.cleavage_diagnostics(float(np.max(sig)), T_K),
        }

    def shelf_audit(self, T, t_total):
        sigma = self.f.sigma_cap if self.f.sigma_cap > 0.0 else 30.0e9
        le = self.lambda_emit(sigma, T)[0]
        lc = self.lambda_cleave(sigma, T)[0]
        t_min = 1.0 / lc if lc > 0.0 else math.inf
        return {
            "lambda_c_max": lc,
            "lambda_e_max": le,
            "t_min_fire_s": t_min,
            "clock_completable": bool(t_min < float(t_total)),
            "lambda_e_shelf": le,
            "lambda_c_shelf": lc,
            "t_emit_min": 1.0 / le if le > 0.0 else math.inf,
            "t_cleave_min": t_min,
            "t_total": float(t_total),
            "material_class": self.manifest.name,
        }
