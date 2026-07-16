"""
Cycle-block fatigue controller for the sharp-front Arrhenius fracture model.

Version-1 goal
--------------
This module DOES NOT impose a Paris law and DOES NOT advance the crack once per
cycle block.  It integrates cyclic plasticity hazards over many cycles, updates
front-local process-zone state, and then lets the existing cleavage/fracture
clock decide whether the crack advances.

Intended coupling:

    cyclic K(t)
      -> dislocation emission / Peierls glide / Taylor depinning hazards
      -> retained process-zone ledger N_em, blunting, back stress, stored energy
      -> existing cleavage hazard with reduced effective barrier
      -> existing renewal advance when B >= 1

The plasticity barriers are EXP-floor free-energy barriers:

    DeltaG(s,T) = Gfloor(T) + [G0(T)-Gfloor(T)] exp[-a (s/sigc(T))^n]
    G0(T)       = G00 + gT (T-Tref)
    sigc(T)     = sigc0 + sT (T-Tref)

The controller evaluates DeltaG directly.  It never reconstructs
H - T S - sigma v, so the stress and temperature dependence are not double
counted.  S*, H*, and phi v* are diagnostics/derivatives of DeltaG, not
independent fatigue knobs.

Version-2 note
--------------
A later implementation should replace/front-load the scalar front-local ledger
with spatial fields D_fat(x), rho_pz(x), Gc_fat(x) around the process zone.  This
module is deliberately written so the same cycle-integrated hazard quantities
can be deposited into spatial fields later.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Iterable, List, Optional, Tuple
import csv
import json
import math
import os

import numpy as np

from .config import KB, EV_TO_J


# -----------------------------------------------------------------------------
# EXP-floor plasticity barriers
# -----------------------------------------------------------------------------

@dataclass
class ExpFloorBarrierParams:
    """EXP-floor activation free-energy barrier parameters.

    Defaults are W[100]-like values from the nanopillar export.  The a,n values
    are deliberately CLI-exposed because the pasted regression summaries listed
    G0(T) and sigc(T), but not always the shape parameters.
    """

    name: str = "W[100]"
    G00_eV: float = 1.94
    gT_eV_per_K: float = 0.003934
    sigc0_Pa: float = 2.298e9
    sT_Pa_per_K: float = -6.564e5
    Tref_K: float = 481.33
    a: float = 0.0845685
    n: float = 1.0
    Gfloor_fraction: float = 0.02
    Gfloor_min_eV: float = 1.0e-4
    Gfloor_max_fraction: float = 0.95

    @staticmethod
    def preset(name: str) -> "ExpFloorBarrierParams":
        key = str(name).strip().lower().replace(" ", "")
        presets = {
            "ta[111]": ExpFloorBarrierParams(
                name="Ta[111]", G00_eV=2.116, gT_eV_per_K=0.003072,
                sigc0_Pa=7.131e8, sT_Pa_per_K=-7.386e5, Tref_K=381.33,
                a=0.0845685, n=1.0),
            "w[100]": ExpFloorBarrierParams(
                name="W[100]", G00_eV=1.94, gT_eV_per_K=0.003934,
                sigc0_Pa=2.298e9, sT_Pa_per_K=-6.564e5, Tref_K=481.33,
                a=0.0845685, n=1.0),
            "al0.7cocrfeni-bcc": ExpFloorBarrierParams(
                name="Al0.7CoCrFeNi-BCC", G00_eV=0.8543, gT_eV_per_K=0.003515,
                sigc0_Pa=2.847e9, sT_Pa_per_K=-2.713e6, Tref_K=159.33,
                a=0.0845685, n=1.0),
            "al0.7cocrfeni-fcc": ExpFloorBarrierParams(
                name="Al0.7CoCrFeNi-FCC", G00_eV=0.6837, gT_eV_per_K=0.003995,
                sigc0_Pa=1.662e9, sT_Pa_per_K=-2.333e6, Tref_K=159.33,
                a=0.0845685, n=1.0),
            "cu": ExpFloorBarrierParams(
                name="Cu", G00_eV=2.114, gT_eV_per_K=0.003319,
                sigc0_Pa=5.487e8, sT_Pa_per_K=-9.226e5, Tref_K=481.33,
                a=0.0845685, n=1.0),
        }
        return presets.get(key, presets["w[100]"])


@dataclass
class ScaledExpFloorBarrier:
    """Mechanism-specific scaling of a nanopillar EXP-floor free-energy family.

    energy_scale multiplies the reference barrier height.  entropy_scale scales
    the T-slope gT independently so sensitivity studies can either scale the
    entire free-energy surface or preserve/enhance entropy-driven temperature
    dependence.  stress_scale multiplies sigc(T), making the mechanism harder
    for stress_scale > 1.
    """

    base: ExpFloorBarrierParams = field(default_factory=ExpFloorBarrierParams)
    mechanism: str = "nucleation"
    energy_scale: float = 1.0
    entropy_scale: float = 1.0
    stress_scale: float = 1.0
    rate_prefactor: float = 1.0e11

    def G0_eV(self, T_K: float) -> float:
        return max(
            self.energy_scale * self.base.G00_eV
            + self.entropy_scale * self.base.gT_eV_per_K * (T_K - self.base.Tref_K),
            1.0e-10,
        )

    def sigc_Pa(self, T_K: float) -> float:
        sigc = self.base.sigc0_Pa + self.base.sT_Pa_per_K * (T_K - self.base.Tref_K)
        return max(self.stress_scale * sigc, 1.0e6)

    def Gfloor_eV(self, T_K: float) -> float:
        G0 = self.G0_eV(T_K)
        raw = max(self.base.Gfloor_min_eV * max(self.energy_scale, 1.0e-12),
                  self.base.Gfloor_fraction * G0)
        return min(self.base.Gfloor_max_fraction * G0, raw)

    def deltaG_eV(self, sigma_Pa: np.ndarray | float, T_K: float) -> np.ndarray:
        sigma = np.asarray(sigma_Pa, dtype=float)
        G0 = self.G0_eV(T_K)
        Gfloor = self.Gfloor_eV(T_K)
        sigc = self.sigc_Pa(T_K)
        a = max(float(self.base.a), 0.0)
        n = max(float(self.base.n), 1.0e-9)
        x = np.maximum(np.abs(sigma), 0.0) / sigc
        DG = Gfloor + (G0 - Gfloor) * np.exp(-a * np.power(x, n))
        return np.maximum(DG, 0.0)

    def rate(self, sigma_Pa: np.ndarray | float, T_K: float) -> np.ndarray:
        DG_J = self.deltaG_eV(sigma_Pa, T_K) * EV_TO_J
        exponent = -DG_J / max(KB * T_K, 1.0e-30)
        return self.rate_prefactor * np.exp(np.clip(exponent, -700.0, 0.0))

    def entropy_over_kB_numeric(self, sigma_Pa: float, T_K: float, dT: float = 1.0) -> float:
        """Diagnostic S*/kB = -dDeltaG/dT / kB using centered finite difference."""
        T1 = max(T_K - dT, 1.0)
        T2 = T_K + dT
        dGdT_eV = (float(self.deltaG_eV(sigma_Pa, T2)) - float(self.deltaG_eV(sigma_Pa, T1))) / (T2 - T1)
        return -dGdT_eV * EV_TO_J / KB

    def dG_dsigma_eV_per_GPa_numeric(self, sigma_Pa: float, T_K: float) -> float:
        """Diagnostic derivative dDeltaG/dsigma [eV/GPa]."""
        sig = abs(float(sigma_Pa))
        h = max(1.0e5, 1.0e-5 * max(sig, 1.0))
        sm = max(sig - h, 0.0)
        sp = sig + h
        Gp = float(self.deltaG_eV(sp, T_K))
        Gm = float(self.deltaG_eV(sm, T_K))
        return (Gp - Gm) / max(sp - sm, 1.0) * 1.0e9

    def vstar_b3_numeric(self, sigma_Pa: float, T_K: float, b: float = 2.74e-10) -> float:
        """Diagnostic phi v* = -dDeltaG/dsigma in b^3."""
        dG_eV_per_Pa = self.dG_dsigma_eV_per_GPa_numeric(sigma_Pa, T_K) / 1.0e9
        v_m3 = -(dG_eV_per_Pa * EV_TO_J)
        return v_m3 / max(b**3, 1.0e-300)

    def as_dict(self) -> dict:
        out = asdict(self)
        out["base"] = asdict(self.base)
        return out


# -----------------------------------------------------------------------------
# Fatigue waveform and controller
# -----------------------------------------------------------------------------

@dataclass
class FatigueWaveform:
    """Tensile cyclic K waveform.

    Kmax is in Pa*sqrt(m).  R=Kmin/Kmax.  The default clips negative K_eff to
    zero, which is a simple first-pass closure/contact proxy.  More detailed
    closure should be a later addition.
    """

    Kmax: float
    R: float = 0.1
    frequency_Hz: float = 1000.0
    closure_clip: bool = True

    def K_phase(self, phase: np.ndarray) -> np.ndarray:
        Kmin = self.R * self.Kmax
        Kmean = 0.5 * (self.Kmax + Kmin)
        Kamp = 0.5 * (self.Kmax - Kmin)
        # cosine phase starts at Kmax, which is useful for diagnostics.
        K = Kmean + Kamp * np.cos(phase)
        if self.closure_clip:
            K = np.maximum(K, 0.0)
        return K

    @property
    def period_s(self) -> float:
        return 1.0 / max(float(self.frequency_Hz), 1.0e-300)

    @property
    def DeltaK(self) -> float:
        return self.Kmax - self.R * self.Kmax


@dataclass
class FatigueControllerConfig:
    """Cycle-block controls.

    The adaptive block size limits changes in the process-zone ledger and the
    cleavage clock.  That is the key numerical distinction from an imposed
    crack-growth law: the controller may take large cycle jumps near threshold,
    but it cannot jump across a process-zone/fracture transition.
    """

    n_phase: int = 96
    block_cycles: float = 1000.0
    adaptive_cycles: bool = True
    max_block_cycles: float = 1.0e6
    min_block_cycles: float = 1.0e-6
    target_dB: float = 0.20
    target_dN_store: float = 0.25
    recovery_per_s: float = 0.0
    N_sat: float = float("inf")
    storage_model: str = "escape_limited"  # escape_limited | all_retained | fixed_fraction
    fixed_retained_fraction: float = 1.0
    # Cycle-block controller.  requested_cap reproduces the older behavior:
    # --block-cycles is a hard upper bound.  hazard_limited treats
    # --max-block-cycles as the upper bound and chooses the largest block that
    # keeps every active clock/state increment below its target.
    cycle_block_mode: str = "requested_cap"  # requested_cap | hazard_limited
    target_dN_emit: float = float("inf")
    target_dN_mobile: float = float("inf")
    target_dN_escape: float = float("inf")
    target_dN_peierls: float = float("inf")
    target_dN_taylor: float = float("inf")


@dataclass
class CycleHazardResult:
    mu_emit: float
    mu_peierls: float
    mu_taylor: float
    mu_escape: float
    mu_cleave: float
    store_per_cycle: float
    mobile_per_cycle: float
    escape_per_cycle: float
    peierls_per_cycle: float
    taylor_per_cycle: float
    avg_sigma_tip: float
    max_sigma_tip: float
    avg_sigma_emit_eff: float
    storage_fraction: float


class FatigueCycleHazardController:
    """Cycle-block fatigue process-zone controller for one sharp front."""

    def __init__(
        self,
        cfg: FatigueControllerConfig,
        emit_barrier: ScaledExpFloorBarrier,
        peierls_barrier: ScaledExpFloorBarrier,
        taylor_barrier: ScaledExpFloorBarrier,
    ):
        self.cfg = cfg
        self.emit_barrier = emit_barrier
        self.peierls_barrier = peierls_barrier
        self.taylor_barrier = taylor_barrier

    def _phases(self) -> np.ndarray:
        n = max(int(self.cfg.n_phase), 8)
        return (np.arange(n, dtype=float) + 0.5) * (2.0 * np.pi / n)

    @staticmethod
    def _series_rate(lam1: np.ndarray, lam2: np.ndarray) -> np.ndarray:
        """Harmonic/series rate for sequential Peierls + Taylor bottlenecks."""
        lam1 = np.maximum(np.asarray(lam1, dtype=float), 0.0)
        lam2 = np.maximum(np.asarray(lam2, dtype=float), 0.0)
        return 1.0 / (1.0 / np.maximum(lam1, 1.0e-300) + 1.0 / np.maximum(lam2, 1.0e-300))

    def integrate_one_cycle(self, front, waveform: FatigueWaveform, T_K: float) -> CycleHazardResult:
        phase = self._phases()
        K = waveform.K_phase(phase)
        dtw = waveform.period_s / len(phase)

        sig_tip = np.array([front.sigma_tip(float(k)) for k in K], dtype=float)
        sig_back = float(front.sigma_back())
        sig_emit_eff = np.maximum(sig_tip - sig_back, 0.0)

        lam_emit = self.emit_barrier.rate(sig_emit_eff, T_K)
        lam_P = self.peierls_barrier.rate(sig_emit_eff, T_K)
        lam_T = self.taylor_barrier.rate(sig_emit_eff, T_K)
        lam_escape = self._series_rate(lam_P, lam_T)

        mu_emit = float(np.sum(lam_emit) * dtw)
        mu_P = float(np.sum(lam_P) * dtw)
        mu_T = float(np.sum(lam_T) * dtw)
        mu_escape = float(np.sum(lam_escape) * dtw)

        if self.cfg.storage_model == "all_retained":
            storage_fraction = 1.0
        elif self.cfg.storage_model == "fixed_fraction":
            storage_fraction = float(np.clip(self.cfg.fixed_retained_fraction, 0.0, 1.0))
        else:
            # If glide/depinning escape hazards are fast over a cycle, emitted
            # dislocations escape the near-tip ledger; if escape is slow, they
            # remain stored and embrittle/shield the process zone.
            storage_fraction = float(np.exp(-min(mu_escape, 700.0)))
        store_per_cycle = mu_emit * storage_fraction

        # Cleavage hazard is evaluated from the current/front state here.  The
        # block step below re-evaluates it after the process-zone update before
        # committing dB, so this value is a conservative predictor used for
        # adaptive cycle sizing.
        lam_c = []
        for s in sig_tip:
            lc, _, _ = front.lambda_cleave(float(s), T_K)
            lam_c.append(max(float(lc), 0.0))
        mu_c = float(np.sum(lam_c) * dtw)

        w = np.maximum(lam_emit, 0.0)
        if np.sum(w) > 0:
            avg_sigma_emit_eff = float(np.sum(w * sig_emit_eff) / np.sum(w))
        else:
            avg_sigma_emit_eff = float(np.mean(sig_emit_eff))

        return CycleHazardResult(
            mu_emit=mu_emit,
            mu_peierls=mu_P,
            mu_taylor=mu_T,
            mu_escape=mu_escape,
            mu_cleave=mu_c,
            store_per_cycle=float(store_per_cycle),
            mobile_per_cycle=float(max(mu_emit - store_per_cycle, 0.0)),
            escape_per_cycle=float(mu_escape),
            peierls_per_cycle=float(mu_P),
            taylor_per_cycle=float(mu_T),
            avg_sigma_tip=float(np.mean(sig_tip)),
            max_sigma_tip=float(np.max(sig_tip)),
            avg_sigma_emit_eff=avg_sigma_emit_eff,
            storage_fraction=storage_fraction,
        )

    def choose_block_cycles_diagnostic(self, pred: CycleHazardResult, user_block_cycles: Optional[float] = None) -> dict:
        """Choose a cycle-block size and report the active limiter.

        The adaptive variable is the physical number of cycles, ΔN.  We estimate
        per-cycle increments for fracture and process-zone state, then choose
        the largest ΔN such that no monitored increment exceeds its target.
        This is the fatigue analogue of adaptive time stepping; it allows very
        large jumps in VHCF/rare-event regimes and fractional cycles in
        high-hazard regimes.
        """
        req = float(user_block_cycles if user_block_cycles is not None else self.cfg.block_cycles)
        maxb = float(self.cfg.max_block_cycles)
        minb = max(float(self.cfg.min_block_cycles), 0.0)
        if not self.cfg.adaptive_cycles:
            cyc = float(np.clip(req, minb, maxb))
            return {"cycles": cyc, "limiter": "fixed", "unlimited_cycles": req, "candidate_limits": {"fixed": req}}

        mode = str(getattr(self.cfg, "cycle_block_mode", "requested_cap") or "requested_cap").lower()
        if mode in ("hazard", "hazard_limited", "rate", "auto"):
            base = maxb
            base_name = "max_block_cycles"
        else:
            base = min(maxb, req)
            base_name = "block_cycles"

        limits = {base_name: float(base)}
        def _add(name: str, target: float, rate_per_cycle: float):
            try:
                target = float(target); rate_per_cycle = float(rate_per_cycle)
            except Exception:
                return
            if target > 0.0 and np.isfinite(target) and rate_per_cycle > 0.0 and np.isfinite(rate_per_cycle):
                limits[name] = target / rate_per_cycle

        _add("cleavage_clock", self.cfg.target_dB, pred.mu_cleave)
        _add("stored_pz", self.cfg.target_dN_store, pred.store_per_cycle)
        _add("emitted_pz", self.cfg.target_dN_emit, pred.mu_emit)
        _add("mobile_pz", self.cfg.target_dN_mobile, pred.mobile_per_cycle)
        _add("escape_pz", self.cfg.target_dN_escape, pred.escape_per_cycle)
        _add("peierls_clock", self.cfg.target_dN_peierls, pred.peierls_per_cycle)
        _add("taylor_clock", self.cfg.target_dN_taylor, pred.taylor_per_cycle)

        limiter = min(limits, key=lambda k: limits[k]) if limits else base_name
        cyc_raw = float(limits.get(limiter, base))
        cyc = float(np.clip(cyc_raw, minb, maxb))
        if cyc != cyc_raw:
            limiter = "min_block_cycles" if cyc <= minb + 1e-300 else "max_block_cycles"
        return {"cycles": cyc, "limiter": limiter, "unlimited_cycles": cyc_raw, "candidate_limits": limits}

    def choose_block_cycles(self, pred: CycleHazardResult, user_block_cycles: Optional[float] = None) -> float:
        return float(self.choose_block_cycles_diagnostic(pred, user_block_cycles)["cycles"])

    def _commit_cleavage_clock(self, front, waveform: FatigueWaveform, T_K: float, cycles: float) -> float:
        """Accumulate the existing cleavage clock after the PZ update."""
        phase = self._phases()
        K = waveform.K_phase(phase)
        dtw_block = cycles * waveform.period_s / len(phase)
        dB = 0.0
        for k in K:
            sig = front.sigma_tip(float(k))
            lam_c, _, _ = front.lambda_cleave(sig, T_K)
            dB += max(float(lam_c), 0.0) * dtw_block
        front.B += dB
        return float(dB)

    @staticmethod
    def _renew_if_fired(front, dt_block_s: float) -> dict:
        """Mirror FrontEngine.step renewal semantics without imposing a fatigue law."""
        N_em_pre = float(front.N_em)
        sigma_back_pre = float(front.sigma_back())
        r_eff_pre = float(front.r_eff())
        dG_emb_pre = float(front.dG_emb())

        if not np.isfinite(front.B):
            front.B = 0.0
        n_fire = int(np.floor(min(max(front.B, 0.0), 1.0e7)))
        fired = n_fire >= 1
        N_retained = N_em_pre
        N_shed = 0.0
        if fired:
            front.B -= n_fire
            retain = float(np.clip(front.f.wake_retain, 0.0, 1.0)) ** n_fire
            N_retained = N_em_pre * retain
            N_shed = N_em_pre * (1.0 - retain)
            front.N_em = N_retained
            front.a_adv += front.f.da * n_fire
            front.n_adv += n_fire
        return {
            "fired": bool(fired),
            "n_fire": int(n_fire),
            "v_crack": (front.f.da * n_fire / dt_block_s) if dt_block_s > 0 else 0.0,
            "N_em_pre_renewal": N_em_pre,
            "N_em_retained": N_retained,
            "N_em_shed_to_wake": N_shed,
            "sigma_back_pre_renewal": sigma_back_pre,
            "r_eff_pre_renewal": r_eff_pre,
            "dG_emb_pre_renewal_eV": dG_emb_pre / EV_TO_J,
        }

    def cycle_step_front(
        self,
        front,
        waveform: FatigueWaveform,
        T_K: float,
        requested_cycles: Optional[float] = None,
        force_cycles: Optional[float] = None,
    ) -> dict:
        """Advance the process-zone state by a cycle block.

        Crack advance occurs only if the existing cleavage clock reaches B>=1.
        The cycle block itself is not a crack-growth increment.
        """
        if hasattr(front, "cycle_step_waveform"):
            return front.cycle_step_waveform(
                self, waveform, T_K,
                requested_cycles=requested_cycles,
                force_cycles=force_cycles,
            )
        pred = self.integrate_one_cycle(front, waveform, T_K)
        if force_cycles is None:
            stepdiag = self.choose_block_cycles_diagnostic(pred, requested_cycles)
            cycles = float(stepdiag["cycles"])
        else:
            cycles = max(float(force_cycles), 0.0)
            stepdiag = {"cycles": cycles, "limiter": "global_forced", "unlimited_cycles": cycles, "candidate_limits": {"global_forced": cycles}}
        dt_block = cycles * waveform.period_s

        # Commit process-zone plasticity first.  This makes fatigue growth emerge
        # through shielding/embrittlement of the existing crack-opening hazard.
        dN_emit = cycles * pred.mu_emit
        dN_peierls = cycles * pred.mu_peierls
        dN_taylor = cycles * pred.mu_taylor
        dN_escape = cycles * pred.mu_escape
        dN_store = cycles * pred.store_per_cycle
        if np.isfinite(self.cfg.N_sat) and self.cfg.N_sat > 0:
            dN_store *= max(1.0 - front.N_em / self.cfg.N_sat, 0.0)
        # The mobile increment is an explicit audit/state source for 2-D coupling:
        # emission creates mobile content; the chosen storage law decides how much
        # remains retained near the front.  The v8 spatial adapter deposits this
        # quantity into a mobile process-zone field and lets the cyclic FEM
        # plasticity/transport tools move or annihilate it.
        dN_mobile = max(dN_emit - dN_store, 0.0)
        dN_recover = self.cfg.recovery_per_s * front.N_em * dt_block

        front.W_emit += pred.avg_sigma_emit_eff * front.b * front.f.L_pz * max(dN_emit, 0.0)
        front.N_em = max(float(front.N_em) + dN_store - dN_recover, 0.0)

        # Then commit cleavage hazard with the updated PZ state.  This is the
        # existing fracture clock; it is affected by N_em through r_eff,
        # sigma_back, dG_emb, and any existing shielding settings.
        dB = self._commit_cleavage_clock(front, waveform, T_K, cycles)
        # Diagnostic: effective cleavage barrier at the peak cyclic driving, after
        # the process-zone ledger update and before/after possible renewal.
        _lc_diag, _lcraw_diag, Gcleave_eff_diag = front.lambda_cleave(pred.max_sigma_tip, T_K)
        front.t += dt_block
        front.K_prev = waveform.Kmax
        renew = self._renew_if_fired(front, dt_block)

        out = {
            "cycles": float(cycles),
            "cycle_limiter": str(stepdiag.get("limiter", "unknown")),
            "cycle_unlimited": float(stepdiag.get("unlimited_cycles", cycles)),
            "time_s": float(front.t),
            "Kmax_Pa_sqrt_m": float(waveform.Kmax),
            "DeltaK_Pa_sqrt_m": float(waveform.DeltaK),
            "R": float(waveform.R),
            "frequency_Hz": float(waveform.frequency_Hz),
            "T_K": float(T_K),
            "mu_emit": pred.mu_emit,
            "mu_peierls": pred.mu_peierls,
            "mu_taylor": pred.mu_taylor,
            "mu_escape": pred.mu_escape,
            "mu_cleave_pred": pred.mu_cleave,
            # Compatibility keys for the existing 2-D sharp-front diagnostics.
            # They are rates per second obtained from cycle hazards multiplied
            # by cycling frequency.
            "lambda_e": float(pred.mu_emit * waveform.frequency_Hz),
            "lambda_c": float(pred.mu_cleave * waveform.frequency_Hz),
            "lambda_c_raw": float(pred.mu_cleave * waveform.frequency_Hz),
            "G_emit_eV": float(self.emit_barrier.deltaG_eV(pred.avg_sigma_emit_eff, T_K)),
            "S_emit_kB": float(self.emit_barrier.entropy_over_kB_numeric(pred.avg_sigma_emit_eff, T_K)),
            "dGemit_dsigma_eV_per_GPa": float(self.emit_barrier.dG_dsigma_eV_per_GPa_numeric(pred.avg_sigma_emit_eff, T_K)),
            "vstar_emit_b3": float(self.emit_barrier.vstar_b3_numeric(pred.avg_sigma_emit_eff, T_K, getattr(front, 'b', 2.74e-10))),
            "G_peierls_eV": float(self.peierls_barrier.deltaG_eV(pred.avg_sigma_emit_eff, T_K)),
            "S_peierls_kB": float(self.peierls_barrier.entropy_over_kB_numeric(pred.avg_sigma_emit_eff, T_K)),
            "G_taylor_eV": float(self.taylor_barrier.deltaG_eV(pred.avg_sigma_emit_eff, T_K)),
            "S_taylor_kB": float(self.taylor_barrier.entropy_over_kB_numeric(pred.avg_sigma_emit_eff, T_K)),
            "G_cleave_eff_eV": float(Gcleave_eff_diag / EV_TO_J),
            **front.cleavage_diagnostics(pred.max_sigma_tip, T_K),
            "sigma_tip": float(pred.max_sigma_tip),
            "sigma_back": float(front.sigma_back()),
            "r_eff": float(front.r_eff()),
            "store_per_cycle": pred.store_per_cycle,
            "mobile_per_cycle": pred.mobile_per_cycle,
            "escape_per_cycle": pred.escape_per_cycle,
            "peierls_per_cycle": pred.peierls_per_cycle,
            "taylor_per_cycle": pred.taylor_per_cycle,
            "storage_fraction": pred.storage_fraction,
            "dN_emit_block": float(dN_emit),
            "dN_peierls_block": float(dN_peierls),
            "dN_taylor_block": float(dN_taylor),
            "dN_escape_block": float(dN_escape),
            "dN_mobile_block": float(dN_mobile),
            "dN_store_block": float(dN_store),
            "dN_recover_block": float(dN_recover),
            "dB_block": float(dB),
            "B": float(front.B),
            "N_em": float(front.N_em),
            "r_eff_m": float(front.r_eff()),
            "sigma_back_Pa": float(front.sigma_back()),
            "dG_emb_eV": float(front.dG_emb() / EV_TO_J),
            "a_adv_m": float(front.a_adv),
            "n_adv": int(front.n_adv),
            "avg_sigma_tip_Pa": pred.avg_sigma_tip,
            "max_sigma_tip_Pa": pred.max_sigma_tip,
            "avg_sigma_emit_eff_Pa": pred.avg_sigma_emit_eff,
        }
        out.update(renew)
        return out

    def write_config(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "controller": asdict(self.cfg),
            "emit_barrier": self.emit_barrier.as_dict(),
            "peierls_barrier": self.peierls_barrier.as_dict(),
            "taylor_barrier": self.taylor_barrier.as_dict(),
        }
        with open(path, "w") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)


def build_controller_from_namespace(args) -> FatigueCycleHazardController:
    """Build a fatigue controller from argparse-like attributes.

    This is shared by the K-controlled smoke driver and the full-field 2-D
    adapter so the mechanism scalings remain identical.
    """
    base = ExpFloorBarrierParams.preset(getattr(args, "exp_system", "W[100]"))
    exp_a = getattr(args, "exp_a", None)
    exp_n = getattr(args, "exp_n", None)
    if exp_a is not None:
        base.a = float(exp_a)
    if exp_n is not None:
        base.n = float(exp_n)

    emit = ScaledExpFloorBarrier(
        base=base, mechanism="crack_tip_dislocation_emission",
        energy_scale=float(getattr(args, "emit_energy_scale", 1.0)),
        entropy_scale=float(getattr(args, "emit_entropy_scale", 1.0)),
        stress_scale=float(getattr(args, "emit_stress_scale", 1.0)),
        rate_prefactor=float(getattr(args, "nu0_emit_pz", 1.0e11)),
    )
    peierls = ScaledExpFloorBarrier(
        base=base, mechanism="peierls_glide_escape",
        energy_scale=float(getattr(args, "peierls_energy_scale", 0.02)),
        entropy_scale=float(getattr(args, "peierls_entropy_scale", 0.02)),
        stress_scale=float(getattr(args, "peierls_stress_scale", 1.0)),
        rate_prefactor=float(getattr(args, "nu0_peierls", 1.0e12)),
    )
    taylor = ScaledExpFloorBarrier(
        base=base, mechanism="taylor_junction_depinning_escape",
        energy_scale=float(getattr(args, "taylor_energy_scale", 0.10)),
        entropy_scale=float(getattr(args, "taylor_entropy_scale", 0.10)),
        stress_scale=float(getattr(args, "taylor_stress_scale", 1.0)),
        rate_prefactor=float(getattr(args, "nu0_taylor", 1.0e11)),
    )
    cfg = FatigueControllerConfig(
        n_phase=int(getattr(args, "n_phase", 96)),
        block_cycles=float(getattr(args, "block_cycles", getattr(args, "fatigue_block_cycles", 1.0e4))),
        adaptive_cycles=not bool(getattr(args, "no_adaptive_cycles", False)),
        max_block_cycles=float(getattr(args, "max_block_cycles", 1.0e6)),
        min_block_cycles=float(getattr(args, "min_block_cycles", 1.0)),
        target_dB=float(getattr(args, "target_dB", 0.2)),
        target_dN_store=float(getattr(args, "target_dN_store", 0.25)),
        recovery_per_s=float(getattr(args, "pz_recovery_per_s", 0.0)),
        N_sat=float(getattr(args, "N_sat", float("inf"))),
        storage_model=str(getattr(args, "storage_model", "escape_limited")),
        fixed_retained_fraction=float(getattr(args, "fixed_retained_fraction", 1.0)),
        cycle_block_mode=str(getattr(args, "cycle_block_mode", "requested_cap")),
        target_dN_emit=float(getattr(args, "target_dN_emit", float("inf"))),
        target_dN_mobile=float(getattr(args, "target_dN_mobile", float("inf"))),
        target_dN_escape=float(getattr(args, "target_dN_escape", float("inf"))),
        target_dN_peierls=float(getattr(args, "target_dN_peierls", float("inf"))),
        target_dN_taylor=float(getattr(args, "target_dN_taylor", float("inf"))),
    )
    return FatigueCycleHazardController(cfg, emit, peierls, taylor)


def write_history_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    # Add any late keys without disturbing the first-row order.
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
