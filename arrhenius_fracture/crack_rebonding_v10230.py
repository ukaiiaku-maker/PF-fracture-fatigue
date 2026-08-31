"""Engine-integration layer for optional crack-face rebonding (v10.2.30).

Imports only from ``crack_rebonding_kinetics_v10230`` (pure math) — never from
``unified_front``/``kinetic_tip_cell``/etc, to avoid an import cycle and to keep
this layer duck-typed against whatever engine object it is installed on (it only
ever calls ``engine.r_eff()``, ``engine.K_shield()``, ``engine.G``, ``engine.nu``).

See ``docs/v10_2_30_crack_rebonding_equation_lineage.md`` for the equation-by-
equation mapping from the mission specification to the functions below, and
``docs/v10_2_30_crack_rebonding_approved_plan.md`` for the reviewed design
(two rounds of correction folded in, including the phase-resolved
event-time-coupling requirement this module implements in
``solve_coupled_event_time``/``phase_resolved_action``).
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import types
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.linalg import expm

from .crack_rebonding_kinetics_v10230 import (
    CONTACT_SEMANTICS_LABEL,
    CrackRebondingControls,
    InitialPrecrackWakeMode,
    RebondModelLevel,
    bond_formation_rate,
    bond_rupture_rate,
    build_Q,
    cooperative_hazard,
    contact_pressure,
    depassivation_rate,
    opening_stress,
    repassivation_rate,
)

_ACTIVE_MODEL_LEVELS = frozenset(
    {RebondModelLevel.CLEAN_REVERSIBLE_REBOND, RebondModelLevel.PASSIVATION_GATED_REBOND}
)


def reduced_modulus_Pa(G_Pa: float, nu: float) -> float:
    """Plane-strain reduced modulus E' = 2G/(1-nu), computed from the engine's
    own shear modulus and Poisson ratio (avoids depending on the diagnostic-only
    global material-observer singleton used elsewhere for E')."""
    return 2.0 * float(G_Pa) / max(1.0 - float(nu), 1.0e-12)


def chronological_phase_offset_rad(elapsed_time_s: float, period_s: float) -> float:
    """The wake's own continuous phase-clock offset, mod one period.

    Adding this to the solver's fixed relative-phase sample points before
    evaluating ``K_phase`` gives the wake's genuinely elapsed-time-continuous
    view of the waveform, while the array used unshifted (``Kvals``/``sig``
    driving cleavage/emission) remains exactly as the existing solver
    already computes it. See ``RebondingWakeState``'s docstring for the
    full rationale.
    """
    if period_s <= 0.0:
        return 0.0
    return 2.0 * math.pi * (elapsed_time_s % period_s) / period_s


def wake_weight(s_m: float, L_h_m: float, L_w_m: float) -> float:
    if L_h_m <= 0.0 or L_w_m <= 0.0:
        return 0.0
    denom = L_w_m * (1.0 - math.exp(-L_h_m / L_w_m))
    if denom <= 1.0e-300:
        return 0.0
    return math.exp(-max(s_m, 0.0) / L_w_m) / denom


def contact_diagnostics(K_s_Pa_sqrt_m: float, s_j_m: float, r_contact_m: float, cfg: CrackRebondingControls) -> dict:
    """Compressive/opening exposure diagnostics, computed unconditionally
    (used for RB1 CONTACT_PROXY_ONLY archiving) but never fed into a nonzero
    Markov generator unless model_level is RB2/RB3."""
    return {
        "sigma_comp_Pa": contact_pressure(K_s_Pa_sqrt_m, cfg.contact_pressure_scale, r_contact_m, s_j_m, cfg.contact_pressure_cap_Pa),
        "sigma_open_Pa": opening_stress(K_s_Pa_sqrt_m, r_contact_m, s_j_m, cfg.opening_stress_cap_Pa),
        "K_s_sign": 1 if K_s_Pa_sqrt_m > 0.0 else (-1 if K_s_Pa_sqrt_m < 0.0 else 0),
        "compressive_phase": bool(K_s_Pa_sqrt_m < 0.0),
    }


def patch_Q(
    K_s_Pa_sqrt_m: float,
    s_j_m: float,
    r_contact_m: float,
    cfg: CrackRebondingControls,
    T_K: float,
) -> np.ndarray:
    """Build the patch's 3x3 conservative generator at one instant. Exactly
    zero (identity propagator) for REBOND_OFF/CONTACT_PROXY_ONLY, so RB0/RB1
    never alter bonded state — RB1's diagnostics are computed separately via
    ``contact_diagnostics`` and archived without feeding back here."""
    if cfg.model_level not in _ACTIVE_MODEL_LEVELS:
        return build_Q(0.0, 0.0, 0.0, 0.0)

    diag = contact_diagnostics(K_s_Pa_sqrt_m, s_j_m, r_contact_m, cfg)
    lam_bond_raw, _ = bond_formation_rate(diag["sigma_comp_Pa"], T_K, cfg.chemistry_factor, cfg)
    k_CB = cooperative_hazard(lam_bond_raw, cfg.healing_cooperative_order, cfg.healing_correlation_time_s)
    k_BC, _ = bond_rupture_rate(diag["sigma_open_Pa"], T_K, cfg, compressive_phase=diag["compressive_phase"])

    if cfg.model_level is RebondModelLevel.PASSIVATION_GATED_REBOND:
        k_PC, _ = depassivation_rate(diag["sigma_comp_Pa"], T_K, cfg, compressive_phase=diag["compressive_phase"])
        k_CP, _ = repassivation_rate(T_K, cfg, K_s_sign=diag["K_s_sign"])
    else:
        k_PC = 0.0
        k_CP = 0.0

    return build_Q(k_CB=k_CB, k_BC=k_BC, k_PC=k_PC, k_CP=k_CP)


@dataclass
class WakePatch:
    patch_id: int
    length_m: float
    s_j_m: float
    p_P: float
    p_C: float
    p_B: float
    creation_event_index: int
    age_s: float = 0.0
    retired: bool = False

    def state_vector(self) -> np.ndarray:
        return np.array([self.p_P, self.p_C, self.p_B], dtype=float)

    def set_state(self, p: np.ndarray) -> None:
        total = float(p[0] + p[1] + p[2])
        if total > 1.0e-12:
            p = np.asarray(p, dtype=float) / total
        self.p_P, self.p_C, self.p_B = (
            float(max(p[0], 0.0)),
            float(max(p[1], 0.0)),
            float(max(p[2], 0.0)),
        )


class RebondingWakeState:
    """Segment ledger of crack-wake patches and the current cohesive-coupling
    ledger (H_b, K_rebond). Owns no engine reference; all engine-specific
    quantities (K signed waveform, r_eff, K_shield, E') are passed in by the
    caller at each use site.

    Chronological phase semantics (round-3 review correction): the existing
    (pre-rebonding) solver's cleavage/emission channels sample a fixed,
    block-invariant phase array every block (``FatigueCycleHazardController
    ._phases()`` is a pure function of ``n_phase`` alone, unrelated to
    elapsed physical time -- confirmed in
    docs/v10_2_30_crack_rebonding_equation_lineage.md), and that convention
    must not change (existing mechanics unchanged). The wake's OWN notion of
    "where in the loading cycle we are," however, is new physics this module
    owns, and the compression-formation / opening-rupture asymmetry the
    mission specifies is only physically meaningful if that clock is
    continuous across blocks and events -- a patch created mid-cycle must
    evolve through the *actual remaining fraction* of that cycle, not a
    fresh archetypal cycle restarted at phase zero. ``elapsed_time_s`` is
    that persistent chronological clock (mod the waveform period), advanced
    by the caller after every committed block or event and included in
    snapshot/restore and checkpoint round-trips. It shifts only the phase
    ARRAY passed to the wake's own K(phase) sampling (see
    ``kinetic_tip_cell.py::cycle_step_waveform``); ``Kvals``/``sig`` driving
    cleavage/emission are computed from the unshifted array exactly as
    before, so this is purely additive and never perturbs those channels.
    """

    def __init__(self, cfg: CrackRebondingControls):
        self.cfg = cfg
        self.active: list[WakePatch] = []
        self.retired: list[WakePatch] = []
        self._next_patch_id = 0
        self.total_created_length_m = 0.0
        self.formation_action_total = 0.0
        self.rupture_action_total = 0.0
        self.depassivation_action_total = 0.0
        self.repassivation_action_total = 0.0
        self.H_b = 0.0
        self.K_rebond_max_Pa_sqrt_m = 0.0
        self.K_rebond_Pa_sqrt_m = 0.0
        self.elapsed_time_s = 0.0
        self._seed_initial_precrack_patch()

    def _allocate_patch_id(self) -> int:
        pid = self._next_patch_id
        self._next_patch_id += 1
        return pid

    def _seed_initial_precrack_patch(self) -> None:
        mode = self.cfg.initial_precrack_wake_mode
        if mode is InitialPrecrackWakeMode.NO_INITIAL_ACTIVE_WAKE:
            return
        if mode is InitialPrecrackWakeMode.INITIAL_WAKE_CLEAN:
            p_C, p_P = 1.0, 0.0
        else:
            p_C, p_P = 0.0, 1.0
        self.active.append(
            WakePatch(
                patch_id=self._allocate_patch_id(),
                length_m=0.0,
                s_j_m=0.0,
                p_P=p_P,
                p_C=p_C,
                p_B=0.0,
                creation_event_index=-1,
            )
        )

    # -- snapshot/restore for transactional rollback -----------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "active": [dataclasses.replace(p) for p in self.active],
            "retired": [dataclasses.replace(p) for p in self.retired],
            "next_patch_id": self._next_patch_id,
            "total_created_length_m": self.total_created_length_m,
            "formation_action_total": self.formation_action_total,
            "rupture_action_total": self.rupture_action_total,
            "depassivation_action_total": self.depassivation_action_total,
            "repassivation_action_total": self.repassivation_action_total,
            "H_b": self.H_b,
            "K_rebond_max_Pa_sqrt_m": self.K_rebond_max_Pa_sqrt_m,
            "K_rebond_Pa_sqrt_m": self.K_rebond_Pa_sqrt_m,
            "elapsed_time_s": self.elapsed_time_s,
        }

    def restore(self, snap: dict[str, Any] | None) -> None:
        if snap is None:
            return
        self.active = [dataclasses.replace(p) for p in snap["active"]]
        self.retired = [dataclasses.replace(p) for p in snap["retired"]]
        self._next_patch_id = snap["next_patch_id"]
        self.total_created_length_m = snap["total_created_length_m"]
        self.formation_action_total = snap["formation_action_total"]
        self.rupture_action_total = snap["rupture_action_total"]
        self.depassivation_action_total = snap["depassivation_action_total"]
        self.repassivation_action_total = snap["repassivation_action_total"]
        self.H_b = snap["H_b"]
        self.K_rebond_max_Pa_sqrt_m = snap["K_rebond_max_Pa_sqrt_m"]
        self.K_rebond_Pa_sqrt_m = snap["K_rebond_Pa_sqrt_m"]
        self.elapsed_time_s = snap.get("elapsed_time_s", 0.0)

    # -- coupling ------------------------------------------------------------
    def rebuild_coupling(self, Eprime_Pa: float) -> None:
        L_h = self.cfg.wake_length_m
        L_w = self.cfg.wake_weight_length_m
        total = 0.0
        for patch in self.active:
            if patch.retired:
                continue
            w = wake_weight(patch.s_j_m, L_h, L_w)
            total += patch.p_B * w * patch.length_m
        H_b = min(1.0, max(total, 0.0))
        G_max = self.cfg.restored_work_of_separation_J_m2
        eta_K = self.cfg.rebond_K_geometry_factor
        K_rebond_max = eta_K * math.sqrt(max(Eprime_Pa, 0.0) * max(G_max, 0.0))
        self.H_b = H_b
        self.K_rebond_max_Pa_sqrt_m = K_rebond_max
        self.K_rebond_Pa_sqrt_m = K_rebond_max * H_b

    def bonded_fraction_for_state(self, patch_states: dict[int, np.ndarray]) -> float:
        """H_b evaluated at an arbitrary trial P/C/B state per patch (used by
        the event-time root-finder for trial durations, without mutating the
        committed wake state)."""
        L_h = self.cfg.wake_length_m
        L_w = self.cfg.wake_weight_length_m
        total = 0.0
        for patch in self.active:
            if patch.retired:
                continue
            p = patch_states.get(patch.patch_id, patch.state_vector())
            w = wake_weight(patch.s_j_m, L_h, L_w)
            total += float(p[2]) * w * patch.length_m
        return min(1.0, max(total, 0.0))

    # -- transaction: event commit -------------------------------------------
    def commit_event(
        self,
        *,
        accepted_length_m: float,
        event_index: int,
        pre_event_states: dict[int, np.ndarray] | None,
        Eprime_Pa: float,
    ) -> None:
        """Translate pre-existing active patches by the accepted length,
        applying ``pre_event_states`` (the exact state at the true firing
        instant t_e, from the event-time root-finder) if supplied, then
        create the new patch fresh (never inheriting pre-event bonding) and
        retire patches beyond wake_length_m."""
        for patch in self.active:
            if patch.retired:
                continue
            if pre_event_states is not None and patch.patch_id in pre_event_states:
                patch.set_state(pre_event_states[patch.patch_id])
            patch.s_j_m += float(accepted_length_m)

        new_patch = WakePatch(
            patch_id=self._allocate_patch_id(),
            length_m=float(accepted_length_m),
            s_j_m=0.0,
            p_P=1.0 - self.cfg.fresh_surface_clean_fraction,
            p_C=self.cfg.fresh_surface_clean_fraction,
            p_B=0.0,
            creation_event_index=int(event_index),
        )
        self.active.append(new_patch)
        self.total_created_length_m += float(accepted_length_m)

        self._retire_beyond_wake_length()
        self.rebuild_coupling(Eprime_Pa)

    def commit_no_event_block(self, end_states: dict[int, np.ndarray], Eprime_Pa: float) -> None:
        """Finalize a block that did not fire: apply the exact full-block end
        state to every active patch (no length translation, no new patch)."""
        for patch in self.active:
            if patch.retired:
                continue
            if patch.patch_id in end_states:
                patch.set_state(end_states[patch.patch_id])
        self.rebuild_coupling(Eprime_Pa)

    def _retire_beyond_wake_length(self) -> None:
        still_active: list[WakePatch] = []
        for patch in self.active:
            if not patch.retired and patch.s_j_m > self.cfg.wake_length_m:
                patch.retired = True
                self.retired.append(patch)
            else:
                still_active.append(patch)
        self.active = still_active

    def active_length_m(self) -> float:
        return sum(p.length_m for p in self.active if not p.retired)

    def retired_length_m(self) -> float:
        return sum(p.length_m for p in self.retired)


def cleavage_stress_with_rebond(K_Pa_sqrt_m: float, K_shield_Pa_sqrt_m: float, K_rebond_Pa_sqrt_m: float, r_eff_m: float) -> float:
    """sigma_c = [K+ - K_shield - K_rebond]_+ / sqrt(2*pi*r_eff); the HAZARD_ONLY
    injection: mirrors ``sigma_tip``'s exact arithmetic (unified_front.py:84-89)
    with one extra subtraction, called only from the guarded cyclic-path call
    site -- never installed as an override of ``sigma_tip`` itself."""
    K_eff = max(float(K_Pa_sqrt_m) - float(K_shield_Pa_sqrt_m) - float(K_rebond_Pa_sqrt_m), 0.0)
    return K_eff / math.sqrt(2.0 * math.pi * max(float(r_eff_m), 1.0e-30))


# ---------------------------------------------------------------------------
# Phase-resolved representative-cycle construction (block hazard bias) and
# the rebonding-coupled event-time root-finder.
# ---------------------------------------------------------------------------


def build_patch_phase_generators(
    patch: WakePatch, K_phase: np.ndarray, r_contact_m: float, cfg: CrackRebondingControls, T_K: float
) -> list[np.ndarray]:
    return [patch_Q(float(K), patch.s_j_m, r_contact_m, cfg, T_K) for K in K_phase]


def representative_cycle_K_rebond(
    *,
    active_patches: list[WakePatch],
    patch_states: dict[int, np.ndarray],
    K_phase: np.ndarray,
    dt_phase: float,
    r_contact_m: float,
    cfg: CrackRebondingControls,
    T_K: float,
    Eprime_Pa: float,
) -> np.ndarray:
    """One Strang-split representative cycle, starting from each patch's given
    state, producing a length-N_phase K_rebond(phase) array (never a single
    scalar). This is a trial/provisional evaluation: it does not mutate the
    caller's committed wake state."""
    n_phase = len(K_phase)
    L_h = cfg.wake_length_m
    L_w = cfg.wake_weight_length_m
    G_max = cfg.restored_work_of_separation_J_m2
    eta_K = cfg.rebond_K_geometry_factor
    K_rebond_max = eta_K * math.sqrt(max(Eprime_Pa, 0.0) * max(G_max, 0.0))

    p_by_patch = {p.patch_id: patch_states.get(p.patch_id, p.state_vector()).copy() for p in active_patches}
    K_rebond_phase = np.zeros(n_phase)
    for k in range(n_phase):
        K_s = float(K_phase[k])
        total_bonded_weighted = 0.0
        for patch in active_patches:
            Q = patch_Q(K_s, patch.s_j_m, r_contact_m, cfg, T_K)
            half = expm(Q * (0.5 * dt_phase))
            p_mid = half @ p_by_patch[patch.patch_id]
            w = wake_weight(patch.s_j_m, L_h, L_w)
            total_bonded_weighted += p_mid[2] * w * patch.length_m
            p_by_patch[patch.patch_id] = half @ p_mid
        H_b_k = min(1.0, max(total_bonded_weighted, 0.0))
        K_rebond_phase[k] = K_rebond_max * H_b_k
    return K_rebond_phase


def phase_resolved_action(
    *,
    active_patches: list[WakePatch],
    patch_states: dict[int, np.ndarray],
    k0: int,
    t_interval: float,
    K_phase_fn: Callable[[int], float],
    dt_phase: float,
    n_phase: int,
    r_contact_m: float,
    cfg: CrackRebondingControls,
    T_K: float,
    Eprime_Pa: float,
    K_shield_Pa_sqrt_m: float,
    r_eff_m: float,
    lambda_cleave_fn: Callable[[float], float],
) -> tuple[float, dict[int, np.ndarray], int]:
    """Delta B_c(t_interval) = integral_0^t_interval lambda_c[K(tau), K_rebond(tau)] dtau,
    exact for the piecewise-constant phase discretization, starting from
    ``patch_states`` at phase index ``k0``. Returns (action, end_states, end_phase_index).

    This is the phase-resolved evaluator required by the event-time coupling
    guardrail: no block-midpoint or block-constant shortcut is used here.

    KNOWN LIMITATION: this steps every phase bin explicitly, O(n_phase *
    n_cycles), unlike the exact O(log n_cycles) ``propagate`` state
    propagator in the kinetics module. That is acceptable for the interval
    widths exercised by the event-time root-finder and the Stage 2 block
    verifier in this pass (bounded by the candidate block, not by the full
    VHCF horizon), but a production block spanning billions of cycles with
    no event firing would need a representative-cycle bulk approximation
    (one Strang cycle's action reused for the repeated-cycle portion, exact
    stepping only for the leading/trailing partial cycles) before this
    function is called on such an interval. Not implemented in this pass;
    documented here and in the equation-lineage doc as follow-up work.
    """
    L_h = cfg.wake_length_m
    L_w = cfg.wake_weight_length_m
    G_max = cfg.restored_work_of_separation_J_m2
    eta_K = cfg.rebond_K_geometry_factor
    K_rebond_max = eta_K * math.sqrt(max(Eprime_Pa, 0.0) * max(G_max, 0.0))

    p_by_patch = {p.patch_id: patch_states.get(p.patch_id, p.state_vector()).copy() for p in active_patches}

    steps_f = t_interval / dt_phase
    n_full = int(math.floor(steps_f + 1.0e-9))
    frac = max(steps_f - n_full, 0.0)
    if frac > 1.0 - 1.0e-9:
        n_full += 1
        frac = 0.0

    total_action = 0.0
    idx = k0 % n_phase

    def _one_bin(K_s: float, bin_frac: float) -> None:
        nonlocal total_action
        total_weighted = 0.0
        new_states: dict[int, np.ndarray] = {}
        for patch in active_patches:
            Q = patch_Q(K_s, patch.s_j_m, r_contact_m, cfg, T_K)
            half = expm(Q * (0.5 * bin_frac * dt_phase))
            p_mid = half @ p_by_patch[patch.patch_id]
            w = wake_weight(patch.s_j_m, L_h, L_w)
            total_weighted += p_mid[2] * w * patch.length_m
            new_states[patch.patch_id] = half @ p_mid
        H_b_bin = min(1.0, max(total_weighted, 0.0))
        K_rebond_bin = K_rebond_max * H_b_bin
        sigma_c = cleavage_stress_with_rebond(K_s, K_shield_Pa_sqrt_m, K_rebond_bin, r_eff_m)
        lam_c = lambda_cleave_fn(sigma_c)
        total_action += lam_c * (bin_frac * dt_phase)
        p_by_patch.update(new_states)

    for _ in range(n_full):
        K_s = K_phase_fn(idx)
        _one_bin(K_s, 1.0)
        idx = (idx + 1) % n_phase

    if frac > 1.0e-12:
        K_s = K_phase_fn(idx)
        _one_bin(K_s, frac)

    return total_action, p_by_patch, idx


def solve_coupled_event_time(
    *,
    integrate_coupled_fn: Callable[[float], dict[str, Any]],
    phase_resolved_action_fn: Callable[[float], tuple[float, dict[int, np.ndarray], int]],
    lambda_avg_uncoupled: float,
    B_start: float,
    B_threshold: float,
    eps_B: float,
    dt_block: float | None = None,
    max_iterations: int = 60,
) -> dict[str, Any]:
    """Root-find the true event time under the phase-resolved, rebonding-biased
    cleavage action, then use the existing ``_integrate_coupled``-style
    transactional integrator only to obtain a self-consistent bookkeeping call
    at the converged rate -- never trusting a block-constant guess as the
    physical model on its own.

    Since every cleavage rate is non-negative, ``B_start + action(dt)`` is
    monotonically non-decreasing in ``dt``; bisection on ``dt`` against the
    genuine phase-resolved action (``phase_resolved_action_fn``) is therefore
    both correct and numerically stable, unlike a naive fixed-point rescaling
    of the equivalent constant rate, which can diverge for super-linear
    time-dependence. The existing ``integrate_coupled_fn`` (whatever calls the
    production ``_integrate_coupled``) is used exactly once, first as the
    initial predictor/bracket seed (its uncoupled ``dt_consumed``), and once
    more at the end purely to perform the real engine's transactional
    bookkeeping (advance B, draw the next threshold, etc.) at the converged,
    self-consistent equivalent rate -- this final call's own ``dt_consumed``
    is algebraically guaranteed to reproduce the converged ``dt_used``
    (``lambda_avg = action(dt_used)/dt_used`` by construction), so no further
    iteration is required once the phase-resolved root is found.
    """
    baseline = integrate_coupled_fn(max(float(lambda_avg_uncoupled), 1.0e-300))
    if not baseline.get("fired", False):
        return {"fired": False, "iterations": 0}

    def residual(dt: float) -> tuple[float, float, dict[int, np.ndarray], int]:
        action, end_states, end_idx = phase_resolved_action_fn(dt)
        return (B_start + action) - B_threshold, action, end_states, end_idx

    dt_lo = 0.0
    dt_hi = max(float(baseline["dt_consumed"]), float(dt_block or 0.0), 1.0e-300)
    g_hi, _, _, _ = residual(dt_hi)
    expansions = 0
    while g_hi < 0.0 and expansions < 60:
        dt_hi *= 2.0
        g_hi, _, _, _ = residual(dt_hi)
        expansions += 1
    if g_hi < 0.0:
        return {"fired": False, "iterations": expansions}

    dt_mid = dt_hi
    g_mid = g_hi
    action = 0.0
    end_states: dict[int, np.ndarray] = {}
    end_idx = 0
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        dt_mid = 0.5 * (dt_lo + dt_hi)
        g_mid, action, end_states, end_idx = residual(dt_mid)
        if abs(g_mid) <= eps_B:
            break
        if g_mid < 0.0:
            dt_lo = dt_mid
        else:
            dt_hi = dt_mid

    lambda_avg_final = action / max(dt_mid, 1.0e-300)
    confirm = integrate_coupled_fn(lambda_avg_final)

    return {
        "fired": True,
        "dt_used": dt_mid,
        "converged": abs(g_mid) <= eps_B,
        "iterations": iterations,
        "patch_states": end_states,
        "final_phase_index": end_idx,
        "action_closure_residual": g_mid,
        "lambda_avg_used": lambda_avg_final,
        "integrate_coupled_result": confirm,
    }


# ---------------------------------------------------------------------------
# Block-size limiter (Stage 1 cheap preliminary + Stage 2 exact verification)
# ---------------------------------------------------------------------------


def stage1_block_cycle_limits(
    *,
    active_patches: list[WakePatch],
    patch_states: dict[int, np.ndarray],
    K_s0: float,
    r_contact_m: float,
    cfg: CrackRebondingControls,
    T_K: float,
    Eprime_Pa: float,
    period_s: float,
) -> list[float]:
    """Cheap, unit-corrected (seconds -> cycles) linearized candidate limits
    from the block-start local rate, folded into the caller's existing
    cycle-count ``limits`` list before ``cycles`` is finalized."""
    limits: list[float] = []
    L_h = cfg.wake_length_m
    L_w = cfg.wake_weight_length_m
    G_max = cfg.restored_work_of_separation_J_m2
    eta_K = cfg.rebond_K_geometry_factor
    K_rebond_max = eta_K * math.sqrt(max(Eprime_Pa, 0.0) * max(G_max, 0.0))

    dHb_dt = 0.0
    max_dpB_dt = 0.0
    max_dpC_dt = 0.0
    for patch in active_patches:
        p = patch_states.get(patch.patch_id, patch.state_vector())
        Q = patch_Q(K_s0, patch.s_j_m, r_contact_m, cfg, T_K)
        dpdt = Q @ p
        w = wake_weight(patch.s_j_m, L_h, L_w)
        dHb_dt += dpdt[2] * w * patch.length_m
        max_dpB_dt = max(max_dpB_dt, abs(dpdt[2]))
        max_dpC_dt = max(max_dpC_dt, abs(dpdt[1]))

    floor = 1.0e-300
    if max_dpB_dt > floor:
        limits.append((cfg.rebonding_block_max_dpB / max_dpB_dt) / period_s)
    if max_dpC_dt > floor:
        limits.append((cfg.rebonding_block_max_dpC / max_dpC_dt) / period_s)
    dK_rebond_dt = abs(dHb_dt) * K_rebond_max
    K_scale = max(K_rebond_max, 1.0)
    if dK_rebond_dt > floor:
        limits.append((cfg.rebonding_block_max_dK_rebond_frac * K_scale / dK_rebond_dt) / period_s)
    return limits


def stage2_verify_block(
    *,
    active_patches: list[WakePatch],
    patch_states: dict[int, np.ndarray],
    k0: int,
    dt_candidate: float,
    K_phase_fn: Callable[[int], float],
    dt_phase: float,
    n_phase: int,
    r_contact_m: float,
    cfg: CrackRebondingControls,
    T_K: float,
    Eprime_Pa: float,
    K_shield_Pa_sqrt_m: float,
    r_eff_m: float,
    lambda_cleave_fn: Callable[[float], float],
    representative_action_estimate: float,
) -> dict[str, Any]:
    """Exact trial propagation of the candidate block; returns actual deltas
    and a pass/fail verdict the caller should bisect on when failed."""
    action_exact, end_states, end_idx = phase_resolved_action(
        active_patches=active_patches,
        patch_states=patch_states,
        k0=k0,
        t_interval=dt_candidate,
        K_phase_fn=K_phase_fn,
        dt_phase=dt_phase,
        n_phase=n_phase,
        r_contact_m=r_contact_m,
        cfg=cfg,
        T_K=T_K,
        Eprime_Pa=Eprime_Pa,
        K_shield_Pa_sqrt_m=K_shield_Pa_sqrt_m,
        r_eff_m=r_eff_m,
        lambda_cleave_fn=lambda_cleave_fn,
    )

    max_dpB = 0.0
    max_dpC = 0.0
    for patch in active_patches:
        p0 = patch_states.get(patch.patch_id, patch.state_vector())
        p1 = end_states.get(patch.patch_id, p0)
        max_dpB = max(max_dpB, abs(float(p1[2] - p0[2])))
        max_dpC = max(max_dpC, abs(float(p1[1] - p0[1])))

    L_h, L_w = cfg.wake_length_m, cfg.wake_weight_length_m
    G_max, eta_K = cfg.restored_work_of_separation_J_m2, cfg.rebond_K_geometry_factor
    K_rebond_max = eta_K * math.sqrt(max(Eprime_Pa, 0.0) * max(G_max, 0.0))

    def _H_b(states: dict[int, np.ndarray]) -> float:
        total = 0.0
        for patch in active_patches:
            p = states.get(patch.patch_id, patch.state_vector())
            w = wake_weight(patch.s_j_m, L_h, L_w)
            total += float(p[2]) * w * patch.length_m
        return min(1.0, max(total, 0.0))

    H_b_start = _H_b(patch_states)
    H_b_end = _H_b(end_states)
    dK_rebond = abs(K_rebond_max * (H_b_end - H_b_start))
    K_scale = max(K_rebond_max, 1.0)

    action_error = abs(action_exact - representative_action_estimate) / max(
        abs(representative_action_estimate), abs(action_exact), 1.0e-300
    )

    passed = (
        max_dpB <= cfg.rebonding_block_max_dpB
        and max_dpC <= cfg.rebonding_block_max_dpC
        and dK_rebond <= cfg.rebonding_block_max_dK_rebond_frac * K_scale
        and action_error <= cfg.rebonding_block_action_consistency_tol
    )

    return {
        "passed": passed,
        "max_dpB": max_dpB,
        "max_dpC": max_dpC,
        "dK_rebond": dK_rebond,
        "action_error": action_error,
        "action_exact": action_exact,
        "end_states": end_states,
        "end_phase_index": end_idx,
    }


# ---------------------------------------------------------------------------
# Installer, config-hash idempotency, checkpoint round trip
# ---------------------------------------------------------------------------


def install_crack_rebonding(engine: Any, cfg: CrackRebondingControls) -> None:
    """The single authoritative installer, called only when cfg.enabled. Only
    ever invoked from ``reduced_shared_state_v1023.build_shared_engine`` in the
    production path; idempotent in general as defense in depth."""
    cfg = cfg.validate()
    if not cfg.enabled:
        return

    existing = getattr(engine, "_rebonding_state", None)
    existing_cfg = getattr(engine, "_rebonding_cfg", None)
    if existing is not None:
        if existing_cfg is not None and existing_cfg.config_hash() == cfg.config_hash():
            return
        raise RuntimeError(
            "crack rebonding already installed on this engine with a different "
            "configuration; refusing a silent double install"
        )

    # cfg.validate() (above) already raises NotImplementedError for any
    # feedback_mode other than HAZARD_ONLY_REBOND_SHIELD, before any engine
    # attribute is touched -- nothing further to enforce here.
    engine._rebonding_state = RebondingWakeState(cfg)
    engine._rebonding_cfg = cfg
    engine.rebonding_acceleration_qualified = False
    engine.contact_semantics = CONTACT_SEMANTICS_LABEL


def serialize_rebonding_checkpoint(engine: Any) -> dict[str, Any] | None:
    state: RebondingWakeState | None = getattr(engine, "_rebonding_state", None)
    if state is None:
        return None

    def _patch_dict(p: WakePatch) -> dict[str, Any]:
        return {
            "patch_id": p.patch_id,
            "length_m": p.length_m,
            "s_j_m": p.s_j_m,
            "p_P": p.p_P,
            "p_C": p.p_C,
            "p_B": p.p_B,
            "creation_event_index": p.creation_event_index,
            "age_s": p.age_s,
            "retired": p.retired,
        }

    return {
        "schema": "v10.2.30_crack_rebonding_checkpoint_v1",
        "active": [_patch_dict(p) for p in state.active],
        "retired": [_patch_dict(p) for p in state.retired],
        "next_patch_id": state._next_patch_id,
        "total_created_length_m": state.total_created_length_m,
        "formation_action_total": state.formation_action_total,
        "rupture_action_total": state.rupture_action_total,
        "depassivation_action_total": state.depassivation_action_total,
        "repassivation_action_total": state.repassivation_action_total,
        "H_b": state.H_b,
        "K_rebond_max_Pa_sqrt_m": state.K_rebond_max_Pa_sqrt_m,
        "K_rebond_Pa_sqrt_m": state.K_rebond_Pa_sqrt_m,
        "elapsed_time_s": state.elapsed_time_s,
        "config_hash": state.cfg.config_hash(),
    }


def restore_rebonding_checkpoint(engine: Any, payload: dict[str, Any] | None) -> None:
    state: RebondingWakeState | None = getattr(engine, "_rebonding_state", None)
    if state is None or payload is None:
        return

    def _patch_from_dict(d: dict[str, Any]) -> WakePatch:
        return WakePatch(
            patch_id=d["patch_id"],
            length_m=d["length_m"],
            s_j_m=d["s_j_m"],
            p_P=d["p_P"],
            p_C=d["p_C"],
            p_B=d["p_B"],
            creation_event_index=d["creation_event_index"],
            age_s=d.get("age_s", 0.0),
            retired=d.get("retired", False),
        )

    state.active = [_patch_from_dict(d) for d in payload.get("active", [])]
    state.retired = [_patch_from_dict(d) for d in payload.get("retired", [])]
    state._next_patch_id = payload.get("next_patch_id", state._next_patch_id)
    state.total_created_length_m = payload.get("total_created_length_m", 0.0)
    state.formation_action_total = payload.get("formation_action_total", 0.0)
    state.rupture_action_total = payload.get("rupture_action_total", 0.0)
    state.depassivation_action_total = payload.get("depassivation_action_total", 0.0)
    state.repassivation_action_total = payload.get("repassivation_action_total", 0.0)
    state.H_b = payload.get("H_b", 0.0)
    state.K_rebond_max_Pa_sqrt_m = payload.get("K_rebond_max_Pa_sqrt_m", 0.0)
    state.K_rebond_Pa_sqrt_m = payload.get("K_rebond_Pa_sqrt_m", 0.0)
    state.elapsed_time_s = payload.get("elapsed_time_s", 0.0)


__all__ = [
    "reduced_modulus_Pa",
    "wake_weight",
    "contact_diagnostics",
    "patch_Q",
    "WakePatch",
    "RebondingWakeState",
    "cleavage_stress_with_rebond",
    "build_patch_phase_generators",
    "representative_cycle_K_rebond",
    "phase_resolved_action",
    "solve_coupled_event_time",
    "stage1_block_cycle_limits",
    "stage2_verify_block",
    "install_crack_rebonding",
    "serialize_rebonding_checkpoint",
    "restore_rebonding_checkpoint",
]
