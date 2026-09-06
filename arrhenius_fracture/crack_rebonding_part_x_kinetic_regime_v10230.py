"""PX2 (Part X): analytical, zero-fitting kinetic-regime characterization.

Evaluates the TRUE periodic-orbit occupancy trajectory, one-cycle
transition actions/fluxes, and barrier-floor/cooperative-saturation
diagnostics for one representative wake patch under a fully-specified
loading schedule (Kmax, R, frequency, minimum-load hold), using only the
existing analytical phase-resolved P/C/B model
(``crack_rebonding_kinetics_v10230``/``crack_rebonding_v10230``) --
never fitting any barrier or rate to da/dN, a desired Paris slope, or an
observed suppression (mission section 6).

Reuses, rather than reimplements, PX1's exact machinery: the
heterogeneous ``propagate``/``build_phase_factors`` (PX1.1) for the exact
one-cycle monodromy, and ``transition_actions_and_fluxes`` (PX1.2, the
augmented-matrix-exponential exact occupancy-time-integral) for the
one-cycle action/flux report -- so this module is a thin composition
layer over already-validated primitives, not a fresh reimplementation.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from .crack_rebonding_kinetics_v10230 import (
    CrackRebondingControls,
    RebondModelLevel,
    _partial_product,
    bond_formation_rate,
    bond_rupture_rate,
    build_Q,
    build_phase_factors,
    cooperative_hazard,
    depassivation_rate,
    propagate,
    repassivation_rate,
    transition_actions_and_fluxes,
)
from .crack_rebonding_v10230 import (
    _ACTIVE_MODEL_LEVELS,
    contact_diagnostics,
    reduced_modulus_Pa,
)
from .fatigue_v1 import FatigueWaveform

MODEL_ID = "v10230_part_x_analytical_periodic_orbit_v1"


def _per_bin_rates_and_diagnostics(
    K_phase: np.ndarray, s_j_m: float, r_contact_m: float, cfg: CrackRebondingControls, T_K: float
) -> dict[str, Any]:
    """Per-bin rate constants and floor/saturation diagnostics for all four
    channels -- the same underlying rate functions patch_rate_constants
    uses, but retaining the per-bin diagnostic dicts those functions
    return (patch_rate_constants discards them, since the real physics
    path never needed them; PX2's analytical characterization does)."""
    n = len(K_phase)
    k_CB = np.zeros(n)
    k_BC = np.zeros(n)
    k_PC = np.zeros(n)
    k_CP = np.zeros(n)
    floored_any = {"CB": False, "BC": False, "PC": False, "CP": False}
    saturated_any = {"CB": False, "BC": False, "PC": False, "CP": False}
    max_floor_fraction = {"CB": 0.0, "BC": 0.0, "PC": 0.0, "CP": 0.0}

    active = cfg.model_level in _ACTIVE_MODEL_LEVELS
    for i, K_s in enumerate(K_phase):
        if not active:
            continue
        diag = contact_diagnostics(float(K_s), s_j_m, r_contact_m, cfg)
        lam_bond_raw, diag_cb = bond_formation_rate(diag["sigma_comp_Pa"], T_K, cfg.chemistry_factor, cfg)
        if diag["compressive_phase"]:
            k_CB[i] = cooperative_hazard(lam_bond_raw, cfg.healing_cooperative_order, cfg.healing_correlation_time_s)
        else:
            k_CB[i] = 0.0
        k_BC[i], diag_bc = bond_rupture_rate(diag["sigma_open_Pa"], T_K, cfg, compressive_phase=diag["compressive_phase"])

        if cfg.model_level is RebondModelLevel.PASSIVATION_GATED_REBOND:
            k_PC[i], diag_pc = depassivation_rate(diag["sigma_comp_Pa"], T_K, cfg, compressive_phase=diag["compressive_phase"])
            k_CP[i], diag_cp = repassivation_rate(T_K, cfg, K_s_sign=diag["K_s_sign"])
        else:
            diag_pc = {"barrier_floor_fraction": 0.0, "floored": False, "saturated": False}
            diag_cp = {"barrier_floor_fraction": 0.0, "floored": False, "saturated": False}

        for key, diag_i in (("CB", diag_cb), ("BC", diag_bc), ("PC", diag_pc), ("CP", diag_cp)):
            floored_any[key] = floored_any[key] or diag_i["floored"]
            saturated_any[key] = saturated_any[key] or diag_i["saturated"]
            max_floor_fraction[key] = max(max_floor_fraction[key], diag_i["barrier_floor_fraction"])

    return {
        "k_CB": k_CB, "k_BC": k_BC, "k_PC": k_PC, "k_CP": k_CP,
        "floored_any": floored_any, "saturated_any": saturated_any,
        "max_barrier_floor_fraction": max_floor_fraction,
    }


def analytical_periodic_orbit(
    *,
    cfg: CrackRebondingControls,
    T_K: float,
    Kmax_Pa_sqrt_m: float,
    R: float,
    frequency_Hz: float,
    minimum_load_hold_s: float = 0.0,
    s_j_m: float = 0.0,
    r_contact_m: float,
    n_phase: int = 64,
    forward_iterations: int = 400,
) -> dict[str, Any]:
    """The true periodic fixed point of the phase-resolved P/C/B chain for
    one representative patch (fixed geometry ``s_j_m``/``r_contact_m``)
    under a fully specified cycle schedule, plus the one-cycle transition
    actions/fluxes at that periodic state and barrier-floor/saturation
    diagnostics -- the complete, zero-fitting analytical characterization
    of a candidate kinetic row at one operating condition.

    ``p_star`` is obtained by repeated squaring of the one-cycle monodromy
    (``matrix_power``, O(log forward_iterations)), landing in whichever
    invariant subspace an arbitrary starting state projects onto --
    the same robust approach ``_periodic_orbit_certificate`` uses for the
    live engine's bulk-cycle acceleration, reused here for a from-scratch
    (no committed trajectory) analytical evaluation.
    """
    waveform = FatigueWaveform(
        Kmax=Kmax_Pa_sqrt_m, R=R, frequency_Hz=frequency_Hz, minimum_load_hold_s=minimum_load_hold_s
    )
    K_phase, dt_array = waveform.cycle_schedule(n_phase, signed=True)
    n_total = len(K_phase)

    rates = _per_bin_rates_and_diagnostics(K_phase, s_j_m, r_contact_m, cfg, T_K)
    Q_list = [
        build_Q(rates["k_CB"][i], rates["k_BC"][i], rates["k_PC"][i], rates["k_CP"][i])
        for i in range(n_total)
    ]
    factors = build_phase_factors(Q_list, dt_array)
    M_cycle, _ = _partial_product(factors, 0, n_total)

    p0 = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    p_star = np.linalg.matrix_power(M_cycle, forward_iterations) @ p0
    p_star = np.clip(p_star, 0.0, None)
    total = float(p_star.sum())
    p_star = p_star / total if total > 1.0e-12 else np.array([1.0, 0.0, 0.0])

    # Verify convergence: one more cycle should not move p_star (within
    # tolerance) -- an explicit, checkable certificate rather than trusting
    # forward_iterations blindly.
    p_star_next = M_cycle @ p_star
    convergence_residual = float(np.linalg.norm(p_star_next - p_star))

    # Phase-resolved trajectory over one cycle starting at the periodic state.
    trajectory = np.zeros((n_total + 1, 3))
    trajectory[0] = p_star
    p = p_star.copy()
    for i in range(n_total):
        p = factors[i] @ p
        trajectory[i + 1] = p

    total_cycle_s = float(dt_array.sum())
    af = transition_actions_and_fluxes(
        p_star, Q_list, {"CB": rates["k_CB"], "BC": rates["k_BC"], "PC": rates["k_PC"], "CP": rates["k_CP"]},
        k0=0, dt=total_cycle_s, dt_phase=dt_array,
    )

    # Duration-weighted cycle-mean occupancy (bin i's contribution uses the
    # trajectory value AT THE START of that bin, i.e. trajectory[i], since
    # that is the state held during that bin's dt in this piecewise-
    # constant-generator discretization).
    p_B_per_bin = trajectory[:-1, 2]
    p_C_per_bin = trajectory[:-1, 1]
    p_P_per_bin = trajectory[:-1, 0]
    mean_p_B = float(np.sum(p_B_per_bin * dt_array) / total_cycle_s)
    mean_p_C = float(np.sum(p_C_per_bin * dt_array) / total_cycle_s)
    mean_p_P = float(np.sum(p_P_per_bin * dt_array) / total_cycle_s)

    n_phase_only = n_phase  # sinusoidal-only bin count (dwell, if any, is the extra bin)
    contact_time_s = float(
        np.sum(dt_array[np.asarray([K_phase[i] < 0.0 for i in range(n_total)])])
    )
    contact_time_sinusoid_s = float(
        np.sum(dt_array[:n_phase_only][K_phase[:n_phase_only] < 0.0])
    )
    contact_time_dwell_s = contact_time_s - contact_time_sinusoid_s

    return {
        "schema": MODEL_ID,
        "conditions": {
            "Kmax_Pa_sqrt_m": float(Kmax_Pa_sqrt_m), "R": float(R),
            "frequency_Hz": float(frequency_Hz), "minimum_load_hold_s": float(minimum_load_hold_s),
            "T_K": float(T_K), "n_phase": int(n_phase), "s_j_m": float(s_j_m),
            "r_contact_m": float(r_contact_m), "chemistry_factor": float(cfg.chemistry_factor),
        },
        "p_star": p_star, "trajectory": trajectory,
        "convergence_residual": convergence_residual,
        "mean_p_P": mean_p_P, "mean_p_C": mean_p_C, "mean_p_B": mean_p_B,
        "max_p_B": float(np.max(trajectory[:, 2])), "min_p_B": float(np.min(trajectory[:, 2])),
        "A_CB": af["A_CB"], "A_BC": af["A_BC"], "A_PC": af["A_PC"], "A_CP": af["A_CP"],
        "F_CB": af["F_CB"], "F_BC": af["F_BC"], "F_PC": af["F_PC"], "F_CP": af["F_CP"],
        "contact_time_s": contact_time_s,
        "contact_time_sinusoid_s": contact_time_sinusoid_s,
        "contact_time_dwell_s": contact_time_dwell_s,
        "total_cycle_s": total_cycle_s,
        "barrier_floor_fraction": rates["max_barrier_floor_fraction"],
        "any_floored": rates["floored_any"],
        "any_saturated": rates["saturated_any"],
    }


def predicted_static_shield_equivalent_K_b(
    periodic_orbit: dict[str, Any], cfg: CrackRebondingControls, Eprime_Pa: float
) -> float:
    """K_rebond,max * (cycle-mean bonded occupancy) -- the frozen
    periodic-orbit-matched static control value PX5 needs, computed here
    from the analytical periodic orbit alone (never fit to physical
    S_h)."""
    eta_K = cfg.rebond_K_geometry_factor
    G_max = cfg.restored_work_of_separation_J_m2
    K_rebond_max = eta_K * math.sqrt(max(Eprime_Pa, 0.0) * max(G_max, 0.0))
    return float(K_rebond_max * periodic_orbit["mean_p_B"])


__all__ = [
    "MODEL_ID",
    "analytical_periodic_orbit",
    "predicted_static_shield_equivalent_K_b",
]
