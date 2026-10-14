"""Diagnostic-only factorization of historical and native Taylor--Peierls laws.

This module deliberately does not expose a runtime selector.  It evaluates
saved material rows at frozen states and keeps the two historical consumers
separate because the old moving-tip and bulk implementations were not the same
law.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .material_manifest import KB_EV_PER_K, ExpFloorBarrier
from .taylor_release import (
    completion_rate,
    density_completion_order,
    historical_legacy_implicit_1s,
    signed_arrhenius_components,
)

HISTORICAL_PHI_CAP = 20.0
HISTORICAL_JUMP_FLOOR_M = 2.5e-10
HISTORICAL_RATE_CAP_S = 1.0e12
HISTORICAL_BULK_RENEWAL_S = 1.0e-9
DEFAULT_BURGERS_M = 2.74e-10


def raw_scales(row: dict[str, str], temperature_K: float) -> dict[str, float]:
    """Return unclamped raw scales used to decide whether a state is admissible."""
    T = float(temperature_K)
    dT = T - float(row["Tref_K"])
    emit_sigc = (
        float(row["emit_sigc0_GPa"])
        + float(row["emit_sT_GPa_per_K"]) * dT
    ) * 1.0e9
    return {
        "cleavage_G0_eV": float(row["cleave_G00_eV"])
        + float(row["cleave_gT_eV_per_K"]) * dT,
        "cleavage_sigc_Pa": (
            float(row["cleave_sigc0_GPa"])
            + float(row["cleave_sT_GPa_per_K"]) * dT
        ) * 1.0e9,
        "emission_G0_eV": float(row["emit_G00_eV"])
        + float(row["emit_gT_eV_per_K"]) * dT,
        "emission_sigc_Pa": emit_sigc,
        "peierls_G0_eV": float(row["peierls_H0_eV"])
        - float(row["peierls_activation_entropy_kB"]) * KB_EV_PER_K * dT,
        "peierls_surface_sigc_Pa": emit_sigc,
        "peierls_projected_equivalent_sigc_Pa": (
            float(row["peierls_stress_fraction"]) * emit_sigc
        ),
        "taylor_G0_eV": float(row["taylor_H0_eV"])
        - float(row["taylor_activation_entropy_kB"]) * KB_EV_PER_K * dT,
        "taylor_surface_sigc_Pa": emit_sigc,
        "taylor_projected_equivalent_sigc_Pa": (
            float(row["taylor_stress_fraction"]) * emit_sigc
        ),
    }


def raw_domain_status(row: dict[str, str], temperature_K: float) -> dict[str, Any]:
    values = raw_scales(row, temperature_K)
    invalid = {key: value for key, value in values.items() if value <= 0.0}
    return {
        "temperature_K": float(temperature_K),
        "status": "ADMISSIBLE" if not invalid else "UNSUPPORTED_RAW_SCALE",
        "raw_scales": values,
        "invalid_raw_scales": invalid,
    }


def select_temperature_anchors(
    row: dict[str, str], declared_temperatures_K: tuple[float, ...]
) -> tuple[list[float], list[dict[str, Any]]]:
    """Select low/middle/high grid anchors without evaluating invalid states."""
    domain = [raw_domain_status(row, T) for T in declared_temperatures_K]
    valid = [item["temperature_K"] for item in domain if item["status"] == "ADMISSIBLE"]
    if not valid:
        return [], domain
    anchors = [valid[0], valid[len(valid) // 2], valid[-1]]
    return list(dict.fromkeys(anchors)), domain


def shifted_density_order(rho_m2: float, scale: float, rho_c_m2: float) -> float:
    """Exact order used by the pre-repair moving-tip and bulk consumers."""
    ratio = math.sqrt(max(float(rho_m2), 0.0) / max(float(rho_c_m2), 1.0))
    return 1.0 + max(float(scale), 0.0) * max(ratio - 1.0, 0.0)


def _safe_log_rate(rate_s: float) -> float | None:
    """Log an unsigned elementary rate only; signed net rates never enter here."""
    rate = float(rate_s)
    return math.log(rate) if rate > 0.0 else None


def _barrier_payload(surface: ExpFloorBarrier, stress_Pa: float, T_K: float) -> dict[str, Any]:
    diagnostic = surface.diagnostic(max(float(stress_Pa), 0.0), T_K)
    return {
        "stress_Pa": max(float(stress_Pa), 0.0),
        "barrier_eV": diagnostic["barrier_eV"],
        "floor_eV": diagnostic["floor_eV"],
        "floor_branch": diagnostic["floor_branch"],
        "activation_volume_m3": diagnostic["activation_volume_m3"],
        "activation_entropy_fixed_stress_state_eV_K": diagnostic[
            "activation_entropy_fixed_stress_state_eV_K"
        ],
        "derivative_convention": diagnostic["derivative_convention"],
    }


def _surface(row: dict[str, str], prefix: str) -> ExpFloorBarrier:
    H0 = float(row[f"{prefix}_H0_eV"])
    return ExpFloorBarrier(
        G00_eV=H0,
        gT_eV_per_K=-float(row[f"{prefix}_activation_entropy_kB"]) * KB_EV_PER_K,
        sigc0_Pa=float(row[f"{prefix}_stress_fraction"])
        * float(row["emit_sigc0_GPa"]) * 1.0e9,
        sT_Pa_per_K=float(row[f"{prefix}_stress_fraction"])
        * float(row["emit_sT_GPa_per_K"]) * 1.0e9,
        alpha=float(row[f"{prefix}_exp_a"]),
        exponent=float(row[f"{prefix}_exp_n"]),
        floor_fraction=float(row["emit_floor_frac"]),
        floor_min_eV=1.0e-4 * H0 / max(float(row["emit_G00_eV"]), 1.0e-30),
        floor_max_fraction=0.95,
        Tref_K=float(row["Tref_K"]),
        attempt_frequency_s=float(row[f"{prefix}_nu0_s"]),
    )


def _series_rate(peierls_rate_s: float, completion_rate_s: float) -> float:
    denom = float(peierls_rate_s) + float(completion_rate_s)
    if denom <= 0.0:
        return 0.0
    return float(peierls_rate_s) * float(completion_rate_s) / denom


def evaluate_same_state(
    row: dict[str, str], *, temperature_K: float, signed_local_stress_Pa: float,
    raw_density_m2: float, burgers_m: float = DEFAULT_BURGERS_M,
) -> dict[str, Any]:
    """Evaluate isolated switches plus exact old-tip, old-bulk and current bundles."""
    domain = raw_domain_status(row, temperature_K)
    if domain["status"] != "ADMISSIBLE":
        return {
            "family": row["material_class"],
            "option_key": row["option_key"],
            "temperature_K": float(temperature_K),
            "signed_local_stress_Pa": float(signed_local_stress_Pa),
            "raw_density_m2": float(raw_density_m2),
            "status": "NOT_EVALUATED_UNSUPPORTED_RAW_SCALE",
            "domain": domain,
        }

    rho_floor = float(row["rho_forest_floor_m2"])
    rho = max(float(raw_density_m2), rho_floor, 1.0)
    rho_c = float(row["taylor_corr_rho_c_m2"])
    scale = float(row["taylor_corr_scale"])
    b = abs(float(burgers_m))
    spacing = 1.0 / (2.0 * math.sqrt(rho))
    phi_raw = spacing / max(b, 1.0e-30)
    phi_capped = min(phi_raw, HISTORICAL_PHI_CAP)
    jump_raw = spacing
    jump_floored = max(spacing, HISTORICAL_JUMP_FLOOR_M)
    signed_stress = float(signed_local_stress_Pa)
    p_fraction = float(row["peierls_stress_fraction"])
    t_fraction = float(row["taylor_stress_fraction"])
    p_surface = _surface(row, "peierls")
    t_surface = _surface(row, "taylor")

    p_stress = p_fraction * signed_stress
    t_stress = t_fraction * signed_stress * phi_raw
    t_stress_capped_phi = t_fraction * signed_stress * phi_capped
    p_parts = signed_arrhenius_components(p_surface, p_stress, temperature_K)
    t_parts = signed_arrhenius_components(t_surface, t_stress, temperature_K)
    t_cap_parts = signed_arrhenius_components(
        t_surface, t_stress_capped_phi, temperature_K
    )
    pf = float(p_parts["forward_rate_s"])
    pr = float(p_parts["reverse_rate_s"])
    pn = float(p_parts["net_rate_s"])
    tf = float(t_parts["forward_rate_s"])
    tr = float(t_parts["reverse_rate_s"])
    tn = float(t_parts["net_rate_s"])
    tn_phi_cap = float(t_cap_parts["net_rate_s"])
    order_current = float(density_completion_order(
        rho, correlation_scale=scale, correlation_density_m2=rho_c
    ))
    order_shifted = shifted_density_order(rho, scale, rho_c)

    def variant(name: str, *, taylor_net: float = tn, order: float = order_current,
                completion: str = "native", jump: float = jump_raw) -> dict[str, Any]:
        positive = max(float(taylor_net), 0.0)
        rate_entering = positive
        if completion == "native_rate_cap":
            rate_entering = min(rate_entering, HISTORICAL_RATE_CAP_S)
            completed = float(completion_rate(rate_entering, order))
        elif completion == "historical_1s":
            completed = float(historical_legacy_implicit_1s(
                rate_entering, order, rate_cap_s=HISTORICAL_RATE_CAP_S
            ))
        elif completion == "historical_bulk_1ns":
            exposure = min(rate_entering * HISTORICAL_BULK_RENEWAL_S, 1.0e12)
            from scipy.special import gammainc
            completed = float(gammainc(max(order, 1.0), exposure)
                              / HISTORICAL_BULK_RENEWAL_S)
        else:
            completed = float(completion_rate(rate_entering, order))
        series = _series_rate(max(pn, 0.0), completed)
        return {
            "name": name,
            "taylor_net_rate_s": float(taylor_net),
            "positive_taylor_rate_entering_completion_s": positive,
            "rate_after_optional_cap_s": rate_entering,
            "order": order,
            "completion_mapping": completion,
            "completion_rate_s": completed,
            "completion_time_s": 1.0 / completed if completed > 0.0 else None,
            "jump_length_m": jump,
            "peierls_velocity_m_s": jump * pn,
            "positive_series_rate_s": series,
        }

    variants = [
        variant("current_combined"),
        variant("completion_historical_implicit_1s_only", completion="historical_1s"),
        variant("shifted_density_order_only", order=order_shifted),
        variant("phi_cap_20_only", taylor_net=tn_phi_cap),
        variant("elementary_rate_cap_1e12_only", completion="native_rate_cap"),
        variant("jump_floor_2p5e-10_m_only", jump=jump_floored),
    ]

    # Exact pre-repair moving-tip bundle: tensile clipping, one-way elementary
    # rates, shifted order, one-second gamma completion, no phi/jump cap.
    old_tip_stress = max(signed_stress, 0.0)
    old_tip_p = float(p_surface.rate(p_fraction * old_tip_stress, temperature_K))
    old_tip_t = float(t_surface.rate(
        t_fraction * old_tip_stress * phi_raw, temperature_K
    ))
    old_tip_completion = float(historical_legacy_implicit_1s(
        old_tip_t, order_shifted, rate_cap_s=HISTORICAL_RATE_CAP_S
    ))
    historical_tip = {
        "name": "historical_moving_tip_exact",
        "stress_policy": "maximum(local_stress,0); one_way_elementary_rate",
        "peierls_one_way_rate_s": old_tip_p,
        "taylor_one_way_rate_s": old_tip_t,
        "order": order_shifted,
        "phi": phi_raw,
        "jump_length_m": jump_raw,
        "completion_mapping": "historical_legacy_implicit_1s",
        "completion_rate_s": old_tip_completion,
        "completion_time_s": 1.0 / old_tip_completion if old_tip_completion > 0 else None,
        "peierls_velocity_m_s": jump_raw * old_tip_p,
    }

    # Exact old bulk policy at the same scalar state: positive-stress gate,
    # symmetric detailed balance, phi cap, shifted order, 1 ns renewal and jump floor.
    bulk_stress = max(signed_stress, 0.0)
    bulk_p_tau = p_fraction * bulk_stress
    bulk_t_tau = t_fraction * bulk_stress * phi_capped

    def old_bulk_net(surface: ExpFloorBarrier, tau: float) -> tuple[float, float, float]:
        forward_barrier = float(surface.values_eV(tau, temperature_K))
        zero_barrier = float(surface.values_eV(0.0, temperature_K))
        reverse_barrier = max(2.0 * zero_barrier - forward_barrier, zero_barrier)
        kT = max(KB_EV_PER_K * float(temperature_K), 1.0e-30)
        forward = surface.attempt_frequency_s * math.exp(
            max(-745.0, min(0.0, -forward_barrier / kT)))
        reverse = surface.attempt_frequency_s * math.exp(
            max(-745.0, min(0.0, -reverse_barrier / kT)))
        net = max(forward - reverse, 0.0) if bulk_stress != 0.0 else 0.0
        return forward, reverse, net

    old_bpf, old_bpr, old_bpn = old_bulk_net(p_surface, bulk_p_tau)
    old_btf, old_btr, old_btn = old_bulk_net(t_surface, bulk_t_tau)
    old_bulk_completion = variant(
        "historical_bulk_exact", taylor_net=old_btn, order=order_shifted,
        completion="historical_bulk_1ns", jump=jump_floored,
    )
    old_bulk_completion.update({
        "stress_policy": "maximum(equivalent_stress,0); symmetric_detailed_balance",
        "peierls_forward_rate_s": old_bpf,
        "peierls_reverse_rate_s": old_bpr,
        "peierls_net_rate_s": old_bpn,
        "taylor_forward_rate_s": old_btf,
        "taylor_reverse_rate_s": old_btr,
        "phi": phi_capped,
        "peierls_velocity_m_s": jump_floored * old_bpn,
        "positive_series_rate_s": _series_rate(old_bpn, old_bulk_completion["completion_rate_s"]),
    })

    return {
        "family": row["material_class"],
        "option_key": row["option_key"],
        "candidate_id": row["candidate_id"],
        "temperature_K": float(temperature_K),
        "signed_local_stress_Pa": signed_stress,
        "raw_density_m2": float(raw_density_m2),
        "consumed_forest_density_m2": rho,
        "density_floor_was_active": rho != float(raw_density_m2),
        "status": "EVALUATED_ADMISSIBLE_PARENT",
        "domain": domain,
        "geometry": {
            "burgers_m": b,
            "spacing_m": spacing,
            "phi_uncapped": phi_raw,
            "phi_historical_cap": HISTORICAL_PHI_CAP,
            "phi_capped": phi_capped,
            "jump_native_m": jump_raw,
            "jump_historical_floor_m": HISTORICAL_JUMP_FLOOR_M,
            "jump_floored_m": jump_floored,
            "amplification_derivation": "phi=ell_f/b with ell_f=1/(2*sqrt(rho_f)); tau_T=f_T*sigma_local*phi",
            "local_stress_counting": "sigma_local enters once; no additional crack-tip concentration is applied in this diagnostic",
        },
        "orders": {"current_unshifted": order_current, "historical_shifted": order_shifted},
        "current_elementary": {
            "peierls": {
                "projected_signed_stress_Pa": p_stress,
                "forward_rate_s": pf, "reverse_rate_s": pr, "signed_net_rate_s": pn,
                "log_forward_rate": _safe_log_rate(pf),
                "log_reverse_rate": _safe_log_rate(pr),
                "forward_barrier": _barrier_payload(p_surface, max(p_stress, 0.0), temperature_K),
                "reverse_barrier": _barrier_payload(p_surface, max(-p_stress, 0.0), temperature_K),
            },
            "taylor": {
                "projected_signed_stress_Pa": t_stress,
                "forward_rate_s": tf, "reverse_rate_s": tr, "signed_net_rate_s": tn,
                "log_forward_rate": _safe_log_rate(tf),
                "log_reverse_rate": _safe_log_rate(tr),
                "forward_barrier": _barrier_payload(t_surface, max(t_stress, 0.0), temperature_K),
                "reverse_barrier": _barrier_payload(t_surface, max(-t_stress, 0.0), temperature_K),
            },
        },
        "isolated_variants": variants,
        "historical_moving_tip_exact": historical_tip,
        "historical_bulk_exact": old_bulk_completion,
    }


__all__ = [
    "DEFAULT_BURGERS_M", "HISTORICAL_BULK_RENEWAL_S",
    "HISTORICAL_JUMP_FLOOR_M", "HISTORICAL_PHI_CAP",
    "HISTORICAL_RATE_CAP_S", "evaluate_same_state", "raw_domain_status",
    "raw_scales", "select_temperature_anchors", "shifted_density_order",
]
