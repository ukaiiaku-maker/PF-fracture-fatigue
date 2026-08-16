#!/usr/bin/env python3
"""Derivative-based fracture/fatigue analysis for existing shared candidates.

This analysis is read-only with respect to all production campaigns.  It uses
the exact EXP-floor derivatives, censor-safe developed rates, saved cycle
hazard perturbations, and saved monotonic first-passage states.  Quantities
that are not integrated-hazard sensitivities are explicitly labelled as
instantaneous proxies.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


KB_EV = 8.617333262145e-5
EV_J = 1.602176634e-19
R_LOAD = 0.1
R0_M = 1.0e-6
SIGMA_CAP_PA = 30.0e9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causality-root", type=Path, required=True)
    parser.add_argument("--historical-fatigue-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def finite(value, default=np.nan) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _surface_scalars(row: pd.Series, prefix: str, temperature_K: float) -> tuple[float, ...]:
    tref = finite(row.get("Tref_K"), 300.0)
    g0 = max(
        finite(row[f"{prefix}_G00_eV"])
        + finite(row.get(f"{prefix}_gT_eV_per_K"), 0.0) * (temperature_K - tref),
        1.0e-12,
    )
    g0_T = finite(row.get(f"{prefix}_gT_eV_per_K"), 0.0)
    sigc = max(
        (finite(row[f"{prefix}_sigc0_GPa"])
         + finite(row.get(f"{prefix}_sT_GPa_per_K"), 0.0) * (temperature_K - tref))
        * 1.0e9,
        1.0,
    )
    sigc_T = finite(row.get(f"{prefix}_sT_GPa_per_K"), 0.0) * 1.0e9
    a = max(finite(row[f"{prefix}_exp_a"]), 1.0e-30)
    n = max(finite(row[f"{prefix}_exp_n"]), 1.0e-12)
    floor_fraction = finite(row[f"{prefix}_floor_frac"])
    raw_floor = floor_fraction * g0
    if raw_floor <= 1.0e-4:
        floor, floor_T = 1.0e-4, 0.0
    elif raw_floor >= 0.95 * g0:
        floor, floor_T = 0.95 * g0, 0.95 * g0_T
    else:
        floor, floor_T = raw_floor, floor_fraction * g0_T
    return g0, g0_T, sigc, sigc_T, a, n, floor, floor_T


def exp_floor_from_stress(
    row: pd.Series, stress_Pa: np.ndarray | float, temperature_K: float, prefix: str
) -> dict[str, np.ndarray]:
    """Exact EXP-floor value and first/second stress derivatives."""
    g0, g0_T, sigc, sigc_T, a, n, floor, floor_T = _surface_scalars(
        row, prefix, temperature_K
    )
    sigma = np.maximum(np.asarray(stress_Pa, dtype=float), 0.0)
    x = sigma / sigc
    z = a * np.power(x, n)
    exponential = np.exp(-z)
    drop = g0 - floor
    drop_T = g0_T - floor_T
    barrier = floor + drop * exponential
    positive = sigma > 0.0
    dG_dsigma = np.zeros_like(sigma)
    d2G_dsigma2 = np.zeros_like(sigma)
    np.divide(
        -drop * exponential * n * z,
        sigma,
        out=dG_dsigma,
        where=positive,
    )
    np.divide(
        drop * exponential * n * z * (n * z - n + 1.0),
        np.square(sigma),
        out=d2G_dsigma2,
        where=positive,
    )
    z_T = -n * z * sigc_T / sigc
    dG_dT = floor_T + drop_T * exponential - drop * exponential * z_T
    return {
        "G_eV": barrier,
        "dG_dsigma_eV_per_Pa": dG_dsigma,
        "d2G_dsigma2_eV_per_Pa2": d2G_dsigma2,
        "dG_dT_eV_per_K": dG_dT,
        "G0_eV": np.full_like(sigma, g0),
        "Gfloor_eV": np.full_like(sigma, floor),
        "sigma_c_Pa": np.full_like(sigma, sigc),
    }


def exp_floor_from_deltaK(
    row: pd.Series,
    deltaK_MPa_sqrt_m: np.ndarray | float,
    temperature_K: float,
    prefix: str,
) -> dict[str, np.ndarray]:
    deltaK = np.asarray(deltaK_MPa_sqrt_m, dtype=float)
    scale = 1.0e6 / ((1.0 - R_LOAD) * math.sqrt(2.0 * math.pi * R0_M))
    uncapped = deltaK * scale
    stress = np.minimum(uncapped, SIGMA_CAP_PA)
    result = exp_floor_from_stress(row, stress, temperature_K, prefix)
    active = uncapped < SIGMA_CAP_PA
    result["dG_dK_eV_per_MPa_sqrt_m"] = result["dG_dsigma_eV_per_Pa"] * scale * active
    result["d2G_dK2_eV_per_MPa2_m"] = result["d2G_dsigma2_eV_per_Pa2"] * scale**2 * active
    result["stress_Pa"] = stress
    return result


def conservative_local_slopes(curve: pd.DataFrame) -> pd.DataFrame:
    """Three-point local fits, never crossing a mode/censor boundary."""
    output: list[dict[str, object]] = []
    for (candidate, mode), group in curve.groupby(["candidate_id", "integration_mode"]):
        q = group[group.developed_da_dN_m_per_cycle.gt(0.0)].copy()
        q = q.sort_values("deltaK_MPa_sqrt_m").drop_duplicates("deltaK_MPa_sqrt_m")
        if len(q) < 2:
            continue
        x = q.deltaK_MPa_sqrt_m.to_numpy(float)
        lx = np.log(x)
        y = np.log(q.developed_da_dN_m_per_cycle.to_numpy(float))
        local_m, local_sk = [], []
        for index in range(len(q)):
            lo, hi = max(0, index - 1), min(len(q), index + 2)
            if hi - lo < 2:
                lo, hi = max(0, hi - 2), min(len(q), lo + 2)
            local_m.append(float(stats.linregress(lx[lo:hi], y[lo:hi]).slope))
            local_sk.append(float(stats.linregress(x[lo:hi], y[lo:hi] / math.log(10.0)).slope))
        dm = np.gradient(np.asarray(local_m), lx)
        for (_, source), mm, sk, change in zip(q.iterrows(), local_m, local_sk, dm):
            output.append(
                {
                    **source.to_dict(),
                    "local_m": mm,
                    "local_S_K_per_MPa_sqrt_m": sk,
                    "dm_dln_deltaK": float(change),
                    "local_slope_method": "THREE_POINT_LOCAL_LINEAR_WITH_MODE_BOUNDARIES",
                }
            )
    return pd.DataFrame(output)


def segment_fit(group: pd.DataFrame, coordinate: str, response: str) -> dict[str, object]:
    q = group[[coordinate, response]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(q)
    empty = {
        "slope": np.nan,
        "intercept": np.nan,
        "standard_error": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "r2": np.nan,
        "n_points": n,
        "coordinate_span": np.nan,
        "dynamic_rate_span_decades": np.nan,
        "fit_quality": "INSUFFICIENT",
    }
    if n < 2 or np.ptp(q[coordinate]) <= 0.0:
        return empty
    fit = stats.linregress(q[coordinate], q[response])
    if n >= 3:
        critical = stats.t.ppf(0.975, n - 2)
        ci = (fit.slope - critical * fit.stderr, fit.slope + critical * fit.stderr)
        quality = "QUALIFIED_OLS" if n >= 3 else "INSUFFICIENT"
    else:
        ci = (np.nan, np.nan)
        quality = "TWO_POINT_DESCRIPTIVE_ONLY"
    return {
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "standard_error": float(fit.stderr),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "r2": float(fit.rvalue**2),
        "n_points": n,
        "coordinate_span": float(np.ptp(q[coordinate])),
        "dynamic_rate_span_decades": float(np.ptp(q[response])),
        "fit_quality": quality,
    }


def load_state_screens(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("state_screen.json")):
        payload = json.loads(path.read_text())
        if "candidate_id" not in payload or "points" not in payload:
            continue
        for point in payload["points"]:
            record = {
                key: value
                for key, value in point.items()
                if key not in {"active_state_vector", "state_relative_changes_per_cycle"}
            }
            record["candidate_id"] = payload["candidate_id"]
            record["temperature_K"] = payload["temperature_K"]
            eps = finite(point.get("frozen_state_deltaK_relative_perturbation"))
            minus = finite(point.get("frozen_state_hazard_minus"))
            plus = finite(point.get("frozen_state_hazard_plus"))
            record["cycle_hazard_log_derivative_m_frozen"] = (
                math.log(plus / minus) / math.log((1.0 + eps) / (1.0 - eps))
                if eps > 0.0 and minus > 0.0 and plus > 0.0
                else np.nan
            )
            rows.append(record)
    return pd.DataFrame(rows)


def prospective_curves(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis = root / "prospective_fatigue" / "analysis_final"
    rates = pd.read_csv(analysis / "prospective_joint_fatigue_rates.csv")
    loads = pd.read_csv(root / "prospective_fatigue" / "load_selection" / "prospective_fatigue_adaptive_loads.csv")
    events = pd.read_csv(analysis / "prospective_joint_fatigue_event_statistics.csv")
    finite_rates = rates[rates.status_class.eq("developed_target_reached")].copy()
    collapsed = (
        finite_rates.groupby(
            ["candidate_id", "integration_mode", "normalized_f", "deltaK_MPa_sqrt_m"],
            as_index=False,
        )
        .agg(
            developed_da_dN_m_per_cycle=("developed_da_dN_m_per_cycle", "median"),
            seed_count=("seed", "nunique"),
        )
    )
    collapsed = collapsed.merge(
        loads[["candidate_id", "normalized_f", "selection_regime"]],
        on=["candidate_id", "normalized_f"],
        how="left",
    )
    regime_map = {
        "VHCF_1E6": "VHCF",
        "HCF_1E4": "HCF",
        "RARE_HCF_LOWER": "HCF",
        "RARE_HCF_20PLUS": "HCF",
        "HCF_LCF_OVERLAP": "HCF",
        "TRANSITION_3_TO_10": "HCF_TO_LCF",
        "LCF_1_TO_3": "LCF",
        "SUBCYCLE_0P1_TO_1": "LCF",
        "SCREEN_UPPER_ENDPOINT": "NEAR_MONOTONIC",
    }
    collapsed["regime"] = collapsed.selection_regime.map(regime_map).fillna("UNCLASSIFIED")
    local = conservative_local_slopes(collapsed)

    event_size = (
        events.groupby(
            ["candidate_id", "integration_mode", "normalized_f", "deltaK_MPa_sqrt_m"],
            as_index=False,
        )
        .agg(
            mean_committed_event_advance_m=("committed_advance_m", "mean"),
            median_committed_event_advance_m=("committed_advance_m", "median"),
            event_observations=("event_index", "count"),
        )
    )
    size_local = event_size.rename(
        columns={"mean_committed_event_advance_m": "developed_da_dN_m_per_cycle"}
    )
    size_slopes = conservative_local_slopes(size_local)
    size_slopes = size_slopes.rename(
        columns={"local_m": "event_size_log_slope_m_delta_a"}
    )[["candidate_id", "integration_mode", "normalized_f", "event_size_log_slope_m_delta_a"]]
    local = local.merge(
        size_slopes,
        on=["candidate_id", "integration_mode", "normalized_f"],
        how="left",
    )
    return collapsed, local, event_size


def prospective_fits(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (candidate, regime), group in curves.groupby(["candidate_id", "regime"]):
        # HCF uses accelerated points; LCF/near-monotonic uses explicit points.
        if regime in {"VHCF", "HCF"}:
            group = group[group.integration_mode.eq("accelerated")]
        elif regime in {"HCF_TO_LCF", "LCF", "NEAR_MONOTONIC"}:
            explicit = group[group.integration_mode.eq("explicit")]
            group = explicit if not explicit.empty else group
        if group.empty:
            continue
        group = group.assign(
            log10_deltaK=np.log10(group.deltaK_MPa_sqrt_m),
            log10_rate=np.log10(group.developed_da_dN_m_per_cycle),
        )
        for coordinate, label in (
            ("log10_deltaK", "m"),
            ("deltaK_MPa_sqrt_m", "S_K"),
        ):
            rows.append(
                {
                    "candidate_id": candidate,
                    "evidence_population": "SEVEN_PROSPECTIVE_EXACT_TRANSFERS",
                    "regime": regime,
                    "slope_coordinate": label,
                    **segment_fit(group, coordinate, "log10_rate"),
                }
            )
    return pd.DataFrame(rows)


def attach_fatigue_predictors(
    local: pd.DataFrame, registry: pd.DataFrame, screens: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    registry = registry.set_index("candidate_id")
    for source in local.itertuples(index=False):
        material = registry.loc[source.candidate_id]
        dk = float(source.deltaK_MPa_sqrt_m)
        bare = exp_floor_from_deltaK(material, dk, 300.0, "cleave")
        emission = exp_floor_from_deltaK(material, dk, 300.0, "emit")
        candidate_screen = screens[screens.candidate_id.eq(source.candidate_id)].sort_values("fraction")
        def interp(column: str) -> float:
            good = candidate_screen[["fraction", column]].replace([np.inf, -np.inf], np.nan).dropna()
            return (
                float(np.interp(source.normalized_f, good.fraction, good[column]))
                if len(good)
                else np.nan
            )
        dgc = float(np.asarray(bare["dG_dK_eV_per_MPa_sqrt_m"]))
        d2gc = float(np.asarray(bare["d2G_dK2_eV_per_MPa2_m"]))
        dge = float(np.asarray(emission["dG_dK_eV_per_MPa_sqrt_m"]))
        m_bare = -dk * dgc / (KB_EV * 300.0)
        hazard = interp("cycle_hazard_log_derivative_m_frozen")
        direct_eff = interp("direct_effective_barrier_derivative_eV_per_MPa_sqrt_m")
        state_eff = interp("state_mediated_effective_barrier_derivative_eV_per_MPa_sqrt_m")
        total_eff = interp("total_effective_barrier_derivative_eV_per_MPa_sqrt_m")
        m_evolved = -dk * total_eff / (KB_EV * 300.0) if math.isfinite(total_eff) else np.nan
        m_state = -dk * state_eff / (KB_EV * 300.0) if math.isfinite(state_eff) else np.nan
        event_size = finite(getattr(source, "event_size_log_slope_m_delta_a", np.nan))
        rows.append(
            {
                **source._asdict(),
                "cleavage_barrier_eV": float(np.asarray(bare["G_eV"])),
                "cleavage_dG_dDeltaK_eV_per_MPa_sqrt_m": dgc,
                "cleavage_d2G_dDeltaK2_eV_per_MPa2_m": d2gc,
                "emission_dG_dDeltaK_eV_per_MPa_sqrt_m": dge,
                "relative_emission_minus_cleavage_derivative_eV_per_MPa_sqrt_m": dge - dgc,
                "m_bare_cleavage": m_bare,
                "m_cycle_hazard_frozen": hazard,
                "direct_effective_barrier_derivative_eV_per_MPa_sqrt_m": direct_eff,
                "state_mediated_effective_barrier_derivative_eV_per_MPa_sqrt_m": state_eff,
                "total_effective_barrier_derivative_eV_per_MPa_sqrt_m": total_eff,
                "m_state_correction": m_state,
                "m_evolved_state_predictor": m_evolved,
                "m_event_size": event_size,
                "m_evolved_plus_event_size": m_evolved + event_size,
                "predicted_dm_dlnDeltaK_bare": (
                    -dk * (dgc + dk * d2gc) / (KB_EV * 300.0)
                ),
                "predictor_semantics": "A_BARE_EXACT_EXP_FLOOR;B_FROZEN_EXACT_CYCLE_HAZARD;C_SCREEN_EVOLVED_STATE",
            }
        )
    return pd.DataFrame(rows)


def detailed_barriers(
    registry: pd.DataFrame,
    fracture_points: pd.DataFrame,
    fracture_states: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    descriptors: list[dict[str, object]] = []
    sensitivities: list[dict[str, object]] = []
    registry = registry.set_index("candidate_id")
    for candidate, material in registry.iterrows():
        points = fracture_points[fracture_points.candidate_id.eq(candidate)].copy()
        if points.empty:
            continue
        points = points.sort_values("temperature_K")
        actual_slope = np.gradient(points.K50_MPa_sqrt_m, points.temperature_K)
        points["actual_dK_dT_MPa_sqrt_m_per_K"] = actual_slope
        states = fracture_states[
            fracture_states.candidate_id.eq(candidate) & fracture_states.event_index.eq(0)
        ]
        state_by_temperature = states.set_index("temperature_K")
        for point in points.itertuples(index=False):
            temperature = float(point.temperature_K)
            kfracture = float(point.K50_MPa_sqrt_m)
            for prefix in ("cleave", "emit"):
                grid = np.linspace(0.0, max(80.0, 1.5 * kfracture), 1601)
                surface = exp_floor_from_deltaK(material, grid, temperature, prefix)
                barrier = surface["G_eV"]
                derivative = surface["dG_dK_eV_per_MPa_sqrt_m"]
                curvature = surface["d2G_dK2_eV_per_MPa2_m"]
                at = exp_floor_from_deltaK(material, kfracture, temperature, prefix)
                g0 = float(np.asarray(at["G0_eV"])); floor = float(np.asarray(at["Gfloor_eV"]))
                def crossing(remaining_fraction: float) -> float:
                    target = floor + remaining_fraction * (g0 - floor)
                    ids = np.flatnonzero(barrier <= target)
                    return float(grid[ids[0]]) if len(ids) else np.nan
                descriptors.append(
                    {
                        "candidate_id": candidate,
                        "temperature_K": temperature,
                        "mechanism": prefix,
                        "K_fracture_MPa_sqrt_m": kfracture,
                        "G0_eV": g0,
                        "Gfloor_eV": floor,
                        "available_barrier_drop_eV": g0 - floor,
                        "normalized_floor_fraction": floor / g0,
                        "characteristic_stress_GPa": float(np.asarray(at["sigma_c_Pa"])) * 1e-9,
                        "K_x90_MPa_sqrt_m": crossing(0.90),
                        "K_x75_MPa_sqrt_m": crossing(0.75),
                        "K_x50_MPa_sqrt_m": crossing(0.50),
                        "K_x25_MPa_sqrt_m": crossing(0.25),
                        "K_x10_MPa_sqrt_m": crossing(0.10),
                        "transition_width_x90_to_x10_MPa_sqrt_m": crossing(0.10) - crossing(0.90),
                        "maximum_minus_dG_dK_eV_per_MPa_sqrt_m": float(np.max(-derivative)),
                        "derivative_at_fracture_eV_per_MPa_sqrt_m": float(np.asarray(at["dG_dK_eV_per_MPa_sqrt_m"])),
                        "maximum_abs_curvature_eV_per_MPa2_m": float(np.max(np.abs(curvature))),
                        "curvature_at_fracture_eV_per_MPa2_m": float(np.asarray(at["d2G_dK2_eV_per_MPa2_m"])),
                        "floor_proximity_at_fracture": (float(np.asarray(at["G_eV"])) - floor) / max(g0 - floor, 1e-30),
                    }
                )
            if temperature not in state_by_temperature.index:
                continue
            state = state_by_temperature.loc[temperature]
            stress = finite(state.sigma_local_effective_Pa)
            local = exp_floor_from_stress(material, stress, temperature, "cleave")
            radius = max(finite(state.tip_radius_m), 1.0e-30)
            d_sigma_dK = 1.0e6 / math.sqrt(2.0 * math.pi * radius)
            dGdK = float(np.asarray(local["dG_dsigma_eV_per_Pa"])) * d_sigma_dK
            G = float(np.asarray(local["G_eV"]))
            dGdT = float(np.asarray(local["dG_dT_eV_per_K"]))
            A_K_proxy = -dGdK / (KB_EV * temperature)
            A_T_proxy = G / (KB_EV * temperature**2) - dGdT / (KB_EV * temperature)
            sensitivities.append(
                {
                    "candidate_id": candidate,
                    "temperature_K": temperature,
                    "K_fracture_MPa_sqrt_m": kfracture,
                    "A_K_instantaneous_peak_rate_per_MPa_sqrt_m": A_K_proxy,
                    "A_T_instantaneous_peak_rate_per_K": A_T_proxy,
                    "predicted_dK_dT_instantaneous_peak_rate_proxy": -A_T_proxy / A_K_proxy if A_K_proxy else np.nan,
                    "actual_dK_dT_MPa_sqrt_m_per_K": float(point.actual_dK_dT_MPa_sqrt_m_per_K),
                    "effective_activation_volume_m3": -float(np.asarray(local["dG_dsigma_eV_per_Pa"])) * EV_J,
                    "effective_activation_volume_A3": -float(np.asarray(local["dG_dsigma_eV_per_Pa"])) * EV_J * 1e30,
                    "K_shield_MPa_sqrt_m": finite(state.K_shield_MPa_sqrt_m),
                    "backstress_mean_Pa": finite(state.backstress_mean_Pa),
                    "tip_radius_m": radius,
                    "monotonic_hazard_sensitivity_semantics": "INSTANTANEOUS_PEAK_RATE_PROXY_NOT_INTEGRATED_HAZARD",
                }
            )
    return pd.DataFrame(descriptors), pd.DataFrame(sensitivities)


def savefig(fig: plt.Figure, out: Path, stem: str, data: pd.DataFrame) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=190, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    data.to_csv(out / f"{stem}_plot_data.csv", index=False)


def figures(
    out: Path,
    master: pd.DataFrame,
    fits: pd.DataFrame,
    fracture_sensitivity: pd.DataFrame,
) -> None:
    prospective = master[master.evidence_population.eq("SEVEN_PROSPECTIVE_EXACT_TRANSFERS")]
    historical = fits[
        fits.evidence_population.eq("HISTORICAL_EXACT_FINGERPRINT_INTERSECTION")
        & fits.regime.eq("HCF")
        & fits.slope_coordinate.eq("m")
    ]
    hcf = fits[
        fits.evidence_population.eq("SEVEN_PROSPECTIVE_EXACT_TRANSFERS")
        & fits.regime.eq("HCF")
        & fits.slope_coordinate.eq("m")
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if len(historical): ax.scatter(historical.candidate_id, historical.slope, color="0.65", label="historical shared")
    if len(hcf): ax.scatter(hcf.candidate_id, hcf.slope, color="#0072B2", label="prospective seven")
    ax.tick_params(axis="x", rotation=90, labelsize=6); ax.set(ylabel=r"developed HCF $m$", title="Paris-like slopes for exact shared candidates"); ax.legend()
    savefig(fig, out, "paris_slopes_all_shared_candidates", pd.concat([historical, hcf], ignore_index=True))

    fig, ax = plt.subplots(figsize=(9, 6))
    for candidate, group in prospective.groupby("candidate_id"):
        ax.plot(group.deltaK_MPa_sqrt_m, group.local_m, "o-", label=candidate.replace("v913_prospective_", ""), alpha=.8)
    ax.set(xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)", ylabel=r"local $m(\Delta K)$", title="Local Paris-slope evolution without crossing mode boundaries")
    ax.legend(fontsize=6, ncol=2)
    savefig(fig, out, "local_m_vs_deltaK", prospective)

    summary = prospective.groupby("candidate_id", as_index=False).agg(
        measured_m_HCF=("local_m", "median"),
        bare_m_HCF=("m_bare_cleavage", "median"),
        hazard_m_HCF=("m_cycle_hazard_frozen", "median"),
        evolved_m_HCF=("m_evolved_state_predictor", "median"),
        cleavage_derivative=("cleavage_dG_dDeltaK_eV_per_MPa_sqrt_m", "median"),
        cleavage_curvature=("cleavage_d2G_dDeltaK2_eV_per_MPa2_m", "median"),
        delta_m=("dm_dln_deltaK", "median"),
    )
    fig, ax = plt.subplots(figsize=(6.6, 5.5)); ax.scatter(-summary.cleavage_derivative, summary.measured_m_HCF, c="#0072B2")
    ax.set(xlabel=r"median $-\partial G_c/\partial\Delta K$", ylabel=r"median measured local $m$", title="HCF slope versus cleavage-barrier sensitivity")
    savefig(fig, out, "m_HCF_vs_cleavage_barrier_derivative", summary)
    fig, ax = plt.subplots(figsize=(6.6, 5.5)); ax.scatter(summary.cleavage_curvature, summary.delta_m, c="#D55E00")
    ax.set(xlabel=r"median $\partial^2G_c/\partial\Delta K^2$", ylabel=r"median $dm/d\ln\Delta K$", title="Paris-slope evolution versus barrier curvature")
    savefig(fig, out, "delta_m_vs_cleavage_curvature", summary)
    fig, ax = plt.subplots(figsize=(6.6, 5.5)); ax.scatter(prospective.m_cycle_hazard_frozen, prospective.local_m, c=prospective.normalized_f, cmap="viridis")
    lo = np.nanmin([prospective.m_cycle_hazard_frozen.min(), prospective.local_m.min()]); hi=np.nanmax([prospective.m_cycle_hazard_frozen.max(), prospective.local_m.max()]); ax.plot([lo,hi],[lo,hi],"k--",lw=1)
    ax.set(xlabel="frozen-state cycle-hazard prediction", ylabel="measured local m", title="Measured versus cycle-integrated hazard slope")
    savefig(fig, out, "measured_vs_hazard_predicted_paris_slope", prospective)
    fig, ax = plt.subplots(figsize=(6.6, 5.5)); q=fracture_sensitivity.replace([np.inf,-np.inf],np.nan).dropna(subset=["predicted_dK_dT_instantaneous_peak_rate_proxy","actual_dK_dT_MPa_sqrt_m_per_K"]); ax.scatter(q.predicted_dK_dT_instantaneous_peak_rate_proxy,q.actual_dK_dT_MPa_sqrt_m_per_K,c=q.temperature_K,cmap="plasma")
    ax.set(xlabel="instantaneous peak-rate proxy",ylabel="measured dK/dT",title="Fracture thermal sensitivity: proxy pending integrated-hazard runs")
    savefig(fig,out,"fracture_dKdT_measured_vs_hazard_predicted",q)
    fig, ax=plt.subplots(figsize=(7,5.5)); joined=summary.merge(fracture_sensitivity.groupby("candidate_id",as_index=False).actual_dK_dT_MPa_sqrt_m_per_K.mean(),on="candidate_id"); sc=ax.scatter(-joined.cleavage_derivative,joined.actual_dK_dT_MPa_sqrt_m_per_K,c=joined.measured_m_HCF,cmap="viridis",s=80,edgecolor="black"); fig.colorbar(sc,ax=ax,label="measured HCF m"); ax.set(xlabel="crack-opening load sensitivity",ylabel="mean fracture dK/dT",title="Common fracture–fatigue sensitivity map")
    savefig(fig,out,"fracture_fatigue_common_sensitivity_map",joined)


def main() -> int:
    args = parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    root = args.causality_root
    registry = pd.read_csv(root / "prospective_fatigue_registry" / "prospective_fatigue_registry.csv")
    curves, local, event_size = prospective_curves(root)
    screens = load_state_screens(root / "prospective_fatigue" / "state_screen")
    local = attach_fatigue_predictors(local, registry, screens)
    prospective_fit = prospective_fits(curves)

    historical_ids = set(pd.read_csv(root / "joint_existing" / "joint_fracture_fatigue_candidate_master.csv").candidate_id)
    old_fit = pd.read_csv(args.historical_fatigue_root / "fatigue_regime_fits.csv")
    old_fit = old_fit[old_fit.candidate_id.isin(historical_ids)].rename(columns={
        "coordinate": "slope_coordinate", "slope": "slope", "se": "standard_error",
        "ci_low": "ci95_low", "ci_high": "ci95_high", "n": "n_points",
        "span": "coordinate_span", "quality": "fit_quality",
    })
    old_fit["evidence_population"] = "HISTORICAL_EXACT_FINGERPRINT_INTERSECTION"
    old_fit["dynamic_rate_span_decades"] = np.nan
    fits = pd.concat([prospective_fit, old_fit[prospective_fit.columns]], ignore_index=True)

    historical_local = pd.read_csv(args.historical_fatigue_root / "fatigue_local_slopes.csv")
    historical_local = historical_local[historical_local.candidate_id.isin(historical_ids)].copy()
    historical_local["evidence_population"] = "HISTORICAL_EXACT_FINGERPRINT_INTERSECTION"
    local["evidence_population"] = "SEVEN_PROSPECTIVE_EXACT_TRANSFERS"
    local["reference_envelope_status"] = "NO_QUANTITATIVE_REFERENCE_ENVELOPE"

    fracture_points = pd.read_csv(root / "prospective_fracture_analysis_final" / "prospective_fracture_response_points.csv")
    k300 = registry[["candidate_id", "stageA_K50_300K_MPa_sqrt_m"]].rename(columns={"stageA_K50_300K_MPa_sqrt_m":"K50_MPa_sqrt_m"}); k300["temperature_K"] = 300.0
    fracture_points = pd.concat([fracture_points[fracture_points.candidate_id.isin(registry.candidate_id)], k300], ignore_index=True).drop_duplicates(["candidate_id","temperature_K"])
    fracture_states = pd.read_csv(root / "prospective_fracture_analysis_final" / "prospective_fracture_state_at_first_passage.csv", low_memory=False)
    barriers, fracture_sensitivity = detailed_barriers(registry, fracture_points, fracture_states)

    local.to_csv(args.out / "local_paris_slope_curves.csv", index=False)
    fits.to_csv(args.out / "paris_slope_master.csv", index=False)
    fits.to_csv(args.out / "joint_paris_slope_master.csv", index=False)
    barriers.to_csv(args.out / "fracture_barrier_detailed_descriptors.csv", index=False)
    fracture_sensitivity.to_csv(args.out / "fracture_hazard_sensitivity.csv", index=False)
    local.to_csv(args.out / "fatigue_hazard_sensitivity.csv", index=False)
    event_size.to_csv(args.out / "fatigue_event_size_statistics.csv", index=False)
    joint = (
        prospective_fit[prospective_fit.regime.eq("HCF") & prospective_fit.slope_coordinate.eq("m")]
        .merge(
            local.groupby("candidate_id", as_index=False).agg(
                median_bare_m=("m_bare_cleavage", "median"),
                median_cycle_hazard_m=("m_cycle_hazard_frozen", "median"),
                median_evolved_m=("m_evolved_state_predictor", "median"),
                median_state_m=("m_state_correction", "median"),
                median_event_size_m=("m_event_size", "median"),
                median_cleavage_dG_dK=("cleavage_dG_dDeltaK_eV_per_MPa_sqrt_m", "median"),
                median_cleavage_curvature=("cleavage_d2G_dDeltaK2_eV_per_MPa2_m", "median"),
            ), on="candidate_id"
        )
        .merge(
            fracture_sensitivity.groupby("candidate_id", as_index=False).agg(
                measured_mean_dK_dT=("actual_dK_dT_MPa_sqrt_m_per_K", "mean"),
                peak_rate_proxy_mean_dK_dT=("predicted_dK_dT_instantaneous_peak_rate_proxy", "mean"),
                mean_activation_volume_A3=("effective_activation_volume_A3", "mean"),
            ), on="candidate_id"
        )
    )
    joint["monotonic_integrated_hazard_status"] = "PENDING_PROSPECTIVE_INSTRUMENTED_RUN"
    joint["reference_envelope_status"] = "NO_QUANTITATIVE_REFERENCE_ENVELOPE"
    joint.to_csv(args.out / "fracture_fatigue_hazard_sensitivity_master.csv", index=False)
    figures(args.out, local, fits, fracture_sensitivity)
    manifest = {
        "schema": "v913_joint_paris_slope_existing_baseline_v1",
        "prospective_candidates": int(local.candidate_id.nunique()),
        "historical_exact_shared_candidates": len(historical_ids),
        "prospective_finite_rate_points": int(len(curves)),
        "prospective_local_slope_points": int(len(local)),
        "state_screen_points": int(len(screens)),
        "fracture_barrier_descriptor_rows": int(len(barriers)),
        "monotonic_integrated_hazard_status": "PENDING_PROSPECTIVE_INSTRUMENTED_RUN",
        "reference_envelope_status": "NO_QUANTITATIVE_REFERENCE_ENVELOPE",
        "physics_changed": False,
    }
    (args.out / "baseline_analysis_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(f"JOINT_PARIS_SLOPE_BASELINE_COMPLETE candidates={manifest['prospective_candidates']} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
