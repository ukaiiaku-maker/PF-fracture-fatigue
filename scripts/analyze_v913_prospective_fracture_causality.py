#!/usr/bin/env python3
"""Analyze prospective v9.13 fracture-causality trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


COORDS = {
    "F1": "achieved__F1_delta_mu",
    "F2": "achieved__F2_activation_window_overlap",
    "F3": "achieved__F3_delta_Theta_sigma_900",
    "F4": "achieved__F4_lowT_plastic_bottleneck",
}
COLORS = {
    "DBTT": "#2563EB",
    "Peak-T": "#EA580C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, nargs="+", required=True)
    parser.add_argument("--historical-grid-root", type=Path, nargs="+", required=True)
    parser.add_argument("--k300-root", type=Path, nargs="+", required=True)
    parser.add_argument("--design-audit", type=Path, nargs="+", required=True)
    parser.add_argument("--anchor-audit", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def crossing_temperature(T: np.ndarray, y: np.ndarray, level: float) -> float:
    for index in range(len(T) - 1):
        a, b = y[index] - level, y[index + 1] - level
        if a == 0.0:
            return float(T[index])
        if a * b <= 0.0 and b != a:
            return float(T[index] + (T[index + 1] - T[index]) * -a / (b - a))
    return float("nan")


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) <= 0.0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def read_root_tables(roots: list[Path], filename: str) -> pd.DataFrame:
    """Read disjoint production batches without copying or rewriting them."""
    tables = [pd.read_csv(root / filename) for root in roots]
    return pd.concat(tables, ignore_index=True)


def morphology(T: np.ndarray, K: np.ndarray) -> str:
    imax = int(np.argmax(K))
    prominence = min(K[imax] - K[0], K[imax] - K[-1])
    delta = K[-1] - K[0]
    span = float(np.ptp(K))
    if 0 < imax < len(K) - 1 and prominence >= 5.0:
        return "PEAK_T"
    if delta >= 5.0:
        return "DBTT_LIKE"
    if span <= 5.0:
        return "WEAK_T"
    if delta <= -5.0:
        return "CERAMIC_OR_INVERSE_T"
    return "INTERMEDIATE"


def response_summary(cases: pd.DataFrame, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    points: list[dict[str, object]] = []
    meta = registry.set_index("prospective_candidate_id")
    for candidate_id, group in cases.groupby("candidate_id", sort=False):
        g = group[group.status.eq("complete")].sort_values("temperature_K")
        historical = g[g.temperature_K.ne(300.0)]
        q300 = g[g.temperature_K.eq(300.0)]
        if len(historical) != 10 or len(q300) != 1:
            continue
        T = historical.temperature_K.to_numpy(float)
        K = historical.K_50um_MPa_sqrt_m.to_numpy(float)
        deriv = np.gradient(K, T)
        imax = int(np.argmax(K))
        imin = int(np.argmin(K))
        prominence = min(K[imax] - K[0], K[imax] - K[-1])
        delta = float(K[-1] - K[0])
        t20 = t80 = float("nan")
        if delta > 0.0:
            t20 = crossing_temperature(T, K, K[0] + 0.2 * delta)
            t80 = crossing_temperature(T, K, K[0] + 0.8 * delta)
        row = {
            "candidate_id": candidate_id,
            "design_family": meta.loc[candidate_id, "design_family"],
            "design_role": meta.loc[candidate_id, "design_role"],
            "target_code": meta.loc[candidate_id, "target_code"],
            "parameter_fingerprint": meta.loc[candidate_id, "parameter_fingerprint"],
            "K300_MPa_sqrt_m": float(q300.K_50um_MPa_sqrt_m.iloc[0]),
            "K700_MPa_sqrt_m": float(K[0]),
            "K1400_MPa_sqrt_m": float(K[-1]),
            "K_min_MPa_sqrt_m": float(K[imin]),
            "K_max_MPa_sqrt_m": float(K[imax]),
            "temperature_at_min_K": float(T[imin]),
            "temperature_at_max_K": float(T[imax]),
            "K_span_MPa_sqrt_m": float(np.ptp(K)),
            "fractional_K_span_over_K300": float(np.ptp(K) / q300.K_50um_MPa_sqrt_m.iloc[0]),
            "DBTT_magnitude_MPa_sqrt_m": delta,
            "DBTT_T20_K": t20,
            "DBTT_T80_K": t80,
            "DBTT_width_K": float(t80 - t20) if np.isfinite(t20 + t80) else float("nan"),
            "peak_prominence_MPa_sqrt_m": float(prominence),
            "peak_temperature_K": float(T[imax]) if 0 < imax < len(T) - 1 else float("nan"),
            "slope_low_MPa_sqrt_m_per_K": linear_slope(T[T <= 900], K[T <= 900]),
            "slope_mid_MPa_sqrt_m_per_K": linear_slope(T[(T >= 900) & (T <= 1100)], K[(T >= 900) & (T <= 1100)]),
            "slope_high_MPa_sqrt_m_per_K": linear_slope(T[T >= 1100], K[T >= 1100]),
            "max_positive_slope_MPa_sqrt_m_per_K": float(np.max(deriv)),
            "max_negative_slope_MPa_sqrt_m_per_K": float(np.min(deriv)),
            "slope_reversal_count": int(np.count_nonzero(np.diff(np.sign(deriv)))),
            "morphology_class": morphology(T, K),
            "historical_grid_complete": True,
            "physics_changed": False,
        }
        for key, column in COORDS.items():
            row[key] = float(meta.loc[candidate_id, column])
        rows.append(row)
        baseline = float(q300.K_50um_MPa_sqrt_m.iloc[0])
        for temperature, value in zip(T, K):
            points.append(
                {
                    "candidate_id": candidate_id,
                    "design_family": row["design_family"],
                    "target_code": row["target_code"],
                    "temperature_K": temperature,
                    "K50_MPa_sqrt_m": value,
                    "K50_over_K300": value / baseline,
                    "morphology_class": row["morphology_class"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(points)


def mechanism_table(states: pd.DataFrame, responses: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    first = states[states.event_index.eq(0)].copy()
    nu0 = registry.set_index("prospective_candidate_id")["peierls_nu0_s"].astype(float)
    reconstructed = []
    for row in first.itertuples(index=False):
        with np.load(row.state_npz) as archive:
            positions = np.flatnonzero(archive["event_index"] == int(row.event_index))
            if len(positions) != 1:
                raise RuntimeError(f"state NPZ event lookup failed: {row.state_npz} event={row.event_index}")
            barrier = np.asarray(archive["peierls_barrier_eV"][positions[0]], dtype=float)
        rate = float(nu0.loc[row.candidate_id]) * np.exp(
            np.clip(-barrier / (8.617333262145e-5 * float(row.temperature_K)), -745.0, 700.0)
        )
        reconstructed.append(float(np.sum(rate)))
    first["peierls_aggregate_rate_s_raw_field"] = first["peierls_aggregate_rate_s"]
    first["peierls_aggregate_rate_s"] = reconstructed
    first["peierls_rate_reconstruction"] = "EXACT_FROM_SAVED_BARRIER_ARRAY_AND_CANDIDATE_NU0"
    first["log10_emission_over_cleavage_rate"] = np.log10(
        np.maximum(first.emission_aggregate_rate_s, 1e-300)
        / np.maximum(first.cleavage_rate_s, 1e-300)
    )
    first["log10_peierls_over_cleavage_rate"] = np.log10(
        np.maximum(first.peierls_aggregate_rate_s, 1e-300)
        / np.maximum(first.cleavage_rate_s, 1e-300)
    )
    first["log10_taylor_over_cleavage_rate"] = np.log10(
        np.maximum(first.taylor_aggregate_rate_s, 1e-300)
        / np.maximum(first.cleavage_rate_s, 1e-300)
    )
    first["barrier_gap_emit_minus_cleave_eV"] = (
        first.emission_barrier_eV_mean - first.cleavage_barrier_eV
    )
    first["transport_bottleneck_log10_s"] = -np.log10(
        np.maximum(
            np.minimum(
                first.peierls_aggregate_rate_s,
                first.taylor_aggregate_rate_s,
            ),
            1e-300,
        )
    )
    first["state_accumulation_index"] = np.log10(
        1.0
        + np.maximum(first.mobile_population_sum_m2, 0.0)
        + np.maximum(first.retained_population_sum_m2, 0.0)
        + np.maximum(first.accumulated_slip_sum_m2, 0.0)
    )
    keep = responses[["candidate_id", "design_family", "morphology_class"]]
    return first.merge(keep, on="candidate_id", how="left", validate="many_to_one")


def _correlation_test(data: pd.DataFrame, x: str, y: str, hypothesis: str, family: str, note: str) -> dict[str, object]:
    q = data[data.design_family.eq(family)][[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(q) < 4 or q[x].nunique() < 2 or q[y].nunique() < 2:
        return {"hypothesis": hypothesis, "design_family": family, "predictor": x, "response": y, "n": len(q), "spearman_rho": np.nan, "spearman_p": np.nan, "evidence": "UNDERPOWERED_OR_CONSTANT", "interpretation": note}
    result = stats.spearmanr(q[x], q[y])
    return {"hypothesis": hypothesis, "design_family": family, "predictor": x, "response": y, "n": len(q), "spearman_rho": float(result.statistic), "spearman_p": float(result.pvalue), "evidence": "PROSPECTIVE_CONTROLLED_ASSOCIATION", "interpretation": note}


def hypothesis_tests(responses: pd.DataFrame, mechanisms: pd.DataFrame) -> pd.DataFrame:
    records = [
        _correlation_test(responses, "F1", "DBTT_magnitude_MPa_sqrt_m", "F-H1", "DBTT", "larger relative activation-window separation should strengthen DBTT"),
        _correlation_test(responses, "F2", "DBTT_width_K", "F-H2", "DBTT", "greater overlap should broaden or weaken DBTT"),
        _correlation_test(responses, "F3", "temperature_at_max_K", "F-H3", "Peak-T", "relative thermal stress-scale motion should move the response maximum"),
        _correlation_test(responses, "F4", "DBTT_magnitude_MPa_sqrt_m", "F-H4", "DBTT", "a stronger low-temperature plastic bottleneck should strengthen DBTT"),
        _correlation_test(responses, "F1", "peak_prominence_MPa_sqrt_m", "F-H5", "Peak-T", "Peak-T should track nonmonotonic closest competition"),
    ]
    summaries = []
    for candidate_id, group in mechanisms.groupby("candidate_id"):
        g = group[group.temperature_K.ne(300.0)].sort_values("temperature_K")
        value = g.log10_emission_over_cleavage_rate.to_numpy(float)
        summaries.append(
            {
                "candidate_id": candidate_id,
                "competition_span_decades": float(np.ptp(value)),
                "competition_slope_reversals": int(np.count_nonzero(np.diff(np.sign(np.gradient(value))))),
                "competition_exact_crossings": int(np.count_nonzero(np.diff(np.sign(value)))),
                "mean_state_accumulation_index": float(g.state_accumulation_index.mean()),
                "mean_backstress_Pa": float(g.backstress_mean_Pa.mean()),
            }
        )
    summary = responses.merge(pd.DataFrame(summaries), on="candidate_id")
    records.append(_correlation_test(summary, "competition_slope_reversals", "peak_prominence_MPa_sqrt_m", "F-H5", "Peak-T", "nonmonotonic competition, not crossing count alone, should support Peak-T"))
    records.append(_correlation_test(summary, "competition_span_decades", "K_span_MPa_sqrt_m", "F-H6", "Peak-T", "weak-T should occur near cancellation with small competition change"))
    records.append(_correlation_test(summary, "mean_state_accumulation_index", "K_span_MPa_sqrt_m", "F-H7", "DBTT", "state accumulation should mediate response amplitude"))
    return pd.DataFrame(records)


def save_figure(fig: plt.Figure, out: Path, stem: str, data: pd.DataFrame) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=190, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    data.to_csv(out / f"{stem}_plot_data.csv", index=False)


def figures(out: Path, responses: pd.DataFrame, points: pd.DataFrame, mechanisms: pd.DataFrame) -> None:
    for family, stem in (("DBTT", "DBTT_causal_design_K_vs_T"), ("Peak-T", "PeakT_causal_design_K_vs_T")):
        q = points[points.design_family.eq(family)]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
        for cid, group in q.groupby("candidate_id"):
            center = "CENTER" in cid
            ax1.plot(group.temperature_K, group.K50_MPa_sqrt_m, "o-", lw=2.5 if center else 1.0, alpha=1.0 if center else 0.58, label="canonical center" if center else None)
            ax2.plot(group.temperature_K, group.K50_over_K300, "o-", lw=2.5 if center else 1.0, alpha=1.0 if center else 0.58)
        ax1.set(xlabel="Temperature (K)", ylabel=r"$K_{50}$ (MPa$\sqrt{m}$)", title=f"{family}-centered dimensional response")
        ax2.set(xlabel="Temperature (K)", ylabel=r"$K_{50}/K_{50}(300 K)$", title="300 K normalized morphology")
        ax1.legend(fontsize=8)
        save_figure(fig, out, stem, q)

    specs = [
        ("F1", "DBTT_magnitude_MPa_sqrt_m", "causal_response_vs_delta_mu", r"$\Delta\mu$", "DBTT magnitude"),
        ("F2", "DBTT_width_K", "causal_response_vs_overlap", r"$O_{ce}$", "DBTT width (K)"),
        ("F3", "temperature_at_max_K", "causal_response_vs_deltaThetaSigma", r"$\Delta\Theta_\sigma$", "Temperature at maximum K (K)"),
        ("F4", "DBTT_magnitude_MPa_sqrt_m", "causal_response_vs_plastic_bottleneck", r"$B_p(700 K)$", "DBTT magnitude"),
    ]
    for x, y, stem, xlabel, ylabel in specs:
        q = responses[["candidate_id", "design_family", "target_code", "morphology_class", x, y]].dropna(subset=[x, y])
        fig, ax = plt.subplots(figsize=(6.6, 5.0))
        for family, group in q.groupby("design_family"):
            ax.scatter(group[x], group[y], s=46, alpha=0.78, edgecolor="black", linewidth=0.3, color=COLORS.get(family), label=family)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=stem.replace("_", " "))
        ax.legend()
        save_figure(fig, out, stem, q)

    first = mechanisms[mechanisms.event_index.eq(0)]
    agg = first.groupby("candidate_id", as_index=False).agg(
        competition_span=("log10_emission_over_cleavage_rate", lambda x: float(np.ptp(x))),
        mean_backstress_Pa=("backstress_mean_Pa", "mean"),
    )
    q = responses.merge(agg, on="candidate_id")
    fig, ax = plt.subplots(figsize=(7.0, 5.3))
    for label, group in q.groupby("morphology_class"):
        ax.scatter(group.competition_span, group.mean_backstress_Pa * 1e-9, s=60, label=label, alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.set(xlabel="First-passage competition span (decades)", ylabel="Mean first-passage backstress (GPa)", title="Prospective causal mechanism-transition map")
    ax.legend(fontsize=7)
    save_figure(fig, out, "causal_mechanism_transition_map", q)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    registry = pd.concat([pd.read_csv(path) for path in args.registry], ignore_index=True)
    if registry.prospective_candidate_id.duplicated().any():
        raise RuntimeError("prospective registry batches contain duplicate candidate IDs")
    if registry.parameter_fingerprint.duplicated().any():
        raise RuntimeError("prospective registry batches contain duplicate fingerprints")
    grid_cases = read_root_tables(
        args.historical_grid_root, "prospective_fracture_case_results.csv"
    )
    k300_cases = read_root_tables(
        args.k300_root, "prospective_fracture_case_results.csv"
    )
    cases = pd.concat([k300_cases, grid_cases], ignore_index=True)
    states = pd.concat(
        [
            read_root_tables(
                args.k300_root, "prospective_fracture_state_at_first_passage.csv"
            ),
            read_root_tables(
                args.historical_grid_root,
                "prospective_fracture_state_at_first_passage.csv",
            ),
        ],
        ignore_index=True,
    )
    responses, points = response_summary(cases, registry)
    mechanisms = mechanism_table(states, responses, registry)
    tests = hypothesis_tests(responses, mechanisms)

    registry.to_csv(args.out / "prospective_fracture_candidate_registry.csv", index=False)
    pd.concat([pd.read_csv(path) for path in args.design_audit], ignore_index=True).to_csv(
        args.out / "prospective_candidate_design_audit.csv", index=False
    )
    pd.concat([pd.read_csv(path) for path in args.anchor_audit], ignore_index=True).to_csv(
        args.out / "prospective_K300_anchor_audit.csv", index=False
    )
    responses.to_csv(args.out / "prospective_fracture_response_summary.csv", index=False)
    points.to_csv(args.out / "prospective_fracture_response_points.csv", index=False)
    states.to_csv(args.out / "prospective_fracture_state_at_first_passage.csv", index=False)
    mechanisms.to_csv(args.out / "prospective_fracture_mechanism_decomposition.csv", index=False)
    tests.to_csv(args.out / "prospective_fracture_hypothesis_tests.csv", index=False)
    figures(args.out, responses, points, mechanisms)
    manifest = {
        "schema": "v913_prospective_fracture_causality_analysis_v1",
        "qualified_candidates": len(registry),
        "analyzed_candidates": len(responses),
        "complete_historical_cases": int(grid_cases.status.eq("complete").sum()),
        "complete_K300_cases": int(k300_cases.status.eq("complete").sum()),
        "full_state_snapshots": len(states),
        "hypothesis_tests": len(tests),
        "physics_changed": False,
        "analysis_only": True,
    }
    (args.out / "prospective_fracture_analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"V913_PROSPECTIVE_FRACTURE_ANALYSIS_COMPLETE candidates={len(responses)} out={args.out}")
    return 0 if len(responses) == len(registry) else 2


if __name__ == "__main__":
    raise SystemExit(main())
