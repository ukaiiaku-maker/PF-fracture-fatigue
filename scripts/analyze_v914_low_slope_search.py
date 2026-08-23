#!/usr/bin/env python3
"""Exploratory low-Paris-slope diagnostics for the v9.14 v7 search."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammainc
from scipy.stats import spearmanr

KB_EV_PER_K = 8.617333262145e-5
SLOPE_STATUSES = {
    "PROVISIONAL_THREE_POINT_LOCAL_SLOPE",
    "FOUR_POINT_INVALID_REGION",
    "VALID_PARIS_REGION",
}

PHYSICS_ROLES = {
    "cleave_G00_eV": "cleavage intrinsic zero-stress barrier; slope shape and rate level",
    "cleave_gT_eV_per_K": "cleavage barrier temperature dependence; inactive at Tref in this campaign",
    "cleave_sigc0_GPa": "cleavage activation-stress scale; slope shape",
    "cleave_sT_GPa_per_K": "cleavage activation-stress temperature dependence; inactive at Tref",
    "cleave_exp_a": "cleavage EXP stress sensitivity amplitude; slope shape",
    "cleave_exp_n": "cleavage EXP stress-shape exponent; slope shape",
    "cleave_floor_frac": "cleavage barrier floor fraction; high-drive shape and level",
    "emit_G00_eV": "emission/source zero-stress barrier; source competition and rate level",
    "emit_gT_eV_per_K": "emission barrier temperature dependence; inactive at Tref",
    "emit_sigc0_GPa": "emission activation-stress scale; source competition",
    "emit_sT_GPa_per_K": "emission activation-stress temperature dependence; inactive at Tref",
    "emit_exp_a": "emission stress sensitivity amplitude; source competition",
    "emit_exp_n": "emission stress-shape exponent; source competition",
    "emit_floor_frac": "emission barrier floor; source competition and feasibility",
    "peierls_H0_eV": "mobile-dislocation transport enthalpy; retained/mobile state",
    "peierls_activation_entropy_kB": "mobile-dislocation transport entropy; retained/mobile state",
    "peierls_exp_a": "mobile-dislocation transport stress sensitivity amplitude",
    "peierls_exp_n": "mobile-dislocation transport stress-shape exponent",
    "taylor_H0_eV": "retained/backstress obstacle enthalpy",
    "taylor_activation_entropy_kB": "retained/backstress obstacle entropy",
    "taylor_exp_a": "retained/backstress obstacle stress sensitivity amplitude",
    "taylor_exp_n": "retained/backstress obstacle stress-shape exponent",
    "rho_source0_m2": "persistent source population/activity; feasibility and level",
    "taylor_corr_rho_c_m2": "density-correlation/backstress transition density",
    "taylor_corr_scale": "density-correlation/backstress state scale",
    "c_blunt": "source/blunting competition through tip-radius feedback",
}


def finite(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def log_fit(x: list[float], y: list[float]) -> dict[str, float]:
    xa, ya = np.log10(np.asarray(x, float)), np.log10(np.asarray(y, float))
    slope, intercept = np.polyfit(xa, ya, 1)
    predicted = slope * xa + intercept
    total = float(np.sum((ya - np.mean(ya)) ** 2))
    r2 = 1.0 - float(np.sum((ya - predicted) ** 2)) / total if total > 0.0 else math.nan
    return {"m": float(slope), "C": float(10.0**intercept), "R2": r2}


def association(values: list[tuple[float, float]]) -> tuple[float, int]:
    clean = [(x, y) for x, y in values if math.isfinite(x) and math.isfinite(y)]
    if len(clean) < 3 or len({x for x, _ in clean}) < 2 or len({y for _, y in clean}) < 2:
        return math.nan, len(clean)
    result = spearmanr([x for x, _ in clean], [y for _, y in clean])
    return float(result.statistic), len(clean)


def developed_rate(events: list[dict[str, Any]], limit_intervals: int | None = None) -> float:
    ordered = sorted(events, key=lambda event: finite(event.get("cumulative_cycles")))
    if len(ordered) < 4:
        return math.nan
    start = max(1, int(math.floor(0.4 * len(ordered))) - 1)
    selected = ordered[start:]
    if limit_intervals is not None:
        selected = selected[: limit_intervals + 1]
    if len(selected) < 2:
        return math.nan
    cycles = finite(selected[-1].get("cumulative_cycles")) - finite(selected[0].get("cumulative_cycles"))
    extension = finite(selected[-1].get("cumulative_extension_m")) - finite(selected[0].get("cumulative_extension_m"))
    return extension / cycles if cycles > 0.0 and extension > 0.0 else math.nan


def barrier_components(row: dict[str, str], delta_k: float, common: dict[str, Any]) -> dict[str, float]:
    temperature, R = 300.0, 0.1
    g0 = finite(row["cleave_G00_eV"]) + finite(row["cleave_gT_eV_per_K"]) * (temperature - finite(row["Tref_K"]))
    sigc = (finite(row["cleave_sigc0_GPa"]) + finite(row["cleave_sT_GPa_per_K"]) * (temperature - finite(row["Tref_K"]))) * 1.0e9
    floor = min(0.95 * g0, max(1.0e-4, finite(row["cleave_floor_frac"]) * g0))
    kmax = delta_k / (1.0 - R)
    radius = finite(common.get("r0_m"), 1.0e-6)
    sigma = kmax * 1.0e6 / math.sqrt(2.0 * math.pi * radius)
    barrier = floor + (g0 - floor) * math.exp(-finite(row["cleave_exp_a"]) * (sigma / sigc) ** finite(row["cleave_exp_n"]))
    raw = finite(common.get("cleavage_nu0_s"), 1.0e12) * math.exp(-barrier / (KB_EV_PER_K * temperature))
    tau = finite(common.get("cleavage_correlation_time_s"), 1.0e-6)
    hits = finite(common.get("cleavage_hits"), 3.0)
    exposure = min(raw * tau, 1.0e12)
    production = raw if hits <= 1.0 + 1.0e-12 else float(gammainc(hits, exposure) / tau)
    return {
        "DeltaK_MPa_sqrt_m": delta_k, "Kmax_MPa_sqrt_m": kmax,
        "cleavage_stress_GPa": sigma / 1.0e9, "raw_cleavage_barrier_eV": barrier,
        "raw_one_hit_rate_s": raw, "three_hit_production_rate_s": production,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    registry, pool, fits, points = map(read_csv, (args.registry, args.pool, args.fits, args.points))
    by_registry = {row["candidate_id"]: row for row in registry}
    coordinates = [field for field in pool[0] if field.startswith("u__")]

    descriptor = {}
    for fit in fits:
        if fit.get("fidelity") == "level1" and fit.get("descriptor_status") in SLOPE_STATUSES and math.isfinite(finite(fit.get("m"))):
            descriptor[fit["candidate_id"]] = fit
    point_by_candidate: dict[str, list[dict[str, str]]] = {}
    for point in points:
        point_by_candidate.setdefault(point["candidate_id"], []).append(point)

    responses: dict[str, dict[str, float]] = {}
    for candidate, rows in point_by_candidate.items():
        level0 = [row for row in rows if row.get("fidelity") == "level0"]
        developed = [finite(row.get("developed_da_dN_m_per_cycle")) for row in rows if finite(row.get("developed_da_dN_m_per_cycle")) > 0.0]
        growth = [row for row in level0 if row.get("status") == "growth_target_reached"]
        bursts = [row for row in growth if finite(row.get("post_initiation_cycles"), 0.0) <= 1.0]
        responses[candidate] = {
            "developed_m": finite(descriptor.get(candidate, {}).get("m"))
            if descriptor.get(candidate, {}).get("descriptor_status") == "VALID_PARIS_REGION" else math.nan,
            "provisional_developed_local_slope": finite(descriptor.get(candidate, {}).get("m"))
            if descriptor.get(candidate, {}).get("descriptor_status") == "PROVISIONAL_THREE_POINT_LOCAL_SLOPE" else math.nan,
            "combined_slope_descriptor": finite(descriptor.get(candidate, {}).get("m")),
            "log_developed_da_dN": float(np.median(np.log10(developed))) if developed else math.nan,
            "censor_probability": sum(row.get("status") != "growth_target_reached" for row in level0) / len(level0) if level0 else math.nan,
            "burst_arrest_probability": len(bursts) / len(growth) if growth else 0.0 if level0 else math.nan,
        }

    sensitivity = []
    for field in coordinates:
        physical = field.removeprefix("u__")
        metrics = {}
        for response in (
            "developed_m", "provisional_developed_local_slope", "combined_slope_descriptor",
            "log_developed_da_dN", "censor_probability", "burst_arrest_probability",
        ):
            rho, count = association([(finite(by_registry[c][field]), values[response]) for c, values in responses.items() if c in by_registry])
            metrics[f"spearman_{response}"] = rho; metrics[f"n_{response}"] = count
        a = finite(by_registry.get("v914_endurance_knee_0462", {}).get(field))
        steep = [finite(by_registry.get(candidate, {}).get(field)) for candidate in ("v914_endurance_knee_0849", "v914_endurance_knee_0917")]
        contrast = float(np.mean([value for value in steep if math.isfinite(value)]) - a) if math.isfinite(a) and any(math.isfinite(value) for value in steep) else math.nan
        slope_rho, rate_rho = metrics["spearman_combined_slope_descriptor"], metrics["spearman_log_developed_da_dN"]
        feasibility = max(abs(metrics["spearman_censor_probability"]), abs(metrics["spearman_burst_arrest_probability"]))
        if math.isfinite(slope_rho) and abs(slope_rho) >= 0.5:
            interpretation = "decreasing it flattens m" if slope_rho > 0.0 else "increasing it flattens m"
        elif math.isfinite(rate_rho) and abs(rate_rho) >= 0.55:
            interpretation = "primarily controls rate magnitude rather than slope"
        elif math.isfinite(feasibility) and feasibility >= 0.55:
            interpretation = "primarily controls feasibility/initiation"
        else:
            interpretation = "unclear/nonmonotone"
        sensitivity.append({
            "coordinate": physical,
            "physics_role": PHYSICS_ROLES.get(physical, "other constitutive coordinate"),
            "normalized_A_to_mean_0849_0917": contrast,
            **metrics,
            "exploratory_interpretation": interpretation,
        })
    write_csv(args.out / "parameter_slope_sensitivity.csv", sensitivity)
    lines = ["# Exploratory parameter-to-slope sensitivity", "", "Only real developed/provisional slopes are used; censors and bursts have no artificial m. The one formal developed m and two provisional slopes are reported separately; their three-point combined descriptor is used only for exploratory direction signs, not inference.", "", "| parameter | physical role | developed m rho (n) | provisional rho (n) | combined rho (n) | log-rate rho | censor rho | burst rho | interpretation |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in sensitivity:
        lines.append(f"| `{row['coordinate']}` | {row['physics_role']} | {row['spearman_developed_m']:.3g} ({row['n_developed_m']}) | {row['spearman_provisional_developed_local_slope']:.3g} ({row['n_provisional_developed_local_slope']}) | {row['spearman_combined_slope_descriptor']:.3g} ({row['n_combined_slope_descriptor']}) | {row['spearman_log_developed_da_dN']:.3g} | {row['spearman_censor_probability']:.3g} | {row['spearman_burst_arrest_probability']:.3g} | {row['exploratory_interpretation']} |")
    (args.out / "parameter_slope_sensitivity.md").write_text("\n".join(lines) + "\n")

    # Matched A fidelity comparison, including the terminal n=128 work-budget
    # point because it contains five post-initiation intervals and a complete audit.
    matched = [10.24, 10.7388310162, 11.7972560356, 12.96]
    comparison = []
    for fidelity in ("level1", "level2"):
        selected = []
        for load in matched:
            candidates = [row for row in point_by_candidate["v914_endurance_knee_0462"] if row["fidelity"] == fidelity and abs(finite(row["DeltaK_MPa_sqrt_m"]) - load) < 1e-6]
            selected.append(max(candidates, key=lambda row: finite(row.get("post_initiation_event_count"))))
        rates = [finite(row["developed_da_dN_m_per_cycle"]) for row in selected]
        fit = log_fit(matched, rates)
        for row, load, rate in zip(selected, matched, rates):
            comparison.append({"fidelity": fidelity, "DeltaK_MPa_sqrt_m": load, "developed_da_dN_m_per_cycle": rate, "terminal_status": row["status"], **fit})
    write_csv(args.out / "A_n64_n128_matched_comparison.csv", comparison)

    common_payload = json.loads(args.physics.read_text())
    common = common_payload.get("common_physics", common_payload)
    arow = by_registry["v914_endurance_knee_0462"]
    causal = []
    for load in matched:
        point = max((row for row in point_by_candidate["v914_endurance_knee_0462"] if row["fidelity"] == "level1" and abs(finite(row["DeltaK_MPa_sqrt_m"]) - load) < 1e-6), key=lambda row: finite(row.get("post_initiation_event_count")))
        result = json.loads(Path(point["result_path"]).read_text()); events = result["events"]
        post = events[max(1, int(math.floor(0.4 * len(events))) - 1):]
        waits = [finite(event["interval_cycles"]) for event in post[1:]]
        sizes = [finite(event["committed_advance_m"]) for event in post[1:]]
        source = [finite(event.get("gross_source_slip_line_content"), 0.0) / max(finite(event["interval_cycles"]), 1e-300) for event in post[1:]]
        row = barrier_components(arow, load, common)
        row.update({
            "developed_da_dN_m_per_cycle": developed_rate(events),
            "first_3_developed_intervals_da_dN": developed_rate(events, 3),
            "median_developed_wait_cycles": float(np.median(waits)),
            "median_event_size_m": float(np.median(sizes)),
            "median_source_activity_per_cycle": float(np.median(source)),
            "tip_radius_m": finite(post[-1].get("tip_radius_m")),
            "Kshield_MPa_sqrt_m": finite(post[-1].get("shielding_MPa_sqrt_m")),
        }); causal.append(row)
    slopes = {
        key: log_fit(matched, [row[key] for row in causal])["m"]
        for key in ("developed_da_dN_m_per_cycle", "first_3_developed_intervals_da_dN", "raw_one_hit_rate_s", "three_hit_production_rate_s", "median_developed_wait_cycles", "median_event_size_m", "median_source_activity_per_cycle")
        if all(finite(row[key]) > 0.0 for row in causal)
    }
    intrinsic = slopes["raw_one_hit_rate_s"]
    three_hit = slopes["three_hit_production_rate_s"] - intrinsic
    event_size = slopes["median_event_size_m"]
    state_residual = slopes["developed_da_dN_m_per_cycle"] - slopes["three_hit_production_rate_s"] - event_size
    summary = [{
        "measured_full_m": slopes["developed_da_dN_m_per_cycle"],
        "first_3_interval_m": slopes["first_3_developed_intervals_da_dN"],
        "intrinsic_barrier_stress_sensitivity_contribution": intrinsic,
        "three_hit_transformation_contribution": three_hit,
        "blunting_contribution": 0.0,
        "shielding_contribution": 0.0,
        "event_size_variation_contribution": event_size,
        "state_waiting_residual_contribution": state_residual,
        "waiting_time_log_slope": slopes["median_developed_wait_cycles"],
        "source_activity_log_slope": slopes["median_source_activity_per_cycle"],
        "short_descriptor_relative_m_error": abs(slopes["first_3_developed_intervals_da_dN"] - slopes["developed_da_dN_m_per_cycle"]) / abs(slopes["developed_da_dN_m_per_cycle"]),
    }]
    write_csv(args.out / "A_slope_causal_points.csv", causal)
    write_csv(args.out / "A_slope_causal_decomposition.csv", summary)

    discovery_validation = []
    for candidate, fit_row in descriptor.items():
        loads = [float(value) for value in fit_row.get("fit_point_DeltaK", "").split(";") if value]
        full_rates, short_rates, used_loads = [], [], []
        for load in loads:
            choices = [row for row in point_by_candidate.get(candidate, []) if row.get("fidelity") == "level1" and abs(finite(row.get("DeltaK_MPa_sqrt_m")) - load) < 1e-6 and row.get("result_path")]
            if not choices:
                continue
            point = max(choices, key=lambda row: finite(row.get("post_initiation_event_count")))
            events = json.loads(Path(point["result_path"]).read_text()).get("events", [])
            full, short = developed_rate(events), developed_rate(events, 3)
            if full > 0.0 and short > 0.0:
                used_loads.append(load); full_rates.append(full); short_rates.append(short)
        if len(used_loads) >= 3:
            full_fit, short_fit = log_fit(used_loads, full_rates), log_fit(used_loads, short_rates)
            discovery_validation.append({
                "candidate_id": candidate, "n_points": len(used_loads),
                "full_history_m": full_fit["m"], "first_3_intervals_m": short_fit["m"],
                "relative_m_error": abs(short_fit["m"] - full_fit["m"]) / abs(full_fit["m"]),
                "full_history_rank_target": finite(fit_row["m"]),
            })
    discovery_validation.sort(key=lambda row: row["full_history_m"])
    for rank, row in enumerate(discovery_validation, 1):
        row["full_history_rank"] = rank
    for rank, row in enumerate(sorted(discovery_validation, key=lambda item: item["first_3_intervals_m"]), 1):
        row["first_3_intervals_rank"] = rank
    write_csv(args.out / "search_descriptor_target_validation.csv", discovery_validation)

    # Minimum measured slope by adaptive iteration and current-domain GP lower prediction.
    history = []
    for candidate, fit in descriptor.items():
        iteration = int(finite(by_registry.get(candidate, {}).get("surrogate_iteration"), 0.0))
        history.append((iteration, candidate, finite(fit["m"])))
    best_rows = []
    for iteration in range(max([item[0] for item in history], default=0) + 1):
        available = [item for item in history if item[0] <= iteration]
        best = min(available, key=lambda item: item[2])
        best_rows.append({"iteration": iteration, "best_candidate_id": best[1], "observed_best_m": best[2]})
    write_csv(args.out / "best_m_vs_iteration.csv", best_rows)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.step([r["iteration"] for r in best_rows], [r["observed_best_m"] for r in best_rows], where="post", marker="o")
    ax.axhspan(2, 10, color="tab:green", alpha=.15, label="provisional target"); ax.set(xlabel="adaptive iteration", ylabel="minimum measured developed m", title="Best developed Paris slope by search iteration"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(args.out / "best_m_vs_iteration.png", dpi=180); plt.close(fig)
    print(f"V914_LOW_SLOPE_ANALYSIS_COMPLETE descriptors={len(descriptor)} best_m={min((finite(x['m']) for x in descriptor.values()), default=math.nan):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
