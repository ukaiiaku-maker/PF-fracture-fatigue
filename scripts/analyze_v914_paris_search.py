#!/usr/bin/env python3
"""Censor-safe Paris-first analysis for the v9.14 reparameterization search."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TERMINAL = {"growth_target_reached", "cycle_horizon_reached", "work_budget_exhausted"}
ABCD = {
    "v914_endurance_knee_0462": "A",
    "v914_endurance_knee_0658": "B",
    "v914_endurance_knee_0554": "C",
    "v914_endurance_knee_0133": "D",
}


def slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def finite(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def post_initiation_rate(events: list[dict[str, Any]], minimum_events: int) -> dict[str, float]:
    ordered = sorted(events, key=lambda event: finite(event.get("cumulative_cycles")))
    if len(ordered) < minimum_events + 1:
        return {
            "developed_da_dN_m_per_cycle": math.nan,
            "post_initiation_event_count": max(len(ordered) - 1, 0),
            "post_initiation_cycles": math.nan,
            "event_interval_rate_median": math.nan,
            "event_interval_rate_iqr": math.nan,
        }
    # Drop initiation and, for longer histories, the first 40% of committed
    # events.  The response is an extension-weighted established-growth rate.
    start = max(1, int(math.floor(0.4 * len(ordered))) - 1)
    first, last = ordered[start], ordered[-1]
    delta_cycles = finite(last.get("cumulative_cycles")) - finite(first.get("cumulative_cycles"))
    delta_extension = finite(last.get("cumulative_extension_m")) - finite(first.get("cumulative_extension_m"))
    rates = []
    for event in ordered[start + 1:]:
        interval = finite(event.get("interval_cycles"))
        advance = finite(event.get("committed_advance_m"))
        if interval > 0.0 and advance > 0.0:
            rates.append(advance / interval)
    return {
        "developed_da_dN_m_per_cycle": delta_extension / delta_cycles if delta_cycles > 0.0 and delta_extension > 0.0 else math.nan,
        "post_initiation_event_count": len(ordered) - start - 1,
        "post_initiation_cycles": delta_cycles,
        "event_interval_rate_median": float(np.median(rates)) if rates else math.nan,
        "event_interval_rate_iqr": float(np.subtract(*np.percentile(rates, [75, 25]))) if rates else math.nan,
    }


def point_from_result(
    case: dict[str, Any], result_path: Path, config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    payload = json.loads(result_path.read_text())
    contract_path = result_path.with_name("run_contract.json")
    if not contract_path.is_file():
        raise ValueError(f"terminal result lacks run contract: {result_path}")
    contract = json.loads(contract_path.read_text())
    identity = payload.get("identity_contract", {})
    loading = identity.get("loading", {})
    status = str(payload.get("status", "invalid"))
    if status not in TERMINAL:
        raise ValueError(f"unrecognized terminal status {status!r}: {result_path}")
    if int(payload.get("restart_count", -1)) != 0:
        raise ValueError(f"restart result is not admissible: {result_path}")
    if contract.get("repository_clean") is not True:
        raise ValueError(f"run contract is not clean: {contract_path}")
    if contract.get("candidate") != case["candidate_id"]:
        raise ValueError(f"candidate mismatch: {contract_path}")
    if not math.isclose(finite(contract.get("deltaK_MPa_sqrt_m")), finite(case["deltaK_MPa_sqrt_m"]), rel_tol=1.0e-12):
        raise ValueError(f"DeltaK mismatch: {contract_path}")
    if contract.get("registry_sha256") != manifest["candidate_registry_sha256"]:
        raise ValueError(f"candidate registry hash mismatch: {contract_path}")
    if contract.get("physics_sha256") != manifest["fixed_current_physics_sha256"]:
        raise ValueError(f"fixed physics hash mismatch: {contract_path}")
    if contract.get("repository_head") != identity.get("repository_head"):
        raise ValueError(f"result/contract HEAD mismatch: {result_path}")
    if contract.get("return_control") != "REV" or contract.get("configuration_label") != "CAL-TRANS-REV":
        raise ValueError(f"wrong v7 control configuration: {contract_path}")
    events = list(payload.get("events", []))
    minimum_events = int(config["minimum_post_initiation_events"])
    developed = post_initiation_rate(events, minimum_events)
    cycles = finite(payload.get("completed_physical_cycles"))
    extension = finite(payload.get("cumulative_extension_m"), 0.0)
    total_rate = extension / cycles if cycles > 0.0 else math.nan
    last = events[-1] if events else {}
    developed_usable = bool(
        status == "growth_target_reached"
        and int(payload.get("restart_count", -1)) == 0
        and developed["post_initiation_event_count"] >= minimum_events
        and developed["post_initiation_cycles"] >= float(config["minimum_post_initiation_cycles"])
        and finite(developed["developed_da_dN_m_per_cycle"]) > 0.0
    )
    return {
        "candidate_id": case["candidate_id"],
        "fidelity": case["fidelity"],
        "seed": int(case["seed"]),
        "R": finite(case["R"]),
        "frequency_Hz": finite(case["frequency_Hz"]),
        "temperature_K": finite(case["temperature_K"]),
        "DeltaK_MPa_sqrt_m": finite(case["deltaK_MPa_sqrt_m"]),
        "status": status,
        "censor_type": "none" if status == "growth_target_reached" else status,
        "restart_count": int(payload.get("restart_count", -1)),
        "event_count": len(events),
        "completed_physical_cycles": cycles,
        "cumulative_extension_m": extension,
        "total_da_dN_m_per_cycle": total_rate,
        **developed,
        "developed_point_usable": developed_usable,
        "tip_radius_m": finite(last.get("tip_radius_m")),
        "blunting_increment_m": finite(last.get("tip_radius_m")) - 1.0e-6 if last else math.nan,
        "shielding_MPa_sqrt_m": finite(last.get("shielding_MPa_sqrt_m")),
        "mobile_total_m2": finite(last.get("mobile_total_m2")),
        "retained_total_m2": finite(last.get("retained_total_m2")),
        "physical_return_fraction": finite(last.get("reversible_physical_return_fraction_of_emitted")),
        "total_cycle_map_evaluations": int(payload.get("total_cycle_map_evaluations", 0)),
        "accepted_cycle_map_evaluations": int(payload.get("accepted_cycle_map_evaluations", 0)),
        "accepted_block_count": int(payload.get("accepted_block_count", 0)),
        "rejected_block_count": int(payload.get("rejected_block_count", 0)),
        "maximum_accepted_stride": int(payload.get("maximum_accepted_stride", 0)),
        "dmd_map_refresh_count": int(payload.get("dmd_map_refresh_count", 0)),
        "dmd_map_reuse_count": int(payload.get("dmd_map_reuse_count", 0)),
        "dmd_map_invalidation_count": int(payload.get("dmd_map_invalidation_count", 0)),
        "repository_head": identity.get("repository_head", ""),
        "engine_id": payload.get("engine_id", ""),
        "result_path": str(result_path.resolve()),
        "run_contract_path": str(contract_path.resolve()),
        "run_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    }


def fit_segment(points: list[dict[str, Any]]) -> dict[str, float]:
    x = np.log10([finite(point["DeltaK_MPa_sqrt_m"]) for point in points])
    y = np.log10([finite(point["developed_da_dN_m_per_cycle"]) for point in points])
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    residuals = y - predicted
    ss_res = float(np.sum(residuals**2))
    ss_total = float(np.sum((y - np.mean(y))**2))
    return {
        "n_points": len(points),
        "DeltaK_min": float(10 ** np.min(x)),
        "DeltaK_max": float(10 ** np.max(x)),
        "relative_DeltaK_span": float((10 ** np.max(x) - 10 ** np.min(x)) / (10 ** np.min(x))),
        "da_dN_min": float(10 ** np.min(y)),
        "da_dN_max": float(10 ** np.max(y)),
        "da_dN_span_decades": float(np.ptp(y)),
        "m": float(slope),
        "log10_C": float(intercept),
        "C": float(10 ** intercept) if -300.0 < intercept < 300.0 else (0.0 if intercept <= -300.0 else math.inf),
        "R2": float(1.0 - ss_res / ss_total) if ss_total > 0.0 else 1.0,
        "residual_rmse_log10": float(math.sqrt(ss_res / len(points))),
        "fit_point_DeltaK": ";".join(f"{10 ** value:.12g}" for value in x),
    }


def band_distance(value: float, low: float, high: float, center: float) -> float:
    scale = max(high - low, 1.0e-12)
    if low <= value <= high:
        return 0.25 * abs(value - center) / scale
    edge = low if value < low else high
    return abs(value - edge) / scale + 0.25 * abs(value - center) / scale


def segment_objective(fit: dict[str, float], config: dict[str, Any]) -> dict[str, float]:
    weights = config["objective_weights"]
    slope = band_distance(fit["m"], config["target_m_min"], config["target_m_max"], config["target_m_center"])
    midpoint_log_rate = 0.5 * (math.log10(fit["da_dN_min"]) + math.log10(fit["da_dN_max"]))
    rate = band_distance(
        midpoint_log_rate,
        config["target_log10_da_dN_min_m_per_cycle"],
        config["target_log10_da_dN_max_m_per_cycle"],
        config["target_log10_da_dN_center_m_per_cycle"],
    )
    span = max(0.0, config["minimum_relative_DeltaK_span"] - fit["relative_DeltaK_span"]) / config["minimum_relative_DeltaK_span"]
    span += max(0.0, config["preferred_da_dN_span_decades"] - fit["da_dN_span_decades"]) / config["preferred_da_dN_span_decades"]
    objective = (
        weights["slope_mismatch"] * slope
        + weights["rate_mismatch"] * rate
        + weights["fit_rmse_log10"] * fit["residual_rmse_log10"]
        + weights["span_shortfall"] * span
    )
    return {"slope_mismatch": slope, "rate_mismatch": rate, "span_shortfall": span, "objective": objective}


def best_fit(group: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    ordered = sorted(group, key=lambda point: point["DeltaK_MPa_sqrt_m"])
    usable = [point for point in ordered if point["developed_point_usable"]]
    contiguous_runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for point in ordered:
        if point["developed_point_usable"]:
            current.append(point)
        elif current:
            contiguous_runs.append(current)
            current = []
    if current:
        contiguous_runs.append(current)
    candidates: list[dict[str, Any]] = []
    minimum = int(config["minimum_Paris_points"])
    for run in contiguous_runs:
        for start in range(len(run)):
            for stop in range(start + minimum, len(run) + 1):
                segment = run[start:stop]
                fit = fit_segment(segment)
                metrics = segment_objective(fit, config)
                valid = bool(
                    fit["relative_DeltaK_span"] >= config["minimum_relative_DeltaK_span"]
                    and fit["R2"] >= config["minimum_fit_R2"]
                )
                candidates.append({**fit, **metrics, "valid_Paris_region": valid})
    if candidates:
        valid = [fit for fit in candidates if fit["valid_Paris_region"]]
        chosen = min(valid or candidates, key=lambda fit: (fit["objective"], -fit["n_points"]))
    else:
        chosen = {
            "n_points": len(usable), "DeltaK_min": math.nan, "DeltaK_max": math.nan,
            "relative_DeltaK_span": math.nan, "da_dN_min": math.nan, "da_dN_max": math.nan,
            "da_dN_span_decades": math.nan, "m": math.nan, "log10_C": math.nan, "C": math.nan,
            "R2": math.nan, "residual_rmse_log10": math.nan, "fit_point_DeltaK": "",
            "slope_mismatch": math.inf, "rate_mismatch": math.inf, "span_shortfall": math.inf,
            "objective": math.inf, "valid_Paris_region": False,
        }
    censors = [point for point in group if point["censor_type"] != "none"]
    threshold = bool(censors and any(point["event_count"] == 0 for point in censors))
    return {**chosen, "usable_developed_points": len(usable), "censor_points": len(censors), "threshold_observed": threshold}


def group_key(point: dict[str, Any]) -> tuple[Any, ...]:
    return (
        point["candidate_id"], point["fidelity"], point["seed"], point["R"],
        point["frequency_Hz"], point["temperature_K"],
    )


def level0_decision(points: list[dict[str, Any]], expected_points: int, minimum_cycles: float) -> tuple[bool, str, str]:
    if len(points) < expected_points:
        return False, "LEVEL0_INCOMPLETE", "level0"
    any_growth = any(point["status"] == "growth_target_reached" for point in points)
    all_inert = all(point["event_count"] == 0 for point in points)
    all_catastrophic = all(
        point["status"] == "growth_target_reached" and point["completed_physical_cycles"] <= minimum_cycles
        for point in points
    )
    promote = any_growth and not all_inert and not all_catastrophic
    return (
        promote,
        "LEVEL0_FINITE_NONCATASTROPHIC_GROWTH" if promote else "LEVEL0_EARLY_REJECTION",
        "level1" if promote else "none",
    )


def expected_level0_points(registry_row: dict[str, str], config: dict[str, Any]) -> int:
    level0 = config["level0"]
    expected = len(level0["DeltaK_MPa_sqrt_m"])
    k50 = finite(registry_row.get("historical_monotonic_K50_300K_MPa_sqrt_m"))
    if math.isfinite(k50):
        supplemental = min(
            float(level0["inert_supplement_max_DeltaK_MPa_sqrt_m"]),
            float(level0["inert_supplement_historical_K50_factor"]) * k50,
        )
        if supplemental - max(level0["DeltaK_MPa_sqrt_m"]) >= float(level0["inert_supplement_minimum_separation_MPa_sqrt_m"]):
            expected += 1
    return expected


def plot_outputs(out: Path, points: list[dict[str, Any]], fits: list[dict[str, Any]], registry: list[dict[str, str]], config: dict[str, Any]) -> None:
    colors = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73", "D": "#CC79A7"}
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for candidate, letter in ABCD.items():
        q = sorted((p for p in points if p["candidate_id"] == candidate), key=lambda p: p["DeltaK_MPa_sqrt_m"])
        good = [p for p in q if finite(p["developed_da_dN_m_per_cycle"]) > 0.0]
        if good:
            ax.loglog([p["DeltaK_MPa_sqrt_m"] for p in good], [p["developed_da_dN_m_per_cycle"] for p in good], "o-", label=letter, color=colors[letter])
    ax.set(xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)", ylabel=r"developed $da/dN$ (m/cycle)", title="A-D Paris response under current v7")
    ax.grid(True, which="both", alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout(); fig.savefig(out / "A_ABCD_Paris_curves.png", dpi=180); plt.close(fig)

    valid = [fit for fit in fits if math.isfinite(finite(fit["m"]))]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axhspan(config["target_m_min"], config["target_m_max"], color="#56B4E9", alpha=0.18, label="external target band")
    ax.scatter(range(len(valid)), [fit["m"] for fit in valid], c=[fit["objective"] for fit in valid], cmap="viridis_r")
    ax.set(xlabel="evaluated candidate fit", ylabel="Paris slope m", title="Paris slopes for all evaluated candidates")
    ax.grid(True, axis="y", alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(out / "B_all_candidate_Paris_slopes.png", dpi=180); plt.close(fig)

    coordinates = [field for field in registry[0] if field.startswith("u__")][:6] if registry else []
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
    reg = {row["candidate_id"]: row for row in registry}
    for ax, field in zip(axes.flat, coordinates):
        q = [fit for fit in valid if fit["candidate_id"] in reg]
        ax.scatter([finite(reg[fit["candidate_id"]][field]) for fit in q], [fit["m"] for fit in q], c=[fit["objective"] for fit in q], cmap="viridis_r")
        ax.set(xlabel=field.removeprefix("u__"), ylabel="m"); ax.grid(True, alpha=0.2)
    fig.suptitle("Paris slope versus normalized historical search coordinates"); fig.tight_layout(); fig.savefig(out / "C_m_vs_search_coordinates.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    q = [fit for fit in valid if finite(fit["C"]) > 0.0]
    if q:
        ax.scatter([fit["m"] for fit in q], [fit["C"] for fit in q], c=[fit["objective"] for fit in q], cmap="viridis_r")
    ax.set_yscale("log"); ax.set(xlabel="m", ylabel="C", title="Paris slope versus coefficient"); ax.grid(True, which="both", alpha=0.25); fig.tight_layout(); fig.savefig(out / "D_m_vs_C.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.text(0.5, 0.55, "Experimental Paris curve unavailable", ha="center", transform=ax.transAxes, fontsize=13)
    ax.text(0.5, 0.45, "External target configuration is preserved for replacement", ha="center", transform=ax.transAxes)
    ax.set_axis_off(); fig.tight_layout(); fig.savefig(out / "E_experimental_vs_best_model_Paris_curves.png", dpi=180); plt.close(fig)

    by_candidate: dict[str, dict[str, float]] = {}
    for fit in valid:
        by_candidate.setdefault(fit["candidate_id"], {})[fit["fidelity"]] = fit["m"]
    paired = [value for value in by_candidate.values() if "level1" in value and "level2" in value]
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    if paired:
        ax.scatter([value["level1"] for value in paired], [value["level2"] for value in paired])
    ax.set(xlabel=r"$m_{64}$", ylabel=r"$m_{128}$", title="Low-fidelity versus n=128 Paris slope")
    ax.grid(True, alpha=0.25); fig.tight_layout(); fig.savefig(out / "F_low_fidelity_vs_n128_m.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if valid:
        scatter = ax.scatter([fit["slope_mismatch"] for fit in valid], [fit["residual_rmse_log10"] + fit["span_shortfall"] for fit in valid], c=[fit["rate_mismatch"] for fit in valid], cmap="plasma")
        fig.colorbar(scatter, ax=ax, label="rate mismatch")
    ax.set(xlabel="slope mismatch", ylabel="fit/span penalty", title="Paris-search Pareto responses"); ax.grid(True, alpha=0.25); fig.tight_layout(); fig.savefig(out / "G_Paris_objective_Pareto.png", dpi=180); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/v914_paris_reparameterization_v1"))
    parser.add_argument("--registry", type=Path, default=Path("artifacts/v914_paris_reparameterization/candidate_registry.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/v914_paris_search_target.json"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    registry = read_csv(args.registry)

    points: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for manifest_path in sorted(args.root.glob("level*/launch_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        for case in manifest["cases"]:
            run_dir = manifest_path.parent / case["candidate_id"] / f"DeltaK{slug(case['deltaK_MPa_sqrt_m'])}_seed{case['seed']}"
            result_path = run_dir / "result.json"
            if result_path.is_file():
                point = point_from_result(case, result_path, config, manifest)
                points.append(point)
                history.append({**case, "state": "terminal", "status": point["status"], "result_path": str(result_path.resolve())})
            elif (run_dir / "checkpoint.json").is_file():
                history.append({**case, "state": "active_or_interrupted", "status": "checkpoint_only", "result_path": ""})
            else:
                history.append({**case, "state": "not_started", "status": "missing", "result_path": ""})

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for point in points:
        groups.setdefault(group_key(point), []).append(point)
    fits = []
    for key, group in groups.items():
        candidate, fidelity, seed, R, frequency, temperature = key
        fit = best_fit(group, config)
        fits.append({
            "candidate_id": candidate, "fidelity": fidelity, "seed": seed, "R": R,
            "frequency_Hz": frequency, "temperature_K": temperature, **fit,
        })

    promotions = []
    for row in registry:
        candidate = row["candidate_id"]
        candidate_points = [point for point in points if point["candidate_id"] == candidate]
        level0 = [point for point in candidate_points if point["fidelity"] == "level0"]
        level1_fit = next((fit for fit in fits if fit["candidate_id"] == candidate and fit["fidelity"] == "level1"), None)
        if level1_fit is not None:
            promote = bool(level1_fit["valid_Paris_region"])
            reason = "VALID_LEVEL1_PARIS_REGION" if promote else "NO_VALID_LEVEL1_PARIS_REGION"
            next_fidelity = "level2" if promote else "none"
            objective = level1_fit["objective"]
        elif level0:
            promote, reason, next_fidelity = level0_decision(
                level0,
                expected_level0_points(row, config),
                float(config["minimum_post_initiation_cycles"]),
            )
            objective = math.nan
        else:
            promote, reason, next_fidelity, objective = False, "UNEVALUATED", "level0", math.nan
        promotions.append({
            "candidate_id": candidate, "current_fidelity": "level1" if level1_fit else "level0" if level0 else "none",
            "promote": promote, "next_fidelity": next_fidelity, "promotion_reason": reason, "objective": objective,
        })

    objectives = [{
        "candidate_id": fit["candidate_id"], "fidelity": fit["fidelity"], "objective": fit["objective"],
        "slope_mismatch": fit["slope_mismatch"], "rate_mismatch": fit["rate_mismatch"],
        "fit_rmse_log10": fit["residual_rmse_log10"], "span_shortfall": fit["span_shortfall"],
        "valid_Paris_region": fit["valid_Paris_region"],
    } for fit in fits]

    point_fields = [
        "candidate_id", "fidelity", "seed", "R", "frequency_Hz", "temperature_K", "DeltaK_MPa_sqrt_m",
        "status", "censor_type", "restart_count", "event_count", "completed_physical_cycles", "cumulative_extension_m",
        "total_da_dN_m_per_cycle", "developed_da_dN_m_per_cycle", "developed_point_usable", "post_initiation_event_count",
        "post_initiation_cycles", "event_interval_rate_median", "event_interval_rate_iqr", "tip_radius_m", "blunting_increment_m",
        "shielding_MPa_sqrt_m", "mobile_total_m2", "retained_total_m2", "physical_return_fraction",
        "total_cycle_map_evaluations", "accepted_cycle_map_evaluations", "accepted_block_count", "rejected_block_count",
        "maximum_accepted_stride", "dmd_map_refresh_count", "dmd_map_reuse_count", "dmd_map_invalidation_count",
        "repository_head", "engine_id", "result_path", "run_contract_path", "run_contract_sha256",
    ]
    fit_fields = [
        "candidate_id", "fidelity", "seed", "R", "frequency_Hz", "temperature_K", "n_points", "DeltaK_min", "DeltaK_max",
        "relative_DeltaK_span", "da_dN_min", "da_dN_max", "da_dN_span_decades", "m", "C", "log10_C", "R2",
        "residual_rmse_log10", "usable_developed_points", "censor_points", "valid_Paris_region", "threshold_observed",
        "slope_mismatch", "rate_mismatch", "span_shortfall", "objective", "fit_point_DeltaK",
    ]
    write_csv(out / "paris_search_points.csv", points, point_fields)
    write_csv(out / "paris_candidate_fits.csv", fits, fit_fields)
    write_csv(out / "candidate_objectives.csv", objectives, list(objectives[0]) if objectives else ["candidate_id", "fidelity", "objective", "slope_mismatch", "rate_mismatch", "fit_rmse_log10", "span_shortfall", "valid_Paris_region"])
    write_csv(out / "candidate_promotions.csv", promotions, list(promotions[0]))
    write_csv(out / "search_history.csv", history, list(history[0]) if history else ["candidate_id", "state", "status"])
    # Required copy is kept beside response tables for self-contained re-ranking.
    write_csv(out / "candidate_registry.csv", registry, list(registry[0]))
    plot_outputs(out, points, fits, registry, config)
    audit = {
        "schema": "v914_paris_search_analysis_audit_v1", "terminal_points": len(points), "candidate_fits": len(fits),
        "valid_Paris_regions": sum(bool(fit["valid_Paris_region"]) for fit in fits),
        "experimental_reference_status": config["target_source_status"], "restart_results_accepted": False,
    }
    (out / "analysis_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"V914_PARIS_ANALYSIS_COMPLETE points={len(points)} fits={len(fits)} valid={audit['valid_Paris_regions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
