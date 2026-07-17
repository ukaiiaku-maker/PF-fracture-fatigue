#!/usr/bin/env python3
"""Analyze the v10.1.7 two-temperature DBTT tuning matrix.

Each full-model event curve is initiation-referenced and compared with the single
matched no-plasticity curve at the same temperature.  Developed-state history is
reported separately from the first transient population maximum.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _mean(records: Iterable[dict[str, Any]], key: str) -> float:
    values = [_float(r.get(key)) for r in records]
    values = [v for v in values if math.isfinite(v)]
    return float(np.mean(values)) if values else math.nan


def _last(records: list[dict[str, Any]], key: str) -> float:
    for record in reversed(records):
        value = _float(record.get(key))
        if math.isfinite(value):
            return value
    return math.nan


def _load_case(case_dir: Path, kind: str, temperature: float,
               backstress_scale: float, refresh_scale: float) -> dict[str, Any]:
    summary = json.loads((case_dir / "summary.json").read_text())[0]
    audit = json.loads((case_dir / "kinetic_tip_cell_audit_v101.json").read_text())
    records = audit.get("records", [])
    if not records:
        raise ValueError(f"no diagnostic records in {case_dir}")

    fired_indices = [i for i, r in enumerate(records) if bool(r.get("fired", False))]
    if not fired_indices:
        raise ValueError(f"no crack-advance records in {case_dir}")
    event_K = [_float(records[i].get("K_Pa_sqrt_m")) / 1.0e6 for i in fired_indices]
    event_K = [v for v in event_K if math.isfinite(v)]
    if not event_K:
        raise ValueError(f"no finite event K values in {case_dir}")

    first_i = fired_indices[0]
    pre_first = records[max(first_i - 1, 0)]
    first = records[first_i]
    final_advance = max(_float(r.get("micro_advance_total_m"), 0.0) for r in records)
    late_threshold = 0.75 * final_advance
    late = [r for r in records if _float(r.get("micro_advance_total_m"), 0.0) >= late_threshold]
    if not late:
        late = records[-max(1, len(records) // 4):]

    K_init = event_K[0]
    late_event = event_K[-min(3, len(event_K)):]
    row = {
        "kind": kind,
        "temperature_K": temperature,
        "backstress_scale": backstress_scale,
        "refresh_scale": refresh_scale,
        "outdir": str(case_dir),
        "K_init_MPa_sqrt_m": K_init,
        "R_rise_final_MPa_sqrt_m": event_K[-1] - K_init,
        "R_rise_late_MPa_sqrt_m": float(np.mean(late_event)) - K_init,
        "R_rise_peak_MPa_sqrt_m": max(event_K) - K_init,
        "n_advances": int(summary.get("n_advances", len(event_K))),
        "pre_first_mobile": _float(pre_first.get("developed_state_mobile_count"), 0.0),
        "pre_first_retained": _float(pre_first.get("developed_state_retained_count"), 0.0),
        "first_mobile": _float(first.get("developed_state_mobile_count"), 0.0),
        "first_retained": _float(first.get("developed_state_retained_count"), 0.0),
        "late_mobile_mean": _mean(late, "developed_state_mobile_count"),
        "late_retained_mean": _mean(late, "developed_state_retained_count"),
        "late_active_mean": _mean(late, "developed_state_active_count"),
        "late_retained_fraction_mean": _mean(late, "developed_state_retained_fraction"),
        "late_backstress_GPa": _mean(late, "sigma_emission_backstress_Pa") / 1.0e9,
        "late_K_shield_MPa_sqrt_m": _mean(late, "campaign_active_K_shield_effective_Pa_sqrt_m") / 1.0e6,
        "max_active_population": max(_float(r.get("developed_state_active_count"), 0.0) for r in records),
        "max_backstress_GPa": max(_float(r.get("sigma_emission_backstress_Pa"), 0.0) for r in records) / 1.0e9,
        "max_K_shield_MPa_sqrt_m": max(abs(_float(r.get("campaign_active_K_shield_effective_Pa_sqrt_m"), 0.0)) for r in records) / 1.0e6,
        "cumulative_emitted": _last(records, "developed_state_cumulative_emitted"),
        "cumulative_refreshed": _last(records, "developed_state_cumulative_refreshed"),
        "cumulative_trapped": _last(records, "developed_state_cumulative_trapped"),
        "cumulative_released": _last(records, "developed_state_cumulative_released"),
        "cumulative_recovered": _last(records, "developed_state_cumulative_recovered"),
        "cumulative_escaped": _last(records, "developed_state_cumulative_escaped"),
        "mobile_residence_count_s": _last(records, "developed_state_mobile_residence_count_s"),
        "retained_residence_count_s": _last(records, "developed_state_retained_residence_count_s"),
        "active_residence_count_s": _last(records, "developed_state_active_residence_count_s"),
        "final_source_budget_remaining": _last(records, "campaign_source_budget_remaining"),
        "final_source_budget_consumed": _last(records, "campaign_source_budget_consumed"),
        "event_K_MPa_sqrt_m": event_K,
    }
    return row


def _pair(full: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    row = {k: v for k, v in full.items() if k != "event_K_MPa_sqrt_m"}
    row.update({
        "baseline_K_init_MPa_sqrt_m": baseline["K_init_MPa_sqrt_m"],
        "plastic_initiation_shift_MPa_sqrt_m": full["K_init_MPa_sqrt_m"] - baseline["K_init_MPa_sqrt_m"],
        "plastic_R_rise_final_MPa_sqrt_m": full["R_rise_final_MPa_sqrt_m"] - baseline["R_rise_final_MPa_sqrt_m"],
        "plastic_R_rise_late_MPa_sqrt_m": full["R_rise_late_MPa_sqrt_m"] - baseline["R_rise_late_MPa_sqrt_m"],
        "plastic_R_rise_peak_MPa_sqrt_m": full["R_rise_peak_MPa_sqrt_m"] - baseline["R_rise_peak_MPa_sqrt_m"],
    })
    return row


def _rank_candidates(paired: list[dict[str, Any]], temperatures: list[float],
                     low_limit: float = 0.5, high_min: float = 1.0,
                     emergence_min: float = 1.0,
                     initiation_tolerance: float = 1.0) -> list[dict[str, Any]]:
    low_T, high_T = min(temperatures), max(temperatures)
    grouped: dict[tuple[float, float], dict[float, dict[str, Any]]] = {}
    for row in paired:
        key = (float(row["backstress_scale"]), float(row["refresh_scale"]))
        grouped.setdefault(key, {})[float(row["temperature_K"])] = row

    reference_key = min(
        grouped,
        key=lambda key: abs(math.log(max(key[0], 1e-30))) + abs(math.log(max(key[1], 1e-30))),
    )
    reference = grouped[reference_key]
    ref_low = reference[low_T]["plastic_initiation_shift_MPa_sqrt_m"]
    ref_high = reference[high_T]["plastic_initiation_shift_MPa_sqrt_m"]

    ranked = []
    for (back, refresh), by_T in grouped.items():
        if low_T not in by_T or high_T not in by_T:
            continue
        low = by_T[low_T]
        high = by_T[high_T]
        low_rise = float(low["plastic_R_rise_late_MPa_sqrt_m"])
        high_rise = float(high["plastic_R_rise_late_MPa_sqrt_m"])
        emergence = high_rise - low_rise
        init_deviation = (
            abs(float(low["plastic_initiation_shift_MPa_sqrt_m"]) - ref_low)
            + abs(float(high["plastic_initiation_shift_MPa_sqrt_m"]) - ref_high)
        )
        score = high_rise - 1.5 * abs(low_rise) - 0.25 * init_deviation
        row = {
            "backstress_scale": back,
            "refresh_scale": refresh,
            "low_temperature_K": low_T,
            "high_temperature_K": high_T,
            "low_plastic_R_rise_late_MPa_sqrt_m": low_rise,
            "high_plastic_R_rise_late_MPa_sqrt_m": high_rise,
            "R_rise_emergence_MPa_sqrt_m": emergence,
            "low_plastic_initiation_shift_MPa_sqrt_m": low["plastic_initiation_shift_MPa_sqrt_m"],
            "high_plastic_initiation_shift_MPa_sqrt_m": high["plastic_initiation_shift_MPa_sqrt_m"],
            "initiation_deviation_from_scale1_MPa_sqrt_m": init_deviation,
            "low_late_active_mean": low["late_active_mean"],
            "high_late_active_mean": high["late_active_mean"],
            "late_active_contrast": high["late_active_mean"] - low["late_active_mean"],
            "low_late_retained_mean": low["late_retained_mean"],
            "high_late_retained_mean": high["late_retained_mean"],
            "late_retained_contrast": high["late_retained_mean"] - low["late_retained_mean"],
            "low_cumulative_emitted": low["cumulative_emitted"],
            "high_cumulative_emitted": high["cumulative_emitted"],
            "low_cumulative_refreshed": low["cumulative_refreshed"],
            "high_cumulative_refreshed": high["cumulative_refreshed"],
            "low_late_backstress_GPa": low["late_backstress_GPa"],
            "high_late_backstress_GPa": high["late_backstress_GPa"],
            "low_late_K_shield_MPa_sqrt_m": low["late_K_shield_MPa_sqrt_m"],
            "high_late_K_shield_MPa_sqrt_m": high["late_K_shield_MPa_sqrt_m"],
            "score": score,
        }
        row["low_T_guardrail_pass"] = abs(low_rise) <= low_limit
        row["high_T_developed_pass"] = high_rise >= high_min
        row["emergence_pass"] = emergence >= emergence_min
        row["initiation_preserved"] = init_deviation <= initiation_tolerance
        row["candidate_pass"] = bool(
            row["low_T_guardrail_pass"]
            and row["high_T_developed_pass"]
            and row["emergence_pass"]
            and row["initiation_preserved"]
        )
        ranked.append(row)

    ranked.sort(key=lambda row: (bool(row["candidate_pass"]), float(row["score"])), reverse=True)
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key == "event_K_MPa_sqrt_m" or key in seen:
                continue
            keys.append(key)
            seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in keys} for row in rows)


def _heatmap(path: Path, ranked: list[dict[str, Any]], backs: list[float],
             refreshes: list[float], key: str, title: str) -> None:
    array = np.full((len(backs), len(refreshes)), np.nan)
    lookup = {(float(r["backstress_scale"]), float(r["refresh_scale"])): r for r in ranked}
    for i, back in enumerate(backs):
        for j, refresh in enumerate(refreshes):
            row = lookup.get((back, refresh))
            if row:
                array[i, j] = float(row[key])
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    image = ax.imshow(array, aspect="auto", origin="lower")
    ax.set_xticks(range(len(refreshes)), [f"{x:g}" for x in refreshes])
    ax.set_yticks(range(len(backs)), [f"{x:g}" for x in backs])
    ax.set_xlabel("Refresh-length scale")
    ax.set_ylabel("Back-stress scale")
    ax.set_title(title)
    for i in range(len(backs)):
        for j in range(len(refreshes)):
            if math.isfinite(array[i, j]):
                ax.text(j, i, f"{array[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--temperatures", type=float, nargs="+", required=True)
    parser.add_argument("--backstress-scales", type=float, nargs="+", required=True)
    parser.add_argument("--refresh-scales", type=float, nargs="+", required=True)
    parser.add_argument("--theta", type=float, default=45.0)
    parser.add_argument("--low-limit", type=float, default=0.5)
    parser.add_argument("--high-min", type=float, default=1.0)
    parser.add_argument("--emergence-min", type=float, default=1.0)
    parser.add_argument("--initiation-tolerance", type=float, default=1.0)
    args = parser.parse_args()

    root = args.root
    temperatures = sorted(args.temperatures)
    backs = sorted(args.backstress_scales)
    refreshes = sorted(args.refresh_scales)
    theta_tag = f"{args.theta:g}"

    baselines: dict[float, dict[str, Any]] = {}
    case_rows: list[dict[str, Any]] = []
    for temperature in temperatures:
        path = root / "baseline" / f"T{temperature:g}_th{theta_tag}"
        row = _load_case(path, "baseline", temperature, 1.0, 1.0)
        baselines[temperature] = row
        case_rows.append(row)

    paired: list[dict[str, Any]] = []
    event_payload: dict[str, Any] = {"baseline": {}, "full": {}}
    for temperature, row in baselines.items():
        event_payload["baseline"][f"{temperature:g}"] = row["event_K_MPa_sqrt_m"]

    for back in backs:
        for refresh in refreshes:
            candidate = f"bs{_tag(back)}_rf{_tag(refresh)}"
            event_payload["full"][candidate] = {}
            for temperature in temperatures:
                path = root / "full" / candidate / f"T{temperature:g}_th{theta_tag}"
                full = _load_case(path, "full", temperature, back, refresh)
                case_rows.append(full)
                paired.append(_pair(full, baselines[temperature]))
                event_payload["full"][candidate][f"{temperature:g}"] = full["event_K_MPa_sqrt_m"]

    ranked = _rank_candidates(
        paired,
        temperatures,
        low_limit=args.low_limit,
        high_min=args.high_min,
        emergence_min=args.emergence_min,
        initiation_tolerance=args.initiation_tolerance,
    )

    _write_csv(root / "dbtt_tuning_case_summary.csv", case_rows)
    _write_csv(root / "dbtt_tuning_temperature_summary.csv", paired)
    _write_csv(root / "dbtt_tuning_ranking.csv", ranked)
    (root / "dbtt_tuning_event_K.json").write_text(json.dumps(event_payload, indent=2))

    assessment = {
        "schema": "v10.1.7_dbtt_developed_state_tuning",
        "temperatures_K": temperatures,
        "backstress_scales": backs,
        "refresh_scales": refreshes,
        "thresholds": {
            "low_abs_late_plastic_R_rise_max_MPa_sqrt_m": args.low_limit,
            "high_late_plastic_R_rise_min_MPa_sqrt_m": args.high_min,
            "emergence_min_MPa_sqrt_m": args.emergence_min,
            "initiation_deviation_max_MPa_sqrt_m": args.initiation_tolerance,
        },
        "best_candidate": ranked[0] if ranked else None,
        "passing_candidates": [r for r in ranked if r["candidate_pass"]],
        "overall_pass": any(r["candidate_pass"] for r in ranked),
    }
    (root / "dbtt_tuning_assessment.json").write_text(json.dumps(assessment, indent=2))

    _heatmap(
        root / "dbtt_R_rise_emergence_heatmap.png", ranked, backs, refreshes,
        "R_rise_emergence_MPa_sqrt_m", "DBTT high-minus-low plastic R-rise",
    )
    _heatmap(
        root / "dbtt_highT_late_R_rise_heatmap.png", ranked, backs, refreshes,
        "high_plastic_R_rise_late_MPa_sqrt_m", "High-temperature late plastic R-rise",
    )
    _heatmap(
        root / "dbtt_late_active_contrast_heatmap.png", ranked, backs, refreshes,
        "late_active_contrast", "High-minus-low late active population",
    )

    print("\nDBTT developed-state tuning ranking")
    print("rank  back  refresh  low_R  high_R  emergence  late_active_delta  init_dev  pass")
    for row in ranked:
        print(
            f"{row['rank']:4d}  {row['backstress_scale']:4g}  {row['refresh_scale']:7g}  "
            f"{row['low_plastic_R_rise_late_MPa_sqrt_m']:6.3f}  "
            f"{row['high_plastic_R_rise_late_MPa_sqrt_m']:6.3f}  "
            f"{row['R_rise_emergence_MPa_sqrt_m']:9.3f}  "
            f"{row['late_active_contrast']:17.3f}  "
            f"{row['initiation_deviation_from_scale1_MPa_sqrt_m']:8.3f}  "
            f"{str(row['candidate_pass']):>5s}"
        )
    print(f"\nassessment: {'PASS' if assessment['overall_pass'] else 'REVIEW'}")
    print(root / "dbtt_tuning_assessment.json")


if __name__ == "__main__":
    main()
