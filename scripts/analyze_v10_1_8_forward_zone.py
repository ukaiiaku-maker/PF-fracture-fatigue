#!/usr/bin/env python3
"""Analyze v10.1.8 forward interaction-zone DBTT cases."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _f(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(records: list[dict[str, Any]], key: str) -> float:
    vals = [_f(r.get(key)) for r in records]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else math.nan


def _last(records: list[dict[str, Any]], key: str) -> float:
    for record in reversed(records):
        value = _f(record.get(key))
        if math.isfinite(value):
            return value
    return math.nan


def _tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _load_case(path: Path, kind: str, temperature: float,
               length_scale: float, retention_scale: float) -> dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text())[0]
    audit = json.loads((path / "kinetic_tip_cell_audit_v101.json").read_text())
    records = audit.get("records", [])
    if not records:
        raise ValueError(f"no kinetic records in {path}")
    fired = [r for r in records if bool(r.get("fired", False))]
    event_K = [_f(r.get("K_Pa_sqrt_m")) / 1.0e6 for r in fired]
    event_K = [v for v in event_K if math.isfinite(v)]
    if not event_K:
        raise ValueError(f"no finite crack events in {path}")

    final_advance = max(_f(r.get("micro_advance_total_m"), 0.0) for r in records)
    late = [
        r for r in records
        if _f(r.get("micro_advance_total_m"), 0.0) >= 0.75 * final_advance
    ]
    if not late:
        late = records[-max(1, len(records) // 4):]

    K0 = event_K[0]
    late_events = event_K[-min(5, len(event_K)):]
    return {
        "kind": kind,
        "temperature_K": temperature,
        "interaction_length_scale": length_scale,
        "retention_scale": retention_scale,
        "outdir": str(path),
        "K_init_MPa_sqrt_m": K0,
        "R_rise_final_MPa_sqrt_m": event_K[-1] - K0,
        "R_rise_late_MPa_sqrt_m": float(np.mean(late_events)) - K0,
        "R_rise_peak_MPa_sqrt_m": max(event_K) - K0,
        "n_advances": int(summary.get("n_advances", len(event_K))),
        "late_mobile_mean": _mean(late, "forward_active_mobile_count"),
        "late_retained_mean": _mean(late, "forward_active_retained_count"),
        "late_active_mean": _mean(late, "forward_active_total_count"),
        "late_source_available_mean": _mean(late, "forward_source_available_total"),
        "late_source_available_fraction_mean": _mean(late, "forward_source_available_fraction"),
        "late_source_centroid_um": 1.0e6 * _mean(late, "forward_source_available_centroid_m"),
        "late_depletion_centroid_um": 1.0e6 * _mean(late, "forward_source_depletion_centroid_m"),
        "late_emission_centroid_um": 1.0e6 * _mean(late, "forward_source_last_emission_centroid_m"),
        "late_backstress_GPa": _mean(late, "sigma_emission_backstress_Pa") / 1.0e9,
        "late_K_shield_MPa_sqrt_m": _mean(late, "campaign_active_K_shield_effective_Pa_sqrt_m") / 1.0e6,
        "cumulative_source_consumed": _last(records, "forward_source_cumulative_consumed"),
        "cumulative_source_inflow": _last(records, "forward_source_cumulative_inflow"),
        "cumulative_available_outflow": _last(records, "forward_source_cumulative_available_outflow"),
        "final_source_available": _last(records, "forward_source_available_total"),
        "interaction_length_um": 1.0e6 * _last(records, "forward_interaction_length_m"),
        "event_K_MPa_sqrt_m": event_K,
    }


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


def _rank(rows: list[dict[str, Any]], temperatures: list[float]) -> list[dict[str, Any]]:
    low_T, high_T = min(temperatures), max(temperatures)
    grouped: dict[tuple[float, float], dict[float, dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["interaction_length_scale"]), float(row["retention_scale"]))
        grouped.setdefault(key, {})[float(row["temperature_K"])] = row

    ranked = []
    for (length, retention), by_T in grouped.items():
        if low_T not in by_T or high_T not in by_T:
            continue
        low, high = by_T[low_T], by_T[high_T]
        low_r = float(low["plastic_R_rise_late_MPa_sqrt_m"])
        high_r = float(high["plastic_R_rise_late_MPa_sqrt_m"])
        emergence = high_r - low_r
        init_penalty = abs(float(low["plastic_initiation_shift_MPa_sqrt_m"])) + abs(float(high["plastic_initiation_shift_MPa_sqrt_m"]))
        score = high_r - 1.5 * abs(low_r) - 0.15 * init_penalty
        candidate = {
            "interaction_length_scale": length,
            "retention_scale": retention,
            "low_temperature_K": low_T,
            "high_temperature_K": high_T,
            "low_plastic_R_rise_late_MPa_sqrt_m": low_r,
            "high_plastic_R_rise_late_MPa_sqrt_m": high_r,
            "R_rise_emergence_MPa_sqrt_m": emergence,
            "low_plastic_initiation_shift_MPa_sqrt_m": low["plastic_initiation_shift_MPa_sqrt_m"],
            "high_plastic_initiation_shift_MPa_sqrt_m": high["plastic_initiation_shift_MPa_sqrt_m"],
            "low_late_active_mean": low["late_active_mean"],
            "high_late_active_mean": high["late_active_mean"],
            "late_active_contrast": high["late_active_mean"] - low["late_active_mean"],
            "low_late_retained_mean": low["late_retained_mean"],
            "high_late_retained_mean": high["late_retained_mean"],
            "late_retained_contrast": high["late_retained_mean"] - low["late_retained_mean"],
            "low_cumulative_source_consumed": low["cumulative_source_consumed"],
            "high_cumulative_source_consumed": high["cumulative_source_consumed"],
            "low_cumulative_source_inflow": low["cumulative_source_inflow"],
            "high_cumulative_source_inflow": high["cumulative_source_inflow"],
            "low_late_backstress_GPa": low["late_backstress_GPa"],
            "high_late_backstress_GPa": high["late_backstress_GPa"],
            "low_late_K_shield_MPa_sqrt_m": low["late_K_shield_MPa_sqrt_m"],
            "high_late_K_shield_MPa_sqrt_m": high["late_K_shield_MPa_sqrt_m"],
            "score": score,
        }
        candidate["low_T_guardrail_pass"] = abs(low_r) <= 0.5
        candidate["high_T_developed_pass"] = high_r >= 1.0
        candidate["emergence_pass"] = emergence >= 1.0
        candidate["candidate_pass"] = bool(
            candidate["low_T_guardrail_pass"]
            and candidate["high_T_developed_pass"]
            and candidate["emergence_pass"]
        )
        ranked.append(candidate)

    ranked.sort(key=lambda r: (bool(r["candidate_pass"]), float(r["score"])), reverse=True)
    for i, row in enumerate(ranked, 1):
        row["rank"] = i
    return ranked


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key == "event_K_MPa_sqrt_m" or key in seen:
                continue
            seen.add(key)
            keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in keys} for row in rows)


def _plot(path: Path, ranked: list[dict[str, Any]]) -> None:
    labels = [f"L={r['interaction_length_scale']:g}, R={r['retention_scale']:g}" for r in ranked]
    x = np.arange(len(ranked))
    low = [r["low_plastic_R_rise_late_MPa_sqrt_m"] for r in ranked]
    high = [r["high_plastic_R_rise_late_MPa_sqrt_m"] for r in ranked]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(ranked)), 5.5))
    width = 0.36
    ax.bar(x - width / 2, low, width, label="Low T")
    ax.bar(x + width / 2, high, width, label="High T")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Matched late plastic R-rise (MPa sqrt(m))")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--temperatures", type=float, nargs="+", required=True)
    parser.add_argument("--interaction-scales", type=float, nargs="+", required=True)
    parser.add_argument("--retention-scales", type=float, nargs="+", required=True)
    parser.add_argument("--theta", type=float, default=45.0)
    args = parser.parse_args()

    temperatures = sorted(args.temperatures)
    theta = f"{args.theta:g}"
    baselines = {}
    all_rows = []
    for T in temperatures:
        row = _load_case(args.root / "baseline" / f"T{T:g}_th{theta}", "baseline", T, 1.0, 1.0)
        baselines[T] = row
        all_rows.append(row)

    paired = []
    events: dict[str, Any] = {"baseline": {}, "full": {}}
    for T, row in baselines.items():
        events["baseline"][f"{T:g}"] = row["event_K_MPa_sqrt_m"]
    for length in sorted(args.interaction_scales):
        for retention in sorted(args.retention_scales):
            name = f"L{_tag(length)}_R{_tag(retention)}"
            events["full"][name] = {}
            for T in temperatures:
                path = args.root / "full" / name / f"T{T:g}_th{theta}"
                row = _load_case(path, "full", T, length, retention)
                all_rows.append(row)
                paired.append(_pair(row, baselines[T]))
                events["full"][name][f"{T:g}"] = row["event_K_MPa_sqrt_m"]

    ranked = _rank(paired, temperatures)
    _write_csv(args.root / "forward_zone_case_summary.csv", all_rows)
    _write_csv(args.root / "forward_zone_temperature_summary.csv", paired)
    _write_csv(args.root / "forward_zone_ranking.csv", ranked)
    (args.root / "forward_zone_event_K.json").write_text(json.dumps(events, indent=2))
    assessment = {
        "schema": "v10.1.8_forward_interaction_zone_matrix",
        "best_candidate": ranked[0] if ranked else None,
        "passing_candidates": [r for r in ranked if r["candidate_pass"]],
        "overall_pass": any(r["candidate_pass"] for r in ranked),
    }
    (args.root / "forward_zone_assessment.json").write_text(json.dumps(assessment, indent=2))
    _plot(args.root / "forward_zone_plastic_R_rise.png", ranked)

    print("\nForward interaction-zone DBTT ranking")
    print("rank  Lscale  retention  low_R  high_R  emergence  active_delta  retained_delta  pass")
    for row in ranked:
        print(
            f"{row['rank']:4d}  {row['interaction_length_scale']:6g}  {row['retention_scale']:9g}  "
            f"{row['low_plastic_R_rise_late_MPa_sqrt_m']:6.3f}  "
            f"{row['high_plastic_R_rise_late_MPa_sqrt_m']:6.3f}  "
            f"{row['R_rise_emergence_MPa_sqrt_m']:9.3f}  "
            f"{row['late_active_contrast']:12.3f}  "
            f"{row['late_retained_contrast']:14.3f}  "
            f"{str(row['candidate_pass']):>5s}"
        )
    print(f"\nassessment: {'PASS' if assessment['overall_pass'] else 'REVIEW'}")


if __name__ == "__main__":
    main()
