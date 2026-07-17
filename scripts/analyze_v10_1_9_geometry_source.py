#!/usr/bin/env python3
"""Analyze the v10.1.9 DBTT geometry/source-capacity matrix."""
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
               geometry_gain: float) -> dict[str, Any]:
    summary = json.loads((case_dir / "summary.json").read_text())[0]
    audit = json.loads((case_dir / "kinetic_tip_cell_audit_v101.json").read_text())
    records = audit.get("records", [])
    if not records:
        raise ValueError(f"no kinetic records in {case_dir}")

    fired_indices = [i for i, r in enumerate(records) if bool(r.get("fired", False))]
    if not fired_indices:
        raise ValueError(f"no crack advances in {case_dir}")
    event_K = [_float(records[i].get("K_Pa_sqrt_m")) / 1.0e6 for i in fired_indices]
    event_K = [v for v in event_K if math.isfinite(v)]
    if not event_K:
        raise ValueError(f"no finite event K values in {case_dir}")

    first_i = fired_indices[0]
    first = records[first_i]
    final_advance = max(_float(r.get("micro_advance_total_m"), 0.0) for r in records)
    late_threshold = 0.75 * final_advance
    late = [r for r in records if _float(r.get("micro_advance_total_m"), 0.0) >= late_threshold]
    if not late:
        late = records[-max(1, len(records) // 4):]

    K_init = event_K[0]
    late_event = event_K[-min(3, len(event_K)):]
    return {
        "kind": kind,
        "temperature_K": temperature,
        "geometry_gain": geometry_gain,
        "outdir": str(case_dir),
        "K_init_MPa_sqrt_m": K_init,
        "R_rise_final_MPa_sqrt_m": event_K[-1] - K_init,
        "R_rise_late_MPa_sqrt_m": float(np.mean(late_event)) - K_init,
        "R_rise_peak_MPa_sqrt_m": max(event_K) - K_init,
        "n_advances": int(summary.get("n_advances", len(event_K))),
        "first_reference_radius_m": _float(first.get("geometry_source_reference_radius_m")),
        "first_capacity_ratio": _float(first.get("geometry_source_capacity_ratio"), 1.0),
        "first_cumulative_exposed": _float(first.get("geometry_source_cumulative_exposed"), 0.0),
        "late_capacity_ratio": _mean(late, "geometry_source_capacity_ratio"),
        "max_capacity_ratio": max(_float(r.get("geometry_source_capacity_ratio"), 1.0) for r in records),
        "final_cumulative_exposed": _last(records, "geometry_source_cumulative_exposed"),
        "late_normalized_blunting": _mean(late, "geometry_source_normalized_blunting"),
        "late_active_mean": _mean(late, "developed_state_active_count"),
        "late_retained_mean": _mean(late, "developed_state_retained_count"),
        "late_mobile_mean": _mean(late, "developed_state_mobile_count"),
        "late_backstress_GPa": _mean(late, "sigma_emission_backstress_Pa") / 1.0e9,
        "late_K_shield_MPa_sqrt_m": _mean(late, "campaign_active_K_shield_effective_Pa_sqrt_m") / 1.0e6,
        "cumulative_emitted": _last(records, "developed_state_cumulative_emitted"),
        "cumulative_refreshed": _last(records, "developed_state_cumulative_refreshed"),
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


def _rank(paired: list[dict[str, Any]], temperatures: list[float],
          low_limit: float, high_min: float, emergence_min: float,
          first_passage_relative_tolerance: float) -> list[dict[str, Any]]:
    low_T, high_T = min(temperatures), max(temperatures)
    grouped: dict[float, dict[float, dict[str, Any]]] = {}
    for row in paired:
        grouped.setdefault(float(row["geometry_gain"]), {})[
            float(row["temperature_K"])
        ] = row
    if 0.0 not in grouped:
        raise ValueError("GEOMETRY_GAINS must include 0 for the scale-preserving control")
    control = grouped[0.0]

    ranked = []
    for gain, by_T in grouped.items():
        if low_T not in by_T or high_T not in by_T:
            continue
        low = by_T[low_T]
        high = by_T[high_T]
        low_rise = float(low["plastic_R_rise_late_MPa_sqrt_m"])
        high_rise = float(high["plastic_R_rise_late_MPa_sqrt_m"])
        emergence = high_rise - low_rise
        low_init_dev = abs(float(low["K_init_MPa_sqrt_m"]) - float(control[low_T]["K_init_MPa_sqrt_m"]))
        high_init_dev = abs(float(high["K_init_MPa_sqrt_m"]) - float(control[high_T]["K_init_MPa_sqrt_m"]))
        low_tol = first_passage_relative_tolerance * max(abs(float(control[low_T]["K_init_MPa_sqrt_m"])), 1.0e-12)
        high_tol = first_passage_relative_tolerance * max(abs(float(control[high_T]["K_init_MPa_sqrt_m"])), 1.0e-12)
        first_passage_pass = low_init_dev <= low_tol and high_init_dev <= high_tol
        score = high_rise - 1.5 * abs(low_rise) + 0.05 * (
            float(high["late_active_mean"]) - float(low["late_active_mean"])
        )
        row = {
            "geometry_gain": gain,
            "low_temperature_K": low_T,
            "high_temperature_K": high_T,
            "low_plastic_R_rise_late_MPa_sqrt_m": low_rise,
            "high_plastic_R_rise_late_MPa_sqrt_m": high_rise,
            "R_rise_emergence_MPa_sqrt_m": emergence,
            "low_K_init_deviation_from_gain0_MPa_sqrt_m": low_init_dev,
            "high_K_init_deviation_from_gain0_MPa_sqrt_m": high_init_dev,
            "first_passage_preserved": first_passage_pass,
            "low_late_capacity_ratio": low["late_capacity_ratio"],
            "high_late_capacity_ratio": high["late_capacity_ratio"],
            "low_final_cumulative_exposed": low["final_cumulative_exposed"],
            "high_final_cumulative_exposed": high["final_cumulative_exposed"],
            "low_late_active_mean": low["late_active_mean"],
            "high_late_active_mean": high["late_active_mean"],
            "late_active_contrast": high["late_active_mean"] - low["late_active_mean"],
            "low_late_retained_mean": low["late_retained_mean"],
            "high_late_retained_mean": high["late_retained_mean"],
            "late_retained_contrast": high["late_retained_mean"] - low["late_retained_mean"],
            "low_late_backstress_GPa": low["late_backstress_GPa"],
            "high_late_backstress_GPa": high["late_backstress_GPa"],
            "low_late_K_shield_MPa_sqrt_m": low["late_K_shield_MPa_sqrt_m"],
            "high_late_K_shield_MPa_sqrt_m": high["late_K_shield_MPa_sqrt_m"],
            "score": score,
        }
        row["low_T_guardrail_pass"] = abs(low_rise) <= low_limit
        row["high_T_developed_pass"] = high_rise >= high_min
        row["emergence_pass"] = emergence >= emergence_min
        row["candidate_pass"] = bool(
            row["first_passage_preserved"]
            and row["low_T_guardrail_pass"]
            and row["high_T_developed_pass"]
            and row["emergence_pass"]
        )
        ranked.append(row)

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
        writer.writerows({k: row.get(k, "") for k in keys} for row in rows)


def _plot(root: Path, ranked: list[dict[str, Any]]) -> None:
    rows = sorted(ranked, key=lambda r: float(r["geometry_gain"]))
    gains = [float(r["geometry_gain"]) for r in rows]
    low = [float(r["low_plastic_R_rise_late_MPa_sqrt_m"]) for r in rows]
    high = [float(r["high_plastic_R_rise_late_MPa_sqrt_m"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(gains, low, marker="o", label="Low T")
    ax.plot(gains, high, marker="s", label="High T")
    ax.set_xlabel("Geometry source gain")
    ax.set_ylabel("Matched late plastic R-rise (MPa sqrt(m))")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "geometry_source_plastic_R_rise.png", dpi=200)
    plt.close(fig)

    low_ratio = [float(r["low_late_capacity_ratio"]) for r in rows]
    high_ratio = [float(r["high_late_capacity_ratio"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(gains, low_ratio, marker="o", label="Low T")
    ax.plot(gains, high_ratio, marker="s", label="High T")
    ax.set_xlabel("Geometry source gain")
    ax.set_ylabel("Late source-capacity ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "geometry_source_capacity_ratio.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--temperatures", type=float, nargs="+", required=True)
    parser.add_argument("--geometry-gains", type=float, nargs="+", required=True)
    parser.add_argument("--theta", type=float, default=45.0)
    parser.add_argument("--low-limit", type=float, default=0.5)
    parser.add_argument("--high-min", type=float, default=1.0)
    parser.add_argument("--emergence-min", type=float, default=1.0)
    parser.add_argument("--first-passage-relative-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    root = args.root
    temperatures = sorted(args.temperatures)
    gains = sorted(args.geometry_gains)
    theta_tag = f"{args.theta:g}"

    baselines: dict[float, dict[str, Any]] = {}
    case_rows: list[dict[str, Any]] = []
    event_payload: dict[str, Any] = {"baseline": {}, "full": {}}
    for temperature in temperatures:
        path = root / "baseline" / f"T{temperature:g}_th{theta_tag}"
        row = _load_case(path, "baseline", temperature, 0.0)
        baselines[temperature] = row
        case_rows.append(row)
        event_payload["baseline"][f"{temperature:g}"] = row["event_K_MPa_sqrt_m"]

    paired: list[dict[str, Any]] = []
    for gain in gains:
        tag = f"G{_tag(gain)}"
        event_payload["full"][tag] = {}
        for temperature in temperatures:
            path = root / "full" / tag / f"T{temperature:g}_th{theta_tag}"
            full = _load_case(path, "full", temperature, gain)
            case_rows.append(full)
            paired.append(_pair(full, baselines[temperature]))
            event_payload["full"][tag][f"{temperature:g}"] = full["event_K_MPa_sqrt_m"]

    ranked = _rank(
        paired,
        temperatures,
        args.low_limit,
        args.high_min,
        args.emergence_min,
        args.first_passage_relative_tolerance,
    )
    _write_csv(root / "geometry_source_case_summary.csv", case_rows)
    _write_csv(root / "geometry_source_temperature_summary.csv", paired)
    _write_csv(root / "geometry_source_ranking.csv", ranked)
    (root / "geometry_source_event_K.json").write_text(json.dumps(event_payload, indent=2))
    assessment = {
        "schema": "v10.1.9_geometry_source_feedback_matrix",
        "temperatures_K": temperatures,
        "geometry_gains": gains,
        "thresholds": {
            "low_abs_late_plastic_R_rise_max_MPa_sqrt_m": args.low_limit,
            "high_late_plastic_R_rise_min_MPa_sqrt_m": args.high_min,
            "emergence_min_MPa_sqrt_m": args.emergence_min,
            "first_passage_relative_tolerance": args.first_passage_relative_tolerance,
        },
        "best_candidate": ranked[0] if ranked else None,
        "passing_candidates": [r for r in ranked if r["candidate_pass"]],
        "overall_pass": any(r["candidate_pass"] for r in ranked),
    }
    (root / "geometry_source_assessment.json").write_text(json.dumps(assessment, indent=2))
    _plot(root, ranked)

    print("\nGeometry/source feedback ranking")
    print("rank  gain  low_R  high_R  emergence  cap_low  cap_high  active_delta  first_pass  pass")
    for row in ranked:
        print(
            f"{row['rank']:4d}  {row['geometry_gain']:4g}  "
            f"{row['low_plastic_R_rise_late_MPa_sqrt_m']:6.3f}  "
            f"{row['high_plastic_R_rise_late_MPa_sqrt_m']:6.3f}  "
            f"{row['R_rise_emergence_MPa_sqrt_m']:9.3f}  "
            f"{row['low_late_capacity_ratio']:7.3f}  "
            f"{row['high_late_capacity_ratio']:8.3f}  "
            f"{row['late_active_contrast']:12.3f}  "
            f"{str(row['first_passage_preserved']):>10s}  "
            f"{str(row['candidate_pass']):>5s}"
        )
    print(f"\nassessment: {'PASS' if assessment['overall_pass'] else 'REVIEW'}")
    print(root / "geometry_source_assessment.json")


if __name__ == "__main__":
    main()
