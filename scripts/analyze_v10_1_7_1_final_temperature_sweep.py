#!/usr/bin/env python3
"""Analyze the v10.1.7.1 three-class final production temperature sweep."""
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


def _float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(records: list[dict[str, Any]], key: str) -> float:
    values = [_float(record.get(key)) for record in records]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def _last(records: list[dict[str, Any]], key: str) -> float:
    for record in reversed(records):
        value = _float(record.get(key))
        if math.isfinite(value):
            return value
    return math.nan


def _load_case(case_dir: Path, mode: str, material_class: str,
               temperature: float, checkpoint_advance_um: float) -> dict[str, Any]:
    summary = json.loads((case_dir / "summary.json").read_text())[0]
    audit = json.loads((case_dir / "kinetic_tip_cell_audit_v101.json").read_text())
    records = audit.get("records", [])
    if not records:
        raise ValueError(f"no kinetic records in {case_dir}")

    fired = [record for record in records if bool(record.get("fired", False))]
    if not fired:
        raise ValueError(f"no crack advances in {case_dir}")

    event_k = [_float(record.get("K_Pa_sqrt_m")) / 1.0e6 for record in fired]
    event_k = [value for value in event_k if math.isfinite(value)]
    if not event_k:
        raise ValueError(f"no finite event K values in {case_dir}")

    event_ext = []
    for index, record in enumerate(fired):
        advance_m = _float(
            record.get("kinetic_micro_advance_total_m", record.get("micro_advance_total_m"))
        )
        if math.isfinite(advance_m):
            event_ext.append(advance_m * 1.0e6)
        else:
            event_ext.append((index + 1) * checkpoint_advance_um)
    first_ext = event_ext[0]
    event_ext = [max(value - first_ext, 0.0) for value in event_ext]

    late_n = min(10, len(event_k))
    late_mean = float(np.mean(event_k[-late_n:]))
    final_advance = max(
        _float(record.get("kinetic_micro_advance_total_m", record.get("micro_advance_total_m")), 0.0)
        for record in records
    )
    late_threshold = 0.8 * final_advance
    late_records = [
        record for record in records
        if _float(record.get("kinetic_micro_advance_total_m", record.get("micro_advance_total_m")), 0.0)
        >= late_threshold
    ]
    if not late_records:
        late_records = records[-max(1, len(records) // 5):]

    k_init = event_k[0]
    return {
        "mode": mode,
        "class": material_class,
        "temperature_K": temperature,
        "case_dir": str(case_dir),
        "K_init_MPa_sqrt_m": k_init,
        "K_late_mean_MPa_sqrt_m": late_mean,
        "K_final_event_MPa_sqrt_m": event_k[-1],
        "K_peak_event_MPa_sqrt_m": max(event_k),
        "R_rise_late_MPa_sqrt_m": late_mean - k_init,
        "R_rise_final_MPa_sqrt_m": event_k[-1] - k_init,
        "R_rise_peak_MPa_sqrt_m": max(event_k) - k_init,
        "n_events": len(event_k),
        "n_advances": int(summary.get("n_advances", len(event_k))),
        "final_extension_um": final_advance * 1.0e6,
        "max_active_population": max(
            _float(record.get("developed_state_active_count"), 0.0) for record in records
        ),
        "late_active_population": _mean(late_records, "developed_state_active_count"),
        "late_mobile_population": _mean(late_records, "developed_state_mobile_count"),
        "late_retained_population": _mean(late_records, "developed_state_retained_count"),
        "max_backstress_GPa": max(
            _float(record.get("sigma_emission_backstress_Pa"), 0.0) for record in records
        ) / 1.0e9,
        "late_backstress_GPa": _mean(late_records, "sigma_emission_backstress_Pa") / 1.0e9,
        "max_active_shielding_MPa_sqrt_m": max(
            abs(_float(record.get("campaign_active_K_shield_effective_Pa_sqrt_m"), 0.0))
            for record in records
        ) / 1.0e6,
        "late_active_shielding_MPa_sqrt_m": _mean(
            late_records, "campaign_active_K_shield_effective_Pa_sqrt_m"
        ) / 1.0e6,
        "cumulative_emitted": _last(records, "developed_state_cumulative_emitted"),
        "cumulative_refreshed": _last(records, "developed_state_cumulative_refreshed"),
        "event_extension_um": event_ext,
        "event_K_MPa_sqrt_m": event_k,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    excluded = {"event_extension_um", "event_K_MPa_sqrt_m"}
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in excluded or key in seen:
                continue
            seen.add(key)
            keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in keys} for row in rows)


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["class"], float(row["temperature_K"]), row["mode"]): row
        for row in rows
    }
    paired = []
    keys = sorted({(row["class"], float(row["temperature_K"])) for row in rows})
    for material_class, temperature in keys:
        full = indexed.get((material_class, temperature, "full"))
        off = indexed.get((material_class, temperature, "plasticity_off"))
        if full is None or off is None:
            continue
        paired.append({
            "class": material_class,
            "temperature_K": temperature,
            "plastic_initiation_shift_MPa_sqrt_m": (
                full["K_init_MPa_sqrt_m"] - off["K_init_MPa_sqrt_m"]
            ),
            "plastic_R_rise_late_MPa_sqrt_m": (
                full["R_rise_late_MPa_sqrt_m"] - off["R_rise_late_MPa_sqrt_m"]
            ),
            "plastic_R_rise_final_MPa_sqrt_m": (
                full["R_rise_final_MPa_sqrt_m"] - off["R_rise_final_MPa_sqrt_m"]
            ),
            "plastic_R_rise_peak_MPa_sqrt_m": (
                full["R_rise_peak_MPa_sqrt_m"] - off["R_rise_peak_MPa_sqrt_m"]
            ),
        })
    return paired


def _plot_temperature_metric(root: Path, rows: list[dict[str, Any]], metric: str,
                             ylabel: str, filename: str, mode: str = "full") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for material_class in ("ceramic", "weakT", "DBTT"):
        selected = sorted(
            [row for row in rows if row["class"] == material_class and row["mode"] == mode],
            key=lambda row: float(row["temperature_K"]),
        )
        if selected:
            ax.plot(
                [row["temperature_K"] for row in selected],
                [row[metric] for row in selected],
                marker="o",
                label=material_class,
            )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / filename, dpi=220)
    plt.close(fig)


def _plots(root: Path, rows: list[dict[str, Any]], paired: list[dict[str, Any]]) -> None:
    _plot_temperature_metric(
        root, rows, "K_init_MPa_sqrt_m", "Initiation toughness (MPa√m)",
        "production_K_init_vs_temperature.png",
    )
    _plot_temperature_metric(
        root, rows, "K_late_mean_MPa_sqrt_m", "Late resistance (MPa√m)",
        "production_K_late_vs_temperature.png",
    )
    _plot_temperature_metric(
        root, rows, "R_rise_late_MPa_sqrt_m", "Late R-curve rise (MPa√m)",
        "production_R_rise_vs_temperature.png",
    )
    _plot_temperature_metric(
        root, rows, "late_active_population", "Late active population",
        "production_late_active_population_vs_temperature.png",
    )

    for material_class in ("ceramic", "weakT", "DBTT"):
        selected = sorted(
            [row for row in rows if row["class"] == material_class and row["mode"] == "full"],
            key=lambda row: float(row["temperature_K"]),
        )
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for row in selected:
            ax.plot(
                row["event_extension_um"], row["event_K_MPa_sqrt_m"],
                marker="o", markersize=3, label=f"{row['temperature_K']:g} K",
            )
        ax.set_xlabel("Crack extension after initiation (µm)")
        ax.set_ylabel("K at advance (MPa√m)")
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(root / f"production_R_curves_{material_class}.png", dpi=220)
        plt.close(fig)

    if paired:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for material_class in ("ceramic", "weakT", "DBTT"):
            selected = sorted(
                [row for row in paired if row["class"] == material_class],
                key=lambda row: float(row["temperature_K"]),
            )
            if selected:
                ax.plot(
                    [row["temperature_K"] for row in selected],
                    [row["plastic_R_rise_late_MPa_sqrt_m"] for row in selected],
                    marker="o", label=material_class,
                )
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Matched plastic late R-rise (MPa√m)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "production_matched_plastic_R_rise_vs_temperature.png", dpi=220)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--temperatures", nargs="+", type=float, required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--theta", type=float, default=45.0)
    parser.add_argument("--checkpoint-advance-um", type=float, default=5.0)
    args = parser.parse_args()

    root = args.root.resolve()
    rows = []
    events: dict[str, Any] = {}
    for mode in args.modes:
        events[mode] = {}
        for material_class in args.classes:
            events[mode][material_class] = {}
            for temperature in args.temperatures:
                case_dir = root / mode / material_class / f"T{temperature:g}_th{args.theta:g}"
                row = _load_case(
                    case_dir, mode, material_class, temperature,
                    args.checkpoint_advance_um,
                )
                rows.append(row)
                events[mode][material_class][f"{temperature:g}"] = {
                    "extension_um": row["event_extension_um"],
                    "K_MPa_sqrt_m": row["event_K_MPa_sqrt_m"],
                }

    rows.sort(key=lambda row: (row["mode"], row["class"], float(row["temperature_K"])))
    paired = _paired_rows(rows)
    _write_csv(root / "final_production_case_summary.csv", rows)
    _write_csv(root / "final_production_matched_ablation_summary.csv", paired)
    (root / "final_production_event_curves.json").write_text(json.dumps(events, indent=2))
    assessment = {
        "schema": "v10.1.7.1_final_production_temperature_sweep",
        "classes": args.classes,
        "temperatures_K": args.temperatures,
        "modes": args.modes,
        "n_cases": len(rows),
        "all_requested_cases_loaded": len(rows) == len(args.classes) * len(args.temperatures) * len(args.modes),
        "minimum_final_extension_um": min(row["final_extension_um"] for row in rows),
        "maximum_final_extension_um": max(row["final_extension_um"] for row in rows),
    }
    (root / "final_production_assessment.json").write_text(json.dumps(assessment, indent=2))
    _plots(root, rows, paired)

    print("\nFinal production sweep summary")
    print("class      T(K)  mode             Kinit    Klate    Rlate  events  ext_um")
    for row in rows:
        print(
            f"{row['class']:<9s} {row['temperature_K']:5.0f}  {row['mode']:<15s} "
            f"{row['K_init_MPa_sqrt_m']:7.3f}  {row['K_late_mean_MPa_sqrt_m']:7.3f}  "
            f"{row['R_rise_late_MPa_sqrt_m']:7.3f}  {row['n_events']:6d}  "
            f"{row['final_extension_um']:7.1f}"
        )
    print(root / "final_production_assessment.json")


if __name__ == "__main__":
    main()
