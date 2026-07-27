#!/usr/bin/env python3
"""Aggregate v10.2.29 developed-state fixed-DeltaK fatigue measurements."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCHEMA = "v10.2.29_developed_fatigue_campaign_v1"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def load_cases(root: Path) -> tuple[list[dict], list[dict]]:
    case_rows: list[dict] = []
    measurement_rows: list[dict] = []
    for path in sorted(root.rglob("fatigue_developed_growth_*K.json")):
        payload = json.loads(path.read_text())
        if payload.get("schema") != "v10.2.29_developed_fatigue_growth_v1":
            continue
        summary = dict(payload["summary"])
        run_root = path.parent
        control_path = run_root / "v10_2_29_fixed_deltaK_control.json"
        control = json.loads(control_path.read_text()) if control_path.is_file() else {}
        case = {
            "run_root": str(run_root),
            "parameter_option": payload.get("parameter_option"),
            "candidate_id": control.get("candidate_id"),
            "temperature_K": float(payload["temperature_K"]),
            "target_deltaK_MPa_sqrt_m": float(payload["target_deltaK_MPa_sqrt_m"]),
            "target_Kmax_MPa_sqrt_m": float(payload["target_Kmax_MPa_sqrt_m"]),
            "R": float(payload["R"]),
            "frequency_Hz": float(payload["frequency_Hz"]),
            "hazard_seed": control.get("cleavage_hazard_seed"),
            **summary,
        }
        case_rows.append(case)
        for row in payload.get("developed_measurements", []):
            item = {
                "run_root": str(run_root),
                "parameter_option": payload.get("parameter_option"),
                "temperature_K": float(payload["temperature_K"]),
                "target_deltaK_MPa_sqrt_m": float(payload["target_deltaK_MPa_sqrt_m"]),
                "target_Kmax_MPa_sqrt_m": float(payload["target_Kmax_MPa_sqrt_m"]),
                "R": float(payload["R"]),
                "frequency_Hz": float(payload["frequency_Hz"]),
                "hazard_seed": control.get("cleavage_hazard_seed"),
                **row,
            }
            measurement_rows.append(item)
    return case_rows, measurement_rows


def aggregate(cases: list[dict], measurements: list[dict]) -> list[dict]:
    keys = sorted({
        (
            str(row["parameter_option"]),
            float(row["temperature_K"]),
            float(row["target_deltaK_MPa_sqrt_m"]),
        )
        for row in cases
    })
    rows: list[dict] = []
    for option, temperature, deltaK in keys:
        case_subset = [
            row for row in cases
            if str(row["parameter_option"]) == option
            and float(row["temperature_K"]) == temperature
            and float(row["target_deltaK_MPa_sqrt_m"]) == deltaK
        ]
        measure_subset = [
            row for row in measurements
            if str(row["parameter_option"]) == option
            and float(row["temperature_K"]) == temperature
            and float(row["target_deltaK_MPa_sqrt_m"]) == deltaK
        ]
        rates = [
            float(row["da_dN_m_per_cycle"])
            for row in measure_subset
            if math.isfinite(float(row["da_dN_m_per_cycle"]))
            and float(row["da_dN_m_per_cycle"]) > 0.0
        ]
        extensions = [
            float(row["extension_since_initiation_mid_m"])
            for row in measure_subset
            if math.isfinite(float(row["extension_since_initiation_mid_m"]))
        ]
        rows.append({
            "parameter_option": option,
            "temperature_K": temperature,
            "target_deltaK_MPa_sqrt_m": deltaK,
            "target_Kmax_MPa_sqrt_m": (
                float(case_subset[0]["target_Kmax_MPa_sqrt_m"])
                if case_subset else float("nan")
            ),
            "n_cases": len(case_subset),
            "n_right_censored_no_event": sum(
                row["status"] == "right_censored_no_event" for row in case_subset
            ),
            "n_initiated_only": sum(
                row["status"] == "initiated_only" for row in case_subset
            ),
            "n_predeveloped_propagation": sum(
                row["status"] == "propagated_before_developed_window"
                for row in case_subset
            ),
            "n_cases_with_developed_measurements": sum(
                int(row["developed_measurement_count"]) > 0 for row in case_subset
            ),
            "n_developed_intervals": len(rates),
            "median_da_dN_m_per_cycle": (
                float(np.median(rates)) if rates else float("nan")
            ),
            "p16_da_dN_m_per_cycle": percentile(rates, 16.0),
            "p84_da_dN_m_per_cycle": percentile(rates, 84.0),
            "geometric_mean_da_dN_m_per_cycle": (
                float(np.exp(np.mean(np.log(np.asarray(rates, dtype=float)))))
                if rates else float("nan")
            ),
            "median_extension_since_initiation_m": (
                float(np.median(extensions)) if extensions else float("nan")
            ),
        })
    return rows


def plot_summary(
    path: Path,
    rows: list[dict],
    *,
    x_key: str,
    x_label: str,
) -> None:
    valid = [
        row for row in rows
        if math.isfinite(float(row["median_da_dN_m_per_cycle"]))
        and float(row["median_da_dN_m_per_cycle"]) > 0.0
    ]
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    if valid:
        x = np.asarray([float(row[x_key]) for row in valid])
        y = np.asarray([float(row["median_da_dN_m_per_cycle"]) for row in valid])
        lo = y - np.asarray([float(row["p16_da_dN_m_per_cycle"]) for row in valid])
        hi = np.asarray([float(row["p84_da_dN_m_per_cycle"]) for row in valid]) - y
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([np.maximum(lo, 0.0), np.maximum(hi, 0.0)]),
            marker="o",
            linestyle="-",
            capsize=3,
        )
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Developed-state $da/dN$ (m/cycle)")
    ax.set_title("Rates after the configured crack-extension development window")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    cases, measurements = load_cases(root)
    if not cases:
        raise SystemExit(f"no developed-fatigue case outputs found below {root}")
    summary = aggregate(cases, measurements)

    write_csv(root / "developed_fatigue_case_summary.csv", cases)
    write_csv(root / "developed_fatigue_event_rates.csv", measurements)
    write_csv(root / "developed_fatigue_deltaK_summary.csv", summary)
    plot_summary(
        root / "developed_fatigue_da_dN_vs_deltaK.png",
        summary,
        x_key="target_deltaK_MPa_sqrt_m",
        x_label=r"$\Delta K$ (MPa$\sqrt{\mathrm{m}}$)",
    )
    plot_summary(
        root / "developed_fatigue_da_dN_vs_Kmax.png",
        summary,
        x_key="target_Kmax_MPa_sqrt_m",
        x_label=r"$K_{\max}$ (MPa$\sqrt{\mathrm{m}}$)",
    )

    payload = {
        "schema": SCHEMA,
        "root": str(root),
        "case_count": len(cases),
        "developed_interval_count": len(measurements),
        "deltaK_level_count": len(summary),
        "initiation_excluded": True,
        "smoothing_or_Paris_fit_applied": False,
        "summary": summary,
    }
    (root / "developed_fatigue_campaign.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
