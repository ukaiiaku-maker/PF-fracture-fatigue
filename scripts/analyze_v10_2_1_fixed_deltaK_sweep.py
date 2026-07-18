#!/usr/bin/env python3
"""Analyze v10.2.1 fixed-DeltaK cases and construct da/dN versus DeltaK.

The first event is treated as fatigue initiation.  Propagation rates use only
subsequent event intervals:

    rate_i = da_i / (N_i - N_{i-1}),  i >= 2

A seed-level interval rate is also reported from the first event through the
last event.  Right-censored and initiation-only cases remain in the case table
but are not converted into artificial positive propagation rates.
"""
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


def _read_numeric_csv(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"empty CSV: {path}")
    header = [item.strip() for item in lines[0].lstrip("# ").split(",")]
    rows: list[dict[str, float]] = []
    for line in lines[1:]:
        values = [item.strip() for item in line.split(",")]
        if len(values) != len(header):
            continue
        row: dict[str, float] = {}
        for key, value in zip(header, values):
            try:
                row[key] = float(value)
            except ValueError:
                row[key] = float("nan")
        rows.append(row)
    return header, rows


def _key(header: list[str], *candidates: str) -> str:
    for candidate in candidates:
        if candidate in header:
            return candidate
    raise KeyError(f"none of {candidates} found in columns {header}")


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")


def analyze_case(audit_path: Path) -> tuple[dict, list[dict]]:
    root = audit_path.parent
    control = json.loads(audit_path.read_text())
    fatigue_audit_path = root / "v10_2_0_fatigue_reintegration.json"
    fatigue_audit = json.loads(fatigue_audit_path.read_text()) if fatigue_audit_path.is_file() else {}
    step_paths = sorted(root.glob("steps_*K.csv"))
    if len(step_paths) != 1:
        raise RuntimeError(f"expected one steps CSV in {root}, found {step_paths}")
    header, rows = _read_numeric_csv(step_paths[0])
    cycle_key = _key(header, "fatigue_cycles", "cycles")
    da_key = _key(header, "da_block_m", "crack_advance_block_m")
    extension_key = _key(header, "crack_extension_m", "a_extension_m")

    cumulative_cycles = 0.0
    event_rows: list[dict] = []
    for index, row in enumerate(rows):
        cycles = max(float(row.get(cycle_key, 0.0)), 0.0)
        cumulative_cycles += cycles
        da = max(float(row.get(da_key, 0.0)), 0.0)
        if da > 1.0e-15:
            event_rows.append({
                "row_index": index,
                "cycles_cumulative": cumulative_cycles,
                "event_advance_m": da,
                "crack_extension_m": max(float(row.get(extension_key, 0.0)), 0.0),
            })

    target_deltaK = float(control["target_deltaK_MPa_sqrt_m"])
    seed = int(fatigue_audit.get("cleavage_hazard_seed", -1))
    event_rates: list[dict] = []
    for event_index in range(1, len(event_rows)):
        previous = event_rows[event_index - 1]
        current = event_rows[event_index]
        dN = current["cycles_cumulative"] - previous["cycles_cumulative"]
        da = current["event_advance_m"]
        if dN > 0.0 and da > 0.0:
            event_rates.append({
                "case_dir": str(root),
                "target_deltaK_MPa_sqrt_m": target_deltaK,
                "seed": seed,
                "propagation_event_index": event_index,
                "cycles_interval": dN,
                "event_advance_m": da,
                "da_dN_m_per_cycle": da / dN,
                "cycles_cumulative": current["cycles_cumulative"],
                "crack_extension_m": current["crack_extension_m"],
            })

    cycles_to_first = event_rows[0]["cycles_cumulative"] if event_rows else float("nan")
    interval_rate = float("nan")
    propagation_extension = 0.0
    propagation_cycles = 0.0
    if len(event_rows) >= 2:
        propagation_extension = sum(event["event_advance_m"] for event in event_rows[1:])
        propagation_cycles = (
            event_rows[-1]["cycles_cumulative"] - event_rows[0]["cycles_cumulative"]
        )
        if propagation_cycles > 0.0:
            interval_rate = propagation_extension / propagation_cycles

    if len(event_rows) == 0:
        status = "right_censored_no_event"
    elif len(event_rows) == 1:
        status = "initiated_only"
    else:
        status = "propagated"

    case = {
        "case_dir": str(root),
        "target_deltaK_MPa_sqrt_m": target_deltaK,
        "target_Kmax_MPa_sqrt_m": float(control["target_Kmax_MPa_sqrt_m"]),
        "R": float(control["R"]),
        "frequency_Hz": float(control["frequency_Hz"]),
        "seed": seed,
        "status": status,
        "event_count": len(event_rows),
        "propagation_event_count": max(len(event_rows) - 1, 0),
        "cycles_total": cumulative_cycles,
        "cycles_to_first_event": cycles_to_first,
        "crack_extension_total_m": max(float(rows[-1].get(extension_key, 0.0)), 0.0) if rows else 0.0,
        "propagation_extension_m": propagation_extension,
        "propagation_cycles": propagation_cycles,
        "interval_da_dN_m_per_cycle": interval_rate,
        "fixed_deltaK_max_abs_error_Pa_sqrt_m": float(
            control.get("maximum_abs_target_error_Pa_sqrt_m", float("nan"))
        ),
        "fixed_deltaK_exact": bool(
            control.get("fixed_deltaK_exact_within_relative_1e-12", False)
        ),
    }
    return case, event_rates


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_cases(cases: list[dict], event_rates: list[dict]) -> list[dict]:
    summary: list[dict] = []
    for deltaK in sorted({float(row["target_deltaK_MPa_sqrt_m"]) for row in cases}):
        subset = [row for row in cases if float(row["target_deltaK_MPa_sqrt_m"]) == deltaK]
        rates = [
            float(row["interval_da_dN_m_per_cycle"])
            for row in subset
            if math.isfinite(float(row["interval_da_dN_m_per_cycle"]))
            and float(row["interval_da_dN_m_per_cycle"]) > 0.0
        ]
        pooled = [
            float(row["da_dN_m_per_cycle"])
            for row in event_rates
            if float(row["target_deltaK_MPa_sqrt_m"]) == deltaK
            and float(row["da_dN_m_per_cycle"]) > 0.0
        ]
        geometric = (
            float(np.exp(np.mean(np.log(np.asarray(rates))))) if rates else float("nan")
        )
        summary.append({
            "target_deltaK_MPa_sqrt_m": deltaK,
            "n_seeds": len(subset),
            "n_propagating_seeds": len(rates),
            "n_right_censored": sum(row["status"] == "right_censored_no_event" for row in subset),
            "n_initiated_only": sum(row["status"] == "initiated_only" for row in subset),
            "median_interval_da_dN_m_per_cycle": float(np.median(rates)) if rates else float("nan"),
            "geometric_mean_interval_da_dN_m_per_cycle": geometric,
            "p16_interval_da_dN_m_per_cycle": _percentile(rates, 16.0),
            "p84_interval_da_dN_m_per_cycle": _percentile(rates, 84.0),
            "pooled_event_count": len(pooled),
            "median_event_da_dN_m_per_cycle": float(np.median(pooled)) if pooled else float("nan"),
        })
    return summary


def render_plot(out: Path, cases: list[dict], summary: list[dict]) -> dict:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for row in cases:
        rate = float(row["interval_da_dN_m_per_cycle"])
        if math.isfinite(rate) and rate > 0.0:
            ax.plot(
                float(row["target_deltaK_MPa_sqrt_m"]), rate,
                marker="o", markersize=6, markerfacecolor="none", linestyle="none",
                alpha=0.55,
            )
    valid = [
        row for row in summary
        if math.isfinite(float(row["median_interval_da_dN_m_per_cycle"]))
        and float(row["median_interval_da_dN_m_per_cycle"]) > 0.0
    ]
    if valid:
        x = np.asarray([float(row["target_deltaK_MPa_sqrt_m"]) for row in valid])
        y = np.asarray([float(row["median_interval_da_dN_m_per_cycle"]) for row in valid])
        lo = y - np.asarray([float(row["p16_interval_da_dN_m_per_cycle"]) for row in valid])
        hi = np.asarray([float(row["p84_interval_da_dN_m_per_cycle"]) for row in valid]) - y
        ax.errorbar(x, y, yerr=np.vstack([np.maximum(lo, 0.0), np.maximum(hi, 0.0)]),
                    marker="s", markersize=7, markerfacecolor="none", linewidth=1.5,
                    capsize=3, label="Seed median, 16–84%")
    fit = {"paris_fit_available": False, "paris_exponent_m": None, "paris_prefactor_C_SI": None}
    if len(valid) >= 3:
        x = np.asarray([float(row["target_deltaK_MPa_sqrt_m"]) for row in valid])
        y = np.asarray([float(row["geometric_mean_interval_da_dN_m_per_cycle"]) for row in valid])
        mask = np.isfinite(y) & (y > 0.0)
        if np.count_nonzero(mask) >= 3:
            slope, intercept = np.polyfit(np.log10(x[mask]), np.log10(y[mask]), 1)
            xx = np.geomspace(np.min(x[mask]), np.max(x[mask]), 100)
            yy = 10.0 ** intercept * xx ** slope
            ax.plot(xx, yy, linestyle="--", label=f"Log–log fit, m={slope:.2f}")
            fit = {
                "paris_fit_available": True,
                "paris_exponent_m": float(slope),
                "paris_prefactor_C_SI": float(10.0 ** intercept),
                "fit_deltaK_unit": "MPa*sqrt(m)",
                "fit_rate_unit": "m/cycle",
            }
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta K$ (MPa$\sqrt{\mathrm{m}}$)")
    ax.set_ylabel(r"$da/dN$ (m/cycle)")
    if valid:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fixed_deltaK_da_dN_vs_deltaK.png", dpi=300)
    plt.close(fig)
    return fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    audits = sorted(root.rglob("v10_2_1_fixed_deltaK_control.json"))
    if not audits:
        raise SystemExit(f"no v10.2.1 fixed-DeltaK cases found below {root}")

    cases: list[dict] = []
    events: list[dict] = []
    for audit in audits:
        case, event_rows = analyze_case(audit)
        cases.append(case)
        events.extend(event_rows)
    summary = aggregate_cases(cases, events)
    _write_csv(root / "fixed_deltaK_case_summary.csv", cases)
    _write_csv(root / "fixed_deltaK_event_rates.csv", events)
    _write_csv(root / "fixed_deltaK_paris_summary.csv", summary)
    fit = render_plot(root, cases, summary)
    assessment = {
        "schema": "v10.2.1_fixed_deltaK_paris_assessment",
        "n_cases": len(cases),
        "n_deltaK_levels": len(summary),
        "n_propagating_cases": sum(row["status"] == "propagated" for row in cases),
        "n_right_censored_cases": sum(row["status"] == "right_censored_no_event" for row in cases),
        "all_fixed_deltaK_exact": all(bool(row["fixed_deltaK_exact"]) for row in cases),
        **fit,
    }
    (root / "fixed_deltaK_paris_assessment.json").write_text(json.dumps(assessment, indent=2))
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
