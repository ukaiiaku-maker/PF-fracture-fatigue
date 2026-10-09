#!/usr/bin/env python3
"""Aggregate provenance-complete four-class fatigue growth and censor records."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCHEMA = "v10.2.30_four_class_fatigue_campaign/1"


def _read(path: Path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        if not fields:
            return
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _case(root: Path) -> tuple[dict, list[dict]]:
    growth = _read(root / "developed_fatigue_growth_summary.json", {})
    control = _read(root / "v10_2_30_fixed_deltaK_control.json", {})
    provenance = growth.get("provenance", {})
    exit_code_path = root / "exit_code.txt"
    exit_code = int(exit_code_path.read_text().strip()) if exit_code_path.is_file() else 0
    reached = bool(growth.get("target_reached", False))
    event_count = int(growth.get("event_count", 0))
    if reached:
        status = "completed"
        reason = f"post_target_wrapper_exit_{exit_code}" if exit_code else None
    elif growth.get("status") == "no_committed_crack_event" or event_count == 0:
        status = "censored"
        reason = control.get("censor_status", "right_censored_no_event")
    else:
        status = "failed"
        reason = f"wrapper_exit_{exit_code}" if exit_code else str(growth.get("status"))
    developed = growth.get("developed_interval", {})
    row = {
        "run_root": str(root),
        "parameter_option": provenance.get("parameter_option"),
        "status": status,
        "censor_or_failure_reason": reason,
        "event_count": event_count,
        "cycles_consumed": growth.get("cycles_consumed"),
        "final_projected_extension_um": growth.get("final_projected_extension_um"),
        "target_reached": reached,
        "stable_growth_provisional": bool(growth.get("stable_growth_provisional", False)),
        "developed_event_count": developed.get("event_count"),
        "developed_da_dN_m_per_cycle": developed.get("da_dN"),
        "deltaK_MPa_sqrt_m": provenance.get("deltaK_MPa_sqrt_m"),
        "hazard_seed": provenance.get("hazard_seed"),
        "git_head": provenance.get("git_head"),
        "exit_code": exit_code,
    }
    events = []
    for event in growth.get("event_measurements", []):
        item = dict(event)
        item["run_root"] = str(root)
        item["case_status"] = status
        item.setdefault("parameter_option", row["parameter_option"])
        item.setdefault("deltaK_MPa_sqrt_m", row["deltaK_MPa_sqrt_m"])
        item.setdefault("hazard_seed", row["hazard_seed"])
        events.append(item)
    return row, events


def _paris_fit(parameter_option: str, rows: list[dict]):
    selected = [
        row for row in rows
        if row.get("parameter_option") == parameter_option
        and row.get("status") == "completed"
        and row.get("stable_growth_provisional")
        and row.get("deltaK_MPa_sqrt_m") not in (None, 0.0)
        and row.get("developed_da_dN_m_per_cycle") not in (None, 0.0)
    ]
    selected.sort(key=lambda row: float(row["deltaK_MPa_sqrt_m"]))
    if len(selected) < 2:
        return None, []
    x = np.log(np.asarray([row["deltaK_MPa_sqrt_m"] for row in selected], dtype=float))
    y = np.log(np.asarray([row["developed_da_dN_m_per_cycle"] for row in selected], dtype=float))
    m, log_c = np.polyfit(x, y, 1)
    predicted = log_c + m * x
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if total == 0.0 else 1.0 - residual / total
    local = []
    for left, right in zip(selected, selected[1:]):
        dk0 = float(left["deltaK_MPa_sqrt_m"])
        dk1 = float(right["deltaK_MPa_sqrt_m"])
        rate0 = float(left["developed_da_dN_m_per_cycle"])
        rate1 = float(right["developed_da_dN_m_per_cycle"])
        local.append({
            "parameter_option": parameter_option,
            "deltaK_low_MPa_sqrt_m": dk0,
            "deltaK_high_MPa_sqrt_m": dk1,
            "local_m": math.log(rate1 / rate0) / math.log(dk1 / dk0),
        })
    fit = {
        "parameter_option": parameter_option,
        "point_count": len(selected),
        "m": float(m),
        "C_m_per_cycle": float(math.exp(log_c)),
        "R2": r2,
        "response_class": "credible_low_slope" if m <= 8.0 else "steep_or_threshold_like",
    }
    return fit, local


def _plot(rows: list[dict], out: Path) -> None:
    valid = [
        row for row in rows
        if row.get("status") == "completed"
        and row.get("deltaK_MPa_sqrt_m") not in (None, 0.0)
        and row.get("developed_da_dN_m_per_cycle") not in (None, 0.0)
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for option in sorted({str(row["parameter_option"]) for row in valid}):
        selected = [row for row in valid if str(row["parameter_option"]) == option]
        ax.scatter(
            [row["deltaK_MPa_sqrt_m"] for row in selected],
            [row["developed_da_dN_m_per_cycle"] for row in selected],
            label=option,
        )
    if valid:
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    ax.set_xlabel("Delta K (MPa sqrt(m))")
    ax.set_ylabel("Developed da/dN (m/cycle)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "four_class_da_dN_vs_deltaK.png", dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_roots", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cases, events = [], []
    for root in args.run_roots:
        case, case_events = _case(root.resolve())
        cases.append(case)
        events.extend(case_events)
    fits, slopes = [], []
    for option in sorted({str(row.get("parameter_option")) for row in cases}):
        fit, local = _paris_fit(option, cases)
        if fit is not None:
            fits.append(fit)
            slopes.extend(local)
    payload = {
        "schema": SCHEMA,
        "case_count": len(cases),
        "completed_count": sum(row["status"] == "completed" for row in cases),
        "censored_count": sum(row["status"] == "censored" for row in cases),
        "failed_count": sum(row["status"] == "failed" for row in cases),
        "event_interval_count": len(events),
        "cases": cases,
        "paris_fits": fits,
        "local_slopes": slopes,
    }
    _write_csv(args.out / "four_class_fatigue_cases.csv", cases)
    _write_csv(args.out / "four_class_fatigue_censor_failure_table.csv", [
        row for row in cases if row.get("censor_or_failure_reason")
    ])
    _write_csv(args.out / "four_class_event_intervals.csv", events)
    _plot(cases, args.out)
    (args.out / "four_class_fatigue_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
