#!/usr/bin/env python3
"""Build four-class v10.2.30 driving-force/rate endpoint data products."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


CASE_RE = re.compile(r"(?P<label>peak|dbtt|weakT|ceramic)_f(?P<whole>\d+)p(?P<frac>\d+)_seed", re.I)
CLASS_ORDER = ("peak", "dbtt", "weakt", "ceramic")


def fraction(path: str) -> tuple[str, float] | None:
    match = CASE_RE.search(path)
    if not match:
        return None
    token = f"{match.group('whole')}.{match.group('frac')}"
    return match.group("label").lower(), float(token)


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def locator_max(output: Path) -> int | None:
    pointer = output / "run_state_checkpoint.json"
    if not pointer.exists():
        return None
    try:
        generation = json.loads(pointer.read_text())["generation"]
        outer = json.loads((output / "run_state_generations" / generation / "outer.json").read_text())
    except (KeyError, OSError, json.JSONDecodeError):
        return None
    trials = []
    for audit in outer.get("history", {}).get("kinetic_audit_records", []):
        for mode in audit.get("coupled_hazard_modes", []):
            if mode.get("mode") == "first_passage_cycle_locator":
                value = mode.get("trial_evaluations")
                if value is not None:
                    trials.append(int(value))
    return max(trials) if trials else None


def regime(row: dict) -> str:
    """Classify using event-spacing distribution plus rate, not life alone."""
    if row.get("status") == "censored" or finite(row.get("developed_da_dN_m_per_cycle")) is None:
        return "VHCF_OR_CENSORED"
    subcycle = finite(row.get("fraction_subcycle_intervals")) or 0.0
    median = finite(row.get("median_event_spacing_cycles"))
    rate = finite(row.get("developed_da_dN_m_per_cycle")) or 0.0
    if subcycle >= 0.5 and median is not None and median < 1.0:
        return "NEAR_MONOTONIC_CYCLIC_FAILURE"
    if rate < 1e-15:
        return "VHCF"
    if rate < 1e-12:
        return "HCF"
    if rate < 1e-9:
        return "ACCELERATED_FATIGUE"
    return "LCF"


def case_row(case: dict) -> dict | None:
    parsed = fraction(case["run_path"])
    if parsed is None:
        return None
    label, f = parsed
    output = Path(case["run_path"])
    developed_path = output / "developed_fatigue_growth_summary.json"
    developed = json.loads(developed_path.read_text()) if developed_path.exists() else {}
    events = developed.get("event_measurements", [])
    intervals = np.asarray([
        float(event["cycles_between_events"]) for event in events
        if finite(event.get("cycles_between_events")) is not None
    ])
    event_rates = np.asarray([
        float(event["da_dN_m_per_cycle"]) for event in events
        if finite(event.get("da_dN_m_per_cycle")) is not None
    ])
    tortuosity = np.asarray([
        float(event["tortuosity"]) for event in events
        if finite(event.get("tortuosity")) is not None
    ])
    windows = developed.get("moving_windows", [])
    row = {
        "class": label,
        "f": f,
        "parameter_option": case.get("parameter_option"),
        "seed": case.get("hazard_seed"),
        "status": case.get("status"),
        "deltaK_MPa_sqrt_m": case.get("deltaK_MPa_sqrt_m"),
        "Kmax_MPa_sqrt_m": case.get("Kmax_MPa_sqrt_m"),
        "Kmin_MPa_sqrt_m": (finite(case.get("Kmax_MPa_sqrt_m")) or 0.0) * (finite(case.get("R")) or 0.0),
        "R": case.get("R"),
        "frequency_Hz": case.get("frequency_Hz"),
        "temperature_K": case.get("temperature_K"),
        "cycles_to_first_event": events[0].get("cycles_post") if events else None,
        "cycles_to_target": case.get("cycles_reached") if case.get("status") == "completed" else None,
        "projected_extension_um": case.get("projected_extension_um"),
        "path_extension_um": (events[-1].get("path_extension_post_m", 0.0) * 1e6 if events else None),
        "event_count": case.get("event_count"),
        "developed_da_dN_m_per_cycle": case.get("developed_da_dN_m_per_cycle"),
        "stable_growth_provisional": case.get("stable_growth_provisional", False),
        "event_rate_cv": (float(np.std(event_rates, ddof=1) / np.mean(event_rates))
                          if event_rates.size > 1 and np.mean(event_rates) else None),
        "event_da_dN_min": float(np.min(event_rates)) if event_rates.size else None,
        "event_da_dN_max": float(np.max(event_rates)) if event_rates.size else None,
        "mean_event_spacing_cycles": float(np.mean(intervals)) if intervals.size else None,
        "median_event_spacing_cycles": float(np.median(intervals)) if intervals.size else None,
        "min_event_spacing_cycles": float(np.min(intervals)) if intervals.size else None,
        "max_event_spacing_cycles": float(np.max(intervals)) if intervals.size else None,
        "fraction_subcycle_intervals": float(np.mean(intervals < 1.0)) if intervals.size else None,
        "mean_tortuosity": float(np.mean(tortuosity)) if tortuosity.size else None,
        "late_to_early_rate_ratio": developed.get("late_to_early_rate_ratio"),
        "window_0_25_da_dN": windows[0].get("da_dN") if len(windows) > 0 else None,
        "window_25_50_da_dN": windows[1].get("da_dN") if len(windows) > 1 else None,
        "window_50_75_da_dN": windows[2].get("da_dN") if len(windows) > 2 else None,
        "window_75_100_da_dN": windows[3].get("da_dN") if len(windows) > 3 else None,
        "locator_max_trials": locator_max(output),
        "restart_generation": (json.loads((output / "run_state_checkpoint.json").read_text()).get("generation")
                               if (output / "run_state_checkpoint.json").exists() else None),
        "output_root": str(output),
    }
    row["regime"] = regime(row)
    return row


def fit_rows(rows: list[dict], label: str) -> dict | None:
    selected = [row for row in rows if row["class"] == label and
                .875 <= row["f"] <= .975 and
                finite(row["developed_da_dN_m_per_cycle"]) and
                finite(row["deltaK_MPa_sqrt_m"])]
    if len(selected) < 3:
        return None
    x = np.log10([row["deltaK_MPa_sqrt_m"] for row in selected])
    y = np.log10([row["developed_da_dN_m_per_cycle"] for row in selected])
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "class": label,
        "f_range": [min(row["f"] for row in selected), max(row["f"] for row in selected)],
        "deltaK_range_MPa_sqrt_m": [min(row["deltaK_MPa_sqrt_m"] for row in selected),
                                     max(row["deltaK_MPa_sqrt_m"] for row in selected)],
        "m": float(slope),
        "log10_prefactor": float(intercept),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot else 1.0,
        "point_count": len(selected),
        "interpretation": "descriptive local power-law fit; no Paris law used by the solver",
    }


def plot(rows: list[dict], xkey: str, ykey: str, ylabel: str, filename: Path,
         *, ylim: tuple[float, float] | None = None) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for label, marker in zip(CLASS_ORDER, ("o", "s", "^", "D")):
        selected = sorted((row for row in rows if row["class"] == label and
                           finite(row.get(xkey)) and finite(row.get(ykey))), key=lambda row: row[xkey])
        axis.plot([row[xkey] for row in selected], [row[ykey] for row in selected],
                  marker=marker, label=label.upper(), linewidth=1.6)
    axis.set_xlabel("f" if xkey == "f" else r"$\Delta K$ (MPa $\sqrt{m}$)")
    axis.set_ylabel(ylabel)
    axis.set_yscale("log")
    if ylim:
        axis.set_ylim(*ylim)
    axis.grid(True, which="both", alpha=.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def rate_endpoints(rows: list[dict], target: float = 1e-5) -> list[dict]:
    endpoints = []
    for label in CLASS_ORDER:
        selected = sorted((row for row in rows if row["class"] == label and
                           finite(row.get("developed_da_dN_m_per_cycle"))), key=lambda row: row["f"])
        if not selected:
            continue
        nearest = min(selected, key=lambda row: abs(math.log10(row["developed_da_dN_m_per_cycle"] / target)))
        low = max((row for row in selected if row["developed_da_dN_m_per_cycle"] <= target),
                  key=lambda row: row["developed_da_dN_m_per_cycle"], default=None)
        high = min((row for row in selected if row["developed_da_dN_m_per_cycle"] >= target),
                   key=lambda row: row["developed_da_dN_m_per_cycle"], default=None)
        endpoints.append({
            "class": label,
            "target_da_dN_m_per_cycle": target,
            "nearest": {key: nearest.get(key) for key in
                        ("f", "deltaK_MPa_sqrt_m", "cycles_to_target", "developed_da_dN_m_per_cycle", "regime")},
            "lower_bracket_f": low.get("f") if low else None,
            "lower_bracket_rate": low.get("developed_da_dN_m_per_cycle") if low else None,
            "upper_bracket_f": high.get("f") if high else None,
            "upper_bracket_rate": high.get("developed_da_dN_m_per_cycle") if high else None,
        })
    return endpoints


def regime_map(rows: list[dict], filename: Path) -> None:
    names = ["VHCF", "HCF", "ACCELERATED_FATIGUE", "LCF", "NEAR_MONOTONIC_CYCLIC_FAILURE"]
    ordinate = {name: index for index, name in enumerate(names)}
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for label, marker in zip(CLASS_ORDER, ("o", "s", "^", "D")):
        selected = [row for row in rows if row["class"] == label and row["regime"] in ordinate]
        axis.scatter([row["deltaK_MPa_sqrt_m"] for row in selected],
                     [ordinate[row["regime"]] for row in selected], marker=marker, label=label.upper())
    axis.set_yticks(range(len(names)), [name.replace("_", " ") for name in names])
    axis.set_xlabel(r"$\Delta K$ (MPa $\sqrt{m}$)")
    axis.grid(True, alpha=.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text())
    # Later aggregate entries are deliberate refinements/repeats and supersede
    # earlier copies of the same class/f point.
    deduplicated = {}
    for case in payload["cases"]:
        row = case_row(case)
        if row is not None:
            deduplicated[(row["class"], row["f"])] = row
    rows = list(deduplicated.values())
    rows.sort(key=lambda row: (row["class"], row["f"]))
    args.out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.out / "four_class_driving_force_ladder.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fits = [fit for label in CLASS_ORDER if (fit := fit_rows(rows, label))]
    result = {"schema": "v10.2.30_four_class_driving_force_ladder_v2", "cases": rows,
              "descriptive_power_law_fits": fits, "rate_endpoints": rate_endpoints(rows)}
    (args.out / "four_class_driving_force_ladder.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    plot(rows, "f", "developed_da_dN_m_per_cycle", "Developed da/dN (m/cycle)",
         args.out / "developed_da_dN_vs_f.png")
    plot(rows, "deltaK_MPa_sqrt_m", "developed_da_dN_m_per_cycle", "Developed da/dN (m/cycle)",
         args.out / "developed_da_dN_vs_deltaK.png")
    plot(rows, "f", "cycles_to_first_event", "Cycles to first event",
         args.out / "cycles_to_first_event_vs_f.png")
    plot(rows, "f", "cycles_to_target", "Cycles to ~100 µm target",
         args.out / "cycles_to_target_vs_f.png")
    plot(rows, "deltaK_MPa_sqrt_m", "cycles_to_first_event", "Cycles to first event",
         args.out / "cycles_to_first_event_vs_deltaK.png")
    plot(rows, "deltaK_MPa_sqrt_m", "cycles_to_target", "Cycles to ~100 µm target",
         args.out / "cycles_to_target_vs_deltaK.png")
    plot(rows, "deltaK_MPa_sqrt_m", "median_event_spacing_cycles", "Median event spacing (cycles)",
         args.out / "event_spacing_vs_deltaK.png")
    plot(rows, "deltaK_MPa_sqrt_m", "developed_da_dN_m_per_cycle", "Developed da/dN (m/cycle)",
         args.out / "high_rate_zoom.png", ylim=(1e-8, 1e-4))
    regime_map(rows, args.out / "four_class_regime_map.png")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
