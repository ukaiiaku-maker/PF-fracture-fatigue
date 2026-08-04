#!/usr/bin/env python3
"""Analyze complete qualified-plus-extended four-class 0.95 trajectories."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "v913_paper_peak01_0242980_persistent_sites": "peak",
    "v913_paper_dbtt01_0202500_persistent_sites": "dbtt",
    "v913_paper_weakT01_0129902_persistent_sites": "weakT",
    "v913_paper_ceramic01_0077080_persistent_sites": "ceramic",
}


def load(path: Path, default=None):
    try: return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError): return {} if default is None else default


def rate(rows: list[dict]) -> float | None:
    dn = sum(float(row["delta_N"]) for row in rows)
    return sum(float(row["committed_delta_a_m"]) for row in rows) / dn if dn > 0 else None


def analyze_case(root: Path) -> tuple[dict, list[dict]]:
    summary = load(root / "developed_fatigue_growth_summary.json")
    geometry = load(root / "stochastic_avalanche_geometry_events.json", [])
    source = load(root.parent / "extension_source_manifest.json")
    provenance = summary.get("provenance", {})
    by_index = {int(row.get("event_index", -1)) + 1: row for row in geometry}
    option = provenance.get("parameter_option")
    label = LABELS[option]
    rows = []
    for event in summary.get("event_measurements", []):
        index = int(event["event_index"]); geom = by_index.get(index, {})
        direction = (geom.get("direction_audit", {}).get("direction") or [None, None])
        rows.append({
            "class": label, "parameter_option": option,
            "seed": event.get("hazard_seed"), "deltaK_MPa_sqrt_m": event.get("deltaK_MPa_sqrt_m"),
            "Kmax_MPa_sqrt_m": event.get("Kmax_MPa_sqrt_m"), "R": event.get("R"),
            "frequency_Hz": event.get("frequency_Hz"), "event_index": index,
            "cycle_start": event.get("cycles_pre"), "cycle_end": event.get("cycles_post"),
            "delta_N": event.get("cycles_between_events"),
            "crack_extension_start_m": event.get("projected_extension_pre_m"),
            "crack_extension_end_m": event.get("projected_extension_post_m"),
            "committed_delta_a_m": event.get("projected_advance_m"),
            "path_advance_m": event.get("path_advance_m"), "da_dN_m_per_cycle": event.get("da_dN_m_per_cycle"),
            "ds_dN_m_per_cycle": event.get("ds_dN_m_per_cycle"), "tortuosity": event.get("tortuosity"),
            "cumulative_extension_m": event.get("projected_extension_post_m"),
            "threshold_action": event.get("threshold_action"),
            "physical_hazard_action": event.get("physical_hazard_action"),
            "event_proposal_m": event.get("stochastic_proposed_advance_m"),
            "energy_admitted_m": event.get("energy_admissible_advance_m"),
            "selected_direction_x": direction[0], "selected_direction_y": direction[1],
            "endpoint_x_m": geom.get("x1"), "endpoint_y_m": geom.get("y1"),
            "energy_gate_result": event.get("energy_gate_outcome"),
            "geometry_commit_result": event.get("geometry_commit_inserted"),
            "acceleration_mode": event.get("acceleration_modes"),
            "restart_provenance": "qualification" if index <= int(source.get("starting_event_count", 0)) else "developed_extension_restart",
            "private_trials_counted_as_cycles": event.get("private_trials_counted_as_cycles"),
        })
    half = max(1, len(rows) // 2)
    first, second = rows[:half], rows[half:]
    final_m = float(summary.get("final_projected_extension_um", 0.0)) * 1e-6
    late = [row for row in rows if float(row["cumulative_extension_m"]) > final_m - 50e-6]
    first_rate, second_rate, late_rate = rate(first), rate(second), rate(late)
    ratios = {
        "second_to_first_rate_ratio": second_rate / first_rate if first_rate and second_rate else None,
        "late_50um_to_full_rate_ratio": late_rate / rate(rows) if late_rate and rate(rows) else None,
    }
    waits = np.asarray([float(row["delta_N"]) for row in rows], float)
    lengths = np.asarray([float(row["committed_delta_a_m"]) for row in rows], float)
    def log_slope(values):
        if len(values) < 3 or np.any(values <= 0): return None
        return float(np.polyfit(np.arange(len(values)), np.log10(values), 1)[0])
    enough = len(late) >= 5 and len(first) >= 3 and len(second) >= 3
    ratio = ratios["second_to_first_rate_ratio"]
    stationary = bool(enough and ratio is not None and 0.5 <= ratio <= 2.0)
    case = {
        "class": label, "parameter_option": option, "seed": provenance.get("hazard_seed"),
        "deltaK_MPa_sqrt_m": provenance.get("deltaK_MPa_sqrt_m"),
        "status": summary.get("status"), "starting_cycles": source.get("starting_cycles"),
        "final_cycles": summary.get("cycles_consumed"), "starting_event_count": source.get("starting_event_count"),
        "final_event_count": len(rows), "starting_extension_um": source.get("starting_extension_um"),
        "final_extension_um": summary.get("final_projected_extension_um"),
        "extension_added_um": float(summary.get("final_projected_extension_um", 0.0)) - float(source.get("starting_extension_um", 0.0)),
        "full_trajectory_da_dN_m_per_cycle": rate(rows),
        "late_50um_da_dN_m_per_cycle": late_rate, "late_50um_event_count": len(late),
        "first_half_da_dN_m_per_cycle": first_rate, "second_half_da_dN_m_per_cycle": second_rate,
        "event_da_dN_log10_std": float(np.std(np.log10([float(row["da_dN_m_per_cycle"]) for row in rows]))),
        "waiting_cycles_log10_slope_per_event": log_slope(waits),
        "event_length_log10_slope_per_event": log_slope(lengths),
        "stationary_growth_supported": stationary,
        "stationarity_assessment": "supported_by_half_rate_ratio_and_event_count" if stationary else "insufficient_or_nonstationary_at_100um",
        **ratios,
    }
    return case, rows


def write_csv(path: Path, rows: list[dict]):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def plots(cases: list[dict], rows: list[dict], out: Path) -> list[str]:
    outputs = []
    grouped = {case["class"]: [row for row in rows if row["class"] == case["class"]] for case in cases}
    specs = [
        ("da_dN_vs_extension.png", "cumulative_extension_m", "da_dN_m_per_cycle", "Cumulative extension (um)", "Event da/dN (m/cycle)", True),
        ("event_length_vs_event_index.png", "event_index", "committed_delta_a_m", "Event index", "Committed event length (um)", False),
        ("waiting_cycles_vs_event_index.png", "event_index", "delta_N", "Event index", "Waiting cycles", True),
    ]
    for name, xkey, ykey, xlabel, ylabel, logy in specs:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for label, selected in grouped.items():
            x = [float(row[xkey]) * (1e6 if xkey == "cumulative_extension_m" else 1) for row in selected]
            y = [float(row[ykey]) * (1e6 if ykey == "committed_delta_a_m" else 1) for row in selected]
            ax.plot(x, y, marker="o", label=label)
        if logy: ax.set_yscale("log")
        ax.set(xlabel=xlabel, ylabel=ylabel); ax.grid(True, which="both", alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/name, dpi=180); plt.close(fig); outputs.append(name)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for label, selected in grouped.items():
        running = []
        for index in range(1, len(selected)+1): running.append(rate(selected[:index]))
        ax.plot([float(row["cumulative_extension_m"])*1e6 for row in selected], running, marker="o", label=label)
    ax.set_yscale("log"); ax.set(xlabel="Cumulative extension (um)", ylabel="Running da/dN (m/cycle)"); ax.grid(True, which="both", alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/"running_da_dN_vs_extension.png", dpi=180); plt.close(fig); outputs.append("running_da_dN_vs_extension.png")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for label, selected in grouped.items():
        window = [rate(selected[max(0, i-3):i+1]) for i in range(len(selected))]
        ax.plot([float(row["cumulative_extension_m"])*1e6 for row in selected], window, marker="o", label=label)
    ax.set_yscale("log"); ax.set(xlabel="Cumulative extension (um)", ylabel="Four-event window da/dN (m/cycle)"); ax.grid(True, which="both", alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/"windowed_da_dN_vs_extension.png", dpi=180); plt.close(fig); outputs.append("windowed_da_dN_vs_extension.png")
    return outputs


def main(argv=None) -> int:
    p=argparse.ArgumentParser(); p.add_argument("campaign", type=Path); p.add_argument("--out", type=Path); args=p.parse_args(argv)
    root=args.campaign.resolve(); out=(args.out or root/"production_analysis").resolve(); out.mkdir(parents=True, exist_ok=True)
    case_results=[]; rows=[]
    for case_dir in sorted(root.glob("*_f0p95_seed*")):
        case, events=analyze_case(case_dir/"output"); case_results.append(case); rows.extend(events)
        write_csv(out/f'{case["class"]}_event_intervals.csv', events)
    write_csv(out/"complete_event_intervals.csv", rows)
    write_csv(out/"production_cases.csv", case_results)
    failures=[case for case in case_results if case["status"] != "growth_target_reached"]
    write_csv(out/"censor_failure_table.csv", failures)
    plot_files=plots(case_results, rows, out)
    payload={"schema":"v10.2.30_developed_extension_analysis_v1","case_count":len(case_results),"event_interval_count":len(rows),"cases":case_results,"plots":plot_files,"empirical_Paris_law_fit":False}
    (out/"production_summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    (out/"restart_checkpoint_provenance.json").write_text(json.dumps({"schema":"v10.2.30_extension_restart_provenance_v1","cases":[load(case/"extension_source_manifest.json") for case in sorted(root.glob("*_f0p95_seed*"))]},indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
