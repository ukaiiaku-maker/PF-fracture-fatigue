#!/usr/bin/env python3
"""Aggregate completed v10.2.30 event-growth runs into da/dN versus DeltaK."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


SCHEMA = "v10.2.30_four_class_fatigue_da_dN_campaign_v2"


def _load(path: Path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def summarize_run(root: Path) -> dict:
    summary = _load(root / "developed_fatigue_growth_summary.json", {})
    control = _load(root / "v10_2_30_fixed_deltaK_control.json", {})
    manifest = _load(root / "high_cycle_run_manifest.json", {})
    provenance = summary.get("provenance", {})
    developed = summary.get("developed_interval", {})
    early = summary.get("stability_early_interval", {})
    late = summary.get("stability_late_interval", {})
    event_count = int(summary.get("event_count", 0))
    exit_code_path = root / "exit_code.txt"
    exit_code = int(exit_code_path.read_text().strip()) if exit_code_path.is_file() else None
    # A wrapper watchdog can expire during terminal plotting/analysis after the
    # solver has already committed the physical target and the developed-growth
    # analyzer has written a complete record.  The physical terminal artifact
    # is authoritative; retain the nonzero wrapper code as diagnostic context.
    if summary.get("target_reached") and event_count > 0:
        status = "completed"
        reason = f"post_target_wrapper_exit_{exit_code}" if exit_code not in (None, 0) else None
    elif exit_code not in (None, 0):
        status = "failed"
        reason = f"exit_code_{exit_code}"
    elif event_count > 0:
        status = "completed" if summary.get("target_reached") else "incomplete"
        reason = None
    else:
        status = "censored" if control.get("censor_status") else "incomplete"
        reason = control.get("censor_status") or summary.get("censor_or_failure_reason")
    return {
        "run_path": str(root.resolve()),
        "parameter_option": provenance.get("parameter_option") or control.get("parameter_option"),
        "temperature_K": provenance.get("temperature_K", summary.get("temperature_K")),
        "deltaK_MPa_sqrt_m": provenance.get("deltaK_MPa_sqrt_m") or control.get(
            "target_deltaK_MPa_sqrt_m"
        ),
        "Kmax_MPa_sqrt_m": provenance.get("Kmax_MPa_sqrt_m"),
        "R": provenance.get("R", control.get("R")),
        "frequency_Hz": provenance.get("frequency_Hz", control.get("frequency_Hz")),
        "hazard_seed": provenance.get("hazard_seed", control.get("cleavage_hazard_seed")),
        "git_head": provenance.get("git_head", manifest.get("git_head")),
        "command": provenance.get("command", manifest.get("generic_launcher")),
        "status": status,
        "censor_or_failure_reason": reason,
        "cycles_reached": summary.get("cycles_consumed", 0.0),
        "event_count": event_count,
        "projected_extension_um": summary.get("final_projected_extension_um", 0.0),
        "developed_event_count": developed.get("event_count", 0),
        "developed_da_dN_m_per_cycle": developed.get("da_dN"),
        "early_da_dN_m_per_cycle": early.get("da_dN"),
        "late_da_dN_m_per_cycle": late.get("da_dN"),
        "late_to_early_rate_ratio": summary.get("late_to_early_rate_ratio"),
        "stable_growth_provisional": summary.get("stable_growth_provisional", False),
    }


def event_intervals(root: Path) -> list[dict]:
    summary = _load(root / "developed_fatigue_growth_summary.json", {})
    geometry = _load(root / "stochastic_avalanche_geometry_events.json", [])
    by_index = {int(row.get("event_index", -1)) + 1: row for row in geometry}
    provenance = summary.get("provenance", {})
    rows = []
    for event in summary.get("event_measurements", []):
        index = int(event["event_index"]); geom = by_index.get(index, {})
        audit = geom.get("direction_audit", {})
        rows.append({
            "run_path": str(root.resolve()), "parameter_option": event.get("parameter_option"),
            "deltaK_MPa_sqrt_m": event.get("deltaK_MPa_sqrt_m"), "hazard_seed": event.get("hazard_seed"),
            "event_index": index, "interval_start_cycles": event.get("cycles_pre"),
            "interval_end_cycles": event.get("cycles_post"), "delta_N": event.get("cycles_between_events"),
            "interval_start_extension_m": event.get("projected_extension_pre_m"),
            "interval_end_extension_m": event.get("projected_extension_post_m"),
            "committed_delta_a_m": event.get("projected_advance_m"), "da_dN_m_per_cycle": event.get("da_dN_m_per_cycle"),
            "threshold_action": event.get("threshold_action"), "physical_hazard_action": event.get("physical_hazard_action"),
            "event_proposal_m": event.get("stochastic_proposed_advance_m"),
            "selected_direction_x": (audit.get("direction") or [None, None])[0],
            "selected_direction_y": (audit.get("direction") or [None, None])[1],
            "endpoint_x_m": geom.get("x1"), "endpoint_y_m": geom.get("y1"),
            "energy_gate_result": event.get("energy_gate_outcome"),
            "geometry_commit_result": event.get("geometry_commit_inserted"),
            "restart_status": "restored_trajectory" if provenance.get("environment", {}).get("V10230_RESTART_CHECKPOINT_DIR") else "continuous",
            "acceleration_mode": event.get("acceleration_modes"),
            "private_trials_counted_as_cycles": event.get("private_trials_counted_as_cycles"),
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _valid(row: dict) -> bool:
    try:
        return float(row["developed_da_dN_m_per_cycle"]) > 0.0
    except (TypeError, ValueError, KeyError):
        return False


def _short_name(option: str) -> str:
    folded = option.lower()
    for name in ("peak", "dbtt", "weakt", "ceramic"):
        if name in folded:
            return "weak-T" if name == "weakt" else name
    return option


def _paris_fit(option: str, rows: list[dict]) -> tuple[dict | None, list[dict]]:
    selected = sorted(
        [row for row in rows if row.get("parameter_option") == option
         and row.get("status") == "completed" and _valid(row)],
        key=lambda row: float(row["deltaK_MPa_sqrt_m"]),
    )
    local = []
    for lower, upper in zip(selected, selected[1:]):
        k0, k1 = (float(lower["deltaK_MPa_sqrt_m"]),
                  float(upper["deltaK_MPa_sqrt_m"]))
        r0, r1 = (float(lower["developed_da_dN_m_per_cycle"]),
                  float(upper["developed_da_dN_m_per_cycle"]))
        local.append({
            "parameter_option": option, "parameter_class": _short_name(option),
            "lower_deltaK_MPa_sqrt_m": k0, "upper_deltaK_MPa_sqrt_m": k1,
            "lower_da_dN_m_per_cycle": r0, "upper_da_dN_m_per_cycle": r1,
            "local_m": math.log(r1 / r0) / math.log(k1 / k0),
        })
    if len(selected) < 3:
        return None, local
    x = [math.log10(float(row["deltaK_MPa_sqrt_m"])) for row in selected]
    y = [math.log10(float(row["developed_da_dN_m_per_cycle"])) for row in selected]
    xbar, ybar = sum(x) / len(x), sum(y) / len(y)
    sxx = sum((value - xbar) ** 2 for value in x)
    m = sum((xi - xbar) * (yi - ybar) for xi, yi in zip(x, y)) / sxx
    intercept = ybar - m * xbar
    residual = sum((yi - (intercept + m * xi)) ** 2 for xi, yi in zip(x, y))
    total = sum((yi - ybar) ** 2 for yi in y)
    fit = {
        "parameter_option": option, "parameter_class": _short_name(option),
        "point_count": len(selected), "m": m, "C_m_per_cycle": 10.0 ** intercept,
        "R2": 1.0 - residual / total if total > 0.0 else 1.0,
        "deltaK_min_MPa_sqrt_m": 10.0 ** min(x),
        "deltaK_max_MPa_sqrt_m": 10.0 ** max(x),
        "deltaK_window_width_MPa_sqrt_m": 10.0 ** max(x) - 10.0 ** min(x),
        "rate_min_m_per_cycle": 10.0 ** min(y),
        "rate_max_m_per_cycle": 10.0 ** max(y),
        "rate_span": 10.0 ** (max(y) - min(y)),
        "response_class": "credible_low_slope" if 2.0 <= m <= 8.0 else "too_steep" if m > 8.0 else "nonstandard_low_or_negative_slope",
    }
    return fit, local


def _plot(rows: list[dict], fits: list[dict], out: Path) -> list[str]:
    outputs = []
    options = sorted({str(row.get("parameter_option")) for row in rows})
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for option in options:
        selected = sorted(
            [row for row in rows if row.get("parameter_option") == option and _valid(row)],
            key=lambda row: float(row["deltaK_MPa_sqrt_m"]),
        )
        if not selected:
            continue
        x = [float(row["deltaK_MPa_sqrt_m"]) for row in selected]
        y = [float(row["developed_da_dN_m_per_cycle"]) for row in selected]
        short = _short_name(option)
        ax.plot(x, y, marker="o", label=short)
        censored = [row for row in rows if row.get("parameter_option") == option
                    and row.get("status") == "censored"]
        if censored:
            floor = min(y) / 3.0
            ax.scatter([float(row["deltaK_MPa_sqrt_m"]) for row in censored],
                       [floor] * len(censored), marker="v", facecolors="none",
                       edgecolors=ax.lines[-1].get_color(), zorder=4)

        single, single_ax = plt.subplots(figsize=(7.5, 5.0))
        single_ax.plot(x, y, marker="o")
        if censored:
            single_ax.scatter([float(row["deltaK_MPa_sqrt_m"]) for row in censored],
                              [min(y) / 3.0] * len(censored), marker="v",
                              facecolors="none", edgecolors="tab:red",
                              label="right-censored (no event)")
        single_ax.set_yscale("log")
        single_ax.set_xlabel("DeltaK (MPa sqrt(m))")
        single_ax.set_ylabel("Developed da/dN (m/cycle)")
        single_ax.set_title(option)
        single_ax.grid(True, which="both", alpha=0.25)
        if censored:
            single_ax.legend()
        name = f"da_dN_vs_deltaK_{option}.png"
        single.tight_layout(); single.savefig(out / name, dpi=180); plt.close(single)
        outputs.append(name)
    ax.set_yscale("log")
    ax.set_xlabel("DeltaK (MPa sqrt(m))")
    ax.set_ylabel("Developed da/dN (m/cycle)")
    ax.set_title("Four-parameterization fatigue crack-growth comparison")
    ax.axhline(1.0e-3, color="black", linestyle="--", linewidth=1.0,
               alpha=0.7, label=r"$10^{-3}$ target")
    ax.grid(True, which="both", alpha=0.25)
    if ax.lines:
        ax.legend(fontsize=7)
    name = "four_class_da_dN_vs_deltaK.png"
    fig.tight_layout(); fig.savefig(out / name, dpi=180); plt.close(fig)
    outputs.append(name)
    return outputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_roots", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = [summarize_run(root.resolve()) for root in args.run_roots]
    rows.sort(key=lambda row: (str(row.get("parameter_option")), float(row.get("deltaK_MPa_sqrt_m") or math.inf), int(row.get("hazard_seed") or 0)))
    _write_csv(out / "four_class_fatigue_cases.csv", rows)
    intervals = [row for root in args.run_roots for row in event_intervals(root.resolve())]
    _write_csv(out / "four_class_event_intervals.csv", intervals)
    per_case = out / "per_case_event_intervals"; per_case.mkdir(exist_ok=True)
    for root in args.run_roots:
        _write_csv(per_case / f"{root.resolve().parent.name}_event_intervals.csv", event_intervals(root.resolve()))
    exceptions = [row for row in rows if row["status"] in {"failed", "censored", "incomplete"}]
    _write_csv(out / "four_class_fatigue_censor_failure_table.csv", exceptions)
    fits, local_slopes = [], []
    for option in sorted({str(row.get("parameter_option")) for row in rows}):
        fit, local = _paris_fit(option, rows)
        if fit is not None:
            fits.append(fit)
        local_slopes.extend(local)
    _write_csv(out / "four_class_paris_fits.csv", fits)
    _write_csv(out / "four_class_local_slopes.csv", local_slopes)
    plots = _plot(rows, fits, out)
    payload = {
        "schema": SCHEMA,
        "case_count": len(rows),
        "completed_count": sum(row["status"] == "completed" for row in rows),
        "censored_count": sum(row["status"] == "censored" for row in rows),
        "failed_count": sum(row["status"] == "failed" for row in rows),
        "cases": rows,
        "event_interval_count": len(intervals),
        "event_interval_csv": "four_class_event_intervals.csv",
        "plots": plots,
        "paris_fits": fits,
        "local_slopes": local_slopes,
        "fit_rate_definition": "developed post-initiation event-to-event projected da/dN",
        "censor_marker": "open downward triangle; plotting ordinate is visual only",
        "empirical_Paris_law_fit": True,
    }
    (out / "four_class_fatigue_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
