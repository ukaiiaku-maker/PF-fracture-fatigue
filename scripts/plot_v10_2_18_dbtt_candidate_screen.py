#!/usr/bin/env python3
"""Analyze the v10.2.18 short-distance DBTT candidate screen."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "plot_v10_2_17_stage3_temperature_metrics.py"
SPEC = importlib.util.spec_from_file_location("v10217_metrics", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not import {BASE_SCRIPT}")
BASE = importlib.util.module_from_spec(SPEC)
# Python 3.12 dataclasses resolves annotations through sys.modules while the
# imported module is executing.  Register the module before exec_module.
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)

OPTIONS = (
    "dbtt_primary",
    "dbtt_broad_shielding",
    "dbtt_intrinsic_control",
    "dbtt_moderate_shielding_reference",
)
LABELS = {
    "dbtt_primary": "Current primary",
    "dbtt_broad_shielding": "Broad shielding",
    "dbtt_intrinsic_control": "Intrinsic control",
    "dbtt_moderate_shielding_reference": "Moderate shielding",
}
MARKERS = {
    "dbtt_primary": "o",
    "dbtt_broad_shielding": "s",
    "dbtt_intrinsic_control": "^",
    "dbtt_moderate_shielding_reference": "D",
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _v10218_selection(case_root: Path) -> dict:
    path = case_root / "v10_2_18_dbtt_parameter_selection.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _target_extension_um(case_root: Path) -> float | None:
    path = case_root / "stage3_case_status.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text()).get("target_extension_um")
    return None if value is None else float(value)


def _target_text(target_extension_um: float | None) -> str:
    return "end" if target_extension_um is None else f"{target_extension_um:g} µm"


def _transition_diagnostics(option: str, rows: list) -> dict:
    subset = sorted((r for r in rows if r.option_key == option), key=lambda r: r.temperature_K)
    T = np.asarray([r.temperature_K for r in subset], dtype=float)
    K = np.asarray([r.K_initial_MPa_sqrt_m for r in subset], dtype=float)
    low = float(np.mean(K[T <= 500.0]))
    high = float(np.mean(K[T >= 1000.0]))
    rise = high - low
    increments = np.diff(K)
    positive_total = float(np.sum(np.maximum(increments, 0.0)))
    imax = int(np.argmax(increments)) if increments.size else 0
    max_jump = float(increments[imax]) if increments.size else float("nan")
    localization = max_jump / positive_total if positive_total > 0.0 else float("nan")

    linear_slope, linear_intercept = np.polyfit(T, K, 1)
    pred = linear_intercept + linear_slope * T
    ss_res = float(np.sum((K - pred) ** 2))
    ss_tot = float(np.sum((K - np.mean(K)) ** 2))
    linear_r2 = float("nan") if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot

    midpoint = low + 0.5 * rise
    transition_T = float("nan")
    if rise > 0.0:
        for i in range(len(T) - 1):
            y0, y1 = K[i], K[i + 1]
            if (y0 - midpoint) * (y1 - midpoint) <= 0.0 and y1 != y0:
                transition_T = float(T[i] + (midpoint - y0) * (T[i + 1] - T[i]) / (y1 - y0))
                break

    return {
        "option_key": option,
        "label": LABELS[option],
        "candidate_id": subset[0].candidate_id if subset else "",
        "K_initial_low_shelf_mean_300_500": low,
        "K_initial_high_shelf_mean_1000_1200": high,
        "K_initial_high_minus_low": rise,
        "largest_adjacent_100K_jump": max_jump,
        "largest_jump_interval_start_K": float(T[imax]) if increments.size else float("nan"),
        "largest_jump_interval_end_K": float(T[imax + 1]) if increments.size else float("nan"),
        "positive_rise_localization_fraction": localization,
        "linear_fit_R2": linear_r2,
        "midpoint_crossing_temperature_K": transition_T,
    }


def _plot_metric(rows: list, metric: str, ylabel: str, path: Path, formats: list[str], dpi: int) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for option in OPTIONS:
        subset = sorted((r for r in rows if r.option_key == option), key=lambda r: r.temperature_K)
        ax.plot(
            [r.temperature_K for r in subset],
            [getattr(r, metric) for r in subset],
            marker=MARKERS[option],
            markerfacecolor="none",
            markersize=7,
            linewidth=1.6,
            label=LABELS[option],
        )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="out")
    ax.legend(frameon=False)
    written = []
    for fmt in formats:
        target = path.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
        written.append(str(target))
    plt.close(fig)
    return written


def _plot_three_panel(
    rows: list,
    path: Path,
    formats: list[str],
    dpi: int,
    target_extension_um: float | None,
) -> list[str]:
    target = _target_text(target_extension_um)
    specs = (
        ("K_initial_MPa_sqrt_m", r"$K_{initial}$ (MPa$\sqrt{m}$)"),
        ("K_end_MPa_sqrt_m", rf"$K_{{{target}}}$ (MPa$\sqrt{{m}}$)"),
        ("Rcurve_slope_MPa_sqrt_m_per_100um", r"Early R-curve slope (MPa$\sqrt{m}$ per 100 $\mu$m)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, specs):
        for option in OPTIONS:
            subset = sorted((r for r in rows if r.option_key == option), key=lambda r: r.temperature_K)
            ax.plot(
                [r.temperature_K for r in subset],
                [getattr(r, metric) for r in subset],
                marker=MARKERS[option], markerfacecolor="none",
                markersize=6, linewidth=1.4, label=LABELS[option],
            )
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(ylabel)
        ax.tick_params(direction="out")
    axes[0].legend(frameon=False, fontsize=8)
    written = []
    for fmt in formats:
        output = path.with_suffix(f".{fmt}")
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        written.append(str(output))
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outroot", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    root = args.outroot.expanduser().resolve()
    outdir = (args.output_dir or root / "analysis_v10_2_18_dbtt_candidate_screen").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = []
    events = []
    failures = []
    target_values: set[float] = set()
    for case in sorted(root.glob("*/T*_th*_seed*")):
        try:
            metric, case_events = BASE.analyze_case(case)
            selection = _v10218_selection(case)
            if selection:
                metric.option_key = str(selection.get("option_key", metric.option_key))
                metric.option_label = LABELS.get(metric.option_key, metric.option_key)
                metric.candidate_id = str(selection.get("candidate_id", metric.candidate_id))
                for event in case_events:
                    event["option_key"] = metric.option_key
            target_value = _target_extension_um(case)
            if target_value is not None:
                target_values.add(target_value)
            if metric.status != "complete_target_extension":
                raise ValueError(f"case status is {metric.status}")
            if metric.option_key not in OPTIONS:
                raise ValueError(f"unexpected option {metric.option_key}")
            metrics.append(metric)
            events.extend(case_events)
        except Exception as exc:
            failures.append({"case_root": str(case), "error": f"{type(exc).__name__}: {exc}"})

    if failures:
        (outdir / "analysis_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise SystemExit(f"analysis failed for {len(failures)} case(s)")
    if len(metrics) != 40:
        raise SystemExit(f"expected 40 complete cases; analyzed {len(metrics)}")
    if len(target_values) > 1:
        raise SystemExit(f"mixed target extensions found: {sorted(target_values)}")
    target_extension_um = next(iter(target_values)) if target_values else None

    metrics.sort(key=lambda r: (OPTIONS.index(r.option_key), r.temperature_K))
    metric_rows = [asdict(r) for r in metrics]
    _write_csv(outdir / "dbtt_candidate_short_screen_metrics.csv", metric_rows)
    (outdir / "dbtt_candidate_short_screen_metrics.json").write_text(
        json.dumps(metric_rows, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(outdir / "dbtt_candidate_short_screen_event_Rcurves.csv", events)

    diagnostics = [_transition_diagnostics(option, metrics) for option in OPTIONS]
    _write_csv(outdir / "dbtt_candidate_transition_diagnostics.csv", diagnostics)
    (outdir / "dbtt_candidate_transition_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )

    target = _target_text(target_extension_um)
    plots = []
    plots += _plot_metric(metrics, "K_initial_MPa_sqrt_m", r"$K_{initial}$ (MPa$\sqrt{m}$)", outdir / "K_initial_vs_temperature_all_DBTT_candidates", args.formats, args.dpi)
    plots += _plot_metric(metrics, "K_end_MPa_sqrt_m", rf"$K_{{{target}}}$ (MPa$\sqrt{{m}}$)", outdir / "K_end_vs_temperature_all_DBTT_candidates", args.formats, args.dpi)
    plots += _plot_metric(metrics, "Rcurve_slope_MPa_sqrt_m_per_100um", r"Early R-curve slope (MPa$\sqrt{m}$ per 100 $\mu$m)", outdir / "early_Rcurve_slope_vs_temperature_all_DBTT_candidates", args.formats, args.dpi)
    plots += _plot_three_panel(metrics, outdir / "DBTT_candidate_short_screen_3panel", args.formats, args.dpi, target_extension_um)

    manifest = {
        "schema": "v10.2.18_dbtt_candidate_short_screen_analysis",
        "cases_analyzed": len(metrics),
        "target_extension_um": target_extension_um,
        "common_random_number_design": True,
        "definitions": {
            "K_initial": "summary.json Kc_first_MPa_sqrt_m",
            "K_end": "K_J at final accepted short-screen growth event",
            "early_Rcurve_slope": "OLS event-level K_J versus cumulative primary path extension",
        },
        "plots": plots,
    }
    (outdir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "cases_analyzed": len(metrics),
        "target_extension_um": target_extension_um,
        "output_dir": str(outdir),
        "plots": len(plots),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
