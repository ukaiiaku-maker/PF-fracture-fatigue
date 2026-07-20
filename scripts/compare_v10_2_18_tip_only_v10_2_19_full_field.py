#!/usr/bin/env python3
"""Compare matched v10.2.18 tip-only and v10.2.19 full-field screens."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
METRICS_FILE = "dbtt_candidate_short_screen_metrics.csv"
DIAGNOSTICS_FILE = "dbtt_candidate_transition_diagnostics.csv"


def _find_analysis(root: Path, preferred: str) -> Path:
    direct = root / preferred
    if (direct / METRICS_FILE).is_file():
        return direct
    matches = [p.parent for p in root.rglob(METRICS_FILE)]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"could not uniquely locate {METRICS_FILE} beneath {root}; found {matches}"
        )
    return matches[0]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric_index(rows: list[dict[str, str]]) -> dict[tuple[str, float], dict[str, str]]:
    out = {}
    for row in rows:
        key = (row["option_key"], float(row["temperature_K"]))
        if key in out:
            raise ValueError(f"duplicate metric row {key}")
        out[key] = row
    return out


def _diagnostic_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["option_key"]: row for row in rows}


def _save(fig, base: Path, formats: list[str], dpi: int) -> list[str]:
    written = []
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(str(path))
    plt.close(fig)
    return written


def _plot_candidate_panels(rows: list[dict], outdir: Path, formats: list[str], dpi: int) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0), constrained_layout=True)
    for ax, option in zip(axes.flat, OPTIONS):
        sub = sorted((r for r in rows if r["option_key"] == option), key=lambda r: r["temperature_K"])
        T = [r["temperature_K"] for r in sub]
        ax.plot(T, [r["K_initial_tip_only"] for r in sub], marker=MARKERS[option],
                markerfacecolor="none", linestyle="--", linewidth=1.4, label="Tip only")
        ax.plot(T, [r["K_initial_full_field"] for r in sub], marker=MARKERS[option],
                linewidth=1.7, label="Tip + bulk")
        ax.set_title(LABELS[option])
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(r"$K_{initial}$ (MPa$\sqrt{m}$)")
        ax.tick_params(direction="out")
        ax.legend(frameon=False)
    return _save(fig, outdir / "K_initial_tip_only_vs_full_field_by_candidate", formats, dpi)


def _plot_delta(rows: list[dict], outdir: Path, formats: list[str], dpi: int) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for option in OPTIONS:
        sub = sorted((r for r in rows if r["option_key"] == option), key=lambda r: r["temperature_K"])
        ax.plot(
            [r["temperature_K"] for r in sub],
            [r["delta_K_initial_full_minus_tip"] for r in sub],
            marker=MARKERS[option], markerfacecolor="none", linewidth=1.6,
            label=LABELS[option],
        )
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$K_{initial}^{full}-K_{initial}^{tip}$ (MPa$\sqrt{m}$)")
    ax.tick_params(direction="out")
    ax.legend(frameon=False)
    return _save(fig, outdir / "delta_K_initial_full_field_minus_tip_only", formats, dpi)


def _plot_sharpness(rows: list[dict], outdir: Path, formats: list[str], dpi: int) -> list[str]:
    x = np.arange(len(OPTIONS), dtype=float)
    width = 0.36
    tip = [r["tip_positive_rise_localization_fraction"] for r in rows]
    full = [r["full_positive_rise_localization_fraction"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.bar(x - width / 2.0, tip, width=width, label="Tip only")
    ax.bar(x + width / 2.0, full, width=width, label="Tip + bulk")
    ax.set_xticks(x, [LABELS[o] for o in OPTIONS], rotation=18, ha="right")
    ax.set_ylabel("Largest-jump fraction of positive rise")
    ax.tick_params(direction="out")
    ax.legend(frameon=False)
    return _save(fig, outdir / "transition_localization_tip_only_vs_full_field", formats, dpi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tip-only-root", required=True, type=Path)
    parser.add_argument("--full-field-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    tip_root = args.tip_only_root.expanduser().resolve()
    full_root = args.full_field_root.expanduser().resolve()
    tip_analysis = _find_analysis(tip_root, "analysis_v10_2_18_dbtt_candidate_screen")
    full_analysis = _find_analysis(full_root, "analysis_v10_2_19_full_field_dbtt_screen")
    outdir = (args.output_dir or full_root / "analysis_tip_only_vs_full_field").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    tip_metrics = _metric_index(_read_csv(tip_analysis / METRICS_FILE))
    full_metrics = _metric_index(_read_csv(full_analysis / METRICS_FILE))
    if set(tip_metrics) != set(full_metrics):
        missing_tip = sorted(set(full_metrics) - set(tip_metrics))
        missing_full = sorted(set(tip_metrics) - set(full_metrics))
        raise SystemExit(f"unmatched cases: missing_tip={missing_tip}, missing_full={missing_full}")
    if len(tip_metrics) != 40:
        raise SystemExit(f"expected 40 matched cases; found {len(tip_metrics)}")

    metric_rows = []
    for option, temperature in sorted(tip_metrics, key=lambda k: (OPTIONS.index(k[0]), k[1])):
        tip = tip_metrics[(option, temperature)]
        full = full_metrics[(option, temperature)]
        row = {
            "option_key": option,
            "candidate_id": full.get("candidate_id", tip.get("candidate_id", "")),
            "temperature_K": temperature,
            "K_initial_tip_only": float(tip["K_initial_MPa_sqrt_m"]),
            "K_initial_full_field": float(full["K_initial_MPa_sqrt_m"]),
            "K_end_tip_only": float(tip["K_end_MPa_sqrt_m"]),
            "K_end_full_field": float(full["K_end_MPa_sqrt_m"]),
            "Rcurve_slope_tip_only_per_100um": float(tip["Rcurve_slope_MPa_sqrt_m_per_100um"]),
            "Rcurve_slope_full_field_per_100um": float(full["Rcurve_slope_MPa_sqrt_m_per_100um"]),
        }
        row["delta_K_initial_full_minus_tip"] = row["K_initial_full_field"] - row["K_initial_tip_only"]
        row["delta_K_end_full_minus_tip"] = row["K_end_full_field"] - row["K_end_tip_only"]
        row["delta_Rcurve_slope_full_minus_tip_per_100um"] = (
            row["Rcurve_slope_full_field_per_100um"] - row["Rcurve_slope_tip_only_per_100um"]
        )
        metric_rows.append(row)
    _write_csv(outdir / "tip_only_vs_full_field_metrics.csv", metric_rows)

    tip_diag = _diagnostic_index(_read_csv(tip_analysis / DIAGNOSTICS_FILE))
    full_diag = _diagnostic_index(_read_csv(full_analysis / DIAGNOSTICS_FILE))
    diagnostics = []
    for option in OPTIONS:
        t = tip_diag[option]
        f = full_diag[option]
        diagnostics.append({
            "option_key": option,
            "candidate_id": f.get("candidate_id", t.get("candidate_id", "")),
            "tip_linear_fit_R2": float(t["linear_fit_R2"]),
            "full_linear_fit_R2": float(f["linear_fit_R2"]),
            "tip_largest_adjacent_100K_jump": float(t["largest_adjacent_100K_jump"]),
            "full_largest_adjacent_100K_jump": float(f["largest_adjacent_100K_jump"]),
            "tip_positive_rise_localization_fraction": float(t["positive_rise_localization_fraction"]),
            "full_positive_rise_localization_fraction": float(f["positive_rise_localization_fraction"]),
            "tip_midpoint_crossing_temperature_K": float(t["midpoint_crossing_temperature_K"]),
            "full_midpoint_crossing_temperature_K": float(f["midpoint_crossing_temperature_K"]),
            "delta_localization_full_minus_tip": float(f["positive_rise_localization_fraction"]) - float(t["positive_rise_localization_fraction"]),
        })
    _write_csv(outdir / "tip_only_vs_full_field_transition_diagnostics.csv", diagnostics)

    plots = []
    plots += _plot_candidate_panels(metric_rows, outdir, args.formats, args.dpi)
    plots += _plot_delta(metric_rows, outdir, args.formats, args.dpi)
    plots += _plot_sharpness(diagnostics, outdir, args.formats, args.dpi)
    manifest = {
        "schema": "v10.2.18_tip_only_vs_v10.2.19_full_field",
        "tip_only_root": str(tip_root),
        "full_field_root": str(full_root),
        "matched_cases": len(metric_rows),
        "controlled_change": "bulk_plasticity_mode tip_only -> full_field",
        "plots": plots,
    }
    (outdir / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"matched_cases": len(metric_rows), "output_dir": str(outdir), "plots": len(plots)}, sort_keys=True))


if __name__ == "__main__":
    main()
