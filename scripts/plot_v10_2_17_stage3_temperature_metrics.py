#!/usr/bin/env python3
"""Plot v10.2.17 Stage 3 temperature metrics from completed case outputs.

Definitions
-----------
K_initial
    First recorded crack-advance toughness from summary.json.
K_end
    K_J at the final accepted crack-growth event.
R-curve slope
    Ordinary least-squares slope of event-level K_J versus cumulative geometric
    crack-path extension.  A synthetic (Delta a = 0, K = K_initial) point anchors
    the fit at initiation.  Both slope per micrometre and per 100 micrometres are
    written to the output table.

The script is non-destructive.  It only reads completed case directories and
writes plots/tables beneath --output-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OPTION_ORDER = (
    "ceramic_primary",
    "weakT_primary",
    "dbtt_primary",
    "peak_primary",
)
OPTION_LABELS = {
    "ceramic_primary": "Ceramic-like",
    "weakT_primary": "Weak-T",
    "dbtt_primary": "DBTT",
    "peak_primary": "Peak",
}
OPTION_MARKERS = {
    "ceramic_primary": "o",
    "weakT_primary": "s",
    "dbtt_primary": "^",
    "peak_primary": "D",
}


@dataclass
class CaseMetric:
    option_key: str
    option_label: str
    candidate_id: str
    temperature_K: float
    hazard_seed: int | None
    case_root: str
    status: str
    projected_extension_um: float | None
    path_extension_um: float
    n_path_segments: int
    n_event_points: int
    event_mapping_mode: str
    K_initial_MPa_sqrt_m: float
    K_end_MPa_sqrt_m: float
    delta_K_MPa_sqrt_m: float
    Rcurve_slope_MPa_sqrt_m_per_um: float
    Rcurve_slope_MPa_sqrt_m_per_100um: float
    Rcurve_endpoint_slope_MPa_sqrt_m_per_um: float
    Rcurve_fit_R2: float
    Rcurve_fit_points: int
    steps_file: str
    crack_path_file: str


def _as_rows(array: np.ndarray) -> np.ndarray:
    return np.atleast_1d(array)


def _load_named_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=float,
        encoding=None,
        autostrip=True,
    )
    names = data.dtype.names
    if not names:
        raise ValueError(f"no named columns found in {path}")
    return {name: _as_rows(np.asarray(data[name], dtype=float)) for name in names}


def _load_plain_xy(path: Path) -> np.ndarray:
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=float,
        encoding=None,
        autostrip=True,
    )
    names = data.dtype.names
    if not names or len(names) < 2:
        raise ValueError(f"expected x/y columns in {path}")
    xy = np.column_stack([
        _as_rows(np.asarray(data[names[0]], dtype=float)),
        _as_rows(np.asarray(data[names[1]], dtype=float)),
    ])
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if xy.shape[0] < 2:
        raise ValueError(f"fewer than two finite crack-path points in {path}")
    return xy


def _single_summary(case_root: Path) -> dict:
    payload = json.loads((case_root / "summary.json").read_text())
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"expected one summary row in {case_root / 'summary.json'}")
    return dict(payload[0])


def _selection(case_root: Path) -> dict:
    for name in ("v10_2_17_parameter_selection.json", "v10_2_15_parameter_selection.json"):
        path = case_root / name
        if path.is_file():
            return json.loads(path.read_text())
    return {}


def _status(case_root: Path) -> dict:
    path = case_root / "stage3_case_status.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _stack(case_root: Path) -> dict:
    path = case_root / "v10_2_17_final_signed_stochastic_stack.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _temperature_file(case_root: Path, stem: str, temperature_K: float) -> Path:
    expected = case_root / f"{stem}_{int(round(temperature_K)):04d}K.csv"
    if expected.is_file():
        return expected
    expected_unpadded = case_root / f"{stem}_{int(round(temperature_K))}K.csv"
    if expected_unpadded.is_file():
        return expected_unpadded
    candidates = sorted(case_root.glob(f"{stem}_*K.csv"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"could not uniquely locate {stem} CSV in {case_root}")


def _crack_path_file(case_root: Path, temperature_K: float) -> Path:
    expected = case_root / f"crack_path_{int(round(temperature_K))}K.csv"
    if expected.is_file():
        return expected
    candidates = [
        p for p in sorted(case_root.glob("crack_path_*K.csv"))
        if "front" not in p.name and "branch" not in p.name
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"could not uniquely locate primary crack path in {case_root}")


def _event_k_values(
    steps: dict[str, np.ndarray],
    path_xy: np.ndarray,
) -> tuple[np.ndarray, str]:
    K = np.asarray(steps["KJ_Pa_sqrtm"], dtype=float) / 1.0e6
    da = np.asarray(steps.get("da_block_m", np.zeros_like(K)), dtype=float)
    nfire = np.asarray(steps.get("n_fire", np.zeros_like(K)), dtype=float)
    extension = np.asarray(steps.get("crack_extension_m", np.zeros_like(K)), dtype=float)
    finite = np.isfinite(K)

    expanded: list[float] = []
    for kval, da_i, nf_i, ok in zip(K, da, nfire, finite):
        if not ok:
            continue
        count = int(max(round(float(nf_i)), 0))
        if count <= 0 and float(da_i) > 1.0e-15:
            count = 1
        if count > 0:
            expanded.extend([float(kval)] * count)

    nsegments = path_xy.shape[0] - 1
    if len(expanded) == nsegments:
        return np.asarray(expanded, dtype=float), "n_fire_exact"

    # Fallback: map every path endpoint to the first accepted row whose projected
    # extension has reached that endpoint.  This handles multi-event rows or old
    # outputs whose n_fire count was summarized differently.
    x_projected = path_xy[:, 0] - path_xy[0, 0]
    event_values: list[float] = []
    valid_rows = np.where(finite & np.isfinite(extension))[0]
    if valid_rows.size == 0:
        raise ValueError("no finite K/extension rows")
    for target in x_projected[1:]:
        reached = valid_rows[extension[valid_rows] >= float(target) - 1.0e-12]
        idx = int(reached[0]) if reached.size else int(valid_rows[-1])
        event_values.append(float(K[idx]))
    return np.asarray(event_values, dtype=float), "projected_extension_lookup"


def _collapse_duplicate_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    out_x: list[float] = []
    out_y: list[float] = []
    for xv, yv in zip(x, y):
        if out_x and abs(float(xv) - out_x[-1]) <= 1.0e-10:
            out_y[-1] = float(yv)
        else:
            out_x.append(float(xv))
            out_y.append(float(yv))
    return np.asarray(out_x), np.asarray(out_y)


def _fit_line(x_um: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    finite = np.isfinite(x_um) & np.isfinite(y)
    x = x_um[finite]
    yy = y[finite]
    x, yy = _collapse_duplicate_x(x, yy)
    if x.size < 2 or float(np.ptp(x)) <= 0.0:
        return math.nan, math.nan, int(x.size)
    slope, intercept = np.polyfit(x, yy, 1)
    pred = intercept + slope * x
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    r2 = 1.0 if ss_tot <= 1.0e-30 and ss_res <= 1.0e-30 else (
        math.nan if ss_tot <= 1.0e-30 else 1.0 - ss_res / ss_tot
    )
    return float(slope), float(r2), int(x.size)


def analyze_case(case_root: Path) -> tuple[CaseMetric, list[dict[str, float | str | int]]]:
    summary = _single_summary(case_root)
    selection = _selection(case_root)
    status = _status(case_root)
    stack = _stack(case_root)
    temperature = float(summary["T"])
    steps_path = _temperature_file(case_root, "steps", temperature)
    path_file = _crack_path_file(case_root, temperature)
    steps = _load_named_csv(steps_path)
    path_xy = _load_plain_xy(path_file)

    segment_lengths_um = np.linalg.norm(np.diff(path_xy, axis=0), axis=1) * 1.0e6
    path_extension_um = np.cumsum(segment_lengths_um)
    event_k, mapping_mode = _event_k_values(steps, path_xy)
    if event_k.size != path_extension_um.size:
        raise ValueError(
            f"event/path mismatch in {case_root}: K={event_k.size}, path={path_extension_um.size}"
        )

    K_initial = summary.get("Kc_first_MPa_sqrt_m")
    if K_initial is None or not np.isfinite(float(K_initial)):
        if event_k.size == 0:
            raise ValueError(f"no initial or event K value in {case_root}")
        K_initial = float(event_k[0])
    K_initial = float(K_initial)

    fit_x = np.concatenate(([0.0], path_extension_um))
    fit_y = np.concatenate(([K_initial], event_k))
    slope, r2, nfit = _fit_line(fit_x, fit_y)
    K_end = float(event_k[-1]) if event_k.size else K_initial
    total_path = float(path_extension_um[-1]) if path_extension_um.size else 0.0
    endpoint_slope = (
        (K_end - K_initial) / total_path if total_path > 0.0 else math.nan
    )

    option = str(selection.get("option_key", case_root.parent.name))
    candidate = str(selection.get("candidate_id", ""))
    projected = status.get("projected_extension_um")
    seed = stack.get("cleavage_hazard_seed")
    metric = CaseMetric(
        option_key=option,
        option_label=OPTION_LABELS.get(option, option),
        candidate_id=candidate,
        temperature_K=temperature,
        hazard_seed=None if seed is None else int(seed),
        case_root=str(case_root),
        status=str(status.get("status", "unknown")),
        projected_extension_um=None if projected is None else float(projected),
        path_extension_um=total_path,
        n_path_segments=int(path_xy.shape[0] - 1),
        n_event_points=int(event_k.size),
        event_mapping_mode=mapping_mode,
        K_initial_MPa_sqrt_m=K_initial,
        K_end_MPa_sqrt_m=K_end,
        delta_K_MPa_sqrt_m=K_end - K_initial,
        Rcurve_slope_MPa_sqrt_m_per_um=slope,
        Rcurve_slope_MPa_sqrt_m_per_100um=100.0 * slope,
        Rcurve_endpoint_slope_MPa_sqrt_m_per_um=endpoint_slope,
        Rcurve_fit_R2=r2,
        Rcurve_fit_points=nfit,
        steps_file=str(steps_path),
        crack_path_file=str(path_file),
    )
    events = [
        {
            "option_key": option,
            "temperature_K": temperature,
            "event_index": int(i),
            "path_extension_um": float(x),
            "KJ_MPa_sqrt_m": float(k),
            "case_root": str(case_root),
        }
        for i, (x, k) in enumerate(zip(path_extension_um, event_k), start=1)
    ]
    return metric, events


def _save_figure(fig: plt.Figure, base: Path, formats: Iterable[str], dpi: int) -> list[str]:
    written: list[str] = []
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(str(path))
    plt.close(fig)
    return written


def _style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="out")


def _plot_metric_individual(
    rows: list[CaseMetric],
    metric: str,
    ylabel: str,
    stem: str,
    outdir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    written: list[str] = []
    for option in OPTION_ORDER:
        subset = sorted((r for r in rows if r.option_key == option), key=lambda r: r.temperature_K)
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        marker = OPTION_MARKERS[option]
        ax.plot(
            [r.temperature_K for r in subset],
            [getattr(r, metric) for r in subset],
            marker=marker,
            markerfacecolor="none",
            linewidth=1.6,
            markersize=7,
            label=OPTION_LABELS[option],
        )
        _style_axis(ax, ylabel)
        ax.legend(frameon=False)
        written.extend(_save_figure(fig, outdir / f"{stem}_{option}", formats, dpi))
    return written


def _plot_metric_combined(
    rows: list[CaseMetric],
    metric: str,
    ylabel: str,
    stem: str,
    outdir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for option in OPTION_ORDER:
        subset = sorted((r for r in rows if r.option_key == option), key=lambda r: r.temperature_K)
        if not subset:
            continue
        ax.plot(
            [r.temperature_K for r in subset],
            [getattr(r, metric) for r in subset],
            marker=OPTION_MARKERS[option],
            markerfacecolor="none",
            linewidth=1.6,
            markersize=7,
            label=OPTION_LABELS[option],
        )
    _style_axis(ax, ylabel)
    ax.legend(frameon=False, ncol=2)
    return _save_figure(fig, outdir / f"{stem}_all_options", formats, dpi)


def _plot_three_panel(
    rows: list[CaseMetric],
    outdir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    specs = (
        ("K_initial_MPa_sqrt_m", r"$K_{initial}$ (MPa$\sqrt{m}$)"),
        ("K_end_MPa_sqrt_m", r"$K_{end}$ (MPa$\sqrt{m}$)"),
        ("Rcurve_slope_MPa_sqrt_m_per_100um", r"R-curve slope (MPa$\sqrt{m}$ per 100 $\mu$m)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 4.8), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, specs):
        for option in OPTION_ORDER:
            subset = sorted((r for r in rows if r.option_key == option), key=lambda r: r.temperature_K)
            if not subset:
                continue
            ax.plot(
                [r.temperature_K for r in subset],
                [getattr(r, metric) for r in subset],
                marker=OPTION_MARKERS[option],
                markerfacecolor="none",
                linewidth=1.4,
                markersize=6,
                label=OPTION_LABELS[option],
            )
        _style_axis(ax, ylabel)
    axes[0].legend(frameon=False, ncol=2)
    return _save_figure(fig, outdir / "temperature_metrics_all_options_3panel", formats, dpi)


def _plot_r_curves_by_option(
    rows: list[CaseMetric],
    events: list[dict[str, float | str | int]],
    outdir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    written: list[str] = []
    for option in OPTION_ORDER:
        subset = sorted((r for r in rows if r.option_key == option), key=lambda r: r.temperature_K)
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        for row in subset:
            ev = sorted(
                (e for e in events if e["case_root"] == row.case_root),
                key=lambda e: int(e["event_index"]),
            )
            x = [0.0] + [float(e["path_extension_um"]) for e in ev]
            y = [row.K_initial_MPa_sqrt_m] + [float(e["KJ_MPa_sqrt_m"]) for e in ev]
            ax.plot(x, y, marker="o", markerfacecolor="none", markersize=3.5,
                    linewidth=1.0, label=f"{row.temperature_K:g} K")
        ax.set_xlabel(r"Cumulative crack-path extension, $\Delta a$ ($\mu$m)")
        ax.set_ylabel(r"$K_J$ (MPa$\sqrt{m}$)")
        ax.tick_params(direction="out")
        ax.legend(frameon=False, ncol=2, fontsize=8)
        written.extend(_save_figure(fig, outdir / f"Rcurves_{option}_all_temperatures", formats, dpi))
    return written


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outroot", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-r-curves", action="store_true")
    parser.add_argument("--require-complete", action="store_true", default=True)
    args = parser.parse_args()

    root = args.outroot.expanduser().resolve()
    outdir = (args.output_dir or (root / "analysis_v10_2_17_temperature_metrics")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    case_roots = sorted(root.glob("*/T*_th*_seed*"))
    if not case_roots:
        raise SystemExit(f"no Stage 3 case directories found beneath {root}")

    metrics: list[CaseMetric] = []
    events: list[dict[str, float | str | int]] = []
    failures: list[dict[str, str]] = []
    for case_root in case_roots:
        try:
            metric, case_events = analyze_case(case_root)
            if args.require_complete and metric.status != "complete_target_extension":
                raise ValueError(f"case status is {metric.status!r}, not complete_target_extension")
            metrics.append(metric)
            events.extend(case_events)
        except Exception as exc:  # analysis must report every unusable case
            failures.append({"case_root": str(case_root), "error": f"{type(exc).__name__}: {exc}"})

    if failures:
        (outdir / "analysis_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise SystemExit(
            f"analysis failed for {len(failures)} case(s); see {outdir / 'analysis_failures.json'}"
        )
    if len(metrics) != 40:
        raise SystemExit(f"expected 40 complete cases; analyzed {len(metrics)}")

    metrics.sort(key=lambda r: (OPTION_ORDER.index(r.option_key), r.temperature_K))
    metric_dicts = [asdict(row) for row in metrics]
    _write_csv(outdir / "stage3_temperature_metrics.csv", metric_dicts)
    (outdir / "stage3_temperature_metrics.json").write_text(
        json.dumps(metric_dicts, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(outdir / "stage3_event_level_Rcurves.csv", events)

    plot_files: list[str] = []
    specs = (
        (
            "K_initial_MPa_sqrt_m",
            r"$K_{initial}$ (MPa$\sqrt{m}$)",
            "K_initial_vs_temperature",
        ),
        (
            "K_end_MPa_sqrt_m",
            r"$K_{end}$ (MPa$\sqrt{m}$)",
            "K_end_vs_temperature",
        ),
        (
            "Rcurve_slope_MPa_sqrt_m_per_100um",
            r"R-curve slope (MPa$\sqrt{m}$ per 100 $\mu$m)",
            "Rcurve_slope_vs_temperature",
        ),
    )
    for metric, ylabel, stem in specs:
        plot_files.extend(_plot_metric_individual(metrics, metric, ylabel, stem, outdir, args.formats, args.dpi))
        plot_files.extend(_plot_metric_combined(metrics, metric, ylabel, stem, outdir, args.formats, args.dpi))
    plot_files.extend(_plot_three_panel(metrics, outdir, args.formats, args.dpi))
    if not args.no_r_curves:
        plot_files.extend(_plot_r_curves_by_option(metrics, events, outdir, args.formats, args.dpi))

    manifest = {
        "schema": "v10.2.17_stage3_temperature_metrics_analysis",
        "outroot": str(root),
        "output_dir": str(outdir),
        "cases_analyzed": len(metrics),
        "options": list(OPTION_ORDER),
        "definitions": {
            "K_initial": "summary.json Kc_first_MPa_sqrt_m",
            "K_end": "K_J at final accepted crack-growth event",
            "crack_length_axis": "cumulative Euclidean length of primary crack_path_<T>K.csv",
            "Rcurve_slope": "OLS fit of event-level K_J versus cumulative path extension, anchored at (0,K_initial)",
            "slope_plot_units": "MPa sqrt(m) per 100 um",
        },
        "tables": [
            str(outdir / "stage3_temperature_metrics.csv"),
            str(outdir / "stage3_temperature_metrics.json"),
            str(outdir / "stage3_event_level_Rcurves.csv"),
        ],
        "plots": plot_files,
    }
    (outdir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "cases_analyzed": len(metrics),
        "plots_written": len(plot_files),
        "output_dir": str(outdir),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
