#!/usr/bin/env python3
"""Analyze the v10.2.20 v9.12 five-candidate peak screen.

The frozen solver reports the J-derived equivalent ``KJ_Pa_sqrtm``.  This
postprocessor reports the corresponding plane-strain energy release value
``J = KJ**2 / Eprime`` and labels it explicitly as reconstructed from KJ.
The current sharp-front geometry law advances only when the cleavage clock
fires; it does not contain a separate stable ductile-tearing criterion.  Thus
J_init for stable tearing and CTOD are reported as unavailable rather than
being inferred from arbitrary thresholds.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts" / "plot_v10_2_17_stage3_temperature_metrics.py"
SPEC = importlib.util.spec_from_file_location("v10217_temperature_metrics_v10220", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not import {BASE_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)

OPTIONS = (
    "v912_peak_0368",
    "v912_peak_0314",
    "v912_peak_0162",
    "v912_late_0118",
    "v912_plateau_0403",
)
LABELS = {
    "v912_peak_0368": "Sharp peak 0368",
    "v912_peak_0314": "Broad onset 0314",
    "v912_peak_0162": "Broad tail 0162",
    "v912_late_0118": "Late response 0118",
    "v912_plateau_0403": "Exhausted shelf 0403",
}
MARKERS = {
    "v912_peak_0368": "o",
    "v912_peak_0314": "s",
    "v912_peak_0162": "^",
    "v912_late_0118": "D",
    "v912_plateau_0403": "v",
}
E_PA = 410.0e9
NU = 0.28
EPRIME_PA = E_PA / (1.0 - NU**2)


def _selection(case_root: Path) -> dict:
    for name in (
        "v10_2_20_v912_parameter_selection.json",
        "v10_2_18_dbtt_parameter_selection.json",
        "v10_2_17_parameter_selection.json",
        "v10_2_15_parameter_selection.json",
    ):
        path = case_root / name
        if path.is_file():
            return json.loads(path.read_text())
    return {}


BASE._selection = _selection
BASE.OPTION_LABELS.update(LABELS)
BASE.OPTION_MARKERS.update(MARKERS)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_steps(path: Path) -> dict[str, np.ndarray]:
    return BASE._load_named_csv(path)


def _cumulative_work(U: np.ndarray, F: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    F = np.asarray(F, dtype=float)
    out = np.zeros_like(U)
    if U.size > 1:
        dU = np.diff(U)
        out[1:] = np.cumsum(0.5 * (F[1:] + F[:-1]) * dU)
    return out


def _first_event_index(steps: dict[str, np.ndarray]) -> int:
    nfire = np.asarray(steps.get("n_fire", []), dtype=float)
    da = np.asarray(steps.get("da_block_m", np.zeros_like(nfire)), dtype=float)
    indices = np.flatnonzero((nfire > 0.5) | (da > 1.0e-15))
    return int(indices[0]) if indices.size else max(len(nfire) - 1, 0)


def _safe_at(values: np.ndarray | None, index: int) -> float:
    if values is None:
        return math.nan
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return math.nan
    i = min(max(int(index), 0), array.size - 1)
    return float(array[i])


def _case_roots(outroot: Path) -> list[Path]:
    roots = []
    for option in OPTIONS:
        option_root = outroot / option
        if not option_root.is_dir():
            continue
        for case in sorted(option_root.iterdir()):
            if case.is_dir() and (case / "summary.json").is_file():
                roots.append(case)
    return roots


def _save(fig, base: Path, formats: list[str], dpi: int) -> list[str]:
    written = []
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(str(path))
    plt.close(fig)
    return written


def _plot_temperature(
    rows: list[dict],
    field: str,
    ylabel: str,
    output: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for option in OPTIONS:
        sub = sorted(
            (row for row in rows if row["option_key"] == option),
            key=lambda row: row["temperature_K"],
        )
        if not sub:
            continue
        ax.plot(
            [row["temperature_K"] for row in sub],
            [row[field] for row in sub],
            marker=MARKERS[option],
            markerfacecolor="none",
            linewidth=1.6,
            label=LABELS[option],
        )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="out")
    ax.legend(frameon=False)
    return _save(fig, output, formats, dpi)


def _peak_diagnostics(rows: list[dict]) -> list[dict]:
    result = []
    for option in OPTIONS:
        sub = sorted(
            (row for row in rows if row["option_key"] == option),
            key=lambda row: row["temperature_K"],
        )
        if not sub:
            continue
        K = np.asarray([row["K_initial_MPa_sqrt_m"] for row in sub], dtype=float)
        J = np.asarray([row["J_c_from_KJ_kJ_per_m2"] for row in sub], dtype=float)
        T = np.asarray([row["temperature_K"] for row in sub], dtype=float)
        peak = int(np.nanargmax(K))
        idx1200 = int(np.argmin(np.abs(T - 1200.0)))
        idx300 = int(np.argmin(np.abs(T - 300.0)))
        selection = json.loads(
            (Path(sub[0]["case_root"]) / "v10_2_20_v912_parameter_selection.json").read_text()
        )
        result.append({
            "option_key": option,
            "label": LABELS[option],
            "candidate_id": sub[0]["candidate_id"],
            "source_exhausted_control": bool(selection.get("source_exhausted_control", False)),
            "peak_temperature_K": float(T[peak]),
            "peak_K_initial_MPa_sqrt_m": float(K[peak]),
            "peak_J_c_from_KJ_kJ_per_m2": float(J[peak]),
            "K_initial_300K_MPa_sqrt_m": float(K[idx300]),
            "K_initial_1200K_MPa_sqrt_m": float(K[idx1200]),
            "drop_peak_to_1200K_MPa_sqrt_m": float(K[peak] - K[idx1200]),
            "rise_300K_to_peak_MPa_sqrt_m": float(K[peak] - K[idx300]),
            "J_c_1200K_from_KJ_kJ_per_m2": float(J[idx1200]),
            "available_site_fraction_at_1200K": float(sub[idx1200]["available_site_fraction_end"]),
            "absorbed_work_end_1200K_J_per_modeled_thickness": float(
                sub[idx1200]["absorbed_work_end_J_per_modeled_thickness"]
            ),
            "intended_peak_temperature_K": selection.get("intended_peak_temperature_K"),
            "intended_peak_delta_K_micro_MPa_sqrt_m": selection.get(
                "intended_peak_delta_K_micro_MPa_sqrt_m"
            ),
            "intended_delta_K_micro_1200K_MPa_sqrt_m": selection.get(
                "intended_delta_K_micro_1200K_MPa_sqrt_m"
            ),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outroot", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--require-cases", type=int, default=0)
    args = parser.parse_args()

    outroot = args.outroot.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else outroot / "analysis_v10_2_20_v912_top5_peak_screen"
    )
    output.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict] = []
    event_rows: list[dict] = []
    roots = _case_roots(outroot)
    if args.require_cases and len(roots) != args.require_cases:
        raise SystemExit(f"expected {args.require_cases} cases; found {len(roots)}")

    for case_root in roots:
        metric, events = BASE.analyze_case(case_root)
        status = json.loads((case_root / "stage3_case_status.json").read_text())
        if status.get("complete") is not True:
            raise SystemExit(f"incomplete case cannot be analyzed: {case_root}")
        selection = _selection(case_root)
        metric_dict = BASE.asdict(metric)
        metric_dict["option_label"] = LABELS.get(metric.option_key, metric.option_key)
        metric_dict["candidate_id"] = str(selection.get("candidate_id", metric.candidate_id))

        steps = _load_steps(Path(metric.steps_file))
        U = np.asarray(steps["Uapp_m"], dtype=float)
        F = np.asarray(steps["Ftop_N"], dtype=float)
        work = _cumulative_work(U, F)
        first = _first_event_index(steps)
        available = steps.get("mpz_available_site_fraction")
        shield = steps.get("mpz_K_shield_Pa_sqrt_m")
        K_initial_pa = metric.K_initial_MPa_sqrt_m * 1.0e6
        K_end_pa = metric.K_end_MPa_sqrt_m * 1.0e6
        metric_dict.update({
            "J_c_from_KJ_J_per_m2": K_initial_pa**2 / EPRIME_PA,
            "J_c_from_KJ_kJ_per_m2": K_initial_pa**2 / EPRIME_PA / 1000.0,
            "J_end_from_KJ_J_per_m2": K_end_pa**2 / EPRIME_PA,
            "J_end_from_KJ_kJ_per_m2": K_end_pa**2 / EPRIME_PA / 1000.0,
            "absorbed_work_at_cleavage_J_per_modeled_thickness": _safe_at(work, first),
            "absorbed_work_end_J_per_modeled_thickness": _safe_at(work, len(work) - 1),
            "available_site_fraction_at_cleavage": _safe_at(available, first),
            "available_site_fraction_end": _safe_at(available, len(U) - 1),
            "signed_K_shield_at_cleavage_MPa_sqrt_m": _safe_at(shield, first) / 1.0e6,
            "signed_K_shield_end_MPa_sqrt_m": _safe_at(shield, len(U) - 1) / 1.0e6,
            "J_reconstruction_Eprime_Pa": EPRIME_PA,
            "J_init_stable_tearing": "not represented",
            "CTOD": "not available in frozen solver output",
            "high_temperature_K_interpretation": "cleavage-equivalent K_J; not presumed valid K_IC",
        })
        case_rows.append(metric_dict)

        for event in events:
            K_pa = float(event["KJ_MPa_sqrt_m"]) * 1.0e6
            event_rows.append({
                **event,
                "candidate_id": metric_dict["candidate_id"],
                "J_from_KJ_J_per_m2": K_pa**2 / EPRIME_PA,
                "J_from_KJ_kJ_per_m2": K_pa**2 / EPRIME_PA / 1000.0,
            })

    case_rows.sort(key=lambda row: (OPTIONS.index(row["option_key"]), row["temperature_K"]))
    event_rows.sort(key=lambda row: (OPTIONS.index(row["option_key"]), row["temperature_K"], row["event_index"]))
    peaks = _peak_diagnostics(case_rows)

    _write_csv(output / "v912_top5_case_metrics_K_J_work.csv", case_rows)
    _write_csv(output / "v912_top5_event_J_R_curves.csv", event_rows)
    _write_csv(output / "v912_top5_peak_diagnostics.csv", peaks)

    plots = []
    plots += _plot_temperature(case_rows, "K_initial_MPa_sqrt_m", r"Cleavage initiation $K_J$ (MPa$\sqrt{m}$)", output / "K_initial_vs_temperature_v912_top5", args.formats, args.dpi)
    plots += _plot_temperature(case_rows, "J_c_from_KJ_kJ_per_m2", r"$J_c=K_J^2/E'$ (kJ m$^{-2}$)", output / "J_c_vs_temperature_v912_top5", args.formats, args.dpi)
    plots += _plot_temperature(case_rows, "K_end_MPa_sqrt_m", r"Endpoint $K_J$ (MPa$\sqrt{m}$)", output / "K_end_vs_temperature_v912_top5", args.formats, args.dpi)
    plots += _plot_temperature(case_rows, "J_end_from_KJ_kJ_per_m2", r"Endpoint $J=K_J^2/E'$ (kJ m$^{-2}$)", output / "J_end_vs_temperature_v912_top5", args.formats, args.dpi)
    plots += _plot_temperature(case_rows, "Rcurve_slope_MPa_sqrt_m_per_100um", r"Early R-curve slope (MPa$\sqrt{m}$ per 100 $\mu$m)", output / "Rcurve_slope_vs_temperature_v912_top5", args.formats, args.dpi)
    plots += _plot_temperature(case_rows, "absorbed_work_end_J_per_modeled_thickness", "Integrated load-displacement work (J per modeled thickness)", output / "work_vs_temperature_v912_top5", args.formats, args.dpi)
    plots += _plot_temperature(case_rows, "available_site_fraction_end", "Remaining tip-source fraction at endpoint", output / "source_fraction_vs_temperature_v912_top5", args.formats, args.dpi)

    manifest = {
        "schema": "v10.2.20_v912_top5_peak_analysis",
        "outroot": str(outroot),
        "cases_analyzed": len(case_rows),
        "Eprime_Pa": EPRIME_PA,
        "J_definition": "KJ_Pa_sqrtm squared divided by Eprime",
        "J_init_stable_tearing_available": False,
        "CTOD_available": False,
        "load_displacement_work_available": True,
        "high_temperature_K_label": "cleavage-equivalent K_J; not automatically K_IC",
        "plots": plots,
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases_analyzed": len(case_rows), "output_dir": str(output), "plots": len(plots)}, sort_keys=True))


if __name__ == "__main__":
    main()
