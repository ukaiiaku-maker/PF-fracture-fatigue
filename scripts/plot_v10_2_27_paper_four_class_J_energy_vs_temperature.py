#!/usr/bin/env python3
"""Postprocess direct configurational J and energy absorption versus temperature.

This script keeps the physically distinct quantities separate:

* ``J_direct`` is the signed-positive configurational FEM J used by the crack
  driver.  It is read directly from the accepted root-front J diagnostics.
* ``J_emit`` is cumulative crack-tip emission work divided by projected new
  crack area (unit thickness).
* ``J_bulk`` is cumulative bulk plastic work divided by projected new crack
  area.  It is exactly zero for audited ``tip_only`` runs and is read from the
  accepted-step energy ledger for future validated ``full_field`` runs.
* ``J_irreversible`` is ``(W_ext-U_elastic)/Delta a`` when the stored elastic
  energy ledger is available.
* ``J_fracture_residual`` is
  ``(W_ext-U_elastic-W_bulk-W_emit)/Delta a`` and therefore contains fracture
  surface work plus numerical energy-balance error.

The postprocessor supports legacy completed v10.2.27 cases as well as future
cases written with ``energy_ledger_output_v10227``.  For legacy tip-only cases,
external work is reconstructed from the accepted load-displacement history,
``J_direct`` is read from ``fronts_*K.csv``, and ``J_emit`` is read from the
legacy cumulative emission-work column.  Quantities that cannot be recovered
without the new ledger are left unavailable rather than guessed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CASE_RE = re.compile(
    r"T(?P<T>\d+(?:\.\d+)?)K_th(?P<theta>[-+0-9.]+)_seed(?P<seed>\d+)$"
)
DEFAULT_OPTION_ORDER = (
    "v913_paper_peak01_0242980_persistent_sites",
    "v913_paper_dbtt01_0202500_persistent_sites",
    "v913_paper_weakT01_0257068_persistent_sites",
    "v913_paper_ceramic01_0189364_persistent_sites",
)
SHORT_LABELS = {
    DEFAULT_OPTION_ORDER[0]: "Peak 0242980",
    DEFAULT_OPTION_ORDER[1]: "DBTT 0202500",
    DEFAULT_OPTION_ORDER[2]: "Weak-T/FCC-like 0257068",
    DEFAULT_OPTION_ORDER[3]: "Ceramic-like 0189364",
}
DIRECT_METRICS = (
    ("J_direct_initial_kJ_m2", "Initial"),
    ("J_direct_intermediate_kJ_m2", "Intermediate (500 µm)"),
    ("J_direct_end_kJ_m2", "End (1000 µm)"),
    ("J_direct_average_kJ_m2", "Extension-averaged"),
)
ENERGY_METRICS = (
    ("J_emit_target_kJ_m2", "Tip-emission work / crack area"),
    ("J_bulk_plastic_target_kJ_m2", "Bulk plastic work / crack area"),
    ("J_irreversible_target_kJ_m2", "Irreversible work / crack area"),
    ("J_fracture_residual_target_kJ_m2", "Fracture residual / crack area"),
    ("J_external_work_target_kJ_m2", "External work / crack area"),
)


def _read_structured(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return np.atleast_1d(data)


def _load_steps(case: Path) -> tuple[np.ndarray, Path]:
    files = sorted(case.glob("steps_*K.csv"))
    if len(files) != 1:
        raise RuntimeError(f"expected one steps CSV in {case}; found {files}")
    data = _read_structured(files[0])
    required = {
        "step",
        "Uapp_m",
        "Ftop_N",
        "KJ_Pa_sqrtm",
        "crack_extension_m",
        "da_block_m",
        "W_emit_J_per_m",
        "n_fire",
    }
    missing = required - set(data.dtype.names or ())
    if missing:
        raise RuntimeError(f"{files[0]} missing columns {sorted(missing)}")
    return data, files[0]


def _case_bulk_mode(case: Path) -> str:
    candidates = (
        case / "v10_1_driver_modes.json",
        case / "v10_0_1_driver_modes.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        value = payload.get("bulk_plasticity_mode")
        if value:
            return str(value).strip().lower()
    return "unknown"


def reconstruct_external_work(data: np.ndarray) -> np.ndarray:
    """Cumulative accepted external work per unit thickness, J/m."""
    displacement = np.asarray(data["Uapp_m"], dtype=float)
    force = np.asarray(data["Ftop_N"], dtype=float)
    previous_u = np.concatenate(([0.0], displacement[:-1]))
    previous_f = np.concatenate(([0.0], force[:-1]))
    increments = 0.5 * (force + previous_f) * (displacement - previous_u)
    return np.cumsum(increments)


def _front_direct_j(case: Path, steps: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    names = set(steps.dtype.names or ())
    if "J_effective_direct_J_per_m2" in names:
        effective = np.asarray(steps["J_effective_direct_J_per_m2"], dtype=float)
        signed = (
            np.asarray(steps["J_signed_direct_J_per_m2"], dtype=float)
            if "J_signed_direct_J_per_m2" in names
            else effective.copy()
        )
        return effective, signed, "accepted_step_energy_ledger"

    files = sorted(case.glob("fronts_*K.csv"))
    if len(files) != 1:
        raise RuntimeError(
            f"legacy direct-J recovery requires one fronts CSV in {case}; found {files}"
        )
    fronts = _read_structured(files[0])
    front_names = set(fronts.dtype.names or ())
    required = {"step", "front_id", "J_effective_trial", "J_signed_trial"}
    missing = required - front_names
    if missing:
        raise RuntimeError(f"{files[0]} missing direct-J columns {sorted(missing)}")

    root = fronts[np.asarray(fronts["front_id"], dtype=float) == 0.0]
    mapping = {
        int(round(float(row["step"]))): (
            float(row["J_effective_trial"]),
            float(row["J_signed_trial"]),
        )
        for row in root
    }
    effective = np.full(len(steps), np.nan, dtype=float)
    signed = np.full(len(steps), np.nan, dtype=float)
    for index, step in enumerate(np.asarray(steps["step"], dtype=float)):
        values = mapping.get(int(round(float(step))))
        if values is not None:
            effective[index], signed[index] = values
    if not np.all(np.isfinite(effective)):
        missing_steps = np.asarray(steps["step"], dtype=float)[~np.isfinite(effective)]
        raise RuntimeError(
            f"missing root-front direct J for accepted steps {missing_steps[:10].tolist()} in {case}"
        )
    return effective, signed, "legacy_root_front_direct_J"


def _energy_ledgers(
    case: Path,
    data: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    names = set(data.dtype.names or ())
    bulk_mode = _case_bulk_mode(case)

    if "W_ext_cumulative_J_per_m" in names:
        w_ext = np.asarray(data["W_ext_cumulative_J_per_m"], dtype=float)
        external_source = "accepted_step_energy_ledger"
    else:
        w_ext = reconstruct_external_work(data)
        external_source = "reconstructed_accepted_load_displacement"

    if "W_tip_emit_cumulative_J_per_m" in names:
        w_emit = np.asarray(data["W_tip_emit_cumulative_J_per_m"], dtype=float)
        emit_source = "accepted_step_total_tip_emission_ledger"
    else:
        w_emit = np.asarray(data["W_emit_J_per_m"], dtype=float)
        emit_source = "legacy_primary_tip_emission_ledger"

    if "W_bulk_plastic_cumulative_J_per_m" in names:
        w_bulk = np.asarray(data["W_bulk_plastic_cumulative_J_per_m"], dtype=float)
        bulk_available = True
        bulk_source = "accepted_step_bulk_plastic_ledger"
    elif bulk_mode == "tip_only":
        w_bulk = np.zeros(len(data), dtype=float)
        bulk_available = True
        bulk_source = "audited_tip_only_exact_zero"
    else:
        w_bulk = np.full(len(data), np.nan, dtype=float)
        bulk_available = False
        bulk_source = "unavailable_legacy_non_tip_only_or_unknown"

    if "U_elastic_J_per_m" in names:
        u_elastic = np.asarray(data["U_elastic_J_per_m"], dtype=float)
        elastic_available = True
        elastic_source = "accepted_step_stored_elastic_energy"
    else:
        u_elastic = np.full(len(data), np.nan, dtype=float)
        elastic_available = False
        elastic_source = "unavailable_in_legacy_steps_csv"

    if "W_fracture_residual_cumulative_J_per_m" in names:
        residual = np.asarray(
            data["W_fracture_residual_cumulative_J_per_m"], dtype=float
        )
        residual_available = True
    elif bulk_available and elastic_available:
        residual = w_ext - u_elastic - w_bulk - w_emit
        residual_available = True
    else:
        residual = np.full(len(data), np.nan, dtype=float)
        residual_available = False

    irreversible = (
        w_ext - u_elastic
        if elastic_available
        else np.full(len(data), np.nan, dtype=float)
    )
    return {
        "W_ext": w_ext,
        "U_elastic": u_elastic,
        "W_bulk": w_bulk,
        "W_emit": w_emit,
        "W_irreversible": irreversible,
        "W_fracture_residual": residual,
    }, {
        "bulk_plasticity_mode": bulk_mode,
        "external_work_source": external_source,
        "tip_emission_work_source": emit_source,
        "bulk_plastic_work_source": bulk_source,
        "stored_elastic_energy_source": elastic_source,
        "bulk_plastic_work_available": bulk_available,
        "stored_elastic_energy_available": elastic_available,
        "fracture_residual_available": residual_available,
    }


def event_curve(
    data: np.ndarray,
    direct_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fired = np.asarray(data["n_fire"], dtype=float) > 0.0
    if not np.any(fired):
        empty = np.array([], dtype=float)
        return empty, empty, empty
    post = 1.0e6 * np.asarray(data["crack_extension_m"], dtype=float)[fired]
    increment = 1.0e6 * np.asarray(data["da_block_m"], dtype=float)[fired]
    pre = post - increment
    values = np.asarray(direct_j, dtype=float)[fired] / 1.0e3
    valid = (
        np.isfinite(pre)
        & np.isfinite(post)
        & np.isfinite(values)
        & (post >= pre)
        & (values >= 0.0)
    )
    pre = np.maximum(pre[valid], 0.0)
    post = np.maximum(post[valid], pre)
    values = values[valid]
    order = np.argsort(pre, kind="stable")
    return pre[order], post[order], values[order]


def value_at_extension(
    pre: np.ndarray,
    post: np.ndarray,
    values: np.ndarray,
    target_um: float,
    achieved_um: float,
) -> float:
    if pre.size == 0 or not np.isfinite(achieved_um):
        return float("nan")
    target = float(target_um)
    tol = 1.0e-9 * max(abs(achieved_um), abs(target), 1.0)
    if target < -tol or target > achieved_um + tol:
        return float("nan")
    if target <= pre[0] + tol:
        return float(values[0])
    for index in range(pre.size):
        if pre[index] - tol <= target <= post[index] + tol:
            return float(values[index])
        if index + 1 < pre.size and post[index] < target < pre[index + 1]:
            return float(
                np.interp(
                    target,
                    [post[index], pre[index + 1]],
                    [values[index], values[index + 1]],
                )
            )
    if target <= achieved_um + tol:
        return float(values[-1])
    return float("nan")


def extension_weighted_average(
    pre: np.ndarray,
    post: np.ndarray,
    values: np.ndarray,
    target_um: float,
    achieved_um: float,
) -> float:
    target = float(target_um)
    if target <= 0.0 or pre.size == 0 or achieved_um < target:
        return float("nan")
    knots = [0.0, target]
    for left, right in zip(pre, post):
        if 0.0 < left < target:
            knots.append(float(left))
        if 0.0 < right < target:
            knots.append(float(right))
    x = np.asarray(sorted(set(knots)), dtype=float)
    area = 0.0
    for left, right in zip(x[:-1], x[1:]):
        midpoint = 0.5 * (left + right)
        value = value_at_extension(pre, post, values, midpoint, achieved_um)
        if not math.isfinite(value):
            return float("nan")
        area += value * (right - left)
    return float(area / target)


def cumulative_at_target(
    extension_um: np.ndarray,
    cumulative: np.ndarray,
    target_um: float,
) -> float:
    extension = np.asarray(extension_um, dtype=float)
    values = np.asarray(cumulative, dtype=float)
    valid = np.isfinite(extension) & np.isfinite(values)
    extension = extension[valid]
    values = values[valid]
    if extension.size == 0:
        return float("nan")
    tolerance = 1.0e-9 * max(abs(float(target_um)), 1.0)
    indices = np.flatnonzero(extension >= float(target_um) - tolerance)
    if indices.size == 0:
        return float("nan")
    return float(values[int(indices[0])])


def work_per_crack_area_kj_m2(
    extension_um: np.ndarray,
    cumulative_work_j_per_m: np.ndarray,
    target_um: float,
) -> float:
    work = cumulative_at_target(extension_um, cumulative_work_j_per_m, target_um)
    target_m = float(target_um) * 1.0e-6
    if not math.isfinite(work) or target_m <= 0.0:
        return float("nan")
    return float(work / target_m / 1.0e3)


def _manifest_or_infer(
    outroot: Path,
    cases: list[tuple[str, Path, re.Match[str]]],
    target_um: float | None,
) -> dict[str, Any]:
    path = outroot / "v10_2_27_campaign_manifest.json"
    if path.is_file():
        return json.loads(path.read_text())
    options = [option for option in DEFAULT_OPTION_ORDER if any(c[0] == option for c in cases)]
    options.extend(sorted({c[0] for c in cases} - set(options)))
    temperatures = sorted({float(c[2].group("T")) for c in cases})
    return {
        "schema": "inferred_v10.2.27_campaign_manifest",
        "options": options,
        "temperatures_K": temperatures,
        "planned_case_count": len(cases),
        "target_crack_extension_um": 1000.0 if target_um is None else float(target_um),
        "manifest_inferred": True,
    }


def _discover_cases(outroot: Path) -> list[tuple[str, Path, re.Match[str]]]:
    cases: list[tuple[str, Path, re.Match[str]]] = []
    for option_dir in sorted(path for path in outroot.iterdir() if path.is_dir()):
        option = option_dir.name
        for case in sorted(path for path in option_dir.iterdir() if path.is_dir()):
            match = CASE_RE.match(case.name)
            if match and (case / "COMPLETE").is_file():
                cases.append((option, case, match))
    return cases


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_finite_series(ax, x, y, label, marker) -> bool:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    if not np.any(mask):
        return False
    ax.plot(
        xv[mask],
        yv[mask],
        marker=marker,
        linewidth=1.4,
        markersize=5.0,
        label=label,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outroot", required=True)
    parser.add_argument("--plot-dir")
    parser.add_argument("--target-extension-um", type=float)
    args = parser.parse_args()

    outroot = Path(args.outroot).expanduser().resolve()
    if not outroot.is_dir():
        raise FileNotFoundError(outroot)
    discovered = _discover_cases(outroot)
    if not discovered:
        raise SystemExit(f"no complete v10.2.27 cases found below {outroot}")
    manifest = _manifest_or_infer(outroot, discovered, args.target_extension_um)
    target_um = float(
        args.target_extension_um
        if args.target_extension_um is not None
        else manifest["target_crack_extension_um"]
    )
    intermediate_um = 0.5 * target_um
    plot_dir = (
        Path(args.plot_dir).expanduser().resolve()
        if args.plot_dir
        else outroot / "plots" / "J_energy_vs_temperature"
    )

    records: list[dict[str, Any]] = []
    for option, case, match in discovered:
        data, steps_path = _load_steps(case)
        direct_effective, direct_signed, direct_source = _front_direct_j(case, data)
        pre, post, direct_curve = event_curve(data, direct_effective)
        achieved_values = 1.0e6 * np.asarray(data["crack_extension_m"], dtype=float)
        finite_achieved = achieved_values[np.isfinite(achieved_values)]
        achieved_um = float(np.max(finite_achieved)) if finite_achieved.size else float("nan")
        ledgers, energy_audit = _energy_ledgers(case, data)

        direct_average = extension_weighted_average(
            pre, post, direct_curve, target_um, achieved_um
        )
        record: dict[str, Any] = {
            "option_key": option,
            "label": SHORT_LABELS.get(option, option),
            "temperature_K": float(match.group("T")),
            "theta_deg": float(match.group("theta")),
            "seed": int(match.group("seed")),
            "target_extension_um": target_um,
            "intermediate_extension_um": intermediate_um,
            "achieved_extension_um": achieved_um,
            "J_direct_initial_kJ_m2": float(direct_curve[0]) if direct_curve.size else float("nan"),
            "J_direct_intermediate_kJ_m2": value_at_extension(
                pre, post, direct_curve, intermediate_um, achieved_um
            ),
            "J_direct_end_kJ_m2": value_at_extension(
                pre, post, direct_curve, target_um, achieved_um
            ),
            "J_direct_average_kJ_m2": direct_average,
            "J_emit_target_kJ_m2": work_per_crack_area_kj_m2(
                achieved_values, ledgers["W_emit"], target_um
            ),
            "J_bulk_plastic_target_kJ_m2": work_per_crack_area_kj_m2(
                achieved_values, ledgers["W_bulk"], target_um
            ),
            "J_external_work_target_kJ_m2": work_per_crack_area_kj_m2(
                achieved_values, ledgers["W_ext"], target_um
            ),
            "J_irreversible_target_kJ_m2": work_per_crack_area_kj_m2(
                achieved_values, ledgers["W_irreversible"], target_um
            ),
            "J_fracture_residual_target_kJ_m2": work_per_crack_area_kj_m2(
                achieved_values, ledgers["W_fracture_residual"], target_um
            ),
            "direct_J_source": direct_source,
            "steps_csv": str(steps_path),
            "case_root": str(case),
            **energy_audit,
        }
        emit = float(record["J_emit_target_kJ_m2"])
        record["J_emit_over_J_direct_average"] = (
            emit / direct_average
            if math.isfinite(emit) and math.isfinite(direct_average) and direct_average > 0.0
            else float("nan")
        )
        records.append(record)

    options_requested = [str(v) for v in manifest.get("options", DEFAULT_OPTION_ORDER)]
    present = {str(row["option_key"]) for row in records}
    options = [option for option in options_requested if option in present]
    options.extend(sorted(present - set(options)))
    rank = {option: index for index, option in enumerate(options)}
    records.sort(key=lambda row: (rank.get(str(row["option_key"]), 999), row["temperature_K"]))

    expected = len(manifest.get("options", [])) * len(manifest.get("temperatures_K", []))
    if expected and len(records) != expected:
        raise RuntimeError(
            f"J/energy summary found {len(records)} complete curves; expected {expected}"
        )
    for row in records:
        for key, _ in DIRECT_METRICS:
            if not math.isfinite(float(row[key])):
                raise RuntimeError(
                    f"non-finite {key} for {row['option_key']} at {row['temperature_K']} K"
                )
        if not math.isfinite(float(row["J_emit_target_kJ_m2"])):
            raise RuntimeError(
                f"non-finite J_emit for {row['option_key']} at {row['temperature_K']} K"
            )

    csv_path = outroot / "v10_2_27_paper_four_class_J_energy_vs_temperature_summary.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")

    markers = ["o", "s", "^", "D", "v", "P"]
    for option in options:
        subset = [row for row in records if row["option_key"] == option]
        subset.sort(key=lambda row: row["temperature_K"])
        temperatures = [row["temperature_K"] for row in subset]

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for index, (key, label) in enumerate(DIRECT_METRICS):
            _plot_finite_series(
                ax,
                temperatures,
                [row[key] for row in subset],
                label,
                markers[index % len(markers)],
            )
        ax.set_xlabel("Temperature, T (K)")
        ax.set_ylabel("Direct configurational J (kJ/m²)")
        ax.set_title(f"{SHORT_LABELS.get(option, option)}: direct FEM J versus T")
        ax.set_ylim(bottom=0.0)
        ax.legend(frameon=True)
        _save(fig, plot_dir / "by_candidate" / f"{option}_direct_J_metrics_vs_temperature.png")

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        comparison = (
            ("J_direct_average_kJ_m2", "Extension-averaged direct FEM J"),
            ("J_emit_target_kJ_m2", "Tip-emission work / target crack area"),
            ("J_bulk_plastic_target_kJ_m2", "Bulk plastic work / target crack area"),
            ("J_fracture_residual_target_kJ_m2", "Fracture residual / target crack area"),
        )
        plotted = False
        for index, (key, label) in enumerate(comparison):
            plotted |= _plot_finite_series(
                ax,
                temperatures,
                [row[key] for row in subset],
                label,
                markers[index % len(markers)],
            )
        ax.set_xlabel("Temperature, T (K)")
        ax.set_ylabel("Energy per projected crack area (kJ/m²)")
        ax.set_title(f"{SHORT_LABELS.get(option, option)}: J and dissipation comparison")
        ax.set_ylim(bottom=0.0)
        if plotted:
            ax.legend(frameon=True)
        _save(fig, plot_dir / "by_candidate" / f"{option}_J_direct_vs_dissipation_vs_temperature.png")

    for key, label in DIRECT_METRICS + ENERGY_METRICS:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        plotted = False
        for index, option in enumerate(options):
            subset = [row for row in records if row["option_key"] == option]
            subset.sort(key=lambda row: row["temperature_K"])
            plotted |= _plot_finite_series(
                ax,
                [row["temperature_K"] for row in subset],
                [row[key] for row in subset],
                SHORT_LABELS.get(option, option),
                markers[index % len(markers)],
            )
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Temperature, T (K)")
        ax.set_ylabel("Energy per projected crack area (kJ/m²)")
        ax.set_title(f"{label} versus temperature")
        ax.set_ylim(bottom=0.0)
        ax.legend(frameon=True)
        _save(fig, plot_dir / "by_metric" / f"{key}_vs_temperature.png")

    audit = {
        "schema": "v10.2.27_four_class_direct_J_and_energy_vs_temperature_v1",
        "target_extension_um": target_um,
        "intermediate_extension_um": intermediate_um,
        "direct_J_definition": (
            "signed-positive configurational FEM J read directly from accepted "
            "root-front diagnostics"
        ),
        "J_emit_definition": "W_tip_emit(target)/target_projected_crack_extension",
        "J_bulk_definition": "W_bulk_plastic(target)/target_projected_crack_extension",
        "J_irreversible_definition": "(W_ext-U_elastic)/target_projected_crack_extension",
        "J_fracture_residual_definition": (
            "(W_ext-U_elastic-W_bulk_plastic-W_tip_emit)/"
            "target_projected_crack_extension; fracture work plus numerical error"
        ),
        "direct_J_not_reconstructed_with_apparent_modulus": True,
        "supports_tip_only_and_full_field_bulk_plasticity": True,
        "legacy_tip_only_recovery_supported": True,
        "case_count": len(records),
        "expected_case_count": expected or None,
        "all_expected_cases_processed": len(records) == expected if expected else None,
        "manifest_inferred": bool(manifest.get("manifest_inferred", False)),
        "summary_csv": str(csv_path),
        "summary_json": str(json_path),
        "plot_directory": str(plot_dir),
    }
    audit_path = outroot / "v10_2_27_J_energy_vs_temperature_postprocess_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"Wrote direct-J and energy summaries for {len(records)} cases: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
