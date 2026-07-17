#!/usr/bin/env python3
"""Analyze v10.1.6 class/temperature/ablation calculations.

The central response metric is initiation-referenced and geometry-corrected:

    plastic_late_rise = (late K - K_init)_full
                        - (late K - K_init)_plasticity_off

This removes the common directional-J/crack-path shape to first order.  The
script reports the raw curves and the matched ablation difference; it does not
feed any temperature-dependent target back into the constitutive model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _load_case(case_dir: Path, mode: str, material_class: str, temperature: float) -> dict[str, Any]:
    summary_path = case_dir / "summary.json"
    audit_path = case_dir / "kinetic_tip_cell_audit_v101.json"
    if not summary_path.exists() or not audit_path.exists():
        raise FileNotFoundError(f"missing summary/audit in {case_dir}")

    summary = json.loads(summary_path.read_text())
    if not summary or not isinstance(summary, list):
        raise ValueError(f"invalid summary in {summary_path}")
    row = summary[0]

    audit = json.loads(audit_path.read_text())
    records = audit.get("records", [])
    if not records:
        raise ValueError(f"no kinetic records in {audit_path}")

    event_records = [r for r in records if bool(r.get("fired", False))]
    event_k = [_float(r.get("K_Pa_sqrt_m")) / 1.0e6 for r in event_records]
    event_k = [v for v in event_k if math.isfinite(v)]
    k_init = _float(row.get("Kc_first_MPa_sqrt_m"))
    if not event_k and math.isfinite(k_init):
        event_k = [k_init]
    if not event_k:
        raise ValueError(f"no crack events in {case_dir}")

    first = event_k[0]
    late_n = min(3, len(event_k))
    late_mean = sum(event_k[-late_n:]) / late_n
    peak = max(event_k)

    def values(*keys: str) -> list[float]:
        found: list[float] = []
        for record in records:
            for key in keys:
                if key in record:
                    value = _float(record[key])
                    if math.isfinite(value):
                        found.append(value)
                    break
        return found

    active_mobile = values("active_mobile")
    active_retained = values("active_retained")
    backstress = values(
        "sigma_emission_backstress_Pa",
        "tip_source_backstress_equivalent_Pa",
    )
    shielding = values(
        "campaign_active_K_shield_effective_Pa_sqrt_m",
        "active_K_shield_signed_Pa_sqrt_m",
    )
    budget_consumed = values("campaign_source_budget_consumed")
    budget_total = values("campaign_source_budget_total")

    return {
        "mode": mode,
        "class": material_class,
        "temperature_K": temperature,
        "case_dir": str(case_dir),
        "K_init_MPa_sqrt_m": first,
        "K_final_event_MPa_sqrt_m": event_k[-1],
        "K_peak_event_MPa_sqrt_m": peak,
        "R_rise_final_MPa_sqrt_m": event_k[-1] - first,
        "R_rise_late_MPa_sqrt_m": late_mean - first,
        "R_rise_peak_MPa_sqrt_m": peak - first,
        "n_events": len(event_k),
        "n_advances": int(row.get("n_advances", len(event_k))),
        "max_active_mobile": max(active_mobile, default=0.0),
        "max_active_retained": max(active_retained, default=0.0),
        "max_active_population": max(
            (
                _float(r.get("active_mobile"), 0.0)
                + _float(r.get("active_retained"), 0.0)
                for r in records
            ),
            default=0.0,
        ),
        "final_active_population": (
            _float(records[-1].get("active_mobile"), 0.0)
            + _float(records[-1].get("active_retained"), 0.0)
        ),
        "max_emission_backstress_GPa": max(backstress, default=0.0) / 1.0e9,
        "max_active_K_shield_MPa_sqrt_m": max((abs(v) for v in shielding), default=0.0) / 1.0e6,
        "max_source_budget_consumed": max(budget_consumed, default=0.0),
        "source_budget_total": max(budget_total, default=0.0),
        "r_eff_over_r0_at_initiation": _float(row.get("r_eff_over_r0_init")),
        "event_K_MPa_sqrt_m": event_k,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], excluded: set[str] | None = None) -> None:
    excluded = excluded or set()
    if not rows:
        path.write_text("")
        return
    fields = [k for k in rows[0] if k not in excluded]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _paired_rows(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["class"], row["temperature_K"], row["mode"]): row
        for row in case_rows
    }
    pairs: list[dict[str, Any]] = []
    keys = sorted({(r["class"], r["temperature_K"]) for r in case_rows})
    for material_class, temperature in keys:
        full = indexed.get((material_class, temperature, "full"))
        off = indexed.get((material_class, temperature, "plasticity_off"))
        if full is None or off is None:
            continue
        pairs.append({
            "class": material_class,
            "temperature_K": temperature,
            "K_init_full_MPa_sqrt_m": full["K_init_MPa_sqrt_m"],
            "K_init_off_MPa_sqrt_m": off["K_init_MPa_sqrt_m"],
            "plastic_initiation_shift_MPa_sqrt_m": (
                full["K_init_MPa_sqrt_m"] - off["K_init_MPa_sqrt_m"]
            ),
            "plastic_R_rise_final_MPa_sqrt_m": (
                full["R_rise_final_MPa_sqrt_m"] - off["R_rise_final_MPa_sqrt_m"]
            ),
            "plastic_R_rise_late_MPa_sqrt_m": (
                full["R_rise_late_MPa_sqrt_m"] - off["R_rise_late_MPa_sqrt_m"]
            ),
            "plastic_R_rise_peak_MPa_sqrt_m": (
                full["R_rise_peak_MPa_sqrt_m"] - off["R_rise_peak_MPa_sqrt_m"]
            ),
            "max_active_population": full["max_active_population"],
            "max_emission_backstress_GPa": full["max_emission_backstress_GPa"],
            "max_active_K_shield_MPa_sqrt_m": full["max_active_K_shield_MPa_sqrt_m"],
            "r_eff_over_r0_at_initiation": full["r_eff_over_r0_at_initiation"],
        })
    return pairs


def _assessment(
    paired: list[dict[str, Any]],
    dbtt_low_max: float,
    dbtt_min_emergence: float,
    flat_max_span: float,
) -> dict[str, Any]:
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in paired:
        by_class.setdefault(str(row["class"]), []).append(row)
    for rows in by_class.values():
        rows.sort(key=lambda r: float(r["temperature_K"]))

    result: dict[str, Any] = {
        "metric": "plastic_R_rise_late_MPa_sqrt_m",
        "thresholds_are_validation_only": True,
        "thresholds": {
            "dbtt_low_max_abs_MPa_sqrt_m": dbtt_low_max,
            "dbtt_min_high_minus_low_MPa_sqrt_m": dbtt_min_emergence,
            "weakT_and_ceramic_max_temperature_span_MPa_sqrt_m": flat_max_span,
        },
    }

    dbtt = by_class.get("DBTT", [])
    if len(dbtt) >= 2:
        low = float(dbtt[0]["plastic_R_rise_late_MPa_sqrt_m"])
        high = float(dbtt[-1]["plastic_R_rise_late_MPa_sqrt_m"])
        result["DBTT"] = {
            "low_temperature_K": dbtt[0]["temperature_K"],
            "high_temperature_K": dbtt[-1]["temperature_K"],
            "low_plastic_R_rise": low,
            "high_plastic_R_rise": high,
            "emergence_high_minus_low": high - low,
            "low_T_weak": abs(low) <= dbtt_low_max,
            "high_T_emergent": (high - low) >= dbtt_min_emergence,
        }

    for material_class in ("weakT", "ceramic"):
        rows = by_class.get(material_class, [])
        if rows:
            values = [float(r["plastic_R_rise_late_MPa_sqrt_m"]) for r in rows]
            span = max(values) - min(values)
            result[material_class] = {
                "temperature_span": span,
                "comparatively_temperature_flat": span <= flat_max_span,
                "values": values,
                "temperatures_K": [r["temperature_K"] for r in rows],
            }

    statuses: list[bool] = []
    for section in result.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if key in {"low_T_weak", "high_T_emergent", "comparatively_temperature_flat"}:
                    statuses.append(bool(value))
    result["overall_pass"] = bool(statuses) and all(statuses)
    return result


def _plots(root: Path, case_rows: list[dict[str, Any]], paired: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    for metric, ylabel, filename in (
        ("plastic_R_rise_late_MPa_sqrt_m", "Plastic late R-curve rise (MPa√m)", "plastic_R_rise_vs_temperature.png"),
        ("plastic_initiation_shift_MPa_sqrt_m", "Plastic initiation shift (MPa√m)", "plastic_initiation_shift_vs_temperature.png"),
        ("max_active_population", "Maximum active population", "active_population_vs_temperature.png"),
        ("max_emission_backstress_GPa", "Maximum emission back stress (GPa)", "emission_backstress_vs_temperature.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for material_class in sorted({str(r["class"]) for r in paired}):
            rows = sorted(
                [r for r in paired if r["class"] == material_class],
                key=lambda r: float(r["temperature_K"]),
            )
            ax.plot(
                [r["temperature_K"] for r in rows],
                [r[metric] for r in rows],
                marker="o",
                label=material_class,
            )
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / filename, dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for material_class in sorted({str(r["class"]) for r in case_rows}):
        rows = sorted(
            [r for r in case_rows if r["class"] == material_class and r["mode"] == "full"],
            key=lambda r: float(r["temperature_K"]),
        )
        ax.plot(
            [r["temperature_K"] for r in rows],
            [r["K_init_MPa_sqrt_m"] for r in rows],
            marker="o",
            label=material_class,
        )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("First-passage toughness (MPa√m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "K_init_vs_temperature.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=["ceramic", "weakT", "DBTT"])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[300.0, 700.0, 1100.0])
    parser.add_argument("--modes", nargs="+", default=["full", "plasticity_off"])
    parser.add_argument("--theta", type=float, default=45.0)
    parser.add_argument("--dbtt-low-max", type=float, default=0.5)
    parser.add_argument("--dbtt-min-emergence", type=float, default=1.0)
    parser.add_argument("--flat-max-span", type=float, default=1.0)
    args = parser.parse_args()

    root = args.root.resolve()
    case_rows: list[dict[str, Any]] = []
    for mode in args.modes:
        for material_class in args.classes:
            for temperature in args.temperatures:
                tlabel = f"{temperature:g}"
                case_dir = root / mode / material_class / f"T{tlabel}_th{args.theta:g}"
                case_rows.append(_load_case(case_dir, mode, material_class, temperature))

    case_rows.sort(key=lambda r: (str(r["class"]), float(r["temperature_K"]), str(r["mode"])))
    paired = _paired_rows(case_rows)
    _write_csv(root / "temperature_matrix_case_summary.csv", case_rows, {"event_K_MPa_sqrt_m"})
    _write_csv(root / "temperature_matrix_ablation_summary.csv", paired)
    (root / "temperature_matrix_event_K.json").write_text(json.dumps([
        {
            "mode": r["mode"],
            "class": r["class"],
            "temperature_K": r["temperature_K"],
            "event_K_MPa_sqrt_m": r["event_K_MPa_sqrt_m"],
        }
        for r in case_rows
    ], indent=2))

    assessment = _assessment(
        paired,
        args.dbtt_low_max,
        args.dbtt_min_emergence,
        args.flat_max_span,
    )
    (root / "temperature_emergence_assessment.json").write_text(json.dumps(assessment, indent=2))
    _plots(root, case_rows, paired)

    print("\nInitiation-referenced plastic R-curve contribution")
    print("class      T(K)   init_shift   late_rise   peak_rise   max_active   backstress_GPa")
    for row in paired:
        print(
            f"{row['class']:<9s} {row['temperature_K']:6.0f} "
            f"{row['plastic_initiation_shift_MPa_sqrt_m']:11.4f} "
            f"{row['plastic_R_rise_late_MPa_sqrt_m']:11.4f} "
            f"{row['plastic_R_rise_peak_MPa_sqrt_m']:11.4f} "
            f"{row['max_active_population']:11.4f} "
            f"{row['max_emission_backstress_GPa']:14.4f}"
        )
    print(f"\nassessment: {'PASS' if assessment.get('overall_pass') else 'REVIEW'}")
    print(root / "temperature_emergence_assessment.json")


if __name__ == "__main__":
    main()
