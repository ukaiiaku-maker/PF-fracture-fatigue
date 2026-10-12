#!/usr/bin/env python3
"""Reduce the predeclared rate, orientation, and source PF sensitivities."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_final_baseline_campaign import first_event


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "finalization_control" / "CAMPAIGN_MATRIX.csv"
OUT = ROOT / "review_packet" / "final_fracture_campaign_a5a5883"
ARMS = ("opening_only", "accommodation_enabled")
ARM_LABELS = {
    "opening_only": "Opening only",
    "accommodation_enabled": "Accommodation enabled",
}
RATE_CONDITIONS = (
    ("DBTT", 950), ("DBTT", 1000), ("DBTT", 1200),
    ("Peak", 800), ("Peak", 900),
)
ORIENTATION_CONDITIONS = (
    ("DBTT", 300), ("DBTT", 1200), ("Peak", 300), ("Peak", 900),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_matrix() -> list[dict[str, str]]:
    with MATRIX.open(newline="") as stream:
        return list(csv.DictReader(stream))


def reduce_row(row: dict[str, str]) -> dict[str, Any]:
    if row["status"] != "complete_validated":
        raise ValueError(f"sensitivity row is not validated: {row['case_id']}")
    case = ROOT / row["reuse_evidence"]
    attestation_path = case / "shared_runtime_attestation.post.json"
    summary_path = case / "summary.json"
    attestation = json.loads(attestation_path.read_text())
    if attestation.get("execution_verified") is not True:
        raise ValueError(f"execution is not verified: {case}")
    summary_payload = json.loads(summary_path.read_text())
    summary = summary_payload[0] if isinstance(summary_payload, list) else summary_payload
    event = first_event(case)
    if abs(float(summary["Kc_first_MPa_sqrt_m"]) - event["KJ_MPa_sqrt_m"]) > 1e-9:
        raise ValueError(f"summary/observer mismatch: {case}")
    return {
        "case_id": row["case_id"],
        "material_row": row["material_row"],
        "temperature_K": int(row["temperature_K"]),
        "arm": row["arm"],
        "seed": int(row["seed"]),
        "orientation_deg": float(row["orientation_deg"]),
        "rate_factor": float(row["rate_factor"]),
        "source_geography": row["source_geography"],
        "KJ_MPa_sqrt_m": event["KJ_MPa_sqrt_m"],
        "first_event_opening_um": event["opening_um"],
        "emitted_population": event["emitted_population"],
        "back_stress_GPa": event["back_stress_GPa"],
        "run_root": str(case.relative_to(ROOT)),
        "producer_commit": row["producer_commit"],
        "summary_sha256": sha256(summary_path),
        "attestation_sha256": sha256(attestation_path),
    }


def baseline_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    selected = [row for row in rows if row["case_id"].startswith("baseline_")]
    return {
        (record["material_row"], record["temperature_K"], record["arm"]): record
        for record in (reduce_row(row) for row in selected)
    }


def rate_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (r["material_row"], r["temperature_K"], r["arm"], r["rate_factor"]): r
        for r in records
    }
    result = []
    for material, temperature in RATE_CONDITIONS:
        for arm in ARMS:
            reference = lookup[(material, temperature, arm, 1.0)]["KJ_MPa_sqrt_m"]
            for factor in (0.01, 1.0, 100.0):
                row = lookup[(material, temperature, arm, factor)]
                result.append({
                    **row,
                    "KJ_over_1x": row["KJ_MPa_sqrt_m"] / reference,
                    "KJ_change_from_1x_percent": 100.0 * (
                        row["KJ_MPa_sqrt_m"] / reference - 1.0
                    ),
                })
    return result


def orientation_pairs(
    records: list[dict[str, Any]], baseline: dict[tuple[str, int, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    lookup = {
        (r["material_row"], r["temperature_K"], r["arm"]): r for r in records
    }
    result = []
    for material, temperature in ORIENTATION_CONDITIONS:
        for arm in ARMS:
            zero = baseline[(material, temperature, arm)]
            tilted = lookup[(material, temperature, arm)]
            result.append({
                "material_row": material,
                "temperature_K": temperature,
                "arm": arm,
                "zero_deg_KJ_MPa_sqrt_m": zero["KJ_MPa_sqrt_m"],
                "forty_five_deg_KJ_MPa_sqrt_m": tilted["KJ_MPa_sqrt_m"],
                "orientation_KJ_change_percent": 100.0 * (
                    tilted["KJ_MPa_sqrt_m"] / zero["KJ_MPa_sqrt_m"] - 1.0
                ),
                "zero_deg_run_root": zero["run_root"],
                "forty_five_deg_run_root": tilted["run_root"],
            })
    return result


def source_rows(
    records: list[dict[str, Any]], baseline: dict[tuple[str, int, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    tip = baseline[("DBTT", 1200, "accommodation_enabled")]
    selected = [{**tip, "source_geography": "tip_surface"}, *records]
    result = []
    for row in selected:
        result.append({
            **row,
            "KJ_change_from_tip_percent": 100.0 * (
                row["KJ_MPa_sqrt_m"] / tip["KJ_MPa_sqrt_m"] - 1.0
            ),
        })
    order = {"tip_surface": 0, "bulk_uniform": 1, "tip_plus_bulk": 2}
    return sorted(result, key=lambda row: order[row["source_geography"]])


def plot_rate(rows: list[dict[str, Any]], path: Path) -> None:
    colors = {950: "#2166ac", 1000: "#67a9cf", 1200: "#d6604d", 800: "#2166ac", 900: "#b2182b"}
    styles = {"opening_only": "--", "accommodation_enabled": "-"}
    markers = {"opening_only": "o", "accommodation_enabled": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=True)
    for ax, material in zip(axes, ("DBTT", "Peak")):
        conditions = [condition for condition in RATE_CONDITIONS if condition[0] == material]
        for _, temperature in conditions:
            for arm in ARMS:
                selected = [
                    row for row in rows
                    if row["material_row"] == material
                    and row["temperature_K"] == temperature and row["arm"] == arm
                ]
                ax.plot(
                    [row["rate_factor"] for row in selected],
                    [row["KJ_over_1x"] for row in selected],
                    color=colors[temperature], linestyle=styles[arm],
                    marker=markers[arm], markersize=3.5, linewidth=1.15,
                    label=f"{temperature} K, {ARM_LABELS[arm]}",
                )
        ax.set_xscale("log")
        ax.set_xticks([0.01, 1.0, 100.0], ["0.01", "1", "100"])
        ax.axhline(1.0, color="#777777", linewidth=0.7)
        ax.text(0.04, 0.94, material, transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xlabel("Opening-rate factor")
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(frameon=False, fontsize=6.8, loc="best")
    axes[0].set_ylabel(r"First-event $K_J/K_J(1\times)$")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_orientation(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.45), sharey=True)
    for ax, (material, temperature) in zip(axes.flat, ORIENTATION_CONDITIONS):
        selected = [
            row for row in rows
            if row["material_row"] == material and row["temperature_K"] == temperature
        ]
        selected.sort(key=lambda row: ARMS.index(row["arm"]))
        for index, row in enumerate(selected):
            ax.plot(
                [index - 0.11, index + 0.11],
                [row["zero_deg_KJ_MPa_sqrt_m"], row["forty_five_deg_KJ_MPa_sqrt_m"]],
                color="#999999", linewidth=0.8,
            )
            ax.scatter(index - 0.11, row["zero_deg_KJ_MPa_sqrt_m"], color="#666666", s=18)
            ax.scatter(index + 0.11, row["forty_five_deg_KJ_MPa_sqrt_m"], color="#b2182b", s=18)
        ax.set_xticks([0, 1], ["Opening\nonly", "Accommodation\nenabled"])
        ax.text(0.04, 0.94, f"{material}, {temperature} K", transform=ax.transAxes,
                va="top", fontsize=9)
        ax.tick_params(direction="in", top=True, right=True)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"First-event $K_J$ (MPa$\sqrt{\mathrm{m}}$)")
    axes[0, 0].scatter([], [], color="#666666", s=18, label="0°")
    axes[0, 0].scatter([], [], color="#b2182b", s=18, label="45°")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_source(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.25))
    labels = {"tip_surface": "Tip", "bulk_uniform": "Bulk", "tip_plus_bulk": "Tip + bulk"}
    x = range(len(rows))
    ax.plot(x, [row["KJ_MPa_sqrt_m"] for row in rows], color="#777777", linewidth=0.9)
    ax.scatter(x, [row["KJ_MPa_sqrt_m"] for row in rows], color="#b2182b", s=28)
    ax.set_xticks(list(x), [labels[row["source_geography"]] for row in rows])
    ax.set_ylabel(r"First-event $K_J$ (MPa$\sqrt{\mathrm{m}}$)")
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = load_matrix()
    sensitivity = [row for row in matrix if row["case_id"].startswith("sensitivity_")]
    if len(sensitivity) != 40:
        raise ValueError(f"expected 40 sensitivity rows, found {len(sensitivity)}")
    records = [reduce_row(row) for row in sensitivity]
    baseline = baseline_lookup(matrix)
    rate = rate_pairs([row for row in records if row["case_id"].startswith("sensitivity_rate_")])
    orientation = orientation_pairs(
        [row for row in records if row["case_id"].startswith("sensitivity_orientation_")],
        baseline,
    )
    source = source_rows(
        [row for row in records if row["case_id"].startswith("sensitivity_source_")],
        baseline,
    )
    data_path = OUT / "PF_SENSITIVITIES_DATA.csv"
    rate_path = OUT / "PF_RATE_SENSITIVITY.csv"
    orientation_path = OUT / "PF_ORIENTATION_SENSITIVITY.csv"
    source_path = OUT / "PF_SOURCE_GEOGRAPHY.csv"
    write_csv(data_path, records)
    write_csv(rate_path, rate)
    write_csv(orientation_path, orientation)
    write_csv(source_path, source)
    rate_figure = OUT / "PF_RATE_SENSITIVITY.png"
    orientation_figure = OUT / "PF_ORIENTATION_SENSITIVITY.png"
    source_figure = OUT / "PF_SOURCE_GEOGRAPHY.png"
    plot_rate(rate, rate_figure)
    plot_orientation(orientation, orientation_figure)
    plot_source(source, source_figure)

    max_rate = max(rate, key=lambda row: abs(row["KJ_change_from_1x_percent"]))
    max_orientation = max(
        orientation, key=lambda row: abs(row["orientation_KJ_change_percent"])
    )
    report = {
        "schema": "final-pf-rate-orientation-source/1",
        "decision": "COMPLETE_VALIDATED",
        "sensitivity_matrix_row_count": len(records),
        "rate": {
            "row_count": len(rate),
            "maximum_absolute_change_from_1x_percent": abs(max_rate["KJ_change_from_1x_percent"]),
            "maximum_change_case": {
                key: max_rate[key] for key in (
                    "material_row", "temperature_K", "arm", "rate_factor",
                    "KJ_change_from_1x_percent",
                )
            },
            "data": str(rate_path.relative_to(ROOT)),
            "figure_png": str(rate_figure.relative_to(ROOT)),
            "figure_pdf": str(rate_figure.with_suffix(".pdf").relative_to(ROOT)),
        },
        "orientation": {
            "paired_comparison_count": len(orientation),
            "change_percent_range": [
                min(row["orientation_KJ_change_percent"] for row in orientation),
                max(row["orientation_KJ_change_percent"] for row in orientation),
            ],
            "maximum_absolute_change_case": {
                key: max_orientation[key] for key in (
                    "material_row", "temperature_K", "arm",
                    "orientation_KJ_change_percent",
                )
            },
            "data": str(orientation_path.relative_to(ROOT)),
            "figure_png": str(orientation_figure.relative_to(ROOT)),
            "figure_pdf": str(orientation_figure.with_suffix(".pdf").relative_to(ROOT)),
        },
        "source_geography": {
            "condition": "DBTT, 1200 K, accommodation enabled, seed 1733",
            "rows": [
                {
                    key: row[key] for key in (
                        "source_geography", "KJ_MPa_sqrt_m", "KJ_change_from_tip_percent"
                    )
                } for row in source
            ],
            "data": str(source_path.relative_to(ROOT)),
            "figure_png": str(source_figure.relative_to(ROOT)),
            "figure_pdf": str(source_figure.with_suffix(".pdf").relative_to(ROOT)),
        },
        "all_case_data": str(data_path.relative_to(ROOT)),
        "analysis_command": (
            "/opt/homebrew/Caskroom/miniconda/base/envs/"
            "arrhenius-sharp-front-v10-codex/bin/python "
            "scripts/analyze_final_pf_sensitivities.py"
        ),
        "scope_note": (
            "These are question-driven local sensitivities at predeclared anchors, "
            "not a Cartesian parameter atlas. The 45-degree operator is the original "
            "FEM-measured, two-pass self-consistent signed-kernel family."
        ),
    }
    json_path = OUT / "PF_SENSITIVITIES.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    source_lines = [
        f"| {row['source_geography']} | {row['KJ_MPa_sqrt_m']:.3f} | "
        f"{row['KJ_change_from_tip_percent']:+.2f}% |" for row in source
    ]
    markdown = [
        "# PF rate, orientation, and source sensitivities", "",
        "**Decision:** `COMPLETE_VALIDATED`", "",
        "All 40 predeclared sensitivity-matrix rows have verified original spatial-PF executions and physical first events.", "",
        "## Loading rate", "",
        f"Across the five anchors, two arms, and 0.01×–100× range, the largest absolute first-event change from the matched 1× case is {report['rate']['maximum_absolute_change_from_1x_percent']:.2f}%.", "",
        "## Crystal orientation", "",
        f"The paired 45° minus 0° first-event changes span {report['orientation']['change_percent_range'][0]:+.2f}% to {report['orientation']['change_percent_range'][1]:+.2f}% across the four selected conditions and two arms.", "",
        "## Source geography", "",
        "DBTT, 1200 K, accommodation enabled, seed 1733:", "",
        "| Source geography | First-event KJ (MPa√m) | Change from tip |",
        "|---|---:|---:|", *source_lines, "",
        report["scope_note"], "",
        f"Machine-readable report: `{json_path.relative_to(ROOT)}`", "",
        f"Analysis command: `{report['analysis_command']}`", "",
    ]
    (OUT / "PF_SENSITIVITIES.md").write_text("\n".join(markdown))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
