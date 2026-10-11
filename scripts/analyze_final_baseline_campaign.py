#!/usr/bin/env python3
"""Reduce and audit the 96-case final original-PF baseline campaign."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "finalization_control" / "CAMPAIGN_MATRIX.csv"
OUT = ROOT / "review_packet" / "final_fracture_campaign_a5a5883"
MATERIAL_ORDER = ("DBTT", "Peak", "Weak-T", "Ceramic-like")
ARM_ORDER = ("opening_only", "accommodation_enabled")
ARM_LABEL = {
    "opening_only": "Opening only",
    "accommodation_enabled": "Accommodation enabled",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def first_event(case: Path) -> dict[str, Any]:
    events = []
    with (case / "fracture_observations_online.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("stage") == "pre_event" and row.get("event_id"):
                events.append(row)
    if not events:
        raise ValueError(f"no pre-event observation: {case}")
    event = events[0]
    step_path = Path(event["source_file"])
    if not step_path.is_file():
        step_path = case / step_path.name
    with step_path.open(newline="") as stream:
        steps = {
            number: row for number, row in enumerate(csv.DictReader(stream), 2)
        }
    step = steps[int(event["source_row"])]
    return {
        "event_id": event["event_id"],
        "observer_stage": event["stage"],
        "observer_valid": bool(event["observer_valid"]),
        "source_row": int(event["source_row"]),
        "opening_um": float(step["Uapp_m"]) * 1.0e6,
        "KJ_MPa_sqrt_m": float(event["K_J_Pa_sqrt_m"]) * 1.0e-6,
        "J_J_per_m2": float(event["J_J_per_m2"]),
        "emitted_population": float(step["N_em"]),
        "back_stress_GPa": float(step["sigma_back_Pa"]) * 1.0e-9,
        "tip_emission_work_J_per_m": float(step["W_tip_emit_cumulative_J_per_m"]),
        "observation_file": str(
            (case / "fracture_observations_online.jsonl").relative_to(ROOT)
        ),
        "steps_file": str(step_path.relative_to(ROOT)),
    }


def reduce_campaign() -> list[dict[str, Any]]:
    records = []
    with MATRIX.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    baseline = [row for row in rows if row["case_id"].startswith("baseline_")]
    if len(baseline) != 96:
        raise ValueError(f"expected 96 baseline rows, found {len(baseline)}")
    for row in baseline:
        if row["status"] != "complete_validated":
            raise ValueError(f"baseline row not validated: {row['case_id']}")
        case = ROOT / row["reuse_evidence"]
        attestation_path = case / "shared_runtime_attestation.post.json"
        attestation = load_json(attestation_path)
        if attestation.get("execution_verified") is not True:
            raise ValueError(f"execution not verified: {case}")
        summary_path = case / "summary.json"
        payload = load_json(summary_path)
        summary = payload[0] if isinstance(payload, list) else payload
        event = first_event(case)
        if abs(float(summary["Kc_first_MPa_sqrt_m"]) - event["KJ_MPa_sqrt_m"]) > 1.0e-9:
            raise ValueError(f"summary/observer first-event mismatch: {case}")
        records.append({
            "case_id": row["case_id"],
            "material_row": row["material_row"],
            "temperature_K": int(row["temperature_K"]),
            "arm": row["arm"],
            "seed": int(row["seed"]),
            "orientation_deg": float(row["orientation_deg"]),
            "rate_factor": float(row["rate_factor"]),
            "source_geography": row["source_geography"],
            "status": row["status"],
            "run_root": str(case.relative_to(ROOT)),
            "producer_commit": row["producer_commit"],
            "summary_sha256": sha256(summary_path),
            "attestation_sha256": sha256(attestation_path),
            **event,
            "final_extension_um": (
                float(summary["geometry_projected_extension_m"]) * 1.0e6
            ),
            "geometry_events": int(summary["n_geometry_events"]),
        })
    return sorted(
        records,
        key=lambda row: (
            MATERIAL_ORDER.index(row["material_row"]),
            row["temperature_K"],
            ARM_ORDER.index(row["arm"]),
        ),
    )


def paired(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (r["material_row"], r["temperature_K"], r["arm"]): r
        for r in records
    }
    result = []
    for material in MATERIAL_ORDER:
        temperatures = sorted({
            r["temperature_K"] for r in records if r["material_row"] == material
        })
        for temperature in temperatures:
            opening = lookup[(material, temperature, "opening_only")]
            enabled = lookup[(material, temperature, "accommodation_enabled")]
            result.append({
                "material_row": material,
                "temperature_K": temperature,
                "opening_only_KJ_MPa_sqrt_m": opening["KJ_MPa_sqrt_m"],
                "accommodation_enabled_KJ_MPa_sqrt_m": enabled["KJ_MPa_sqrt_m"],
                "accommodation_K_change_percent": 100.0 * (
                    enabled["KJ_MPa_sqrt_m"] / opening["KJ_MPa_sqrt_m"] - 1.0
                ),
                "opening_only_first_event_opening_um": opening["opening_um"],
                "accommodation_enabled_first_event_opening_um": enabled["opening_um"],
            })
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(records: list[dict[str, Any]], path: Path) -> None:
    colors = {"opening_only": "#666666", "accommodation_enabled": "#b2182b"}
    markers = {"opening_only": "o", "accommodation_enabled": "s"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
    for label, ax in zip(MATERIAL_ORDER, axes.flat):
        for arm in ARM_ORDER:
            selected = [
                row for row in records
                if row["material_row"] == label and row["arm"] == arm
            ]
            ax.plot(
                [row["temperature_K"] for row in selected],
                [row["KJ_MPa_sqrt_m"] for row in selected],
                color=colors[arm], marker=markers[arm], markersize=3.7,
                linewidth=1.35, label=ARM_LABEL[arm],
            )
        ax.text(0.04, 0.93, label, transform=ax.transAxes, va="top", fontsize=9)
        ax.tick_params(direction="in", top=True, right=True)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"First-event $K_J$ (MPa$\sqrt{\mathrm{m}}$)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Temperature (K)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records = reduce_campaign()
    pairs = paired(records)
    data_path = OUT / "PF_BASELINE_96_DATA.csv"
    pair_path = OUT / "PF_BASELINE_96_PAIRED.csv"
    figure_path = OUT / "PF_BASELINE_96_FIRST_EVENT.png"
    write_csv(data_path, records)
    write_csv(pair_path, pairs)
    plot(records, figure_path)
    extrema = {}
    for material in MATERIAL_ORDER:
        selected = [row for row in pairs if row["material_row"] == material]
        peak = max(selected, key=lambda row: row["accommodation_K_change_percent"])
        extrema[material] = {
            "peak_accommodation_effect_temperature_K": peak["temperature_K"],
            "peak_accommodation_K_change_percent": peak["accommodation_K_change_percent"],
            "enabled_KJ_range_MPa_sqrt_m": [
                min(row["accommodation_enabled_KJ_MPa_sqrt_m"] for row in selected),
                max(row["accommodation_enabled_KJ_MPa_sqrt_m"] for row in selected),
            ],
            "opening_only_KJ_range_MPa_sqrt_m": [
                min(row["opening_only_KJ_MPa_sqrt_m"] for row in selected),
                max(row["opening_only_KJ_MPa_sqrt_m"] for row in selected),
            ],
        }
    report = {
        "schema": "final-pf-baseline-96/1",
        "decision": "COMPLETE_VALIDATED",
        "case_count": len(records),
        "physical_first_event_count": sum(r["event_id"] != "" for r in records),
        "finite_no_event_censor_count": 0,
        "design": {
            "material_rows": list(MATERIAL_ORDER),
            "temperatures_K": sorted({r["temperature_K"] for r in records}),
            "arms": list(ARM_ORDER),
            "seed": 1733,
            "orientation_deg": 0.0,
            "rate_factor": 1.0,
            "source_geography": "tip_surface",
        },
        "material_summary": extrema,
        "interpretation": [
            "DBTT and Peak show abrupt accommodation-associated toughness upturns over the sampled transition range.",
            "Weak-T shows a broad accommodation-associated elevation rather than a sharp transition.",
            "Ceramic-like remains the lowest-toughness row and shows only a modest accommodation effect.",
            "These are single-seed baseline curves; focused independent seeds quantify stochastic spread at selected active anchors.",
        ],
        "figure": {
            "png": str(figure_path.relative_to(ROOT)),
            "pdf": str(figure_path.with_suffix('.pdf').relative_to(ROOT)),
            "plot_data": str(data_path.relative_to(ROOT)),
            "panel_mapping": {
                "upper_left": "DBTT", "upper_right": "Peak",
                "lower_left": "Weak-T", "lower_right": "Ceramic-like",
            },
            "x": "temperature_K",
            "y": "KJ_MPa_sqrt_m",
            "series": "arm",
        },
        "paired_table": str(pair_path.relative_to(ROOT)),
    }
    json_path = OUT / "PF_BASELINE_96.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Original-PF 96-case baseline campaign", "",
        "**Decision:** `COMPLETE_VALIDATED`", "",
        "All 96 predeclared cases executed with verified runtime attestations and a physical first fracture event. There were no finite no-event censors.", "",
        "| Material row | Enabled K range (MPa√m) | Opening-only K range (MPa√m) | Largest accommodation effect |", "|---|---:|---:|---:|",
    ]
    for material in MATERIAL_ORDER:
        item = extrema[material]
        lines.append(
            f"| {material} | {item['enabled_KJ_range_MPa_sqrt_m'][0]:.2f}–{item['enabled_KJ_range_MPa_sqrt_m'][1]:.2f} | "
            f"{item['opening_only_KJ_range_MPa_sqrt_m'][0]:.2f}–{item['opening_only_KJ_range_MPa_sqrt_m'][1]:.2f} | "
            f"{item['peak_accommodation_K_change_percent']:.1f}% at {item['peak_accommodation_effect_temperature_K']} K |"
        )
    lines += [
        "", "The DBTT and Peak rows exhibit abrupt accommodation-associated toughness upturns. Weak-T has a broad elevated response, while Ceramic-like remains lower-toughness with a modest effect. These curves are the seed-1733 baseline; independent-seed comparisons remain the uncertainty gate.", "",
        f"Figure: `{figure_path.relative_to(ROOT)}`", "",
        f"Plotted data: `{data_path.relative_to(ROOT)}`", "",
        f"Paired effects: `{pair_path.relative_to(ROOT)}`", "",
    ]
    (OUT / "PF_BASELINE_96.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
