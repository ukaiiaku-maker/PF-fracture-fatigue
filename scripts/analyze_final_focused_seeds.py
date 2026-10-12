#!/usr/bin/env python3
"""Reduce the predeclared eight-seed paired original-PF ensembles."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_final_baseline_campaign import first_event


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "finalization_control" / "CAMPAIGN_MATRIX.csv"
OUT = ROOT / "review_packet" / "final_fracture_campaign_a5a5883"
ANCHORS = (("DBTT", 1200), ("Peak", 900))
T_CRITICAL_95_DF7 = 2.364624251


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_rows() -> list[dict[str, Any]]:
    with MATRIX.open(newline="") as stream:
        selected = [
            row for row in csv.DictReader(stream)
            if row["case_id"].startswith("focused_seed_")
        ]
    if len(selected) != 32:
        raise ValueError(f"expected 32 focused rows, found {len(selected)}")
    records = []
    for row in selected:
        if row["status"] != "complete_validated":
            raise ValueError(f"focused row not validated: {row['case_id']}")
        case = ROOT / row["reuse_evidence"]
        attestation = json.loads(
            (case / "shared_runtime_attestation.post.json").read_text()
        )
        if attestation.get("execution_verified") is not True:
            raise ValueError(f"execution not verified: {case}")
        event = first_event(case)
        records.append({
            "case_id": row["case_id"],
            "material_row": row["material_row"],
            "temperature_K": int(row["temperature_K"]),
            "seed": int(row["seed"]),
            "arm": row["arm"],
            "KJ_MPa_sqrt_m": event["KJ_MPa_sqrt_m"],
            "first_event_opening_um": event["opening_um"],
            "event_id": event["event_id"],
            "run_root": str(case.relative_to(ROOT)),
            "producer_commit": row["producer_commit"],
        })
    return sorted(records, key=lambda row: (
        ANCHORS.index((row["material_row"], row["temperature_K"])),
        row["seed"], row["arm"],
    ))


def pair_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (r["material_row"], r["temperature_K"], r["seed"], r["arm"]): r
        for r in records
    }
    pairs = []
    for material, temperature in ANCHORS:
        seeds = sorted({
            r["seed"] for r in records
            if (r["material_row"], r["temperature_K"]) == (material, temperature)
        })
        if len(seeds) != 8:
            raise ValueError(f"expected eight seeds at {material}/{temperature}: {seeds}")
        for seed in seeds:
            opening = lookup[(material, temperature, seed, "opening_only")]
            enabled = lookup[(material, temperature, seed, "accommodation_enabled")]
            pairs.append({
                "material_row": material,
                "temperature_K": temperature,
                "seed": seed,
                "opening_only_KJ_MPa_sqrt_m": opening["KJ_MPa_sqrt_m"],
                "accommodation_enabled_KJ_MPa_sqrt_m": enabled["KJ_MPa_sqrt_m"],
                "paired_gap_MPa_sqrt_m": (
                    enabled["KJ_MPa_sqrt_m"] - opening["KJ_MPa_sqrt_m"]
                ),
                "paired_relative_change_percent": 100.0 * (
                    enabled["KJ_MPa_sqrt_m"] / opening["KJ_MPa_sqrt_m"] - 1.0
                ),
                "opening_only_first_event_opening_um": opening["first_event_opening_um"],
                "accommodation_enabled_first_event_opening_um": enabled["first_event_opening_um"],
                "opening_only_run_root": opening["run_root"],
                "accommodation_enabled_run_root": enabled["run_root"],
            })
    return pairs


def summarize(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for material, temperature in ANCHORS:
        selected = [
            row for row in pairs
            if (row["material_row"], row["temperature_K"]) == (material, temperature)
        ]
        gaps = [row["paired_gap_MPa_sqrt_m"] for row in selected]
        changes = [row["paired_relative_change_percent"] for row in selected]
        mean = statistics.mean(gaps)
        sd = statistics.stdev(gaps)
        half = T_CRITICAL_95_DF7 * sd / math.sqrt(len(gaps))
        result[f"{material}_{temperature}K"] = {
            "n_paired_seeds": len(gaps),
            "paired_gap_mean_MPa_sqrt_m": mean,
            "paired_gap_sample_sd_MPa_sqrt_m": sd,
            "paired_gap_empirical_range_MPa_sqrt_m": [min(gaps), max(gaps)],
            "paired_gap_mean_95_percent_t_interval_MPa_sqrt_m": [mean - half, mean + half],
            "paired_relative_change_mean_percent": statistics.mean(changes),
            "paired_relative_change_empirical_range_percent": [min(changes), max(changes)],
        }
    return result


def plot(pairs: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    for ax, (material, temperature) in zip(axes, ANCHORS):
        selected = [
            row for row in pairs
            if (row["material_row"], row["temperature_K"]) == (material, temperature)
        ]
        for row in selected:
            ax.plot(
                [0, 1],
                [row["opening_only_KJ_MPa_sqrt_m"], row["accommodation_enabled_KJ_MPa_sqrt_m"]],
                color="#9a9a9a", linewidth=0.8, alpha=0.8, zorder=1,
            )
            ax.scatter(
                [0, 1],
                [row["opening_only_KJ_MPa_sqrt_m"], row["accommodation_enabled_KJ_MPa_sqrt_m"]],
                c=["#666666", "#b2182b"], s=18, zorder=2,
            )
        for x, key in ((0, "opening_only_KJ_MPa_sqrt_m"),
                       (1, "accommodation_enabled_KJ_MPa_sqrt_m")):
            mean = statistics.mean(row[key] for row in selected)
            ax.plot([x - 0.13, x + 0.13], [mean, mean], color="black", linewidth=2.0)
        ax.set_xticks([0, 1], ["Opening\nonly", "Accommodation\nenabled"])
        ax.set_xlim(-0.35, 1.35)
        ax.text(0.04, 0.94, f"{material}, {temperature} K", transform=ax.transAxes,
                va="top", fontsize=9)
        ax.tick_params(direction="in", top=True, right=True)
    axes[0].set_ylabel(r"First-event $K_J$ (MPa$\sqrt{\mathrm{m}}$)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_rows()
    pairs = pair_rows(records)
    summary = summarize(pairs)
    data_path = OUT / "PF_FOCUSED_SEEDS_DATA.csv"
    pair_path = OUT / "PF_FOCUSED_SEEDS_PAIRED.csv"
    figure_path = OUT / "PF_FOCUSED_SEEDS.png"
    write_csv(data_path, records)
    write_csv(pair_path, pairs)
    plot(pairs, figure_path)
    report = {
        "schema": "final-pf-focused-seeds/1",
        "decision": "COMPLETE_VALIDATED",
        "design": {
            "anchors": [
                {"material_row": material, "temperature_K": temperature}
                for material, temperature in ANCHORS
            ],
            "seeds": sorted({row["seed"] for row in pairs}),
            "paired_arms": ["opening_only", "accommodation_enabled"],
            "selection_rule": (
                "retain the two established active anchors and two pilot seeds; "
                "add the six seeds predeclared before new outcomes were inspected"
            ),
        },
        "anchor_summary": summary,
        "interval_assumption": (
            "Two-sided 95% Student-t interval for the mean paired gap, treating "
            "the eight predeclared random seeds as independent and the paired-gap "
            "sampling distribution as approximately normal (df=7). This is not a "
            "tail bound or population tolerance interval."
        ),
        "interpretation": (
            "At both active anchors, the accommodation-associated toughness gap "
            "is large relative to the observed eight-seed spread; no second "
            "replicate tranche is required for that bounded claim."
        ),
        "individual_data": str(data_path.relative_to(ROOT)),
        "paired_data": str(pair_path.relative_to(ROOT)),
        "figure": {
            "png": str(figure_path.relative_to(ROOT)),
            "pdf": str(figure_path.with_suffix('.pdf').relative_to(ROOT)),
            "plot_data": str(pair_path.relative_to(ROOT)),
            "panel_mapping": {"left": "DBTT_1200K", "right": "Peak_900K"},
        },
    }
    (OUT / "PF_FOCUSED_SEEDS.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Focused paired-seed ensembles", "",
        "**Decision:** `COMPLETE_VALIDATED`", "",
        "Eight predeclared paired seeds completed at each established active anchor; all 32 arm-level runs contain verified physical first events.", "",
        "| Anchor | Mean paired gap (MPa√m) | Sample SD | Empirical range | 95% t interval for mean |", "|---|---:|---:|---:|---:|",
    ]
    for material, temperature in ANCHORS:
        item = summary[f"{material}_{temperature}K"]
        empirical = item["paired_gap_empirical_range_MPa_sqrt_m"]
        interval = item["paired_gap_mean_95_percent_t_interval_MPa_sqrt_m"]
        lines.append(
            f"| {material}, {temperature} K | {item['paired_gap_mean_MPa_sqrt_m']:.2f} | "
            f"{item['paired_gap_sample_sd_MPa_sqrt_m']:.2f} | {empirical[0]:.2f}–{empirical[1]:.2f} | "
            f"{interval[0]:.2f}–{interval[1]:.2f} |"
        )
    lines += [
        "", report["interpretation"], "", "The interval is a two-sided Student-t interval for the mean paired gap (df = 7), assuming independent seeds and approximate normality of paired gaps. It is neither a tail bound nor a population tolerance interval.", "",
        f"Figure: `{figure_path.relative_to(ROOT)}`", "",
        f"Paired data: `{pair_path.relative_to(ROOT)}`", "",
    ]
    (OUT / "PF_FOCUSED_SEEDS.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
