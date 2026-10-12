#!/usr/bin/env python3
"""Publish the accepted hot/cold developed-growth and resistance evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.fracture_observation_history import ObservationStage
from scripts.export_physical_growth_packet import CASES, event_pairs, history_rows


SOURCE_PACKET = ROOT / "review_packet" / "current_science_pilots_6abbf10"
OUT = ROOT / "review_packet" / "final_fracture_campaign_a5a5883"
COLORS = {"cold_300K": "#2166ac", "hot_1200K": "#b2182b"}
PRODUCER_COMMITS = {
    "cold_300K": "6abbf10b8cb49b12b8e3d6cc4ad323f27a4b578b",
    "hot_1200K": "d77abff138f9a7f23bbabba7b3ba582cde35ce68",
}
IMPACT_PATHS = (
    "arrhenius_fracture/sharp_front.py",
    "arrhenius_fracture/energy_ledger_output_v10227.py",
    "arrhenius_fracture/fracture_observation_history.py",
    "arrhenius_fracture/sharp_front_v10_2_27.py",
    "arrhenius_fracture/sharp_front_v10_2_28.py",
    "arrhenius_fracture/sharp_front_v10_2_30_hazard_energy_gated.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return math.nan if value in (None, "") else float(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def selected_resistance(label, history, steps) -> list[dict[str, Any]]:
    """Select exact native pre-event resistance over each committed jump."""
    rows = []
    for event_number, (pre, committed) in enumerate(event_pairs(history), 1):
        if pre.stage is not ObservationStage.PRE_EVENT:
            raise ValueError("resistance selector requires a pre-event observation")
        if committed.stage is not ObservationStage.COMMITTED:
            raise ValueError("resistance selector requires the paired committed geometry")
        if not math.isfinite(pre.K_J_Pa_sqrt_m):
            raise ValueError(f"missing native pre-event K_J: {pre.event_id}")
        step = steps[pre.source_row]
        rows.append({
            "condition": label,
            "event_number": event_number,
            "event_id": pre.event_id,
            "front_id": pre.front_id,
            "topology_generation": pre.topology_generation,
            "pre_event_sequence": pre.sequence,
            "committed_sequence": committed.sequence,
            "source_row": pre.source_row,
            "model_time_s": pre.time_s,
            "opening_um": number(step, "Uapp_m") * 1.0e6,
            "reaction_N": number(step, "Ftop_N"),
            "pre_extension_um": pre.pre_length_m * 1.0e6,
            "committed_extension_um": committed.committed_length_m * 1.0e6,
            "pre_path_arclength_um": pre.path_arclength_m * 1.0e6,
            "committed_path_arclength_um": committed.path_arclength_m * 1.0e6,
            "event_projected_advance_um": (
                committed.committed_length_m - pre.pre_length_m
            ) * 1.0e6,
            "event_path_advance_um": (
                committed.path_arclength_m - pre.path_arclength_m
            ) * 1.0e6,
            "pre_event_K_J_MPa_sqrt_m": pre.K_J_Pa_sqrt_m * 1.0e-6,
            "pre_event_J_J_per_m2": pre.J_J_per_m2,
            "event_multiplicity": pre.event_multiplicity,
            "load_segment": pre.segment_id,
            "load_direction": pre.load_direction.value,
            "active_mobile_count": number(step, "mpz_mobile_count"),
            "active_retained_count": number(step, "mpz_retained_count"),
            "active_mobile_plus_retained_count": number(step, "N_em"),
            "wake_retained_total": number(step, "mpz_wake_retained_total"),
            "source_backstress_GPa": number(step, "sigma_back_Pa") * 1.0e-9,
            "active_K_shield_MPa_sqrt_m": number(step, "mpz_K_shield_Pa_sqrt_m") * 1.0e-6,
            "tip_emission_work_cumulative_J_per_m": number(
                step, "W_tip_emit_cumulative_J_per_m"
            ),
            "resistance_definition": (
                "native K_J at exact pre-event state, carried only across its "
                "paired realized committed jump"
            ),
        })
    return rows


def plot_driving(path: Path, driving: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.7))
    for label in CASES:
        rows = [row for row in driving if row["condition"] == label]
        finite = [row for row in rows if row["K_J_MPa_sqrt_m"] not in (None, "")]
        axes[0].plot(
            [float(row["stage_extension_um"]) for row in finite],
            [float(row["K_J_MPa_sqrt_m"]) for row in finite],
            color=COLORS[label], linewidth=0.8, alpha=0.85,
            label=label.replace("_", " "),
        )
        primary = [row for row in rows if row["observation_stage"] != "committed"]
        axes[1].plot(
            [float(row["model_time_s"]) for row in primary],
            [float(row["opening_um"]) for row in primary],
            color=COLORS[label], linewidth=1.0,
            linestyle="--" if label == "cold_300K" else "-",
        )
    axes[0].set_ylabel(r"Native $K_J$ (MPa$\sqrt{\mathrm{m}}$)")
    axes[0].set_xlabel(r"Projected extension $\Delta a$ ($\mu$m)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_ylabel(r"Applied opening ($\mu$m)")
    axes[1].set_xlabel("Accepted model time (s)")
    for tag, ax in zip(("a", "b"), axes):
        ax.text(0.02, 0.94, tag, transform=ax.transAxes, va="top", fontweight="bold")
        ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_resistance(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True)
    for ax, label in zip(axes, CASES):
        selected = [row for row in rows if row["condition"] == label]
        for index, row in enumerate(selected):
            ax.plot(
                [row["pre_extension_um"], row["committed_extension_um"]],
                [row["pre_event_K_J_MPa_sqrt_m"]] * 2,
                color=COLORS[label], linewidth=0.8,
            )
        ax.scatter(
            [row["pre_extension_um"] for row in selected],
            [row["pre_event_K_J_MPa_sqrt_m"] for row in selected],
            color=COLORS[label], s=5,
        )
        ax.text(0.04, 0.94, label.replace("_", " "), transform=ax.transAxes,
                va="top", fontsize=9)
        ax.set_xlabel(r"Projected extension $\Delta a$ ($\mu$m)")
        ax.tick_params(direction="in", top=True, right=True)
    axes[0].set_ylabel(r"Event resistance $K_J$ (MPa$\sqrt{\mathrm{m}}$)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def impact_record(label: str, case: Path) -> dict[str, Any]:
    attestation = json.loads((case / "shared_runtime_attestation.post.json").read_text())
    source = attestation.get("source", {})
    producer = PRODUCER_COMMITS[label]
    completed = subprocess.run(
        ["git", "diff", "--name-only", f"{producer}..HEAD", "--", *IMPACT_PATHS],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    changed = [line for line in completed.stdout.splitlines() if line]
    unexpected = [path for path in changed if path != "arrhenius_fracture/sharp_front.py"]
    if unexpected:
        raise ValueError(f"unexpected developed-growth impact paths: {unexpected}")
    return {
        "condition": label,
        "run_root": str(case.relative_to(ROOT)),
        "execution_verified": attestation.get("execution_verified") is True,
        "producer_commit": source.get("commit"),
        "producer_tree": source.get("tree"),
        "producer_dirty_recorded": source.get("dirty"),
        "current_relevant_changed_paths": changed,
        "impact_classification": (
            "compatible_saved evidence: the only relevant committed delta records "
            "accepted crack-path/arclength mirrors for later kernel measurement; "
            "it does not alter the saved theta-zero event or mechanics history"
        ),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    resistance: list[dict[str, Any]] = []
    cases: dict[str, dict[str, Any]] = {}
    impacts = []
    for label, (case, token) in CASES.items():
        history, steps, _audit, _records = history_rows(label, case, token)
        selected = selected_resistance(label, history, steps)
        if len(selected) != 203:
            raise ValueError(f"expected 203 accepted events for {label}, got {len(selected)}")
        resistance.extend(selected)
        impacts.append(impact_record(label, case))
        cases[label] = {
            "temperature_K": 300 if label == "cold_300K" else 1200,
            "accepted_event_count": len(selected),
            "terminal_projected_extension_um": selected[-1]["committed_extension_um"],
            "terminal_path_arclength_um": selected[-1]["committed_path_arclength_um"],
            "initial_resistance_MPa_sqrt_m": selected[0]["pre_event_K_J_MPa_sqrt_m"],
            "terminal_event_resistance_MPa_sqrt_m": selected[-1]["pre_event_K_J_MPa_sqrt_m"],
            "resistance_range_MPa_sqrt_m": [
                min(row["pre_event_K_J_MPa_sqrt_m"] for row in selected),
                max(row["pre_event_K_J_MPa_sqrt_m"] for row in selected),
            ],
            "unique_load_segments": sorted({row["load_segment"] for row in selected}),
            "independent_reload_resistance_points": 0,
            "raw_root": str(case.relative_to(ROOT)),
            "observation_sha256": sha256(case / "fracture_observations_online.jsonl"),
            "geometry_sha256": sha256(case / "stochastic_avalanche_geometry_events.json"),
        }

    source_history = SOURCE_PACKET / "PHYSICAL_HISTORY_HOT_COLD_v4.csv"
    source_queries = SOURCE_PACKET / "LENGTH_QUERY_BRACKETS_HOT_COLD_v4.csv"
    with source_history.open(newline="") as stream:
        driving = list(csv.DictReader(stream))
    with source_queries.open(newline="") as stream:
        queries = list(csv.DictReader(stream))
    if len(driving) != 3695 or len(queries) != 12:
        raise ValueError("developed-growth export has an unexpected row count")
    if any(row["interpolation_performed"] != "False" for row in queries):
        raise ValueError("length-query export unexpectedly interpolated a discrete jump")

    driving_path = OUT / "DEVELOPED_DRIVING_HISTORY.csv"
    query_path = OUT / "DEVELOPED_LENGTH_QUERIES.csv"
    resistance_path = OUT / "DEVELOPED_RESISTANCE_EVENTS.csv"
    write_csv(driving_path, driving)
    write_csv(query_path, queries)
    write_csv(resistance_path, resistance)
    driving_figure = OUT / "DEVELOPED_DRIVING_HISTORY.png"
    resistance_figure = OUT / "DEVELOPED_RESISTANCE.png"
    plot_driving(driving_figure, driving)
    plot_resistance(resistance_figure, resistance)

    same_geometry = cases["cold_300K"]["geometry_sha256"] == cases["hot_1200K"]["geometry_sha256"]
    report = {
        "schema": "final-developed-growth-resistance/1",
        "decision": "COMPLETE_VALIDATED",
        "cases": cases,
        "impact_checks": impacts,
        "same_seed_geometry": {
            "byte_identical_geometry_event_files": same_geometry,
            "interpretation": (
                "controlled common-mark geometry: temperature changes load, time and "
                "mechanism state; this is not independent morphology invariance"
            ),
        },
        "driving_history": {
            "definition": (
                "complete chronological accepted native observation history, retaining "
                "reequilibrated, exact pre-event and committed stages and the joined "
                "source-row state channels"
            ),
            "row_count": len(driving),
            "data": str(driving_path.relative_to(ROOT)),
            "figure_png": str(driving_figure.relative_to(ROOT)),
            "figure_pdf": str(driving_figure.with_suffix(".pdf").relative_to(ROOT)),
        },
        "resistance_view": {
            "definition": (
                "native J-derived K_J measured at each exact pre-event state and "
                "associated only with that event's realized committed projected jump"
            ),
            "event_row_count": len(resistance),
            "data": str(resistance_path.relative_to(ROOT)),
            "figure_png": str(resistance_figure.relative_to(ROOT)),
            "figure_pdf": str(resistance_figure.with_suffix(".pdf").relative_to(ROOT)),
            "scope": (
                "event-sampled monotonic resistance history; the single loading segment "
                "does not provide multiple independent unload/reload resistance points"
            ),
        },
        "length_queries": {
            "targets_um": [100, 500, 1000],
            "data": str(query_path.relative_to(ROOT)),
            "interpolation_performed": False,
            "policy": (
                "uncomputed lengths retain measured lower pre-event and upper committed "
                "jump coordinates; no K value is shifted to the requested coordinate"
            ),
        },
        "selected_native_field_source": {
            "path": str((CASES["hot_1200K"][0] / "field_snapshots_1200K.png").relative_to(ROOT)),
            "stored_states": [1, 1600, 1672, 1703],
            "channels": ["damage", "log10 density", "sigma_1 FEM", "equivalent plastic strain"],
            "note": "original native renderer output; no reconstructed bulk field from tip bins",
        },
        "commands": [
            "python3 scripts/export_physical_growth_packet.py",
            "python3 scripts/analyze_final_developed_growth.py",
        ],
    }
    report_path = OUT / "DEVELOPED_GROWTH_RESISTANCE.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = [
        "# Developed crack growth and resistance", "",
        "**Decision:** `COMPLETE_VALIDATED`", "",
        "The accepted 300 K and 1200 K trajectories each contain 203 exact native pre-event/committed event pairs and reach 1004.926 µm projected extension (1008.560 µm path arclength). The seeded geometry-event files are byte-identical, providing a controlled common-mark geometry comparison rather than an independent morphology test.", "",
        "The resistance view reports J-derived K_J at each exact pre-event state across only its realized committed crack jump. It is an event-sampled monotonic history, not a collection of independent unload/reload resistance points; no rising slope is forced.", "",
        "Queries at 100, 500 and 1000 µm are discrete-jump brackets. The lower pre-event coordinate retains its measured K_J and the upper committed coordinate has no invented post-event K/J; interpolation is false for all 12 bracket rows.", "",
        "Current active mobile-plus-retained content is reported as state, not cumulative emission. Projected extension and path arclength remain separate.", "",
        f"Machine-readable report: `{report_path.relative_to(ROOT)}`", "",
        f"Resistance data: `{resistance_path.relative_to(ROOT)}`", "",
        f"Full driving history: `{driving_path.relative_to(ROOT)}`", "",
    ]
    (OUT / "DEVELOPED_GROWTH_RESISTANCE.md").write_text("\n".join(markdown))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
