#!/usr/bin/env python3
"""Versioned, stage-faithful analysis of the current-source PF pilots.

This exporter consumes the online chronological observation history as the
authority for stage, event identity, geometry, and K/J semantics.  The steps
table is joined only by the observation's recorded ``source_row`` to attach
non-observer state channels; it is never used to infer fracture events.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.fracture_observation_history import (
    FractureObservation,
    ObservationHistory,
    ObservationStage,
    with_geometry_event_arclength,
)


RAW = ROOT / "integration_evidence" / "current_science_pilots_6abbf10"
OUT = ROOT / "review_packet" / "current_science_pilots_6abbf10"

CASES = (
    ("DBTT 300 K", "DBTT_300K", "0300", "accommodation_enabled_run2"),
    ("DBTT 1200 K", "DBTT_1200K", "1200", "accommodation_enabled"),
    ("Peak 900 K", "Peak_900K", "0900", "accommodation_enabled"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_steps(path: Path) -> dict[int, dict[str, str]]:
    """Index states by physical CSV row number (header is row one)."""
    with path.open(newline="") as stream:
        return {row_number: row for row_number, row in enumerate(csv.DictReader(stream), 2)}


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return math.nan if value in (None, "") else float(value)


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def kj_record(row: FractureObservation) -> dict[str, Any]:
    """Export native K channels without promoting K_J to modal K_I."""
    return {
        "K_J_MPa_sqrt_m": finite_or_none(row.K_J_Pa_sqrt_m / 1.0e6),
        "K_I_MPa_sqrt_m": finite_or_none(row.K_I_Pa_sqrt_m / 1.0e6),
        "K_II_MPa_sqrt_m": finite_or_none(row.K_II_Pa_sqrt_m / 1.0e6),
        "K_effective_MPa_sqrt_m": finite_or_none(
            row.K_effective_Pa_sqrt_m / 1.0e6
        ),
        "K_semantic": (
            "J-derived equivalent stress-intensity K_J"
            if math.isfinite(row.K_J_Pa_sqrt_m)
            else "not observed at this stage"
        ),
    }


def state_record(row: FractureObservation, step: dict[str, str]) -> dict[str, Any]:
    total = number(step, "N_em")
    mobile = number(step, "mpz_mobile_count")
    retained = number(step, "mpz_retained_count")
    return {
        "sequence": row.sequence,
        "source_row": row.source_row,
        "source_step": int(round(number(step, "step"))),
        "segment_id": row.segment_id,
        "event_id": row.event_id,
        "stage": row.stage.value,
        "validity_status": row.validity_status.value,
        "validity_reason": row.validity_reason,
        "pre_extension_um": row.pre_length_m * 1.0e6,
        "committed_extension_um": row.committed_length_m * 1.0e6,
        "stage_extension_um": row.stage_extension_m * 1.0e6,
        "path_arclength_um": finite_or_none(row.path_arclength_m * 1.0e6),
        "opening_um": number(step, "Uapp_m") * 1.0e6,
        **kj_record(row),
        "J_effective_J_per_m2": finite_or_none(row.J_J_per_m2),
        "J_signed_J_per_m2": finite_or_none(row.J_signed_J_per_m2),
        "cleavage_clock_B": number(step, "B"),
        "active_mobile_plus_retained_count": total,
        "active_mobile_count": mobile,
        "active_retained_count": retained,
        "active_population_closure_error": total - mobile - retained,
        "persistent_source_backstress_GPa": number(step, "sigma_back_Pa") / 1.0e9,
        "tip_opening_stress_GPa": number(step, "sigma_tip_Pa") / 1.0e9,
        "emission_hazard_s": number(step, "lambda_e"),
        "cleavage_hazard_s": number(step, "lambda_c"),
        "cumulative_tip_emission_work_J_per_m": number(
            step, "W_tip_emit_cumulative_J_per_m"
        ),
    }


def load_case(case: Path, token: str) -> tuple[ObservationHistory, dict[int, dict[str, str]]]:
    history_path = case / "fracture_observations_online.jsonl"
    geometry_path = case / "stochastic_avalanche_geometry_events.json"
    history = with_geometry_event_arclength(
        ObservationHistory.load(history_path), load_json(geometry_path)
    )
    steps = load_steps(case / f"steps_{token}K.csv")
    missing = sorted({row.source_row for row in history.rows} - set(steps))
    if missing:
        raise RuntimeError(f"observation rows do not resolve in steps table: {missing[:5]}")
    return history, steps


def event_pairs(history: ObservationHistory) -> list[tuple[FractureObservation, FractureObservation]]:
    result = []
    for left, right in zip(history.rows, history.rows[1:]):
        if (
            left.stage is ObservationStage.PRE_EVENT
            and right.stage is ObservationStage.COMMITTED
            and left.event_id == right.event_id
            and left.source_row == right.source_row
        ):
            result.append((left, right))
    return result


def stage_query(history: ObservationHistory, target_um: float) -> dict[str, Any]:
    query = history.at_length(target_um * 1.0e-6)
    payload = query.to_dict()
    payload["target_length_um"] = target_um
    payload["measured_crossing"] = [
        {
            "sequence": row.sequence,
            "stage": row.stage.value,
            "event_id": row.event_id,
            "pre_extension_um": row.pre_length_m * 1.0e6,
            "committed_extension_um": row.committed_length_m * 1.0e6,
            **kj_record(row),
        }
        for row in query.crossing
    ]
    payload["measured_lower"] = None if query.lower is None else {
        "sequence": query.lower.sequence,
        "stage": query.lower.stage.value,
        "stage_extension_um": query.lower.stage_extension_m * 1.0e6,
        **kj_record(query.lower),
    }
    payload["measured_upper"] = None if query.upper is None else {
        "sequence": query.upper.sequence,
        "stage": query.upper.stage.value,
        "stage_extension_um": query.upper.stage_extension_m * 1.0e6,
        **kj_record(query.upper),
    }
    return payload


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def primary_state_rows(history: ObservationHistory) -> tuple[FractureObservation, ...]:
    """One state-bearing row per native source row, with committed events retained."""
    selected: dict[int, FractureObservation] = {}
    for row in history.rows:
        if row.stage is ObservationStage.PRE_EVENT:
            continue
        selected[row.source_row] = row
    return tuple(selected[key] for key in sorted(selected))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pair_rows: list[dict[str, Any]] = []
    arm_details: list[dict[str, Any]] = []

    for label, directory, token, enabled_name in CASES:
        arms: dict[str, dict[str, Any]] = {}
        for arm, run_name in (
            ("accommodation_enabled", enabled_name),
            ("opening_only", "opening_only"),
        ):
            case = RAW / "pf" / directory / run_name
            history, steps = load_case(case, token)
            pairs = event_pairs(history)
            first_pre, first_commit = pairs[0]
            final_pre, final_commit = pairs[-1]
            first_state = state_record(first_commit, steps[first_commit.source_row])
            final_state = state_record(final_commit, steps[final_commit.source_row])
            detail = {
                "condition": label,
                "arm": arm,
                "raw_root": str(case.relative_to(ROOT)),
                "observation_source": str(
                    (case / "fracture_observations_online.jsonl").relative_to(ROOT)
                ),
                "accepted_event_count": len(pairs),
                "first_event_pre_observation": {
                    "sequence": first_pre.sequence,
                    "source_row": first_pre.source_row,
                    "event_id": first_pre.event_id,
                    "stage": first_pre.stage.value,
                    "extension_um": first_pre.stage_extension_m * 1.0e6,
                    "opening_um": number(steps[first_pre.source_row], "Uapp_m") * 1.0e6,
                    **kj_record(first_pre),
                },
                "first_event_committed_state": first_state,
                "terminal_event_pre_observation": {
                    "sequence": final_pre.sequence,
                    "source_row": final_pre.source_row,
                    "event_id": final_pre.event_id,
                    "stage": final_pre.stage.value,
                    "extension_um": final_pre.stage_extension_m * 1.0e6,
                    "opening_um": number(steps[final_pre.source_row], "Uapp_m") * 1.0e6,
                    **kj_record(final_pre),
                },
                "terminal_event_committed_state": final_state,
            }
            arms[arm] = detail
            arm_details.append(detail)

        enabled = arms["accommodation_enabled"]
        control = arms["opening_only"]
        ep = enabled["first_event_pre_observation"]
        cp = control["first_event_pre_observation"]
        et = enabled["terminal_event_pre_observation"]
        ct = control["terminal_event_pre_observation"]
        pair_rows.append({
            "condition": label,
            "first_pre_event_K_J_enabled_MPa_sqrt_m": ep["K_J_MPa_sqrt_m"],
            "first_pre_event_K_J_opening_only_MPa_sqrt_m": cp["K_J_MPa_sqrt_m"],
            "first_pre_event_delta_K_J_MPa_sqrt_m": ep["K_J_MPa_sqrt_m"] - cp["K_J_MPa_sqrt_m"],
            "terminal_pre_event_K_J_enabled_MPa_sqrt_m": et["K_J_MPa_sqrt_m"],
            "terminal_pre_event_K_J_opening_only_MPa_sqrt_m": ct["K_J_MPa_sqrt_m"],
            "enabled_first_committed_active_mobile_plus_retained_count": enabled["first_event_committed_state"]["active_mobile_plus_retained_count"],
            "enabled_first_committed_persistent_source_backstress_GPa": enabled["first_event_committed_state"]["persistent_source_backstress_GPa"],
            "committed_extension_um": enabled["first_event_committed_state"]["committed_extension_um"],
            "events_each_arm": enabled["accepted_event_count"],
        })

    write_csv(OUT / "OPENING_ACCOMMODATION_RESULTS_v2.csv", pair_rows)

    long_case = RAW / "pf" / "DBTT_300K" / "developed_1000um_current"
    long_history, long_steps = load_case(long_case, "0300")
    long_pairs = event_pairs(long_history)
    queries = {str(value): stage_query(long_history, value) for value in (100.0, 500.0, 1000.0)}
    (OUT / "DEVELOPED_LENGTH_QUERIES_v2.json").write_text(
        json.dumps({
            "schema": "developed-length-queries/2",
            "coordinate": "projected extension",
            "interpolation_performed": False,
            "queries_um": queries,
        }, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    mechanism_rows = [
        state_record(row, long_steps[row.source_row])
        for row in primary_state_rows(long_history)
    ]
    write_csv(OUT / "DBTT_300K_MECHANISM_HISTORY_v2.csv", mechanism_rows)

    mapping = {
        "schema": "current-science-pilot-export-mapping/2",
        "stage_authority": "fracture_observations_online.jsonl; no steps-based event inference",
        "geometry_authority": "stochastic_avalanche_geometry_events.json joined chronologically to committed native observation rows",
        "state_join": "steps CSV is joined only through native observation source_row",
        "KJ_Pa_sqrtm": "K_J_Pa_sqrt_m: J-derived equivalent stress intensity; never relabeled K_I",
        "K_I_Pa_sqrt_m": "absent/NaN in these PF pilot observations; no modal K_I claim",
        "N_em": "active_mobile_plus_retained_count; current active process-zone state, not cumulative emission",
        "mpz_mobile_count": "active_mobile_count",
        "mpz_retained_count": "active_retained_count",
        "sigma_back_Pa": "persistent source-law Taylor backstress (mean unsigned mobile-plus-retained density channel)",
        "sigma_tip_Pa": "local crack-tip opening stress after shielding/blunting, not source backstress",
        "committed_K_policy": "absent: committed geometry has no invented post-event K/J; next native solve is a distinct reequilibrated row",
        "validity_policy": "unknown where the online observer is finite but no independent contour-validity certificate was emitted",
    }
    (OUT / "OBSERVATION_SCHEMA_MAPPING_v2.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n"
    )

    result = {
        "schema": "fracture-current-source-pilot-results/2",
        "supersedes_for_interpretation": [
            "PILOT_RESULTS.json",
            "OPENING_ACCOMMODATION_RESULTS.csv",
            "current_pf_1mm_K_history.png",
            "opening_accommodation_first_passage.png",
        ],
        "historical_files_modified": False,
        "K_quantity": "K_J (J-derived equivalent stress intensity)",
        "paired_results": pair_rows,
        "arm_details": arm_details,
        "developed_DBTT_300K": {
            "observation_rows": len(long_history.rows),
            "accepted_events": len(long_pairs),
            "terminal_stage": long_history.rows[-1].stage.value,
            "terminal_projected_extension_um": long_history.rows[-1].stage_extension_m * 1.0e6,
            "terminal_path_arclength_um": long_history.rows[-1].path_arclength_m * 1.0e6,
            "terminal_K_J_MPa_sqrt_m": finite_or_none(long_history.rows[-1].K_J_Pa_sqrt_m / 1.0e6),
            "queries": "DEVELOPED_LENGTH_QUERIES_v2.json",
            "mechanism_history": "DBTT_300K_MECHANISM_HISTORY_v2.csv",
        },
        "provenance": {
            "long_observation_sha256": sha256(long_case / "fracture_observations_online.jsonl"),
            "long_geometry_events_sha256": sha256(long_case / "stochastic_avalanche_geometry_events.json"),
            "long_steps_sha256": sha256(long_case / "steps_0300K.csv"),
        },
    }
    (OUT / "PILOT_RESULTS_v2.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    colors = {"accommodation_enabled": "#b24c00", "opening_only": "#087f5b"}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)
    x = range(len(pair_rows))
    width = 0.36
    for offset, arm, label in (
        (-width / 2, "accommodation_enabled", "Accommodation enabled"),
        (width / 2, "opening_only", "Opening only"),
    ):
        key1 = f"first_pre_event_K_J_{'enabled' if arm == 'accommodation_enabled' else 'opening_only'}_MPa_sqrt_m"
        key2 = f"terminal_pre_event_K_J_{'enabled' if arm == 'accommodation_enabled' else 'opening_only'}_MPa_sqrt_m"
        axes[0].bar([i + offset for i in x], [r[key1] for r in pair_rows], width, color=colors[arm], label=label)
        axes[1].bar([i + offset for i in x], [r[key2] for r in pair_rows], width, color=colors[arm], label=label)
    for label, ax in zip(("a", "b"), axes):
        ax.text(0.02, 0.96, label, transform=ax.transAxes, va="top", fontweight="bold")
        ax.set_xticks(list(x), [r["condition"].replace(" ", "\n", 1) for r in pair_rows])
        ax.set_ylabel(r"Pre-event $K_J$ (MPa$\sqrt{m}$)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "opening_accommodation_first_passage_v2.png", dpi=240)
    plt.close(fig)

    pre_rows = [left for left, _ in long_pairs]
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    ax.plot([r.stage_extension_m * 1.0e6 for r in pre_rows],
            [r.K_J_Pa_sqrt_m / 1.0e6 for r in pre_rows], color="#374c80", lw=1.4)
    for target in (100.0, 500.0, 1000.0):
        q = long_history.at_length(target * 1.0e-6)
        lower = q.lower
        crossing = q.crossing[0]
        ax.scatter(lower.stage_extension_m * 1.0e6, lower.K_J_Pa_sqrt_m / 1.0e6,
                   s=25, color="#c43c2b", zorder=3)
        ax.hlines(lower.K_J_Pa_sqrt_m / 1.0e6,
                  crossing.pre_length_m * 1.0e6,
                  crossing.committed_length_m * 1.0e6,
                  color="#c43c2b", lw=2.2, alpha=0.65)
    ax.text(0.02, 0.96, "a", transform=ax.transAxes, va="top", fontweight="bold")
    ax.set_xlabel("Projected crack extension (µm)")
    ax.set_ylabel(r"Pre-event $K_J$ (MPa$\sqrt{m}$)")
    fig.savefig(OUT / "current_pf_1mm_K_history_v2.png", dpi=240)
    plt.close(fig)

    state_x = [row["source_step"] for row in mechanism_rows]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.8), constrained_layout=True)
    axes[0, 0].plot(state_x, [row["opening_um"] for row in mechanism_rows], color="#3b528b")
    axes[0, 0].set_ylabel("Opening (µm)")
    axes[0, 1].plot(state_x, [row["cleavage_clock_B"] for row in mechanism_rows], color="#5ec962")
    axes[0, 1].set_ylabel("Cleavage clock, B")
    axes[1, 0].plot(state_x, [row["active_mobile_count"] for row in mechanism_rows], label="Mobile", color="#21918c")
    axes[1, 0].plot(state_x, [row["active_retained_count"] for row in mechanism_rows], label="Retained", color="#440154")
    axes[1, 0].set_ylabel("Active line-content count")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].plot(state_x, [row["persistent_source_backstress_GPa"] for row in mechanism_rows], color="#d1495b")
    axes[1, 1].set_ylabel("Source backstress (GPa)")
    for label, ax in zip("abcd", axes.flat):
        ax.text(0.02, 0.94, label, transform=ax.transAxes, va="top", fontweight="bold")
        ax.set_xlabel("Accepted source step")
    fig.savefig(OUT / "DBTT_300K_mechanism_history_v2.png", dpi=240)
    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
