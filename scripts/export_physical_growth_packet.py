#!/usr/bin/env python3
"""Export physical-clock PF histories and same-geometry hot/cold comparisons."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_current_science_pilots_v2 import (  # noqa: E402
    OUT, event_pairs, load_case, number,
)

RAW = ROOT / "integration_evidence/current_science_pilots_6abbf10/pf"
CASES = {
    "cold_300K": (RAW / "DBTT_300K/developed_1000um_current", "0300"),
    "hot_1200K": (RAW / "DBTT_1200K/developed_1000um_current_v2_run3", "1200"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(records[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def audit_records(case: Path) -> list[dict]:
    return json.loads((case / "kinetic_tip_cell_audit_v101.json").read_text())["records"]


def history_rows(label: str, case: Path, token: str):
    history, steps = load_case(case, token)
    audit = audit_records(case)
    records = []
    for observation in history.rows:
        step = steps[observation.source_row]
        diagnostic = audit[observation.source_row - 2]
        records.append({
            "condition": label,
            "observation_sequence": observation.sequence,
            "source_row": observation.source_row,
            "accepted_step": int(number(step, "step")),
            "observation_stage": observation.stage.value,
            "event_id": observation.event_id,
            "model_time_s": observation.time_s,
            "opening_um": number(step, "Uapp_m") * 1.0e6,
            "reaction_N": number(step, "Ftop_N"),
            "pre_extension_um": observation.pre_length_m * 1.0e6,
            "committed_extension_um": observation.committed_length_m * 1.0e6,
            "stage_extension_um": observation.stage_extension_m * 1.0e6,
            "path_arclength_um": observation.path_arclength_m * 1.0e6,
            "K_J_MPa_sqrt_m": (
                observation.K_J_Pa_sqrt_m / 1.0e6
                if math.isfinite(observation.K_J_Pa_sqrt_m) else None
            ),
            "K_I_MPa_sqrt_m": (
                observation.K_I_Pa_sqrt_m / 1.0e6
                if math.isfinite(observation.K_I_Pa_sqrt_m) else None
            ),
            "J_effective_J_per_m2": (
                observation.J_J_per_m2
                if math.isfinite(observation.J_J_per_m2) else None
            ),
            "J_signed_J_per_m2": (
                observation.J_signed_J_per_m2
                if math.isfinite(observation.J_signed_J_per_m2) else None
            ),
            "tip_opening_drive_GPa": float(diagnostic["sigma_opening_tip_Pa"]) / 1.0e9,
            "source_backstress_GPa": number(step, "sigma_back_Pa") / 1.0e9,
            "active_K_shield_MPa_sqrt_m": float(
                diagnostic["active_K_shield_signed_Pa_sqrt_m"]
            ) / 1.0e6,
            "wake_K_shield_MPa_sqrt_m": float(
                diagnostic["wake_K_shield_signed_Pa_sqrt_m"]
            ) / 1.0e6,
            "r_eff_um": float(diagnostic["persistent_tip_radius_m"]) * 1.0e6,
            "active_mobile_count": number(step, "mpz_mobile_count"),
            "active_retained_count": number(step, "mpz_retained_count"),
            "active_total_count": number(step, "N_em"),
            "wake_retained_total": number(step, "mpz_wake_retained_total"),
            "cumulative_escaped": number(step, "mpz_escaped_total"),
            "cumulative_recovered": number(step, "mpz_recovered_total"),
            "cleavage_clock_B_resettable": number(step, "B"),
            "hazard_action_current": float(diagnostic["hazard_action_current"]),
            "hazard_threshold_current_action": float(
                diagnostic["hazard_threshold_current_action"]
            ),
            "hazard_event_index": int(diagnostic["hazard_event_index"]),
            "adaptive_fraction": number(step, "adaptive_frac"),
            "accepted_dt_s": number(step, "dt_cur_s"),
            "W_external_cumulative_J_per_m": number(step, "W_ext_cumulative_J_per_m"),
            "U_elastic_J_per_m": number(step, "U_elastic_J_per_m"),
            "W_bulk_plastic_cumulative_J_per_m": number(
                step, "W_bulk_plastic_cumulative_J_per_m"
            ),
            "W_tip_emission_cumulative_J_per_m": number(
                step, "W_tip_emit_cumulative_J_per_m"
            ),
            "W_fracture_residual_cumulative_J_per_m": number(
                step, "W_fracture_residual_cumulative_J_per_m"
            ),
            "joined_state_semantic": (
                "accepted source-row endpoint; K/J retain native observation stage"
            ),
        })
    return history, steps, audit, records


def step_diagnostics(label: str, steps: dict[int, dict[str, str]], audit: list[dict]):
    rows = []
    time_s = 0.0
    previous_extension = 0.0
    for index, source_row in enumerate(sorted(steps)):
        step = steps[source_row]
        time_s += number(step, "dt_cur_s")
        extension = number(step, "crack_extension_m") * 1.0e6
        rows.append({
            "condition": label,
            "source_row": source_row,
            "accepted_step": int(number(step, "step")),
            "model_time_s_from_accepted_dt": time_s,
            "opening_um": number(step, "Uapp_m") * 1.0e6,
            "extension_um": extension,
            "event_committed": extension > previous_extension + 1.0e-12,
            "accepted_dt_s": number(step, "dt_cur_s"),
            "adaptive_fraction": number(step, "adaptive_frac"),
            "cleavage_clock_B_resettable": number(step, "B"),
            "hazard_action_current": float(audit[index]["hazard_action_current"]),
            "hazard_threshold_current_action": float(
                audit[index]["hazard_threshold_current_action"]
            ),
            "hazard_event_index": int(audit[index]["hazard_event_index"]),
        })
        previous_extension = extension
    return rows


def query_rows(label, history, targets=(100.0, 500.0, 1000.0)):
    rows = []
    for target in targets:
        query = history.at_length(target * 1.0e-6)
        for role, observation in (("lower", query.lower), ("upper", query.upper)):
            rows.append({
                "condition": label,
                "target_extension_um": target,
                "bracket_role": role,
                "sequence": observation.sequence,
                "source_row": observation.source_row,
                "stage": observation.stage.value,
                "event_id": observation.event_id,
                "model_time_s": observation.time_s,
                "stage_extension_um": observation.stage_extension_m * 1.0e6,
                "K_J_MPa_sqrt_m": (
                    observation.K_J_Pa_sqrt_m / 1.0e6
                    if math.isfinite(observation.K_J_Pa_sqrt_m) else None
                ),
                "interpolation_performed": False,
            })
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = {}
    all_history, all_steps, all_queries = [], [], []
    for label, (case, token) in CASES.items():
        history, steps, audit, records = history_rows(label, case, token)
        diagnostics = step_diagnostics(label, steps, audit)
        queries = query_rows(label, history)
        loaded[label] = (case, history, steps, records, diagnostics, queries)
        all_history.extend(records)
        all_steps.extend(diagnostics)
        all_queries.extend(queries)
    write_csv(OUT / "PHYSICAL_HISTORY_HOT_COLD_v4.csv", all_history)
    write_csv(OUT / "ACCEPTED_STEP_DIAGNOSTICS_HOT_COLD_v4.csv", all_steps)
    write_csv(OUT / "LENGTH_QUERY_BRACKETS_HOT_COLD_v4.csv", all_queries)

    cold_pairs = event_pairs(loaded["cold_300K"][1])
    hot_pairs = event_pairs(loaded["hot_1200K"][1])
    geometry_files = {
        label: case / "stochastic_avalanche_geometry_events.json"
        for label, (case, *_rest) in loaded.items()
    }
    geometry_sha = {label: sha256(path) for label, path in geometry_files.items()}
    comparisons = []
    for event_index, ((cold_pre, cold_commit), (hot_pre, hot_commit)) in enumerate(
        zip(cold_pairs, hot_pairs), 1
    ):
        comparisons.append({
            "event_number": event_index,
            "committed_extension_um": hot_commit.committed_length_m * 1.0e6,
            "event_advance_um": (
                hot_commit.committed_length_m - hot_pre.pre_length_m
            ) * 1.0e6,
            "cold_pre_time_s": cold_pre.time_s,
            "hot_pre_time_s": hot_pre.time_s,
            "cold_pre_K_J_MPa_sqrt_m": cold_pre.K_J_Pa_sqrt_m / 1.0e6,
            "hot_pre_K_J_MPa_sqrt_m": hot_pre.K_J_Pa_sqrt_m / 1.0e6,
            "same_pre_extension": cold_pre.pre_length_m == hot_pre.pre_length_m,
            "same_committed_extension": (
                cold_commit.committed_length_m == hot_commit.committed_length_m
            ),
        })
    write_csv(OUT / "HOT_COLD_SAME_GEOMETRY_EVENTS_v4.csv", comparisons)

    colors = {"cold_300K": "#2166ac", "hot_1200K": "#b2182b"}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), constrained_layout=True)
    for label, (_case, _history, _steps, _records, diagnostics, _queries) in loaded.items():
        axes[0].plot([r["model_time_s_from_accepted_dt"] for r in diagnostics],
                     [r["opening_um"] for r in diagnostics], color=colors[label],
                     label=label.replace("_", " "))
        axes[1].step([r["model_time_s_from_accepted_dt"] for r in diagnostics],
                     [r["extension_um"] for r in diagnostics], where="post",
                     color=colors[label])
    axes[0].set_ylabel("Opening (µm)"); axes[0].legend(frameon=False)
    axes[1].set_ylabel("Projected extension (µm)"); axes[1].set_xlabel("Model time (s)")
    for tag, ax in zip("ab", axes): ax.text(0.01, 0.95, tag, transform=ax.transAxes, va="top", fontweight="bold")
    fig.savefig(OUT / "physical_time_growth_hot_cold_v4.png", dpi=240)
    plt.close(fig)

    hot_events = [(pre.time_s, commit.committed_length_m * 1.0e6)
                  for pre, commit in hot_pairs]
    t0 = next(pre.time_s for pre, _commit in hot_pairs
              if pre.pre_length_m * 1.0e6 >= 99.0)
    zoom = [(t, extension) for t, extension in hot_events if t >= t0]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.step([(t - t0) * 1.0e6 for t, _ in zoom], [a for _, a in zoom], where="post", color=colors["hot_1200K"])
    ax.set_xlabel("Model time after 99.483 µm pre-event state (µs)")
    ax.set_ylabel("Committed projected extension (µm)")
    ax.text(0.01, 0.95, "Event-localized adaptive sequence; not a measured crack velocity", transform=ax.transAxes, va="top", fontsize=9)
    fig.savefig(OUT / "hot_growth_time_zoom_v4.png", dpi=240)
    plt.close(fig)

    hot_diag = loaded["hot_1200K"][4]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True, constrained_layout=True)
    steps_x = [r["accepted_step"] for r in hot_diag]
    axes[0].semilogy(steps_x, [r["accepted_dt_s"] for r in hot_diag], label="accepted dt")
    axes[0].semilogy(steps_x, [r["adaptive_fraction"] for r in hot_diag], label="adaptive fraction")
    axes[0].legend(frameon=False); axes[0].set_ylabel("Controller scale")
    axes[1].plot(steps_x, [r["cleavage_clock_B_resettable"] for r in hot_diag], color="#4d9221")
    fired = [r for r in hot_diag if r["event_committed"]]
    axes[1].scatter([r["accepted_step"] for r in fired], [r["cleavage_clock_B_resettable"] for r in fired], s=7, color="#762a83", label="accepted event")
    axes[1].set_ylabel("Resettable clock B"); axes[1].set_xlabel("Accepted step")
    axes[1].legend(frameon=False)
    fig.savefig(OUT / "hot_accepted_step_diagnostics_v4.png", dpi=240)
    plt.close(fig)

    hot_q = {(r["target_extension_um"], r["bracket_role"]): r
             for r in loaded["hot_1200K"][5]}
    first = hot_q[(100.0, "lower")]
    last = hot_q[(1000.0, "lower")]
    payload = {
        "schema": "physical-growth-comparison/4",
        "model_time_semantic": "accepted loading/controller clock from native observation history",
        "velocity_claimed": False,
        "hot_100_to_1000_lower_brackets": {
            "extension_change_um": last["stage_extension_um"] - first["stage_extension_um"],
            "elapsed_model_time_s": last["model_time_s"] - first["model_time_s"],
            "interpretation": "event-localized adaptive model sequence, not physical crack velocity",
        },
        "controller": {
            "prescribed_hold_detected": False,
            "opening_continued_to_increase": True,
            "minimum_adaptive_fraction": min(r["adaptive_fraction"] for r in hot_diag),
            "B_semantic": "resettable within-event progress clock; not cumulative hazard history",
            "B_decrease_count": sum(
                hot_diag[i]["cleavage_clock_B_resettable"]
                < hot_diag[i-1]["cleavage_clock_B_resettable"]
                for i in range(1, len(hot_diag))
            ),
        },
        "same_geometry_comparison": {
            "event_count_each": len(comparisons),
            "geometry_event_files_sha256": geometry_sha,
            "geometry_event_files_byte_identical": len(set(geometry_sha.values())) == 1,
            "all_event_brackets_identical": all(
                r["same_pre_extension"] and r["same_committed_extension"]
                for r in comparisons
            ),
            "interpretation": "temperature changes the time/load/K history, while the seeded event-length and mark sequence fixes the common final geometry",
        },
        "files": {
            "history": "PHYSICAL_HISTORY_HOT_COLD_v4.csv",
            "accepted_steps": "ACCEPTED_STEP_DIAGNOSTICS_HOT_COLD_v4.csv",
            "queries": "LENGTH_QUERY_BRACKETS_HOT_COLD_v4.csv",
            "event_comparison": "HOT_COLD_SAME_GEOMETRY_EVENTS_v4.csv",
        },
    }
    (OUT / "PHYSICAL_GROWTH_COMPARISON_v4.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
