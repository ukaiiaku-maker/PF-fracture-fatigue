#!/usr/bin/env python3
"""Derive scientific comparisons and restart inventory from the 12 h archive."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.coupled_opening_reduction_r10 import score_recurrence


def load(path: Path):
    return json.loads(path.read_text())


def write(path: Path, rows) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def effective_result(path: Path):
    result = load(path)
    progress_path = path.with_name("progress.json")
    if progress_path.is_file():
        progress = load(progress_path)
        if float(progress.get("accepted_cycles", 0.)) > float(result.get("accepted_cycles", 0.)):
            result = {**result, **progress, "result_disposition": result.get("disposition")}
            result["accepted_seconds"] = (float(result["accepted_cycles"]) /
                                          float(result["frequency_Hz"]))
            scored = score_recurrence(
                result.get("events", []), float(result.get("observation_path_m", 120e-6)),
                bool(result.get("unsupported_validities", [])))
            result["developed_rate_m_per_cycle"] = scored["developed_rate_m_per_cycle"]
            result["developed_window"] = scored["developed_window"]
    return result


def event_metrics(path: Path, result: dict):
    events = result.get("events", [])
    cycles = [float(row["cycle"]) for row in events]
    gaps = [b - a for a, b in zip(cycles, cycles[1:]) if b > a]
    window = result.get("developed_window") or {}
    post_escape_unresolved = (not events and any(
        row.get("branch_state") == "POST_ESCAPE_TRANSIT"
        for row in result.get("history", [])))
    return {
        "candidate_id": result.get("candidate_id"),
        "temperature_K": result.get("temperature_K"),
        "R": result.get("R"),
        "frequency_Hz": result.get("frequency_Hz"),
        "Kmax_MPa_sqrt_m": result.get("Kmax_MPa_sqrt_m"),
        "DeltaK_MPa_sqrt_m": result.get("DeltaK_MPa_sqrt_m"),
        "reference_factor": result.get("reference_factor"),
        "accepted_cycles": result.get("accepted_cycles"),
        "accepted_seconds": result.get("accepted_seconds"),
        "events": len(events),
        "first_event_cycle": cycles[0] if cycles else None,
        "positive_interevent_count": len(gaps),
        "median_positive_interevent_cycles": median(gaps) if gaps else None,
        "minimum_positive_interevent_cycles": min(gaps) if gaps else None,
        "maximum_positive_interevent_cycles": max(gaps) if gaps else None,
        "developed_rate_m_per_cycle": result.get("developed_rate_m_per_cycle"),
        "developed_window_duration_cycles": window.get("duration_cycles"),
        "disposition": ("UNRESOLVED_OPENING_PHYSICS" if post_escape_unresolved
                        else result.get("disposition")),
        "plastic_feedback_reference": result.get("plastic_feedback_reference", "EVOLVING_FULL_STATE"),
        "result_path": str(path),
    }


def slope_rows(rows):
    groups = {}
    for row in rows:
        rate = row.get("developed_rate_m_per_cycle")
        if rate in (None, "", 0, 0.):
            continue
        key = (row["candidate_id"], row["temperature_K"], row["R"],
               row["frequency_Hz"], row["plastic_feedback_reference"])
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in groups.items():
        group.sort(key=lambda row: float(row["DeltaK_MPa_sqrt_m"]))
        for low, high in zip(group, group[1:]):
            x0, x1 = float(low["DeltaK_MPa_sqrt_m"]), float(high["DeltaK_MPa_sqrt_m"])
            y0 = float(low["developed_rate_m_per_cycle"])
            y1 = float(high["developed_rate_m_per_cycle"])
            output.append({
                "candidate_id": key[0], "temperature_K": key[1], "R": key[2],
                "frequency_Hz": key[3], "plastic_feedback_reference": key[4],
                "DeltaK_low_MPa_sqrt_m": x0, "DeltaK_high_MPa_sqrt_m": x1,
                "rate_low_m_per_cycle": y0, "rate_high_m_per_cycle": y1,
                "local_log_slope": math.log(y1 / y0) / math.log(x1 / x0),
                "low_result_path": low["result_path"], "high_result_path": high["result_path"],
            })
    return output


def thermal_rows(campaign: Path):
    output = []
    for summary in sorted(campaign.glob("thermal_ramps_*/ramp_dispositions.csv")):
        with summary.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_candidate = {}
        for row in rows:
            by_candidate.setdefault(row["candidate_id"], []).append(row)
        for candidate, group in by_candidate.items():
            roots = [float(row["first_committed_separation_K_MPa_sqrt_m"])
                     for row in group if row.get("first_committed_separation_K_MPa_sqrt_m")]
            censored = [float(row["temperature_K"]) for row in group
                        if row.get("status") == "KMAX_CENSOR"]
            complete = len(roots) == len(group) == 7
            output.append({
                "candidate_id": candidate,
                "source_summary": str(summary),
                "temperature_points": len(group),
                "resolved_points": len(roots),
                "censored_temperatures_K": ";".join(f"{x:g}" for x in censored),
                "minimum_root_MPa_sqrt_m": min(roots) if roots else None,
                "maximum_root_MPa_sqrt_m": max(roots) if roots else None,
                "resolved_span_ratio": max(roots) / min(roots) if roots else None,
                "all_seven_resolved": complete,
                "bounded_weak_T_span_pass": complete and max(roots) / min(roots) <= 1.25,
            })
    return output


def mechanism_comparison(campaign: Path):
    full_path = campaign / "hcf_round_1_block_2/R10F_V2_22871__MS/T_300__R_0.1__Hz_1000/f_1.0100/result.json"
    frozen_path = campaign / "hcf_mechanism_controls_22871/frozen/result.json"
    if not full_path.is_file() or not frozen_path.is_file():
        return []
    full, frozen = effective_result(full_path), effective_result(frozen_path)
    rows = []
    for index, (a, b) in enumerate(zip(full.get("events", []), frozen.get("events", [])), 1):
        af, bf = float(a["cycle"]), float(b["cycle"])
        rows.append({
            "event_index": index, "full_state_cycle": af, "frozen_state_cycle": bf,
            "absolute_difference_cycles": af - bf,
            "relative_difference": (af - bf) / bf if bf else None,
        })
    return rows


def restart_rows(campaign: Path):
    rows = []
    for progress_path in campaign.rglob("progress.json"):
        if "eventful_adapter_qualification" in progress_path.parts:
            continue
        checkpoint = progress_path.parent / "checkpoint/high_cycle_live_checkpoint.json"
        if not checkpoint.is_file():
            continue
        progress = load(progress_path)
        result_path = progress_path.with_name("result.json")
        result = load(result_path) if result_path.is_file() else {}
        if (result.get("disposition") not in {"RESOURCE_STOP", None} and
                float(result.get("accepted_cycles", 0.)) >= float(progress.get("accepted_cycles", 0.))):
            continue
        rows.append({
            "candidate_id": result.get("candidate_id"),
            "accepted_cycles": progress.get("accepted_cycles"),
            "events": len(progress.get("events", [])),
            "declared_horizon_cycles": result.get("exposure_target_cycles"),
            "last_result_disposition": result.get("disposition"),
            "checkpoint_path": str(checkpoint),
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    items = []
    native_items = []
    for path in args.campaign.rglob("result.json"):
        result = load(path)
        if result.get("schema") == "r10.czm-owned-hcf-trajectory/1" and "eventful_adapter_qualification" not in path.parts:
            items.append((path, effective_result(path)))
        elif "events" in result and "cycle_history" in result:
            scored = score_recurrence(
                result.get("events", []),
                float(result.get("cohesive_path_observation_limit_m", 120e-6)),
                bool(result.get("unsupported_crossover_encountered", False)))
            history = result.get("cycle_history", [])
            native_items.append((path, {
                **result,
                "accepted_cycles": len(history),
                "accepted_seconds": history[-1].get("physical_time_s") if history else 0.,
                "developed_rate_m_per_cycle": scored["developed_rate_m_per_cycle"],
                "developed_window": scored["developed_window"],
                "disposition": result.get("stop_category"),
                "plastic_feedback_reference": "NATIVE_FULL_STATE",
            }))
    events = [event_metrics(path, result) for path, result in items]
    native_events = [event_metrics(path, result) for path, result in native_items]
    write(args.out / "event_waiting_and_windows.csv", events)
    write(args.out / "local_developed_slopes.csv", slope_rows(events + native_events))
    write(args.out / "thermal_classification.csv", thermal_rows(args.campaign))
    write(args.out / "matched_22871_event_comparison.csv", mechanism_comparison(args.campaign))
    write(args.out / "restart_inventory.csv", restart_rows(args.campaign))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
