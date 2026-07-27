#!/usr/bin/env python3
"""Identify DBTT/peak fatigue cases with meaningful coupled hazard transients."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "v10.2.29_coupled_transient_screen_v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def cyclic_records(root: Path) -> list[dict]:
    audit = root / "kinetic_tip_cell_audit_v101.json"
    if not audit.is_file():
        return []
    payload = load_json(audit)
    return [
        dict(record)
        for record in payload.get("records", [])
        if str(record.get("loading_mode", "")) == "cyclic"
    ]


def first_last(records: list[dict], key: str) -> tuple[float, float]:
    if not records:
        return 0.0, 0.0
    return finite(records[0].get(key, 0.0)), finite(records[-1].get(key, 0.0))


def summarize_case(control_path: Path) -> dict | None:
    root = control_path.parent
    records = cyclic_records(root)
    if not records:
        return None
    control = load_json(control_path)

    cumulative = 0.0
    event_cycle: float | None = None
    event_count = 0
    max_log_span = 0.0
    max_state_ratio = 0.0
    transient_cycles = 0.0
    stationary_cycles = 0.0
    coupled_segments = 0
    rejected_splits = 0
    lambda_min = math.inf
    lambda_max = 0.0

    for record in records:
        consumed = max(finite(record.get("cycles_consumed", 0.0)), 0.0)
        cumulative += consumed
        if bool(record.get("fired", False)):
            event_count += 1
            if event_cycle is None:
                event_cycle = cumulative
        max_log_span = max(
            max_log_span,
            max(finite(record.get("coupled_hazard_log_lambda_span_decades", 0.0)), 0.0),
        )
        transient_cycles += max(
            finite(record.get("coupled_hazard_transient_cycles", 0.0)), 0.0
        )
        stationary_cycles += max(
            finite(record.get("coupled_hazard_stationary_tail_cycles", 0.0)), 0.0
        )
        coupled_segments += int(record.get("coupled_hazard_accepted_segments", 0) or 0)
        rejected_splits += int(record.get("coupled_hazard_rejected_splits", 0) or 0)
        lo = max(finite(record.get("coupled_hazard_lambda_min_s", 0.0)), 0.0)
        hi = max(finite(record.get("coupled_hazard_lambda_max_s", 0.0)), 0.0)
        if lo > 0.0:
            lambda_min = min(lambda_min, lo)
        lambda_max = max(lambda_max, hi)
        for segment in record.get("coupled_hazard_segments", []) or []:
            max_state_ratio = max(
                max_state_ratio,
                max(finite(segment.get("state_target_ratio", 0.0)), 0.0),
            )

    if not math.isfinite(lambda_min):
        lambda_min = 0.0
    if lambda_max > 0.0 and lambda_min > 0.0:
        global_log_span: float | None = math.log10(lambda_max / lambda_min)
    elif lambda_max > 0.0:
        # A zero-to-positive transition is physically an unbounded logarithmic span.
        # Store a null JSON value and use infinity only for candidate classification.
        global_log_span = None
    else:
        global_log_span = 0.0

    sigma_back_initial, sigma_back_final = first_last(
        records, "persistent_sigma_back_Pa"
    )
    mobile_initial, mobile_final = first_last(records, "state_mobile_count")
    retained_initial, retained_final = first_last(records, "state_retained_count")
    emitted_initial, emitted_final = first_last(records, "state_emitted_total")
    shield_initial, shield_final = first_last(
        records, "state_active_K_shield_signed_Pa_sqrt_m"
    )
    B_initial, B_final = first_last(records, "B")

    return {
        "run_root": str(root),
        "parameter_option": control.get("parameter_option"),
        "temperature_K": finite(records[0].get("temperature_K", 0.0)),
        "target_deltaK_MPa_sqrt_m": finite(
            control.get("target_deltaK_MPa_sqrt_m", 0.0)
        ),
        "target_Kmax_MPa_sqrt_m": finite(control.get("target_Kmax_MPa_sqrt_m", 0.0)),
        "R": finite(control.get("R", 0.0)),
        "cycles_horizon": finite(control.get("cycles_max", 0.0)),
        "cycles_consumed": cumulative,
        "record_count": len(records),
        "event_count": event_count,
        "first_event_cycle": event_cycle,
        "right_censored": event_count == 0,
        "maximum_record_log_lambda_span_decades": max_log_span,
        "global_log_lambda_span_decades": global_log_span,
        "lambda_started_from_zero": bool(lambda_max > 0.0 and lambda_min <= 0.0),
        "maximum_segment_state_target_ratio": max_state_ratio,
        "lambda_min_s": lambda_min,
        "lambda_max_s": lambda_max,
        "transient_cycles": transient_cycles,
        "stationary_tail_cycles": stationary_cycles,
        "coupled_internal_segments": coupled_segments,
        "coupled_rejected_splits": rejected_splits,
        "sigma_back_initial_Pa": sigma_back_initial,
        "sigma_back_final_Pa": sigma_back_final,
        "mobile_initial": mobile_initial,
        "mobile_final": mobile_final,
        "retained_initial": retained_initial,
        "retained_final": retained_final,
        "emitted_initial": emitted_initial,
        "emitted_final": emitted_final,
        "active_K_shield_initial_Pa_sqrt_m": shield_initial,
        "active_K_shield_final_Pa_sqrt_m": shield_final,
        "B_initial": B_initial,
        "B_final": B_final,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--minimum-log-lambda-span-decades", type=float, default=0.30)
    parser.add_argument("--minimum-state-target-ratio", type=float, default=0.05)
    parser.add_argument("--require-candidate", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    rows = [
        result
        for path in sorted(root.rglob("v10_2_29_fixed_deltaK_control.json"))
        if (result := summarize_case(path)) is not None
    ]
    if not rows:
        raise SystemExit(f"no coupled transient cases found below {root}")

    for row in rows:
        stored_span = row["global_log_lambda_span_decades"]
        classification_span = (
            math.inf if row["lambda_started_from_zero"] else float(stored_span or 0.0)
        )
        row["coupled_transient_candidate"] = bool(
            classification_span >= float(args.minimum_log_lambda_span_decades)
            and float(row["maximum_segment_state_target_ratio"])
            >= float(args.minimum_state_target_ratio)
        )
        first_event = row["first_event_cycle"]
        row["delayed_event_candidate"] = bool(
            row["coupled_transient_candidate"]
            and int(row["event_count"]) > 0
            and first_event is not None
            and float(first_event) > 0.0
        )
        row["censored_transient_candidate"] = bool(
            row["coupled_transient_candidate"] and bool(row["right_censored"])
        )

    candidates = [row for row in rows if row["coupled_transient_candidate"]]
    payload = {
        "schema": SCHEMA,
        "root": str(root),
        "minimum_log_lambda_span_decades": float(
            args.minimum_log_lambda_span_decades
        ),
        "minimum_state_target_ratio": float(args.minimum_state_target_ratio),
        "case_count": len(rows),
        "candidate_count": len(candidates),
        "delayed_event_candidate_count": sum(
            bool(row["delayed_event_candidate"]) for row in rows
        ),
        "censored_transient_candidate_count": sum(
            bool(row["censored_transient_candidate"]) for row in rows
        ),
        "cases": rows,
    }
    write_csv(root / "coupled_transient_screen.csv", rows)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (root / "coupled_transient_screen.json").write_text(text)
    print(text, end="")
    if args.require_candidate and not candidates:
        raise SystemExit("ERROR: coupled transient screen found no qualifying case")


if __name__ == "__main__":
    main()
