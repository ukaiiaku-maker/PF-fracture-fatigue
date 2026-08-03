#!/usr/bin/env python3
"""Analyze bounded weak-T fatigue transients without asserting stationarity.

The output is diagnostic only.  It reports how the normalized cleavage action,
active process-zone state, shielding, backstress, and tip radius evolve across
accepted outer blocks.  A stationary-tail *candidate* is identified only when a
trailing window satisfies explicit drift tests.  This script does not advance the
hazard clock and is not the stationary-tail propagator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


AUDIT_NAME = "kinetic_tip_cell_audit_v101.json"
HISTORY_NAME = "v10_2_30_weakt_transient_history.csv"
SEGMENTS_NAME = "v10_2_30_weakt_transient_segments.csv"
SUMMARY_NAME = "v10_2_30_weakt_transient_summary.json"


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _positive(value: Any, default: float = 0.0) -> float:
    return max(_finite(value, default), 0.0)


def _relative_span(values: Iterable[float], floor: float) -> float:
    data = [float(value) for value in values if math.isfinite(float(value))]
    if not data:
        return math.inf
    return (max(data) - min(data)) / max(max(abs(value) for value in data), floor)


def _log_span(values: Iterable[float]) -> float:
    data = [float(value) for value in values if math.isfinite(float(value)) and value > 0.0]
    if not data:
        return math.inf
    return math.log10(max(data) / min(data)) if min(data) > 0.0 else math.inf


def _load_records(root: Path) -> list[dict[str, Any]]:
    path = root / AUDIT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"missing audit file: {path}")
    payload = json.loads(path.read_text())
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a records list")
    return [dict(row) for row in records if row.get("loading_mode") == "cyclic"]


def _history(records: list[dict[str, Any]], horizon_cycles: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for index, record in enumerate(records):
        consumed = _positive(record.get("cycles_consumed"))
        cumulative += consumed
        B_pre = _finite(record.get("B_pre"))
        B_end = _finite(record.get("B"))
        dB = _positive(record.get("dB_block"))
        if dB <= 0.0 and not bool(record.get("fired", False)):
            dB = max(B_end - B_pre, 0.0)
        dB_per_cycle = dB / consumed if consumed > 0.0 else 0.0
        remaining_action = max(1.0 - B_end, 0.0)
        projected_remaining = (
            remaining_action / dB_per_cycle if dB_per_cycle > 0.0 else math.inf
        )
        projected_total = cumulative + projected_remaining
        lambda_end = _positive(
            record.get(
                "coupled_hazard_lambda_end_per_s",
                record.get("coupled_hazard_lambda_end_s", 0.0),
            )
        )
        row = {
            "outer_index": index,
            "cycles_requested": _positive(record.get("cycles_requested")),
            "cycles_consumed": consumed,
            "cycles_unused": _positive(record.get("cycles_unused")),
            "cumulative_cycles": cumulative,
            "B_pre": B_pre,
            "B_end": B_end,
            "dB_block": dB,
            "dB_per_cycle": dB_per_cycle,
            "projected_remaining_cycles": projected_remaining,
            "projected_total_passage_cycle": projected_total,
            "projected_beyond_horizon": bool(projected_total > horizon_cycles),
            "lambda_cleave_end_per_s": lambda_end,
            "sigma_tip_end_Pa": _positive(record.get("coupled_hazard_sigma_end_Pa")),
            "active_K_shield_end_Pa_sqrt_m": _finite(
                record.get(
                    "coupled_hazard_shield_end_Pa_sqrt_m",
                    record.get("state_active_K_shield_signed_Pa_sqrt_m", 0.0),
                )
            ),
            "sigma_back_end_Pa": _finite(record.get("persistent_sigma_back_Pa")),
            "tip_radius_end_m": _positive(record.get("persistent_tip_radius_m")),
            "mobile_count_end": _positive(record.get("state_mobile_count")),
            "retained_count_end": _positive(record.get("state_retained_count")),
            "emitted_total_end": _positive(record.get("state_emitted_total")),
            "escaped_total_end": _positive(record.get("state_escaped_total")),
            "accepted_segments": int(record.get("coupled_hazard_accepted_segments", 0)),
            "rejected_segments": int(record.get("coupled_hazard_rejected_splits", 0)),
            "trial_integrations": int(record.get("coupled_hazard_trial_integrations", 0)),
            "work_budget_exhausted": bool(
                record.get("coupled_hazard_work_budget_exhausted", False)
            ),
            "partial_return": bool(record.get("coupled_hazard_partial_return", False)),
            "event_localized": bool(record.get("coupled_hazard_event_localized", False)),
            "fired": bool(record.get("fired", False)),
            "marcher_wall_seconds": _positive(record.get("coupled_hazard_wall_seconds")),
        }
        rows.append(row)
    return rows


def _segments(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outer_offset = 0.0
    for outer_index, record in enumerate(records):
        segments = record.get("coupled_hazard_segments", [])
        if not isinstance(segments, list):
            segments = []
        for local_index, segment in enumerate(segments):
            row = {
                "outer_index": outer_index,
                "segment_index": local_index,
                "outer_cycle_offset": outer_offset,
            }
            row.update({str(key): value for key, value in dict(segment).items()})
            local_cumulative = _positive(segment.get("cumulative_cycles"))
            row["global_cumulative_cycles"] = outer_offset + local_cumulative
            rows.append(row)
        outer_offset += _positive(record.get("cycles_consumed"))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    root: Path,
    *,
    horizon_cycles: float,
    window: int,
    lambda_span_decades: float,
    state_relative_tol: float,
    shield_relative_tol: float,
    rate_span_decades: float,
) -> dict[str, Any]:
    records = _load_records(root)
    history = _history(records, horizon_cycles)
    segments = _segments(records)
    _write_csv(root / HISTORY_NAME, history)
    _write_csv(root / SEGMENTS_NAME, segments)

    usable = [row for row in history if row["cycles_consumed"] > 0.0 and not row["fired"]]
    tail = usable[-window:] if len(usable) >= window else []
    metrics = {
        "window_records": len(tail),
        "lambda_span_decades": _log_span(
            row["lambda_cleave_end_per_s"] for row in tail
        ) if tail else math.inf,
        "dB_per_cycle_span_decades": _log_span(
            row["dB_per_cycle"] for row in tail
        ) if tail else math.inf,
        "shield_relative_span": _relative_span(
            (row["active_K_shield_end_Pa_sqrt_m"] for row in tail), 1.0
        ) if tail else math.inf,
        "sigma_relative_span": _relative_span(
            (row["sigma_tip_end_Pa"] for row in tail), 1.0
        ) if tail else math.inf,
        "backstress_relative_span": _relative_span(
            (row["sigma_back_end_Pa"] for row in tail), 1.0
        ) if tail else math.inf,
        "radius_relative_span": _relative_span(
            (row["tip_radius_end_m"] for row in tail), 1.0e-30
        ) if tail else math.inf,
        "mobile_relative_span": _relative_span(
            (row["mobile_count_end"] for row in tail), 1.0
        ) if tail else math.inf,
        "retained_relative_span": _relative_span(
            (row["retained_count_end"] for row in tail), 1.0
        ) if tail else math.inf,
    }
    stationary_candidate = bool(
        tail
        and metrics["lambda_span_decades"] <= lambda_span_decades
        and metrics["dB_per_cycle_span_decades"] <= rate_span_decades
        and metrics["shield_relative_span"] <= shield_relative_tol
        and metrics["sigma_relative_span"] <= state_relative_tol
        and metrics["backstress_relative_span"] <= state_relative_tol
        and metrics["radius_relative_span"] <= state_relative_tol
        and metrics["mobile_relative_span"] <= state_relative_tol
        and metrics["retained_relative_span"] <= state_relative_tol
    )
    latest = history[-1] if history else None
    summary = {
        "schema": "v10.2.30_weakt_transient_diagnostic_v1",
        "root": str(root),
        "record_count": len(history),
        "segment_count": len(segments),
        "horizon_cycles": float(horizon_cycles),
        "cumulative_cycles": float(latest["cumulative_cycles"] if latest else 0.0),
        "fired": bool(any(row["fired"] for row in history)),
        "event_localized": bool(any(row["event_localized"] for row in history)),
        "work_budget_partial_returns": int(
            sum(bool(row["work_budget_exhausted"] and row["partial_return"]) for row in history)
        ),
        "latest": latest,
        "stationarity": {
            "candidate_only": True,
            "stationary_candidate": stationary_candidate,
            "window_requested": int(window),
            "criteria": {
                "lambda_span_decades": float(lambda_span_decades),
                "dB_per_cycle_span_decades": float(rate_span_decades),
                "state_relative_tol": float(state_relative_tol),
                "shield_relative_tol": float(shield_relative_tol),
            },
            "metrics": metrics,
        },
        "stationary_tail_propagation_performed": False,
        "safe_to_resume_four_class_campaign": False,
    }
    (root / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--horizon-cycles", type=float, default=1.0e10)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--lambda-span-decades", type=float, default=0.02)
    parser.add_argument("--rate-span-decades", type=float, default=0.02)
    parser.add_argument("--state-relative-tol", type=float, default=1.0e-3)
    parser.add_argument("--shield-relative-tol", type=float, default=1.0e-3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = analyze(
        args.root,
        horizon_cycles=args.horizon_cycles,
        window=max(args.window, 2),
        lambda_span_decades=max(args.lambda_span_decades, 0.0),
        rate_span_decades=max(args.rate_span_decades, 0.0),
        state_relative_tol=max(args.state_relative_tol, 0.0),
        shield_relative_tol=max(args.shield_relative_tol, 0.0),
    )
    latest = summary.get("latest") or {}
    print(f"records={summary['record_count']}")
    print(f"segments={summary['segment_count']}")
    print(f"cumulative_cycles={summary['cumulative_cycles']:.12g}")
    print(f"fired={summary['fired']}")
    print(
        "stationary_candidate="
        f"{summary['stationarity']['stationary_candidate']}"
    )
    print(f"dB_per_cycle={_finite(latest.get('dB_per_cycle')):.12g}")
    projected = _finite(latest.get("projected_total_passage_cycle"), math.inf)
    print(f"projected_total_passage_cycle={projected:.12g}")
    print(f"summary={args.root / SUMMARY_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
