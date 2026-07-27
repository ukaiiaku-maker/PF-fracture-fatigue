#!/usr/bin/env python3
"""Summarize adaptive cycle-block diagnostics from a v10.2.29 run."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def _finite(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def load_cyclic_records(root: Path) -> list[dict]:
    path = root / "kinetic_tip_cell_audit_v101.json"
    if not path.is_file():
        raise SystemExit(f"ERROR: missing kinetic audit: {path}")
    payload = json.loads(path.read_text())
    records = [
        dict(record)
        for record in payload.get("records", [])
        if str(record.get("loading_mode", "monotonic")) == "cyclic"
    ]
    if not records:
        raise SystemExit(f"ERROR: no cyclic audit records in {path}")
    missing = [
        index
        for index, record in enumerate(records)
        if "cycle_limiter" not in record or "cycle_candidate_limits" not in record
    ]
    if missing:
        raise SystemExit(
            "ERROR: cyclic audit predates limiter diagnostics; rerun with the updated "
            f"v10.2.29 branch (first missing record index {missing[0]})"
        )
    return records


def summarize(records: list[dict], root: Path) -> tuple[dict, list[dict]]:
    requested = [_finite(record.get("cycles_requested")) for record in records]
    consumed = [_finite(record.get("cycles_consumed")) for record in records]
    limiters = [str(record.get("cycle_limiter", "unknown")) for record in records]
    counts = Counter(limiters)
    candidate_by_name: dict[str, list[float]] = defaultdict(list)
    rows: list[dict] = []
    cumulative = 0.0

    for index, record in enumerate(records):
        cumulative += _finite(record.get("cycles_consumed"))
        candidates = dict(record.get("cycle_candidate_limits", {}))
        for key, value in candidates.items():
            number = _finite(value, default=float("nan"))
            if math.isfinite(number) and number >= 0.0:
                candidate_by_name[str(key)].append(number)
        rows.append(
            {
                "record_index": index,
                "cycles_requested": _finite(record.get("cycles_requested")),
                "cycles_consumed": _finite(record.get("cycles_consumed")),
                "cycles_cumulative": cumulative,
                "cycle_limiter": str(record.get("cycle_limiter", "unknown")),
                "cycle_unlimited": _finite(record.get("cycle_unlimited")),
                "B": _finite(record.get("B")),
                "N_em": _finite(record.get("state_N_em", record.get("N_em", 0.0))),
                "mobile_count": _finite(record.get("state_mobile_count")),
                "retained_count": _finite(record.get("state_retained_count")),
                "sigma_back_Pa": _finite(record.get("persistent_sigma_back_Pa")),
                "micro_advance_total_m": _finite(
                    record.get("state_micro_advance_total_m")
                ),
                "fired": int(bool(record.get("fired", False))),
                "event_localized": int(bool(record.get("event_localized", False))),
            }
        )

    total_consumed = sum(consumed)
    limiter_summary = {
        name: {
            "count": count,
            "fraction": count / len(records),
        }
        for name, count in sorted(counts.items())
    }
    candidate_summary = {}
    for name, values in sorted(candidate_by_name.items()):
        candidate_summary[name] = {
            "count": len(values),
            "min": min(values),
            "median": median(values),
            "max": max(values),
        }

    final = records[-1]
    summary = {
        "schema": "v10.2.29_cycle_block_summary_v1",
        "run_root": str(root.resolve()),
        "record_count": len(records),
        "cycles_requested_total": sum(requested),
        "cycles_consumed_total": total_consumed,
        "cycles_per_record": {
            "min": min(consumed),
            "median": median(consumed),
            "max": max(consumed),
            "mean": total_consumed / len(records),
        },
        "records_per_consumed_cycle": (
            len(records) / total_consumed if total_consumed > 0.0 else None
        ),
        "localized_event_records": sum(
            bool(record.get("event_localized", False)) for record in records
        ),
        "fired_records": sum(bool(record.get("fired", False)) for record in records),
        "limiter_summary": limiter_summary,
        "candidate_limit_summary": candidate_summary,
        "final_state": {
            "B": _finite(final.get("B")),
            "N_em": _finite(final.get("state_N_em", final.get("N_em", 0.0))),
            "mobile_count": _finite(final.get("state_mobile_count")),
            "retained_count": _finite(final.get("state_retained_count")),
            "emitted_total": _finite(final.get("state_emitted_total")),
            "escaped_total": _finite(final.get("state_escaped_total")),
            "sigma_back_Pa": _finite(final.get("persistent_sigma_back_Pa")),
            "micro_advance_total_m": _finite(
                final.get("state_micro_advance_total_m")
            ),
            "active_K_shield_signed_Pa_sqrt_m": _finite(
                final.get("state_active_K_shield_signed_Pa_sqrt_m")
            ),
        },
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()

    records = load_cyclic_records(args.run_root)
    summary, rows = summarize(records, args.run_root)
    out_json = args.out_json or args.run_root / "cycle_block_summary.json"
    out_csv = args.out_csv or args.run_root / "cycle_block_records.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with out_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
