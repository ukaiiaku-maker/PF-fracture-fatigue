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


def _reconstruct_limiter(record: dict) -> tuple[str, float, dict[str, float], str]:
    applied = str(record.get("cycle_applied_limiter", record.get("cycle_limiter", "unknown")))
    current = str(record.get("cycle_limiter", "unknown"))
    candidates = {
        str(key): _finite(value, default=float("nan"))
        for key, value in dict(record.get("cycle_candidate_limits", {})).items()
    }
    candidates = {
        key: value for key, value in candidates.items()
        if math.isfinite(value) and value >= 0.0 and key != "global_forced"
    }
    if current not in ("global_forced", "unknown", "") and candidates:
        return current, _finite(record.get("cycle_unlimited")), candidates, applied

    targets = dict(record.get("cycle_effective_target_increments", {}))
    if not targets:
        targets = dict(record.get("cycle_target_increments", {}))
        if "cleavage_clock" in targets:
            targets["cleavage_clock"] = min(
                _finite(targets["cleavage_clock"]),
                max(1.0 - _finite(record.get("B")), 0.0),
            )
    rates = dict(record.get("cycle_predicted_increments_per_cycle", {}))
    mode = str(record.get("cycle_block_mode", "unknown") or "unknown").lower()
    max_block = _finite(record.get("cycle_max_block_cycles"))
    nominal = _finite(record.get("cycle_nominal_block_cycles"))
    if mode in ("hazard", "hazard_limited", "rate", "auto"):
        candidates["max_block_cycles"] = max_block
    else:
        candidates["block_cycles"] = min(max_block, nominal)
    for name, target_value in targets.items():
        target = _finite(target_value)
        rate = _finite(rates.get(name))
        if target > 0.0 and rate > 0.0:
            candidates[str(name)] = target / rate
    if not candidates:
        return current, _finite(record.get("cycle_unlimited")), {}, applied
    limiter = min(candidates, key=candidates.get)
    return limiter, candidates[limiter], candidates, applied


def summarize(records: list[dict], root: Path) -> tuple[dict, list[dict]]:
    requested = [_finite(record.get("cycles_requested")) for record in records]
    consumed = [_finite(record.get("cycles_consumed")) for record in records]
    reconstructed = [_reconstruct_limiter(record) for record in records]
    limiters = [item[0] for item in reconstructed]
    applied_limiters = [item[3] for item in reconstructed]
    counts = Counter(limiters)
    applied_counts = Counter(applied_limiters)
    candidate_by_name: dict[str, list[float]] = defaultdict(list)
    rows: list[dict] = []
    cumulative = 0.0

    for index, (record, diagnostic) in enumerate(zip(records, reconstructed)):
        limiter, unlimited, candidates, applied = diagnostic
        cumulative += _finite(record.get("cycles_consumed"))
        for key, number in candidates.items():
            if math.isfinite(number) and number >= 0.0:
                candidate_by_name[str(key)].append(number)
        rows.append(
            {
                "record_index": index,
                "cycles_requested": _finite(record.get("cycles_requested")),
                "cycles_consumed": _finite(record.get("cycles_consumed")),
                "cycles_cumulative": cumulative,
                "cycle_limiter": limiter,
                "cycle_applied_limiter": applied,
                "cycle_unlimited": unlimited,
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
        name: {"count": count, "fraction": count / len(records)}
        for name, count in sorted(counts.items())
    }
    applied_limiter_summary = {
        name: {"count": count, "fraction": count / len(records)}
        for name, count in sorted(applied_counts.items())
    }
    candidate_summary = {
        name: {
            "count": len(values),
            "min": min(values),
            "median": median(values),
            "max": max(values),
        }
        for name, values in sorted(candidate_by_name.items())
    }

    final = records[-1]
    summary = {
        "schema": "v10.2.29_cycle_block_summary_v2",
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
        "applied_limiter_summary": applied_limiter_summary,
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
