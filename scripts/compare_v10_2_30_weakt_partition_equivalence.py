#!/usr/bin/env python3
"""Compare one-block and partitioned weak-T v10.2.30 transients.

This is a numerical-equivalence gate, not a stationary-tail propagator.  Both
runs must use the same material option, loading, seed, total cycle horizon, and
physics.  Only the outer proposal partition may differ.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

AUDIT_NAME = "kinetic_tip_cell_audit_v101.json"
OUTPUT_NAME = "v10_2_30_weakt_partition_equivalence.json"


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _records(root: Path) -> list[dict[str, Any]]:
    path = root / AUDIT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"missing audit: {path}")
    payload = json.loads(path.read_text())
    rows = payload.get("records", [])
    return [dict(row) for row in rows if row.get("loading_mode") == "cyclic"]


def _endpoint(root: Path) -> dict[str, Any]:
    rows = _records(root)
    if not rows:
        raise ValueError(f"no cyclic records in {root / AUDIT_NAME}")
    cumulative = sum(max(_finite(row.get("cycles_consumed")), 0.0) for row in rows)
    row = rows[-1]
    return {
        "root": str(root),
        "record_count": len(rows),
        "cumulative_cycles": cumulative,
        "fired": bool(any(item.get("fired", False) for item in rows)),
        "B": _finite(row.get("B")),
        "mobile_count": _finite(row.get("state_mobile_count")),
        "retained_count": _finite(row.get("state_retained_count")),
        "emitted_total": _finite(row.get("state_emitted_total")),
        "escaped_total": _finite(row.get("state_escaped_total")),
        "sigma_back_Pa": _finite(row.get("persistent_sigma_back_Pa")),
        "tip_radius_m": _finite(row.get("persistent_tip_radius_m")),
        "lambda_cleave_per_s": _finite(
            row.get(
                "coupled_hazard_lambda_end_per_s",
                row.get("coupled_hazard_lambda_end_s"),
            )
        ),
        "sigma_tip_Pa": _finite(row.get("coupled_hazard_sigma_end_Pa")),
        "active_K_shield_Pa_sqrt_m": _finite(
            row.get(
                "coupled_hazard_shield_end_Pa_sqrt_m",
                row.get("state_active_K_shield_signed_Pa_sqrt_m"),
            )
        ),
    }


def _relative_error(a: float, b: float, floor: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), floor)


def _log_error(a: float, b: float) -> float:
    if a <= 0.0 and b <= 0.0:
        return 0.0
    if a <= 0.0 or b <= 0.0:
        return math.inf
    return abs(math.log10(a / b))


def compare(
    reference_root: Path,
    partitioned_root: Path,
    *,
    state_relative_tol: float,
    clock_relative_tol: float,
    lambda_log_tol_decades: float,
    shield_absolute_tol: float,
) -> dict[str, Any]:
    reference = _endpoint(reference_root)
    partitioned = _endpoint(partitioned_root)

    comparisons = {
        "cumulative_cycles": {
            "error": _relative_error(
                reference["cumulative_cycles"], partitioned["cumulative_cycles"], 1.0
            ),
            "tolerance": 1.0e-12,
        },
        "B": {
            "error": _relative_error(reference["B"], partitioned["B"], 1.0e-30),
            "tolerance": clock_relative_tol,
        },
        "mobile_count": {
            "error": _relative_error(
                reference["mobile_count"], partitioned["mobile_count"], 1.0
            ),
            "tolerance": state_relative_tol,
        },
        "retained_count": {
            "error": _relative_error(
                reference["retained_count"], partitioned["retained_count"], 1.0
            ),
            "tolerance": state_relative_tol,
        },
        "emitted_total": {
            "error": _relative_error(
                reference["emitted_total"], partitioned["emitted_total"], 1.0
            ),
            "tolerance": state_relative_tol,
        },
        "escaped_total": {
            "error": _relative_error(
                reference["escaped_total"], partitioned["escaped_total"], 1.0
            ),
            "tolerance": state_relative_tol,
        },
        "sigma_back_Pa": {
            "error": _relative_error(
                reference["sigma_back_Pa"], partitioned["sigma_back_Pa"], 1.0
            ),
            "tolerance": state_relative_tol,
        },
        "tip_radius_m": {
            "error": _relative_error(
                reference["tip_radius_m"], partitioned["tip_radius_m"], 1.0e-30
            ),
            "tolerance": state_relative_tol,
        },
        "lambda_cleave_per_s": {
            "error": _log_error(
                reference["lambda_cleave_per_s"], partitioned["lambda_cleave_per_s"]
            ),
            "tolerance": lambda_log_tol_decades,
            "metric": "absolute_log10_ratio_decades",
        },
        "sigma_tip_Pa": {
            "error": _relative_error(
                reference["sigma_tip_Pa"], partitioned["sigma_tip_Pa"], 1.0
            ),
            "tolerance": state_relative_tol,
        },
        "active_K_shield_Pa_sqrt_m": {
            "error": abs(
                reference["active_K_shield_Pa_sqrt_m"]
                - partitioned["active_K_shield_Pa_sqrt_m"]
            ),
            "tolerance": shield_absolute_tol,
            "metric": "absolute_Pa_sqrt_m",
        },
    }
    for item in comparisons.values():
        item["passed"] = bool(item["error"] <= item["tolerance"])

    same_event_status = reference["fired"] == partitioned["fired"]
    passed = bool(same_event_status and all(item["passed"] for item in comparisons.values()))
    result = {
        "schema": "v10.2.30_weakt_partition_equivalence_v1",
        "reference": reference,
        "partitioned": partitioned,
        "same_event_status": same_event_status,
        "comparisons": comparisons,
        "passed": passed,
        "stationary_tail_propagation_validated": False,
        "safe_to_resume_four_class_campaign": False,
    }
    (partitioned_root / OUTPUT_NAME).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("partitioned_root", type=Path)
    parser.add_argument("--state-relative-tol", type=float, default=1.0e-3)
    parser.add_argument("--clock-relative-tol", type=float, default=1.0e-3)
    parser.add_argument("--lambda-log-tol-decades", type=float, default=0.01)
    parser.add_argument("--shield-absolute-tol", type=float, default=1.0e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = compare(
        args.reference_root,
        args.partitioned_root,
        state_relative_tol=max(args.state_relative_tol, 0.0),
        clock_relative_tol=max(args.clock_relative_tol, 0.0),
        lambda_log_tol_decades=max(args.lambda_log_tol_decades, 0.0),
        shield_absolute_tol=max(args.shield_absolute_tol, 0.0),
    )
    print(f"reference_records={result['reference']['record_count']}")
    print(f"partitioned_records={result['partitioned']['record_count']}")
    print(f"cycles={result['partitioned']['cumulative_cycles']:.12g}")
    print(f"passed={result['passed']}")
    print(f"output={args.partitioned_root / OUTPUT_NAME}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
