#!/usr/bin/env python3
"""Compare equal-cycle v10.2.29 adaptive-target runs against the tightest case."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

STATE_KEYS = (
    "B",
    "N_em",
    "mobile_count",
    "retained_count",
    "emitted_total",
    "escaped_total",
    "sigma_back_Pa",
    "micro_advance_total_m",
    "active_K_shield_signed_Pa_sqrt_m",
)


def _load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"ERROR: missing summary: {path}")
    return json.loads(path.read_text())


def _tag(value: str) -> str:
    return value.replace("-", "m").replace(".", "p").replace("+", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--minimum-cycles", type=float, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    cases = []
    for target in args.targets:
        run = args.root / f"target_{_tag(target)}"
        payload = _load(run / "cycle_block_summary.json")
        cycles = float(payload.get("cycles_consumed_total", 0.0))
        if cycles + 1.0e-10 < args.minimum_cycles:
            raise SystemExit(
                f"ERROR: target {target} consumed only {cycles} cycles; "
                f"required {args.minimum_cycles}"
            )
        cases.append(
            {
                "target": float(target),
                "run_root": str(run.resolve()),
                "cycles_consumed": cycles,
                "record_count": int(payload.get("record_count", 0)),
                "records_per_consumed_cycle": payload.get(
                    "records_per_consumed_cycle"
                ),
                "limiter_summary": payload.get("limiter_summary", {}),
                "final_state": payload.get("final_state", {}),
            }
        )

    baseline = cases[0]
    baseline_state = baseline["final_state"]
    for case in cases:
        differences = {}
        for key in STATE_KEYS:
            ref = float(baseline_state.get(key, 0.0))
            value = float(case["final_state"].get(key, 0.0))
            absolute = value - ref
            scale = max(abs(ref), 1.0e-30)
            differences[key] = {
                "absolute": absolute,
                "relative_to_tightest": absolute / scale,
            }
        case["difference_from_tightest"] = differences

    payload = {
        "schema": "v10.2.29_block_target_convergence_v1",
        "baseline_target": baseline["target"],
        "minimum_cycles_required": args.minimum_cycles,
        "cases": cases,
    }
    out = args.out or args.root / "block_target_convergence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
