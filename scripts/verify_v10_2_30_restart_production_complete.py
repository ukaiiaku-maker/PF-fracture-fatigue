#!/usr/bin/env python3
"""Fail-closed production comparison of uninterrupted and restarted runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from arrhenius_fracture.run_state_checkpoint_v10230 import (
    load_combined_checkpoint,
    validate_cross_layer,
)


SCHEMA = "v10.2.30_restart_production_complete_verification_v1"
EXPECTED_EVENT2_UM = 2.2973400956248734


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def compare_json(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if type(left) is not type(right):
        return [{"path": path, "control": left, "restarted": right}]
    if isinstance(left, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                differences.append({"path": child, "control": left.get(key),
                                    "restarted": right.get(key)})
            else:
                differences.extend(compare_json(left[key], right[key], child))
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return [{"path": path + ".length", "control": len(left),
                     "restarted": len(right)}]
        differences = []
        for index, (a, b) in enumerate(zip(left, right)):
            differences.extend(compare_json(a, b, f"{path}[{index}]"))
        return differences
    return [] if left == right else [{"path": path, "control": left,
                                      "restarted": right}]


def terminal(root: Path) -> dict[str, Any]:
    outer, kinetic, arrays = load_combined_checkpoint(root)
    validate_cross_layer(outer, kinetic)
    summary = load_json(root / "developed_fatigue_growth_summary.json")
    events = summary.get("event_measurements", [])
    if len(events) < 2:
        raise ValueError(f"terminal run lacks event 2: {root}")
    return {"outer": outer, "kinetic": kinetic, "arrays": arrays,
            "summary": summary, "event2": events[1]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("restarted", type=Path)
    parser.add_argument("--deltaK", type=float, default=12.0677888)
    parser.add_argument("--head", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    control = terminal(args.control.resolve())
    restarted = terminal(args.restarted.resolve())
    checks: dict[str, bool] = {}
    for label, state in (("control", control), ("restarted", restarted)):
        provenance = state["summary"].get("provenance", {})
        status = state["summary"].get("status")
        checks[f"{label}_terminal"] = status == "growth_target_reached" or (
            status == "partial_growth"
            and float(state["summary"].get("cycles_consumed", -1.0)) == 1_000_000.0
        )
        checks[f"{label}_head"] = provenance.get("git_head") == args.head
        checks[f"{label}_deltaK"] = provenance.get("deltaK_MPa_sqrt_m") == args.deltaK
        proposal_um = 1e6 * float(state["event2"]["stochastic_proposed_advance_m"])
        checks[f"{label}_event2_proposal"] = proposal_um == EXPECTED_EVENT2_UM

    event2_differences = compare_json(control["event2"], restarted["event2"], "event2")
    outer_differences = []
    for key in ("case", "cycles_total", "driver", "geometry", "history"):
        outer_differences.extend(compare_json(control["outer"].get(key),
                                              restarted["outer"].get(key), key))
    kinetic_left = {k: v for k, v in control["kinetic"].items()
                    if k != "timestamp_unix_s"}
    kinetic_right = {k: v for k, v in restarted["kinetic"].items()
                     if k != "timestamp_unix_s"}
    kinetic_differences = compare_json(kinetic_left, kinetic_right, "kinetic")
    array_differences = []
    for name in sorted(set(control["arrays"]) | set(restarted["arrays"])):
        a = control["arrays"].get(name); b = restarted["arrays"].get(name)
        equal = a is not None and b is not None and np.array_equal(a, b, equal_nan=True)
        if not equal:
            array_differences.append({
                "name": name,
                "control_shape": None if a is None else list(a.shape),
                "restarted_shape": None if b is None else list(b.shape),
                "maximum_absolute_difference": None if a is None or b is None or a.shape != b.shape
                else float(np.nanmax(np.abs(a - b))),
            })
    checks.update({
        "event2_identical": not event2_differences,
        "terminal_outer_state_identical": not outer_differences,
        "terminal_kinetic_state_identical": not kinetic_differences,
        "terminal_arrays_identical": not array_differences,
    })
    report = {
        "schema": SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "control": str(args.control.resolve()),
        "restarted": str(args.restarted.resolve()),
        "event2_expected_advance_um": EXPECTED_EVENT2_UM,
        "event2_differences": event2_differences[:100],
        "terminal_outer_differences": outer_differences[:100],
        "terminal_kinetic_differences": kinetic_differences[:100],
        "terminal_array_differences": array_differences,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
