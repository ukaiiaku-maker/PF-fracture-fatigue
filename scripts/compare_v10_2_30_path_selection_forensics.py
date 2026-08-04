#!/usr/bin/env python3
"""Fail-closed comparison of continuous/restored path-selection records."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from arrhenius_fracture.path_selection_forensics_v10230 import FILENAME, SCHEMA


PROVENANCE_ONLY = {"execution", "combined_checkpoint_generation"}


def _records(root: Path, event: int) -> list[dict[str, Any]]:
    payload = json.loads((root / FILENAME).read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported forensic schema in {root}")
    rows = [copy.deepcopy(row) for row in payload.get("records", [])
            if int(row.get("event_index", -1)) == event]
    if not rows:
        raise ValueError(f"no event-{event} forensic records in {root}")
    for row in rows:
        for key in PROVENANCE_ONLY:
            row.pop(key, None)
    return rows


def _first_difference(left: Any, right: Any, path: str = "$") -> dict | None:
    if type(left) is not type(right):
        return {"path": path, "control": repr(left), "restored": repr(right)}
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return {"path": f"{path}.{key}", "control": left.get(key, "<missing>"),
                        "restored": right.get(key, "<missing>")}
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": f"{path}.length", "control": len(left), "restored": len(right)}
        for index, (a, b) in enumerate(zip(left, right)):
            difference = _first_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return {"path": path, "control": left, "restored": right}
    return None


def _category(path: str) -> str:
    if any(part in path for part in ("selector_input", "front", "hashes")):
        return "A"
    if "candidate_order" in path or ".candidates" in path:
        return "B"
    if any(part in path for part in ("selected_stable_candidate_id", "winning_score", "tie_break")):
        return "C"
    if any(part in path for part in ("selected_direction", "projected_endpoint")):
        return "D"
    if "rng" in path:
        return "E"
    return "A"


def compare(control: Path, restored: Path, event: int = 2) -> dict[str, Any]:
    left = _records(control, event)
    right = _records(restored, event)
    difference = _first_difference(left, right)
    return {
        "schema": "v10.2.30_path_selection_forensic_comparison_v1",
        "event_index": event,
        "equivalent": difference is None,
        "classification": None if difference is None else _category(difference["path"]),
        "first_difference": difference,
        "control_record_count": len(left),
        "restored_record_count": len(right),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("restored", type=Path)
    parser.add_argument("--event", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.control, args.restored, args.event)
    except Exception as exc:
        result = {"schema": "v10.2.30_path_selection_forensic_comparison_v1",
                  "equivalent": False, "classification": "A",
                  "first_difference": {"path": "$", "error": str(exc)}}
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
