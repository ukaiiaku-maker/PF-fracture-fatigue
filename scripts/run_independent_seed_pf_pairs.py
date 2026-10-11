#!/usr/bin/env python3
"""Run one additional common seed for the two active paired PF anchors."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.shared_execution import execute_shared
from arrhenius_fracture.shared_options import SharedOptions


SEED = 91733
PILOT_ROOT = ROOT / "integration_evidence" / "current_science_pilots_6abbf10" / "pf"
CASES = (
    ("DBTT_1200K", "accommodation_enabled"),
    ("DBTT_1200K", "opening_only"),
    ("Peak_900K", "accommodation_enabled"),
    ("Peak_900K", "opening_only"),
)
DEFAULT_OUTPUT = (
    ROOT / "integration_evidence" / "active_fracture_comparisons_0a5d55e" /
    "independent_seed_91733"
)


def request_path(anchor: str, arm: str) -> Path:
    return PILOT_ROOT / anchor / arm / "shared_execution_request.json"


def case_matrix() -> dict:
    rows = []
    for anchor, arm in CASES:
        source = request_path(anchor, arm)
        request = json.loads(source.read_text())
        rows.append({
            "anchor": anchor,
            "arm": arm,
            "source_request": str(source),
            "source_request_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "temperature_K": 1200.0 if anchor == "DBTT_1200K" else 900.0,
            "target_crack_extension_um": 10.0,
            "mechanics": request["options"]["mechanics"],
            "native_accommodation": arm == "accommodation_enabled",
        })
    return {
        "schema": "independent-seed-paired-pf-check/1",
        "new_seed": SEED,
        "existing_seed": 1733,
        "scope": "two-seed pilot variability check only",
        "physical_loading_rate_changed": False,
        "numerical_resolution_changed": False,
        "cases": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-only", action="store_true")
    parsed = parser.parse_args(argv)
    matrix = case_matrix()
    if parsed.plan_only:
        print(json.dumps(matrix, indent=2, sort_keys=True))
        return 0
    output = parsed.out.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing seed matrix: {output}")
    output.mkdir(parents=True)
    (output / "RUN_MATRIX.json").write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    os.environ.update({
        "CLEAVAGE_HAZARD_MODE": "exponential",
        "CLEAVAGE_EVENT_LENGTH_MODE": "threshold_scaled",
        "CLEAVAGE_EVENT_MIN_FACTOR": "0.5",
        "CLEAVAGE_EVENT_MAX_FACTOR": "4.0",
    })
    for anchor, arm in CASES:
        request = json.loads(request_path(anchor, arm).read_text())
        configured = dict(request["options"])
        configured["random_seed"] = SEED
        execute_shared(
            SharedOptions(**configured),
            request["native_args"],
            output / anchor / arm,
            source_policy="qualified",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
