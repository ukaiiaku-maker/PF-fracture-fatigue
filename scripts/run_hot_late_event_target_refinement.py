#!/usr/bin/env python3
"""Refine the event-action tolerance from one accepted hot-path checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.shared_execution import execute_shared
from arrhenius_fracture.shared_options import SharedOptions
from scripts.run_hot_late_floor_refinement import (
    SOURCE_REQUEST, file_sha256, prepare_restart_root, replace_argument,
)


TARGETS = (0.15, 0.075, 0.0375, 0.01875, 0.009375)
DEFAULT_OUTPUT = (
    ROOT / "integration_evidence" / "active_fracture_comparisons_0a5d55e" /
    "hot_late_event_target_refinement"
)


def native_args_for_target(source: list[str], target: float) -> list[str]:
    args = list(source)
    replace_argument(args, "--steps", "1760")
    replace_argument(args, "--target-crack-extension-um", "1000")
    replace_argument(args, "--print-every", "1")
    replace_argument(args, "--adaptive-min-frac", "1e-10")
    replace_argument(args, "--adaptive-event-target", f"{target:.17g}")
    return args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-only", action="store_true")
    parsed = parser.parse_args(argv)
    output = parsed.out.resolve()
    request = json.loads(SOURCE_REQUEST.read_text())
    plan = {
        "schema": "hot-late-event-target-refinement/1",
        "source_request": str(SOURCE_REQUEST),
        "source_request_sha256": file_sha256(SOURCE_REQUEST),
        "source_generation": "85f8ef062b9a298f1f2a92fb",
        "source_step": 1701,
        "source_events": 202,
        "source_action": 1.2310809524553992,
        "source_pending_mark": 1.4655725624469025,
        "event_targets": list(TARGETS),
        "adaptive_min_frac": 1e-10,
        "common_endpoint": {"maximum_step": 1760, "target_crack_extension_um": 1000.0},
        "fixed_physics": {
            "nominal_dU_m": 2e-7,
            "nominal_dt_s": 8.4,
            "hazard_seed": 1733,
            "event_length_mode": "threshold_scaled",
            "event_length_factor_bounds": [0.5, 4.0],
        },
    }
    if parsed.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing refinement root: {output}")
    output.mkdir(parents=True)
    (output / "RUN_MATRIX.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    restart = prepare_restart_root(output)
    os.environ.update({
        "CLEAVAGE_HAZARD_MODE": "exponential",
        "CLEAVAGE_EVENT_LENGTH_MODE": "threshold_scaled",
        "CLEAVAGE_EVENT_MIN_FACTOR": "0.5",
        "CLEAVAGE_EVENT_MAX_FACTOR": "4.0",
    })
    options = SharedOptions(**request["options"])
    for target in TARGETS:
        label = f"target_{target:.4f}".replace(".", "p")
        execute_shared(
            options,
            native_args_for_target(request["native_args"], target),
            output / label,
            source_policy="qualified",
            restart_from=restart,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
