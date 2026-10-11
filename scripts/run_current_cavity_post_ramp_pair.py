#!/usr/bin/env python3
"""Run the declared 10 us post-ramp dwell for the retained V11 cavity pair."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.shared_execution import execute_shared
from arrhenius_fracture.shared_options import SharedOptions


PAIR = ROOT / "review_packet" / "current_science_pilots_6abbf10" / "v11_migration_pair.json"
DEFAULT_OUTPUT = (
    ROOT / "integration_evidence" / "active_fracture_comparisons_0a5d55e" /
    "cavity_post_ramp_dwell_10us"
)
NATIVE_ARGS = (
    "--duration-s", "1e-5",
    "--steps", "1",
    "--transport-refinement-min-duration-s", "1e-8",
    "--material-class", "DBTT",
    "--temperature-K", "1200",
    "--vacancy-energy-variant", "reference",
)


def options() -> SharedOptions:
    return SharedOptions(
        cavity="full_order_v5",
        crack_path="single_front",
        integration="explicit_event",
        kinetics="native_signed",
        loading="monotonic",
        mechanics="spatial_pf",
        output="publication_restart",
        plasticity="tip_plus_full_field",
        random_seed=3621,
        source_lifecycle="current_source",
        source_location="tip_plus_bulk",
    )


def matrix() -> dict:
    pair = json.loads(PAIR.read_text())
    return {
        "schema": "current-cavity-post-ramp-pair/1",
        "restart_pair": str(PAIR),
        "restart_pair_sha256": hashlib.sha256(PAIR.read_bytes()).hexdigest(),
        "restart_physical_time_s": pair["physical_time_s"],
        "restart_opening_m": 6.3e-6,
        "protocol": "explicit_post_ramp_fixed_opening_dwell",
        "original_ramp_status": "completed_declared_6.18_to_6.30_um_window",
        "additional_opening_m": 0.0,
        "additional_time_s": 1e-5,
        "temperature_K": 1200.0,
        "hazard_seed": 3621,
        "both_arms_evolved": True,
        "native_args": list(NATIVE_ARGS),
        "options": options().to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-only", action="store_true")
    parsed = parser.parse_args(argv)
    plan = matrix()
    if parsed.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    output = parsed.out.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing cavity result: {output}")
    output.mkdir(parents=True)
    (output / "RUN_MATRIX.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    execute_shared(
        options(), NATIVE_ARGS, output,
        source_policy="qualified", restart_from=PAIR,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
