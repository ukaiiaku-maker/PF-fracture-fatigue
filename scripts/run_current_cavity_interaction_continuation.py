#!/usr/bin/env python3
"""Resume the qualified cavity pair under the historical 1 um/s opening ramp."""
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


DEFAULT_PAIR = (
    ROOT / "integration_evidence" / "active_fracture_comparisons_0a5d55e" /
    "cavity_post_ramp_dwell_10us_completion_run2" / "checkpoint" /
    "accepted_v5_pair.json"
)
DEFAULT_OUTPUT = (
    ROOT / "integration_evidence" / "final_fracture_campaign" /
    "cavity_interaction_ramp"
)
HISTORICAL_OPENING_RATE_M_PER_S = 1.0e-6


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--restart-pair", type=Path, default=DEFAULT_PAIR)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--transport-floor-s", type=float, default=1.0e-8)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    pair = args.restart_pair.resolve()
    native_args = (
        "--duration-s", repr(float(args.duration_s)),
        "--steps", str(int(args.steps)),
        "--transport-refinement-min-duration-s",
        repr(float(args.transport_floor_s)),
        "--opening-rate-m-per-s", repr(HISTORICAL_OPENING_RATE_M_PER_S),
        "--material-class", "DBTT",
        "--temperature-K", "1200",
        "--vacancy-energy-variant", "reference",
    )
    plan = {
        "schema": "current-cavity-interaction-continuation/1",
        "restart_pair": str(pair),
        "restart_pair_sha256": hashlib.sha256(pair.read_bytes()).hexdigest(),
        "protocol": "historical_linear_total_opening_ramp_continuation",
        "opening_rate_m_per_s": HISTORICAL_OPENING_RATE_M_PER_S,
        "duration_s_per_requested_interval": float(args.duration_s),
        "requested_intervals": int(args.steps),
        "finite_ceiling_policy": (
            "bounded immutable suffixes; stop on first accepted structural event, "
            "physical admissibility limit, or 12.6 um total opening"
        ),
        "starting_opening_m": 6.3e-6,
        "maximum_total_opening_m": 12.6e-6,
        "temperature_K": 1200.0,
        "hazard_seed": 3621,
        "native_args": list(native_args),
        "options": options().to_dict(),
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    output = args.out.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing cavity result: {output}")
    output.mkdir(parents=True)
    (output / "RUN_MATRIX.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    execute_shared(
        options(), native_args, output,
        source_policy="qualified", restart_from=pair,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
