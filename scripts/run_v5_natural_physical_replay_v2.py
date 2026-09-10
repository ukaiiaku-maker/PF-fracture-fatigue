#!/usr/bin/env python3
"""Produce or independently replay one natural V2 physical-equality row."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.closure_lifecycle_evidence import CFG, advance_transition
from arrhenius_fracture.closure_mechanics_evidence import canonical_data
from arrhenius_fracture.natural_future_physical_replay_v2 import compare_states, exact_projection, solver_budget
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
from arrhenius_fracture.voiding_lifecycle_driver_v5 import NATURAL_WINDOW_S, advance_production_void_interval
from arrhenius_fracture.voiding_production_v5 import build_production_void_state
from arrhenius_fracture.voiding_v5 import VoidPhase


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(canonical_data(value), sort_keys=True, indent=2, allow_nan=False) + "\n")


def advance_half(state, partitions):
    operations = []
    for _ in range(partitions):
        state, trace, result = advance_production_void_interval(
            state, NATURAL_WINDOW_S / (2 * partitions), config=CFG,
        )
        operations.extend(trace)
        if result["failure"] is not None or result["elapsed_duration_s"] != NATURAL_WINDOW_S / (2 * partitions):
            raise RuntimeError("natural half-window did not complete: " + repr(result))
    return state, operations


def subsequent_growth_crossing(state):
    cavity = state.void_state.cavities[0] if state.void_state.cavities else None
    if cavity is None or cavity.phase != VoidPhase.STABLE_SUBGRID_VOID:
        raise RuntimeError("V2 registered subsequent crossing requires the retained stable-subgrid terminal")
    margin = 5.0e-5 - cavity.radius_m
    if margin <= 0.0:
        raise RuntimeError("subsequent growth crossing has no positive radius margin")
    grown, first = advance_transition(state, "subgrid_growth", 1)
    promoted, second = advance_transition(grown, "promotion", 1)
    return promoted, {
        "real_crossing_executed": True,
        "selected_event_identity": "SUBGRID_RADIUS_CROSSING_THEN_GEOMETRIC_PROMOTION",
        "minimum_event_selection_margin_action": margin,
        "event_selection_margin_quantity": "remaining_radius_to_promotion_m",
        "operations": first + second,
    }


def clean_head():
    status = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"), cwd=ROOT, text=True
    ).strip()
    if status:
        raise RuntimeError("physical replay requires committed tracked implementation")
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()


def source(args):
    sha = clean_head()
    if args.output.exists():
        raise ValueError("refusing to overwrite source evidence")
    args.output.mkdir(parents=True)
    initial, _ = build_production_void_state(stochastic=True, seed=args.seed)
    midpoint, first = advance_half(initial, args.partitions)
    terminal, second = advance_half(midpoint, args.partitions)
    write_checkpoint(midpoint, args.output / "midpoint.json", compression="gzip")
    write_checkpoint(terminal, args.output / "reference_terminal.json", compression="gzip")
    payload = {
        "schema": "v5.natural-future-physical-replay-source/2",
        "executed_code_sha": sha, "seed": args.seed, "partition_count": args.partitions,
        "duration_s": NATURAL_WINDOW_S, "midpoint_time_s": NATURAL_WINDOW_S / 2,
        "initial_fingerprint": fingerprint(initial), "midpoint_fingerprint": fingerprint(midpoint),
        "reference_terminal_fingerprint": fingerprint(terminal),
        "first_half_operations": first, "second_half_operations": second,
    }
    write_json(args.output / "source.json", payload)
    return {"mode": "source", "seed": args.seed, "terminal": payload["reference_terminal_fingerprint"]}


def replay(args):
    sha = clean_head()
    if args.output.exists():
        raise ValueError("refusing to overwrite replay evidence")
    source_record = json.loads((args.source / "source.json").read_text())
    if source_record["executed_code_sha"] != sha or source_record["seed"] != args.seed:
        raise ValueError("source identity differs from replay implementation or seed")
    midpoint = restore_checkpoint(args.source / "midpoint.json")
    reference = restore_checkpoint(args.source / "reference_terminal.json")
    terminal, operations = advance_half(midpoint, source_record["partition_count"])
    reference_crossed, crossing_a = subsequent_growth_crossing(reference)
    replay_crossed, crossing_b = subsequent_growth_crossing(terminal)
    budget_a, budget_b = solver_budget(reference), solver_budget(terminal)
    exact_after_crossing = exact_projection(reference_crossed) == exact_projection(replay_crossed)
    crossing = {
        **crossing_a,
        "real_crossing_executed": crossing_a["real_crossing_executed"] and crossing_b["real_crossing_executed"],
        "selected_event_identity_exact": crossing_a["selected_event_identity"] == crossing_b["selected_event_identity"],
        "accepted_topology_exact": exact_after_crossing,
        "categorical_terminal_exact": (
            reference_crossed.void_state.cavities[0].phase == replay_crossed.void_state.cavities[0].phase
        ),
        "minimum_event_selection_margin_action": min(
            crossing_a["minimum_event_selection_margin_action"],
            crossing_b["minimum_event_selection_margin_action"],
        ),
        "maximum_event_selection_perturbation_action": abs(
            crossing_a["minimum_event_selection_margin_action"] -
            crossing_b["minimum_event_selection_margin_action"]
        ),
        "reference_post_crossing_fingerprint": fingerprint(reference_crossed),
        "replay_post_crossing_fingerprint": fingerprint(replay_crossed),
    }
    result = compare_states(
        reference, terminal, case_identity=str(args.seed), seed=args.seed,
        solver_condition_number=max(budget_a["condition_number"], budget_b["condition_number"]),
        free_residual_relative=max(budget_a["free_residual_relative"], budget_b["free_residual_relative"]),
        subsequent_crossing=crossing,
    )
    payload = {
        **result, "executed_code_sha": sha,
        "source_execution": source_record,
        "replay_operations": operations,
        "solver_budgets": {"reference": budget_a, "replay": budget_b},
        "V1_bitwise_terminal_fingerprint_equal": fingerprint(reference) == fingerprint(terminal),
        "V1_result_preserved": "NATURAL_BITWISE_SOURCE_REPLAY_V1_REMAINS_10_OF_160",
    }
    args.output.mkdir(parents=True)
    report = args.output / "report.json"
    write_json(report, payload)
    write_json(args.output / "sha256_manifest.json", {
        "report.json": hashlib.sha256(report.read_bytes()).hexdigest(),
    })
    return {"mode": "replay", "seed": args.seed, "passed": payload["passed"],
            "bitwise": payload["V1_bitwise_terminal_fingerprint_equal"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("source", "replay"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--partitions", type=int, choices=(1, 2, 4, 8, 16), default=1)
    args = parser.parse_args()
    if args.mode == "replay" and args.source is None:
        parser.error("--source is required in replay mode")
    result = source(args) if args.mode == "source" else replay(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
