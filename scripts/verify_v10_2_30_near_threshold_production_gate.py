#!/usr/bin/env python3
"""Real production gate for immutable v10.2.30 near-threshold checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from arrhenius_fracture.run_state_checkpoint_v10230 import (
    load_combined_checkpoint,
    validate_cross_layer,
)

REPOSITORY = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY / "tests/fixtures/v10230_near_threshold"
DEFAULT_FAMILY = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/"
    "v10_2_28_kernel_cache/"
    "4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/"
    "family.json"
)
LOCALIZATION_ABS_CYCLE_TOL = 1.0e-6
MAX_LOCATOR_EVALUATIONS = 100

CASES = {
    "dbtt": {
        "parameter_option": "v913_paper_dbtt01_0202500_persistent_sites",
        "deltaK": "19.448409577436756", "fraction": "0.925", "seed": "1001723",
        "start_cycles": 390997291255.2484, "start_events": 5,
        "event_cycles": 390997291255.2692,
        "completed_Xi": 0.35940039563036524,
        "completed_H": 0.3594003955958084,
        "next_Xi": 0.6369055091541016,
        "proposal_m": 2.2973400956248734e-6,
        "admitted_m": 2.2973400956248828e-6,
        "projected_advance_m": 2.2295932447991733e-6,
        "path_advance_m": 2.297340095624884e-6,
        "projected_extension_m": 4.8247475897612944e-5,
        "path_extension_m": 5.1618170395689034e-5,
    },
    "peak": {
        "parameter_option": "v913_paper_peak01_0242980_persistent_sites",
        "deltaK": "19.1605918185452", "fraction": "0.900", "seed": "1720",
        "start_cycles": 19610961067.619057, "start_events": 1,
        "event_cycles": 19610961067.63989,
        "completed_Xi": 0.4332087756327596,
        "completed_H": 0.4332087755905508,
        "next_Xi": 0.004735693787404665,
        "proposal_m": 2.2973400956248734e-6,
        "admitted_m": 2.2973400956248993e-6,
        "projected_advance_m": 2.2454112809293467e-6,
        "path_advance_m": 2.2973400956249005e-6,
        "projected_extension_m": 4.525751767915621e-6,
        "path_extension_m": 4.59468019124982e-6,
    },
}


def _close(actual, expected, *, atol, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=atol):
        raise AssertionError(f"{label}: expected {expected:.17g}, got {actual:.17g}")


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def validate_fixture(name):
    outer, kinetic, arrays = load_combined_checkpoint(FIXTURE_ROOT / name)
    validate_cross_layer(outer, kinetic)
    expected = CASES[name]
    stochastic = kinetic["stochastic"]
    assert outer["cycles_total"] == expected["start_cycles"]
    assert outer["geometry"]["committed_event_count"] == expected["start_events"]
    assert stochastic["hazard_threshold_action"] == expected["completed_Xi"]
    assert stochastic["rng_state"]
    assert arrays["kinetic_active_vector"].size == 2882
    assert np.isfinite(arrays["kinetic_active_vector"]).all()
    return {"outer": outer, "kinetic": kinetic, "arrays": arrays}


def _assert_exact_copy(source, copied):
    assert copied["outer"] == source["outer"]
    assert copied["kinetic"] == source["kinetic"]
    assert copied["arrays"].keys() == source["arrays"].keys()
    for key in source["arrays"]:
        assert np.array_equal(copied["arrays"][key], source["arrays"][key])


def verify_result(name, root, source):
    expected = CASES[name]
    outer, kinetic, _ = load_combined_checkpoint(root)
    validate_cross_layer(outer, kinetic)
    stochastic = kinetic["stochastic"]
    geometry = outer["geometry"]
    assert geometry["committed_event_count"] == expected["start_events"] + 1
    assert stochastic["hazard_event_index"] == expected["start_events"] + 1
    old_history = source["kinetic"]["stochastic"]["hazard_threshold_history"]
    assert stochastic["hazard_threshold_history"] == old_history + [expected["completed_Xi"]]
    assert stochastic["hazard_threshold_action"] == expected["next_Xi"]
    assert stochastic["rng_state"] != source["kinetic"]["stochastic"]["rng_state"]

    summary = json.loads((root / "developed_fatigue_growth_summary.json").read_text())
    event = summary["event_measurements"][-1]
    checks = (
        ("threshold_action", "completed_Xi", 1e-14),
        ("physical_hazard_action", "completed_H", 1e-10),
        ("stochastic_proposed_advance_m", "proposal_m", 1e-18),
        ("energy_admissible_advance_m", "admitted_m", 1e-18),
        ("projected_advance_m", "projected_advance_m", 1e-18),
        ("path_advance_m", "path_advance_m", 1e-18),
        ("projected_extension_post_m", "projected_extension_m", 1e-18),
        ("path_extension_post_m", "path_extension_m", 1e-18),
    )
    _close(event["cycles_post"], expected["event_cycles"],
           atol=LOCALIZATION_ABS_CYCLE_TOL, label=f"{name} event cycle")
    for actual_key, expected_key, atol in checks:
        _close(event[actual_key], expected[expected_key], atol=atol,
               label=f"{name} {actual_key}")
    assert event["energy_gate_outcome"] == "stochastic_proposal_reached"
    assert event["geometry_commit_inserted"] is True
    assert event["private_trials_counted_as_cycles"] is False

    near = [row for row in outer["history"]["kinetic_audit_records"]
            if row.get("fired") and row.get("B_pre", 0.0) > 0.999999]
    assert len(near) == 1
    locator = [row for row in _walk(near[0])
               if row.get("mode") == "first_passage_cycle_locator"]
    assert locator and locator[-1]["fired"] is True
    evaluations = sum(int(row["trial_evaluations"]) for row in locator)
    assert evaluations < MAX_LOCATOR_EVALUATIONS

    live = [json.loads(line)
            for line in (root / "high_cycle_live_history.jsonl").read_text().splitlines()
            if line.strip()]
    event_index = expected["start_events"] + 1
    first_passage = [row for row in live
        if row["reason"] == "first_passage"
        and row["stochastic"]["hazard_event_index"] == event_index]
    assert len(first_passage) == 1
    assert first_passage[0]["high_cycle_cache"]["invalidated_reason"] == "first_passage_event"
    continued = [row for row in live
        if row["stochastic"]["hazard_event_index"] == event_index
        and row["reason"] in {"exact_cycle_progress", "validated_dmd_segment", "integrator_return"}
        and row.get("cycles_from_engine_time") is not None
        and row["cycles_from_engine_time"] > expected["event_cycles"]]
    assert continued
    cache = kinetic["high_cycle_cache"]
    assert "invalidated_reason" not in cache
    assert cache["geometry_signature"][0] == event_index
    return {
        "case": name, "event_cycles": event["cycles_post"],
        "completed_Xi": event["threshold_action"],
        "completed_H": event["physical_hazard_action"],
        "locator_evaluations": evaluations,
        "locator_wall_seconds": near[0]["coupled_hazard_wall_seconds"],
        "renewed_Xi": stochastic["hazard_threshold_action"],
        "post_event_mode": continued[-1]["reason"],
    }


def run_case(name, *, family, python, keep_root):
    source = validate_fixture(name)
    temporary = Path(tempfile.mkdtemp(prefix=f"v10230-{name}-locator-gate-"))
    root = temporary / name
    shutil.copytree(FIXTURE_ROOT / name, root)
    values = load_combined_checkpoint(root)
    copied = dict(zip(("outer", "kinetic", "arrays"), values))
    validate_cross_layer(copied["outer"], copied["kinetic"])
    _assert_exact_copy(source, copied)
    expected = CASES[name]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip()
    env = os.environ.copy()
    env.update({
        "PYTHON_BIN": str(python),
        "CONDA_ENV": "arrhenius-sharp-front-v10-codex",
        "CONDA_DEFAULT_ENV": "arrhenius-sharp-front-v10-codex",
        "EXPECTED_BRANCH": "codex/v10.2.30-fatigue-da-dN",
        "EXPECTED_HEAD": head, "FAMILY_JSON": str(family),
        "PARAMETER_OPTION": expected["parameter_option"],
        "TARGET_DELTAK": expected["deltaK"], "TARGET_FRACTION": expected["fraction"],
        "RUN_LABEL": f"{name}_near_threshold_production_gate",
        "TARGET_EXT_UM": "100",
        "CYCLES_MAX": f'{expected["start_cycles"] + 4.0:.17g}',
        "HAZARD_SEED": expected["seed"], "MAX_WALL_SECONDS": "600",
        "OUTROOT": str(root), "V10230_RESTART_CHECKPOINT_DIR": str(root),
        "V10230_HIGH_CYCLE_CHECKPOINT_DIR": str(root),
    })
    started = time.monotonic()
    with (root / "production_gate.log").open("w") as log:
        subprocess.run(
            ["bash", "scripts/run_v10_2_30_weakt_high_cycle_1e12.sh"],
            cwd=REPOSITORY, env=env, stdout=log, stderr=subprocess.STDOUT,
            check=True, timeout=660)
    result = verify_result(name, root, source)
    result["gate_wall_seconds"] = time.monotonic() - started
    if keep_root is not None:
        destination = keep_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(root), destination)
        result["artifacts"] = str(destination)
    shutil.rmtree(temporary, ignore_errors=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures-only", action="store_true")
    parser.add_argument("--case", choices=("dbtt", "peak", "all"), default="all")
    parser.add_argument("--family-json", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--keep-root", type=Path)
    args = parser.parse_args()
    names = list(CASES) if args.case == "all" else [args.case]
    for name in names:
        validate_fixture(name)
    if args.fixtures_only:
        print(json.dumps({"fixtures_valid": names}, indent=2))
        return 0
    if not args.family_json.is_file():
        parser.error(f"missing immutable production family: {args.family_json}")
    results = [run_case(
        name, family=args.family_json.resolve(), python=args.python.resolve(),
        keep_root=args.keep_root.resolve() if args.keep_root else None)
        for name in names]
    print(json.dumps({
        "schema": "v10.2.30_near_threshold_production_gate_v1",
        "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
