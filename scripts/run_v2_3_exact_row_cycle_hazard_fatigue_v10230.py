#!/usr/bin/env python3
"""Run the three frozen V2.3 exact-row reduced fatigue trajectories."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.select_v2_3_exact_row_cycle_hazard_loads_v10230 import (
    CANDIDATE_ID, OUT, exact_candidate_and_physics, load_external_fatigue,
)


FROZEN_HEAD = "0b986300bffb12bb2aa020541e05dc1c8ecde66a"
FROZEN_EXECUTION_TAG = "refs/tags/v10230-v2-3-exact-row-execution"
RUN_ROOT = ROOT / "runs/v2_3_exact_row_cycle_hazard_fatigue"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def run_case(fatigue, candidate, physics, load: dict, *, explicit: bool = False) -> dict:
    Kmax = float(load["Kmax_MPa_sqrt_m"])
    label = f"Kmax_{Kmax:g}".replace(".", "p") + ("_explicit_overlap" if explicit else "")
    case_dir = RUN_ROOT / label
    checkpoint = case_dir / "live_checkpoint.json"
    result_path = case_dir / "result.json"
    loading = fatigue.FatigueLoading(
        deltaK_MPa_sqrt_m=0.9 * Kmax, R=0.1, frequency_Hz=1000.0,
        temperature_K=300.0, phase_steps=64,
    )
    kwargs = dict(
        maximum_cycles=1e8, target_extension_m=25e-6,
        base_event_length_m=5e-6, event_length_factor_min=0.5,
        event_length_factor_max=4.0, checkpoint_wait_cycles=1e6,
    )
    if explicit:
        kwargs.update(
            maximum_explicit_cycles=1_000_000,
            periodic_confirmations=1_000_000_000,
            dmd_minimum_project_cycles=1_000_000_000,
        )
    numerics = fatigue.FatigueNumerics(**kwargs)
    if result_path.is_file():
        saved = json.loads(result_path.read_text())
        if saved["candidate_id"] != CANDIDATE_ID or saved["loading"] != {
            "R": 0.1, "deltaK_MPa_sqrt_m": 0.9 * Kmax, "frequency_Hz": 1000.0,
            "phase_steps": 64, "temperature_K": 300.0,
        }:
            raise RuntimeError(f"refuse mismatched completed result at {result_path}")
        return saved
    result = fatigue.run_cyclic_fatigue(
        candidate, physics, loading, seed=1720, numerics=numerics,
        checkpoint_path=checkpoint,
        restart_from=checkpoint if checkpoint.is_file() else None,
    )
    payload = result.as_dict()
    payload.update({
        "integration_role": "EXPLICIT_CYCLE_OVERLAP" if explicit else "QUALIFIED_ACCELERATED_OR_EXACT",
        "selected_load_record": load,
        "frozen_selection_head": FROZEN_HEAD,
        "raw_run_output_not_added_to_git": True,
    })
    atomic_json(result_path, payload)
    print(json.dumps({"case": label, "status": result.status,
                      "events": len(result.events), "cycles": result.final_cycles,
                      "extension_m": result.final_extension_m}, sort_keys=True), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("production", "overlap", "all"), default="all")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    expected_head = subprocess.check_output(
        ["git", "rev-parse", FROZEN_EXECUTION_TAG], cwd=ROOT, text=True
    ).strip()
    if head != expected_head:
        raise SystemExit("physical launch requires HEAD at the immutable execution tag")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", FROZEN_HEAD, head], cwd=ROOT
    ).returncode != 0:
        raise SystemExit("execution tag does not contain the frozen load selection")
    if subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT, text=True).strip():
        raise SystemExit("physical launch requires a clean worktree")
    fatigue = load_external_fatigue()
    candidate, physics, _ = exact_candidate_and_physics(fatigue)
    selection = json.loads((OUT / "selected_exact_row_fatigue_loads.json").read_text())
    loads = selection["selected_loads_in_ascending_Kmax"]
    results = []
    if args.mode in {"production", "all"}:
        for load in loads:
            results.append(run_case(fatigue, candidate, physics, load))
    if args.mode in {"overlap", "all"}:
        results.append(run_case(fatigue, candidate, physics, loads[1], explicit=True))
    atomic_json(RUN_ROOT / "run_index.json", {
        "schema": "v10.2.30_v2_3_exact_row_fatigue_run_index_v1",
        "results": [{"Kmax_MPa_sqrt_m": x["loading"]["deltaK_MPa_sqrt_m"] / 0.9,
                     "integration_role": x["integration_role"], "status": x["status"],
                     "events": len(x["events"]), "cycles": x["final_cycles"],
                     "extension_m": x["final_extension_m"]} for x in results],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
