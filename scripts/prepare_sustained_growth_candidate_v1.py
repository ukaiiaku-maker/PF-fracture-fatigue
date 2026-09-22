#!/usr/bin/env python3
"""Create the selected candidate before its first affected emission history."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint


SOURCE = ROOT / "runs/managed_natural_void_v1_20260919/output/restart_002100ms.json"
OUT = ROOT / "runs/managed_sustained_growth_v1_20260920/output"
ROW_ID = "hypothesis.sustained-growth-v1:krec=0.01:depletion=0.01"


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"candidate output already exists: {OUT}")
    state = restore_checkpoint(SOURCE)
    actual_time = float(state.junction_process_state.get(
        "production_time_s", state.vacancy_transport_state_v2.transport_time_s))
    if actual_time != 2.09999:
        raise RuntimeError(f"unexpected reusable-prefix time: {actual_time!r}")
    if state.void_state.embryo_growth_states:
        raise RuntimeError("candidate must be initialized before affected emission history")
    configuration = state.void_state.embryo_growth_configuration
    old_row = configuration.cavity_emission_parameter_row
    new_row = replace(
        old_row, row_id=ROW_ID, recovery_rate_s=0.01,
        depletion_per_loop=0.01)
    state = replace(state, void_state=replace(
        state.void_state,
        embryo_growth_configuration=replace(
            configuration, cavity_emission_parameter_row=new_row)))

    OUT.mkdir(parents=True)
    checkpoint = OUT / "restart_002100ms.json"
    manifest = write_checkpoint(state, checkpoint, compression="gzip")
    declared = {
        "schema": "unified-2d.sustained-growth-candidate/1",
        "classification": "PROSPECTIVE_UNCALIBRATED_PARAMETER_HYPOTHESIS",
        "trajectory_time_origin_s": 0.0,
        "observation_horizon_s": 10000.0,
        "reused_prefix_checkpoint": str(SOURCE),
        "reused_prefix_actual_time_s": actual_time,
        "prefix_independence": "NO_EMBRYO_OR_CAVITY_EMISSION_OWNER_EXISTS",
        "parameter_row": asdict(new_row),
        "changed_fields": {
            "recovery_rate_s": [old_row.recovery_rate_s, new_row.recovery_rate_s],
            "depletion_per_loop": [old_row.depletion_per_loop,
                                   new_row.depletion_per_loop],
            "row_id": [old_row.row_id, new_row.row_id],
        },
        "unchanged": [
            "birth_parameters", "fracture_and_void_barriers",
            "elasticity_and_plasticity", "burgers_vector", "loop_projection",
            "vacancy_energies", "temperature", "loading", "stochastic_state",
        ],
        "initial_accepted_state_fingerprint": complete_accepted_state_fingerprint(state),
        "checkpoint_state_file": manifest["state_file"],
    }
    dump(OUT/"declared_inputs.json", declared)
    dump(OUT/"progress.json", {
        "accepted_time_s": actual_time,
        "latest_restart": checkpoint.name,
        "latest_restart_state_file": manifest["state_file"],
        "history": [], "adaptive_step_s": 0.1,
        "stop_reason": "CANDIDATE_INITIALIZED_BEFORE_AFFECTED_EMISSION",
    })
    print(OUT)
    print("ACTUAL_PREFIX_TIME_S", actual_time)
    print("STATE", complete_accepted_state_fingerprint(state))


if __name__ == "__main__":
    main()
