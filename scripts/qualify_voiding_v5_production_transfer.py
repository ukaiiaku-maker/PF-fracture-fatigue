#!/usr/bin/env python3
"""Actual fixed-crack production transfer and enforced unavailable-first-passage policy.

Each resolution independently executes the real lifecycle and ligament trial.
An unavailable connection/reference is a recorded failure, never a surrogate
static source or a fabricated downstream event.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from arrhenius_fracture.checkpoint_v11 import write_checkpoint, restore_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
from arrhenius_fracture.voiding_production_v5 import (
    deterministic_trajectory, ligament_transaction, downstream_front_transaction,
    cavity_boundary_tensor, directional_clock_rates,
)
from arrhenius_fracture.finalization_v3_schema import SCIENTIFIC_ACCEPTANCE_TOLERANCES as LIMITS

LEVELS = ((32, 12), (64, 24), (128, 48), (512, 192))
# Same prospectively fixed laboratory crack used by the accepted offset study.
PATH = ((0., 0.), (0.0005725993004046688, 0.))


def canonical(value):
    if isinstance(value, np.ndarray): return canonical(value.tolist())
    if isinstance(value, np.generic): return canonical(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0 else "-infinity" if value < 0 else "NaN"
    if isinstance(value, dict): return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [canonical(v) for v in value]
    return value


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(canonical(payload), indent=2, sort_keys=True, allow_nan=False)+"\n")


def candidate_measurements(state):
    from arrhenius_fracture.closure_production_evidence import candidate_measurements as measured
    return measured(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("output", type=Path)
    parser.add_argument('--compress-checkpoints',action='store_true')
    args = parser.parse_args(); out = args.output
    def checkpoint(state,path):
        return write_checkpoint(state,path,compression='gzip' if args.compress_checkpoints else None)
    if out.exists() and any(out.iterdir()): raise ValueError("refusing to overwrite evidence")
    if subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip():
        raise RuntimeError("production transfer requires a clean committed implementation")
    head = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    rows = []
    for segments, layers in LEVELS:
        key = f"production:{segments}:{layers}"
        print("Executing actual production lifecycle "+key, flush=True)
        row = {"case_id": key, "executed_code_sha": head,
               "input_configuration": {"crack_path_m": PATH, "cavity_center_m": (7e-4, 0.),
                    "boundary_segments": segments, "radial_layers": layers, "seed": 3621},
               "connection_executed": False, "first_passage_qualified": False}
        trace = []
        try:
            pre, history = deterministic_trajectory(stop_before_ligament=True, crack_path_m=PATH,
                boundary_segments=segments, radial_layers=layers, state_trace=trace)
            checkpoint(pre, out/"checkpoints"/(key.replace(":", "_")+"_pre.json"))
            connected, result = ligament_transaction(pre)
            row.update({"connection_executed": result.accepted, "initial_state_fingerprint": fingerprint(pre),
                        "connected_state_fingerprint": fingerprint(connected),
                        "candidates": candidate_measurements(connected),
                        "actual_history": history})
            checkpoint_path = out/"checkpoints"/(key.replace(":", "_")+"_connected.json")
            checkpoint(connected, checkpoint_path)
            restored = restore_checkpoint(checkpoint_path)
            guarded, trial, operations, audit = downstream_front_transaction(connected)
            replay, replay_trial, _, replay_audit = downstream_front_transaction(restored)
            row.update({"guarded_state_fingerprint": fingerprint(guarded),
                        "guard_audit": audit, "guard_operations": operations,
                        "guard_preserves_complete_state": fingerprint(guarded) == fingerprint(connected),
                        "checkpoint_restart_exact": fingerprint(replay) == fingerprint(guarded) and replay_audit == audit,
                        "child_created": len(guarded.crack_network.branches) != len(connected.crack_network.branches),
                        "active_tip_ids": guarded.crack_network.active_tip_ids,
                        "event_transaction_created": trial is not None or replay_trial is not None})
            from arrhenius_fracture.closure_production_evidence import REFINEMENT_CASES, execute_refinement_peer
            if key in REFINEMENT_CASES:
                print("Executing bounded source refinement peer "+key,flush=True)
                row["source_refinement_peer"] = execute_refinement_peer(connected,REFINEMENT_CASES[key])
        except Exception as error:
            row["failure"] = {"type": type(error).__name__, "message": str(error),
                              "last_accepted_stage": trace[-1][0] if trace else None}
            if trace:
                checkpoint(trace[-1][1], out/"checkpoints"/(key.replace(":", "_")+"_last_accepted.json"))
        rows.append(row)
        write_json(out/(key.replace(":", "_")+".json"), row)
    from arrhenius_fracture.closure_production_evidence import transfer_comparisons, SCHEMA as TRANSFER_SCHEMA
    comparisons = transfer_comparisons(rows)
    write_json(out/"transfer_manifest.json", {"schema": TRANSFER_SCHEMA, "executed_code_sha": head,
        "physical_registry": LEVELS, "comparisons": comparisons, "production_resolution_cavity_tensor_qualified": False,
        "static_traction_reference_is_not_matched_production": True,
        "policy": "C_FIRST_PASSAGE_UNAVAILABLE_UNTIL_QUALIFIED",
        "frozen_tensor_limit": LIMITS["tensor_probe_relative"],
        "frozen_rate_barrier_mesh_transfer_limits": "NOT_DEFINED_IN_ACCEPTED_REGISTRY"})
    write_json(out/"sha256_manifest.json", {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*")) if p.is_file()})


if __name__ == "__main__": main()
