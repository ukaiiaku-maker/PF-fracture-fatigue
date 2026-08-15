#!/usr/bin/env python3
"""Run isolated accelerated/explicit v10.2.32 LCF comparison cases."""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess

from arrhenius_fracture.emergent_gnd_campaign_v913 import candidate_from_registry_row
from arrhenius_fracture.endurance_knee_v914 import physics_for_row
from arrhenius_fracture.emergent_gnd_types_v913 import CommonPhysics
from arrhenius_fracture import fatigue_v914 as base
from v1032_explicit_cycle_lcf import run_explicit_cycle_fatigue


def load_physics(path: Path) -> CommonPhysics:
    values = json.loads(path.read_text()); values = values.get("common_physics", values)
    names = {f.name for f in fields(CommonPhysics)}; selected = {k: v for k, v in values.items() if k in names}
    for name in ("emission_signs", "emission_schmid_factors", "shielding_orientation_factors",
                 "forest_interaction_matrix", "gnd_stress_projection_matrix",
                 "activation_to_line_content_per_system", "emission_geometry_extension_m",
                 "emission_geometry_factors"):
        if name in selected: selected[name] = tuple(tuple(x) if isinstance(x, list) else x for x in selected[name])
    result = CommonPhysics(**selected); result.validate(); return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, required=True); ap.add_argument("--physics", type=Path, required=True)
    ap.add_argument("--candidate", required=True); ap.add_argument("--deltaK", type=float, required=True)
    ap.add_argument("--mode", "--cycle-integration-mode", dest="mode",
                    choices=["accelerated", "explicit"], required=True)
    ap.add_argument("--phase-steps", type=int, default=32); ap.add_argument("--target-um", type=float, default=100)
    ap.add_argument("--maximum-cycles", type=float, default=50); ap.add_argument("--seed", type=int, default=1720)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--restart", action="store_true")
    ap.add_argument("--checkpoint-every-cycle", action="store_true",
                    help="checkpoint every completed cycle and committed event instead of every phase")
    ap.add_argument("--pause-after-phases", type=int, default=0)
    ap.add_argument("--normalized-f", type=float, default=None)
    ap.add_argument("--expected-head", default=None)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repository, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    worktree = subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True)
    if args.expected_head and head != args.expected_head:
        raise SystemExit(f"HEAD mismatch: expected {args.expected_head}, found {head}")
    if args.expected_head and worktree.strip():
        raise SystemExit("authoritative launch requires a clean worktree")
    rows = list(csv.DictReader(args.registry.open())); row = next((r for r in rows if r["candidate_id"] == args.candidate), None)
    if row is None: raise ValueError("candidate missing from registry")
    candidate = candidate_from_registry_row(row); physics = physics_for_row(load_physics(args.physics), row)
    loading = base.FatigueLoading(args.deltaK, R=.1, frequency_Hz=1000, temperature_K=300, phase_steps=args.phase_steps)
    numerics = base.FatigueNumerics(maximum_cycles=args.maximum_cycles, target_extension_m=args.target_um*1e-6,
                                    maximum_explicit_cycles=max(int(args.maximum_cycles)+2, 4096))
    checkpoint = args.out / "live_checkpoint.json"
    if args.mode == "explicit":
        result = run_explicit_cycle_fatigue(candidate, physics, loading, seed=args.seed, numerics=numerics,
            checkpoint_path=checkpoint, restart_from=checkpoint if args.restart else None,
            maximum_physical_cycles=int(args.maximum_cycles),
            checkpoint_each_phase=not args.checkpoint_every_cycle,
            checkpoint_cycle_interval=1 if args.checkpoint_every_cycle else None,
            pause_after_phase_advances=args.pause_after_phases or None)
        events, history = result["events"], result["state_history"]
    else:
        r = base.run_cyclic_fatigue(candidate, physics, loading, seed=args.seed, numerics=numerics,
            checkpoint_path=checkpoint, restart_from=checkpoint if args.restart else None)
        result = r.as_dict(); result["mode"] = "accelerated"
        events = result["events"]; history = []
    (args.out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, records in (("event_history.csv", events), ("state_history.csv", history)):
        if records:
            with (args.out / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=records[0].keys()); w.writeheader(); w.writerows(records)
    contract = {"schema": "v10.2.32_lcf_case_contract_v1", "mode": args.mode,
        "candidate": args.candidate, "deltaK_MPa_sqrt_m": args.deltaK, "phase_steps": args.phase_steps,
        "target_um": args.target_um, "maximum_cycles": args.maximum_cycles, "seed": args.seed,
        "checkpoint_cadence": "cycle_and_event" if args.checkpoint_every_cycle else "phase_and_event",
        "normalized_f": args.normalized_f,
        "repository": str(repository), "repository_branch": branch,
        "repository_head": head, "repository_clean": not bool(worktree.strip()),
        "registry": str(args.registry.resolve()), "registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
        "physics": str(args.physics.resolve()), "physics_sha256": hashlib.sha256(args.physics.read_bytes()).hexdigest()}
    (args.out / "run_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(args.candidate, args.mode, result["status"], result["final_cycles"], result["final_extension_m"])
    return 0


if __name__ == "__main__": raise SystemExit(main())
