#!/usr/bin/env python3
"""Run a matched baseline or minimal-reversible v9.14 explicit fatigue case."""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V914 = Path(
    os.environ.get(
        "V914_ROOT",
        "/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search",
    )
)
# The qualified v9.14 constitutive package is external to this driver repo.
# Put it ahead of the driver package before importing arrhenius_fracture.
if str(DEFAULT_V914) not in sys.path:
    sys.path.insert(0, str(DEFAULT_V914))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from arrhenius_fracture.emergent_gnd_campaign_v913 import candidate_from_registry_row
from arrhenius_fracture.endurance_knee_v914 import physics_for_row
from arrhenius_fracture.emergent_gnd_types_v913 import CommonPhysics
from arrhenius_fracture import fatigue_v914 as base

from v1032_explicit_cycle_lcf import run_explicit_cycle_fatigue as run_baseline_explicit
from v914_minimal_reversible_explicit import run_minimal_reversible_explicit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--deltaK", type=float, required=True)
    parser.add_argument("--model", choices=("baseline", "reversible"), required=True)
    parser.add_argument("--R", type=float, default=0.1)
    parser.add_argument("--frequency-Hz", type=float, default=1000.0)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--phase-steps", type=int, default=32)
    parser.add_argument("--target-um", type=float, default=100.0)
    parser.add_argument("--maximum-cycles", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1720)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--checkpoint-cycle-interval", type=int, default=10)
    parser.add_argument("--state-history-cycle-interval", type=int, default=10)
    parser.add_argument("--expected-head", default=None)
    return parser.parse_args()


def load_physics(path: Path) -> CommonPhysics:
    values = json.loads(path.read_text())
    values = values.get("common_physics", values)
    names = {field.name for field in fields(CommonPhysics)}
    selected = {key: value for key, value in values.items() if key in names}
    tuple_names = (
        "emission_signs",
        "emission_schmid_factors",
        "shielding_orientation_factors",
        "forest_interaction_matrix",
        "gnd_stress_projection_matrix",
        "activation_to_line_content_per_system",
        "emission_geometry_extension_m",
        "emission_geometry_factors",
    )
    for name in tuple_names:
        if name in selected:
            selected[name] = tuple(
                tuple(item) if isinstance(item, list) else item
                for item in selected[name]
            )
    result = CommonPhysics(**selected)
    result.validate()
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if args.expected_head and head != args.expected_head:
        raise SystemExit(
            f"HEAD mismatch: expected {args.expected_head}, found {head}"
        )
    if args.expected_head and dirty:
        raise SystemExit("authoritative launch requires a clean worktree")

    rows = list(csv.DictReader(args.registry.open()))
    row = next(
        (entry for entry in rows if entry["candidate_id"] == args.candidate),
        None,
    )
    if row is None:
        raise ValueError(f"candidate missing from registry: {args.candidate}")
    candidate = candidate_from_registry_row(row)
    physics = physics_for_row(load_physics(args.physics), row)
    loading = base.FatigueLoading(
        args.deltaK,
        R=args.R,
        frequency_Hz=args.frequency_Hz,
        temperature_K=args.temperature_K,
        phase_steps=args.phase_steps,
    )
    numerics = base.FatigueNumerics(
        maximum_cycles=float(args.maximum_cycles),
        target_extension_m=args.target_um * 1e-6,
        maximum_explicit_cycles=max(args.maximum_cycles + 2, 4096),
    )
    checkpoint = args.out / "live_checkpoint.json"

    common = dict(
        seed=args.seed,
        numerics=numerics,
        checkpoint_path=checkpoint,
        restart_from=checkpoint if args.restart else None,
        maximum_physical_cycles=args.maximum_cycles,
        checkpoint_each_phase=False,
        checkpoint_cycle_interval=args.checkpoint_cycle_interval,
        state_history_cycle_interval=args.state_history_cycle_interval,
    )
    if args.model == "baseline":
        result = run_baseline_explicit(candidate, physics, loading, **common)
    else:
        result = run_minimal_reversible_explicit(
            candidate, physics, loading, **common
        )

    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    events = result.get("events", [])
    history = result.get("state_history", [])
    for name, records in (
        ("event_history.csv", events),
        ("state_history.csv", history),
    ):
        if records:
            with (args.out / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)

    contract = {
        "schema": "v914_minimal_reversible_case_contract_v1",
        "model": args.model,
        "candidate": args.candidate,
        "deltaK_MPa_sqrt_m": args.deltaK,
        "R": args.R,
        "frequency_Hz": args.frequency_Hz,
        "temperature_K": args.temperature_K,
        "phase_steps": args.phase_steps,
        "target_um": args.target_um,
        "maximum_cycles": args.maximum_cycles,
        "seed": args.seed,
        "repository": str(ROOT),
        "repository_branch": branch,
        "repository_head": head,
        "repository_clean": not bool(dirty),
        "v914_root": str(DEFAULT_V914),
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256(args.registry),
        "physics": str(args.physics.resolve()),
        "physics_sha256": sha256(args.physics),
        "cleavage_physics_changed": False,
        "non_schmid_changed": False,
        "empirical_recovery_fraction": False,
        "minimal_reversible_mobile_return": args.model == "reversible",
    }
    (args.out / "run_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    print(
        args.candidate,
        args.model,
        result["status"],
        result["final_cycles"],
        result["final_extension_m"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
