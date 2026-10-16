#!/usr/bin/env python3
"""Restartable native information batch for the R10 mechanism search."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.paris_slope_r9 import run_trajectory
from arrhenius_fracture.retained_material_r10 import load_offspring_binding
from scripts.run_r10_current_fracture_panel import run as run_fracture
from scripts.run_r10_global_mechanism_search import EVENT_LENGTH_M, phase_action


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def choose_loads(row: dict, temperature_K: float,
                 targets: tuple[float, float, float] = (1e-9, 1e-7, 1e-5)
                 ) -> tuple[list[float], list[dict]]:
    # Invert the same A0 law on a dense load grid.  The targets span four
    # decades but remain below cooperative saturation; no full-state transfer
    # or empirical attenuation is used to choose them.
    grid = [1.0 + 39.0 * i / 779.0 for i in range(780)]
    refs = []
    for Kmax in grid:
        action = phase_action(row, "cleave", float(Kmax), temperature_K)
        refs.append({"Kmax_MPa_sqrt_m": float(Kmax), "DeltaK_MPa_sqrt_m": 0.9 * float(Kmax),
                     "opening_action_per_cycle": action,
                     "A0_physical_m_per_cycle": EVENT_LENGTH_M * action})
    positions = []
    for target in targets:
        candidates = sorted(range(len(refs)), key=lambda i: abs(
            math.log10(max(refs[i]["A0_physical_m_per_cycle"], 1e-300)) - math.log10(target)))
        positions.append(next(i for i in candidates if i not in positions))
    positions.sort()
    return [refs[i]["Kmax_MPa_sqrt_m"] for i in positions], [refs[i] for i in positions]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--descriptors", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("fracture", "fatigue", "all"), default="all")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=10162026)
    p.add_argument("--maximum-cycles", type=float, default=1e12)
    p.add_argument("--candidate-id", action="append", default=[])
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    registry = list(csv.DictReader(args.registry.open(newline="")))
    descriptors = {r["candidate_id"]: r for r in csv.DictReader(args.descriptors.open(newline=""))}
    requested = set(args.candidate_id)
    assigned = [r for i, r in enumerate(registry)
                if i % args.shard_count == args.shard_index
                and (not requested or r["candidate_id"] in requested)]
    for ordinal, record in enumerate(assigned):
        candidate = record["candidate_id"]
        descriptor = descriptors[candidate]
        mechanism = descriptor["mechanism_acquisition_stratum"]
        temperature = {"OPENING_DIRECTED": 300.0, "MIXED_DIRECTED": 600.0,
                       "PLASTICITY_DIRECTED": 900.0}[mechanism]
        row = json.loads(record["complete_parameter_vector"])
        binding = load_offspring_binding(args.registry, candidate)
        case = args.out / candidate
        case.mkdir(parents=True, exist_ok=True)
        terminal_status = case / "batch_case_status.json"
        if terminal_status.exists() and json.loads(terminal_status.read_text()).get("mode_completed") == args.mode:
            print(json.dumps({"candidate_id": candidate, "stage": "skip", "status": "ALREADY_COMPLETE"}), flush=True)
            continue
        preflight = {
            "schema": "r10.mechanism-balanced-native-batch/1",
            "candidate_id": candidate, "family": record["family"],
            "complete_row_sha256": record["complete_row_sha256"],
            "acquisition_stratum_not_final_classification": mechanism,
            "thermal_acquisition_stratum": descriptor["thermal_acquisition_stratum"],
            "fatigue_temperature_K": temperature,
            "native_backend": "R8_RESTORED_PERSISTENT_REVERSIBLE_CURRENT_BACKEND",
            "same_complete_material_for_fracture_fatigue_and_A0": True,
            "event_and_extension_stops": {"events": 12, "extension_um": 70.0},
            "developed_window_um": [20.0, 60.0],
        }
        atomic_json(case / "preflight.json", preflight)
        if args.mode in ("fracture", "all"):
            for T in sorted(set((300.0, temperature))):
                path = case / f"fracture_T_{int(T):04d}K.json"
                if not path.exists():
                    result = run_fracture(binding, T, 0.005, 80.0, 0.05, 10.0, 2_000_000)
                    atomic_json(path, result)
                else:
                    result = json.loads(path.read_text())
                print(json.dumps({"candidate_id": candidate, "stage": "fracture", "T": T,
                                  "status": result["status"], "Kinit": result["K_init_MPa_sqrt_m"]}), flush=True)
        if args.mode in ("fatigue", "all"):
            targets = ((1e-9, 1e-7, 1e-5) if mechanism == "OPENING_DIRECTED"
                       else (1e-5, 1e-4, 1e-3))
            loads, references = choose_loads(row, temperature, targets)
            if mechanism != "OPENING_DIRECTED":
                loads = list(reversed(loads))
            atomic_json(case / "opening_reference.json", {"temperature_K": temperature,
                        "renewal_reward_conversion_m_per_action": EVENT_LENGTH_M,
                        "A0_target_rates_m_per_cycle": targets,
                        "references": references})
            for Kmax in loads:
                load_case = case / ("fatigue_Kmax_" + format(Kmax, ".9g").replace(".", "p"))
                path = load_case / "result.json"
                if not path.exists():
                    load_case.mkdir(parents=True, exist_ok=True)
                    result = run_trajectory(
                        binding, kernel_family_path=R4_KERNEL_FAMILY,
                        Kmax_MPa_sqrt_m=Kmax,
                        seed=args.seed + args.shard_index,
                        checkpoint_directory=load_case / "checkpoint",
                        target_events=12, target_extension_m=70e-6,
                        maximum_events=40, maximum_cycles=args.maximum_cycles,
                        phase_count=64, temperature_K=temperature,
                        R=0.1, frequency_Hz=1000.0,
                    )
                    atomic_json(path, result)
                else:
                    result = json.loads(path.read_text())
                print(json.dumps({"candidate_id": candidate, "stage": "fatigue", "T": temperature,
                                  "Kmax": Kmax, "status": result["status"],
                                  "events": len(result["events"]),
                                  "extension_um": 1e6 * result["final_extension_m"]}), flush=True)
        atomic_json(case / "batch_case_status.json", {"candidate_id": candidate,
                    "ordinal_in_shard": ordinal, "mode_completed": args.mode})
    atomic_json(args.out / f"shard_{args.shard_index:02d}_status.json", {
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "assigned_candidates": [r["candidate_id"] for r in assigned],
        "status": "COMPLETE", "mode": args.mode,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
