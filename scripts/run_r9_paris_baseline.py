#!/usr/bin/env python3
"""Run restartable R9 amplitude trajectories on the restored current backend."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.archived_fatigue_material_r9 import load_archived_material_binding
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.paris_slope_r9 import run_trajectory


DEFAULT_LOADS = (4.781762498950186, 8.598738351065925, 15.462474735549586)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimized-registry", type=Path, required=True)
    parser.add_argument("--parent-registry", type=Path, required=True)
    parser.add_argument("--candidate-id", default="V3_ceramic_V2_P25_007__V3_1_TRUST1")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--loads", default=",".join(map(str, DEFAULT_LOADS)))
    parser.add_argument("--seed", type=int, default=1720)
    parser.add_argument("--target-events", type=int, default=20)
    parser.add_argument("--target-extension-um", type=float, default=100.0)
    parser.add_argument("--maximum-events", type=int, default=60)
    parser.add_argument("--maximum-cycles", type=float, default=1.0e12)
    parser.add_argument("--phase-count", type=int, default=64)
    args = parser.parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)
    binding = load_archived_material_binding(
        args.optimized_registry, args.parent_registry, candidate_id=args.candidate_id,
    )
    loads = tuple(float(value) for value in args.loads.split(",") if value.strip())
    probe_stress = np.asarray([0.0, 1.0e9, 2.0e9, 5.0e9, 10.0e9, 20.0e9, 30.0e9])
    process_evaluations = {}
    for name, surface in (("opening", binding.manifest.cleavage),
                          ("emission", binding.manifest.emission)):
        barrier = np.asarray(surface.values_eV(probe_stress, 300.0), float)
        derivative = np.gradient(barrier, probe_stress)
        process_evaluations[name] = {
            "stress_Pa": probe_stress.tolist(), "temperature_K": 300.0,
            "barrier_eV": barrier.tolist(),
            "dG_dsigma_eV_per_Pa": derivative.tolist(),
            "raw_rate_s-1": np.asarray(surface.rate(probe_stress, 300.0), float).tolist(),
            "consumer": ("engine.lambda_cleave->manifest.cleavage"
                         if name == "opening" else "persistent_site._emit->manifest.emission"),
        }
    preflight = {
        "schema": "r9.production-preflight/1",
        "source_head": "97aaf1cc2a0963f118deab398a17c6cbf3111acd",
        "source_tree": "e7823788ef1ce8ac8722799b6ff87120e6ea98a6",
        "packaging_parent": "d27954d862840b408e8ba6aed55c0d7e980fd4e3",
        "binding": dict(binding.provenance),
        "material": binding.manifest.as_dict(),
        "archived_surface_contract": {
            "opening_surface": binding.optimized_row["opening_surface"],
            "emission_surface": binding.optimized_row["emission_surface"],
        },
        "process_evaluations": process_evaluations,
        "reduced_config": vars(binding.reduced_config),
        "persistent_sites": vars(binding.persistent_sites),
        "kernel_family": {"path": str(R4_KERNEL_FAMILY.resolve()),
                          "sha256": _sha(R4_KERNEL_FAMILY)},
        "mechanics_provider": "RESTORED_REDUCED_PRESCRIBED_K_WITH_R4_EFFECTIVE_OPENING_KERNEL",
        "local_strength_Pa": 30.0e9,
        "local_strength_source": "R3_R4_INDEPENDENT_SPATIAL_COHESIVE_STRENGTH_CONTRACT",
        "event_strategy": "UNIT_EXPONENTIAL_MARK_THRESHOLD_SCALED_BOUNDED_REWARD",
        "random_owner": "PCG64_SEED_SEQUENCE_[seed,1]_PER_AMPLITUDE",
        "waveform": {"temperature_K": 300.0, "R": 0.1, "frequency_Hz": 1000.0,
                     "phase_count": args.phase_count, "start_phase": "KMAX_COSINE_PHASE_ZERO"},
        "loads_Kmax_MPa_sqrt_m": list(loads),
        "loads_DeltaK_MPa_sqrt_m": [0.9*value for value in loads],
        "acceleration": "PHASE_EXACT_OVERLAP_QUALIFIED_MEAN_BLOCK_STOPS_SHORT_OF_PENDING_EVENT",
        "physical_lengths": {
            "committed_crack": "checkpoint_advance_total_m",
            "continuous_mpz": "micro_advance_total_m",
            "no_double_addition": True,
        },
    }
    (args.out_root/"production_preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True, default=str)+"\n"
    )
    summaries = []
    for load in loads:
        label = f"Kmax_{load:.9g}".replace(".", "p")
        case = args.out_root/label
        result_path = case/"result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
        else:
            case.mkdir(parents=True, exist_ok=True)
            result = run_trajectory(
                binding, kernel_family_path=R4_KERNEL_FAMILY,
                Kmax_MPa_sqrt_m=load, seed=args.seed,
                checkpoint_directory=case/"checkpoint",
                target_events=args.target_events,
                target_extension_m=args.target_extension_um*1.0e-6,
                maximum_events=args.maximum_events,
                maximum_cycles=args.maximum_cycles,
                phase_count=args.phase_count,
            )
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
        summary = {key: result[key] for key in (
            "Kmax_MPa_sqrt_m", "DeltaK_MPa_sqrt_m", "status", "final_cycles",
            "final_extension_m", "final_continuous_mpz_progress_m", "seed",
        )}
        summary["event_count"] = len(result["events"])
        summary["result_path"] = str(result_path.resolve())
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    (args.out_root/"campaign_summary.json").write_text(
        json.dumps({"preflight": str((args.out_root/'production_preflight.json').resolve()),
                    "cases": summaries}, indent=2, sort_keys=True)+"\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
