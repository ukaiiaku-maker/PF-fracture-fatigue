#!/usr/bin/env python3
"""Launch fresh, prospective v9.14 v7 model-selection trajectories.

The screen preserves two distinct immutable provenance families: canonical
fracture rows use the qualified calibrated-transport physics, while A--D use
their original endurance-knee common physics and row-level physics overrides.
No checkpoint is ever supplied as a restart source.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
V914 = Path(os.environ.get(
    "V914_ROOT",
    "/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search",
))
CANONICAL_REGISTRY = ROOT / "arrhenius_fracture/data/materials/v10_2_27_paper_four_class_registry.csv"
ABCD_REGISTRY = ROOT / "arrhenius_fracture/data/materials/v10_2_31_endurance_knee_ABCD_registry.csv"
CANONICAL_PHYSICS = ROOT / "artifacts/v914_calibrated_v7_transport_extension/calibrated_v7_transport_extension.json"
ABCD_PHYSICS = V914 / "runs/v9_13_v10222_rcurve_alpha0p95_all50_v1/v9_13_R_curve_calibrated_common_physics.json"
MODELS = (
    ("Peak-T", "v913_zeroD_sobol_0242980", CANONICAL_REGISTRY, CANONICAL_PHYSICS, "CAL-TRANS-REV"),
    ("DBTT", "v913_zeroD_sobol_0202500", CANONICAL_REGISTRY, CANONICAL_PHYSICS, "CAL-TRANS-REV"),
    ("weak-T", "v913_zeroD_sobol_0129902", CANONICAL_REGISTRY, CANONICAL_PHYSICS, "CAL-TRANS-REV"),
    ("ceramic-like", "v913_zeroD_sobol_0077080", CANONICAL_REGISTRY, CANONICAL_PHYSICS, "CAL-TRANS-REV"),
    ("A", "v914_endurance_knee_0462", ABCD_REGISTRY, ABCD_PHYSICS, "LEGACY1D-V7"),
    ("B", "v914_endurance_knee_0658", ABCD_REGISTRY, ABCD_PHYSICS, "LEGACY1D-V7"),
    ("C", "v914_endurance_knee_0554", ABCD_REGISTRY, ABCD_PHYSICS, "LEGACY1D-V7"),
    ("D", "v914_endurance_knee_0133", ABCD_REGISTRY, ABCD_PHYSICS, "LEGACY1D-V7"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def cases() -> list[dict]:
    output = []
    for family, candidate, registry, physics, label in MODELS:
        for R in (0.1, -0.5):
            for kmax in (12.0, 15.0, 18.0, 21.0):
                output.append({
                    "family": family, "candidate": candidate,
                    "registry": str(registry.resolve()), "physics": str(physics.resolve()),
                    "configuration_label": label, "R": R, "Kmax_MPa_sqrt_m": kmax,
                    "deltaK_MPa_sqrt_m": kmax * (1.0 - R),
                })
    return output


def model_provenance() -> list[dict]:
    """Return exact source rows, hashes, barriers, and row-level physics."""
    records = []
    for family, candidate, registry, physics, label in MODELS:
        row = next(r for r in csv.DictReader(registry.open()) if r["candidate_id"] == candidate)
        row_sha = hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        records.append({
            "family": family, "candidate_id": candidate,
            "source_registry": str(registry.resolve()), "source_registry_sha256": sha256(registry),
            "source_row_sha256": row_sha, "common_physics_input": str(physics.resolve()),
            "common_physics_sha256": sha256(physics), "configuration_label": label,
            "barrier_parameters": {key: row[key] for key in row if any(token in key for token in (
                "G00", "gT_", "sigc0", "sT_", "H0_eV", "activation_entropy", "exp_a", "exp_n", "floor_frac"
            ))},
            "model_specific_physics_overrides": {
                key: value for key, value in row.items() if key.startswith("physics__") and value != ""
            },
            "source_transport_settings": {key: row.get(key, "") for key in (
                "rho_source0_m2", "source_sites_per_system", "source_recovery_rate_s",
                "retained_recovery_rate_s", "source_refresh_length_um",
                "legacy_source_sites_active", "legacy_source_refresh_active",
                "explicit_recovery_active", "encounter_efficiency",
            )},
            "exact_registry_row": row,
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs/v914_model_selection_screen_v1")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--maximum-physical-cycles", type=int, default=10**10)
    parser.add_argument("--target-extension-um", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=1720)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.worker_count < 1 or not 0 <= args.worker_index < args.worker_count:
        raise SystemExit("worker index must lie in [0, worker count)")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty and not args.dry_run:
        raise SystemExit("production screen requires a clean worktree")
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "v914_model_selection_screen_manifest_v1",
        "repository_head": head, "fresh_uninterrupted_required": True,
        "restart_supported": False, "temperature_K": 300.0, "frequency_Hz": 1000.0,
        "seed": args.seed, "n_bins": 128,
        "maximum_physical_cycles": args.maximum_physical_cycles,
        "target_extension_um": args.target_extension_um,
        "registries": {str(p.resolve()): sha256(p) for p in (CANONICAL_REGISTRY, ABCD_REGISTRY)},
        "physics_inputs": {str(p.resolve()): sha256(p) for p in (CANONICAL_PHYSICS, ABCD_PHYSICS)},
        "models": model_provenance(),
        "cases": cases(),
    }
    (args.out_root / "screen_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (args.out_root / "screen_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest["cases"][0]))
        writer.writeheader(); writer.writerows(manifest["cases"])
    selected = [c for index, c in enumerate(manifest["cases"]) if index % args.worker_count == args.worker_index]
    for sequence, case in enumerate(selected, 1):
        run_dir = args.out_root / (
            f"{case['family'].replace('-', '_')}_{case['candidate'].split('_')[-1]}"
            f"/R{slug(case['R'])}_Kmax{slug(case['Kmax_MPa_sqrt_m'])}"
        )
        result_path = run_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if int(result.get("restart_count", -1)) == 0 and result.get("status") in {
                "growth_target_reached", "cycle_horizon_reached"
            }:
                print(f"SKIP accepted fresh terminal {run_dir}", flush=True)
                continue
            raise SystemExit(f"refusing to overwrite non-accepted prior attempt: {run_dir}")
        if (run_dir / "checkpoint.json").exists():
            raise SystemExit(f"refusing to resume interrupted attempt: {run_dir}")
        command = [
            sys.executable, str(ROOT / "scripts/run_v914_v7_vhcf_event_to_event.py"),
            "--registry", case["registry"], "--physics", case["physics"],
            "--candidate", case["candidate"], "--deltaK", str(case["deltaK_MPa_sqrt_m"]),
            "--R", str(case["R"]), "--return-control", "REV",
            "--configuration-label", case["configuration_label"],
            "--frequency-Hz", "1000", "--temperature-K", "300", "--seed", str(args.seed),
            "--n-bins", "128", "--maximum-physical-cycles", str(args.maximum_physical_cycles),
            "--target-extension-um", str(args.target_extension_um),
            "--expected-head", head, "--out", str(run_dir),
        ]
        print(f"WORKER {args.worker_index} CASE {sequence}/{len(selected)}: {' '.join(command)}", flush=True)
        if not args.dry_run:
            run_dir.mkdir(parents=True, exist_ok=False)
            subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
