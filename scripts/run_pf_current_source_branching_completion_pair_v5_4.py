#!/usr/bin/env python3
"""Sealed V5.4 completion-pair launcher; dry-run is the default.

Dry-run verifies and replays the family migration in memory but creates no PF
output directory, starts no worker, and performs no accepted state update.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


EXECUTION_COMMIT = "fb6516bb6a9f770fee899d382a902cf0bdf1e701"
EXECUTION_TREE = "7a78506025586dd48337fbccddd63358adc3729f"
EXECUTION_BASE_COMMIT = "e2aff736afe0e1d2d1b600c25743de317a71c7ba"
EXECUTION_BASE_TREE = "14a953037a44a25288c41d317a0ae0c7f36d22bc"
FAMILY_SHA = "423bc3232326b8ccc3ffcca0aa6b5363c67bad2c64debee49925bf1e6413e8cb"
SOURCE_FAMILY_SHA = "b109a2fd6fc393fc986b1f15d6edd7c37366d84c111710ea70bcaba75f426847"
FAMILY_PHYSICS = "e0bc48eb8c3f5526877a21f8551e500046a614df8081693f340084fb73400302"
MECHANICAL_SHA = "3ccc690c103f96da8eca7ee4bc67272653bd888b402bb0878c1b1f6eb9dee6f9"
SOURCE_CHECKPOINT_SHA = {
    "control_max1": "afad328062a2000ea822fd567541ec2619144b2bd182c974ca4bf820482d0aac",
    "enabled_max2": "9928c6ef6e52aa0b01fce70cc82379624faac949c35c3b041e8c6af8cc6e594b",
}
PYTHON = Path(
    "/opt/homebrew/Caskroom/miniconda/base/envs/"
    "arrhenius-sharp-front-v10-codex/bin/python"
)
ENTRY = "arrhenius_fracture.sharp_front_current_source_branching_audited"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()


def stable_json_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def heavy_process_inventory() -> dict:
    completed = subprocess.run(
        ("ps", "-axo", "pid=,ppid=,etime=,command="), text=True,
        capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("cannot inventory heavy processes: " + completed.stderr.strip())
    markers = ("sharp_front", "run_pf_", "dolfin", "fenics", "arrhenius_fracture")
    rows = []
    for line in completed.stdout.splitlines():
        if any(marker in line.lower() for marker in markers) and str(os.getpid()) not in line.split()[:2]:
            rows.append(line.strip())
    return {"command": "ps -axo pid=,ppid=,etime=,command=", "matching_processes": rows}


def acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    os.write(descriptor, f"pid={os.getpid()} time={time.time():.17g}\n".encode())
    return descriptor


def require(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"missing or changed {label}: {path}")


def checkpoint_record(path: Path, role: str, maximum_fronts: int) -> dict:
    manifest = json.loads(path.read_text())
    state = path.with_name(manifest["state_file"])
    if sha256(state) != manifest["state_sha256"]:
        raise RuntimeError(f"migrated checkpoint state mismatch: {path}")
    migration = manifest.get("restart_family_migration_provenance") or {}
    required = {
        "qualification": "PASS", "migration_count": 1,
        "source_family_sha256": (
            "b109a2fd6fc393fc986b1f15d6edd7c37366d84c111710ea70bcaba75f426847"
        ),
        "target_family_sha256": FAMILY_SHA,
        "target_family_physics_fingerprint": FAMILY_PHYSICS,
        "engine_mpz_same_bound_family_object": True,
        "old_family_reachable_after_migration": False,
        "physical_or_stochastic_advance_performed": False,
    }
    if any(migration.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"invalid restart-family migration record: {path}")
    if migration.get("process_state_sha256_before") != migration.get("process_state_sha256_after"):
        raise RuntimeError("migrated checkpoint process state is not exact")
    if migration.get("rng_threshold_sha256_before") != migration.get("rng_threshold_sha256_after"):
        raise RuntimeError("migrated checkpoint RNG/threshold state is not exact")
    return {
        "role": role, "maximum_fronts": maximum_fronts,
        "manifest": str(path.resolve()), "manifest_sha256": sha256(path),
        "state": str(state.resolve()), "state_sha256": sha256(state),
        "migration": migration,
    }


def command(*, output: Path, checkpoint: Path, family: Path, maximum_fronts: int) -> list[str]:
    return [
        str(PYTHON), "-u", "-m", ENTRY,
        "--current-source-branching-capability",
        "--maximum-fronts", str(maximum_fronts),
        "--v11-restart-checkpoint", str(checkpoint),
        "--signed-kernel-family", str(family),
        "--mode", "2d",
        "--parameter-option", "v913_paper_weakT01_0129902_persistent_sites",
        "--temperatures", "700", "--steps", "2000000",
        "--nx", "36", "--ny", "72", "--dU", "2e-7", "--dt", "8.4",
        "--n-stagger", "2", "--tip-h-fine", "1e-6", "--tip-ratio", "1.2",
        "--da-phys", "5e-6", "--target-crack-extension-um", "300",
        "--mpz-length-um", "50", "--mpz-n-bins", "80",
        "--front-state-model", "moving_pz", "--tip-source-model", "continuum",
        "--tip-kinetics-mode", "moving_velocity", "--bulk-plasticity-mode", "tip_only",
        "--directional-j-mode", "root_signed", "--tip-plasticity",
        "--active-shielding", "--signed-active-shielding",
        "--mobile-shield-fraction", "0", "--no-wake-shielding",
        "--crystal-aniso", "--crystal-compete", "--crystal-theta-deg", "40",
        "--crystal-material", "w", "--j-decomposition", "cluster",
        "--crack-backend", "sharp_wake", "--adaptive-events",
        "--adaptive-event-target", "0.15", "--print-every", "200",
        "--save-snapshots", "0", "--no-plots", "--out", str(output),
    ]


def environment(execution: Path, family: Path, mechanical: Path, cache: Path) -> dict[str, str]:
    return {
        "PATH": f"{PYTHON.parent}:/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": "/private/tmp", "LC_ALL": "C", "LANG": "C",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(execution),
        "CONDA_ENV": "arrhenius-sharp-front-v10-codex",
        "CONDA_DEFAULT_ENV": "arrhenius-sharp-front-v10-codex",
        "CONDA_PREFIX": str(PYTHON.parent.parent), "PARAMETER_CAMPAIGN": "1",
        "CLEAVAGE_HAZARD_MODE": "exponential", "CLEAVAGE_HAZARD_SEED": "3621",
        "CLEAVAGE_EVENT_LENGTH_MODE": "threshold_scaled",
        "CLEAVAGE_EVENT_MIN_FACTOR": "0.5", "CLEAVAGE_EVENT_MAX_FACTOR": "4.0",
        "CLEAVAGE_EVENT_SUBSEGMENT_FRACTION": "0.1",
        "ANISOTROPIC_TRANSPORT_MODE": "validated_scalar",
        "ANISOTROPIC_USE_AVALANCHE_BACKEND": "1",
        "ANISOTROPIC_EMISSION_ENABLED": "1", "KERNEL_STRICT_FAMILY_OVERRIDE": "1",
        "PERSISTENT_SOURCE_MIN_WIDTH_UM": "0",
        "ONED_V2_TP_STATE_DIAGNOSTICS": "events",
        "SIGNED_KERNEL_FAMILY_JSON": str(family),
        "MECHANICAL_CONFIG": str(mechanical),
        "MECHANICAL_CONFIG_SHA256": MECHANICAL_SHA,
        "KERNEL_CACHE_ROOT": str(cache),
        "MPLCONFIGDIR": str(cache.parent / ".mplconfig"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-worktree", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--source-family", type=Path, required=True)
    parser.add_argument("--mechanical-config", type=Path, required=True)
    parser.add_argument("--source-control-checkpoint", type=Path, required=True)
    parser.add_argument("--source-enabled-checkpoint", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--enabled-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-out", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    execution = args.execution_worktree.resolve()
    if git(execution, "rev-parse", "HEAD") != EXECUTION_COMMIT:
        raise RuntimeError("execution worktree commit differs from the V5.4 seal")
    if git(execution, "rev-parse", "HEAD^{tree}") != EXECUTION_TREE:
        raise RuntimeError("execution worktree tree differs from the V5.4 seal")
    if git(execution, "status", "--porcelain"):
        raise RuntimeError("execution worktree is not clean")
    require(args.family, FAMILY_SHA, "V5.3 family")
    require(args.source_family, SOURCE_FAMILY_SHA, "archived V5.2 family")
    require(args.mechanical_config, MECHANICAL_SHA, "mechanical configuration")
    checkpoints = {
        "control_max1": checkpoint_record(args.control_checkpoint, "control_max1", 1),
        "enabled_max2": checkpoint_record(args.enabled_checkpoint, "enabled_max2", 2),
    }
    inventory = heavy_process_inventory()
    # Re-run the actual migration from each immutable original in memory and
    # require exact equality with the sealed migrated checkpoint. This invokes
    # no solver or process update.
    sys.path.insert(0, str(execution))
    from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
    from arrhenius_fracture.restart_family_migration_v11 import canonical_hash
    from arrhenius_fracture.signed_kernel_family_v10214 import (
        ActiveOnlySigned2DShieldingKernelFamily,
    )
    from scripts.qualify_pf_branching_integrated_envelope_v5_4 import migrate_checkpoint
    source_family = ActiveOnlySigned2DShieldingKernelFamily.from_json(args.source_family)
    target_family = ActiveOnlySigned2DShieldingKernelFamily.from_json(args.family)
    source_paths = {
        "control_max1": args.source_control_checkpoint,
        "enabled_max2": args.source_enabled_checkpoint,
    }
    migrated_paths = {
        "control_max1": args.control_checkpoint,
        "enabled_max2": args.enabled_checkpoint,
    }
    in_memory_migration = {}
    for role in source_paths:
        require(source_paths[role], SOURCE_CHECKPOINT_SHA[role], f"source {role} checkpoint")
        regenerated, audit = migrate_checkpoint(
            restore_branch_checkpoint(source_paths[role]), source_family, target_family,
        )
        sealed = restore_branch_checkpoint(migrated_paths[role])
        exact = all(
            canonical_hash(getattr(regenerated, field.name))
            == canonical_hash(getattr(sealed, field.name))
            for field in fields(regenerated)
        )
        if not exact:
            raise RuntimeError(f"dry-run migration differs from sealed {role} checkpoint")
        in_memory_migration[role] = {
            "qualification": audit["qualification"],
            "sealed_checkpoint_exact": exact,
            "process_state_exact": (
                audit["process_state_sha256_before"] == audit["process_state_sha256_after"]
            ),
            "rng_threshold_exact": (
                audit["rng_threshold_sha256_before"] == audit["rng_threshold_sha256_after"]
            ),
        }
    cases = []
    for role, maximum in (("control_max1", 1), ("enabled_max2", 2)):
        case = args.output_root / f"theta40_v5_4_{role}_seed3621"
        cache = case / "live_kernel_cache"
        cases.append({
            "role": role, "maximum_fronts": maximum,
            "candidate_id": "oneD_v2_focused_weak_T_0016",
            "temperature_K": 700, "theta_deg": 40, "hazard_seed": 3621,
            "event_length_um": 5, "target_projected_extension_um": 300,
            "output_directory": str(case.resolve()),
            "output_exists_before_launch": case.exists(),
            "command": command(
                output=case, checkpoint=Path(checkpoints[role]["manifest"]),
                family=args.family.resolve(), maximum_fronts=maximum,
            ),
            "environment": environment(execution, args.family.resolve(), args.mechanical_config.resolve(), cache),
        })
    if any(case["output_exists_before_launch"] for case in cases):
        raise RuntimeError("a fresh completion output path already exists")
    pair_lock_path = args.output_root.with_name(args.output_root.name + ".pair.lock")
    global_lock_path = Path("/private/tmp/pf_current_source_branching_v5_4.global_worker.lock")
    locks_absent = not pair_lock_path.exists() and not global_lock_path.exists()
    if not locks_absent:
        raise RuntimeError("completion pair or global heavy-worker lock is already held")
    payload = {
        "schema": "pf_branching_completion_pair_preflight_v5_4/1",
        "qualification": "PASS",
        "mode": "LAUNCH" if args.launch else "DRY_RUN",
        "execution_commit": EXECUTION_COMMIT, "execution_tree": EXECUTION_TREE,
        "corrected_base_commit": EXECUTION_BASE_COMMIT, "corrected_base_tree": EXECUTION_BASE_TREE,
        "family_sha256": FAMILY_SHA, "family_physics_fingerprint": FAMILY_PHYSICS,
        "mechanical_configuration_sha256": MECHANICAL_SHA,
        "checkpoints": checkpoints, "cases": cases,
        "heavy_process_inventory": inventory,
        "interpreter": {
            "path": str(PYTHON), "sha256": sha256(PYTHON),
            "version": subprocess.check_output((str(PYTHON), "--version"), text=True).strip(),
        },
        "environment_fingerprints": {
            case["role"]: stable_json_hash(case["environment"]) for case in cases
        },
        "locks": {
            "pair": str(pair_lock_path), "global_worker": str(global_lock_path),
            "both_absent_at_preflight": locks_absent,
        },
        "worker_limit": 2, "workers_started": 0,
        "completion_run_executed": False, "completion_status": False,
        "daughter_stop_present": False, "theta45_present": False,
        "thousand_um_extension_present": False,
        "dry_run_migration_checked": True,
        "in_memory_migration_replay": in_memory_migration,
    }
    args.preflight_out.parent.mkdir(parents=True, exist_ok=True)
    args.preflight_out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if not args.launch:
        print(args.preflight_out.resolve())
        return 0

    pair_lock = Path(payload["locks"]["pair"])
    global_lock = Path(payload["locks"]["global_worker"])
    pair_descriptor = acquire_lock(pair_lock)
    global_descriptor = acquire_lock(global_lock)

    def run(case):
        out = Path(case["output_directory"]); out.mkdir(parents=True, exist_ok=False)
        stdout_path = out / "worker.stdout.log"; stderr_path = out / "worker.stderr.log"
        worker_manifest = out / "worker_manifest.json"
        worker_manifest.write_text(json.dumps({
            "role": case["role"], "command": case["command"],
            "environment_fingerprint": stable_json_hash(case["environment"]),
            "execution_commit": EXECUTION_COMMIT, "execution_tree": EXECUTION_TREE,
            "status": "RUNNING",
        }, indent=2, sort_keys=True) + "\n")
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            completed = subprocess.run(
                case["command"], cwd=execution, env=case["environment"],
                stdout=stdout, stderr=stderr,
            )
        worker_manifest.write_text(json.dumps({
            "role": case["role"], "command": case["command"],
            "environment_fingerprint": stable_json_hash(case["environment"]),
            "execution_commit": EXECUTION_COMMIT, "execution_tree": EXECUTION_TREE,
            "status": "COMPLETE" if completed.returncode == 0 else "FAILED",
            "returncode": completed.returncode,
            "stdout": str(stdout_path), "stderr": str(stderr_path),
        }, indent=2, sort_keys=True) + "\n")
        return completed.returncode
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            codes = list(pool.map(run, cases))
    finally:
        os.close(global_descriptor); global_lock.unlink(missing_ok=True)
        os.close(pair_descriptor); pair_lock.unlink(missing_ok=True)
    (args.output_root.parent / (args.output_root.name + ".pair_manifest.json")).write_text(
        json.dumps({**payload, "completion_run_executed": True, "returncodes": codes},
                   indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return 0 if codes == [0, 0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
