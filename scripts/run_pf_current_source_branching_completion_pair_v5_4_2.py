#!/usr/bin/env python3
"""Launch the single authorized V5.4.2 theta-40 completion pair.

This is an external audited wrapper.  It runs the immutable detached V5.4.1
execution source and never imports production physics from this record branch.
Dry-run is the default and starts no PF or mechanics worker.
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
from typing import Any


BOUNDARY = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
EXECUTION_COMMIT = "ae9a06d8c42287428e917baef143b0c1142cefd8"
EXECUTION_TREE = "2d0edbc7c7b51d81ab5678065ca68a1cef45d134"
RECORD_PARENT = "040a124afb4c90befa214faa2d8aacfd06f2d95c"
FAMILY_SHA = "423bc3232326b8ccc3ffcca0aa6b5363c67bad2c64debee49925bf1e6413e8cb"
SOURCE_FAMILY_SHA = "b109a2fd6fc393fc986b1f15d6edd7c37366d84c111710ea70bcaba75f426847"
FAMILY_PHYSICS = "e0bc48eb8c3f5526877a21f8551e500046a614df8081693f340084fb73400302"
MECHANICAL_SHA = "3ccc690c103f96da8eca7ee4bc67272653bd888b402bb0878c1b1f6eb9dee6f9"
RESTORE_SENTINEL_SHA = "6912cd00a4e22990d6cc914e55dae7d1fdcec32a06a0d95ae12094e68e5d3516"
PYTHON_SHA = "ba75aea91964fe01b84ca9c58ba0d30e9a61278ed3437ac583f2257695aa3c9b"
PYTHON_VERSION = "Python 3.12.13"
PIP_FREEZE_SHA = "8c418850c7e404a78c4a6b81c286845eb7b32aff6755ca84a20d18eff2bcfbd7"
ENVIRONMENT_BASE_SHA = "3ea63a78b314c2d526c7e312bd94140583e5882cc4a72828ea8826c4a062d997"
SOURCE_CHECKPOINT_SHA = {
    "control_max1": {
        "manifest": "afad328062a2000ea822fd567541ec2619144b2bd182c974ca4bf820482d0aac",
        "state": "4ec9c274c1d0ff7ea79b9cf513f57cf2e4592168087fa3d2052748b6fb04bb9f",
    },
    "enabled_max2": {
        "manifest": "9928c6ef6e52aa0b01fce70cc82379624faac949c35c3b041e8c6af8cc6e594b",
        "state": "d66ea86a94ef126c0bdd5ccba69455ecc741592d7653de05b92a09a18aa7c7ba",
    },
}
MIGRATED_CHECKPOINT_SHA = {
    "control_max1": {
        "manifest": "c3edb26d68f23ee89c4fa4c3f6389996e0d4e895cf650dd5c0d215027f3f3fef",
        "state": "2f1c64130c0d4849e4f1678c2a1ff219d045a7a33b12305d2d3ea72a9995c6e7",
    },
    "enabled_max2": {
        "manifest": "9c82c7305e374d85669f997569516f08aa34bba4af365bebcc598689a26710c2",
        "state": "545ce5d3d8e4e70c451dc039664116e4c18e6fe7809d2a771cec3805dd39f3f0",
    },
}
PYTHON = Path(
    "/opt/homebrew/Caskroom/miniconda/base/envs/"
    "arrhenius-sharp-front-v10-codex/bin/python"
)
ENTRY = "arrhenius_fracture.sharp_front_current_source_branching_audited"
GLOBAL_LOCK = Path("/private/tmp/pf_heavy_worker_global_limit_two.lock")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()


def require(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"changed {label}: expected {expected}, found {actual}: {path}")


def manifest_state(path: Path) -> Path:
    payload = json.loads(path.read_text())
    state = path.with_name(payload["state_file"])
    if payload.get("state_sha256") != sha256(state):
        raise RuntimeError(f"checkpoint manifest/state mismatch: {path}")
    return state


def checkpoint_record(path: Path, role: str, migrated: bool) -> dict[str, Any]:
    expected = MIGRATED_CHECKPOINT_SHA[role] if migrated else SOURCE_CHECKPOINT_SHA[role]
    require(path, expected["manifest"], f"{role} checkpoint manifest")
    state = manifest_state(path)
    require(state, expected["state"], f"{role} checkpoint state")
    result: dict[str, Any] = {
        "manifest": str(path.resolve()),
        "manifest_sha256": sha256(path),
        "state": str(state.resolve()),
        "state_sha256": sha256(state),
    }
    if migrated:
        migration = json.loads(path.read_text()).get("restart_family_migration_provenance") or {}
        required = {
            "qualification": "PASS",
            "migration_count": 1,
            "source_family_sha256": SOURCE_FAMILY_SHA,
            "target_family_sha256": FAMILY_SHA,
            "target_family_physics_fingerprint": FAMILY_PHYSICS,
            "engine_mpz_same_bound_family_object": True,
            "old_family_reachable_after_migration": False,
            "physical_or_stochastic_advance_performed": False,
        }
        if any(migration.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"invalid migrated checkpoint provenance: {path}")
        if migration.get("process_state_sha256_before") != migration.get("process_state_sha256_after"):
            raise RuntimeError(f"non-exact migrated process state: {path}")
        if migration.get("rng_threshold_sha256_before") != migration.get("rng_threshold_sha256_after"):
            raise RuntimeError(f"non-exact migrated RNG/threshold state: {path}")
        result["migration"] = migration
    return result


def environment_base() -> dict[str, str]:
    return {
        "PATH": f"{PYTHON.parent}:/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": "/private/tmp",
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "CONDA_ENV": "arrhenius-sharp-front-v10-codex",
        "CONDA_DEFAULT_ENV": "arrhenius-sharp-front-v10-codex",
        "CONDA_PREFIX": str(PYTHON.parent.parent),
        "PARAMETER_CAMPAIGN": "1",
        "CLEAVAGE_HAZARD_MODE": "exponential",
        "CLEAVAGE_HAZARD_SEED": "3621",
        "CLEAVAGE_EVENT_LENGTH_MODE": "threshold_scaled",
        "CLEAVAGE_EVENT_MIN_FACTOR": "0.5",
        "CLEAVAGE_EVENT_MAX_FACTOR": "4.0",
        "CLEAVAGE_EVENT_SUBSEGMENT_FRACTION": "0.1",
        "ANISOTROPIC_TRANSPORT_MODE": "validated_scalar",
        "ANISOTROPIC_USE_AVALANCHE_BACKEND": "1",
        "ANISOTROPIC_EMISSION_ENABLED": "1",
        "KERNEL_STRICT_FAMILY_OVERRIDE": "1",
        "PERSISTENT_SOURCE_MIN_WIDTH_UM": "0",
        "ONED_V2_TP_STATE_DIAGNOSTICS": "events",
    }


def environment(execution: Path, family: Path, mechanical: Path, cache: Path) -> dict[str, str]:
    result = environment_base()
    result.update({
        "PYTHONPATH": str(execution),
        "SIGNED_KERNEL_FAMILY_JSON": str(family),
        "MECHANICAL_CONFIG": str(mechanical),
        "MECHANICAL_CONFIG_SHA256": MECHANICAL_SHA,
        "KERNEL_CACHE_ROOT": str(cache),
        "MPLCONFIGDIR": str(cache.parent / ".mplconfig"),
    })
    return result


def command(output: Path, checkpoint: Path, family: Path, maximum_fronts: int) -> list[str]:
    if maximum_fronts not in (1, 2):
        raise RuntimeError("V5.4.2 authorizes maximum_fronts 1 and 2 only")
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


def is_heavy_compute_command(command_line: str) -> bool:
    """Classify compute workers, excluding their shell/tee supervisors."""
    words = command_line.split()
    if not words:
        return False
    executable = Path(words[0]).name.lower()
    compute_executable = executable.startswith("python") or executable in {
        "mpirun", "mpiexec", "dolfin", "fenics",
    }
    lower = command_line.lower()
    heavy_markers = (
        "sharp_front_current_source_branching_audited", "dolfin", "fenics",
        "pf-sintering", "pf_sintering", "production_recover",
    )
    return compute_executable and any(marker in lower for marker in heavy_markers)


def process_inventory() -> dict[str, Any]:
    completed = subprocess.run(
        ("ps", "-axo", "pid=,ppid=,%cpu=,%mem=,state=,etime=,command="),
        text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError("cannot inventory processes: " + completed.stderr.strip())
    own_pid = os.getpid()
    broad_markers = (
        "sharp_front", "run_pf_", "dolfin", "fenics", "arrhenius_fracture",
        "pf-sintering", "pf_sintering", "production_recover",
    )
    matches = []
    heavy = []
    for line in completed.stdout.splitlines():
        fields_ = line.strip().split(maxsplit=6)
        if len(fields_) < 7:
            continue
        pid = int(fields_[0])
        if pid == own_pid:
            continue
        lower = fields_[6].lower()
        if any(marker in lower for marker in broad_markers):
            record = {
                "pid": pid,
                "ppid": int(fields_[1]),
                "cpu_percent": float(fields_[2]),
                "memory_percent": float(fields_[3]),
                "state": fields_[4],
                "elapsed": fields_[5],
                "command": fields_[6],
            }
            matches.append(record)
            if is_heavy_compute_command(fields_[6]):
                heavy.append(record)
    return {
        "command": "ps -axo pid=,ppid=,%cpu=,%mem=,state=,etime=,command=",
        "matching_processes": matches,
        "external_heavy_workers": heavy,
        "external_heavy_worker_count": len(heavy),
    }


def interpreter_record() -> dict[str, str]:
    require(PYTHON, PYTHON_SHA, "Python interpreter")
    version = subprocess.check_output((str(PYTHON), "--version"), text=True).strip()
    if version != PYTHON_VERSION:
        raise RuntimeError(f"Python version changed: {version}")
    completed = subprocess.run(
        (str(PYTHON), "-m", "pip", "freeze"), text=True,
        capture_output=True, check=True,
    )
    freeze = "\n".join(sorted(line for line in completed.stdout.splitlines() if line)) + "\n"
    freeze_sha = hashlib.sha256(freeze.encode()).hexdigest()
    if freeze_sha != PIP_FREEZE_SHA:
        raise RuntimeError(f"Python environment changed: {freeze_sha}")
    if stable_json_hash(environment_base()) != ENVIRONMENT_BASE_SHA:
        raise RuntimeError("pinned scientific environment base changed")
    return {
        "path": str(PYTHON),
        "sha256": sha256(PYTHON),
        "version": version,
        "pip_freeze_sha256": freeze_sha,
        "environment_base_sha256": stable_json_hash(environment_base()),
    }


def immutable_fingerprint(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        label: {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for label, path in sorted(paths.items())
    }


def tree_fingerprint(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        records.append({
            "relative_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return {"file_count": len(records), "tree_sha256": stable_json_hash(records), "files": records}


def acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    os.write(descriptor, f"pid={os.getpid()} time={time.time():.17g}\n".encode())
    return descriptor


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
    parser.add_argument("--production-restore-sentinel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-out", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    execution = args.execution_worktree.resolve()
    if git(execution, "rev-parse", "HEAD") != EXECUTION_COMMIT:
        raise RuntimeError("execution worktree commit differs from the reviewed source")
    if git(execution, "rev-parse", "HEAD^{tree}") != EXECUTION_TREE:
        raise RuntimeError("execution worktree tree differs from the reviewed source")
    if git(execution, "status", "--porcelain"):
        raise RuntimeError("execution worktree is not clean")
    if git(execution, "rev-parse", "--abbrev-ref", "HEAD") != "HEAD":
        raise RuntimeError("execution worktree must be detached")

    require(args.family, FAMILY_SHA, "V5.3 append-only family")
    require(args.source_family, SOURCE_FAMILY_SHA, "V5.2 source family")
    require(args.mechanical_config, MECHANICAL_SHA, "mechanical configuration")
    require(args.production_restore_sentinel, RESTORE_SENTINEL_SHA, "V5.4.1 restore sentinel")
    sentinel = json.loads(args.production_restore_sentinel.read_text())
    if sentinel.get("qualification") != "PASS" or sentinel.get("execution_head") != EXECUTION_COMMIT:
        raise RuntimeError("V5.4.1 production restore sentinel does not qualify this source")
    if sentinel.get("pf_workers_started") != 0 or sentinel.get("mechanics_solves_performed") != 0:
        raise RuntimeError("V5.4.1 production restore sentinel crossed its no-solve boundary")
    for role in ("control_max1", "enabled_max2"):
        item = sentinel.get("roles", {}).get(role, {})
        if item.get("qualification") != "PASS" or not all(item.get("checks", {}).values()):
            raise RuntimeError(f"production restore sentinel failed for {role}")

    source_paths = {
        "control_max1": args.source_control_checkpoint.resolve(),
        "enabled_max2": args.source_enabled_checkpoint.resolve(),
    }
    migrated_paths = {
        "control_max1": args.control_checkpoint.resolve(),
        "enabled_max2": args.enabled_checkpoint.resolve(),
    }
    source_checkpoints = {
        role: checkpoint_record(path, role, False) for role, path in source_paths.items()
    }
    checkpoints = {
        role: checkpoint_record(path, role, True) for role, path in migrated_paths.items()
    }

    # Reproduce the reviewed migration in memory and require fieldwise identity
    # with the immutable sealed package.  This performs no provider lookup.
    sys.path.insert(0, str(execution))
    from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
    from arrhenius_fracture.restart_family_migration_v11 import canonical_hash
    from arrhenius_fracture.signed_kernel_family_v10214 import (
        ActiveOnlySigned2DShieldingKernelFamily,
    )
    from scripts.qualify_pf_branching_integrated_envelope_v5_4 import migrate_checkpoint

    source_family = ActiveOnlySigned2DShieldingKernelFamily.from_json(args.source_family)
    target_family = ActiveOnlySigned2DShieldingKernelFamily.from_json(args.family)
    migration_replay = {}
    for role in source_paths:
        regenerated, audit = migrate_checkpoint(
            restore_branch_checkpoint(source_paths[role]), source_family, target_family
        )
        sealed = restore_branch_checkpoint(migrated_paths[role])
        exact = all(
            canonical_hash(getattr(regenerated, field.name))
            == canonical_hash(getattr(sealed, field.name))
            for field in fields(regenerated)
        )
        if not exact:
            raise RuntimeError(f"migration replay differs from sealed {role} checkpoint")
        migration_replay[role] = {
            "qualification": audit.get("qualification"),
            "sealed_checkpoint_exact": exact,
            "migration_count": checkpoints[role]["migration"]["migration_count"],
            "old_family_reachable": checkpoints[role]["migration"]["old_family_reachable_after_migration"],
        }

    output_root = args.output_root.resolve()
    case_specs = (("control_max1", 1), ("enabled_max2", 2))
    cases = []
    for role, maximum_fronts in case_specs:
        case_root = output_root / f"theta40_v5_4_2_{role}_seed3621"
        cache = case_root / "live_kernel_cache"
        env = environment(execution, args.family.resolve(), args.mechanical_config.resolve(), cache)
        cases.append({
            "role": role,
            "maximum_fronts": maximum_fronts,
            "candidate_id": "oneD_v2_focused_weak_T_0016",
            "temperature_K": 700,
            "theta_deg": 40,
            "hazard_seed": 3621,
            "dU_m": 2e-7,
            "dt_s": 8.4,
            "event_length_um": 5.0,
            "target_maximum_forward_reach_um": 300.0,
            "daughter_length_early_stop": False,
            "output_directory": str(case_root),
            "cache_directory": str(cache),
            "command": command(case_root, migrated_paths[role], args.family.resolve(), maximum_fronts),
            "environment": env,
            "environment_sha256": stable_json_hash(env),
        })
    if output_root.exists() or any(Path(case["output_directory"]).exists() for case in cases):
        raise RuntimeError("fresh V5.4.2 output root already exists; relaunch is forbidden")

    pair_lock = output_root.with_name(output_root.name + ".pair.lock")
    if pair_lock.exists() or GLOBAL_LOCK.exists():
        raise RuntimeError("pair or global heavy-worker lock is already held")
    inventory = process_inventory()
    external_heavy = inventory["external_heavy_worker_count"]
    available_workers = 2 - external_heavy
    if available_workers < 1:
        raise RuntimeError(
            f"{external_heavy} unrelated heavy PF/FEM workers leave no authorized capacity"
        )
    pair_workers = min(2, available_workers)
    if external_heavy + pair_workers > 2:
        raise RuntimeError("planned workers would exceed system-wide heavy-worker limit two")

    immutable_paths = {
        "target_family": args.family.resolve(),
        "source_family": args.source_family.resolve(),
        "mechanical_configuration": args.mechanical_config.resolve(),
        "production_restore_sentinel": args.production_restore_sentinel.resolve(),
        "python_interpreter": PYTHON,
    }
    for role in source_paths:
        immutable_paths[f"source_{role}_manifest"] = source_paths[role]
        immutable_paths[f"source_{role}_state"] = Path(source_checkpoints[role]["state"])
        immutable_paths[f"migrated_{role}_manifest"] = migrated_paths[role]
        immutable_paths[f"migrated_{role}_state"] = Path(checkpoints[role]["state"])
    immutable_before = immutable_fingerprint(immutable_paths)
    interpreter = interpreter_record()

    payload: dict[str, Any] = {
        "schema": "pf_branching_completion_pair_preflight_v5_4_2/1",
        "qualification": "PASS",
        "mode": "LAUNCH" if args.launch else "DRY_RUN",
        "boundary": BOUNDARY,
        "authorization": "ONE_FRESH_THETA40_MAX1_MAX2_300UM_COMPLETION_PAIR_ONLY",
        "execution_commit": EXECUTION_COMMIT,
        "execution_tree": EXECUTION_TREE,
        "record_parent": RECORD_PARENT,
        "family_sha256": FAMILY_SHA,
        "family_physics_fingerprint": FAMILY_PHYSICS,
        "source_family_sha256": SOURCE_FAMILY_SHA,
        "mechanical_configuration_sha256": MECHANICAL_SHA,
        "production_restore_sentinel_sha256": RESTORE_SENTINEL_SHA,
        "interpreter": interpreter,
        "source_checkpoints": source_checkpoints,
        "checkpoints": checkpoints,
        "migration_replay": migration_replay,
        "cases": cases,
        "process_inventory": inventory,
        "system_wide_heavy_worker_limit": 2,
        "pair_worker_limit": 2,
        "pair_workers_planned": pair_workers,
        "workers_started": 0,
        "locks": {"pair": str(pair_lock), "global": str(GLOBAL_LOCK)},
        "immutable_inputs_before": immutable_before,
        "output_root_absent": True,
        "cache_roots_absent": True,
        "daughter_length_early_stop": False,
        "theta45_authorized": False,
        "maximum_fronts_greater_than_two_authorized": False,
        "continuation_to_1000um_authorized": False,
        "predictive_branching_physics_validated": False,
    }
    args.preflight_out.parent.mkdir(parents=True, exist_ok=True)
    args.preflight_out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if not args.launch:
        print(args.preflight_out.resolve())
        return 0

    # Recheck process capacity under both locks immediately before output creation.
    pair_descriptor = acquire_lock(pair_lock)
    global_descriptor = -1
    try:
        global_descriptor = acquire_lock(GLOBAL_LOCK)
        launch_inventory = process_inventory()
        launch_external = launch_inventory["external_heavy_worker_count"]
        launch_workers = min(2, 2 - launch_external)
        if launch_workers < 1 or launch_external + launch_workers > 2:
            raise RuntimeError("heavy-worker capacity changed after dry run")
        if launch_workers != pair_workers:
            raise RuntimeError("heavy-worker capacity changed between preflight and launch")
        output_root.mkdir(parents=True, exist_ok=False)

        def run_case(case: dict[str, Any]) -> dict[str, Any]:
            case_root = Path(case["output_directory"])
            case_root.mkdir(exist_ok=False)
            stdout_path = case_root / "worker.stdout.log"
            stderr_path = case_root / "worker.stderr.log"
            manifest_path = case_root / "worker_manifest.json"
            started = time.time()
            running = {
                "schema": "pf_branching_completion_worker_v5_4_2/1",
                "status": "RUNNING",
                "role": case["role"],
                "maximum_fronts": case["maximum_fronts"],
                "command": case["command"],
                "environment_sha256": case["environment_sha256"],
                "execution_commit": EXECUTION_COMMIT,
                "execution_tree": EXECUTION_TREE,
                "started_unix_s": started,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
            manifest_path.write_text(json.dumps(running, indent=2, sort_keys=True) + "\n")
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                completed = subprocess.run(
                    case["command"], cwd=execution, env=case["environment"],
                    stdout=stdout, stderr=stderr,
                )
            finished = {
                **running,
                "status": "COMPLETE" if completed.returncode == 0 else "FAILED",
                "returncode": completed.returncode,
                "finished_unix_s": time.time(),
                "elapsed_s": time.time() - started,
            }
            manifest_path.write_text(json.dumps(finished, indent=2, sort_keys=True) + "\n")
            return finished

        with ThreadPoolExecutor(max_workers=launch_workers) as pool:
            results = list(pool.map(run_case, cases))

        immutable_after = immutable_fingerprint(immutable_paths)
        immutable_exact = immutable_before == immutable_after
        raw_trees = {
            case["role"]: tree_fingerprint(Path(case["output_directory"])) for case in cases
        }
        pair_manifest = {
            **payload,
            "schema": "pf_branching_completion_pair_manifest_v5_4_2/1",
            "mode": "EXECUTED",
            "launch_process_inventory": launch_inventory,
            "pair_workers_used": launch_workers,
            "workers_started": len(results),
            "worker_results": results,
            "immutable_inputs_after": immutable_after,
            "immutable_inputs_exact": immutable_exact,
            "raw_output_trees_before_analysis": raw_trees,
            "completion_pair_executed": True,
            "execution_success": immutable_exact and all(item["returncode"] == 0 for item in results),
        }
        (output_root / "pair_manifest.json").write_text(
            json.dumps(pair_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        return 0 if pair_manifest["execution_success"] else 1
    finally:
        if global_descriptor >= 0:
            os.close(global_descriptor)
            GLOBAL_LOCK.unlink(missing_ok=True)
        os.close(pair_descriptor)
        pair_lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
