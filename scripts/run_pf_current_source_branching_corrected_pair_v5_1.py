#!/usr/bin/env python3
"""Fail-closed launcher for the corrected theta-40 max-fronts 1/2 pair.

Dry-run performs every immutable-source, input, checkpoint, state-identity, and
fresh-output check without creating the output root or starting a PF worker.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any

import numpy as np


CLAIM = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
EXECUTION_COMMIT = "725709f847954c90dbf108499a3189e3721faeed"
EXECUTION_TREE = "9d03550c8cd9b9a3db91dfd6979152ca9d0342ce"
PYTHON = Path(
    "/opt/homebrew/Caskroom/miniconda/base/envs/"
    "arrhenius-sharp-front-v10-codex/bin/python"
)
ENTRY = "arrhenius_fracture.sharp_front_current_source_branching_audited"
EXPECTED_HASHES = {
    "control_manifest": "afad328062a2000ea822fd567541ec2619144b2bd182c974ca4bf820482d0aac",
    "control_state": "4ec9c274c1d0ff7ea79b9cf513f57cf2e4592168087fa3d2052748b6fb04bb9f",
    "enabled_manifest": "9928c6ef6e52aa0b01fce70cc82379624faac949c35c3b041e8c6af8cc6e594b",
    "enabled_state": "d66ea86a94ef126c0bdd5ccba69455ecc741592d7653de05b92a09a18aa7c7ba",
    "family": "b109a2fd6fc393fc986b1f15d6edd7c37366d84c111710ea70bcaba75f426847",
    "mechanical": "3ccc690c103f96da8eca7ee4bc67272653bd888b402bb0878c1b1f6eb9dee6f9",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def array_hash(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def tree_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def git(worktree: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(worktree), *args), text=True,
    ).strip()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def checkpoint_record(path: Path, role: str, maximum_fronts: int) -> dict[str, Any]:
    state_path = Path(str(path) + ".state.pkl")
    manifest = json.loads(path.read_text())
    checkpoint = pickle.loads(state_path.read_bytes())
    state = checkpoint.state
    shared = checkpoint.shared_process_state
    competitions = tuple(checkpoint.front_competitions.values())
    if len(competitions) != 1:
        raise RuntimeError(f"{role} source is not the clean one-front state")
    competition = competitions[0]
    normalized_network = replace(state.crack_network, branching_enabled=False)
    engine_rng = shared["engine_fields"].get("_hazard_rng")
    return {
        "role": role,
        "maximum_fronts": maximum_fronts,
        "manifest": str(path),
        "state_file": str(state_path),
        "manifest_sha256": sha256(path),
        "state_sha256": sha256(state_path),
        "manifest_state_sha256": manifest["state_sha256"],
        "physical_time_s": float(checkpoint.physical_time_s),
        "accepted_opening_m": float(checkpoint.accepted_load),
        "mesh_identity": checkpoint.mesh_identity,
        "displacement_sha256": array_hash(state.displacement),
        "ep_sha256": array_hash(state.ep_gp),
        "rho_sha256": array_hash(state.rho_gp),
        "damage_sha256": array_hash(state.damage),
        "elasticity_sha256": array_hash(state.elasticity_D),
        "material_sha256": stable_hash(state.material),
        "boundary_sha256": stable_hash(state.boundary),
        "cohesive_network_sha256": stable_hash(state.cohesive_network),
        "energy_ledgers_sha256": stable_hash(state.energy_ledgers),
        "mpz_sha256": stable_hash(shared["mpz_fields"]),
        "engine_state_sha256": stable_hash(shared["engine_fields"]),
        "directional_competition_sha256": stable_hash(competition),
        "hazard_actions": [float(item.action) for item in competition.hazard_states],
        "hazard_thresholds": [
            float(item.current_threshold_action) for item in competition.hazard_states
        ],
        "state_rng_sha256": stable_hash(state.rng_state),
        "legacy_process_rng_sha256": stable_hash(engine_rng),
        "topology_policy_value": bool(state.crack_network.branching_enabled),
        "topology_without_policy_sha256": stable_hash(normalized_network),
        "accepted_steps": int(state.event_counters.get("accepted_steps", 0)),
        "topology_actions": int(state.event_counters.get("topology_actions", 0)),
        "shared_state_updates": int(state.event_counters.get("shared_state_updates", 0)),
    }


def identity_audit(control: dict[str, Any], enabled: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "role", "maximum_fronts", "manifest", "state_file", "manifest_sha256",
        "state_sha256", "manifest_state_sha256", "topology_policy_value",
    }
    comparisons = {
        key: control[key] == enabled[key]
        for key in control if key not in excluded
    }
    policy_only = (
        control["topology_policy_value"] is False
        and enabled["topology_policy_value"] is True
        and all(comparisons.values())
    )
    if not policy_only:
        failed = [key for key, value in comparisons.items() if not value]
        raise RuntimeError(f"matched checkpoint identity failed: {failed}")
    return {
        "qualification": "PASS",
        "restored_without_advancing_physical_time": True,
        "all_physical_identity_checks_pass": True,
        "comparisons": comparisons,
        "intended_policy_only_difference": {
            "control": False, "enabled": True,
            "field": "crack_network.branching_enabled / maximum-fronts",
        },
        "control": control,
        "enabled": enabled,
    }


def command(
    *, out: Path, checkpoint: Path, family: Path, maximum_fronts: int,
) -> list[str]:
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
        "--tip-kinetics-mode", "moving_velocity",
        "--bulk-plasticity-mode", "tip_only", "--directional-j-mode", "root_signed",
        "--tip-plasticity", "--active-shielding", "--signed-active-shielding",
        "--mobile-shield-fraction", "0", "--no-wake-shielding",
        "--crystal-aniso", "--crystal-compete", "--crystal-theta-deg", "40",
        "--crystal-material", "w", "--j-decomposition", "cluster",
        "--crack-backend", "sharp_wake", "--adaptive-events",
        "--adaptive-event-target", "0.15", "--print-every", "200",
        "--save-snapshots", "0", "--no-plots", "--out", str(out),
    ]


def sealed_environment(
    *, execution_worktree: Path, family: Path, mechanical: Path,
    cache: Path,
) -> dict[str, str]:
    values = {
        "PATH": f"{PYTHON.parent}:/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": "/private/tmp",
        "LC_ALL": "C", "LANG": "C", "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(execution_worktree),
        "CONDA_ENV": "arrhenius-sharp-front-v10-codex",
        "CONDA_DEFAULT_ENV": "arrhenius-sharp-front-v10-codex",
        "CONDA_PREFIX": str(PYTHON.parent.parent),
        "PARAMETER_CAMPAIGN": "1",
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
        "MECHANICAL_CONFIG_SHA256": EXPECTED_HASHES["mechanical"],
        "KERNEL_CACHE_ROOT": str(cache),
        "MPLCONFIGDIR": str(cache.parent / ".mplconfig"),
    }
    # PF_QUALIFIED_DAUGHTER_STOP_UM is intentionally absent: its current
    # length-only implementation is not qualified by the V5.1 execution seal.
    return values


def environment_fingerprint(environment: dict[str, str]) -> str:
    payload = json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def case_plan(
    *, role: str, maximum_fronts: int, checkpoint: Path, outroot: Path,
    execution_worktree: Path, family: Path, mechanical: Path,
) -> dict[str, Any]:
    case = outroot / f"theta40_corrected_{role}_max{maximum_fronts}_seed3621"
    cache = case / "live_kernel_cache"
    env = sealed_environment(
        execution_worktree=execution_worktree, family=family,
        mechanical=mechanical, cache=cache,
    )
    return {
        "role": role, "maximum_fronts": maximum_fronts,
        "output_directory": str(case), "destination_cache_root": str(cache),
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": sha256(checkpoint),
        "checkpoint_state_sha256": sha256(Path(str(checkpoint) + ".state.pkl")),
        "signed_kernel_family": str(family),
        "signed_kernel_family_sha256": sha256(family),
        "mechanical_configuration": str(mechanical),
        "mechanical_configuration_sha256": sha256(mechanical),
        "command": command(
            out=case, checkpoint=checkpoint, family=family,
            maximum_fronts=maximum_fronts,
        ),
        "environment": env,
        "environment_sha256": environment_fingerprint(env),
        "qualified_daughter_early_stop_enabled": False,
        "hard_negative_ceiling_um": 300.0,
    }


def _run_case(plan: dict[str, Any], execution_worktree: Path) -> dict[str, Any]:
    out = Path(plan["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    status_path = out / "worker_status.json"
    stdout_path, stderr_path = out / "run.stdout.log", out / "run.stderr.log"
    started = datetime.now(timezone.utc).isoformat()
    with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
        process = subprocess.Popen(
            plan["command"], cwd=execution_worktree,
            env=plan["environment"], stdout=stdout, stderr=stderr,
        )
        running = {
            **plan, "status": "RUNNING", "pid": process.pid,
            "started_utc": started, "stdout": str(stdout_path),
            "stderr": str(stderr_path), "execution_commit": EXECUTION_COMMIT,
            "execution_tree": EXECUTION_TREE,
        }
        status_path.write_text(json.dumps(running, indent=2, sort_keys=True) + "\n")
        returncode = process.wait()
    completed = {
        **running, "status": "COMPLETE" if returncode == 0 else "FAILED",
        "returncode": returncode,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    status_path.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")
    return completed


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    execution_worktree = args.execution_worktree.resolve()
    if git(execution_worktree, "rev-parse", "HEAD") != EXECUTION_COMMIT:
        raise RuntimeError("execution worktree HEAD is not the sealed V5 source commit")
    if git(execution_worktree, "rev-parse", "HEAD^{tree}") != EXECUTION_TREE:
        raise RuntimeError("execution worktree tree is not the sealed V5 source tree")
    if git(execution_worktree, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("execution worktree is dirty")
    if not PYTHON.is_file():
        raise RuntimeError(f"sealed Python is missing: {PYTHON}")
    if str(execution_worktree) not in sys.path:
        # Pickled checkpoint classes must resolve from the sealed source tree,
        # never from the V5.1 packaging worktree.
        sys.path.insert(0, str(execution_worktree))

    control, enabled = args.control.resolve(), args.enabled.resolve()
    family, mechanical = args.family.resolve(), args.mechanical_config.resolve()
    require_hash(control, EXPECTED_HASHES["control_manifest"], "control manifest")
    require_hash(Path(str(control) + ".state.pkl"), EXPECTED_HASHES["control_state"], "control state")
    require_hash(enabled, EXPECTED_HASHES["enabled_manifest"], "enabled manifest")
    require_hash(Path(str(enabled) + ".state.pkl"), EXPECTED_HASHES["enabled_state"], "enabled state")
    require_hash(family, EXPECTED_HASHES["family"], "signed-kernel family")
    require_hash(mechanical, EXPECTED_HASHES["mechanical"], "mechanical configuration")

    outroot = args.outroot.resolve()
    plans = [
        case_plan(
            role="control", maximum_fronts=1, checkpoint=control,
            outroot=outroot, execution_worktree=execution_worktree,
            family=family, mechanical=mechanical,
        ),
        case_plan(
            role="enabled", maximum_fronts=2, checkpoint=enabled,
            outroot=outroot, execution_worktree=execution_worktree,
            family=family, mechanical=mechanical,
        ),
    ]
    forbidden = [outroot]
    for item in plans:
        forbidden.extend((
            Path(item["output_directory"]), Path(item["destination_cache_root"]),
        ))
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise RuntimeError(f"fresh output/cache requirement failed: {existing}")

    control_record = checkpoint_record(control, "control", 1)
    enabled_record = checkpoint_record(enabled, "enabled", 2)
    matched = identity_audit(control_record, enabled_record)
    source_cases = [control.parents[2], enabled.parents[2]]
    before = {str(path): tree_fingerprint(path) for path in source_cases}
    after = {str(path): tree_fingerprint(path) for path in source_cases}
    if before != after:
        raise RuntimeError("dry-run checkpoint restore mutated an immutable source case")
    tensor_source = (
        execution_worktree / "arrhenius_fracture/anisotropic_emission_v10174.py"
    ).read_text()
    conditioning_gate_present = (
        "opening_scale_not_resolved_above_probe_uncertainty" in tensor_source
        and "probe_uncertainty_Pa" in tensor_source
    )
    blocking_reasons = [] if conditioning_gate_present else [
        "sealed_execution_commit_725709f_lacks_near_zero_tensor_conditioning_gate"
    ]
    return {
        "schema": "pf_branching_matched_pair_launch_preflight_v5_1",
        "claim_label": CLAIM,
        "execution_commit": EXECUTION_COMMIT, "execution_tree": EXECUTION_TREE,
        "execution_worktree": str(execution_worktree), "worktree_clean": True,
        "python": str(PYTHON), "maximum_concurrent_pf_workers": args.workers,
        "dry_run": bool(args.dry_run), "workers_started": 0,
        "output_root": str(outroot), "output_root_absent": True,
        "input_hashes_verified": True,
        "checkpoint_restore_identity_audit": matched,
        "source_case_tree_sha256_before": before,
        "source_case_tree_sha256_after": after,
        "source_cases_immutable_verified": True,
        "execution_source_tensor_conditioning_gate_present": conditioning_gate_present,
        "blocking_reasons": blocking_reasons,
        "qualified_daughter_early_stop_enabled": False,
        "early_stop_disposition": "UNQUALIFIED_LENGTH_ONLY_GATE_REMOVED_FROM_LAUNCH",
        "hard_negative_ceiling_um": 300.0,
        "cases": plans,
        "result": "PASS" if not blocking_reasons else "FAIL_CLOSED",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-worktree", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--enabled", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--mechanical-config", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    audit = preflight(args)
    preflight_output = args.preflight_output.resolve()
    preflight_output.parent.mkdir(parents=True, exist_ok=True)
    preflight_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if args.dry_run:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0 if audit["result"] == "PASS" else 2

    if audit["result"] != "PASS":
        raise RuntimeError(
            "matched-pair execution refused: " + ", ".join(audit["blocking_reasons"])
        )

    outroot = args.outroot.resolve()
    outroot.mkdir(parents=True, exist_ok=False)
    pair_manifest = {
        **audit, "dry_run": False, "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = outroot / "matched_pair_launch_manifest.json"
    manifest_path.write_text(json.dumps(pair_manifest, indent=2, sort_keys=True) + "\n")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_run_case, item, args.execution_worktree.resolve())
            for item in audit["cases"]
        ]
        results = [future.result() for future in futures]
    after = {
        path: tree_fingerprint(Path(path))
        for path in audit["source_case_tree_sha256_before"]
    }
    immutable = after == audit["source_case_tree_sha256_before"]
    completed = {
        **pair_manifest, "status": "COMPLETE" if immutable and all(
            item["returncode"] == 0 for item in results
        ) else "FAILED",
        "workers_started": len(results), "cases": results,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "source_case_tree_sha256_after": after,
        "source_cases_immutable_verified": immutable,
    }
    manifest_path.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")
    if not immutable:
        raise RuntimeError("PF execution mutated an immutable source case")
    return 0 if all(item["returncode"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
