#!/usr/bin/env python3
"""Prepare durable records for the one authorized V5.2 theta-40 pair."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Volumes/Data/Data/Nanopillar_calculation")
LAUNCHER = ROOT / "scripts/run_pf_current_source_branching_corrected_pair_v5_2.py"
EXECUTION_WORKTREE = Path("/private/tmp/pf-branching-v5-2-execution-e2aff73")
CONTROL = DATA / ("PF-fracture-fatigue_current_source_branching_capability_20260829/"
                  "theta40_matched_pair/theta40_control_max1_seed3621/checkpoint/"
                  "transitions/step0000001_mesh_adaptation_g0002.json")
ENABLED = DATA / ("PF-fracture-fatigue_current_source_branching_capability_20260829/"
                  "theta40_matched_pair/theta40_enabled_max2_seed3621/checkpoint/"
                  "transitions/step0000001_mesh_adaptation_g0002.json")
FAMILY = ROOT / ("analysis_outputs/pf_current_source_branching_execution_seal_v5_1/"
                 "pinned_inputs/theta40_signed_kernel_family.json")
MECHANICAL = ROOT / ("analysis_outputs/pf_current_source_branching_execution_seal_v5_1/"
                     "pinned_inputs/theta40_mechanical_configuration.json")
PROMOTED = "e2aff736afe0e1d2d1b600c25743de317a71c7ba"
PROMOTED_TREE = "14a953037a44a25288c41d317a0ae0c7f36d22bc"
FINAL_RECORD = "a046a9521b4eff04840103baea1a3988312288b4"
LAUNCHER_SHA256 = "b1c933ed7431b64c6e721f70981b60ed0704f561d895c14f8ce2d757ae2ca570"
EXPECTED = {
    "control_manifest": "afad328062a2000ea822fd567541ec2619144b2bd182c974ca4bf820482d0aac",
    "control_state": "4ec9c274c1d0ff7ea79b9cf513f57cf2e4592168087fa3d2052748b6fb04bb9f",
    "enabled_manifest": "9928c6ef6e52aa0b01fce70cc82379624faac949c35c3b041e8c6af8cc6e594b",
    "enabled_state": "d66ea86a94ef126c0bdd5ccba69455ecc741592d7653de05b92a09a18aa7c7ba",
    "family": "b109a2fd6fc393fc986b1f15d6edd7c37366d84c111710ea70bcaba75f426847",
    "mechanical": "3ccc690c103f96da8eca7ee4bc67272653bd888b402bb0878c1b1f6eb9dee6f9",
}
MARKERS = ("arrhenius_fracture", "pf-fracture-fatigue", "sharp_front",
           "fatigue", "fem", "mpirun", "srun")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def main() -> int:
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("V5.2 launcher worktree is dirty")
    if subprocess.call(("git", "merge-base", "--is-ancestor", FINAL_RECORD, "HEAD"), cwd=ROOT):
        raise RuntimeError("a046a952 is not an ancestor of the compact-record branch")
    if sha256(LAUNCHER) != LAUNCHER_SHA256:
        raise RuntimeError("final V5.2 launcher SHA-256 mismatch")
    if git("-C", str(EXECUTION_WORKTREE), "rev-parse", "HEAD") != PROMOTED:
        raise RuntimeError("detached execution worktree commit mismatch")
    if git("-C", str(EXECUTION_WORKTREE), "rev-parse", "HEAD^{tree}") != PROMOTED_TREE:
        raise RuntimeError("detached execution worktree tree mismatch")
    if git("-C", str(EXECUTION_WORKTREE), "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("detached execution worktree is dirty")

    observed = {
        "control_manifest": sha256(CONTROL),
        "control_state": sha256(Path(str(CONTROL) + ".state.pkl")),
        "enabled_manifest": sha256(ENABLED),
        "enabled_state": sha256(Path(str(ENABLED) + ".state.pkl")),
        "family": sha256(FAMILY), "mechanical": sha256(MECHANICAL),
    }
    if observed != EXPECTED:
        raise RuntimeError(f"pinned input hash mismatch: {observed}")

    process = subprocess.run(
        ("/bin/ps", "-axo", "pid=,ppid=,state=,%cpu=,%mem=,etime=,command="),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if process.returncode:
        raise RuntimeError("broad process inventory failed: " + process.stderr.strip())
    matches = []
    active_heavy = []
    for line in process.stdout.splitlines():
        lowered = line.lower()
        if not any(marker in lowered for marker in MARKERS):
            continue
        parts = line.strip().split(None, 6)
        if len(parts) < 7 or not parts[0].isdigit() or int(parts[0]) == os.getpid():
            continue
        state, command = parts[2], parts[6]
        heavy = any(token in command.lower() for token in (
            "arrhenius_fracture", "pf-fracture-fatigue", "sharp_front",
            "mpirun", "srun",
        )) and any(token in command.lower() for token in ("python", "mpirun", "srun"))
        disposition = (
            "SUSPENDED_NOT_ACTIVE" if state.startswith("T") else
            "ZOMBIE_NOT_ACTIVE" if state.startswith("Z") else
            "ACTIVE_HEAVY" if heavy else "MATCH_NOT_CLASSIFIED_HEAVY"
        )
        record = {"raw": line, "pid": int(parts[0]), "ppid": int(parts[1]),
                  "state": state, "command": command, "disposition": disposition}
        matches.append(record)
        if disposition == "ACTIVE_HEAVY":
            active_heavy.append(record)
    if active_heavy or len(active_heavy) + 2 > 2:
        raise RuntimeError(f"unrelated active heavy process blocks launch: {active_heavy}")

    runstamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outroot = DATA / f"PF-fracture-fatigue_current_source_branching_corrected_v5_2_{runstamp}"
    preflight = Path(str(outroot) + "_matched_pair_launch_preflight.json")
    authorization = Path(str(outroot) + "_launch_authorization.json")
    process_audit = Path(str(outroot) + "_prelaunch_process_audit.txt")
    for path in (outroot, preflight, authorization, process_audit):
        if path.exists():
            raise RuntimeError(f"fresh launch path already exists: {path}")

    process_audit.write_text(
        "V5.2 BROAD PRELAUNCH PROCESS INVENTORY\n"
        f"recorded_utc={datetime.now(timezone.utc).isoformat()}\n"
        f"matching_count={len(matches)}\nactive_heavy_count={len(active_heavy)}\n"
        "disposition=PASS_SUSPENDED_PROCESSES_DO_NOT_COUNT_AS_ACTIVE\n\n"
        "MATCHING ENTRIES\n" + "\n".join(
            f"{item['disposition']}\t{item['raw']}" for item in matches
        ) + "\n\nFULL PROCESS TABLE\n" + process.stdout
    )
    record = {
        "schema": "pf_branching_v5_2_one_pair_launch_authorization",
        "claim_label": "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS",
        "authorization_scope": "ONE_THETA40_MAX1_MAX2_PAIR_ONLY",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "launcher": str(LAUNCHER), "launcher_sha256": sha256(LAUNCHER),
        "launcher_worktree_head": git("rev-parse", "HEAD"),
        "launcher_worktree_tree": git("rev-parse", "HEAD^{tree}"),
        "final_v5_2_record_is_ancestor": True,
        "execution_worktree": str(EXECUTION_WORKTREE),
        "execution_commit": PROMOTED, "execution_tree": PROMOTED_TREE,
        "checkpoint_and_input_sha256": observed,
        "control": str(CONTROL), "enabled": str(ENABLED),
        "signed_family": str(FAMILY), "mechanical_configuration": str(MECHANICAL),
        "maximum_workers": 2, "hard_negative_ceiling_um": 300.0,
        "qualified_daughter_early_stop_enabled": False,
        "theta45_authorized": False, "continuation_to_1000um_authorized": False,
        "broad_process_audit": str(process_audit),
        "broad_process_audit_sha256": sha256(process_audit),
        "broad_process_audit_disposition": "PASS_ZERO_ACTIVE_HEAVY",
        "suspended_processes": [item for item in matches if item["state"].startswith("T")],
        "active_heavy_worker_count": len(active_heavy),
        "requested_worker_count": 2, "active_plus_requested": len(active_heavy) + 2,
        "outroot": str(outroot), "outroot_absent": True,
        "preflight": str(preflight), "authorization_granted": True,
    }
    authorization.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "outroot": str(outroot), "preflight": str(preflight),
        "authorization": str(authorization), "process_audit": str(process_audit),
        "authorization_sha256": sha256(authorization), "result": "PASS",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
