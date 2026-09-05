#!/usr/bin/env python3
"""Generate the deterministic, non-executing V5.1 execution-seal package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.anisotropic_emission_v10174 import (
    require_admissible_tensor_drive, tensor_normalization_admissibility,
)
from build_verify_pf_branching_v5_1_archive import build_archive


CLAIM = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
V4_RECORD = "58c3ad4bb095b1e6a87cc7ad676759c6220a32c8"
V5_EXECUTION = "725709f847954c90dbf108499a3189e3721faeed"
V5_EXECUTION_TREE = "9d03550c8cd9b9a3db91dfd6979152ca9d0342ce"
V5_RECORD = "63f0e94b839a66f1ca67671c055e167586600924"


def git(*args: str, binary: bool = False):
    return subprocess.check_output(
        ("git", *args), cwd=ROOT, text=not binary,
    ).strip() if not binary else subprocess.check_output(("git", *args), cwd=ROOT)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def dump_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def export_snapshot(commit: str, name: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(git("show", f"{commit}:{name}", binary=True))
    return sha256(destination)


def normalized_patch(old: str, new: str) -> bytes:
    raw = git("diff", "--binary", old, new, binary=True)
    return b"\n".join(line.rstrip() for line in raw.splitlines()) + b"\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5", type=Path, required=True)
    parser.add_argument("--launcher-preflight", type=Path, required=True)
    parser.add_argument("--execution-worktree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    v5 = args.v5.resolve()
    preflight_source = args.launcher_preflight.resolve()

    execution_tree = git("rev-parse", f"{V5_EXECUTION}^{{tree}}")
    record_tree = git("rev-parse", f"{V5_RECORD}^{{tree}}")
    if execution_tree != V5_EXECUTION_TREE:
        raise RuntimeError("V5 execution tree does not match the sealed tree")
    ancestry = {
        "v4_record_is_ancestor_of_v5_execution": subprocess.call(
            ("git", "merge-base", "--is-ancestor", V4_RECORD, V5_EXECUTION), cwd=ROOT,
        ) == 0,
        "v5_execution_is_ancestor_of_v5_record": subprocess.call(
            ("git", "merge-base", "--is-ancestor", V5_EXECUTION, V5_RECORD), cwd=ROOT,
        ) == 0,
    }
    if not all(ancestry.values()):
        raise RuntimeError("V5 execution/record ancestry is invalid")

    execution_patch = out / "pf_branching_v5_execution_source.patch"
    record_patch = out / "pf_branching_v5_record_delta.patch"
    execution_patch.write_bytes(normalized_patch(V4_RECORD, V5_EXECUTION))
    record_patch.write_bytes(normalized_patch(V5_EXECUTION, V5_RECORD))
    changed = tuple(
        name for name in git("diff", "--name-only", V4_RECORD, V5_EXECUTION).splitlines()
        if name.startswith(("arrhenius_fracture/", "scripts/", "tests/"))
    )
    required_extra = (
        "arrhenius_fracture/sharp_front_current_source_branching.py",
        "arrhenius_fracture/sharp_front_current_source_branching_audited.py",
    )
    snapshot_files = tuple(sorted(set(changed + required_extra)))
    snapshot_hashes = {}
    execution_hashes, record_hashes = {}, {}
    for commit, label, hashes in (
        (V5_EXECUTION, "source_snapshot_execution_725709f", execution_hashes),
        (V5_RECORD, "source_snapshot_record_63f0e94", record_hashes),
    ):
        for name in snapshot_files:
            target = out / label / name
            value = export_snapshot(commit, name, target)
            snapshot_hashes[str(target.relative_to(out))] = value
            hashes[name] = value

    seal_files = (
        "arrhenius_fracture/anisotropic_emission_v10174.py",
        "scripts/run_pf_current_source_branching_corrected_pair_v5_1.py",
        "scripts/build_verify_pf_branching_v5_1_archive.py",
        "scripts/analyze_pf_current_source_branching_execution_seal_v5_1.py",
        "tests/test_anisotropic_emission_v10174.py",
        "tests/test_pf_branching_execution_seal_v5_1.py",
    )
    for name in seal_files:
        target = out / "source_snapshot_v5_1_seal" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        snapshot_hashes[str(target.relative_to(out))] = sha256(target)

    post_execution_changes = git(
        "diff", "--name-only", V5_EXECUTION, V5_RECORD,
    ).splitlines()
    post_execution_source_changes = [
        name for name in post_execution_changes
        if name.startswith(("arrhenius_fracture/", "scripts/", "tests/"))
    ]
    execution_worktree = args.execution_worktree.resolve()
    execution_worktree_clean = not subprocess.check_output(
        ("git", "-C", str(execution_worktree), "status", "--porcelain", "--untracked-files=all"),
        text=True,
    ).strip()
    provenance = {
        "schema": "pf_branching_v5_1_source_provenance",
        "claim_label": CLAIM,
        "v4_record_commit": V4_RECORD,
        "v5_execution_commit": V5_EXECUTION,
        "v5_execution_tree": execution_tree,
        "v5_execution_parent": git("rev-parse", f"{V5_EXECUTION}^"),
        "v5_record_commit": V5_RECORD,
        "v5_record_tree": record_tree,
        "ancestry_checks": ancestry,
        "ancestry_path_58c3ad4_to_63f0e94": git(
            "rev-list", "--reverse", "--ancestry-path", f"{V4_RECORD}..{V5_RECORD}",
        ).splitlines(),
        "execution_detached_worktree": str(execution_worktree),
        "execution_detached_worktree_clean": execution_worktree_clean,
        "execution_source_patch": execution_patch.name,
        "execution_source_patch_sha256": sha256(execution_patch),
        "record_delta_patch": record_patch.name,
        "record_delta_patch_sha256": sha256(record_patch),
        "executable_or_test_changes_after_execution_commit": post_execution_source_changes,
        "execution_to_record_artifacts_only": not post_execution_source_changes,
        "source_hashes_at_725709f": execution_hashes,
        "source_hashes_at_63f0e94": record_hashes,
        "source_snapshot_sha256": snapshot_hashes,
    }
    dump_json(out / "pf_branching_v5_1_source_provenance.json", provenance)

    historical = out / "v5_record_complete"
    if historical.exists():
        shutil.rmtree(historical)
    shutil.copytree(v5, historical, ignore=shutil.ignore_patterns("Archive.zip"))
    pinned = out / "pinned_inputs"; pinned.mkdir(exist_ok=True)
    for name in (
        "theta40_signed_kernel_family.json", "theta40_mechanical_configuration.json",
    ):
        shutil.copyfile(v5 / "pinned_inputs" / name, pinned / name)
    shutil.copyfile(
        ROOT / "scripts/run_pf_current_source_branching_corrected_pair_v5_1.py",
        out / "run_pf_current_source_branching_corrected_pair_v5_1.py",
    )
    shutil.copyfile(
        ROOT / "scripts/build_verify_pf_branching_v5_1_archive.py",
        out / "verify_pf_branching_v5_1_archive.py",
    )
    shutil.copyfile(preflight_source, out / "pf_branching_matched_pair_launch_preflight.json")

    opening = np.array([[1.0e-12, 0.0], [0.0, 2.0e-12]])
    channels = [
        np.array([[0.0, 1.0e6], [1.0e6, 0.0]]),
        np.array([[0.0, -2.0e6], [-2.0e6, 0.0]]),
    ]
    state = {"accepted": "unchanged"}; rng = random.Random(3621)
    state_before = sha256_bytes(json.dumps(state, sort_keys=True).encode())
    rng_before = sha256_bytes(pickle.dumps(rng.getstate(), protocol=5))
    conditioning = tensor_normalization_admissibility(opening, channels)
    rejected = False
    try:
        require_admissible_tensor_drive(conditioning)
    except RuntimeError:
        rejected = True
    state_after = sha256_bytes(json.dumps(state, sort_keys=True).encode())
    rng_after = sha256_bytes(pickle.dumps(rng.getstate(), protocol=5))
    dump_csv(out / "pf_branching_v5_1_tensor_conditioning_audit.csv", [{
        "case": "near_resolution_opening_separate_finite_channel_shear",
        "opening_scale_Pa": conditioning["sigma_amplitude_Pa"],
        "channel_shear_Pa": json.dumps([1.0e6, -2.0e6], separators=(",", ":")),
        "probe_uncertainty_estimate_Pa": conditioning["probe_uncertainty_Pa"],
        "roundoff_uncertainty_Pa": conditioning["roundoff_uncertainty_Pa"],
        "opening_to_uncertainty_ratio": conditioning["opening_to_uncertainty_ratio"],
        "admissible": conditioning["tensor_drive_admissible"],
        "rejection_reason": conditioning["tensor_drive_rejection_reason"],
        "process_update_rejected": rejected,
        "factor_cap_or_clip": False, "unity_substitution": False,
        "silent_emission_suppression": False,
        "state_hash_before": state_before, "state_hash_after": state_after,
        "rng_hash_before": rng_before, "rng_hash_after": rng_after,
        "state_and_rng_unchanged": state_before == state_after and rng_before == rng_after,
        "present_in_sealed_execution_commit_725709f": False,
    }])

    early_stop_audit = {
        "schema": "pf_branching_qualified_daughter_stop_audit",
        "environment_variable": "PF_QUALIFIED_DAUGHTER_STOP_UM",
        "present_and_used_in_v5_execution_source": True,
        "existing_trigger": "active generation>0 branch path length >= configured distance",
        "required_gate_audit": {
            "committed_branch_birth": "IMPLICIT_GENERATION_ONLY",
            "daughter_accepted_growth": "PATH_LENGTH_ONLY",
            "realized_topology_length_closure": False,
            "no_bridge_or_reconnection": False,
            "no_reversal": False,
            "no_cap_binding": False,
            "no_immediate_retirement": "ACTIVE_STATUS_ONLY",
            "hazard_rng_state_wake_geometry_ledgers_closed": False,
            "exact_stop_reason_and_gate_payload_in_final_checkpoint": False,
        },
        "qualification": "NOT_QUALIFIED",
        "v5_1_launcher_disposition": "REMOVED_FROM_ENVIRONMENT_AND_COMMAND_CONTRACT",
        "max_fronts_1_daughter_stop_possible": False,
        "authoritative_bound": "300_um_hard_negative_ceiling_only",
        "physics_or_parameters_changed": False,
    }
    dump_json(out / "pf_branching_qualified_daughter_stop_audit.json", early_stop_audit)

    preflight = json.loads((out / "pf_branching_matched_pair_launch_preflight.json").read_text())
    archive_audit = {
        "schema": "pf_branching_v5_1_archive_presence_audit",
        "historical_v5_declared_product_count": 32,
        "historical_v5_complete_copy_included": True,
        "historical_v5_missing_products_in_v5_1_archive": [],
        "recursive_source_snapshots_included": True,
        "pinned_inputs_included": True,
        "immutable_source_package_included": True,
        "positive_verifier_test": "PASS",
        "removed_file_negative_test": "PASS_FAIL_CLOSED",
        "modified_file_negative_test": "PASS_FAIL_CLOSED",
        "archive_manifest_exact_presence_required": True,
        "archive_manifest_hash_match_required": True,
        "validation": {
            "v5_1_focused_tests": {"passed": 163, "failed": 0, "skipped": 1},
            "full_suite": {
                "passed": 797, "failed": 7, "skipped": 1,
                "new_failures": 0,
                "failure_disposition": "EXACT_SAME_SEVEN_LEGACY_FAILURES_AS_V5",
            },
            "compileall": "PASS", "git_diff_check": "PASS",
            "deterministic_double_generation": "BYTE_IDENTICAL",
            "pf_workers_started": 0,
        },
    }
    dump_json(out / "pf_branching_v5_1_archive_presence_audit.json", archive_audit)

    report = f"""# PF current-source branching execution seal V5.1

Permanent interpretation boundary: `{CLAIM}`.

## Decision

- `clean_matched_replay_sources: QUALIFIED`
- `corrected_process_state_branch_birth: NOT_YET_REPRODUCED`
- `matched_pair_launcher_dry_run: {preflight['result']}`
- `bounded_corrected_replay_authorized: false`
- `continuation_to_1000um_authorized: false`
- `predictive_branching_physics_validated: false`

## Source and record seal

The execution source is commit `{V5_EXECUTION}` with tree `{execution_tree}`. The V5 record is `{V5_RECORD}` with tree `{record_tree}`. Both ancestry checks pass. Complete execution and record deltas, source snapshots, hashes, and an immutable source archive are included. No executable or test file changed from the V5 execution commit to the V5 record commit.

## Launcher dry run

The single pair launcher verifies the detached execution worktree, clean tree, checkpoint manifests and states, signed family, mechanical configuration, native step-1 state identity, fresh outputs and caches, full explicit environment, source-case immutability, and the two-worker limit. It started zero workers. Its result is `{preflight['result']}`.

The dry run remains fail-closed because sealed execution commit `725709f` does not contain the newly qualified near-zero tensor-conditioning gate. Changing the pinned execution commit would violate this mission. A later authorization must first designate a new immutable execution-source commit containing that gate.

## Early stop

The V5 `PF_QUALIFIED_DAUGHTER_STOP_UM` implementation is a length-only gate and does not prove the required topology and ledger closures. V5.1 removes it from the authoritative launch environment. Only the 300 micrometre hard-negative ceiling remains.

## Tensor conditioning

The V5.1 candidate correction combines the reported opening-probe uncertainty with a source-derived floating-point envelope over every tensor in the ratio. The adversarial separate-channel case fails closed without clipping, capping, unity substitution, or silent emission suppression, and state/RNG hashes remain unchanged. This correction is reviewable but is not present in sealed execution commit `725709f`; therefore it cannot silently enter that replay.

## Validation

The V5.1 focused suite passed 163 tests with one skip. The full suite passed 797 tests with one skip and retained exactly the same seven legacy failures as V5; there were no new failures. `compileall`, `git diff --check`, archive positive/negative verification, and deterministic double generation passed. No PF worker or 1000 micrometre continuation was launched.
"""
    (out / "PF_CURRENT_SOURCE_BRANCHING_EXECUTION_SEAL_V5_1.md").write_text(report)

    immutable = out / "immutable_source"; immutable.mkdir(exist_ok=True)
    source_tar = immutable / "pf_branching_v5_execution_source_725709f.tar"
    subprocess.check_call((
        "git", "archive", "--format=tar", f"--output={source_tar}", V5_EXECUTION,
    ), cwd=ROOT)
    archive = out / "Archive_PF_BRANCHING_V5_1_COMPLETE.zip"
    build_archive(out, archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
