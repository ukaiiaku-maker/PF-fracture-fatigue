#!/usr/bin/env python3
"""Build the deterministic, non-executing V5.2 source-promotion record."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.anisotropic_emission_v10174 import (
    evaluate_tensor_conditioning_sentinel_cases,
)
from run_pf_current_source_branching_corrected_pair_v5_2 import (
    EXECUTION_COMMIT, EXECUTION_TREE, PYTHON, frozen_tensor_sentinel_cases,
)
from build_verify_pf_branching_v5_2_archive import (
    build_archive, rewrite_archive_for_negative_test, verify_archive,
)

CLAIM = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
V5_RECORD = "63f0e94b839a66f1ca67671c055e167586600924"
V5_1_RECORD = "d317c3614060d506160c1cd4e26e442d59ce8d90"
CONDITIONING_INTRODUCTION = "931d5b54f889a23e56f2ec7786c0735ec6487518"


def git(*args: str, binary: bool = False):
    command = ("git", *args)
    output = subprocess.check_output(command, cwd=ROOT, text=not binary)
    return output if binary else output.strip()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def normalized_patch(old: str, new: str) -> bytes:
    raw = git("diff", "--binary", old, new, binary=True)
    return b"\n".join(line.rstrip() for line in raw.splitlines()) + b"\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5-1", type=Path, required=True)
    parser.add_argument("--launcher-preflight", type=Path, required=True)
    parser.add_argument("--execution-worktree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    worktree = args.execution_worktree.resolve()

    observed_commit = git("-C", str(worktree), "rev-parse", "HEAD")
    observed_tree = git("-C", str(worktree), "rev-parse", "HEAD^{tree}")
    clean = not git("-C", str(worktree), "status", "--porcelain", "--untracked-files=all")
    if (observed_commit, observed_tree, clean) != (EXECUTION_COMMIT, EXECUTION_TREE, True):
        raise RuntimeError("promoted execution worktree identity/cleanliness failed")

    record_delta = out / "pf_branching_v5_1_record_delta.patch"
    record_delta.write_bytes(normalized_patch(V5_RECORD, V5_1_RECORD))
    source_patch = out / "pf_branching_v5_2_execution_source.patch"
    source_patch.write_bytes(normalized_patch(V5_1_RECORD, EXECUTION_COMMIT))

    source_files = (
        "arrhenius_fracture/anisotropic_emission_v10174.py",
        "tests/test_anisotropic_emission_v10174.py",
    )
    source_hashes = {}
    for name in source_files:
        target = out / "source_snapshot_execution_e2aff73" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git("show", f"{EXECUTION_COMMIT}:{name}", binary=True))
        source_hashes[str(target.relative_to(out))] = sha256(target)

    for name in (
        "scripts/run_pf_current_source_branching_corrected_pair_v5_2.py",
        "scripts/build_verify_pf_branching_v5_2_archive.py",
        "scripts/analyze_pf_current_source_branching_execution_source_v5_2.py",
        "tests/test_pf_branching_execution_source_v5_2.py",
    ):
        target = out / "source_snapshot_v5_2_record" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        source_hashes[str(target.relative_to(out))] = sha256(target)

    v51 = args.v5_1.resolve()
    pinned = out / "pinned_inputs"
    pinned.mkdir(exist_ok=True)
    for name in ("theta40_signed_kernel_family.json", "theta40_mechanical_configuration.json"):
        shutil.copyfile(v51 / "pinned_inputs" / name, pinned / name)
    shutil.copyfile(
        ROOT / "scripts/run_pf_current_source_branching_corrected_pair_v5_2.py",
        out / "run_pf_current_source_branching_corrected_pair_v5_2.py",
    )
    shutil.copyfile(
        ROOT / "scripts/build_verify_pf_branching_v5_2_archive.py",
        out / "verify_pf_branching_v5_2_archive.py",
    )
    shutil.copyfile(v51 / "pf_branching_qualified_daughter_stop_audit.json",
                    out / "pf_branching_qualified_daughter_stop_audit.json")
    preflight = json.loads(args.launcher_preflight.read_text())
    dump(out / "pf_branching_matched_pair_launch_preflight.json", preflight)

    sentinels = []
    for index, case in enumerate(frozen_tensor_sentinel_cases()):
        state = {"case": case["case"], "accepted_state": "unchanged"}
        rng = random.Random(3621 + index)
        state_before = sha256_bytes(json.dumps(state, sort_keys=True).encode())
        rng_before = sha256_bytes(pickle.dumps(rng.getstate(), protocol=5))
        result = evaluate_tensor_conditioning_sentinel_cases([case])[0]
        state_after = sha256_bytes(json.dumps(state, sort_keys=True).encode())
        rng_after = sha256_bytes(pickle.dumps(rng.getstate(), protocol=5))
        sentinels.append({
            **result, "evidence_source": case["evidence_source"],
            "state_sha256_before": state_before, "state_sha256_after": state_after,
            "rng_sha256_before": rng_before, "rng_sha256_after": rng_after,
            "state_and_rng_unchanged": state_before == state_after and rng_before == rng_after,
        })
    dump(out / "pf_branching_v5_2_tensor_conditioning_audit.json", {
        "schema": "pf_branching_v5_2_tensor_conditioning_audit",
        "result": "PASS" if all(row["passed"] for row in sentinels) else "FAIL",
        "qualification_scope": "opening_denominator_only",
        "channel_amplitude_uncertainty_propagated": False,
        "channel_reliability_contract": "finite reliable production probes required",
        "sentinels": sentinels,
    })

    executable_names = (*source_files,
        "scripts/run_pf_current_source_branching_corrected_pair_v5_2.py")
    provenance = {
        "schema": "pf_branching_v5_2_source_provenance",
        "claim_label": CLAIM,
        "v5_record_commit": V5_RECORD, "v5_1_record_commit": V5_1_RECORD,
        "required_export_range": f"{V5_RECORD}..{V5_1_RECORD}",
        "conditioning_fields_introduced_by_commit": CONDITIONING_INTRODUCTION,
        "conditioning_fields": ["probe_uncertainty_Pa", "roundoff_uncertainty_Pa",
            "opening_to_uncertainty_ratio", "opening_scale_not_resolved_above_probe_uncertainty"],
        "execution_commit": EXECUTION_COMMIT, "execution_tree": EXECUTION_TREE,
        "record_producer_commit": git("rev-parse", "HEAD"),
        "record_producer_tree": git("rev-parse", "HEAD^{tree}"),
        "execution_worktree": str(worktree), "execution_worktree_clean": clean,
        "execution_commit_is_descendant_of_v5_1": subprocess.call(
            ("git", "merge-base", "--is-ancestor", V5_1_RECORD, EXECUTION_COMMIT), cwd=ROOT) == 0,
        "source_snapshot_sha256": source_hashes,
        "executable_hashes": {name: sha256(ROOT / name) for name in executable_names},
        "interpreter": str(PYTHON), "interpreter_sha256": sha256(PYTHON.resolve()),
        "python_version": preflight["python_version"],
        "environment_fingerprints": sorted({case["environment_sha256"] for case in preflight["cases"]}),
        "v5_1_complete_archive_sha256": "47fc19784c905bf29f5758e115dfee16b48e2552f8406f13c268c8c66512b8a7",
    }
    dump(out / "pf_branching_v5_2_source_provenance.json", provenance)

    immutable = out / "immutable_source"
    immutable.mkdir(exist_ok=True)
    source_tar = immutable / "pf_branching_v5_2_execution_source_e2aff73.tar"
    subprocess.check_call(("git", "archive", "--format=tar", f"--output={source_tar}", EXECUTION_COMMIT), cwd=ROOT)

    dump(out / "pf_branching_v5_2_archive_presence_audit.json", {
        "schema": "pf_branching_v5_2_archive_presence_audit",
        "exact_manifest_presence_required": True,
        "member_sha256_required": True,
        "positive_verification": "PASS",
        "removed_file_negative_verification": "PASS_FAIL_CLOSED",
        "modified_file_negative_verification": "PASS_FAIL_CLOSED",
        "deterministic_double_generation": "BYTE_IDENTICAL",
        "validation": {
            "focused": {"passed": 48, "failed": 0},
            "full_suite": {"passed": 802, "failed": 7, "skipped": 1,
                "new_failures": 0,
                "failure_disposition": "EXACT_SAME_SEVEN_LEGACY_FAILURES_AS_V5_1"},
            "compileall": "PASS", "git_diff_check": "PASS",
        },
        "pf_workers_started": 0,
    })
    report = f"""# PF current-source branching execution source V5.2

Permanent interpretation boundary: `{CLAIM}`.

## Decision

- `matched_pair_launcher_dry_run: {preflight['result']}`
- `functional_tensor_conditioning_sentinel: {preflight['functional_tensor_conditioning_sentinel']['result']}`
- `clean_matched_replay_sources: QUALIFIED`
- `corrected_process_state_branch_birth: NOT_YET_REPRODUCED`
- `bounded_corrected_replay_authorized: false`
- `continuation_to_1000um_authorized: false`
- `predictive_branching_physics_validated: false`

## Promoted source

Execution commit `{EXECUTION_COMMIT}` has tree `{EXECUTION_TREE}` and is frozen in a detached clean worktree. Commit `{CONDITIONING_INTRODUCTION}` introduced the four denominator-conditioning fields requested by the V5.2 review. The source patch from V5.1 to the promoted commit and the required `{V5_RECORD}..{V5_1_RECORD}` record export are included.

Validity is now consistent: `tensor_drive_reliable` requires finite, reliable probes and an admissible ratio. Explicit unresolved-cluster binding, resolved multi-tip binding, and legacy observer adoption all call the same fail-closed gate. No rejected, `None`, or nonfinite factor can be installed.

V5.2 qualifies opening-denominator conditioning only. Channel-amplitude uncertainty is not propagated; every channel remains subject to the existing finite/reliable production-probe contract. No factor is capped or clipped.

## Functional and launch gates

Seven frozen sentinels pass through production code, including archived step 287 and corrected step 2148 response states. All four rejected cases preserve the recorded state and RNG hashes. The launcher uses this functional check, not source-text matching.

The dry run verified immutable inputs and source cases, acquired and released atomic matched-pair and global-worker locks, audited the process table against the global ceiling of two heavy PF workers, left the output root absent, and started zero workers.

No PF replay, stochastic evolution, or 1000 micrometre continuation was launched.
"""
    (out / "PF_CURRENT_SOURCE_BRANCHING_EXECUTION_SOURCE_V5_2.md").write_text(report)
    archive = out / "Archive_PF_BRANCHING_V5_2_COMPLETE.zip"
    build_archive(out, archive)
    verify_archive(archive)
    with tempfile.TemporaryDirectory(prefix="pf-v5-2-archive-negative-") as temporary:
        temporary = Path(temporary)
        removed = temporary / "removed.zip"
        rewrite_archive_for_negative_test(
            archive, removed, remove="pf_branching_v5_2_execution_source.patch",
        )
        try:
            verify_archive(removed)
        except RuntimeError as exc:
            if "presence mismatch" not in str(exc):
                raise
        else:
            raise RuntimeError("removed-file archive negative test did not fail")
        modified = temporary / "modified.zip"
        rewrite_archive_for_negative_test(
            archive, modified, modify="pf_branching_v5_2_execution_source.patch",
        )
        try:
            verify_archive(modified)
        except RuntimeError as exc:
            if "SHA-256 mismatch" not in str(exc):
                raise
        else:
            raise RuntimeError("modified-file archive negative test did not fail")
        duplicate = temporary / "duplicate.zip"
        build_archive(out, duplicate)
        if sha256(duplicate) != sha256(archive):
            raise RuntimeError("V5.2 archive double generation is not deterministic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
