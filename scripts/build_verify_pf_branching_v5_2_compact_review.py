#!/usr/bin/env python3
"""Build and verify the small V5.2 execution-source review packet."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROMOTED = "e2aff736afe0e1d2d1b600c25743de317a71c7ba"
PROMOTED_TREE = "14a953037a44a25288c41d317a0ae0c7f36d22bc"
PRODUCER = "19a93e9bac0a24492db66bc49796addbab01a077"
FINAL = "a046a9521b4eff04840103baea1a3988312288b4"
FINAL_LAUNCHER_SHA256 = "b1c933ed7431b64c6e721f70981b60ed0704f561d895c14f8ce2d757ae2ca570"
SOURCE_SHA256 = "3387b5e725abafed48bd12396c2595bea151df273cae723def6b8f4ccafa6f28"
TEST_SHA256 = "353e855c53bd77991907eb0a907e406de8eb2dbef1916dec2f4a7ed1c9a53ab3"
ARCHIVE = "PF_BRANCHING_V5_2_SMALL_REVIEW_PACKET.zip"
MANIFEST = "pf_branching_v5_2_small_review_manifest.json"


def git(*args: str, binary: bool = False):
    output = subprocess.check_output(("git", *args), cwd=ROOT, text=not binary)
    return output if binary else output.strip()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def show_record(commit: str) -> str:
    return git("show", "-s", "--format=%H%n%T%n%P%n%s", commit) + "\n"


def build(source: Path) -> dict:
    source.mkdir(parents=True, exist_ok=True)
    paths = {
        "final_launcher/scripts/run_pf_current_source_branching_corrected_pair_v5_2.py": (
            FINAL, "scripts/run_pf_current_source_branching_corrected_pair_v5_2.py"
        ),
        "actual_dry_run/pf_branching_matched_pair_launch_preflight.json": (
            FINAL,
            "analysis_outputs/pf_current_source_branching_execution_source_v5_2/"
            "pf_branching_matched_pair_launch_preflight.json",
        ),
        "source_snapshot_execution_e2aff73/arrhenius_fracture/"
        "anisotropic_emission_v10174.py": (
            PROMOTED, "arrhenius_fracture/anisotropic_emission_v10174.py"
        ),
        "source_snapshot_execution_e2aff73/tests/"
        "test_anisotropic_emission_v10174.py": (
            PROMOTED, "tests/test_anisotropic_emission_v10174.py"
        ),
    }
    for relative, (commit, repository_path) in paths.items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git("show", f"{commit}:{repository_path}", binary=True))

    launcher = source / next(name for name in paths if name.startswith("final_launcher/"))
    dry_path = source / next(name for name in paths if name.startswith("actual_dry_run/"))
    source_snapshot = source / next(
        name for name in paths if name.endswith("anisotropic_emission_v10174.py")
    )
    test_snapshot = source / next(
        name for name in paths if name.endswith("test_anisotropic_emission_v10174.py")
    )
    observed = {
        "launcher": sha256(launcher), "anisotropic_source": sha256(source_snapshot),
        "anisotropic_test": sha256(test_snapshot),
    }
    expected = {
        "launcher": FINAL_LAUNCHER_SHA256, "anisotropic_source": SOURCE_SHA256,
        "anisotropic_test": TEST_SHA256,
    }
    if observed != expected:
        raise RuntimeError(f"compact source hash mismatch: {observed}")

    ancestry = {
        "promoted_is_ancestor_of_producer": subprocess.call(
            ("git", "merge-base", "--is-ancestor", PROMOTED, PRODUCER), cwd=ROOT
        ) == 0,
        "producer_is_ancestor_of_final": subprocess.call(
            ("git", "merge-base", "--is-ancestor", PRODUCER, FINAL), cwd=ROOT
        ) == 0,
    }
    if not all(ancestry.values()):
        raise RuntimeError("V5.2 final-record ancestry failed")
    promoted_to_final = git("diff", "--name-status", f"{PROMOTED}..{FINAL}").splitlines()
    producer_to_final = git("diff", "--name-status", f"{PRODUCER}..{FINAL}").splitlines()
    execution_physics_changes = [
        row for row in promoted_to_final if row.split("\t")[-1].startswith("arrhenius_fracture/")
    ]
    if execution_physics_changes:
        raise RuntimeError(f"execution physics changed after promotion: {execution_physics_changes}")
    capture = (
        f"$ git show -s --format='%H%n%T%n%P%n%s' {PROMOTED}\n"
        + show_record(PROMOTED)
        + f"\n$ git show -s --format='%H%n%T%n%P%n%s' {PRODUCER}\n"
        + show_record(PRODUCER)
        + f"\n$ git show -s --format='%H%n%T%n%P%n%s' {FINAL}\n"
        + show_record(FINAL)
        + f"\n$ git diff --name-status {PROMOTED}..{FINAL}\n"
        + "\n".join(promoted_to_final) + "\n"
        + f"\n$ git diff --name-status {PRODUCER}..{FINAL}\n"
        + "\n".join(producer_to_final) + "\n"
    )
    (source / "pf_branching_v5_2_final_record_ancestry_and_delta.txt").write_text(capture)

    dry = json.loads(dry_path.read_text())
    cache_paths = [Path(case["destination_cache_root"]) for case in dry["cases"]]
    cache_roots_absent = bool(dry["output_root_absent"]) and all(
        str(path).startswith(str(Path(dry["output_root"]))) for path in cache_paths
    )
    binding = {
        "schema": "pf_branching_v5_2_actual_dry_run_compact_binding",
        "actual_dry_run_record_sha256": sha256(dry_path),
        "result": dry["result"], "workers_started": dry["workers_started"],
        "functional_tensor_conditioning_sentinel": dry[
            "functional_tensor_conditioning_sentinel"
        ]["result"],
        "pair_lock_acquired": dry["launch_lock_audit"]["atomic_pair_lock_acquired"],
        "pair_lock_released": dry["launch_lock_audit"]["atomic_pair_lock_released"],
        "global_worker_lock_acquired": dry["launch_lock_audit"]["global_worker_lock_acquired"],
        "global_worker_lock_released": dry["launch_lock_audit"]["global_worker_lock_released"],
        "existing_heavy_pf_worker_count": dry["global_heavy_pf_worker_audit"][
            "existing_worker_count"
        ],
        "input_hashes_verified": dry["input_hashes_verified"],
        "source_checkpoints_restored_identically": dry[
            "checkpoint_restore_identity_audit"
        ]["all_physical_identity_checks_pass"],
        "source_cases_immutable": dry["source_cases_immutable_verified"],
        "output_root_absent": dry["output_root_absent"],
        "destination_cache_roots": [str(path) for path in cache_paths],
        "destination_cache_roots_absent": cache_roots_absent,
    }
    if not (
        binding["result"] == "PASS" and binding["workers_started"] == 0
        and binding["functional_tensor_conditioning_sentinel"] == "PASS"
        and binding["destination_cache_roots_absent"]
    ):
        raise RuntimeError("actual V5.2 dry-run compact binding failed")
    dump(source / "pf_branching_v5_2_actual_dry_run_compact_binding.json", binding)

    provenance = {
        "schema": "pf_branching_v5_2_small_review_provenance",
        "promoted_execution_commit": PROMOTED, "promoted_execution_tree": PROMOTED_TREE,
        "record_producer_commit": PRODUCER, "final_artifact_commit": FINAL,
        "ancestry": ancestry,
        "execution_physics_source_changes_after_promoted_commit": execution_physics_changes,
        "execution_physics_source_unchanged_after_promotion": not execution_physics_changes,
        "launcher_changed_after_promoted_commit": any(
            row.endswith("scripts/run_pf_current_source_branching_corrected_pair_v5_2.py")
            for row in promoted_to_final
        ),
        "final_launcher_sha256": observed["launcher"],
        "promoted_source_snapshot_sha256": {
            "arrhenius_fracture/anisotropic_emission_v10174.py": observed["anisotropic_source"],
            "tests/test_anisotropic_emission_v10174.py": observed["anisotropic_test"],
        },
        "historical_a046_snapshot_export_defect": {
            "classification": "PACKAGING_ONLY_TRAILING_BYTE_STRIP",
            "cause": "binary git-show helper applied bytes.strip()",
            "incorrect_archived_source_hash": "5ceebb3e7d6daa8c5759e38080f38e4d1da9487e0f10b56ee537c24f686a04f3",
            "incorrect_archived_test_hash": "222d906113ee32d2bbb043f4c172eaaf0f4fc363ecb414a099b040bade29644f",
            "promoted_git_blobs_affected": False,
        },
        "large_archive_required_for_review": False,
        "bounded_corrected_replay_authorized": False,
        "continuation_to_1000um_authorized": False,
        "predictive_branching_physics_validated": False,
    }
    dump(source / "pf_branching_v5_2_small_review_provenance.json", provenance)
    report = f"""# V5.2 compact execution-source review packet

This packet supplies the four requested review items without the immutable tree tar, historical packages, pinned family, or large V5.2 archive.

- promoted execution source: `{PROMOTED}` / `{PROMOTED_TREE}`
- final record: `{FINAL}`
- final launcher SHA-256: `{observed['launcher']}`
- exact anisotropic source SHA-256: `{observed['anisotropic_source']}`
- exact anisotropic test SHA-256: `{observed['anisotropic_test']}`
- actual dry run: `PASS`, zero workers
- execution physics source changes after promotion: none

The old `5cee...` / `222d...` snapshot hashes came from a packaging-only exporter defect: `.strip()` was applied to binary `git show` output. The exact Git blobs in the promoted commit were never altered. The final launcher did change after the promoted execution commit and is supplied byte-for-byte here; it launches the frozen `{PROMOTED}` worktree.

Authorization remains false for the bounded corrected replay and for any 1000 micrometre continuation. No PF worker was launched while producing this packet.
"""
    (source / "PF_BRANCHING_V5_2_SMALL_REVIEW_PACKET.md").write_text(report)

    members = {
        path.relative_to(source).as_posix(): path for path in sorted(source.rglob("*"))
        if path.is_file() and path.name not in {ARCHIVE, MANIFEST}
    }
    manifest = {
        "schema": "pf_branching_v5_2_small_review_manifest",
        "declared_file_count": len(members),
        "files_sha256": {name: sha256(path) for name, path in members.items()},
    }
    dump(source / MANIFEST, manifest)
    archive = source / ARCHIVE
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for path in [source / MANIFEST, *members.values()]:
            name = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            stream.writestr(info, path.read_bytes(), compresslevel=9)
    return verify(archive)


def verify(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as stream:
        names = stream.namelist()
        if len(names) != len(set(names)) or any(
            PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            for name in names
        ):
            raise RuntimeError("unsafe or duplicate compact archive path")
        manifest = json.loads(stream.read(MANIFEST))
        declared = manifest["files_sha256"]
        if set(names) != set(declared) | {MANIFEST}:
            raise RuntimeError("compact archive presence mismatch")
        for name, expected in declared.items():
            if sha256_bytes(stream.read(name)) != expected:
                raise RuntimeError(f"compact archive SHA-256 mismatch: {name}")
        provenance = json.loads(stream.read("pf_branching_v5_2_small_review_provenance.json"))
        if provenance["final_launcher_sha256"] != FINAL_LAUNCHER_SHA256:
            raise RuntimeError("final launcher is not hash-bound")
    return {
        "schema": "pf_branching_v5_2_small_review_verification",
        "archive_sha256": sha256(archive), "archive_size_bytes": archive.stat().st_size,
        "declared_file_count": len(declared), "result": "PASS",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("build", "verify"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args(argv)
    result = build(args.source.resolve()) if args.operation == "build" else verify(args.archive.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
