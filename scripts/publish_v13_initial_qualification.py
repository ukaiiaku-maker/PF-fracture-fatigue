#!/usr/bin/env python3
"""Package existing initial-validation evidence; does not run any physics."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--full-log", type=Path, required=True)
    parser.add_argument("--focused-log", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    full, focused = args.full_log.read_text(), args.focused_log.read_text()
    failures = [line.split(" - ")[0][7:] for line in full.splitlines() if line.startswith("FAILED ")]
    if len(failures) != 31 or any("test_v13" in node for node in failures):
        raise RuntimeError("unexpected full-suite failure inventory")
    legacy = [node for node in failures if node.startswith((
        "tests/test_real_signed_builder_v10212.py", "tests/test_v10214_capture_model_id.py",
        "tests/test_v10214_response_model_id.py", "tests/test_v10215_stage3_schema_and_status.py",
        "tests/test_v10_2_27_zero_event_summary.py", "tests/test_v10_2_29_vhcf_nonlinear_selector.py"))]
    historical = [node for node in failures if node not in legacy]
    if len(legacy) != 7 or len(historical) != 24:
        raise RuntimeError("unexpected historical/legacy failure classification")
    if not re.search(r"57 passed", focused) or "FAILED " in focused:
        raise RuntimeError("focused qualification did not pass")
    mc = json.loads((root / "branch_mark_qualification.json").read_text())
    restore = json.loads((root / "process_restore_qualification.json").read_text())
    parent = json.loads((root / "canonical_parent.json").read_text())
    if mc["status"] != "PASS" or parent["status"] != "PASS" or restore["phase12_status"] != "PASS":
        raise RuntimeError("prerequisite did not pass")
    repeat = {}
    for name in ("branch_mark_qualification.json", "frozen_mark_validation.png"):
        repeat[name] = {"canonical_sha256": sha(root/name), "repeat_sha256": sha(args.repeat_root/name)}
        if repeat[name]["canonical_sha256"] != repeat[name]["repeat_sha256"]:
            raise RuntimeError(f"non-deterministic repeated artifact: {name}")
    checkpoints = {}
    for state in mc["frozen_states"]:
        current = sha(state["source_checkpoint"])
        if current != state["checkpoint_sha256"]:
            raise RuntimeError("historical source checkpoint changed")
        checkpoints[state["case"]] = current
    for row in restore["parity"]:
        if checkpoints[row["case"]] != row["checkpoint_sha256"]:
            raise RuntimeError("parity and Monte Carlo source checkpoint mismatch")
    subprocess.run([sys.executable, "-m", "compileall", "-q", "arrhenius_fracture", "scripts", "tests"], check=True)
    subprocess.run(["git", "diff", "--check"], check=True)
    source_files = ["arrhenius_fracture/current_source_runtime_bindings.py",
        "arrhenius_fracture/current_source_multifront_hooks_v12.py", "arrhenius_fracture/sharp_front_v11_branching.py",
        "arrhenius_fracture/conditional_branch_mark_v13.py", "arrhenius_fracture/marked_topology_trial_v13.py",
        "arrhenius_fracture/accepted_step_overlay_v13.py", "scripts/v13_frozen_support.py",
        "scripts/v13_value_fingerprint.py", "scripts/qualify_v13_process_restore.py",
        "scripts/trace_v13_canonical_parent.py", "scripts/qualify_v13_branch_marks.py"]
    verification = {
        "schema": "v13.initial-qualification-verification/1",
        "boundary": "BRANCHING_KINETICS_MODEL_UNCALIBRATED", "python": sys.executable,
        "source_code_commit": mc["producer_code_commit"],
        "restoration_producer_commit": restore["source_commit"],
        "canonical_parent_producer_commit": parent["source_commit"],
        "source_files_sha256": {name: sha(name) for name in source_files},
        "assembler_source_sha256": sha(__file__),
        "full_suite_summary": full.splitlines()[-1], "focused_summary": focused.splitlines()[-1],
        "historical_product_failures": historical, "legacy_failures": legacy,
        "full_suite_green": False, "v13_tests_pass": True,
        "test_isolation": "behavioral source fixtures use fresh interpreter; legacy fatigue tests install global delegates",
        "compileall_pass": True, "git_diff_check_pass": True,
        "repeat_artifact_hashes": repeat, "checkpoint_sha256_reverified": checkpoints,
        "canonical_unique_mark_draws": sum(r["samples"] for r in mc["monte_carlo"]),
        "repeat_uses_same_counter_keys_not_new_physical_ensemble": True,
        "physical_trajectories": 0, "new_fem_equilibrium_solves": 0,
        "campaign_cli_enabled": False,
    }
    write_json(root / "verification.json", verification)
    (root / "full_suite.log").write_text(full)
    (root / "focused_tests.log").write_text(focused)
    paths = [root/name for name in ("process_restore_qualification.json", "canonical_parent.json",
        "branch_mark_qualification.json", "frozen_mark_validation.png", "verification.json",
        "full_suite.log", "focused_tests.log")]
    paths.append(Path("V13_INITIAL_QUALIFICATION.md"))
    manifest = {"schema": "v13.initial-review-manifest/1", "boundary": verification["boundary"],
        "producer_commits": {"restoration_parent": restore["source_commit"], "conditional_mark": mc["producer_code_commit"]},
        "artifacts": {path.name: {"sha256": sha(path), "bytes": path.stat().st_size} for path in paths}}
    write_json(root / "manifest.json", manifest)
    archive = root / "V13_INITIAL_QUALIFICATION_REVIEW.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in [*paths, root / "manifest.json"]:
            info = zipfile.ZipInfo(path.name, date_time=(2026, 9, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            output.writestr(info, path.read_bytes())
    (root / "archive.sha256").write_text(sha(archive) + "  " + archive.name + "\n")
    with zipfile.ZipFile(archive) as source:
        for name, record in manifest["artifacts"].items():
            if hashlib.sha256(source.read(name)).hexdigest() != record["sha256"]:
                raise RuntimeError("packaged artifact hash mismatch")
    print(archive, sha(archive))


if __name__ == "__main__":
    main()
