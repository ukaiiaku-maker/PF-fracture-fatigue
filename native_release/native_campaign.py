#!/usr/bin/env python3
"""Safe operator interface for the staged native-fracture campaign.

Validation and command construction are the default.  Starting a solver needs
both ``--execute`` and the literal production acknowledgement.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO.parents[1]
HERE = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
MATRIX = HERE / "MAIN_SEQUENCE.jsonl"
LOCK = HERE / "SOURCE_INPUT_LOCK.json"
DEFAULT_PYTHON = Path("/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python")
ACK = "START_STAGED_NATIVE_CAMPAIGN"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand(value: Any, output_root: Path | None = None) -> Any:
    replacements = {
        "{PACKAGE_ROOT}": str(PACKAGE),
        "{PF_ROOT}": str(REPO),
        "{FEM_ROOT}": str(PACKAGE / "source/Arrhenius_FEM_CZM_MPZ"),
        "{OUTPUT_ROOT}": str(output_root) if output_root else "{OUTPUT_ROOT}",
    }
    if isinstance(value, dict):
        return {key: expand(item, output_root) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, output_root) for item in value]
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value


def rows() -> list[dict]:
    return [json.loads(line) for line in MATRIX.read_text().splitlines() if line.strip()]


def git(*args: str, root: Path = REPO) -> str:
    return subprocess.check_output(("git", *args), cwd=root, text=True).strip()


def synthesize_options(row: dict) -> dict:
    existing = row.get("shared_options") or row.get("execution", {}).get("shared_options")
    if existing:
        return existing
    from arrhenius_fracture.shared_options import OutputProfile, PRESETS
    key = "fem_czm_generic_current" if row["backend"] == "full_fem_czm" else "pf_monotonic_tip_persistent"
    options = PRESETS[key]
    return replace(
        options,
        random_seed=int(row["seed"]),
        output=OutputProfile(row["output_profile"]),
    ).to_dict()


def command_for(
    row: dict, output_root: Path, python: Path,
    restart_from: Path | None = None,
) -> tuple[list[str], Path, Path]:
    row = expand(row, output_root)
    base = output_root / "cases" / row["case_id"]
    target = base
    attempt = 2
    while target.exists():
        target = base.parent / f"{base.name}_run{attempt}"
        attempt += 1
    request_dir = output_root / "requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    request = request_dir / f"{row['case_id']}.json"
    payload = {
        "schema": "shared-execution-request/1",
        "options": expand(synthesize_options(row), output_root),
        "native_args": row["native_args"],
        "output_dir": str(target),
        "source_policy": "qualified",
        "restart_from": (
            str(restart_from.resolve()) if restart_from is not None
            else row.get("execution", {}).get("restart_from")
        ),
        "release_case_identity": row["case_id"],
    }
    request.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if row["release_execution"]["launcher_kind"] == "direct_fem_native":
        command = [
            str(python), "-m", "arrhenius_fracture.generic_current_pf_companion",
            *row["native_args"], "--out", str(target),
        ]
        if restart_from is not None:
            command.extend(("--restart-from", str(restart_from.resolve())))
        return command, target, PACKAGE / "source/Arrhenius_FEM_CZM_MPZ"
    return [str(python), "-m", "arrhenius_fracture.shared_execution_worker", str(request)], target, REPO


def validation(output_root: Path | None, python: Path) -> dict:
    lock = json.loads(LOCK.read_text())
    errors = []
    checks = []
    pf_head = git("rev-parse", "HEAD")
    base = lock["repositories"]["pf"]["baseline_commit"]
    calculation_delta = git("diff", "--name-only", f"{base}..HEAD", "--", "arrhenius_fracture", "scripts", "config_examples")
    checks.append({"name": "pf_baseline_is_ancestor", "passed": subprocess.run(("git", "merge-base", "--is-ancestor", base, "HEAD"), cwd=REPO).returncode == 0})
    checks.append({"name": "pf_calculation_source_unchanged", "passed": calculation_delta == "", "detail": calculation_delta.splitlines()})
    fem = PACKAGE / "source/Arrhenius_FEM_CZM_MPZ"
    checks.append({"name": "fem_exact_commit", "passed": git("rev-parse", "HEAD", root=fem) == lock["repositories"]["fem"]["commit"]})
    for record in lock["artifacts"]:
        path = PACKAGE / record["path"]
        observed = sha256(path) if path.is_file() else None
        checks.append({"name": "artifact_sha256", "path": record["path"], "passed": observed == record["sha256"], "observed": observed})
    archive = PACKAGE / lock["state_archive"]["path"]
    try:
        with tarfile.open(archive, "r:gz") as handle:
            names = handle.getnames()
            unsafe = [name for name in names if Path(name).is_absolute() or ".." in Path(name).parts]
        checks.append({"name": "state_archive_readable_and_safe", "passed": not unsafe, "member_count": len(names), "unsafe": unsafe})
    except Exception as error:
        checks.append({"name": "state_archive_readable_and_safe", "passed": False, "error": repr(error)})
    kernel = PACKAGE / lock["kernel"]["path"]
    checks.append({"name": "portable_signed_kernel", "passed": kernel.is_file() and sha256(kernel) == lock["kernel"]["sha256"], "path": str(kernel)})
    selected = rows()
    checks.append({"name": "main_sequence_count", "passed": len(selected) == 663, "observed": len(selected)})
    unresolved = []
    for row in selected:
        expanded = expand(row, output_root)
        registry = Path(expanded["parameter_registry"])
        if not registry.is_absolute():
            registry = REPO / registry
        if not registry.is_file():
            unresolved.append({"case_id": row["case_id"], "missing_registry": str(registry)})
        args = expanded["native_args"]
        for index, token in enumerate(args[:-1]):
            if token in {"--signed-kernel-family", "--material-manifest", "--material-registry", "--parameter-registry"}:
                candidate = Path(args[index + 1])
                if not candidate.is_absolute():
                    candidate = REPO / candidate
                if not candidate.is_file():
                    unresolved.append({"case_id": row["case_id"], "flag": token, "missing": str(candidate)})
    checks.append({"name": "case_input_paths", "passed": not unresolved, "unresolved": unresolved[:50], "unresolved_count": len(unresolved)})
    checks.append({"name": "python_exists", "passed": python.is_file(), "path": str(python)})
    checks.append({"name": "output_root_supplied", "passed": output_root is not None, "path": None if output_root is None else str(output_root)})
    for check in checks:
        if not check["passed"]:
            errors.append(check["name"])
    return {
        "schema": "native-fracture-release-validation/1",
        "package_root": str(PACKAGE), "pf_head": pf_head,
        "checks": checks, "passed": not errors, "errors": errors,
        "solver_advanced": False,
    }


def state_path(output_root: Path) -> Path:
    return output_root / "campaign_control/RUN_STATE.json"


def load_state(output_root: Path) -> dict:
    path = state_path(output_root)
    if path.is_file():
        return json.loads(path.read_text())
    return {"schema": "native-fracture-runtime-state/1", "cases": {}, "active": None}


def save_state(output_root: Path, state: dict) -> None:
    path = state_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def select_case(all_rows: list[dict], state: dict, case_id: str | None, phase: int | None) -> dict:
    candidates = [row for row in all_rows if row["release_disposition"] == "READY_AFTER_TARGET_RESOURCE_ADMISSION"]
    if phase is not None:
        candidates = [row for row in candidates if row["release_phase_order"] == phase]
    if case_id is not None:
        candidates = [row for row in candidates if row["case_id"] == case_id]
    candidates = [row for row in candidates if state["cases"].get(row["case_id"], {}).get("status") not in {"COMPLETE", "RUNNING"}]
    if not candidates:
        raise SystemExit("no eligible uncompleted case matches the selection")
    return candidates[0]


def start(args, *, resume: bool = False) -> int:
    output = args.output_root.resolve()
    report = validation(output, args.python)
    if not report["passed"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    state = load_state(output)
    active = state.get("active")
    if active and pid_alive(int(active["pid"])):
        raise SystemExit(f"task-owned worker already active: {active}")
    row = select_case(rows(), state, args.case_id, args.phase)
    if resume and args.restart_from is None:
        raise SystemExit("resume requires --restart-from with an exact compatible checkpoint")
    if args.restart_from is not None and not args.restart_from.exists():
        raise SystemExit(f"restart checkpoint does not exist: {args.restart_from}")
    command, target, working_directory = command_for(
        row, output, args.python, args.restart_from,
    )
    plan = {"case_id": row["case_id"], "command": command, "working_directory": str(working_directory), "output": str(target), "execute": bool(args.execute), "resume_requested": resume}
    if not args.execute:
        print(json.dumps({**plan, "status": "DRY_RUN_ONLY"}, indent=2, sort_keys=True))
        return 0
    if args.acknowledge != ACK:
        raise SystemExit(f"execution requires --acknowledge {ACK}")
    target.mkdir(parents=True, exist_ok=False)
    log = target / "worker.log"
    stream = log.open("ab", buffering=0)
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": str(working_directory),
        "FRACTURE_FEM_ROOT": str(PACKAGE / "source/Arrhenius_FEM_CZM_MPZ"),
        "FRACTURE_GENERIC_FEM_ROOT": str(PACKAGE / "source/Arrhenius_FEM_CZM_MPZ"),
        "FRACTURE_FEM_PYTHON": str(args.python),
    })
    process = subprocess.Popen(command, cwd=working_directory, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    state["active"] = {"case_id": row["case_id"], "pid": process.pid, "command": command, "output": str(target), "log": str(log)}
    state["cases"][row["case_id"]] = {"status": "RUNNING", "pid": process.pid, "output": str(target)}
    save_state(output, state)
    print(json.dumps({**plan, "status": "STARTED", "pid": process.pid}, indent=2, sort_keys=True))
    return 0


def status(output: Path) -> dict:
    state = load_state(output)
    active = state.get("active")
    if active and not pid_alive(int(active["pid"])):
        target = Path(active["output"])
        complete = (target / "shared_runtime_attestation.post.json").is_file() and (target / "summary.json").is_file()
        state["cases"][active["case_id"]]["status"] = "COMPLETE" if complete else "INTERRUPTED_OR_FAILED_REVIEW_REQUIRED"
        state["active"] = None
        save_state(output, state)
    counts = {}
    for record in state["cases"].values():
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return {"schema": "native-fracture-runtime-status/1", "active": state.get("active"), "case_status_counts": counts, "state": str(state_path(output))}


def stop(output: Path) -> dict:
    state = load_state(output)
    active = state.get("active")
    if not active or not pid_alive(int(active["pid"])):
        return {"status": "NO_ACTIVE_TASK_OWNED_WORKER"}
    os.killpg(int(active["pid"]), signal.SIGTERM)
    state["cases"][active["case_id"]]["status"] = "STOP_REQUESTED_CHECKPOINT_PRESERVED_BY_NATIVE_OWNER"
    save_state(output, state)
    return {"status": "STOP_REQUESTED", "case_id": active["case_id"], "pid": active["pid"]}


def analyze(output: Path) -> dict:
    records = []
    for case_dir in sorted((output / "cases").glob("*")) if (output / "cases").is_dir() else []:
        records.append({
            "case_id": case_dir.name,
            "summary": str(case_dir / "summary.json") if (case_dir / "summary.json").is_file() else None,
            "attestation": str(case_dir / "shared_runtime_attestation.post.json") if (case_dir / "shared_runtime_attestation.post.json").is_file() else None,
            "checkpoint_count": sum(1 for path in case_dir.rglob("*") if path.is_file() and "checkpoint" in path.name.lower()),
        })
    report = {"schema": "native-fracture-incremental-analysis-index/1", "case_count": len(records), "cases": records, "classification_policy": {"interruption": "INTERRUPTED/RESUMABLE", "numerical_failure": "FAILURE", "domain_exit": "DOMAIN_EXIT", "no_event_endpoint": "PHYSICAL_CENSOR"}}
    target = output / "analysis/ANALYSIS_INVENTORY.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {**report, "written": str(target)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output-root", type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("validate")
    for name in ("start", "resume"):
        command = sub.add_parser(name)
        command.add_argument("--case-id")
        command.add_argument("--phase", type=int, choices=range(1, 7))
        command.add_argument("--execute", action="store_true")
        command.add_argument("--acknowledge")
        command.add_argument("--restart-from", type=Path)
    sub.add_parser("status")
    sub.add_parser("checkpoint-stop")
    sub.add_parser("analyze")
    args = parser.parse_args(argv)
    if args.output_root is None:
        raise SystemExit("--output-root is required")
    if args.action == "validate":
        report = validation(args.output_root.resolve(), args.python.resolve())
    elif args.action == "start":
        return start(args)
    elif args.action == "resume":
        return start(args, resume=True)
    elif args.action == "status":
        report = status(args.output_root.resolve())
    elif args.action == "checkpoint-stop":
        report = stop(args.output_root.resolve())
    else:
        report = analyze(args.output_root.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
