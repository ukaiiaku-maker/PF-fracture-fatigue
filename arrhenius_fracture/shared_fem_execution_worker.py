"""Standalone fresh-process launcher for the separately versioned FEM backend.

This file is executed by absolute path with the FEM interpreter.  It therefore
does not import the PF package (whose top-level package name is intentionally
the same as the external backend).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ("git", *args), cwd=root, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    ).stdout.strip()


def execute(request_path: str | Path) -> dict:
    request = json.loads(Path(request_path).read_text())
    output = Path(request["output_dir"]).resolve()
    backend = request["fem_backend"]
    root = Path(backend["root"]).resolve()
    observed_commit = _git(root, "rev-parse", "HEAD")
    observed_branch = _git(root, "branch", "--show-current")
    status = _git(root, "--no-optional-locks", "status", "--porcelain")
    tracked_status = _git(
        root, "--no-optional-locks", "status", "--porcelain", "--untracked-files=no",
    )
    if observed_commit != backend["expected_commit"]:
        raise RuntimeError(
            f"FEM commit changed before execution: {observed_commit}"
        )
    if observed_branch != backend["expected_branch"]:
        raise RuntimeError(
            f"FEM branch changed before execution: {observed_branch}"
        )
    if tracked_status:
        raise RuntimeError("FEM tracked calculation source became dirty before execution")

    module = importlib.import_module(request["executor_module"])
    executor = getattr(module, request["executor_attribute"])
    module_file = Path(module.__file__).resolve()
    if str(module_file) != backend["entry_file"]:
        raise RuntimeError(
            "FEM import resolved to the wrong source file: " + str(module_file)
        )
    if _sha256(module_file) != backend["entry_sha256"]:
        raise RuntimeError("FEM entry source changed after dependency inspection")

    argv = [*request["native_args"], "--out", str(output)]
    receipt = {
        "schema": "shared-runtime-attestation/1",
        "status": "constructed",
        "execution_verified": False,
        "options": request["options"],
        "options_sha256": request["options_sha256"],
        "plan": request["plan"],
        "native_argv": argv,
        "worker": {"pid": os.getpid(), "fresh_interpreter": True},
        "python": {"executable": sys.executable, "version": sys.version},
        "source_policy": {
            "policy": "qualified",
            "qualification": "pinned_tracked_clean_external_backend",
            "commit": observed_commit,
            "branch": observed_branch,
            "global_dirty": bool(status),
            "calculation_source_dirty": False,
        },
        "executor": {
            "module": request["executor_module"],
            "module_file": str(module_file),
            "module_sha256": _sha256(module_file),
            "callable": getattr(executor, "__qualname__", str(executor)),
        },
        "inputs": request.get("inputs", {}),
        "seed": {
            "hazard": request["options"]["random_seed"],
            "mode": os.environ.get("CLEAVAGE_HAZARD_MODE", "native_default"),
            "event_length_mode": os.environ.get(
                "CLEAVAGE_EVENT_LENGTH_MODE", "native_default"
            ),
        },
        "checkpoint_policy": {
            "profile": request["options"]["output"],
            "owner": "external_fem_native",
            "rendered_snapshot_count": None,
            "restart_from": request.get("restart_from"),
        },
        "fem_backend": backend,
    }
    (output / "shared_runtime_attestation.pre.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n"
    )
    result = executor(argv)
    final = {
        **receipt,
        "status": "executed",
        "execution_verified": True,
        "return_value": result,
    }
    (output / "shared_runtime_attestation.post.json").write_text(
        json.dumps(final, indent=2, sort_keys=True, default=str) + "\n"
    )
    return final


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("usage: shared_fem_execution_worker.py REQUEST.json")
    request_path = Path(args[0]).resolve()
    request = json.loads(request_path.read_text())
    output = Path(request["output_dir"]).resolve()
    try:
        execute(request_path)
    except BaseException as error:
        payload = {
            "schema": "shared-runtime-failure/1",
            "status": "failed",
            "execution_verified": False,
            "exception_class": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
        }
        (output / "shared_runtime_attestation.failure.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
