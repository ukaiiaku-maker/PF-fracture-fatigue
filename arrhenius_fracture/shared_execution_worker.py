"""Fresh-interpreter worker for one shared PF or FEM/CZM execution."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from .shared_execution import (
    _BUNDLED_V104_MANIFEST, _file_sha256, build_execution_plan,
    construct_and_attest,
)
from .shared_options import MechanicsBackend, OutputProfile, SharedOptions


def _git(root: Path, *args: str) -> str:
    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ("git", *args), cwd=root, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    ).stdout


def _calculation_paths(status: str) -> list[str]:
    prefixes = ("arrhenius_fracture/", "scripts/", "config_examples/", "tests/")
    exact = {"pyproject.toml", "setup.py", "setup.cfg"}
    paths = []
    for line in status.splitlines():
        path = line[3:].split(" -> ")[-1]
        if path.startswith(prefixes) or path in exact:
            paths.append(path)
    return sorted(paths)


def _attest_source(root: Path, output: Path, policy: str) -> dict:
    commit = _git(root, "rev-parse", "HEAD").strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").strip()
    status = _git(root, "--no-optional-locks", "status", "--porcelain").rstrip()
    calculation_changes = _calculation_paths(status)
    receipt = {
        "policy": policy, "commit": commit, "tree": tree,
        "global_dirty": bool(status),
        "calculation_source_dirty": bool(calculation_changes),
        "calculation_changes": calculation_changes,
    }
    if policy == "qualified":
        if calculation_changes:
            raise RuntimeError(
                "qualified execution rejects modified calculation source: "
                + ", ".join(calculation_changes)
            )
        receipt["qualification"] = "committed_calculation_source"
        return receipt
    snapshot = output / "development_source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "tracked.patch").write_text(_git(root, "diff", "--binary", "HEAD", "--"))
    untracked = _git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    copied = []
    for relative in untracked:
        if relative not in calculation_changes:
            continue
        source = root / relative
        if source.is_file():
            target = snapshot / "untracked" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(relative)
    receipt.update({
        "qualification": "sealed_development_snapshot",
        "snapshot": str(snapshot),
        "untracked_calculation_files": copied,
    })
    return receipt


def _verify_source_unchanged(root: Path, before: dict) -> dict:
    """Reject a run whose calculation closure changed after preflight."""
    commit = _git(root, "rev-parse", "HEAD").strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").strip()
    status = _git(root, "--no-optional-locks", "status", "--porcelain").rstrip()
    changes = _calculation_paths(status)
    after = {
        "commit": commit,
        "tree": tree,
        "global_dirty": bool(status),
        "calculation_source_dirty": bool(changes),
        "calculation_changes": changes,
    }
    comparable = ("commit", "tree", "calculation_source_dirty", "calculation_changes")
    changed = [key for key in comparable if before.get(key) != after.get(key)]
    if changed:
        raise RuntimeError(
            "calculation source changed during execution: " + ", ".join(changed)
        )
    return after


def _verify_loaded_files(receipt: dict) -> None:
    rows = [*receipt.get("observed_bindings", []), receipt.get("executor", {})]
    for row in rows:
        raw = row.get("observed_module_file") or row.get("module_file")
        expected = row.get("module_sha256")
        if raw and expected and _file_sha256(raw) != expected:
            raise RuntimeError(f"loaded calculation source changed during execution: {raw}")


def _multifront_restart_flag(path: str | Path) -> str:
    source = Path(path)
    if source.suffix == ".json":
        try:
            schema = str(json.loads(source.read_text()).get("schema", ""))
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot inspect multifront checkpoint: {source}") from error
        if schema.startswith("v11.production-branch-network-checkpoint/"):
            return "--v11-restart-checkpoint"
    return "--v12-restart-checkpoint"


def execute(request_path: str | Path) -> dict:
    request = json.loads(Path(request_path).read_text())
    options = SharedOptions.from_dict(request["options"])
    output = Path(request["output_dir"]).resolve()
    root = Path(__file__).resolve().parents[1]
    source = _attest_source(root, output, request["source_policy"])
    plan = build_execution_plan(options)
    executor, receipt = construct_and_attest(plan, options)
    output_flags = {
        OutputProfile.COMPACT_SMOKE: ["--save-snapshots", "0", "--no-plots"],
        OutputProfile.SELECTED_FIELDS: ["--save-snapshots", "4"],
        OutputProfile.PUBLICATION_RESTART: ["--save-snapshots", "0", "--no-plots"],
    }[options.output]
    native_args = list(request["native_args"])
    if plan.executor_module == "arrhenius_fracture.shared_v5_cavity_adapter":
        native_args.append(
            "--cavity-enabled" if options.cavity.value == "full_order_v5"
            else "--cavity-disabled"
        )
    if plan.executor_module == "arrhenius_fracture.sharp_front_v10_4_bulk_peierls_taylor_audited":
        if any(item == "--material-manifest" or item.startswith("--material-manifest=")
               for item in native_args):
            raise ValueError("--material-manifest is owned by the shared v10.4 adapter")
        manifest = options.material_manifest or str(_BUNDLED_V104_MANIFEST)
        native_args.extend(["--material-manifest", str(Path(manifest).resolve())])
    restart_from = request.get("restart_from")
    if restart_from and plan.restart_family == "v12_accepted_multifront":
        restart_names = ("--v11-restart-checkpoint", "--v12-restart-checkpoint")
        if any(item in restart_names or item.startswith(restart_names[0] + "=")
               or item.startswith(restart_names[1] + "=") for item in native_args):
            raise ValueError("restart checkpoint was supplied twice")
        native_args.extend([_multifront_restart_flag(restart_from), restart_from])
    elif restart_from and plan.restart_family == "v11_native_full_state_v5":
        if any(item == "--restart-checkpoint" or item.startswith("--restart-checkpoint=")
               for item in native_args):
            raise ValueError("restart checkpoint was supplied twice")
        native_args.extend(["--restart-checkpoint", restart_from])
    argv = [*native_args, "--out", str(output), *output_flags]
    checkpoint = {
        "profile": options.output.value,
        "rendered_snapshot_count": int(output_flags[1]),
        "native_checkpoint_required": options.output is OutputProfile.PUBLICATION_RESTART,
        "checkpoint_directory": os.environ.get("V10230_HIGH_CYCLE_CHECKPOINT_DIR"),
        "restart_from": request.get("restart_from"),
    }
    receipt.update({
        "source_policy": source,
        "native_argv": argv,
        "checkpoint_policy": checkpoint,
        "worker": {"pid": os.getpid(), "fresh_interpreter": True},
    })
    (output / "shared_runtime_attestation.pre.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n"
    )
    result = executor(argv)
    source_after = _verify_source_unchanged(root, source)
    _verify_loaded_files(receipt)
    binding_path = output / "monotonic_reversible_runtime_binding.json"
    runtime_bindings = json.loads(binding_path.read_text()) if binding_path.is_file() else None
    final = {
        **receipt, "status": "executed", "execution_verified": True,
        "return_value": result, "runtime_strategy_bindings": runtime_bindings,
        "source_policy_post": source_after,
    }
    (output / "shared_runtime_attestation.post.json").write_text(
        json.dumps(final, indent=2, sort_keys=True, default=str) + "\n"
    )
    return final


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("usage: python -m arrhenius_fracture.shared_execution_worker REQUEST.json")
    request_path = Path(args[0]).resolve()
    output = Path(json.loads(request_path.read_text())["output_dir"]).resolve()
    try:
        execute(request_path)
    except BaseException as error:
        payload = {
            "schema": "shared-runtime-failure/1",
            "status": "failed", "execution_verified": False,
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
