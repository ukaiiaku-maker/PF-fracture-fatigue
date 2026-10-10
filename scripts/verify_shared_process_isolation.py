#!/usr/bin/env python3
"""Execute native/reversible A-B-A isolation sequences through the public API."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.shared_execution import execute_shared
from arrhenius_fracture.shared_options import KineticsStrategy, OutputProfile, SharedOptions


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(options: SharedOptions, args: list[str], output: Path, policy: str):
    receipt = execute_shared(options, args, output, source_policy=policy)
    return {
        "output": str(output),
        "worker_pid": receipt["worker"]["pid"],
        "steps_sha256": _sha(output / "steps_0900K.csv"),
        "summary_sha256": _sha(output / "summary.json"),
        "executor_sha256": receipt["executor"]["module_sha256"],
        "runtime_methods": None if receipt.get("runtime_strategy_bindings") is None else [
            row["observed_methods"]
            for row in receipt["runtime_strategy_bindings"]["bindings"]
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--source-policy", choices=("qualified", "development_snapshot"),
        default="qualified",
    )
    values = parser.parse_args(argv)
    request = json.loads(values.request.read_text())
    base = dict(request["options"])
    base["output"] = OutputProfile.COMPACT_SMOKE.value
    base["kinetics"] = KineticsStrategy.NATIVE_SIGNED.value
    option_a = SharedOptions.from_dict(base)
    base["kinetics"] = KineticsStrategy.REVERSIBLE_SIGNED.value
    option_b = SharedOptions.from_dict(base)
    native = list(request["native_args"])
    step_index = native.index("--steps")
    native[step_index + 1] = "1"
    values.work_root.mkdir(parents=True, exist_ok=False)
    watched = (
        "CLEAVAGE_HAZARD_SEED", "CLEAVAGE_HAZARD_MODE",
        "CLEAVAGE_EVENT_LENGTH_MODE", "SIGNED_KERNEL_FAMILY_JSON",
    )
    environment_before = {name: os.environ.get(name) for name in watched}
    records = {}
    for label, options in (
        ("aba_a1", option_a), ("aba_b", option_b), ("aba_a2", option_a),
        ("bab_b1", option_b), ("bab_a", option_a), ("bab_b2", option_b),
    ):
        records[label] = _run(
            options, native, values.work_root / label, values.source_policy
        )
    failed = None
    try:
        _run(
            option_b, [*native, "--definitely-invalid-shared-isolation-probe"],
            values.work_root / "failed_b", values.source_policy,
        )
    except RuntimeError as error:
        failed = str(error)
    if failed is None:
        raise RuntimeError("the intentional invalid-B isolation probe did not fail")
    records["post_failure_a"] = _run(
        option_a, native, values.work_root / "post_failure_a", values.source_policy
    )
    a_reference = (records["aba_a1"]["steps_sha256"], records["aba_a1"]["summary_sha256"])
    b_reference = (records["aba_b"]["steps_sha256"], records["aba_b"]["summary_sha256"])
    a_labels = ("aba_a1", "aba_a2", "bab_a", "post_failure_a")
    b_labels = ("aba_b", "bab_b1", "bab_b2")
    result = {
        "schema": "shared-process-isolation-evidence/1",
        "source_policy": values.source_policy,
        "records": records,
        "all_worker_pids_distinct": len({row["worker_pid"] for row in records.values()}) == len(records),
        "a_histories_identical": all(
            (records[label]["steps_sha256"], records[label]["summary_sha256"]) == a_reference
            for label in a_labels
        ),
        "b_histories_identical": all(
            (records[label]["steps_sha256"], records[label]["summary_sha256"]) == b_reference
            for label in b_labels
        ),
        "b_methods_identical": all(
            records[label]["runtime_methods"] == records["aba_b"]["runtime_methods"]
            for label in b_labels
        ),
        "intentional_b_failure_observed": True,
        "intentional_b_failure": failed,
        "parent_environment_unchanged": environment_before == {
            name: os.environ.get(name) for name in watched
        },
    }
    result["passed"] = all((
        result["all_worker_pids_distinct"], result["a_histories_identical"],
        result["b_histories_identical"], result["b_methods_identical"],
        result["parent_environment_unchanged"],
    ))
    values.evidence.parent.mkdir(parents=True, exist_ok=True)
    values.evidence.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        raise RuntimeError("shared process-isolation regression failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
