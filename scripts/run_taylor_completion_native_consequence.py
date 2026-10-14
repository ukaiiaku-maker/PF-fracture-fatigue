#!/usr/bin/env python3
"""Two-step full-PF consequence check for the Taylor completion correction.

The historical arm is an explicit diagnostic state transfer from the accepted
step-46 current-law checkpoint.  It is installed only by this script in an
isolated process and is not a production option or fallback.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.shared_execution import execute_shared
from arrhenius_fracture.shared_options import OutputProfile, SharedOptions
import arrhenius_fracture.unified_mpz as unified_mpz

SOURCE_REQUEST = (
    ROOT / "integration_evidence" / "final_fracture_campaign"
    / "baseline_dbtt_950K_accommodation_enabled" / "shared_execution_request.json"
)
SOURCE_CHECKPOINT = (
    ROOT / "integration_evidence" / "f1_native_taylor_peierls_qualification_2276745b"
    / "refinement_step46_checkpoint"
)
FINAL_STEP = 48
DIAGNOSTIC_SITE_DIR = ROOT / "scripts" / "diagnostic_taylor_sitecustomize"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _replace(args: list[str], flag: str, value: str) -> None:
    args[args.index(flag) + 1] = value


def _configuration() -> tuple[SharedOptions, list[str]]:
    request = json.loads(SOURCE_REQUEST.read_text())
    options = dict(request["options"])
    options["output"] = OutputProfile.PUBLICATION_RESTART.value
    native = list(request["native_args"])
    _replace(native, "--steps", str(FINAL_STEP))
    return SharedOptions.from_dict(options), native


def _checkpoint_identity(path: Path) -> dict[str, Any]:
    manifest_path = path / "run_state_checkpoint.json"
    manifest = json.loads(manifest_path.read_text())
    generation = path / "run_state_generations" / manifest["generation"]
    verified = {}
    for name, expected in manifest["files"].items():
        actual = _sha(generation / name)
        if actual != expected:
            raise RuntimeError(f"source checkpoint hash mismatch for {name}")
        verified[name] = actual
    return {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": _sha(manifest_path),
        "generation": manifest["generation"], "verified_files": verified,
    }


def run_arm(root: Path, arm: str) -> None:
    output = root / arm
    if output.exists():
        raise SystemExit(f"refusing to overwrite consequence output: {output}")
    output.mkdir(parents=True)
    checkpoint = _checkpoint_identity(SOURCE_CHECKPOINT)
    overlay = None
    worker_overlay_receipt = output / "worker_diagnostic_overlay.json"
    if arm == "historical_completion_diagnostic":
        overlay = {
            "classification": "DIAGNOSTIC_ONLY_STATE_TRANSFER_NOT_PRODUCTION_SELECTABLE",
            "installation": "isolated worker startup hook outside the production package",
            "sitecustomize_sha256": _sha(DIAGNOSTIC_SITE_DIR / "sitecustomize.py"),
            "expected_worker_receipt": worker_overlay_receipt.name,
            "production_runtime_option_added": False,
        }
    elif arm != "current_native":
        raise ValueError(arm)
    bound = unified_mpz.UnifiedMPZState._transport_rates.__globals__["completion_rate"]
    receipt = {
        "schema": "taylor-completion-native-consequence-arm/1",
        "arm": arm, "source_head": _git_head(),
        "source_request": str(SOURCE_REQUEST.relative_to(ROOT)),
        "source_request_sha256": _sha(SOURCE_REQUEST),
        "source_checkpoint": checkpoint,
        "parent_process_completion_callable": (
            f"{bound.__module__}.{bound.__qualname__}"
        ),
        "overlay": overlay,
        "scope": (
            "two accepted full-PF steps from the same active step-46 state; "
            "historical arm is a marked state-transfer sensitivity"
        ),
        "production_execution_authorized": False,
    }
    (output / "diagnostic_arm_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    options, native = _configuration()
    saved = {
        name: os.environ.get(name) for name in (
            "PYTHONPATH", "FRACTURE_DIAGNOSTIC_TAYLOR_COMPLETION",
            "FRACTURE_DIAGNOSTIC_TAYLOR_RECEIPT",
        )
    }
    try:
        if arm == "historical_completion_diagnostic":
            prior = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = str(DIAGNOSTIC_SITE_DIR) + (
                os.pathsep + prior if prior else ""
            )
            os.environ["FRACTURE_DIAGNOSTIC_TAYLOR_COMPLETION"] = "historical_implicit_1s"
            os.environ["FRACTURE_DIAGNOSTIC_TAYLOR_RECEIPT"] = str(worker_overlay_receipt)
        execute_shared(
            options, native, output, source_policy="qualified",
            restart_from=SOURCE_CHECKPOINT,
        )
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if arm == "historical_completion_diagnostic":
        if not worker_overlay_receipt.is_file():
            raise RuntimeError("isolated worker produced no diagnostic overlay receipt")
        worker_receipt = json.loads(worker_overlay_receipt.read_text())
        worker_attestation = json.loads(
            (output / "shared_runtime_attestation.post.json").read_text()
        )
        if worker_receipt["pid"] != worker_attestation["worker"]["pid"]:
            raise RuntimeError("diagnostic overlay was not installed in the native worker")


def _generation(checkpoint: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = checkpoint / "run_state_checkpoint.json"
    manifest = json.loads(manifest_path.read_text())
    directory = checkpoint / "run_state_generations" / manifest["generation"]
    for name, expected in manifest["files"].items():
        if _sha(directory / name) != expected:
            raise RuntimeError(f"terminal checkpoint hash mismatch: {directory / name}")
    return manifest, directory


def _terminal_step(path: Path) -> dict[str, float]:
    files = sorted(path.glob("steps_*K.csv"))
    if len(files) != 1:
        raise RuntimeError(f"expected one step table in {path}; found {len(files)}")
    with files[0].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or int(float(rows[-1]["step"])) != FINAL_STEP:
        raise RuntimeError(f"arm did not reach step {FINAL_STEP}: {path}")
    names = (
        "Uapp_m", "Ftop_N", "KJ_Pa_sqrtm", "sigma_tip_Pa", "sigma_back_Pa",
        "mpz_K_shield_Pa_sqrt_m", "mpz_mobile_count", "mpz_retained_count",
        "mpz_local_slip_count", "mpz_escaped_total", "J_effective_direct_J_per_m2",
    )
    return {name: float(rows[-1][name]) for name in names}


def compare(root: Path) -> dict[str, Any]:
    arms = {
        name: root / name
        for name in ("current_native", "historical_completion_diagnostic")
    }
    terminal = {name: _terminal_step(path) for name, path in arms.items()}
    vectors = {}
    kinetics = {}
    for name, path in arms.items():
        _, directory = _generation(path / "checkpoint")
        kinetics[name] = json.loads((directory / "kinetic.json").read_text())
        with np.load(directory / "state.npz", allow_pickle=False) as data:
            vectors[name] = data["kinetic_active_vector"].copy()
    a = vectors["current_native"]
    b = vectors["historical_completion_diagnostic"]
    if a.shape != b.shape:
        raise RuntimeError("terminal active-vector shapes differ")
    difference = np.abs(a - b)
    field_differences = {}
    for field in kinetics["current_native"]["fields"]:
        start, stop = int(field["start"]), int(field["stop"])
        part = difference[start:stop]
        field_differences[field["name"]] = {
            "maximum_absolute_difference": float(np.max(part)) if part.size else 0.0,
            "l1_difference": float(np.sum(part)),
        }
    report = {
        "schema": "taylor-completion-native-consequence/1",
        "status": "PASS" if float(np.max(difference)) > 0.0 else "FAIL_NO_ACTIVE_CONSEQUENCE",
        "source_head": _git_head(),
        "comparison": (
            "same accepted step-46 full-PF state and original DBTT/950 K specimen; "
            "current native completion versus exact historical one-second completion only"
        ),
        "current_terminal": terminal["current_native"],
        "historical_terminal": terminal["historical_completion_diagnostic"],
        "terminal_metric_difference_current_minus_historical": {
            key: terminal["current_native"][key]
            - terminal["historical_completion_diagnostic"][key]
            for key in terminal["current_native"]
        },
        "active_vector": {
            "shape": list(a.shape),
            "maximum_absolute_difference": float(np.max(difference)),
            "l1_difference": float(np.sum(difference)),
            "exactly_equal": bool(np.array_equal(a, b, equal_nan=True)),
        },
        "field_differences": field_differences,
        "interpretation_limit": (
            "This two-step marked state-transfer sensitivity establishes an active native "
            "consequence; it is not a historical production rerun or an effect-size estimate."
        ),
        "failed_predecessor": (
            "taylor_completion_native_consequence_d4cdd51e: parent-only overlay did not "
            "cross the isolated worker boundary and correctly failed with exact equality"
        ),
    }
    (root / "native_consequence_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if not report["status"].startswith("PASS"):
        raise SystemExit(report["status"])
    return report


def _run_isolated(root: Path, arm: str) -> None:
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), arm, "--root", str(root)],
        cwd=ROOT, check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("current_native", "historical_completion_diagnostic", "compare", "all")
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == "all":
        _run_isolated(root, "current_native")
        _run_isolated(root, "historical_completion_diagnostic")
        compare(root)
    elif args.phase == "compare":
        compare(root)
    else:
        run_arm(root, args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
