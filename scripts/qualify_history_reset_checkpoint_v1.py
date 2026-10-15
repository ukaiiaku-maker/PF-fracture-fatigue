#!/usr/bin/env python3
"""Qualify a persisted P3 reset against an accepted populated native checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.history_reset_checkpoint_v1 import (
    RESET_FIELDS, transform_history_reset_checkpoint,
)
from arrhenius_fracture.run_state_checkpoint_v10230 import load_combined_checkpoint


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(source: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    reset = output / "reset_checkpoint"
    receipt = transform_history_reset_checkpoint(source, reset)
    outer, kinetic, arrays = load_combined_checkpoint(reset)
    vector = arrays["kinetic_active_vector"]
    fields = {
        field["name"]: vector[int(field["start"]):int(field["stop"])]
        for field in kinetic["fields"]
        if field["owner"] == "mpz" and field["name"] in RESET_FIELDS
    }
    checks = {
        "all_declared_process_fields_present": set(fields) == RESET_FIELDS,
        "all_declared_process_fields_zero": all(
            not values.any() for values in fields.values()
        ),
        "populated_source_was_removed": receipt["removed_population_total"] > 0.0,
        "physical_loss_ledgers_unchanged": receipt["physical_loss_ledgers_unchanged"],
        "native_fixed_load_reequilibrium_converged": receipt[
            "native_reequilibration"
        ]["converged"],
        "restart_generation_readable": bool(receipt["manifest"]["generation"]),
        "stale_high_cycle_cache_cleared": kinetic["high_cycle_cache"] == {},
        "counterfactual_work_retained": receipt["counterfactual_removed_work"][
            "W_emit_J_per_m"
        ] > 0.0,
        "source_and_destination_distinct": source.resolve() != reset.resolve(),
    }
    report = {
        "schema": "p3-history-reset-native-qualification/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True,
        ).strip(),
        "production_execution_authorized": False,
        "source_checkpoint": {
            "path": str(source.resolve()),
            "manifest_sha256": _sha(source / "run_state_checkpoint.json"),
        },
        "reset_checkpoint": {
            "path": str(reset.resolve()),
            "manifest_sha256": _sha(reset / "run_state_checkpoint.json"),
            "generation": receipt["manifest"]["generation"],
        },
        "checks": checks,
        "receipt": receipt,
        "restored_step": int(outer["driver"]["step"]),
        "scope": (
            "qualification-only populated-state history reset; native mechanics "
            "re-equilibrated; no physical calibration or production permission"
        ),
    }
    (output / "qualification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = qualify(args.source.resolve(), args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
