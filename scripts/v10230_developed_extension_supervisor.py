#!/usr/bin/env python3
"""Preflight, stage, and supervise four qualified 0.95 trajectories to 100 um."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

try:
    from scripts import v10230_qualification_supervisor as qualification
    from scripts.v10230_qualification_family import validate as validate_family
except ModuleNotFoundError:
    import v10230_qualification_supervisor as qualification
    from v10230_qualification_family import validate as validate_family

from arrhenius_fracture.run_state_checkpoint_v10230 import (
    load_combined_checkpoint, validate_cross_layer,
)

TARGET_EXTENSION_UM = 100.0
CYCLES_MAX = 1.0e12
SOURCE_MANIFEST = "extension_source_manifest.json"
QUALIFICATION_MATRIX = qualification.matrix


def matrix() -> list[dict]:
    return [row for row in QUALIFICATION_MATRIX() if row["fraction"] == 0.95]


def inspect_source(source_root: Path, row: dict) -> dict:
    case = source_root / row["case"]
    output = case / "output"
    status = qualification.read_json(case / "qualification_status.json")
    summary = qualification.read_json(output / "developed_fatigue_growth_summary.json")
    if status.get("status") != "completed" or not summary.get("target_reached"):
        raise RuntimeError(f'{row["case"]}: source is not a completed growth trajectory')
    if not qualification.checkpoint_valid(case, row):
        raise RuntimeError(f'{row["case"]}: combined checkpoint is not row-valid')
    outer, kinetic, _arrays = load_combined_checkpoint(output)
    validate_cross_layer(outer, kinetic)
    stochastic = kinetic.get("stochastic", {})
    event_count = int(summary.get("event_count", -1))
    cycles = float(summary.get("cycles_consumed", -1.0))
    extension = float(summary.get("final_projected_extension_um", -1.0))
    if cycles != float(outer.get("cycles_total", -2.0)):
        raise RuntimeError(f'{row["case"]}: summary/outer cycle mismatch')
    if event_count != int(outer.get("geometry", {}).get("committed_event_count", -2)):
        raise RuntimeError(f'{row["case"]}: summary/outer event-count mismatch')
    if event_count != int(stochastic.get("hazard_event_index", -2)):
        raise RuntimeError(f'{row["case"]}: outer/kinetic event-count mismatch')
    if not 0.0 < extension < TARGET_EXTENSION_UM:
        raise RuntimeError(f'{row["case"]}: source extension is outside (0, 100) um')
    descriptor = qualification.read_json(output / "run_state_checkpoint.json")
    return {
        **row,
        "source_case": str(case.resolve()),
        "source_checkpoint": str(output.resolve()),
        "source_checkpoint_generation": descriptor.get("generation"),
        "starting_cycles": cycles,
        "starting_event_count": event_count,
        "starting_extension_um": extension,
        "target_extension_um": TARGET_EXTENSION_UM,
        "threshold_action": stochastic.get("hazard_threshold_action"),
        "physical_hazard_action": stochastic.get("hazard_action_current"),
        "normalized_clock_B": stochastic.get("B"),
        "rng_state_present": bool(stochastic.get("rng_state")),
    }


def preflight(source_root: Path, destination_root: Path, minimum_free_gib: float) -> dict:
    repository = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip():
        raise RuntimeError("extension preflight requires a clean worktree")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    family = validate_family(repository / "runtime_inputs/v10_2_30/qualification_family_manifest.json")
    if destination_root.exists():
        raise RuntimeError(f"destination already exists: {destination_root}")
    free = qualification.free_gib(destination_root.parent.resolve())
    if free < minimum_free_gib:
        raise RuntimeError(f"free space {free:.3f} GiB is below {minimum_free_gib:.3f} GiB")
    cases = [inspect_source(source_root.resolve(), row) for row in matrix()]
    return {
        "schema": "v10.2.30_developed_extension_preflight_v1",
        "source_root": str(source_root.resolve()),
        "destination_root": str(destination_root.resolve()),
        "launch_git_head": head,
        "family": family,
        "target_extension_um": TARGET_EXTENSION_UM,
        "cycles_max_censor": CYCLES_MAX,
        "maximum_concurrency": 2,
        "minimum_free_gib": minimum_free_gib,
        "available_free_gib": free,
        "cases": cases,
    }


def prepare(source_root: Path, destination_root: Path, minimum_free_gib: float) -> dict:
    payload = preflight(source_root, destination_root, minimum_free_gib)
    destination_root.mkdir(parents=True)
    for row in payload["cases"]:
        source_case = Path(row["source_case"])
        destination_case = destination_root / row["case"]
        shutil.copytree(source_case, destination_case)
        qualification.atomic_json(destination_case / SOURCE_MANIFEST, row)
        qualification.set_status(
            destination_case, "restartable", pid=None, restart_count=0,
            source_checkpoint_generation=row["source_checkpoint_generation"],
            source_case=row["source_case"], starting_cycles=row["starting_cycles"],
            starting_event_count=row["starting_event_count"],
            starting_extension_um=row["starting_extension_um"],
        )
    qualification.atomic_json(destination_root / "developed_extension_preflight.json", payload)
    return payload


def validate_staged(root: Path) -> dict:
    payload = qualification.read_json(root / "developed_extension_preflight.json")
    if payload.get("schema") != "v10.2.30_developed_extension_preflight_v1":
        raise RuntimeError("missing developed-extension preflight")
    if float(payload.get("target_extension_um", 0.0)) != TARGET_EXTENSION_UM:
        raise RuntimeError("staged target differs from 100 um")
    for row in matrix():
        case = root / row["case"]
        source = qualification.read_json(case / SOURCE_MANIFEST)
        if source.get("source_checkpoint_generation") != qualification.read_json(
                case / "output/run_state_checkpoint.json").get("generation"):
            raise RuntimeError(f'{row["case"]}: staged checkpoint generation changed')
        if not qualification.checkpoint_valid(case, row):
            raise RuntimeError(f'{row["case"]}: staged checkpoint is not row-valid')
    return payload


def extension_classify(case: Path, row: dict | None = None) -> str:
    state = qualification.classify_original(case, row)
    if state == "completed" and row is not None:
        summary = qualification.read_json(qualification.artifacts(case) / "developed_fatigue_growth_summary.json")
        if float(summary.get("final_projected_extension_um", 0.0)) < TARGET_EXTENSION_UM:
            return "restartable" if qualification.checkpoint_valid(case, row) else "failed"
    return state


def run(root: Path, args) -> int:
    validate_staged(root)
    qualification.matrix = matrix
    qualification.classify_original = qualification.classify
    qualification.classify = extension_classify
    qargs = qualification.parser().parse_args([
        "run", str(root), "--max-jobs", "2", "--target-extension-um", "100",
        "--cycles-max", "1e12", "--minimum-free-gib", str(args.minimum_free_gib),
        "--no-progress-seconds", str(args.no_progress_seconds),
        *( ["--recover-stale-lock"] if args.recover_stale_lock else [] ),
    ])
    return qualification.run(qargs)


def monitor(root: Path) -> int:
    for row in matrix():
        case = root / row["case"]
        status = qualification.read_json(case / "qualification_status.json")
        status.update(qualification.progress(case))
        source = qualification.read_json(case / SOURCE_MANIFEST)
        extension = float(status.get("crack_extension_um") or 0.0)
        output = case / "output"
        size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        print(row["case"], json.dumps({
            "status": status.get("status"), "pid": status.get("pid"),
            "cycles_reached": status.get("cycles_reached"),
            "event_count": status.get("event_count"), "total_extension_um": extension,
            "extension_added_um": extension - float(source.get("starting_extension_um", 0.0)),
            "current_mode": status.get("current_mode"), "current_phase": status.get("current_phase"),
            "latest_physical_progress_timestamp": status.get("latest_physical_progress_timestamp"),
            "latest_liveness_timestamp": status.get("latest_liveness_timestamp"),
            "restart_count": status.get("restart_count"), "output_size_bytes": size,
        }, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    for command in ("preflight", "prepare"):
        child = sub.add_parser(command); child.add_argument("source", type=Path); child.add_argument("destination", type=Path)
        child.add_argument("--minimum-free-gib", type=float, default=10.0)
    runp = sub.add_parser("run"); runp.add_argument("root", type=Path)
    runp.add_argument("--minimum-free-gib", type=float, default=10.0)
    runp.add_argument("--no-progress-seconds", type=float, default=900.0)
    runp.add_argument("--recover-stale-lock", action="store_true")
    mon = sub.add_parser("monitor"); mon.add_argument("root", type=Path)
    stop = sub.add_parser("stop"); stop.add_argument("root", type=Path)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(preflight(args.source, args.destination, args.minimum_free_gib), indent=2, sort_keys=True)); return 0
    if args.command == "prepare":
        print(json.dumps(prepare(args.source, args.destination, args.minimum_free_gib), indent=2, sort_keys=True)); return 0
    if args.command == "run": return run(args.root.resolve(), args)
    if args.command == "monitor": return monitor(args.root.resolve())
    return qualification.stop_launcher(args.root.resolve())


if __name__ == "__main__": raise SystemExit(main())
