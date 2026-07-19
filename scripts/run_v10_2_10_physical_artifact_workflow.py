#!/usr/bin/env python3
"""Plan, review-build, authorize, and audit v10.2.10 mechanics artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from arrhenius_fracture.physical_artifact_workflow_v10210 import (
    MODEL_ID,
    inspect_raw_artifacts,
    load_state_requests,
    readiness_report,
    validate_review_approval,
    write_collection_plan,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def _parse_magnitudes(raw: str) -> list[float]:
    values = [float(token) for token in raw.replace(",", " ").split()]
    if not values:
        raise argparse.ArgumentTypeError("at least two magnitudes are required")
    return values


def _run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _build(
    args: argparse.Namespace,
    *,
    authorize: bool,
    approval: dict | None,
) -> dict:
    preflight = inspect_raw_artifacts(
        signed_responses=args.signed_responses,
        tensor_responses=args.tensor_responses,
        normalization=args.normalization,
    )
    outroot = args.outroot
    if outroot.exists():
        raise SystemExit(f"refusing to overwrite output: {outroot}")
    outroot.mkdir(parents=True)
    kernel = outroot / (
        "state_resolved_signed_kernel_family_authorized.json"
        if authorize
        else "state_resolved_signed_kernel_family_review.json"
    )
    drive = outroot / (
        "state_resolved_signed_drive_family_authorized.json"
        if authorize
        else "state_resolved_signed_drive_family_review.json"
    )
    kernel_command = [
        sys.executable,
        "scripts/build_v10_2_6_state_resolved_kernel_family.py",
        "--responses",
        str(args.signed_responses),
        "--normalization",
        str(args.normalization),
        "--out",
        str(kernel),
        "--relative-linearity-tolerance",
        str(args.relative_linearity_tolerance),
        "--fixed-kernel-tolerance",
        str(args.fixed_kernel_tolerance),
        "--minimum-distinct-r",
        str(args.minimum_distinct_r),
        "--minimum-distinct-opening",
        str(args.minimum_distinct_opening),
        "--minimum-distinct-extension",
        str(args.minimum_distinct_extension),
    ]
    if authorize:
        kernel_command.append("--authorize-production-parameterization")
    _run(kernel_command)

    drive_command = [
        sys.executable,
        "scripts/build_v10_2_7_state_resolved_drive_family.py",
        "--responses",
        str(args.tensor_responses),
        "--kernel-family",
        str(kernel),
        "--out",
        str(drive),
        "--relative-repeat-tolerance",
        str(args.relative_repeat_tolerance),
    ]
    if authorize:
        drive_command.append("--authorize-production-parameterization")
    _run(drive_command)

    payload = {
        "schema": MODEL_ID,
        "stage": "authorized_build" if authorize else "review_build",
        "raw_preflight": preflight,
        "independent_review": approval,
        "kernel_family": str(kernel),
        "drive_family": str(drive),
        "production_parameterization_allowed": bool(authorize),
        "automatic_authorization": False,
    }
    _write(outroot / "physical_artifact_workflow_complete.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="write a physical FEM collection manifest")
    plan.add_argument("--states", type=Path, required=True)
    plan.add_argument("--outroot", type=Path, required=True)
    plan.add_argument("--n-systems", type=int, default=2)
    plan.add_argument("--active-bins", type=int, required=True)
    plan.add_argument("--wake-bins", type=int, required=True)
    plan.add_argument(
        "--perturbation-magnitudes",
        type=_parse_magnitudes,
        default=[0.25, 0.50],
        help="space/comma separated positive signed-line magnitudes",
    )

    preflight = sub.add_parser("preflight", help="validate raw physical response inputs")
    preflight.add_argument("--signed-responses", type=Path, required=True)
    preflight.add_argument("--tensor-responses", type=Path, required=True)
    preflight.add_argument("--normalization", type=Path, required=True)
    preflight.add_argument("--out", type=Path, required=True)

    for name in ("build-review", "authorize"):
        build = sub.add_parser(name)
        build.add_argument("--signed-responses", type=Path, required=True)
        build.add_argument("--tensor-responses", type=Path, required=True)
        build.add_argument("--normalization", type=Path, required=True)
        build.add_argument("--outroot", type=Path, required=True)
        build.add_argument("--relative-linearity-tolerance", type=float, default=0.03)
        build.add_argument("--relative-repeat-tolerance", type=float, default=0.03)
        build.add_argument("--fixed-kernel-tolerance", type=float, default=0.05)
        build.add_argument("--minimum-distinct-r", type=int, default=2)
        build.add_argument("--minimum-distinct-opening", type=int, default=3)
        build.add_argument("--minimum-distinct-extension", type=int, default=2)
        if name == "authorize":
            build.add_argument("--independent-review", type=Path, required=True)

    ready = sub.add_parser("readiness")
    ready.add_argument("--kernel-family", type=Path, required=True)
    ready.add_argument("--drive-family", type=Path, required=True)
    ready.add_argument("--engine-template", type=Path, required=True)
    ready.add_argument("--out", type=Path, required=True)
    ready.add_argument("--require-ready", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "plan":
        payload = write_collection_plan(
            load_state_requests(args.states),
            outroot=args.outroot,
            n_systems=args.n_systems,
            active_bins=args.active_bins,
            wake_bins=args.wake_bins,
            perturbation_magnitudes=args.perturbation_magnitudes,
        )
        print(json.dumps(payload, indent=2))
        return
    if args.command == "preflight":
        _write(
            args.out,
            inspect_raw_artifacts(
                signed_responses=args.signed_responses,
                tensor_responses=args.tensor_responses,
                normalization=args.normalization,
            ),
        )
        return
    if args.command == "build-review":
        _build(args, authorize=False, approval=None)
        return
    if args.command == "authorize":
        approval = validate_review_approval(args.independent_review)
        _build(args, authorize=True, approval=approval)
        return
    if args.command == "readiness":
        payload = readiness_report(
            kernel_family=args.kernel_family,
            drive_family=args.drive_family,
            engine_template=args.engine_template,
        )
        _write(args.out, payload)
        if args.require_ready and not payload["ready_for_stage_1"]:
            raise SystemExit(2)
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
