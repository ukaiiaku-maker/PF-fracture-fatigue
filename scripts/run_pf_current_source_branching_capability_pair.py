#!/usr/bin/env python3
"""Launch the fresh current-source max-fronts 1/2 matched pair."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python")
LABEL = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(out: Path, family: Path, maximum_fronts: int, theta: float) -> list[str]:
    return [
        str(PYTHON), "-u", "-m",
        "arrhenius_fracture.sharp_front_current_source_branching_audited",
        "--current-source-branching-capability",
        "--maximum-fronts", str(maximum_fronts),
        "--signed-kernel-family", str(family),
        "--mode", "2d",
        "--parameter-option", "v913_paper_weakT01_0129902_persistent_sites",
        "--temperatures", "700",
        "--steps", "2000000", "--nx", "36", "--ny", "72",
        "--dU", "2e-7", "--dt", "8.4", "--n-stagger", "2",
        "--tip-h-fine", "1e-6", "--tip-ratio", "1.2",
        "--da-phys", "5e-6", "--target-crack-extension-um", "300",
        "--mpz-length-um", "50", "--mpz-n-bins", "80",
        "--front-state-model", "moving_pz", "--tip-source-model", "continuum",
        "--tip-kinetics-mode", "moving_velocity", "--bulk-plasticity-mode", "tip_only",
        "--directional-j-mode", "root_signed", "--tip-plasticity",
        "--active-shielding", "--signed-active-shielding",
        "--mobile-shield-fraction", "0", "--no-wake-shielding",
        "--crystal-aniso", "--crystal-compete",
        "--crystal-theta-deg", f"{theta:g}", "--crystal-material", "w",
        "--j-decomposition", "cluster", "--crack-backend", "sharp_wake",
        "--adaptive-events", "--adaptive-event-target", "0.15",
        "--print-every", "200", "--save-snapshots", "0", "--no-plots",
        "--out", str(out),
    ]


def run_case(out: Path, family: Path, maximum_fronts: int, theta: float) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    cmd = command(out, family, maximum_fronts, theta)
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT),
        "PYTHONUNBUFFERED": "1",
        "CONDA_ENV": "arrhenius-sharp-front-v10-codex",
        "CONDA_DEFAULT_ENV": "arrhenius-sharp-front-v10-codex",
        "PARAMETER_CAMPAIGN": "1",
        "CLEAVAGE_HAZARD_MODE": "exponential",
        "CLEAVAGE_HAZARD_SEED": "3621",
        "CLEAVAGE_EVENT_LENGTH_MODE": "threshold_scaled",
        "CLEAVAGE_EVENT_MIN_FACTOR": "0.5",
        "CLEAVAGE_EVENT_MAX_FACTOR": "4.0",
        "CLEAVAGE_EVENT_SUBSEGMENT_FRACTION": "0.1",
        "ANISOTROPIC_TRANSPORT_MODE": "validated_scalar",
        "ANISOTROPIC_USE_AVALANCHE_BACKEND": "1",
        "ANISOTROPIC_EMISSION_ENABLED": "1",
        "KERNEL_STRICT_FAMILY_OVERRIDE": "1",
        "SIGNED_KERNEL_FAMILY_JSON": str(family),
        "MECHANICAL_CONFIG": str(family.parent / "mechanical_configuration.json"),
        "PERSISTENT_SOURCE_MIN_WIDTH_UM": "0",
        "ONED_V2_TP_STATE_DIAGNOSTICS": "events",
        "MPLCONFIGDIR": "/private/tmp/pf-current-source-branching-mpl",
    })
    started = datetime.now(timezone.utc).isoformat()
    with (out / "run.stdout.log").open("w") as stdout, (out / "run.stderr.log").open("w") as stderr:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    payload = {
        "claim_label": LABEL,
        "case": "control" if maximum_fronts == 1 else "branching_enabled",
        "maximum_fronts": maximum_fronts,
        "theta_deg": theta,
        "hazard_seed": 3621,
        "family": str(family),
        "family_sha256": sha256(family),
        "command": cmd,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": result.returncode,
    }
    (out / "pair_case_result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--theta", type=float, default=40.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--enabled-only", action="store_true",
        help="Run only the prescribed max-fronts=2 orientation fallback.",
    )
    args = parser.parse_args()
    if args.workers not in (1, 2):
        raise SystemExit("workers must be one or two")
    family = args.family.resolve()
    if not family.is_file():
        raise SystemExit(f"qualified family missing: {family}")
    if args.outroot.exists():
        raise SystemExit(f"fresh output root already exists: {args.outroot}")
    args.outroot.mkdir(parents=True)
    specs = [
        (args.outroot / f"theta{args.theta:g}_control_max1_seed3621", 1),
        (args.outroot / f"theta{args.theta:g}_enabled_max2_seed3621", 2),
    ]
    if args.enabled_only:
        specs = specs[1:]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_case, path, family, maximum, args.theta)
            for path, maximum in specs
        ]
        results = [future.result() for future in futures]
    manifest = {
        "schema": (
            "pf_current_source_branching_fallback_launch_v1"
            if args.enabled_only else "pf_current_source_branching_pair_launch_v1"
        ),
        "claim_label": LABEL,
        "workers": args.workers,
        "enabled_only": args.enabled_only,
        "cases": results,
    }
    (args.outroot / "matched_pair_launch_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return 0 if all(item["returncode"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
