#!/usr/bin/env python3
"""Launch or resume the single authorized V12 1000-um cap-six trajectory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(
    "/opt/homebrew/Caskroom/miniconda/base/envs/"
    "arrhenius-sharp-front-v10-codex/bin/python"
)
LABEL = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
BRANCH = "codex/current-source-general-multifront-v12-1000um-cap6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(output: Path, checkpoint: Path, family: Path) -> list[str]:
    return [
        str(PYTHON), "-u", "-m",
        "arrhenius_fracture.sharp_front_current_source_multifront_physical_v12",
        "--maximum-fronts", "6",
        "--v12-restart-checkpoint", str(checkpoint),
        "--v12-target-maximum-forward-reach-um", "1000",
        "--signed-kernel-family", str(family),
        "--mode", "2d",
        "--parameter-option", "v913_paper_weakT01_0129902_persistent_sites",
        "--temperatures", "700", "--steps", "2000000",
        "--nx", "36", "--ny", "72", "--dU", "2e-7", "--dt", "8.4",
        "--n-stagger", "2", "--tip-h-fine", "1e-6", "--tip-ratio", "1.2",
        "--da-phys", "5e-6", "--target-crack-extension-um", "1000",
        "--mpz-length-um", "50", "--mpz-n-bins", "80",
        "--front-state-model", "moving_pz", "--tip-source-model", "continuum",
        "--tip-kinetics-mode", "moving_velocity", "--bulk-plasticity-mode", "tip_only",
        "--directional-j-mode", "root_signed", "--tip-plasticity",
        "--active-shielding", "--signed-active-shielding",
        "--mobile-shield-fraction", "0", "--no-wake-shielding",
        "--crystal-aniso", "--crystal-compete", "--crystal-theta-deg", "40",
        "--crystal-material", "w", "--j-decomposition", "cluster",
        "--crack-backend", "sharp_wake", "--adaptive-events",
        "--adaptive-event-target", "0.15", "--print-every", "200",
        "--save-snapshots", "0", "--no-plots", "--out", str(output),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--expected-family-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    family = args.family.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    branch = subprocess.check_output(
        ("git", "branch", "--show-current"), cwd=ROOT, text=True
    ).strip()
    head = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()
    if branch != BRANCH:
        raise RuntimeError(f"execution branch mismatch: {branch}")
    if subprocess.check_output(
        ("git", "status", "--porcelain"), cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("execution source must be clean and committed")
    if sha256(family) != args.expected_family_sha256:
        raise RuntimeError("qualified family SHA-256 mismatch")
    if sha256(checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError("migrated checkpoint SHA-256 mismatch")
    canonical_checkpoint = output / "checkpoint" / "latest.v12.pkl"
    if checkpoint != canonical_checkpoint:
        raise RuntimeError("launch checkpoint must be the output's atomic restart path")
    if (output / "v12_run_complete.json").exists():
        raise RuntimeError("trajectory already has a terminal record")
    cmd = command(output, checkpoint, family)
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1", "CONDA_ENV": "arrhenius-sharp-front-v10-codex",
        "CONDA_DEFAULT_ENV": "arrhenius-sharp-front-v10-codex",
        "PARAMETER_CAMPAIGN": "1", "CLEAVAGE_HAZARD_MODE": "exponential",
        "CLEAVAGE_HAZARD_SEED": "3621",
        "CLEAVAGE_EVENT_LENGTH_MODE": "threshold_scaled",
        "CLEAVAGE_EVENT_MIN_FACTOR": "0.5", "CLEAVAGE_EVENT_MAX_FACTOR": "4.0",
        "CLEAVAGE_EVENT_SUBSEGMENT_FRACTION": "0.1",
        "ANISOTROPIC_TRANSPORT_MODE": "validated_scalar",
        "ANISOTROPIC_USE_AVALANCHE_BACKEND": "1",
        "ANISOTROPIC_EMISSION_ENABLED": "1",
        "KERNEL_STRICT_FAMILY_OVERRIDE": "1",
        "SIGNED_KERNEL_FAMILY_JSON": str(family),
        "MECHANICAL_CONFIG": str(family.parent / "mechanical_configuration.json"),
        "PERSISTENT_SOURCE_MIN_WIDTH_UM": "0",
        "ONED_V2_TP_STATE_DIAGNOSTICS": "events",
        "MPLCONFIGDIR": "/private/tmp/pf-current-source-v12-1000um-mpl",
    })
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "v12.1000um-cap6-launch/1", "claim_label": LABEL,
        "branch": branch, "execution_commit": head,
        "family": str(family), "family_sha256": sha256(family),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "maximum_fronts": 6, "branch_transaction_limit": None,
        "target_maximum_network_forward_reach_um": 1000.0,
        "hazard_seed": 3621, "temperature_K": 700.0, "theta_deg": 40.0,
        "command": cmd, "started_utc": datetime.now(timezone.utc).isoformat(),
        "predictive_recursive_branching_physics_validated": False,
    }
    (output / "v12_1000um_launch_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    with (output / "worker.stdout.log").open("a") as stdout, (
        output / "worker.stderr.log"
    ).open("a") as stderr:
        completed = subprocess.run(
            cmd, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr
        )
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["returncode"] = completed.returncode
    (output / "v12_1000um_launch_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
