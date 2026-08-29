#!/usr/bin/env python3
"""Fork a current-source branching checkpoint into a resumable output segment.

The source case and its provider cache are read-only evidence.  Every restart
uses a new output directory and a new live-kernel cache, so interrupted or
extended runs form an explicit lineage of immutable segments.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from arrhenius_fracture.branching_qualification_v2 import CLAIM_LABEL


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python")
ENTRY = "arrhenius_fracture.sharp_front_current_source_branching_audited"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big")); digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def _replace_value(tokens: list[str], name: str, value: str) -> list[str]:
    result: list[str] = []
    found = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == name:
            if found or index + 1 >= len(tokens):
                raise ValueError(f"invalid or duplicate source argument: {name}")
            result.extend((name, value)); found = True; index += 2; continue
        if token.startswith(name + "="):
            if found:
                raise ValueError(f"duplicate source argument: {name}")
            result.extend((name, value)); found = True; index += 1; continue
        result.append(token); index += 1
    if not found:
        result.extend((name, value))
    return result


def build_restart_command(
    checkpoint_path: Path, destination: Path, target_extension_um: float,
) -> tuple[list[str], dict]:
    checkpoint_path = checkpoint_path.resolve()
    destination = destination.resolve()
    if not checkpoint_path.is_file():
        raise ValueError(f"restart checkpoint is missing: {checkpoint_path}")
    if destination.exists():
        raise ValueError(f"restart output must be a fresh path: {destination}")
    checkpoint = restore_branch_checkpoint(checkpoint_path)
    if target_extension_um * 1e-6 <= checkpoint.projected_extension_m:
        raise ValueError("restart target must exceed checkpoint projected extension")
    source_case = checkpoint_path.parent.parent
    audit_path = source_case / "pf_current_source_branching_model_audit.json"
    if not audit_path.is_file():
        raise ValueError("source case lacks the current-source model audit")
    audit = json.loads(audit_path.read_text())
    if audit.get("claim_label") != CLAIM_LABEL:
        raise ValueError("source checkpoint is outside the permanent claim boundary")
    args = list(audit.get("argv", ()))
    if "--v11-restart-checkpoint" in args or any(
        token.startswith("--v11-restart-checkpoint=") for token in args
    ):
        # A later segment is still resumable, but its source command must be
        # normalized to the newly selected checkpoint below.
        normalized: list[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            if token == "--v11-restart-checkpoint": index += 2; continue
            if token.startswith("--v11-restart-checkpoint="): index += 1; continue
            normalized.append(token); index += 1
        args = normalized
    args = _replace_value(args, "--out", str(destination))
    args = _replace_value(args, "--target-crack-extension-um", f"{target_extension_um:.17g}")
    args.extend(("--v11-restart-checkpoint", str(checkpoint_path)))
    command = [str(PYTHON), "-u", "-m", ENTRY, *args]
    source_manifest = json.loads(checkpoint_path.read_text())
    plan = {
        "schema": "pf_current_source_branching_restart_plan_v2",
        "claim_label": CLAIM_LABEL,
        "source_case": str(source_case),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_manifest_sha256": sha256(checkpoint_path),
        "source_checkpoint_state_sha256": source_manifest["state_sha256"],
        "source_projected_extension_um": checkpoint.projected_extension_m * 1e6,
        "source_accepted_steps": int(checkpoint.state.event_counters.get("accepted_steps", 0)),
        "source_physical_time_s": checkpoint.physical_time_s,
        "source_accepted_opening_m": checkpoint.accepted_load,
        "source_termination_reason": checkpoint.termination_reason,
        "destination": str(destination),
        "target_extension_um": float(target_extension_um),
        "fresh_output_fork_required": True,
        "source_provider_cache_immutable": True,
        "destination_provider_cache_rebound": True,
        "command": command,
    }
    return command, plan


def restart_environment(command: list[str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1",
        "CONDA_ENV": "arrhenius-sharp-front-v10-codex",
        "CONDA_DEFAULT_ENV": "arrhenius-sharp-front-v10-codex",
        "PARAMETER_CAMPAIGN": "1", "CLEAVAGE_HAZARD_MODE": "exponential",
        "CLEAVAGE_HAZARD_SEED": "3621",
        "CLEAVAGE_EVENT_LENGTH_MODE": "threshold_scaled",
        "CLEAVAGE_EVENT_MIN_FACTOR": "0.5", "CLEAVAGE_EVENT_MAX_FACTOR": "4.0",
        "CLEAVAGE_EVENT_SUBSEGMENT_FRACTION": "0.1",
        "ANISOTROPIC_TRANSPORT_MODE": "validated_scalar",
        "ANISOTROPIC_USE_AVALANCHE_BACKEND": "1", "ANISOTROPIC_EMISSION_ENABLED": "1",
        "KERNEL_STRICT_FAMILY_OVERRIDE": "1", "PERSISTENT_SOURCE_MIN_WIDTH_UM": "0",
        "ONED_V2_TP_STATE_DIAGNOSTICS": "events",
        "MPLCONFIGDIR": "/private/tmp/pf-current-source-branching-mpl",
    })
    family = Path(command[command.index("--signed-kernel-family") + 1]).resolve()
    mechanical = family.parent / "mechanical_configuration.json"
    if not family.is_file() or not mechanical.is_file():
        raise ValueError("restart family or mechanical configuration is missing")
    env["SIGNED_KERNEL_FAMILY_JSON"] = str(family)
    env["MECHANICAL_CONFIG"] = str(mechanical)
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-crack-extension-um", type=float, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    command, plan = build_restart_command(
        args.checkpoint, args.out, args.target_crack_extension_um,
    )
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True)); return 0

    source_case = Path(plan["source_case"])
    before = tree_fingerprint(source_case)
    args.out.mkdir(parents=True, exist_ok=False)
    plan["source_case_tree_sha256_before"] = before
    plan["started_utc"] = datetime.now(timezone.utc).isoformat()
    (args.out / "restart_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    with (args.out / "run.stdout.log").open("w") as stdout, (args.out / "run.stderr.log").open("w") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=restart_environment(command), stdout=stdout, stderr=stderr)
    after = tree_fingerprint(source_case)
    completion = {
        **plan, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": result.returncode, "source_case_tree_sha256_after": after,
        "source_case_immutable_verified": before == after,
    }
    (args.out / "restart_result.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    if before != after:
        raise RuntimeError("restart mutated its immutable source case")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
