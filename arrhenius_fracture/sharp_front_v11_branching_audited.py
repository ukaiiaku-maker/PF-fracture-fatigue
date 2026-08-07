"""Fail-closed audited entry policy for bounded v11 mechanistic branching."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Sequence

from .live_topology_kernel_registry_v11 import PREBRANCH_PROVIDER_ID
from .live_topology_kernel_v11 import PROVIDER_ID


MODEL_ID = "v11.mechanistic_branching.monotonic_tip_only_sharp_wake_causal/2"
AUDIT_NAME = "v11_branching_model_audit.json"


@dataclass(frozen=True)
class AuditedBranchingPolicy:
    mode: str = "2d"
    loading: str = "monotonic"
    plasticity: str = "tip_only"
    crack_backend: str = "sharp_wake"
    crack_representation: str = "sharp_wake_causal_v11"
    maximum_fronts: int = 16
    maximum_branch_births: int = 8
    directional_J: str = "positive_signed_raw"
    branch_process_zone_mode: str = "shared_unresolved_cluster"
    prebranch_mechanics_provider: str = PREBRANCH_PROVIDER_ID
    branch_mechanics_provider: str = PROVIDER_ID
    independent_tip_handoff: str = "junction_reservoir_and_independent_tip_continuation"
    topology_interpolation: str = "disabled"


FORBIDDEN_FLAGS = {
    "--fatigue-cycles": "fatigue",
    "--fatigue-hold-load": "fatigue",
    "--fixed-delta-k": "fixed-delta-K fatigue",
    "--full-field-plasticity": "full-field plasticity",
    "--plastic-dominance-censor": "plastic-dominance censoring",
    "--adaptive-czm-branching": "adaptive-CZM branching",
    "--absolute-directional-j": "absolute directional J",
    "--legacy-root-sign-latching": "legacy root-sign latching",
    "--branch-probability": "legacy branch probabilities",
    "--branch-ratio": "legacy branch ratios",
    "--clone-split": "legacy clone_split",
    "--mpz-partition": "heuristic MPZ partitioning",
    "--topology-interpolation": "topology interpolation",
}


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=root, text=True).strip()


def _value(tokens: Sequence[str], name: str, default=None):
    prefix = name + "="
    for index, token in enumerate(tokens):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(tokens):
            return tokens[index + 1]
    return default


def validate_audited_arguments(argv: Sequence[str]) -> None:
    tokens = tuple(argv)
    if "--mechanistic-branching" not in tokens:
        raise SystemExit("ERROR: audited v11 entry requires --mechanistic-branching")
    for token in tokens:
        name = token.split("=", 1)[0]
        if name in FORBIDDEN_FLAGS:
            raise SystemExit(f"ERROR: audited v11 rejects {FORBIDDEN_FLAGS[name]} ({name})")
    required = {
        "--mode": "2d",
        "--crack-backend": "sharp_wake",
    }
    for name, expected in required.items():
        actual = _value(tokens, name, expected)
        if actual != expected:
            raise SystemExit(f"ERROR: audited v11 forces {name}={expected}, got {actual}")
    maximum = int(_value(tokens, "--maximum-fronts", "16"))
    if maximum != 16:
        raise SystemExit("ERROR: audited v11 forces --maximum-fronts=16")


def audit_payload(
    argv: Sequence[str], *, root: str | Path, provider_transition_state=None,
    topology_fingerprints=(), energy_tolerance: float = 1.0e-8,
) -> dict:
    base = Path(root).resolve()
    policy = AuditedBranchingPolicy()
    return {
        "schema": "v11.branching-model-audit/1",
        "git_head": _git(base, "rev-parse", "HEAD"),
        "dirty_tree": bool(_git(base, "status", "--porcelain")),
        "python_executable": str(Path(sys.executable).resolve()),
        "package_import_path": str(Path(__file__).resolve().parent),
        "model_id": MODEL_ID,
        "material_option": _value(argv, "--parameter-option", _value(argv, "--material-option")),
        "temperature_K": [float(value) for value in (_value(argv, "--temperatures", "").split()) if value],
        "orientation_deg": float(_value(
            argv, "--crystal-theta-deg", _value(argv, "--theta-deg", "30")
        )),
        "hazard_seed": int(os.environ.get(
            "CLEAVAGE_HAZARD_SEED", _value(argv, "--hazard-seed", "3621")
        )),
        "mechanics_provider_sequence": [
            policy.prebranch_mechanics_provider, policy.branch_mechanics_provider,
        ],
        "provider_transition_state": provider_transition_state,
        "topology_fingerprints": list(topology_fingerprints),
        "energy_tolerance": float(energy_tolerance),
        "policy": asdict(policy),
        "forbidden_legacy_controls": sorted(FORBIDDEN_FLAGS),
        "argv": list(argv),
    }


def write_model_audit(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, target)
    return target


def _write_failure(out: Path, error: BaseException) -> None:
    """Persist the complete production exception and latest atomic context."""
    context = {}
    checkpoint = out / "checkpoint" / "latest.json"
    if checkpoint.is_file():
        try:
            manifest = json.loads(checkpoint.read_text())
            context = {
                "step": manifest.get("event_counters", {}).get("accepted_steps"),
                "event_index": manifest.get("event_counters", {}).get("topology_actions"),
                "physical_time_s": manifest.get("physical_time_s"),
                "accepted_load": manifest.get("accepted_load"),
                "topology_fingerprint": manifest.get("topology_fingerprint"),
                "mechanics_provider": manifest.get("mechanics_provider"),
                "active_fronts": manifest.get("active_front_ids", []),
                "branch_count": len(manifest.get("crack_network", {}).get("branches", [])),
                "latest_checkpoint": str(checkpoint.resolve()),
                "latest_successful_action": manifest.get("event_counters", {}).get("latest_successful_action"),
            }
            from .branch_checkpoint_v11 import restore_branch_checkpoint
            from .branch_snapshot_v11 import write_topology_snapshot
            from .network_metrics_v11 import crack_growth_metrics
            restored = restore_branch_checkpoint(checkpoint)
            from .production_counts_v11 import production_front_counts
            context.update(production_front_counts(restored.state))
            context.pop("branch_count", None)
            root_branch = restored.state.crack_network.branch(restored.state.crack_network.primary_branch_id)
            initial_length = (
                ((root_branch.path[1][0] - root_branch.path[0][0]) ** 2 +
                 (root_branch.path[1][1] - root_branch.path[0][1]) ** 2) ** 0.5
                if len(root_branch.path) > 1 else 0.0
            )
            growth = crack_growth_metrics(
                restored.state.crack_network, initial_crack_length_m=initial_length,
            )
            write_topology_snapshot(
                out, restored.state,
                step=int(context.get("step") or 0), reason="failure",
                physical_extension_m=float(restored.physical_extension_m),
                branch_birth_count=int(restored.state.event_counters.get("branch_birth_count", 0)),
                latest_action=context.get("latest_successful_action"),
                growth_metrics=growth.to_dict_um(),
                coalescence_count=int(restored.state.event_counters.get("coalescence_count", 0)),
            )
        except (OSError, ValueError, TypeError):
            context = {"latest_checkpoint": str(checkpoint.resolve())}
    summary = {
        "schema": "v11.production-failure-summary/1",
        "exception_class": type(error).__name__,
        "exception_message": str(error),
        **context,
    }
    write_model_audit(out / "failure_summary.json", summary)
    trace = out / "failure_traceback.txt"
    temporary = trace.with_name(trace.name + ".tmp")
    temporary.write_text(traceback.format_exc())
    os.replace(temporary, trace)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    validate_audited_arguments(args)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-only", action="store_true")
    known, _ = parser.parse_known_args(args)
    root = Path(__file__).resolve().parents[1]
    write_model_audit(Path(known.out) / AUDIT_NAME, audit_payload(args, root=root))
    if known.audit_only:
        return 0
    from .sharp_front_v11_branching import main as production_main
    try:
        return production_main(args, audit_already_written=True)
    except BaseException as error:
        _write_failure(Path(known.out), error)
        raise


if __name__ == "__main__":
    main()


__all__ = [
    "AUDIT_NAME", "AuditedBranchingPolicy", "FORBIDDEN_FLAGS", "MODEL_ID",
    "audit_payload", "main", "validate_audited_arguments", "write_model_audit",
]
