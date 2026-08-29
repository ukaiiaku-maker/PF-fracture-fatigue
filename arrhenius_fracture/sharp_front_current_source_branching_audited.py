"""Fail-closed entry for the bounded current-source matched branching pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


MODEL_ID = "pf.current_source.signed_dislocation_atomic_multifront/1"
CLAIM_LABEL = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
PHYSICAL_SOURCE_COMMIT = "9e884fb0b0845da621d2612bdf1042e481b8df49"
TOPOLOGY_OVERLAY_SOURCE_COMMIT = "2b5e535"
AUDIT_NAME = "pf_current_source_branching_model_audit.json"
FORBIDDEN = {
    "--fatigue-cycles", "--fatigue-hold-load", "--fixed-delta-k",
    "--full-field-plasticity", "--plastic-dominance-censor",
    "--adaptive-czm-branching", "--absolute-directional-j",
    "--legacy-root-sign-latching", "--branch-probability", "--branch-ratio",
    "--clone-split", "--mpz-partition", "--topology-interpolation",
}


def _value(args, name, default=None):
    values = []
    for index, token in enumerate(args):
        if token.startswith(name + "="):
            values.append(token.split("=", 1)[1])
        if token == name and index + 1 < len(args):
            values.append(args[index + 1])
    if len(values) > 1:
        raise SystemExit(f"duplicate launch argument is forbidden: {name}")
    return values[0] if values else default


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(args):
    if "--current-source-branching-capability" not in args:
        raise SystemExit("current-source capability flag is required")
    bad = sorted({token.split("=", 1)[0] for token in args} & FORBIDDEN)
    if bad:
        raise SystemExit("forbidden branching controls: " + ", ".join(bad))
    if _value(args, "--mode", "2d") != "2d":
        raise SystemExit("current-source capability requires 2d mode")
    if _value(args, "--crack-backend", "sharp_wake") != "sharp_wake":
        raise SystemExit("current-source capability requires sharp_wake")
    if _value(args, "--bulk-plasticity-mode", "tip_only") != "tip_only":
        raise SystemExit("current-source capability requires tip_only")
    fronts = int(_value(args, "--maximum-fronts", "2"))
    if fronts not in (1, 2):
        raise SystemExit("matched capability cases require maximum-fronts 1 or 2")
    return fronts


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    fronts = validate(args)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", required=True)
    known, _ = parser.parse_known_args(args)
    root = Path(__file__).resolve().parents[1]
    out = Path(known.out)
    out.mkdir(parents=True, exist_ok=True)
    registry = root / "runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_registry.csv"
    selection = root / "runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_selection.json"
    payload = {
        "schema": "pf_current_source_branching_model_audit_v1",
        "claim_label": CLAIM_LABEL,
        "model_id": MODEL_ID,
        "qualified_physical_source_commit": PHYSICAL_SOURCE_COMMIT,
        "ported_atomic_topology_overlay_source_commit": TOPOLOGY_OVERLAY_SOURCE_COMMIT,
        "historical_v11_executable_invoked": False,
        "historical_v11_material_row_invoked": False,
        "producer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty_tree_at_launch": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "maximum_fronts": fronts,
        "branching_enabled": fronts == 2,
        "material_candidate": "oneD_v2_focused_weak_T_0016",
        "material_option": _value(args, "--parameter-option"),
        "temperature_K": float(_value(args, "--temperatures", "700")),
        "theta_deg": float(_value(args, "--crystal-theta-deg", "40")),
        "hazard_seed": int(os.environ["CLEAVAGE_HAZARD_SEED"]),
        "registry_sha256": _sha256(registry),
        "selection_sha256": _sha256(selection),
        "policy": {
            "branch_nucleation": "directional_first_passage_plus_atomic_whole_topology_energy_transaction",
            "process_state": "shared_unresolved_cluster_then_independent_tip_handoff",
            "assigned_branch_probability": False,
            "heuristic_process_zone_split": False,
            "topology_interpolation": False,
            "directional_J": "positive_signed_raw",
            "target_metric": "maximum_forward_reach_over_all_fronts",
        },
        "argv": args,
    }
    (out / AUDIT_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    forwarded = [token for token in args if token != "--current-source-branching-capability"]
    from .sharp_front_current_source_branching import main as production_main
    try:
        return production_main(forwarded)
    except BaseException as exc:
        (out / "failure_traceback.txt").write_text(traceback.format_exc())
        (out / "failure_summary.json").write_text(json.dumps({
            "exception_class": type(exc).__name__, "exception_message": str(exc),
            "claim_label": CLAIM_LABEL,
        }, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
