#!/usr/bin/env python3
"""Bind R10 to native cavity evidence and optionally resume its accepted state.

The compact qualification is intentionally manifest-first: it verifies the
immutable native checkpoint payload and the already retained, source-bound
physics reports without unpickling a large FEM state.  ``--execute`` delegates
the expensive daughter-birth/continuation work to the native restart runner;
it never substitutes a reduced or injected state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DURABLE_CHECKPOINT = ROOT / (
    "common_framework_readiness/acceptance_review_20261015/"
    "r10_durable_checkpoint/immediately-before-child-birth.json"
)
CHECKPOINT_GLOB = Path(
    "/private/var/folders/zm/6j1kp1m523d_mt4vlnq45zdr0000gn/T"
).glob("unified-one-void-*/immediately-before-child-birth.json")
LIGAMENT_REPORT = ROOT / (
    "review_packet/current_science_pilots_6abbf10/"
    "CAVITY_LIGAMENT_QUALIFICATION_v4.json"
)
INTERACTION_REPORT = ROOT / (
    "review_packet/final_fracture_campaign_a5a5883/"
    "CAVITY_INTERACTION_0P2S_RECOVERED.json"
)
COMBINED_REPORT = ROOT / (
    "review_packet/final_fracture_campaign_a5a5883/"
    "BRANCH_CAVITY_COMPATIBILITY.json"
)


def _load(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_checkpoint() -> Path | None:
    if DURABLE_CHECKPOINT.is_file():
        return DURABLE_CHECKPOINT
    candidates = [path for path in CHECKPOINT_GLOB if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def checkpoint_evidence(path: Path) -> dict[str, Any]:
    path = path.resolve()
    manifest = _load(path)
    if manifest.get("schema") != "v11.live-fem-branch-checkpoint/1":
        raise ValueError("R10 input is not a native V11/V12 live-FEM checkpoint")
    state_file = path.with_name(str(manifest.get("state_file", "")))
    if not state_file.is_file():
        raise ValueError("R10 checkpoint payload is absent")
    actual_sha = _sha256(state_file)
    if actual_sha != manifest.get("state_sha256"):
        raise ValueError("R10 checkpoint payload hash does not match its manifest")
    cavities = manifest.get("production_void_state", {}).get("cavities", [])
    if len(cavities) != 1:
        raise ValueError("R10 checkpoint must contain exactly one native cavity")
    cavity = cavities[0]
    if cavity.get("phase") != "CONNECTED_VOID":
        raise ValueError("R10 checkpoint is not at the accepted connected-cavity state")
    if cavity.get("connection_entry_m") is None or cavity.get("connection_exit_m") is None:
        raise ValueError("R10 connected cavity lacks a physical entry/exit pair")
    if manifest.get("active_tip_ids"):
        raise ValueError("R10 pre-birth checkpoint unexpectedly has an active crack tip")
    ledgers = manifest["production_void_state"].get("length_ledgers", {})
    required_ledgers = {
        "fractured_ligament_length_m", "connected_void_free_span_m",
        "preexisting_void_free_span_m", "active_front_coordinate_advance_m",
    }
    if not required_ledgers.issubset(ledgers):
        raise ValueError("R10 checkpoint lacks separate crack/cavity length ledgers")
    equilibrium = manifest.get("equilibrium_observables") or {}
    return {
        "status": "NATIVE_CONNECTED_ACCEPTED_STATE_VERIFIED",
        "manifest": str(path),
        "payload": str(state_file),
        "payload_sha256": actual_sha,
        "payload_size_bytes": state_file.stat().st_size,
        "source_commit": manifest.get("v12_support_state", {}).get("source_commit"),
        "sharp_wake_model_id": manifest.get("sharp_wake_model_id"),
        "topology_transaction_model_id": manifest.get("topology_transaction_model_id"),
        "mesh_generation": manifest.get("event_counters", {}).get("mesh_generation"),
        "topology_actions": manifest.get("event_counters", {}).get("topology_actions"),
        "active_tip_ids": manifest.get("active_tip_ids"),
        "cavity": {
            "id": cavity.get("cavity_id"), "phase": cavity.get("phase"),
            "radius_m": cavity.get("radius_m"),
            "connection_entry_m": cavity.get("connection_entry_m"),
            "connection_exit_m": cavity.get("connection_exit_m"),
        },
        "length_ledgers": {name: ledgers[name] for name in sorted(required_ledgers)},
        "equilibrium": {
            "free_dof_residual_relative": equilibrium.get("free_dof_residual_relative"),
            "energy_identity_residual": equilibrium.get("energy_identity_residual"),
            "total_opening_m": equilibrium.get("total_opening_m"),
        },
    }


def retained_evidence() -> dict[str, Any]:
    ligament = _load(LIGAMENT_REPORT)
    interaction = _load(INTERACTION_REPORT)
    combined = _load(COMBINED_REPORT)
    if ligament.get("status") != "QUALIFIED_COMPLETE_LIGAMENT_OPERATION":
        raise ValueError("retained complete-ligament operation is not qualified")
    if ligament.get("local_J_substituted") is not False:
        raise ValueError("retained ligament drive substituted a local J")
    arms = interaction.get("arms", {})
    if not {"enabled", "disabled"}.issubset(arms):
        raise ValueError("retained cavity interaction lacks its matched control")
    if not all(arms[name].get("checkpoint_fingerprint_exact") for name in arms):
        raise ValueError("retained cavity interaction restart is not exact")
    if interaction.get("accepted_exposure_s", 0.0) <= 0.0:
        raise ValueError("retained cavity interaction has no finite interval")
    if combined.get("status") != "PASS":
        raise ValueError("retained branch/cavity compatibility did not pass")
    return {
        "complete_ligament": {
            "report": str(LIGAMENT_REPORT.relative_to(ROOT)),
            "drive_source": ligament.get("drive_source"),
            "G_ligament_J_per_m2": ligament.get("G_ligament_J_per_m2"),
            "mesh_relative_error": ligament.get("mesh_relative_error"),
            "mesh_relative_limit": ligament.get("mesh_relative_limit"),
            "accepted_state_preserved": ligament.get("accepted_state_preserved"),
            "current_code_role": (
                "ACTUAL_KINETIC_RATE_ARGUMENT_FOR_CAVITY_INTERSECTING_CANDIDATE"
            ),
            "selected_local_incremental_conformance": False,
            "classification": "COLLECTIVE_FINITE_LIGAMENT_ALTERNATIVE",
            "consumer_source": (
                "arrhenius_fracture.voiding_production_v5."
                "dispatch_cavity_intersecting_candidate_drives -> "
                "directional_sharp_front_rates"
            ),
        },
        "finite_enabled_suppressed_interval": {
            "report": str(INTERACTION_REPORT.relative_to(ROOT)),
            "classification": interaction.get("classification"),
            "accepted_exposure_s": interaction.get("accepted_exposure_s"),
            "native_checkpoint_round_trip_exact": interaction.get(
                "native_checkpoint_round_trip_exact"
            ),
            "enabled_topology_action_change": arms["enabled"].get("topology_action_change"),
            "disabled_topology_action_change": arms["disabled"].get("topology_action_change"),
        },
        "combined_branch_cavity": {
            "report": str(COMBINED_REPORT.relative_to(ROOT)),
            "status": combined.get("status"),
            "source_commit": combined.get("source_commit"),
            "native_checkpoint_round_trips_exact": combined.get(
                "native_checkpoint_round_trips_exact"
            ),
            "active_tip_ids": combined.get("owners", {}).get("active_tip_ids"),
            "terminal_void_phase": combined.get("terminal", {}).get("void_phase"),
        },
    }


def qualification(checkpoint: Path | None) -> dict[str, Any]:
    evidence = retained_evidence()
    connected = None if checkpoint is None else checkpoint_evidence(checkpoint)
    next_command = None
    if connected is not None:
        next_command = [
            sys.executable, "scripts/continue_unified_one_void_from_checkpoint.py",
            "--checkpoint", connected["manifest"], "--output", "<new-output>",
        ]
    return {
        "schema": "r10.cavity-kinetic-role-and-restart-qualification/2",
        "status": (
            "COLLECTIVE_LIGAMENT_RESTART_READY_LOCAL_INCREMENTAL_PENDING"
            if connected is not None else "INCOMPLETE_NATIVE_CHECKPOINT_UNAVAILABLE"
        ),
        "production_execution_authorized": False,
        "qualification_only_constructed_inputs": False,
        "retained_native_evidence": evidence,
        "connected_checkpoint": connected,
        "next_native_command": next_command,
        "execution_boundary": (
            "The next command restores the exact occupied accepted state and performs "
            "native downstream birth, ordinary continuation, post-event equilibrium, "
            "rollback injections and exact active-state checkpoint round trips."
        ),
        "not_claimed": [
            "No downstream child acceptance is claimed until the native continuation exits PASS.",
            "The current complete-ligament rate does not qualify the selected local-incremental law.",
            "The recovered 0.2 s exporter failure does not identify individual event times.",
            "No production campaign or physical recalibration is authorized.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--continuation-output", type=Path)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint or discover_checkpoint()
    report = qualification(checkpoint)
    if args.execute:
        if checkpoint is None or args.continuation_output is None:
            parser.error("--execute requires an available checkpoint and --continuation-output")
        command = [
            sys.executable, str(ROOT / "scripts/continue_unified_one_void_from_checkpoint.py"),
            "--checkpoint", str(checkpoint.resolve()),
            "--output", str(args.continuation_output.resolve()),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        child_report = _load(args.continuation_output / "qualification_report.json")
        if child_report.get("status") != "PASS":
            raise RuntimeError("native R10 continuation did not report PASS")
        report["status"] = (
            "COLLECTIVE_LIGAMENT_CHILD_CONTINUATION_COMPLETE_"
            "LOCAL_INCREMENTAL_PENDING"
        )
        report["native_continuation_report"] = str(
            (args.continuation_output / "qualification_report.json").resolve()
        )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if checkpoint is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
