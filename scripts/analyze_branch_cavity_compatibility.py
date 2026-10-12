#!/usr/bin/env python3
"""Condense the public branch--cavity interrupted/uninterrupted evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_report(uninterrupted: dict, restarted: dict) -> dict:
    same_fields = uninterrupted["shared_fields"] == restarted["shared_fields"]
    same_owners = uninterrupted["owners"] == restarted["owners"]
    exact = bool(restarted["reference_terminal_exact"])
    passed = all((
        uninterrupted["status"] == "PASS",
        restarted["status"] == "PASS",
        uninterrupted["native_checkpoint_round_trip_exact"],
        restarted["native_checkpoint_round_trip_exact"],
        exact,
        same_fields,
        same_owners,
        len(restarted["owners"]["branch_ids"]) >= 2,
        len(restarted["owners"]["cavity_ids"]) >= 1,
        restarted["crack_void_connection_certificate"]["passed"],
    ))
    terminal = uninterrupted["terminal"]
    return {
        "schema": "branch-cavity-compatibility-summary/1",
        "status": "PASS" if passed else "FAIL",
        "source_commit": "bb0c21d9",
        "public_preset": "pf_branching_void_v5",
        "public_restart_family": "v11_branch_cavity_compatibility",
        "terminal_payload_exact": exact,
        "native_checkpoint_round_trips_exact": bool(
            uninterrupted["native_checkpoint_round_trip_exact"]
            and restarted["native_checkpoint_round_trip_exact"]
        ),
        "shared_fields_exact": same_fields,
        "owner_registry_exact": same_owners,
        "owners": uninterrupted["owners"],
        "event_order_uninterrupted": uninterrupted["execution"]["event_sequence"],
        "event_order_restarted": restarted["execution"]["event_sequence"],
        "terminal": {
            key: terminal[key] for key in (
                "graph_length_m", "cavity_radius_m", "reaction_N_per_m",
                "energy_J_per_m", "void_phase", "active_crack_branch_ids",
                "event_counters",
            )
        },
        "field_transfer": uninterrupted["shared_fields"]["field_transfer_audit"],
        "cavity_switch": {
            "enabled_path_executed": uninterrupted["cavity_feedback_switch"] == "enabled",
            "branch_permission_changed_by_toggle": uninterrupted[
                "cavity_toggle_changes_branch_permission"
            ],
            "suppressed_route": "pf_branching_current remains the distinct cavity-disabled V12 route",
        },
        "scientific_scope": {
            "production_eligible": False,
            "recursive_branching_calibrated": False,
            "spontaneous_joint_birth_required": False,
            "interpretation": (
                "Compatibility qualification of existing full-field, topology, cavity, "
                "event-order, transfer, and restart laws; not a new branch-kinetics calibration."
            ),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninterrupted", type=Path, required=True)
    parser.add_argument("--restarted", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(
        json.loads(args.uninterrupted.read_text()),
        json.loads(args.restarted.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    owners = report["owners"]
    terminal = report["terminal"]
    args.out.with_suffix(".md").write_text(
        "# Combined branch--cavity compatibility\n\n"
        f"Status: **{report['status']}**. The public `pf_branching_void_v5` route "
        "executed from a fresh process, retained the root branch, downstream child, "
        "and cavity as separate owners, and reproduced the uninterrupted terminal "
        f"state {'exactly' if report['terminal_payload_exact'] else 'non-exactly'} after restart.\n\n"
        f"Owners: branches `{owners['branch_ids']}`, active tips "
        f"`{owners['active_tip_ids']}`, cavities `{owners['cavity_ids']}`. The "
        f"terminal graph length was {terminal['graph_length_m']:.9e} m and the "
        f"cavity radius was {terminal['cavity_radius_m']:.9e} m.\n\n"
        + report["scientific_scope"]["interpretation"] + "\n"
    )
    if report["status"] != "PASS":
        raise SystemExit("branch-cavity compatibility summary failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
