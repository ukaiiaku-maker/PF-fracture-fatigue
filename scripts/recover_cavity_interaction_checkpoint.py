#!/usr/bin/env python3
"""Recover the accepted interaction pair whose event audit export failed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.shared_v5_cavity_adapter import (
    _directional_provider, _restore_pair,
)
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint,
)
from arrhenius_fracture.voiding_production_v5 import observables


DEFAULT_RESULT = (
    ROOT / "integration_evidence" / "final_fracture_campaign" /
    "cavity_interaction_ramp_0p2s_run1"
)
DEFAULT_PARENT = (
    ROOT / "integration_evidence" / "final_fracture_campaign" /
    "cavity_interaction_ramp_0p1s" / "shared_v5_cavity_audit.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--parent-audit", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = args.result.resolve()
    parent_path = args.parent_audit.resolve()
    pair_path = result / "checkpoint" / "accepted_v5_pair.json"
    failure_path = result / "shared_runtime_attestation.failure.json"
    request_path = result / "shared_execution_request.json"
    states, manifest = _restore_pair(pair_path)
    parent = json.loads(parent_path.read_text())
    failure = json.loads(failure_path.read_text())
    request = json.loads(request_path.read_text())
    restart_manifest = json.loads(Path(request["restart_from"]).read_text())
    observer = _directional_provider(1200.0)
    arms = {}
    for arm, state in states.items():
        start = parent["arms"][arm]["terminal"]
        end = observables(state, "recovered_interaction_terminal")
        _, effective_k, drive, _ = observer(state)
        rates = [
            {
                key: row.get(key)
                for key in (
                    "candidate_id", "effective_rate_s",
                    "G_kinetic_used_J_per_m2", "drive_source",
                    "K_effective_Pa_sqrt_m",
                )
            }
            for row in drive["fracture_engine_rows"]
            if row["directional_drive_status"] == "AVAILABLE"
        ]
        arms[arm] = {
            "start_fingerprint": parent["arms"][arm]["terminal_fingerprint"],
            "terminal_fingerprint": complete_accepted_state_fingerprint(state),
            "checkpoint_fingerprint_exact": (
                complete_accepted_state_fingerprint(state)
                == manifest["fingerprints"][arm]
            ),
            "opening_start_m": start["applied_opening_m"],
            "opening_end_m": end["applied_opening_m"],
            "reaction_start_N_per_m": start["reaction_N_per_m"],
            "reaction_end_N_per_m": end["reaction_N_per_m"],
            "graph_length_start_m": start["graph_length_m"],
            "graph_length_end_m": end["graph_length_m"],
            "graph_length_change_m": (
                end["graph_length_m"] - start["graph_length_m"]
            ),
            "topology_action_change": (
                end["event_counters"].get("topology_actions", 0)
                - start["event_counters"].get("topology_actions", 0)
            ),
            "cavity_radius_start_m": start["cavity_radius_m"],
            "cavity_radius_end_m": end["cavity_radius_m"],
            "cavity_radius_change_m": (
                None if start["cavity_radius_m"] is None else
                end["cavity_radius_m"] - start["cavity_radius_m"]
            ),
            "terminal_crack_void_connection": end[
                "crack_void_connection_certificate"
            ],
            "terminal_length_ledgers": end["length_ledgers"],
            "terminal_effective_K_Pa_sqrt_m": effective_k,
            "terminal_directional_rates": rates,
            "terminal_branches": [
                {
                    "branch_id": branch.branch_id,
                    "status": branch.status,
                    "tip_m": list(branch.tip),
                }
                for branch in state.crack_network.branches
            ],
        }
    payload = {
        "schema": "final-fracture-cavity-interaction-recovered-checkpoint/1",
        "classification": "ACCEPTED_PHYSICS_AUDIT_EXPORT_FAILURE",
        "producer_result": str(result.relative_to(ROOT)),
        "producer_request_sha256": hashlib.sha256(
            request_path.read_bytes()
        ).hexdigest(),
        "producer_failure": {
            "exception_class": failure.get("exception_class"),
            "exception_message": failure.get("exception_message"),
            "execution_verified": failure.get("execution_verified"),
        },
        "native_pair_checkpoint": str(pair_path.relative_to(ROOT)),
        "native_pair_checkpoint_manifest_sha256": hashlib.sha256(
            pair_path.read_bytes()
        ).hexdigest(),
        "native_checkpoint_round_trip_exact": all(
            row["checkpoint_fingerprint_exact"] for row in arms.values()
        ),
        "parent_audit": str(parent_path.relative_to(ROOT)),
        "parent_terminal_to_run_start_exact": all(
            parent["arms"][arm]["terminal_fingerprint"]
            == restart_manifest["fingerprints"][arm]
            for arm in ("disabled", "enabled")
        ),
        "protocol": manifest["configuration"],
        "physical_time_start_s": parent["physical_time_s"],
        "physical_time_end_s": manifest["physical_time_s"],
        "accepted_exposure_s": (
            manifest["physical_time_s"] - parent["physical_time_s"]
        ),
        "arms": arms,
        "interpretation": (
            "Over the matched 0.2 s, 1 um/s continuation, the cavity-suppressed "
            "arm accepted four ordinary 20 um advances (80 um total), whereas "
            "the cavity-enabled arm accepted no crack advance. Its native "
            "complete-ligament energy drive and effective cleavage rate remain "
            "far below the ordinary no-cavity front. This is a finite strong "
            "shielding result; no crack-void connection occurred in the window."
        ),
        "audit_limit": (
            "The event-rich in-memory scheduler history was not published because "
            "the generic JSON normalizer could not serialize a native barrier "
            "dataclass. Terminal native states, exact fingerprints, topology "
            "counts and geometry were written and restored before that exporter "
            "failure. Event times inside the 0.2 s suffix are therefore not claimed."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
