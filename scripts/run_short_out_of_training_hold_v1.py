#!/usr/bin/env python3
"""Independent 0.1 s fixed-state hold from the accepted 2.59999 s prefix."""
from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.global_physical_scheduler_v1 import advance_global_physical_interval
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from scripts.run_active_two_mechanism_natural_void_study import canonical_drive_check


SOURCE = ROOT / "runs/managed_natural_void_v1_20260919/output/restart_002600ms.json"
OUT = ROOT / "artifacts/short_out_of_training_hold_v1"


def phase(state):
    return {
        "site_phases": {site.site_id: site.phase.value for site in state.void_state.sites},
        "direct_action": {site.site_id: site.interaction_action for site in
                          state.void_state.dislocation_interaction_state.sites},
        "direct_hits": {site.site_id: site.interaction_hit_count for site in
                        state.void_state.dislocation_interaction_state.sites},
        "growth": {owner.site_id: {
            "radius_m": owner.radius_m,
            "loop_count": owner.cavity_emission_state.cumulative_loop_count_effective,
        } for owner in state.void_state.embryo_growth_states},
        "cavity_count": len(state.void_state.cavities),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state = restore_checkpoint(SOURCE)
    before = phase(state)
    tensor, K, canonical, load_rows = canonical_drive_check(state)
    advanced, audit = advance_global_physical_interval(
        state, state, 0.1, {
            "kind": "hold", "temperature_K": 1200.0,
            "stress_tensor_Pa": tensor,
            "K_effective_by_front": {ROOT_BRANCH_ID: K},
            "directional_load_rows_by_front": {ROOT_BRANCH_ID: load_rows},
            "maximum_events": 4096,
            "maximum_transactions_per_call": 4096,
        })
    checkpoint = OUT / "restart_hold_002700ms.json"
    manifest = write_checkpoint(advanced, checkpoint, compression="gzip")
    restored = restore_checkpoint(checkpoint)
    selected = [row["selected_event"] for row in audit["events"]
                if row["selected_event"] is not None]
    result = {
        "schema": "unified-2d.short-independent-history/1",
        "prediction_frozen_before_run": str(
            ROOT.parent/"oneD-v3-stateful-voiding-reduction/analysis_outputs/"
            "oneD_v3_bounded_autonomous/result.json"),
        "history": "FIXED_OPENING_ZERO_NEW_ACCEPTED_PLASTIC_INCREMENT",
        "start_time_s": 2.59999, "duration_s": 0.1,
        "before": before, "after": phase(advanced),
        "selected_events": selected,
        "advanced_duration_s": audit["advanced_duration_s"],
        "all_owners_previewed": audit["all_owners_previewed"],
        "canonical_drive": canonical,
        "accepted_state_fingerprint": complete_accepted_state_fingerprint(advanced),
        "checkpoint": checkpoint.name,
        "checkpoint_state_file": manifest["state_file"],
        "direct_restored_equal": (
            complete_accepted_state_fingerprint(advanced)
            == complete_accepted_state_fingerprint(restored)),
    }
    (OUT/"result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(OUT/"result.json")
    print(json.dumps({"selected_events": selected, "after": result["after"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
