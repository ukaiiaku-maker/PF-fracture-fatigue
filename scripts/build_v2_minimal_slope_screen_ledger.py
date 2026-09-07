"""Build a portable, tracked event ledger for the minimal slope screen's
new trajectories (Kmax=15/21, both seeds, reversible regime) -- same
schema as the parent branch's event_ledger.json, built from day one this
time rather than retrofitted.

Usage:
    <pinned interpreter> scripts/build_v2_minimal_slope_screen_ledger.py \\
        --run-root runs/crack_rebonding_minimal_slope_screen_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts/crack_rebonding_minimal_slope_screen_v1"

EVENT_FIELDS = [
    "event_index", "blocks_to_fire", "waiting_time_s_this_event", "cumulative_time_s",
    "accepted_length_m", "cumulative_extension_m",
    "pre_event_max_pB", "pre_event_K_rebond_Pa_sqrt_m",
    "max_pB_post_commit", "max_K_rebond_post_commit_Pa_sqrt_m",
    "max_phase_resolved_K_rebond_Pa_sqrt_m", "action_weighted_K_rebond_Pa_sqrt_m",
    "cleavage_action", "hazard_threshold_action", "hazard_event_index", "engine_id",
    "any_bulk_action_used", "all_bulk_action_qualified",
]
MPZ_STATE_FIELDS = [
    "mpz_mobile_count", "mpz_retained_count", "mpz_emitted_total", "mpz_escaped_total",
    "mpz_recovered_total", "r_eff", "sigma_tip", "mpz_total_K_shield_Pa_sqrt_m",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    trajectories: dict[str, dict] = json.loads((run_root / "trajectories.json").read_text())
    exposure_report: dict = json.loads((run_root / "exposure_gate_report.json").read_text())
    frozen_predictions: dict = json.loads(
        (ARTIFACTS_DIR / "frozen_predictions.json").read_text()
    )

    ledger_trajectories: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for name, res in trajectories.items():
        events_out = []
        for e in res["events"]:
            event_row = {k: e.get(k) for k in EVENT_FIELDS}
            mpz_state = e.get("mpz_state") or {}
            for k in MPZ_STATE_FIELDS:
                event_row[k] = mpz_state.get(k)
            events_out.append(event_row)
            csv_rows.append({"trajectory": name, **event_row})

        ledger_trajectories[name] = {
            "R": res["R"],
            "rebonding_model_level": res["rebonding_model_level"],
            "restored_work_of_separation_J_m2": res["restored_work_of_separation_J_m2"],
            "manifest_audit": res["manifest_audit"],
            "seed": res["seed"],
            "n_accepted_events": res["n_accepted_events"],
            "cumulative_extension_m": res["cumulative_extension_m"],
            "cumulative_time_s": res["cumulative_time_s"],
            "censored": res["censored"],
            "censor_reason": res["censor_reason"],
            "uncensored": res["uncensored"],
            "events": events_out,
            "post_first_event_intervals": res["post_first_event_intervals"],
        }

    ledger = {
        "schema": "v10.2.30_crack_rebonding_minimal_slope_screen_event_ledger_v1",
        "parent_frozen_configuration_sha256": frozen_predictions["parent_frozen_configuration_sha256"],
        "exposure_gate_report": exposure_report,
        "trajectories": ledger_trajectories,
    }

    ledger_path = ARTIFACTS_DIR / "event_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(f"wrote {ledger_path}")

    csv_path = ARTIFACTS_DIR / "event_ledger.csv"
    fieldnames = ["trajectory"] + EVENT_FIELDS + MPZ_STATE_FIELDS
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"wrote {csv_path}  ({len(csv_rows)} event rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
