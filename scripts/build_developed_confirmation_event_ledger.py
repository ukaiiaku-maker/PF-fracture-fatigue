"""Build/extend a portable, tracked event ledger for the developed-
response confirmation campaign, in the same style as
scripts/build_v2_event_ledger.py and scripts/build_v2_second_seed_event_
ledger.py -- so downstream analysis and verification never depend on
gitignored runs/ directories.

Run once per completed stage (Stage 1: seed=1720; Stage 2, conditional:
seed=1001723). Each call MERGES its stage's trajectories into the single
tracked artifacts/crack_rebonding_developed_confirmation/event_ledger.json
(+ .csv) rather than overwriting the other stage's already-recorded
trajectories.

Usage:
    <pinned interpreter> scripts/build_developed_confirmation_event_ledger.py \\
        --run-root runs/crack_rebonding_developed_confirmation/stage1_seed1720
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts/crack_rebonding_developed_confirmation"
PARENT_PILOT_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"

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
    stage_manifest: dict = json.loads((run_root / "stage_manifest.json").read_text())
    frozen = json.loads((PARENT_PILOT_ARTIFACTS / "frozen_configuration.json").read_text())

    if stage_manifest["parent_frozen_configuration_sha256"] != frozen["frozen_configuration_sha256"]:
        raise SystemExit(
            "this stage's manifest was recorded against a different "
            "frozen_configuration_sha256 than the parent pilot's frozen "
            "configuration -- refusing to merge a possibly-drifted config "
            "into the tracked ledger"
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = ARTIFACTS_DIR / "event_ledger.json"
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text())
        if ledger["frozen_configuration_sha256"] != frozen["frozen_configuration_sha256"]:
            raise SystemExit(
                "existing tracked ledger was built against a different "
                "frozen_configuration_sha256 -- refusing to merge"
            )
    else:
        ledger = {
            "schema": "v10.2.30_crack_rebonding_developed_confirmation_event_ledger_v1",
            "frozen_configuration_sha256": frozen["frozen_configuration_sha256"],
            "trajectories": {},
        }

    added = []
    for name, res in trajectories.items():
        if name in ledger["trajectories"]:
            raise SystemExit(
                f"trajectory {name!r} already present in the tracked ledger -- "
                "no restart/resume or silent overwrite authorized; if this is "
                "an intentional re-run under a new name, use a new trajectory "
                "name instead"
            )
        events_out = []
        for e in res["events"]:
            event_row = {k: e.get(k) for k in EVENT_FIELDS}
            mpz_state = e.get("mpz_state") or {}
            for k in MPZ_STATE_FIELDS:
                event_row[k] = mpz_state.get(k)
            events_out.append(event_row)
        ledger["trajectories"][name] = {
            "R": res["R"],
            "rebonding_cfg_hash": res.get("rebonding_cfg_hash"),
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
        added.append(name)

    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(f"wrote {ledger_path}  (+{len(added)} trajectories: {', '.join(added)})")

    csv_rows: list[dict[str, Any]] = []
    for name, traj in ledger["trajectories"].items():
        for event_row in traj["events"]:
            csv_rows.append({"trajectory": name, **event_row})
    csv_path = ARTIFACTS_DIR / "event_ledger.csv"
    fieldnames = ["trajectory"] + EVENT_FIELDS + MPZ_STATE_FIELDS
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"wrote {csv_path}  ({len(csv_rows)} event rows total)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
