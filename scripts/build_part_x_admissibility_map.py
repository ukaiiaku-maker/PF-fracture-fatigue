"""Final closure D/E: one consolidated censor/attempt/invalidation/
admissibility map across the ENTIRE Part X physical campaign (PX3 screen,
PX4/PX4.1 developed, PX5 static-shield), reading only tracked registries
and ledgers -- never runs/.

Every canonical_job_key that was ever authorized in ANY Part X registry
gets exactly one row, with a single final_disposition drawn from:

    ADMITTED
    INTERRUPTED_NOT_SCIENCE
    INVALIDATED_DWELL_DURATION_WEIGHTING_BUG
    SUPERSEDED_WALL_BUDGET_TOO_SMALL
    NOT_LAUNCHED_BLOCKED
    NOT_LAUNCHED_ALIAS

This is the canonical source for figure 18 (censor/attempt/invalidation/
admissibility map).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"
RUN_ROOT = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1"


def main() -> None:
    invalidated = json.loads((ARTIFACTS_DIR / "px3_bug_invalidated_trajectories.json").read_text())
    invalidated_keys = {t["canonical_job_key"]: t["status"] for t in invalidated["invalidated_trajectories"]}

    controller_state = json.loads((RUN_ROOT / "controller_state.json").read_text())
    controller_status_by_key = {k: v["status"] for k, v in controller_state["jobs"].items()}

    rows_by_key: dict[str, dict] = {}

    def _add(campaign: str, key: str, protocol: str, status_in_registry: str, extra_note: str = "") -> None:
        # A canonical_job_key can legitimately appear in more than one
        # registry (e.g. a developed-protocol row aliasing back to an
        # existing PX3 screen result) -- dedupe to ONE row per key, keyed
        # on first appearance (the screen/developed/PX5 registries are
        # processed in that fixed order below, so the first appearance is
        # always the AUTHORITATIVE registration, never a downstream ALIAS),
        # and record every campaign that references it.
        if key in rows_by_key:
            if campaign not in rows_by_key[key]["campaigns_referencing"]:
                rows_by_key[key]["campaigns_referencing"].append(campaign)
            return
        if key in invalidated_keys:
            disposition = invalidated_keys[key]
        elif status_in_registry == "SUPERSEDED_WALL_BUDGET_TOO_SMALL":
            disposition = "SUPERSEDED_WALL_BUDGET_TOO_SMALL"
        elif status_in_registry.startswith("AUTHORIZED"):
            controller_status = controller_status_by_key.get(key)
            if controller_status in ("COMPLETE", "COMPLETE_PREFLIGHT"):
                disposition = "ADMITTED"
            elif controller_status == "INTERRUPTED_NOT_SCIENCE":
                disposition = "INTERRUPTED_NOT_SCIENCE"
            elif controller_status is None:
                disposition = "NOT_LAUNCHED_AUTHORIZED_BUT_NO_ATTEMPT_RECORD"
            else:
                disposition = f"OTHER_CONTROLLER_STATUS_{controller_status}"
        elif status_in_registry == "ALIAS_OF_EXISTING_JOB":
            disposition = "NOT_LAUNCHED_ALIAS"
        elif status_in_registry.startswith("BLOCKED"):
            disposition = "NOT_LAUNCHED_BLOCKED"
        else:
            disposition = f"OTHER_REGISTRY_STATUS_{status_in_registry}"

        rows_by_key[key] = {
            "campaign_of_record": campaign, "canonical_job_key": key, "protocol": protocol,
            "registry_status": status_in_registry, "controller_status": controller_status_by_key.get(key, ""),
            "final_disposition": disposition, "note": extra_note, "campaigns_referencing": [campaign],
        }

    for row in csv.DictReader(open(ARTIFACTS_DIR / "screen_job_registry.csv")):
        _add("PX3_screen", row["canonical_job_key"], row["protocol"], row["status"])

    for row in csv.DictReader(open(ARTIFACTS_DIR / "developed_job_registry.csv")):
        _add("PX4_developed", row["canonical_job_key"], row["protocol"], row["status"])

    for row in csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_job_registry.csv")):
        _add("PX5_static_shield", row["canonical_job_key"], row["protocol"], row["status"])
    for row in csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_wall_budget_retry_registry.csv")):
        _add("PX5_static_shield_retry", row["canonical_job_key"], row["protocol"], row["status"],
             extra_note="wall-budget retry of a superseded original attempt")

    # The 3 quarantined wrong-budget PX4 attempts never appear in developed_
    # job_registry.csv under their OWN pre-fix canonical_job_key (that key was
    # minted before the campaign_stage fix and is not the registry's current
    # key for the same physical row) -- add them explicitly from controller_
    # state so they are not silently absent from the map.
    quarantine_dir = RUN_ROOT / "_quarantine_interrupted"
    if quarantine_dir.is_dir():
        for job_json_path in sorted(quarantine_dir.glob("*.job.json")):
            job = json.loads(job_json_path.read_text())
            key = job["canonical_job_key"]
            if key not in rows_by_key:
                rows_by_key[key] = {
                    "campaign_of_record": "PX4_developed", "canonical_job_key": key, "protocol": job["protocol"],
                    "registry_status": "PRE_FIX_CANONICAL_KEY_NOT_IN_CURRENT_REGISTRY",
                    "controller_status": controller_status_by_key.get(key, ""),
                    "final_disposition": "INTERRUPTED_NOT_SCIENCE",
                    "note": "wrong-budget attempt under the pre-campaign_stage-fix canonical_job_key; "
                            "the corrected key for this same physical row is the ADMITTED row above",
                    "campaigns_referencing": ["PX4_developed"],
                }

    rows_out = list(rows_by_key.values())
    for r in rows_out:
        r["campaigns_referencing"] = ";".join(r["campaigns_referencing"])

    from collections import Counter
    disposition_counts = Counter(r["final_disposition"] for r in rows_out)

    with (ARTIFACTS_DIR / "part_x_admissibility_map.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)
    (ARTIFACTS_DIR / "part_x_admissibility_map.json").write_text(json.dumps({
        "schema": "v10230_part_x_admissibility_map_v1",
        "n_total_rows": len(rows_out), "disposition_counts": dict(disposition_counts), "rows": rows_out,
    }, indent=2, default=str))

    print(f"Wrote part_x_admissibility_map.{{csv,json}}: {len(rows_out)} rows")
    for disposition, count in sorted(disposition_counts.items()):
        print(f"  {disposition}: {count}")


if __name__ == "__main__":
    main()
