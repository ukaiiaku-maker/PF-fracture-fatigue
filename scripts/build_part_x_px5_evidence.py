"""PX5: portable evidence package for the 20 static-shield control
trajectories (D2/D5 x ceiling_static/orbit_matched_static x 5-point
seed=1720 Kmax grid), analogous to PX4 Stage-1's developed_event_ledger/
developed_attempt_registry/developed_censor_registry/developed_raw_
result_hashes.

Writes, under artifacts/crack_rebonding_part_x_v1/:
  px5_event_ledger.json / .csv   -- every admitted PX5 trajectory's every
                                     accepted event (K_b_applied_Pa_sqrt_m
                                     included, since that IS the physics
                                     these trajectories exist to probe).
  px5_raw_result_hashes.json     -- sha256 of every admitted result.json.
  px5_attempt_registry.csv       -- every physical attempt: the 18
                                     original AUTHORIZED_PX5 completions,
                                     the 2 SUPERSEDED_WALL_BUDGET_TOO_SMALL
                                     censored attempts (never admitted),
                                     and the 2 wall-budget-retry
                                     completions that superseded them.
  px5_censor_registry.csv        -- one row per ADMITTED trajectory
                                     (censored=False by construction for
                                     all 20, or this evidence package
                                     would refuse to admit them).
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"
RUN_ROOT = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1"

EXPECTED_ADMITTED = 20
EXPECTED_SUPERSEDED = 2


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    main_rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_job_registry.csv")))
    retry_rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_wall_budget_retry_registry.csv")))

    admitted_rows = [r for r in main_rows if r["status"] == "AUTHORIZED_PX5"] + [r for r in retry_rows if r["status"] == "AUTHORIZED_PX5"]
    superseded_rows = [r for r in main_rows if r["status"] == "SUPERSEDED_WALL_BUDGET_TOO_SMALL"]
    if len(admitted_rows) != EXPECTED_ADMITTED:
        raise RuntimeError(f"expected exactly {EXPECTED_ADMITTED} admitted PX5 rows, found {len(admitted_rows)}")
    if len(superseded_rows) != EXPECTED_SUPERSEDED:
        raise RuntimeError(f"expected exactly {EXPECTED_SUPERSEDED} superseded PX5 rows, found {len(superseded_rows)}")

    results_by_key = {}
    for p in glob.glob(str(RUN_ROOT / "*/result.json")):
        r = json.loads(Path(p).read_text())
        job = r.get("job")
        if job is not None:
            results_by_key[job["canonical_job_key"]] = (Path(p), r)

    event_rows, censor_rows, raw_hashes, attempt_rows = [], [], {}, []

    for row in admitted_rows:
        key = row["canonical_job_key"]
        if key not in results_by_key:
            raise RuntimeError(f"missing completed result.json for admitted PX5 row {key[:12]}")
        path, result = results_by_key[key]
        job, traj = result["job"], result["trajectory"]
        if traj["censored"] or traj["n_accepted_events"] != 30:
            raise RuntimeError(f"admitted PX5 row {key[:12]} is censored or incomplete -- refusing to admit")

        raw_hashes[key] = {
            "sha256": _sha256_file(path), "result_dir_name": path.parent.name,
            "protocol": job["protocol"], "cohesion": job["cohesion"], "Kmax_Pa_sqrt_m": job["Kmax_Pa_sqrt_m"],
        }
        censor_rows.append({
            "protocol": job["protocol"], "cohesion": job["cohesion"], "Kmax_Pa_sqrt_m": job["Kmax_Pa_sqrt_m"],
            "K_rebond_max_target_Pa_sqrt_m": job["K_rebond_max_target_Pa_sqrt_m"], "canonical_job_key": key,
            "n_accepted_events": traj["n_accepted_events"], "censored": traj["censored"],
            "censor_reason": traj["censor_reason"], "cumulative_extension_m": traj["cumulative_extension_m"],
            "cumulative_cycles": traj["cumulative_cycles"], "max_wall_seconds_used": row.get("max_wall_seconds_override") or "",
        })
        attempt_rows.append({
            "canonical_job_key": key, "protocol": job["protocol"], "cohesion": job["cohesion"],
            "Kmax_Pa_sqrt_m": job["Kmax_Pa_sqrt_m"], "physical_producer_sha": job["physical_producer_sha"],
            "attempt_status": "COMPLETE", "admitted": True, "supersedes_key": "", "note": row.get("note", ""),
        })
        for event in traj["events"]:
            event_rows.append({
                "canonical_job_key": key, "protocol": job["protocol"], "cohesion": job["cohesion"],
                "Kmax_Pa_sqrt_m": job["Kmax_Pa_sqrt_m"], "R": job["R"], "frequency_Hz": job["frequency_Hz"],
                "event_index": event["event_index"], "K_b_applied_Pa_sqrt_m": event.get("K_b_applied_Pa_sqrt_m"),
                "hazard_threshold_action": event["hazard_threshold_action"], "accepted_length_m": event["accepted_length_m"],
                "cumulative_extension_m": event["cumulative_extension_m"], "cumulative_cycles": event["cumulative_cycles"],
                "waiting_time_s_this_event": event["waiting_time_s_this_event"],
            })

    retry_key_by_superseded = {}
    for retry_row in retry_rows:
        for orig in superseded_rows:
            if (retry_row["protocol"] == orig["protocol"] and retry_row["cohesion"] == orig["cohesion"]
                    and retry_row["Kmax_Pa_sqrt_m"] == orig["Kmax_Pa_sqrt_m"]):
                retry_key_by_superseded[orig["canonical_job_key"]] = retry_row["canonical_job_key"]

    for row in superseded_rows:
        key = row["canonical_job_key"]
        path, result = results_by_key.get(key, (None, None))
        n_events = result["trajectory"]["n_accepted_events"] if result else None
        attempt_rows.append({
            "canonical_job_key": key, "protocol": row["protocol"], "cohesion": row["cohesion"],
            "Kmax_Pa_sqrt_m": row["Kmax_Pa_sqrt_m"], "physical_producer_sha": row["physical_producer_sha"],
            "attempt_status": f"CENSORED_WALL_TIME_BUDGET_EXHAUSTED_AT_{n_events}_EVENTS",
            "admitted": False, "supersedes_key": "",
            "note": f"superseded by {retry_key_by_superseded.get(key, 'UNKNOWN')[:12]} with a larger wall-clock budget",
        })

    _write_csv(ARTIFACTS_DIR / "px5_event_ledger.csv", event_rows)
    (ARTIFACTS_DIR / "px5_event_ledger.json").write_text(
        json.dumps({"schema": "v10230_part_x_px5_event_ledger_v1", "events": event_rows}, indent=2, default=str)
    )
    _write_csv(ARTIFACTS_DIR / "px5_censor_registry.csv", censor_rows)
    _write_csv(ARTIFACTS_DIR / "px5_attempt_registry.csv", attempt_rows)
    (ARTIFACTS_DIR / "px5_raw_result_hashes.json").write_text(
        json.dumps({"schema": "v10230_part_x_px5_raw_result_hashes_v1", "files": raw_hashes}, indent=2)
    )

    print(f"Wrote px5_event_ledger.{{csv,json}}: {len(event_rows)} events across {len(admitted_rows)} admitted trajectories")
    print(f"Wrote px5_censor_registry.csv: {len(censor_rows)} admitted trajectories, "
          f"{sum(1 for r in censor_rows if r['censored'] == True)} censored")
    print(f"Wrote px5_attempt_registry.csv: {len(attempt_rows)} attempts "
          f"({sum(1 for r in attempt_rows if r['admitted'])} admitted, {sum(1 for r in attempt_rows if not r['admitted'])} superseded/not admitted)")
    print(f"Wrote px5_raw_result_hashes.json: {len(raw_hashes)} files")


if __name__ == "__main__":
    main()
