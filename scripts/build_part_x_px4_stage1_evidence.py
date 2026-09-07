"""PX4.1 section 5: build the complete, portable PX4 Stage-1 evidence
package from the 80 completed runs/crack_rebonding_part_x_v1/*/result.json
developed-campaign files (44 original seed-1720 + 6 seed-1720 grid-
completion + 30 seed-1001723 second-seed-confirmation) plus the 3
quarantined wrong-budget INTERRUPTED_NOT_SCIENCE attempts, so the strict
PX4.1 verifier (verify_part_x_px4_stage1.py) can reproduce every rate,
pairing, and seed-robustness classification from TRACKED artifacts/
alone, with the gitignored runs/ directory hidden.

Writes, under artifacts/crack_rebonding_part_x_v1/:
  developed_event_ledger.json / .csv   -- every developed trajectory's
                                           every accepted event.
  developed_raw_result_hashes.json     -- sha256 of every raw result.json,
                                           keyed by canonical_job_key.
  developed_attempt_registry.csv       -- every physical attempt (admitted
                                           COMPLETE + quarantined
                                           INTERRUPTED_NOT_SCIENCE), so the
                                           quarantine narrative is provable
                                           rather than asserted.
  developed_censor_registry.csv        -- one row per admitted trajectory:
                                           censored or not, reason, n_events,
                                           extension, terminal cumulative
                                           cycles vs the 1e12 horizon.
  px4_stage1_rate_table.csv            -- per-pair S_h_developed and da/dN,
                                           reduced from px4_developed_pair_
                                           analysis.csv (already the
                                           authoritative source; this is a
                                           renamed, stable-name copy for the
                                           Stage-1 evidence bundle).
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

EXPECTED_DEVELOPED_RESULTS = 80
EXPECTED_QUARANTINED_INTERRUPTED = 3


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    result_paths = sorted(glob.glob(str(RUN_ROOT / "*/result.json")))
    developed_paths = []
    for p in result_paths:
        r = json.loads(Path(p).read_text())
        if r.get("schema") == "v10230_part_x_developed_job_result_v1":
            developed_paths.append((p, r))
    if len(developed_paths) != EXPECTED_DEVELOPED_RESULTS:
        raise RuntimeError(f"expected exactly {EXPECTED_DEVELOPED_RESULTS} developed results, found {len(developed_paths)}")

    event_rows: list[dict] = []
    censor_rows: list[dict] = []
    raw_hashes: dict = {}
    attempt_rows: list[dict] = []

    for path_str, result in developed_paths:
        path = Path(path_str)
        job, traj = result["job"], result["trajectory"]
        raw_hashes[job["canonical_job_key"]] = {
            "sha256": _sha256_file(path), "result_dir_name": path.parent.name,
            "protocol": job["protocol"], "row_name": job["row_name"], "cohesion": job["cohesion"],
            "seed": job["seed"],
        }

        max_event_cycles = max((float(e["cumulative_cycles"]) for e in traj["events"]), default=0.0)
        censor_rows.append({
            "seed": job["seed"], "protocol": job["protocol"], "row_name": job["row_name"],
            "cohesion": job["cohesion"], "Kmax_Pa_sqrt_m": job["Kmax_Pa_sqrt_m"], "R": job["R"],
            "frequency_Hz": job["frequency_Hz"], "canonical_job_key": job["canonical_job_key"],
            "n_accepted_events": traj["n_accepted_events"], "censored": traj["censored"],
            "censor_reason": traj["censor_reason"],
            "cumulative_extension_m": traj["cumulative_extension_m"],
            "terminal_cumulative_cycles": traj["cumulative_cycles"],
            "max_event_cumulative_cycles": max_event_cycles,
            "cycle_horizon": 1.0e12,
            "hit_cycle_horizon": traj["censor_reason"] == "complete_physical_cycle_censor" or max_event_cycles >= 1.0e12,
        })

        attempt_rows.append({
            "canonical_job_key": job["canonical_job_key"], "seed": job["seed"], "protocol": job["protocol"],
            "row_name": job["row_name"], "cohesion": job["cohesion"], "Kmax_Pa_sqrt_m": job["Kmax_Pa_sqrt_m"],
            "physical_producer_sha": job["physical_producer_sha"], "attempt_status": "COMPLETE",
            "admitted": True, "quarantine_reason": "",
        })

        for event in traj["events"]:
            event_rows.append({
                "canonical_job_key": job["canonical_job_key"], "seed": job["seed"], "protocol": job["protocol"],
                "row_name": job["row_name"], "cohesion": job["cohesion"], "R": job["R"],
                "frequency_Hz": job["frequency_Hz"], "minimum_load_hold_s": job["minimum_load_hold_s"],
                "Kmax_Pa_sqrt_m": job["Kmax_Pa_sqrt_m"], "config_hash": job["config_hash"],
                "material_row_hash": job["material_row_hash"], "physical_producer_sha": job["physical_producer_sha"],
                "event_index": event["event_index"], "hazard_threshold_action": event["hazard_threshold_action"],
                "accepted_length_m": event["accepted_length_m"],
                "cumulative_extension_m": event["cumulative_extension_m"],
                "cumulative_time_s": event["cumulative_time_s"], "cumulative_cycles": event["cumulative_cycles"],
                "waiting_time_s_this_event": event["waiting_time_s_this_event"],
                "pre_event_max_pB": event["pre_event_max_pB"], "max_pB_post_commit": event["max_pB_post_commit"],
                "pre_event_K_rebond_Pa_sqrt_m": event["pre_event_K_rebond_Pa_sqrt_m"],
                "max_K_rebond_post_commit_Pa_sqrt_m": event["max_K_rebond_post_commit_Pa_sqrt_m"],
                "cleavage_action": event["cleavage_action"],
            })

    quarantine_dir = RUN_ROOT / "_quarantine_interrupted"
    n_quarantined = 0
    for job_json_path in sorted(quarantine_dir.glob("*.job.json")):
        job = json.loads(job_json_path.read_text())
        if not job["protocol"].startswith("D"):
            continue
        n_quarantined += 1
        attempt_rows.append({
            "canonical_job_key": job["canonical_job_key"], "seed": job["seed"], "protocol": job["protocol"],
            "row_name": job["row_name"], "cohesion": job["cohesion"], "Kmax_Pa_sqrt_m": job["Kmax_Pa_sqrt_m"],
            "physical_producer_sha": job["physical_producer_sha"], "attempt_status": "INTERRUPTED_NOT_SCIENCE",
            "admitted": False,
            "quarantine_reason": (
                "Killed mid-run under the pre-805f87d runtime budget-selection bug (part_x_run_one_job.py "
                "was about to apply PX3's screen budget to a developed 'D' protocol row); interrupted before "
                "producing a result.json and quarantined by part_x_physical_controller.py's dead-process "
                "detection, never counted as science. Relaunched fresh into a virgin path after the fix "
                "(805f87d) landed; the relaunch is the COMPLETE attempt row above for the same canonical_job_key "
                "family (a new canonical_job_key was minted post-budget-fix since physical_producer_sha changed)."
            ),
        })
    if n_quarantined != EXPECTED_QUARANTINED_INTERRUPTED:
        raise RuntimeError(f"expected exactly {EXPECTED_QUARANTINED_INTERRUPTED} quarantined-interrupted D-protocol attempts, found {n_quarantined}")

    n_admitted = sum(1 for r in attempt_rows if r["admitted"])
    n_admitted_interrupted = sum(1 for r in attempt_rows if r["admitted"] and r["attempt_status"] != "COMPLETE")
    if n_admitted_interrupted != 0:
        raise RuntimeError(f"expected zero admitted interrupted trajectories, found {n_admitted_interrupted}")

    _write_csv(ARTIFACTS_DIR / "developed_event_ledger.csv", event_rows)
    (ARTIFACTS_DIR / "developed_event_ledger.json").write_text(
        json.dumps({"schema": "v10230_part_x_px4_developed_event_ledger_v1", "events": event_rows}, indent=2, default=str)
    )
    _write_csv(ARTIFACTS_DIR / "developed_censor_registry.csv", censor_rows)
    _write_csv(ARTIFACTS_DIR / "developed_attempt_registry.csv", attempt_rows)
    (ARTIFACTS_DIR / "developed_raw_result_hashes.json").write_text(
        json.dumps({"schema": "v10230_part_x_px4_developed_raw_result_hashes_v1", "files": raw_hashes}, indent=2)
    )

    pair_analysis = json.loads((ARTIFACTS_DIR / "px4_developed_pair_analysis.json").read_text())
    rate_rows = []
    for row in pair_analysis["pairs"]:
        rate_rows.append({
            "seed": row["seed"], "protocol": row["protocol"], "row_name": row["row_name"],
            "Kmax_Pa_sqrt_m": row["Kmax_Pa_sqrt_m"], "R": row["R"], "frequency_Hz": row["frequency_Hz"],
            "finite_developed_da_dN_m_per_cycle": row["finite_developed_da_dN_m_per_cycle"],
            "zero_developed_da_dN_m_per_cycle": row["zero_developed_da_dN_m_per_cycle"],
            "S_h_developed": row["S_h_developed"], "both_stable_growth": row["both_stable_growth"],
        })
    _write_csv(ARTIFACTS_DIR / "px4_stage1_rate_table.csv", rate_rows)

    print(f"Wrote developed_event_ledger.{{csv,json}}: {len(event_rows)} events across {len(developed_paths)} trajectories")
    print(f"Wrote developed_censor_registry.csv: {len(censor_rows)} trajectories, "
          f"{sum(1 for r in censor_rows if r['censored'])} censored, "
          f"{sum(1 for r in censor_rows if r['hit_cycle_horizon'])} hit the cycle horizon")
    print(f"Wrote developed_attempt_registry.csv: {len(attempt_rows)} attempts "
          f"({n_admitted} admitted, {len(attempt_rows) - n_admitted} quarantined, {n_admitted_interrupted} admitted-interrupted)")
    print(f"Wrote developed_raw_result_hashes.json: {len(raw_hashes)} files")
    print(f"Wrote px4_stage1_rate_table.csv: {len(rate_rows)} pairs")


if __name__ == "__main__":
    main()
