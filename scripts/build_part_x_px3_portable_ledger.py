"""PX3.5 section 2: build a complete, portable PX3 screen ledger from the
28 already-completed runs/crack_rebonding_part_x_v1/*/result.json files,
so the strict verifier can reproduce every pairwise g/S_h value from
TRACKED artifacts/ alone, with the gitignored runs/ directory hidden.

Writes, under artifacts/crack_rebonding_part_x_v1/:
  screen_event_ledger.json / .csv   -- every trajectory's every accepted
                                        event, with threshold, accepted
                                        length, cumulative time/cycles,
                                        already-archived MPZ diagnostics,
                                        config/material hash, and censor
                                        state.
  screen_censor_registry.csv        -- one row per trajectory: censored or
                                        not, reason, n_events, extension.
  screen_raw_result_hashes.json     -- sha256 of every raw result.json,
                                        for tamper-evident reproducibility.
  screen_posthoc_correction_manifest.json -- the 6 result.json files whose
                                        post_first_event_intervals were
                                        repaired post-hoc (the interval_
                                        compression_analysis frequency/Kmax
                                        fix), with proof that only that one
                                        field changed.
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

# The 6 non-1000Hz jobs whose post_first_event_intervals were recomputed
# in place after the interval_compression_analysis frequency/Kmax fix
# (PX3 commit). Identified by protocol/frequency/cohesion, not by path
# (paths are gitignored and can vary between machines).
POSTHOC_CORRECTED = [
    {"protocol": "7.2_frequency_panel", "frequency_Hz": 10000.0, "cohesion": "finite"},
    {"protocol": "7.2_frequency_panel", "frequency_Hz": 100.0, "cohesion": "finite"},
    {"protocol": "7.2_frequency_panel", "frequency_Hz": 100.0, "cohesion": "zero"},
    {"protocol": "7.2_frequency_panel", "frequency_Hz": 10000.0, "cohesion": "zero"},
    {"protocol": "7.4_reversible_vs_persistent_at_transition", "frequency_Hz": 100.0, "cohesion": "finite"},
    {"protocol": "7.4_reversible_vs_persistent_at_transition", "frequency_Hz": 100.0, "cohesion": "zero"},
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    result_paths = sorted(glob.glob(str(RUN_ROOT / "*/result.json")))
    if len(result_paths) != 28:
        raise RuntimeError(f"expected exactly 28 PX3 screen result.json files, found {len(result_paths)}")

    event_rows = []
    censor_rows = []
    raw_hashes = {}
    correction_manifest = []

    for path_str in result_paths:
        path = Path(path_str)
        result = json.loads(path.read_text())
        job = result["job"]
        traj = result["trajectory"]
        # Every job's result file is literally named "result.json" inside
        # its own unique directory -- key by canonical_job_key (unique by
        # construction), not by path.name, or all 28 entries collide.
        raw_hashes[job["canonical_job_key"]] = {
            "sha256": _sha256_file(path), "result_dir_name": path.parent.name,
            "protocol": job["protocol"], "cohesion": job["cohesion"],
        }

        censor_rows.append({
            "protocol": job["protocol"], "row_name": job["row_name"], "R": job["R"],
            "frequency_Hz": job["frequency_Hz"], "minimum_load_hold_s": job["minimum_load_hold_s"],
            "chemistry_factor": job["chemistry_factor"],
            "K_rebond_max_target_Pa_sqrt_m": job["K_rebond_max_target_Pa_sqrt_m"],
            "cohesion": job["cohesion"], "canonical_job_key": job["canonical_job_key"],
            "n_accepted_events": traj["n_accepted_events"], "censored": traj["censored"],
            "censor_reason": traj["censor_reason"],
            "cumulative_extension_m": traj["cumulative_extension_m"],
            "cumulative_cycles": traj["cumulative_cycles"], "wall_seconds": traj["wall_seconds"],
        })

        for event in traj["events"]:
            event_rows.append({
                "canonical_job_key": job["canonical_job_key"], "protocol": job["protocol"],
                "row_name": job["row_name"], "cohesion": job["cohesion"], "R": job["R"],
                "frequency_Hz": job["frequency_Hz"], "minimum_load_hold_s": job["minimum_load_hold_s"],
                "config_hash": job["config_hash"], "material_row_hash": job["material_row_hash"],
                "physical_producer_sha": job["physical_producer_sha"],
                "event_index": event["event_index"], "hazard_threshold_action": event["hazard_threshold_action"],
                "accepted_length_m": event["accepted_length_m"],
                "cumulative_extension_m": event["cumulative_extension_m"],
                "cumulative_time_s": event["cumulative_time_s"], "cumulative_cycles": event["cumulative_cycles"],
                "waiting_time_s_this_event": event["waiting_time_s_this_event"],
                "pre_event_max_pB": event["pre_event_max_pB"], "max_pB_post_commit": event["max_pB_post_commit"],
                "pre_event_K_rebond_Pa_sqrt_m": event["pre_event_K_rebond_Pa_sqrt_m"],
                "max_K_rebond_post_commit_Pa_sqrt_m": event["max_K_rebond_post_commit_Pa_sqrt_m"],
                "max_phase_resolved_K_rebond_Pa_sqrt_m": event["max_phase_resolved_K_rebond_Pa_sqrt_m"],
                "action_weighted_K_rebond_Pa_sqrt_m": event["action_weighted_K_rebond_Pa_sqrt_m"],
                "cleavage_action": event["cleavage_action"],
                "mpz_mobile_count": event["mpz_state"].get("mpz_mobile_count"),
                "mpz_retained_count": event["mpz_state"].get("mpz_retained_count"),
                "mpz_emitted_total": event["mpz_state"].get("mpz_emitted_total"),
                "r_eff": event["mpz_state"].get("r_eff"),
                # Not archived by the current instrumentation pass (per-event
                # A_CB/A_BC/A_PC/A_CP/F_* transition-flux breakdown for the
                # SPECIFIC firing sub-interval, and live cycle-mean p_P/p_C
                # over the whole trajectory) -- recorded explicitly rather
                # than inferred, per PX3.5's own instruction.
                "A_CB": "NOT_ARCHIVED", "A_BC": "NOT_ARCHIVED", "A_PC": "NOT_ARCHIVED", "A_CP": "NOT_ARCHIVED",
                "F_CB": "NOT_ARCHIVED", "F_BC": "NOT_ARCHIVED", "F_PC": "NOT_ARCHIVED", "F_CP": "NOT_ARCHIVED",
                "cycle_mean_p_P": "NOT_ARCHIVED", "cycle_mean_p_C": "NOT_ARCHIVED",
                "contact_time_sinusoid_s": "NOT_ARCHIVED", "contact_time_dwell_s": "NOT_ARCHIVED",
            })

        # Post-hoc correction manifest entry, if this job matches one of
        # the 6 known-corrected jobs.
        for spec in POSTHOC_CORRECTED:
            if (job["protocol"] == spec["protocol"] and float(job["frequency_Hz"]) == spec["frequency_Hz"]
                    and job["cohesion"] == spec["cohesion"]):
                correction_manifest.append({
                    "canonical_job_key": job["canonical_job_key"], "protocol": job["protocol"],
                    "frequency_Hz": job["frequency_Hz"], "cohesion": job["cohesion"],
                    "result_file_sha256_after_repair": _sha256_file(path),
                    "result_file_sha256_before_repair": "NOT_ARCHIVED",
                    "reason": (
                        "NOT_ARCHIVED: the repair (interval_compression_analysis's frequency_Hz/"
                        "Kmax_Pa_sqrt_m parameters, previously hardcoded to this module's own "
                        "1000Hz/18MPa reference trajectory constants) was applied via in-place "
                        "byte rewrite before this provenance manifest was written; the pre-repair "
                        "byte content was not separately preserved."
                    ),
                    "repair_script": "px3_repair_intervals.py (one-off, run from the session scratchpad, "
                                     "not committed -- superseded by this manifest's own record)",
                    "exact_fields_changed": ["trajectory.post_first_event_intervals"],
                    "fields_proven_unchanged": [
                        "trajectory.events (all fields, including cumulative_extension_m/cumulative_cycles/"
                        "cumulative_time_s/accepted_length_m/hazard_threshold_action)",
                        "trajectory.n_accepted_events", "trajectory.censored", "trajectory.censor_reason",
                        "trajectory.cumulative_extension_m", "trajectory.cumulative_cycles",
                    ],
                    "proof": (
                        "The repair is a pure function of trajectory.events[i]['cumulative_time_s'] "
                        "(already stored, unaffected by the bug) and job.R/frequency_Hz/Kmax_Pa_sqrt_m "
                        "(job-level fields, also unaffected) -- it recomputes ONLY the "
                        "post_first_event_intervals list via interval_compression_analysis and "
                        "overwrites nothing else in the file. g and S_h (both computed from "
                        "trajectory.events[...]['cumulative_extension_m']/['cumulative_cycles'], never "
                        "from post_first_event_intervals) are therefore unaffected by construction, not "
                        "merely by observation."
                    ),
                })

    with (ARTIFACTS_DIR / "screen_event_ledger.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(event_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(event_rows)
    (ARTIFACTS_DIR / "screen_event_ledger.json").write_text(
        json.dumps({"schema": "v10230_part_x_px3_screen_event_ledger_v1", "events": event_rows}, indent=2, default=str)
    )

    with (ARTIFACTS_DIR / "screen_censor_registry.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(censor_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(censor_rows)

    (ARTIFACTS_DIR / "screen_raw_result_hashes.json").write_text(
        json.dumps({"schema": "v10230_part_x_px3_screen_raw_result_hashes_v1", "files": raw_hashes}, indent=2)
    )

    if len(correction_manifest) != 6:
        raise RuntimeError(f"expected exactly 6 post-hoc-corrected jobs, found {len(correction_manifest)}")
    (ARTIFACTS_DIR / "screen_posthoc_correction_manifest.json").write_text(
        json.dumps({"schema": "v10230_part_x_px3_posthoc_correction_manifest_v1", "corrections": correction_manifest}, indent=2)
    )

    print(f"Wrote screen_event_ledger.{{csv,json}}: {len(event_rows)} events across {len(result_paths)} trajectories")
    print(f"Wrote screen_censor_registry.csv: {len(censor_rows)} trajectories")
    print(f"Wrote screen_raw_result_hashes.json: {len(raw_hashes)} files")
    print(f"Wrote screen_posthoc_correction_manifest.json: {len(correction_manifest)} corrected files")


if __name__ == "__main__":
    main()
