"""PX4.1 section 1: audit the 44 already-completed PX4 developed
trajectories for cycle-horizon compliance, before the exact first-passage
horizon fix (section 2) lands.

The pre-fix implementation only checked max_cumulative_cycles BETWEEN
accepted events, not within a block or exactly at first passage -- a real
correctness gap for the general case, but this audit proves it was
DORMANT for every one of these 44 trajectories: none came remotely close
to the 1e12-cycle horizon (max observed is many orders of magnitude
below), so no accepted event could possibly have crossed it. Per the
review's own instruction, this proof admits the existing trajectories
without rerunning them: "No rerun is required merely because the
implementation's dormant cap branch was defective."
"""
from __future__ import annotations

import csv
import glob
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"
RUN_ROOT = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1"

CYCLE_HORIZON = 1.0e12
REQUIRED_EVENTS = 30
REQUIRED_EXTENSION_M = 150.0e-6


def main() -> None:
    rows = []
    for p in sorted(glob.glob(str(RUN_ROOT / "*/result.json"))):
        r = json.loads(Path(p).read_text())
        if r.get("schema") != "v10230_part_x_developed_job_result_v1":
            continue
        job, traj = r["job"], r["trajectory"]
        event_cycles = [float(e["cumulative_cycles"]) for e in traj["events"]]
        max_event_cycles = max(event_cycles) if event_cycles else 0.0
        terminal_cycles = float(traj["cumulative_cycles"])
        n_events = traj["n_accepted_events"]
        extension_m = float(traj["cumulative_extension_m"])

        ok = (
            max_event_cycles < CYCLE_HORIZON
            and terminal_cycles < CYCLE_HORIZON
            and n_events == REQUIRED_EVENTS
            and abs(extension_m - REQUIRED_EXTENSION_M) < 1.0e-12
            and not traj["censored"]
        )
        rows.append({
            "canonical_job_key": job["canonical_job_key"], "protocol": job["protocol"],
            "row_name": job["row_name"], "cohesion": job["cohesion"], "Kmax_Pa_sqrt_m": job["Kmax_Pa_sqrt_m"],
            "R": job["R"], "frequency_Hz": job["frequency_Hz"], "seed": job["seed"],
            "physical_producer_sha": job["physical_producer_sha"],
            "n_accepted_events": n_events, "extension_m": extension_m,
            "terminal_cumulative_cycles": terminal_cycles, "max_event_cumulative_cycles": max_event_cycles,
            "raw_censored": traj["censored"], "raw_censor_reason": traj["censor_reason"],
            "cycle_horizon": CYCLE_HORIZON,
            "margin_orders_of_magnitude_below_horizon": (
                None if max_event_cycles <= 0 else round(math.log10(CYCLE_HORIZON / max_event_cycles), 2)
            ),
            "admitted": ok,
            "classification": "CYCLE_HORIZON_GUARD_INACTIVE_PROVEN" if ok else "REQUIRES_RERUN_UNDER_EXACT_HORIZON",
        })

    if len(rows) != 44:
        raise RuntimeError(f"expected exactly 44 developed results, found {len(rows)}")

    n_admitted = sum(1 for r in rows if r["admitted"])
    max_overall = max(r["max_event_cumulative_cycles"] for r in rows)
    min_margin = min(r["margin_orders_of_magnitude_below_horizon"] for r in rows if r["margin_orders_of_magnitude_below_horizon"] is not None)

    with (ARTIFACTS_DIR / "px4_pre_horizon_repair_admission.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (ARTIFACTS_DIR / "px4_pre_horizon_repair_admission.json").write_text(json.dumps({
        "schema": "v10230_part_x_px4_pre_horizon_repair_admission_v1",
        "cycle_horizon": CYCLE_HORIZON,
        "n_trajectories": len(rows), "n_admitted": n_admitted,
        "max_event_cumulative_cycles_across_all_trajectories": max_overall,
        "minimum_margin_orders_of_magnitude_below_horizon": min_margin,
        "conclusion": (
            "CYCLE_HORIZON_GUARD_INACTIVE_PROVEN for all 44 trajectories" if n_admitted == 44 else
            f"{44 - n_admitted} trajectories require rerun under the exact horizon"
        ),
        "rows": rows,
    }, indent=2, default=str))

    print(f"admitted {n_admitted}/{len(rows)}")
    print(f"max event cumulative_cycles across ALL trajectories: {max_overall:.6e} "
          f"({min_margin} orders of magnitude below the 1e12 horizon)")


if __name__ == "__main__":
    main()
