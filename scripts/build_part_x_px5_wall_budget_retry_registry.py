"""PX5 wall-budget retry: relaunch the 2 of 20 static-shield jobs that
hit the pilot's default 1800s max_wall_seconds mid-event (D2/D5 at
Kmax=12 MPa*sqrt(m), ceiling_static -- censored at 5/30 events,
censor_reason='wall_time_budget_exhausted_mid_event') with a 21600s
(6-hour) budget, 2x the ~10800s naive linear extrapolation (5 events in
1800s => 360s/event => 30 events ~= 10800s) to 30 events at the observed
per-event rate, for margin against a slower approach to steady state.

Marks the 2 original AUTHORIZED_PX5 rows SUPERSEDED_WALL_BUDGET_TOO_SMALL
in the main registry (so no analysis script ever mistakes the censored
5-event result for a completed developed trajectory) and writes the 2
retry rows to a separate registry file with a fresh canonical_job_key
(new physical_producer_sha -- the wall-budget-override commit -- makes
this a distinct job by construction, launched into a virgin result path,
never a resume of the censored run).
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

from build_part_x_kinetic_regime_registry import canonical_job_key  # noqa: E402

CENSORED_KEYS = {
    "1973ea7df6540425cb421b7baefc70f721eec74fea019f9116391eef39fdd426",  # D5 Kmax=12 ceiling_static
    "95e4e23c6b8495856f82742923c4f54c3a9b85cc1445e1f4d2ccfb40f2d7a48d",  # D2 Kmax=12 ceiling_static
}
MAX_WALL_SECONDS_OVERRIDE = 21600.0
FIELDNAMES = [
    "protocol", "row_name", "config_hash", "material_row_hash", "Kmax_Pa_sqrt_m", "R", "frequency_Hz",
    "minimum_load_hold_s", "chemistry_factor", "K_rebond_max_target_Pa_sqrt_m", "seed", "integrator_mode",
    "physical_producer_sha", "canonical_job_key", "cohesion", "note", "alias_of_protocol", "alias_of_row_name",
    "status", "max_wall_seconds_override",
]


def _current_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    main_path = ARTIFACTS_DIR / "px5_static_shield_job_registry.csv"
    rows = list(csv.DictReader(open(main_path)))
    fieldnames = list(rows[0].keys())
    if "max_wall_seconds_override" not in fieldnames:
        fieldnames.append("max_wall_seconds_override")

    producer_sha = _current_head()
    retry_rows = []
    n_superseded = 0
    for row in rows:
        row.setdefault("max_wall_seconds_override", "")
        if row["canonical_job_key"] in CENSORED_KEYS:
            row["status"] = "SUPERSEDED_WALL_BUDGET_TOO_SMALL"
            n_superseded += 1

            new_key = canonical_job_key(
                material_row_hash=row["material_row_hash"], rebonding_config_hash=row["config_hash"],
                Kmax=float(row["Kmax_Pa_sqrt_m"]), R=float(row["R"]), nominal_frequency_Hz=float(row["frequency_Hz"]),
                minimum_load_hold_s=float(row["minimum_load_hold_s"]), T_K=300.0,
                chemistry_factor=float(row["chemistry_factor"]),
                K_rebond_max_Pa_sqrt_m=float(row["K_rebond_max_target_Pa_sqrt_m"]), seed=int(row["seed"]),
                integrator_mode=row["integrator_mode"], physical_producer_sha=producer_sha,
                campaign_stage="px5_static_shield",
            )
            retry_row = dict(row)
            retry_row.update({
                "physical_producer_sha": producer_sha, "canonical_job_key": new_key,
                "status": "AUTHORIZED_PX5", "max_wall_seconds_override": str(MAX_WALL_SECONDS_OVERRIDE),
                "note": "wall-budget retry of a censored (5/30 events) original attempt",
            })
            retry_rows.append(retry_row)

    if n_superseded != 2:
        raise RuntimeError(f"expected exactly 2 superseded rows, found {n_superseded}")

    with main_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    retry_path = ARTIFACTS_DIR / "px5_static_shield_wall_budget_retry_registry.csv"
    with retry_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(retry_rows)

    print(f"Marked {n_superseded} rows SUPERSEDED_WALL_BUDGET_TOO_SMALL in {main_path}")
    print(f"Wrote {retry_path}: {len(retry_rows)} AUTHORIZED_PX5 retry rows (producer_sha={producer_sha}, "
          f"max_wall_seconds_override={MAX_WALL_SECONDS_OVERRIDE})")


if __name__ == "__main__":
    main()
