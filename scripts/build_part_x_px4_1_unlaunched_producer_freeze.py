"""PX4.1: freeze a clean, current producer identity for every AUTHORIZED_PX4
row that has NOT yet been physically launched -- generalizes build_part_x_
px4_producer_freeze.py (which targeted exactly the original 44 rows) to the
36 newly-authorized grid-completion/second-seed rows, per the review's own
"Regenerate only the UNLAUNCHED PX4 registry rows against that new clean
HEAD" instruction. Rows already launched (any canonical_job_key that
appears in an existing runs/*.job.json) are identified and left completely
untouched -- their producer identity is the honest historical record of
what actually ran.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"
RUN_ROOT = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1"

from build_part_x_kinetic_regime_registry import canonical_job_key  # noqa: E402
from build_part_x_px3_producer_provenance import PHYSICAL_SOURCE_FILES  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    status = _git("status", "--short")
    if status.strip():
        raise RuntimeError(f"refusing to freeze: worktree is not clean:\n{status}")
    launch_head = _git("rev-parse", "HEAD")

    launched_keys = set()
    if RUN_ROOT.exists():
        for job_json in RUN_ROOT.glob("*.job.json"):
            job = json.loads(job_json.read_text())
            launched_keys.add(job.get("canonical_job_key"))

    registry_path = ARTIFACTS_DIR / "developed_job_registry.csv"
    with registry_path.open() as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    existing_keys = {r["canonical_job_key"] for r in rows}

    authorized = [r for r in rows if r["status"] == "AUTHORIZED_PX4"]
    unlaunched = [r for r in authorized if r["canonical_job_key"] not in launched_keys]
    already_launched = [r for r in authorized if r["canonical_job_key"] in launched_keys]
    print(f"AUTHORIZED_PX4 rows: {len(authorized)} total, {len(already_launched)} already launched, {len(unlaunched)} unlaunched")

    n_refreshed = 0
    for row in unlaunched:
        new_key = canonical_job_key(
            material_row_hash=row["material_row_hash"], rebonding_config_hash=row["config_hash"],
            Kmax=float(row["Kmax_Pa_sqrt_m"]), R=float(row["R"]), nominal_frequency_Hz=float(row["frequency_Hz"]),
            minimum_load_hold_s=float(row["minimum_load_hold_s"]), T_K=300.0, chemistry_factor=float(row["chemistry_factor"]),
            K_rebond_max_Pa_sqrt_m=float(row["K_rebond_max_target_Pa_sqrt_m"]), seed=int(row["seed"]),
            integrator_mode=row["integrator_mode"], physical_producer_sha=launch_head, campaign_stage="developed",
        )
        if new_key in existing_keys and new_key != row["canonical_job_key"]:
            raise RuntimeError(f"canonical_job_key collision refreshing {row['protocol']}: {new_key[:12]}")
        existing_keys.discard(row["canonical_job_key"])
        existing_keys.add(new_key)
        row["physical_producer_sha"] = launch_head
        row["canonical_job_key"] = new_key
        n_refreshed += 1

    with registry_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    per_file = {}
    for path in PHYSICAL_SOURCE_FILES:
        blob = subprocess.run(["git", "show", f"{launch_head}:{path}"], cwd=REPO_ROOT, capture_output=True, check=True).stdout
        per_file[path] = hashlib.sha256(blob).hexdigest()
    aggregate_lines = "\n".join(f"{path}:{h}" for path, h in sorted(per_file.items()))
    physical_source_bundle_sha256 = hashlib.sha256(aggregate_lines.encode("utf-8")).hexdigest()

    freeze_record = {
        "schema": "v10230_part_x_px4_1_unlaunched_producer_freeze_v1",
        "launch_HEAD": launch_head, "physical_source_bundle_sha256": physical_source_bundle_sha256,
        "per_file_sha256": per_file, "rows_refreshed": n_refreshed,
        "already_launched_rows_untouched": len(already_launched),
    }
    out_path = ARTIFACTS_DIR / "px4_1_unlaunched_producer_freeze.json"
    out_path.write_text(json.dumps(freeze_record, indent=2))
    print(f"wrote {out_path}")
    print(f"launch_HEAD={launch_head}")
    print(f"rows refreshed: {n_refreshed}")


if __name__ == "__main__":
    main()
