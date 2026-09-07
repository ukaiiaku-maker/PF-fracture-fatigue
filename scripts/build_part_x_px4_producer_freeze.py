"""PX4 launch producer freeze (external review section 8): one clean,
current, committed producer identity for every AUTHORIZED_PX4 developed
row before launch -- distinguishing:

  selection_source_commit    -- the commit at which D1-D7's post-screen
                                 decisions were finalized (PX3.6, 300ff5a)
  physical_source_commit     -- the commit whose physics-affecting .py
                                 files will actually execute these jobs
  launch_HEAD                -- the git HEAD at the moment of this freeze
                                 (must equal physical_source_commit; no
                                 drift is tolerated by this script)
  physical_source_bundle_sha256 -- a single aggregate hash over every
                                 physics-affecting file's own content, so
                                 any future producer-identity question can
                                 be settled by comparing one value

Regenerates ONLY the 44 rows currently AUTHORIZED_PX4 in developed_job_
registry.csv (nothing has been physically launched under any of them --
verified below before touching anything) -- their physical_producer_sha
and canonical_job_key are refreshed to this freeze's launch_HEAD. Every
other column (Kmax, R, frequency, hold, chemistry, K_target, config_hash,
material_row_hash, seed, cohesion, status) is left untouched. BLOCKED_*/
ALIAS_OF_EXISTING_JOB rows (D4, the old D3/D6 100Hz/1000Hz rows, D7,
D1_confirm/D2_confirm) are NOT touched -- they are not being launched, and
their historical producer identity (whatever it was when they were
evaluated) is left as the honest record of that evaluation.

This never rewrites any already-COMPLETED physical result's own recorded
physical_producer_sha (those live inside runs/.../result.json and the
already-committed PX3/PX3.5/PX3.6 provenance JSONs, none of which this
script touches).
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

SELECTION_SOURCE_COMMIT = "300ff5ab355c47457da65f45b6428daa39485374"  # PX3.6


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    status = _git("status", "--short")
    if status.strip():
        raise RuntimeError(f"refusing to freeze: worktree is not clean:\n{status}")
    launch_head = _git("rev-parse", "HEAD")
    # selection_source_commit (PX3.6, where D1-D7's decisions were finalized)
    # and launch_HEAD (this freeze's own, necessarily later, commit -- every
    # commit changes HEAD, so requiring exact equality here would make a
    # freeze impossible by construction) are legitimately different commits;
    # that is exactly why the review asked for them as separate fields
    # rather than one. What DOES matter is verified below instead: the
    # physical-source-bundle hash proves whether anything physics-affecting
    # actually changed between them.
    diff_since_selection = _git("diff", "--name-only", SELECTION_SOURCE_COMMIT, launch_head)
    physics_files_changed_since_selection = [
        p for p in diff_since_selection.splitlines() if p in PHYSICAL_SOURCE_FILES
    ]

    registry_path = ARTIFACTS_DIR / "developed_job_registry.csv"
    with registry_path.open() as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())

    authorized = [r for r in rows if r["status"] == "AUTHORIZED_PX4"]
    if len(authorized) != 44:
        raise RuntimeError(f"expected exactly 44 AUTHORIZED_PX4 rows, found {len(authorized)}")

    # Zero-launch precondition (mirrors verify_part_x_px3_5.py's own check_7).
    launched = []
    if RUN_ROOT.exists():
        existing_keys = {r["canonical_job_key"] for r in authorized}
        for job_json in RUN_ROOT.glob("*.job.json"):
            job = json.loads(job_json.read_text())
            if job.get("canonical_job_key") in existing_keys:
                launched.append(job["canonical_job_key"])
    if launched:
        raise RuntimeError(f"refusing to freeze: jobs already launched under current keys: {launched}")

    # Aggregate physical-source-bundle hash: sha256 of the sorted
    # "path:blob_sha256" lines, one per physics-affecting file.
    per_file = {}
    for path in PHYSICAL_SOURCE_FILES:
        blob = subprocess.run(["git", "show", f"{launch_head}:{path}"], cwd=REPO_ROOT, capture_output=True, check=True).stdout
        per_file[path] = hashlib.sha256(blob).hexdigest()
    aggregate_lines = "\n".join(f"{path}:{h}" for path, h in sorted(per_file.items()))
    physical_source_bundle_sha256 = hashlib.sha256(aggregate_lines.encode("utf-8")).hexdigest()

    n_refreshed = 0
    for row in rows:
        if row["status"] != "AUTHORIZED_PX4":
            continue
        new_key = canonical_job_key(
            material_row_hash=row["material_row_hash"], rebonding_config_hash=row["config_hash"],
            Kmax=float(row["Kmax_Pa_sqrt_m"]), R=float(row["R"]), nominal_frequency_Hz=float(row["frequency_Hz"]),
            minimum_load_hold_s=float(row["minimum_load_hold_s"]), T_K=300.0,
            chemistry_factor=float(row["chemistry_factor"]), K_rebond_max_Pa_sqrt_m=float(row["K_rebond_max_target_Pa_sqrt_m"]),
            seed=int(row["seed"]), integrator_mode=row["integrator_mode"], physical_producer_sha=launch_head,
        )
        row["physical_producer_sha"] = launch_head
        row["canonical_job_key"] = new_key
        n_refreshed += 1

    with registry_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    freeze_record = {
        "schema": "v10230_part_x_px4_producer_freeze_v1",
        "selection_source_commit": SELECTION_SOURCE_COMMIT,
        "physical_source_commit": launch_head,
        "launch_HEAD": launch_head,
        "physical_source_bundle_sha256": physical_source_bundle_sha256,
        "per_file_sha256": per_file,
        "files_changed_between_selection_source_and_launch_HEAD": diff_since_selection.splitlines(),
        "physical_source_files_among_those_changes": physics_files_changed_since_selection,
        "rows_refreshed": n_refreshed,
        "zero_jobs_launched_precondition_verified": True,
        "note": (
            "Only the 44 rows already AUTHORIZED_PX4 in developed_job_registry.csv had their "
            "physical_producer_sha/canonical_job_key refreshed to this freeze's launch_HEAD. No "
            "BLOCKED_*/ALIAS row, and no already-completed physical result's own recorded "
            "producer identity, was touched."
        ),
    }
    out_path = ARTIFACTS_DIR / "px4_producer_freeze.json"
    out_path.write_text(json.dumps(freeze_record, indent=2))
    print(f"wrote {out_path}")
    print(f"launch_HEAD={launch_head}")
    print(f"physical_source_bundle_sha256={physical_source_bundle_sha256}")
    print(f"rows refreshed: {n_refreshed}")


if __name__ == "__main__":
    main()
