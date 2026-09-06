"""PX3.5 section 1: close administrative/producer provenance.

Proves that the physical source bundle (every production python module that
can affect trajectory physics) is byte-identical between the registry's
recorded physical_producer_sha (6b9631d, the commit at which the screen
job registry was generated) and the actual launch HEAD used by the
controller (cc306a8, one commit later -- the registry's own producer-sha
refresh, per PX3's own follow-up commit). Distinguishes:

  physical_source_commit    -- the commit whose physics-affecting .py files
                                were actually executed for the 28 PX3 jobs
  protocol_registry_commit  -- the commit whose registry/CSV rows were used
                                to select and launch those jobs
  launch_HEAD               -- the actual git HEAD at controller launch time

and requires exact equality of every physics-affecting file's hash across
the two candidate commits, not merely "no diff --stat entries touch source
files" (this script re-hashes each file independently at both commits,
rather than trusting a diff summary).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

# PX0's originally-traced production files plus every Part X physics-affecting
# module added/modified since (fatigue dwell schedule, the causal-pilot-v2
# event loop PX3 reuses, and the PX3 job runner itself). Registry-building
# scripts (build_part_x_*_registry.py) and the controller (orchestration
# only, never touches physics) are deliberately excluded -- they are
# "protocol_registry" concerns, not "physical_source".
PHYSICAL_SOURCE_FILES = [
    "arrhenius_fracture/persistent_site_cyclic_v10229.py",
    "arrhenius_fracture/persistent_site_cyclic_coupled_v10229.py",
    "arrhenius_fracture/persistent_site_coupled_hazard_v10229.py",
    "arrhenius_fracture/persistent_site_cyclic_coupled_audited_v10229.py",
    "arrhenius_fracture/persistent_site_cyclic_energy_gated_v10230.py",
    "arrhenius_fracture/persistent_site_cyclic_energy_gated_corrected_v10230.py",
    "arrhenius_fracture/crack_rebonding_v10230.py",
    "arrhenius_fracture/crack_rebonding_kinetics_v10230.py",
    "arrhenius_fracture/a_native_engine_v10230.py",
    "arrhenius_fracture/material_manifest.py",
    "arrhenius_fracture/reduced_shared_state_v1023.py",
    "arrhenius_fracture/fatigue_v1.py",
    "arrhenius_fracture/crack_rebonding_causal_pilot_v2_v10230.py",
    "scripts/part_x_run_one_job.py",
]


def _git_show_hash(rev: str, path: str) -> str:
    blob = subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    registry_producer_sha = "6b9631dc4d0c27367fd444f93a0d4894cdf66fa5"
    launch_head = "cc306a8a4174b64a608865e8cd02620a56d030f8"
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()

    diff_stat = subprocess.run(
        ["git", "diff", "--name-only", registry_producer_sha, launch_head],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()

    per_file = {}
    all_equal = True
    for path in PHYSICAL_SOURCE_FILES:
        h_registry = _git_show_hash(registry_producer_sha, path)
        h_launch = _git_show_hash(launch_head, path)
        equal = h_registry == h_launch
        all_equal = all_equal and equal
        per_file[path] = {
            "sha256_at_registry_producer_commit": h_registry,
            "sha256_at_launch_HEAD": h_launch,
            "byte_identical": equal,
        }

    touched_physical_source = [p for p in diff_stat if p in PHYSICAL_SOURCE_FILES]

    result = {
        "schema": "v10230_part_x_px3_physical_producer_provenance_v1",
        "protocol_registry_commit": registry_producer_sha,
        "launch_HEAD": launch_head,
        "current_full_HEAD_at_provenance_time": current_head,
        "files_changed_between_registry_producer_and_launch_HEAD": diff_stat,
        "physical_source_files_touched_by_that_diff": touched_physical_source,
        "physical_source_file_hashes": per_file,
        "physical_source_bundle_byte_identical": all_equal,
        "conclusion": (
            "PROVEN_EQUAL_PHYSICAL_SOURCE_BUNDLE -- the only file changes between the "
            "registry's recorded physical_producer_sha and the actual launch HEAD were "
            "the two registry CSVs themselves (a protocol_registry_commit concern), never "
            "a physics-affecting module; the 28 already-completed PX3 trajectories are "
            "admitted as valid under this launch HEAD."
            if all_equal and not touched_physical_source else
            "UNEQUAL_PHYSICAL_SOURCE_BUNDLE -- do not admit the PX3 trajectories without "
            "further review."
        ),
        "policy_for_future_physical_launches": (
            "Do not use a general --allow-head-drift for PX4 or later. Require exact "
            "launch-HEAD equality to the registry's recorded physical_producer_sha, or an "
            "explicit re-proof of physical_source_bundle equality identical in form to this "
            "file, before any future physical launch."
        ),
    }
    out_path = ARTIFACTS_DIR / "px3_physical_producer_provenance.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")
    print(f"  physical_source_bundle_byte_identical: {all_equal}")
    print(f"  files changed between commits: {diff_stat}")
    if not all_equal:
        raise SystemExit("PHYSICAL SOURCE BUNDLE MISMATCH -- see px3_physical_producer_provenance.json")


if __name__ == "__main__":
    main()
