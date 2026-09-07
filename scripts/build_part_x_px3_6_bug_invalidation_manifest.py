"""PX3.6 section 1: formally invalidate the two original PX3 finite-cohesion
dwell trajectories (hold=0.0005s, hold=0.002s), which were computed with the
_phase_statistics duration-weighting bug (fixed in PX3.5, commit c720d59).

Per external review: these are not legitimate physical results (the 0.5ms
case is not a legitimate right-censor; the 2ms case is not a legitimate
uncensored measurement) -- they must be excluded from scientific screen
counts, censor statistics, sensitivity classifications, and developed-
protocol decisions, retained ONLY as historical bug-audit evidence. This
script does not delete or alter the original result.json files (mission's
own "do not rerun/alter/delete the 28 existing PX3 trajectories" still
applies) -- it records their invalidation status and points to the
corrected PX3.5 dwell-audit replacement.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"
RUN_ROOT = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1"

INVALIDATED_KEYS = {
    "3d30034af1b472c45d28f9ab22063a56707c5a38b084acfa6dee5871095115d5": "hold=0.0005s finite cohesion",
    "7e27ff50ebf98aefef66e6e3fc08d2266e2765f5ef7b63209bbaa575cdb22274": "hold=0.002s finite cohesion",
}

BUGGY_COMMIT = "ba1731e0c9b20a0b685245bd12af49c6b437b25a"  # PX3 screen commit -- has the bug
FIXED_COMMIT = "c720d590ee1c072460b574399ec28808e5e6727f"  # PX3.5 -- has the fix
DEFECTIVE_FILE = "arrhenius_fracture/persistent_site_coupled_hazard_v10229.py"


def _git_show_hash(rev: str, path: str) -> str:
    blob = subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    key_to_path = {}
    for p in glob.glob(str(RUN_ROOT / "*/result.json")):
        r = json.loads(Path(p).read_text())
        if "job" in r and r["job"]["canonical_job_key"] in INVALIDATED_KEYS:
            key_to_path[r["job"]["canonical_job_key"]] = p

    pairs = json.loads((ARTIFACTS_DIR / "px3_screen_pair_analysis.json").read_text())["pairs"]
    dwell_audit = json.loads((ARTIFACTS_DIR / "px3_5_dwell_audit_classification.json").read_text())

    entries = []
    for key, label in INVALIDATED_KEYS.items():
        pair = next(
            p for p in pairs if p["protocol"] == "7.3_dwell_panel" and p["finite_canonical_job_key"] == key
        )
        replacement_leg = "dynamic_finite_hold0.0005" if "0.0005" in label else "dynamic_finite_hold0.002"
        entries.append({
            "canonical_job_key": key, "label": label,
            "original_result_path": key_to_path.get(key, "NOT_FOUND_ON_THIS_MACHINE"),
            "original_result_sha256": (
                hashlib.sha256(Path(key_to_path[key]).read_bytes()).hexdigest()
                if key in key_to_path else "NOT_ARCHIVED"
            ),
            "original_producer_commit": BUGGY_COMMIT,
            "defective_source_file": DEFECTIVE_FILE,
            "defective_source_sha256_at_producer_commit": _git_show_hash(BUGGY_COMMIT, DEFECTIVE_FILE),
            "fixed_source_sha256_at_px3_5_commit": _git_show_hash(FIXED_COMMIT, DEFECTIVE_FILE),
            "original_reported_S_h": pair["S_h_all"],
            "invalidation_reason": (
                "_phase_statistics's hazard_coupled branch duration-weighted cursor-rotated "
                "cleavage-hazard samples against the unrotated dt_values array -- corrupting "
                "the adaptive block-stepping search's hazard rate whenever hold>0. Fixed in "
                f"commit {FIXED_COMMIT}. This trajectory is NOT a legitimate physical "
                "measurement (neither a valid right-censor at hold=0.0005s nor a valid "
                "uncensored measurement at hold=0.002s) and must not be used in any "
                "scientific screen count, censor statistic, sensitivity classification, or "
                "developed-protocol decision."
            ),
            "status": "INVALIDATED_DWELL_DURATION_WEIGHTING_BUG",
            "corrected_replacement": {
                "source": "px3_5_dwell_audit_classification.json",
                "leg": replacement_leg,
                "corrected_S_h_vs_zero_cohesion": dwell_audit["corrected_audit_S_h_vs_matched_zero_cohesion"][replacement_leg],
            },
        })

    manifest = {
        "schema": "v10230_part_x_px3_6_bug_invalidation_manifest_v1",
        "invalidated_trajectories": entries,
        "exclusion_scope": [
            "scientific screen event/trajectory counts", "censor statistics",
            "axis-sensitivity classifications", "developed-protocol decisions",
            "any manuscript table or figure",
        ],
        "retained_for": "historical bug-audit evidence only; the portable ledger may still reproduce their raw values",
    }
    out_path = ARTIFACTS_DIR / "px3_bug_invalidated_trajectories.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"wrote {out_path}")
    for e in entries:
        print(f"  INVALIDATED: {e['label']} (key={e['canonical_job_key'][:12]}) -> {e['status']}")


if __name__ == "__main__":
    main()
