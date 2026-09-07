"""Final closure F: close producer-versus-launch source provenance for
EVERY admitted PX4/PX4.1/PX5 physical trajectory (several PX5 launches
used --allow-head-drift; several PX4.1 launches did too).

For each distinct (registry producer_sha, actual launch HEAD) pair used
by any admitted row, this script:
  1. hashes every PHYSICAL_SOURCE_FILE (the canonical physics-affecting
     source list, reused verbatim from build_part_x_px3_producer_
     provenance.py) at BOTH commits;
  2. lists every file that differs between the two commits at all
     (git diff --name-only, not restricted to the physics list);
  3. classifies each differing file as PHYSICS_AFFECTING (if it is in
     PHYSICAL_SOURCE_FILES) or REGISTRY_OR_EVIDENCE_ONLY (everything
     else -- CSV/JSON registries, analysis/build scripts that are never
     imported by the physics run path, this closure script itself);
  4. renders a final admissibility decision: PHYSICS_SOURCE_IDENTICAL_
     ALLOW_HEAD_DRIFT_CLOSED if every physics-affecting file hash is
     byte-identical between producer and launch commits (regardless of
     how many non-physics files differ), else PHYSICS_SOURCE_DIVERGED_
     REQUIRES_NARROW_REPLACEMENT.

Covers every canonical_job_key admitted into PX4 (developed_job_
registry.csv, status AUTHORIZED_PX4) and PX5 (px5_static_shield_job_
registry.csv + px5_static_shield_wall_budget_retry_registry.csv, status
AUTHORIZED_PX5), plus the PX3 screen for completeness (its 28 results
are superseded evidence, not part of the PX4/PX5 admission contract, but
are covered here for a complete audit trail).
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

from build_part_x_px3_producer_provenance import PHYSICAL_SOURCE_FILES  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout


def _blob_sha256(rev: str, path: str) -> str:
    blob = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=REPO_ROOT, capture_output=True, check=True).stdout
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    from collections import defaultdict

    controller_state = json.loads((RUN_ROOT / "controller_state.json").read_text())
    launch_head_by_producer: dict[str, set] = defaultdict(set)
    for job_entry in controller_state["jobs"].values():
        prod = job_entry.get("job", {}).get("physical_producer_sha")
        launch = job_entry.get("launch_head")
        if prod and launch:
            launch_head_by_producer[prod].add(launch)

    dev_rows = [r for r in csv.DictReader(open(ARTIFACTS_DIR / "developed_job_registry.csv")) if r["status"] == "AUTHORIZED_PX4"]
    px5_rows = [r for r in csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_job_registry.csv"))
                if r["status"] in ("AUTHORIZED_PX5", "SUPERSEDED_WALL_BUDGET_TOO_SMALL")]
    px5_retry_rows = [r for r in csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_wall_budget_retry_registry.csv"))
                      if r["status"] == "AUTHORIZED_PX5"]
    screen_rows = list(csv.DictReader(open(ARTIFACTS_DIR / "screen_job_registry.csv"))) if (ARTIFACTS_DIR / "screen_job_registry.csv").is_file() else []
    screen_rows = [r for r in screen_rows if r.get("status", "").startswith("AUTHORIZED")]

    groups = {
        "PX4_developed": dev_rows, "PX5_main": px5_rows, "PX5_wall_budget_retry": px5_retry_rows,
        "PX3_screen": screen_rows,
    }

    producer_to_keys: dict[str, dict] = {}
    for group_name, rows in groups.items():
        for row in rows:
            prod = row["physical_producer_sha"]
            entry = producer_to_keys.setdefault(prod, {"group": group_name, "keys": []})
            entry["keys"].append(row["canonical_job_key"])

    results = []
    n_diverged = 0
    for producer_sha, meta in sorted(producer_to_keys.items()):
        launches = launch_head_by_producer.get(producer_sha)
        if not launches:
            raise RuntimeError(f"producer_sha {producer_sha[:12]} (group {meta['group']}) has no recorded launch_head in controller_state.json")
        if len(launches) != 1:
            raise RuntimeError(f"producer_sha {producer_sha[:12]} maps to multiple launch heads: {launches}")
        launch_head = next(iter(launches))

        per_file = {}
        physics_identical = True
        for path in PHYSICAL_SOURCE_FILES:
            h_producer = _blob_sha256(producer_sha, path)
            h_launch = _blob_sha256(launch_head, path)
            equal = h_producer == h_launch
            physics_identical = physics_identical and equal
            per_file[path] = {"sha256_at_producer": h_producer, "sha256_at_launch": h_launch, "identical": equal}

        diff_names = _git("diff", "--name-only", producer_sha, launch_head).strip().splitlines()
        diff_classification = {
            name: ("PHYSICS_AFFECTING" if name in PHYSICAL_SOURCE_FILES else "REGISTRY_OR_EVIDENCE_ONLY")
            for name in diff_names
        }
        any_physics_diff_by_name = any(v == "PHYSICS_AFFECTING" for v in diff_classification.values())
        if any_physics_diff_by_name != (not physics_identical):
            raise RuntimeError(
                f"producer_sha {producer_sha[:12]}: name-based diff classification disagrees with per-file hash "
                f"comparison -- any_physics_diff_by_name={any_physics_diff_by_name}, physics_identical={physics_identical}"
            )

        decision = "PHYSICS_SOURCE_IDENTICAL_ALLOW_HEAD_DRIFT_CLOSED" if physics_identical else "PHYSICS_SOURCE_DIVERGED_REQUIRES_NARROW_REPLACEMENT"
        if not physics_identical:
            n_diverged += 1

        bundle_producer = hashlib.sha256(
            "\n".join(f"{p}:{per_file[p]['sha256_at_producer']}" for p in sorted(PHYSICAL_SOURCE_FILES)).encode()
        ).hexdigest()
        bundle_launch = hashlib.sha256(
            "\n".join(f"{p}:{per_file[p]['sha256_at_launch']}" for p in sorted(PHYSICAL_SOURCE_FILES)).encode()
        ).hexdigest()

        results.append({
            "group": meta["group"], "n_canonical_job_keys": len(meta["keys"]),
            "canonical_job_keys": meta["keys"], "registry_producer_sha": producer_sha, "actual_launch_head": launch_head,
            "physical_source_bundle_sha256_at_producer": bundle_producer,
            "physical_source_bundle_sha256_at_launch": bundle_launch,
            "per_physics_file": per_file, "all_differing_files": diff_classification,
            "n_differing_files_total": len(diff_names), "n_differing_files_physics_affecting": sum(
                1 for v in diff_classification.values() if v == "PHYSICS_AFFECTING"
            ),
            "admissibility_decision": decision,
        })

    if n_diverged > 0:
        raise RuntimeError(
            f"{n_diverged} producer/launch pair(s) show PHYSICS_SOURCE_DIVERGED_REQUIRES_NARROW_REPLACEMENT -- "
            "the narrow replacement policy must be invoked manually before this script can complete; refusing to "
            "silently retain a physics-affecting-source-mismatched trajectory."
        )

    out = {
        "schema": "v10230_part_x_producer_launch_provenance_closure_v1",
        "n_producer_launch_pairs": len(results), "n_diverged": n_diverged,
        "overall_decision": "PHYSICS_SOURCE_IDENTICAL_ALLOW_HEAD_DRIFT_CLOSED_FOR_ALL_PAIRS",
        "pairs": results,
    }
    (ARTIFACTS_DIR / "part_x_producer_launch_provenance_closure.json").write_text(json.dumps(out, indent=2, default=str))

    flat_rows = []
    for r in results:
        flat_rows.append({
            "group": r["group"], "n_canonical_job_keys": r["n_canonical_job_keys"],
            "registry_producer_sha": r["registry_producer_sha"], "actual_launch_head": r["actual_launch_head"],
            "physical_source_bundle_sha256_at_producer": r["physical_source_bundle_sha256_at_producer"],
            "physical_source_bundle_sha256_at_launch": r["physical_source_bundle_sha256_at_launch"],
            "n_differing_files_total": r["n_differing_files_total"],
            "n_differing_files_physics_affecting": r["n_differing_files_physics_affecting"],
            "admissibility_decision": r["admissibility_decision"],
        })
    with (ARTIFACTS_DIR / "part_x_producer_launch_provenance_closure.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"Wrote part_x_producer_launch_provenance_closure.{{json,csv}}: {len(results)} producer/launch pairs")
    for r in results:
        print(f"  {r['group']:24s} keys={r['n_canonical_job_keys']:3d}  producer={r['registry_producer_sha'][:10]}  "
              f"launch={r['actual_launch_head'][:10]}  diff_total={r['n_differing_files_total']}  "
              f"diff_physics={r['n_differing_files_physics_affecting']}  {r['admissibility_decision']}")


if __name__ == "__main__":
    main()
