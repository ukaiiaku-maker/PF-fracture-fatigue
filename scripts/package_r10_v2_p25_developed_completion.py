#!/usr/bin/env python3
"""Create the portable R10 V2-P25 completion and Family-A acquisition receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--v2-rows", type=Path, required=True)
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.destination.exists():
        raise RuntimeError(f"refusing to overwrite existing evidence directory: {args.destination}")
    args.destination.mkdir(parents=True)

    files = {
        ROOT / "scripts/analyze_r10_corrected_eligibility_search.py": "source/scripts/analyze_r10_corrected_eligibility_search.py",
        ROOT / "scripts/design_r10_family_a_discrepancy_acquisition.py": "source/scripts/design_r10_family_a_discrepancy_acquisition.py",
        ROOT / "scripts/run_r10_retained_bridge.py": "source/scripts/run_r10_retained_bridge.py",
        ROOT / "tests/test_r10_family_a_discrepancy_acquisition.py": "source/tests/test_r10_family_a_discrepancy_acquisition.py",
        ROOT / "integration_evidence/r10_joint_search_amendment/inputs/original_four_registry.csv": "inputs/original_four_registry.csv",
        args.v2_rows: "inputs/production_candidate_rows.csv",
        args.v2_manifest: "inputs/production_candidate_manifest.json",
        args.run_root / "analytical_opening_screen.csv": "inputs/analytical_opening_screen.csv",
        args.run_root / "projection_frontier/strict_projection_frontier.csv": "inputs/strict_projection_frontier.csv",
        args.run_root / "current_bridge_local_slopes.csv": "executed/current_bridge_local_slopes.csv",
        args.run_root / "current_bridge_windows.csv": "executed/current_bridge_windows.csv",
        args.run_root / "corrected_eligibility_search/corrected_eligibility.csv": "executed/corrected_eligibility_search/corrected_eligibility.csv",
        args.run_root / "corrected_eligibility_search/selected_candidates.csv": "executed/corrected_eligibility_search/selected_candidates.csv",
        args.run_root / "corrected_eligibility_search/weakt_v2_p25_progress.csv": "executed/corrected_eligibility_search/weakt_v2_p25_progress.csv",
        args.run_root / "corrected_eligibility_search/joint_search_result.json": "executed/corrected_eligibility_search/joint_search_result.json",
        args.run_root / "family_a_discrepancy_acquisition/family_b_A0_to_current_discrepancy.csv": "executed/family_a_acquisition/family_b_A0_to_current_discrepancy.csv",
        args.run_root / "family_a_discrepancy_acquisition/strict_family_a_acquisition.csv": "executed/family_a_acquisition/strict_family_a_acquisition.csv",
        args.run_root / "family_a_discrepancy_acquisition/strict_family_a_candidate_rows.csv": "executed/family_a_acquisition/strict_family_a_candidate_rows.csv",
        args.run_root / "family_a_discrepancy_acquisition/acquisition_decision.json": "executed/family_a_acquisition/acquisition_decision.json",
    }
    cases = {
        args.run_root / "bridges/weakt_v2_p25_seed4810/Kmax_4p60545451": "executed/trajectories/low",
        args.run_root / "bridges/weakt_v2_p25_seed4810/Kmax_10p730645": "executed/trajectories/middle",
        args.run_root / "bridges/weakt_v2_p25_high_seed4810/Kmax_20p2367527": "executed/trajectories/high",
    }
    for source, relative in files.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        copy_file(source, args.destination / relative)
    for source, relative in cases.items():
        if not (source / "result.json").is_file():
            raise FileNotFoundError(source / "result.json")
        shutil.copytree(source, args.destination / relative)

    manifest = []
    for path in sorted(args.destination.rglob("*")):
        if path.is_file():
            manifest.append({
                "path": str(path.relative_to(args.destination)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    (args.destination / "MANIFEST.json").write_text(
        json.dumps({"schema": "r10.v2-p25-developed-completion-package/1", "files": manifest},
                   indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"destination": str(args.destination.resolve()), "file_count": len(manifest)},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
