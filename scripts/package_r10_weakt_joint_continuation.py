#!/usr/bin/env python3
"""Package the compact, reviewable R10 Weak-T joint continuation."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    destination = args.out.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    result_root = ROOT / "runs/r10_joint_search/weakt_joint_continuation"

    # Keep the export compact. The full same-row spatial tree remains in the
    # run directory; its hash, binding, arm fingerprints, restart and geometry
    # evidence are carried by selected_spatial_receipt.json.
    for source in sorted(result_root.rglob("*")):
        if source.is_file() and "spatial" not in source.relative_to(result_root).parts:
            copy(source, destination / "results" / source.relative_to(result_root))
    for name in (
        "design_r10_weakt_joint_continuation.py",
        "design_r10_weakt_joint_update.py",
        "analyze_r10_weakt_event_feedback.py",
        "analyze_r10_weakt_joint_continuation.py",
        "run_r10_current_fracture_panel.py",
        "execute_r8_current_backend_spatial.py",
    ):
        copy(ROOT / "scripts" / name, destination / "source/scripts" / name)
    copy(ROOT / "runs/r10_joint_search/continuation_state.json",
         destination / "continuation_state.json")

    # The exact-contract fatigue source results retain event/phase histories,
    # overlap qualification, and runtime identities for both seeds.
    fatigue_root = ROOT / "runs/r10_joint_search/thermal_accessibility_iteration/fatigue"
    for seed in (4810, 9117):
        campaign = fatigue_root / f"peak_seed{seed}"
        copy(campaign / "campaign_summary.json",
             destination / "exact_300K_fatigue_source" / f"seed{seed}" / "campaign_summary.json")
        for source in sorted(campaign.glob("Kmax_*/result.json")):
            copy(source, destination / "exact_300K_fatigue_source" /
                 f"seed{seed}" / source.parent.name / "result.json")

    commands = """# Executed command identities

Python: `/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python`

- `python scripts/design_r10_weakt_joint_continuation.py --registry runs/r10_joint_search/thermal_accessibility_iteration/thermal_state_candidates.csv --full-state-table runs/r10_joint_search/thermal_accessibility_iteration/full_state_case_table.csv --out-root runs/r10_joint_search/weakt_joint_continuation`
- `python scripts/analyze_r10_weakt_event_feedback.py --fatigue-root runs/r10_joint_search/thermal_accessibility_iteration/fatigue/weakt_seed4810 --out-root runs/r10_joint_search/weakt_joint_continuation/feedback`
- `python scripts/run_r10_current_fracture_panel.py --kind offspring --identifier R10C_WEAKT_PEAK300_G1988_S6P0GPA_P4_EP055_PP010_V1 --offspring-registry runs/r10_joint_search/weakt_joint_continuation/candidate.csv --temperatures 300,450,600,750,900,1050,1200 --Kdot 0.005 --Kmax 80 --dK 0.05 --target-extension-um 50 --out-root runs/r10_joint_search/weakt_joint_continuation/full_state/R10C_WEAKT_PEAK300_G1988_S6P0GPA_P4_EP055_PP010_V1`
- `python scripts/design_r10_weakt_joint_update.py --registry runs/r10_joint_search/thermal_accessibility_iteration/thermal_state_candidates.csv --first-candidate runs/r10_joint_search/weakt_joint_continuation/candidate.csv --first-panel runs/r10_joint_search/weakt_joint_continuation/full_state/R10C_WEAKT_PEAK300_G1988_S6P0GPA_P4_EP055_PP010_V1 --out-root runs/r10_joint_search/weakt_joint_continuation`
- `python scripts/run_r10_current_fracture_panel.py --kind offspring --identifier R10C_WEAKT_PEAK300_G2050_S3P75GPA_P4_LOW_TAYLOR_V2 --offspring-registry runs/r10_joint_search/weakt_joint_continuation/candidate_update.csv --temperatures 300,450,600,750,900,1050,1200 --Kdot 0.005 --Kmax 80 --dK 0.05 --target-extension-um 50 --out-root runs/r10_joint_search/weakt_joint_continuation/full_state/R10C_WEAKT_PEAK300_G2050_S3P75GPA_P4_LOW_TAYLOR_V2`
- `python scripts/execute_r8_current_backend_spatial.py runs/r10_joint_search/weakt_joint_continuation/spatial/R10C_WEAKT_PEAK300_G2050_S3P75GPA_P4_LOW_TAYLOR_V2 --offspring-registry runs/r10_joint_search/weakt_joint_continuation/candidate_update.csv --candidate-id R10C_WEAKT_PEAK300_G2050_S3P75GPA_P4_LOW_TAYLOR_V2`
- `python scripts/analyze_r10_weakt_joint_continuation.py --root runs/r10_joint_search/weakt_joint_continuation --thermal-root runs/r10_joint_search/thermal_accessibility_iteration --continuation-state runs/r10_joint_search/continuation_state.json`
"""
    (destination / "COMMANDS.md").write_text(commands)
    decision = json.loads((result_root / "joint_material_decision.json").read_text())
    readme = f"""# R10 Weak-T joint continuation

Joint decision: `{decision['joint_material_decision']}`.

Selected exact row: `{decision['selected_candidate_id']}`
Complete vector SHA-256: `{decision['complete_row_sha256']}`

The updated row passes the bounded complete-state Weak-T curve, the two-seed
developed fatigue objective through exact 300 K complete-process parity with
the retained Peak background, and the same-row R8 spatial execution without a
refit. The original Weak-T five-point curve remains failed-range evidence; the
first cross-background EP055/PP010 acquisition remains a measured near miss.

`results/feedback` records exact event-conditioned opening/state quantities and
discrete early/developed/late endpoint rules. `exact_300K_fatigue_source`
contains the reused event/phase results that own the transferred rates.
"""
    (destination / "README.md").write_text(readme)
    diff_parts = []
    for name in (
        "design_r10_weakt_joint_continuation.py",
        "design_r10_weakt_joint_update.py",
        "analyze_r10_weakt_event_feedback.py",
        "analyze_r10_weakt_joint_continuation.py",
        "package_r10_weakt_joint_continuation.py",
    ):
        relative = f"scripts/{name}"
        lines = (ROOT / relative).read_text().splitlines(keepends=True)
        diff_parts.extend(difflib.unified_diff(
            [], lines, fromfile="/dev/null", tofile=f"b/{relative}", n=3,
        ))
    diff = "".join(diff_parts)
    (destination / "SOURCE_DIFF.patch").write_text(diff)
    manifest = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest[str(path.relative_to(destination))] = {
                "sha256": sha(path), "bytes": path.stat().st_size,
            }
    (destination / "MANIFEST.json").write_text(json.dumps({
        "schema": "r10.weakt-joint-continuation-package/1",
        "git_head_at_execution": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": manifest,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(destination), "files": len(manifest) + 1}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
