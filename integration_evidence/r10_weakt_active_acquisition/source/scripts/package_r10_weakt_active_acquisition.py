#!/usr/bin/env python3
"""Export the executed active Weak-T acquisition, including restart checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN_NAMES = (
    "weakt_active_acquisition",
    "weakt_active_refinement",
    "weakt_active_thermal_refinement",
    "weakt_active_thermal_refinement_narrow",
    "weakt_active_thermal_refinement_p3_2d",
    "weakt_active_thermal_refinement_low_sigc",
)
SOURCE_NAMES = (
    "design_r10_weakt_active_acquisition.py",
    "design_r10_weakt_active_thermal_refinement.py",
    "analyze_r10_weakt_active_acquisition.py",
    "package_r10_weakt_active_acquisition.py",
    "run_r10_current_fracture_panel.py",
    "run_v10_2_30_weakt_high_cycle_1e12.sh",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    destination = args.out.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)

    for name in RUN_NAMES:
        source = ROOT / "runs/r10_joint_search" / name
        shutil.copytree(
            source, destination / "executed_runs" / name,
            ignore=shutil.ignore_patterns("s3p0_seed4810_.log"),
        )
    shutil.copytree(args.analysis_root, destination / "analysis")
    for name in SOURCE_NAMES:
        source = ROOT / "scripts" / name
        target = destination / "source/scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    test = ROOT / "tests/test_r10_weakt_active_acquisition.py"
    (destination / "source/tests").mkdir(parents=True, exist_ok=True)
    shutil.copy2(test, destination / "source/tests" / test.name)

    decision = json.loads((args.analysis_root / "joint_scientific_decision.json").read_text())
    readme = f"""# R10 active Weak-T joint acquisition

Joint scientific decision: `{decision['joint_scientific_decision']}`.

The existing opening-dominated result is preserved at its stated scope.  This
continuation instead starts from the mechanically active Weak-T row and changes
only permitted EXP_floor material coordinates.  Five P4 cold emission knees
were serialized, four received native 15/22 MPa sqrt(m) trajectories, and the
selected 3.0 GPa row received a full five-amplitude native curve with checkpointed
state evolution.  Its developed fatigue metrics are in `analysis/`.

The 3.0 GPa row is a fatigue-qualified active frontier, not a promoted joint
material.  A dK=0.01 fracture calculation exposed the coarse dK=0.05 thermal
screen as a false positive.  Executed P3 hot-opening brackets did not recover
the complete <=1.25 span, so the same-row 2-D gate was not opened.  The package
retains the complete run results and restart checkpoints supporting both the
positive fatigue conclusion and the negative joint decision.
"""
    (destination / "README.md").write_text(readme)
    (destination / "COMMANDS.md").write_text(
        "# Execution identity\n\n"
        "Python: `/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python`\n\n"
        "The exact candidate vectors, parents, changed fields, surface audits, run-time result "
        "identities, and checkpoint directories are embedded under `executed_runs/`.\n"
    )
    files = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files[str(path.relative_to(destination))] = {
                "sha256": sha256(path), "bytes": path.stat().st_size,
            }
    manifest = {
        "schema": "r10.active-weakt-acquisition-package/1",
        "git_head_before_package_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "joint_scientific_decision": decision["joint_scientific_decision"],
        "files": files,
    }
    (destination / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(destination), "files": len(files) + 1}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
