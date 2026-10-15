#!/usr/bin/env python3
"""Package the source-complete R10 thermal-accessibility continuation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


SELECTED = (
    "R10T_PEAK_P40_G2600_S0P25GPA_P4_LOW_TAYLOR_V1",
    "R10T_DBTT_P40_G3075_S0P25GPA_P4_EP400_PP040_V1",
    "R10T_WEAKT_P40_G1962_S4P5GPA_P4_EP055_PP010_V1",
)


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv); args.out.mkdir(parents=True, exist_ok=True)
    iteration = args.run_root / "thermal_accessibility_iteration"
    names = (
        "thermal_accessibility_scan.csv", "thermal_state_candidates.csv",
        "thermal_accessibility_decision.json", "accessibility_bound_audit.json",
        "thermal_accessibility_result.json",
        "full_state_case_table.csv", "causal_decomposition.csv",
        "fatigue_case_table.csv", "fatigue_windows.csv", "fatigue_two_windows.csv",
    )
    for name in names:
        copy(iteration / name, args.out / "results" / name)
    copy(args.run_root / "continuation_state.json", args.out / "continuation_state.json")
    copy(args.run_root / "reused_sealed_2d_terminal_summary.csv",
         args.out / "spatial" / "reused_sealed_2d_terminal_summary.csv")
    spatial = iteration / "spatial" / "dbtt_900K_current_backend"
    for source in sorted(x for x in spatial.rglob("*") if x.is_file()):
        copy(source, args.out / "spatial" / "dbtt_900K_current_backend" / source.relative_to(spatial))
    for candidate in SELECTED:
        for source in sorted((iteration / "full_state" / candidate).glob("T_*K.json")):
            copy(source, args.out / "full_state" / candidate / source.name)
    for campaign in ("peak_seed4810", "peak_seed9117"):
        root = iteration / "fatigue" / campaign
        for source in sorted(root.glob("Kmax_*/result.json")):
            copy(source, args.out / "fatigue" / campaign / source.parent.name / source.name)
        for name in ("campaign_summary.json", "production_preflight.json"):
            copy(root / name, args.out / "fatigue" / campaign / name)
    weak = iteration / "fatigue" / "weakt_seed4810"
    for source in sorted(x for x in weak.rglob("*") if x.is_file()):
        copy(source, args.out / "fatigue" / "weakt_seed4810" / source.relative_to(weak))
    for campaign in ("r10j_dbtt_p40_lshift_036mev_seed4810", "r10j_dbtt_p40_lshift_036mev_seed9117"):
        for source in sorted((args.run_root / "bridges" / campaign).glob("Kmax_*/result.json")):
            copy(source, args.out / "fatigue" / "reused_dbtt_control" / campaign / source.parent.name / source.name)

    repository = Path(__file__).resolve().parents[1]
    for name in (
        "design_r10_thermal_accessibility_iteration.py",
        "analyze_r10_thermal_accessibility_iteration.py",
        "run_r10_current_fracture_panel.py", "run_r10_retained_bridge.py",
        "execute_r8_current_backend_spatial.py",
        "run_off_axis_prescribed_void_benchmark_v2.py",
        "package_r10_thermal_accessibility.py",
    ):
        copy(repository / "scripts" / name, args.out / "source" / "scripts" / name)
    for name in ("thermal_accessibility_r10.py", "retained_material_r10.py",
                 "unified_fracture_material_v5.py", "paris_slope_r9.py",
                 "persistent_site_first_passage_locator_v10230.py"):
        copy(repository / "arrhenius_fracture" / name,
             args.out / "source" / "arrhenius_fracture" / name)
    copy(repository / "tests" / "test_r10_thermal_accessibility.py",
         args.out / "source" / "tests" / "test_r10_thermal_accessibility.py")
    copy(repository / "tests" / "test_paris_slope_r9.py",
         args.out / "source" / "tests" / "test_paris_slope_r9.py")

    commands = """# Reproduction commands

Use `/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python`.

```bash
python scripts/design_r10_thermal_accessibility_iteration.py --joint-registry runs/r10_joint_search/joint_level_iteration/joint_level_candidates.csv --out-root runs/r10_joint_search/thermal_accessibility_iteration
python scripts/run_r10_current_fracture_panel.py --help
python scripts/run_r10_retained_bridge.py --help
python scripts/analyze_r10_thermal_accessibility_iteration.py --root runs/r10_joint_search/thermal_accessibility_iteration --continuation-state runs/r10_joint_search/continuation_state.json
python -m pytest -q
```
"""
    (args.out / "COMMANDS.md").write_text(commands)
    result = json.loads((iteration / "thermal_accessibility_result.json").read_text())
    spatial_status = result["spatial"]["status"]
    readme = f"""# R10 thermal-accessibility continuation

This package continues the existing family-A search in the saved P3/P4 coordinates. It adds a certified, channel-local F0 opening bound; preserves exact 300 K process surfaces; enforces positive raw-domain endpoints on prospective rows; and records complete-state thermal and fatigue evidence without changing the barrier, event, or mechanics laws.

Current selections are `{SELECTED[0]}` (opening-dominated Peak-T topology and two-seed fatigue pass), `{SELECTED[1]}` (intrinsically rising DBTT with important hot current-state enhancement and exact-contract fatigue reuse), and `{SELECTED[2]}` (bounded compensated Weak-T fracture response). The Weak-T mean-cycle accelerator remains rejected, but the retained native phase-resolved engine completed all five requested developed trajectories. The five-point curve fails the established slope/width objective and returns to the existing class-specific selection loop; this is now a physical candidate result rather than a numerical censor. The ceramic rows remain controls without a Paris requirement.

The selected DBTT row was passed through a narrow exact-row spatial binding that retains the qualified non-fracture owners; its current status is `{spatial_status}`. The sealed historical same-row controls are also copied under `spatial/`.

Machine-readable dispositions are in `results/thermal_accessibility_result.json`. The result schema is `{result['schema']}`.
"""
    (args.out / "README.md").write_text(readme)
    manifest = {}
    for path in sorted(x for x in args.out.rglob("*") if x.is_file() and x.name != "MANIFEST.json"):
        manifest[str(path.relative_to(args.out))] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
        }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"files": len(manifest), "out": str(args.out.resolve())}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
