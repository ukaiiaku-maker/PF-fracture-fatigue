#!/usr/bin/env python3
"""Export the reconciled constitutive audit and executed coupled continuation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.npz", "*.tmp"))
    else:
        shutil.copy2(source, destination)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(a.out)
    run = ROOT / "runs/r10_joint_search"
    sources = {
        "audit": run / "constitutive_temperature_reconciliation",
        "coupled_scale": run / "weakt_coupled_scale_continuation",
        "scale_response": run / "weakt_coupled_scale_bracket",
        "opening_response": run / "weakt_coupled_opening_scale",
        "analysis": run / "constitutive_coupled_continuation_analysis",
    }
    for name, source in sources.items():
        copy(source, a.out / "executed" / name)
    for name in (
        "reconcile_r10_constitutive_temperature_lineage.py",
        "design_r10_weakt_coupled_scale_continuation.py",
        "design_r10_weakt_coupled_opening_scale.py",
        "analyze_r10_constitutive_coupled_continuation.py",
        "package_r10_constitutive_coupled_continuation.py",
    ):
        copy(ROOT / "scripts" / name, a.out / "source/scripts" / name)
    copy(ROOT / "arrhenius_fracture/material_manifest.py",
         a.out / "source/arrhenius_fracture/material_manifest.py")
    copy(ROOT / "arrhenius_fracture/retained_material_r10.py",
         a.out / "source/arrhenius_fracture/retained_material_r10.py")
    copy(ROOT / "scripts/run_r10_current_fracture_panel.py",
         a.out / "source/scripts/run_r10_current_fracture_panel.py")
    copy(ROOT / "scripts/run_r10_retained_bridge.py",
         a.out / "source/scripts/run_r10_retained_bridge.py")

    decision = json.loads((sources["analysis"] / "joint_scientific_decision.json").read_text())
    readme = f"""# R10 constitutive-temperature reconciliation and coupled continuation

The portable W[100] source was reconciled before new candidate generation. Independent cold/hot endpoint rows are preserved as non-promotable historical evidence. No empirical temperature law was introduced and no activation entropy was zeroed.

The corrected candidate `{decision['candidate_id']}` uses one multiplicative emission characteristic-stress scale: both `emit_sigc0` and `emit_sT` move together, while Peierls and Taylor inherit the same `sigc(T)`. Its opening surface, all affine energy slopes, nonzero transport entropies, floors, shapes, state evolution, event law, and mechanics remain intact.

Fresh fracture and fatigue calculations are bound to the same complete-row hash `{decision['complete_row_sha256']}`. The fatigue high pair is recorded in `executed/analysis/corrected_fatigue_high_pair.csv`; the refined seven-temperature fracture curve fails the 1.25 weak-temperature span gate, so there is no promotion and same-row 2-D is correctly withheld. Adjacent coupled-coordinate response rows are retained as negative information; dominated interpolants were not executed.
"""
    (a.out / "README.md").write_text(readme)
    manifest = {}
    for path in sorted(x for x in a.out.rglob("*") if x.is_file()):
        rel = str(path.relative_to(a.out))
        manifest[rel] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (a.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(a.out), "files": len(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
