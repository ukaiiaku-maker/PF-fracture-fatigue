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
        "ancestry_qualified_continuation": run / "ancestry_qualified_continuation",
        "original_weakt_fracture": run / "fracture/original_weakt_ancestry_qualified_refined",
        "original_weakt_fatigue_attempt": run / "bridges/original_weakt_ancestry_qualified_seed4810",
        "p40_thermal_diagnostic": run / "fracture/p40_transfer_ancestry_qualified_refined",
        "preserved_class_causal_record.json": run / "thermal_accessibility_iteration/thermal_accessibility_result.json",
    }
    for name, source in sources.items():
        copy(source, a.out / "executed" / name)
    for name in (
        "reconcile_r10_constitutive_temperature_lineage.py",
        "design_r10_weakt_coupled_scale_continuation.py",
        "design_r10_weakt_coupled_opening_scale.py",
        "analyze_r10_constitutive_coupled_continuation.py",
        "analyze_r10_ancestry_qualified_continuation.py",
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
    next_result = json.loads((sources["ancestry_qualified_continuation"] / "joint_result.json").read_text())
    readme = f"""# R10 complete-lineage reconciliation and ancestry-qualified continuation

The full original -> projection -> level-shift -> P3/P4 endpoint -> final-scale ancestry is reconciled in `executed/audit/reconciliation.json` and its source-to-field companion CSV. The portable W[100] export establishes the formula and its 298-673 K fit only; it does not authorize the later four-process 1200 K endpoint constructions. Those rows and all calculations are preserved, but the endpoint-descended row is non-promotable. No empirical temperature law was introduced and no entropy coefficient was zeroed.

The tested candidate `{decision['candidate_id']}` (hash `{decision['complete_row_sha256']}`) is retained at its actual scope: a fresh two-point 15/22 MPa sqrt(m) fatigue secant of {decision['fatigue']['slope']:.6f} over {decision['fatigue']['new_rate_span_decades']:.6f} rate decades and a refined fracture span of {decision['fracture']['span']:.6f}. Its coherent stress scale preserves named energy/entropy coefficients, but it rescales the derived entropy landscape in stress; it does not cure the inherited endpoint ancestry.

The post-audit search resumed from canonical original Weak-T row `{next_result['selected_candidate']['complete_row_sha256']}` without a new fit. Its current-backend fracture response completed 55 um at 300 K with K-init 14.431541 MPa sqrt(m), then produced zero events through Kmax 80 MPa sqrt(m) at 450 K. The same-row native fatigue attempt committed no event or post-initial atomic checkpoint in 283 seconds and is reported only as a quantified numerical limit, not a Paris slope. Joint promotion and same-row 2-D were therefore withheld. The P40 fatigue control was separately diagnosed and rejected as a thermal parent because its opening/emission affine temperature slopes are zero.

The unexecuted 2.6/2.8 emission-scale and 0.6/0.8 opening-scale hypotheses remain `UNEVALUATED_PREDICTED_UNCOMPETITIVE`, not dominated. The next search remains open to every source-authorized nucleation and motion coordinate with its original coupling; there is no P1/P2-only or opening-only restriction.

Existing DBTT and Peak-T classifications are preserved at their demonstrated diagnostic scope. DBTT remains intrinsically rising with hot-state enhancement; plasticity-created DBTT was not established. Peak-T remains opening-dominated with its A0 maximum already at 600 K; plasticity-generated peaking was not established, and the 600-to-750 K root difference was below the executed scan resolution. These causal records do not promote their endpoint-constructed descendants.
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
