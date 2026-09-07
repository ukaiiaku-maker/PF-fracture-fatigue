#!/usr/bin/env python3
"""Fail-closed verifier for the all-1-D analytical-overlay bundle."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/A_native_analytical_overlay_all_1d_v1"
EXPECTED = {
    "physical_condition_inventory.csv",
    "analytical_predictions.csv",
    "analytical_error_summary.csv",
    "stationary_state_comparison.csv",
    "CT_analytical_life_comparison.csv",
    "CT_analytical_curves.csv",
    "analytical_input_manifest.json",
    "analytical_overlay_decision.md",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def sha(path: Path) -> str:
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def rows(name: str):
    with (OUT/name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    missing=[name for name in EXPECTED if not (OUT/name).is_file()]
    if missing: fail(f"missing artifacts: {missing}")
    manifest=json.loads((OUT/"analytical_input_manifest.json").read_text())
    if manifest["source_HEAD"] != "d727cbde36f240214086ac2134985bcd023742fc": fail("wrong source HEAD")
    if manifest["production_solver_sha256"] != "c15a957161e8cdb43bf944243d8a18a64839dbfe0518aa2d1c5f0c93773b817b": fail("wrong solver hash")
    if manifest["fitted_to_da_dN"] or manifest["empirical_Paris_law_used"]: fail("forbidden fit")
    if manifest["predeclared_descriptive_error_bands_decade"] != [0.05,0.1,0.3]: fail("error bands changed")
    registry=ROOT/manifest["parameter_registry"]
    if sha(registry) != manifest["parameter_registry_sha256"]: fail("parameter registry changed")
    for rel, expected in manifest["result_summary_sha256"].items():
        path=ROOT/rel
        if not path.exists() or sha(path)!=expected: fail(f"archived result modified: {rel}")
    inventory=rows("physical_condition_inventory.csv")
    if len(inventory)!=146: fail(f"inventory count {len(inventory)}")
    if len({r["source_result_path"] for r in inventory})!=146: fail("duplicate source admitted")
    classes={key:sum(r["stationarity_classification"]==key for r in inventory) for key in {r["stationarity_classification"] for r in inventory}}
    if classes != {"STEADY_STATE_QUALIFIED":110,"NUMERICAL_VALIDATION_DUPLICATE":27,"TRANSIENT_NOT_QUALIFIED":9}: fail(f"qualification counts {classes}")
    if any(r["analytical_status"] != "PREDICTED_IN_DOMAIN" for r in inventory): fail("unclassified extrapolation")
    errors=rows("analytical_error_summary.csv")
    overall=[r for r in errors if r["grouping"]=="overall"]
    if len(overall)!=3 or any(int(r["count"])!=110 for r in overall): fail("nonstationary error admission")
    if len(rows("CT_analytical_life_comparison.csv"))!=42: fail("virtual path count")
    figures=list((OUT/"figures").glob("*.png"))
    if len(figures)!=9 or any(p.stat().st_size<10_000 for p in figures): fail("figure gate")
    report=(OUT/"analytical_overlay_decision.md").read_text()
    if "ANALYTICAL_STEADY_STATE_PARTIAL" not in report: fail("missing decision")
    if "No archived trajectory was modified and no physics calculation was run" not in report: fail("missing no-physics statement")
    ps=subprocess.run(["pgrep","-fal","sharp_front_v10_2_30|run_v10_2_30"],text=True,capture_output=True)
    active=[line for line in ps.stdout.splitlines() if "verify_v10_2_30" not in line]
    if active: fail(f"active physics workers: {active}")
    branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    payload={
        "schema":"v10.2.30_analytical_overlay_verification_v1",
        "result":"PASS",
        "branch":branch,
        "head":head,
        "source_head":manifest["source_HEAD"],
        "solver_sha256":manifest["production_solver_sha256"],
        "inventory_count":146,
        "steady_state_qualified_count":110,
        "numerical_validation_duplicate_count":27,
        "transient_not_qualified_count":9,
        "virtual_path_count":42,
        "figure_count":9,
        "active_worker_count":0,
        "archived_result_hashes_unchanged":True,
        "fitted_parameter_count":0,
        "physics_calculations_rerun":False,
        "worktree_clean":not bool(dirty),
    }
    (OUT/"analytical_overlay_verification.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,sort_keys=True))


if __name__ == "__main__":
    main()
