#!/usr/bin/env python3
"""Single idempotent entry point for the additive R10 cohesive campaign."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE=ROOT/"integration_evidence/r10_cohesive_arrhenius_joint_sweep"
DEFAULT_RUNS=ROOT/"runs/r10_cohesive_arrhenius_joint_sweep"
ORIGINAL=ROOT/"arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv"
PARENTS=ROOT/"integration_evidence/r10_joint_search_continuation/inputs/fatigue_parent_registry.csv"


def call(*args):
    print(json.dumps({"command":[str(x) for x in args]}),flush=True)
    subprocess.run([str(x) for x in args],cwd=ROOT,check=True)


def selected_candidate(registry: Path) -> str:
    with registry.open(newline="") as stream:return next(csv.DictReader(stream))["candidate_id"]


def write_ledger(evidence: Path,runs: Path):
    protocol=json.loads((evidence/"prospective_search_protocol.json").read_text())
    expansion_path=evidence/"thermal_hardening_expansion/decision.json"
    expansion=(json.loads(expansion_path.read_text()) if expansion_path.exists() else
               {"allocated":0,"analytically_evaluated":0,"domain_valid":0,"native_queued_not_executed":0})
    fatigue=(list(runs.glob("native/*/fatigue/Kmax_*/result.json"))+
             list(runs.glob("native_parallel/*/fatigue/Kmax_*/result.json")))
    fracture=list(runs.glob("native/*/fracture/T_*K.json"))
    spatial=list(runs.glob("spatial/*/qualification.json"))
    developed=0;censored=0;restartable=0
    for path in fatigue:
        result=json.loads(path.read_text())
        if result.get("status")=="developed_target_completed":developed+=1
        else:censored+=1
        if (result.get("status")!="developed_target_completed" and
                (path.parent/"restart/high_cycle_live_checkpoint.json").exists()):restartable+=1
    payload={"schema":"r10.cohesive-campaign-ledger/1","family":protocol["family"],
        "allocated":protocol["allocated"]+expansion["allocated"],
        "analytically_evaluated":protocol["analytically_evaluated"]+expansion["analytically_evaluated"],
        "analytical_domain_valid":protocol["domain_valid"]+expansion["domain_valid"],
        "state_aware_allocated":protocol["state_aware_allocated_not_executed"],
        "state_aware_evaluated":0,"native_queued":max(
            protocol["native_queued_not_executed"]+expansion["native_queued_not_executed"]-2,0),
        "native_fatigue_executed":len(fatigue),"native_fracture_executed":len(fracture),
        "developed":developed,"physically_censored":censored,"restartable":restartable,
        "kinetic_czm_spatial_qualifications":len(spatial),"numerically_unresolved":0,
        "qualified_joint_materials":0,
        "classification_status":"ACQUISITION_AND_EXECUTION_IN_PROGRESS",
        "open_cells":"reported in portfolio.csv; unexecuted cells remain NOT_YET_FOUND",
        "counts_are_execution_scoped":True}
    path=evidence/"continuation_ledger.json";tmp=path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");tmp.replace(path)
    print(json.dumps(payload,indent=2,sort_keys=True))


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("qualification","acquisition","expansion","spatial","native","ledger","all"))
    p.add_argument("--evidence",type=Path,default=DEFAULT_EVIDENCE);p.add_argument("--runs",type=Path,default=DEFAULT_RUNS)
    p.add_argument("--candidate-id");p.add_argument("--native-mode",choices=("fracture","fatigue","all"),default="all")
    p.add_argument("--loads");p.add_argument("--maximum-cycles",type=float,default=1e12)
    a=p.parse_args(argv);a.evidence.mkdir(parents=True,exist_ok=True);a.runs.mkdir(parents=True,exist_ok=True)
    py=sys.executable
    if a.stage in ("qualification","all"):
        call(py,"-m","pytest","-q","tests/test_cohesive_arrhenius_r10.py","tests/test_retained_material_r10.py")
    if a.stage in ("acquisition","all") and not (a.evidence/"prospective_search_protocol.json").exists():
        call(py,"scripts/run_r10_cohesive_arrhenius_search.py","--original-registry",ORIGINAL,
             "--parent-registry",PARENTS,"--out",a.evidence,"--log2-bank","15","--native-count","48")
    expansion=a.evidence/"thermal_hardening_expansion"
    if a.stage in ("expansion","all") and not (expansion/"decision.json").exists():
        call(py,"scripts/run_r10_cohesive_thermal_expansion.py","--original-registry",ORIGINAL,
             "--parent-registry",PARENTS,"--out",expansion,"--log2-bank","13")
    registry=a.evidence/"native_batch_registry.csv"
    if a.stage in ("spatial","native","all"):
        if not registry.exists():raise FileNotFoundError("run acquisition before native/spatial stages")
        candidate=a.candidate_id or selected_candidate(registry)
        if a.stage in ("spatial","all"):
            call(py,"scripts/qualify_r10_cohesive_2d.py","--registry",registry,"--candidate-id",candidate,
                 "--out",a.runs/"spatial"/candidate)
        if a.stage in ("native","all"):
            command=[py,"scripts/run_r10_cohesive_native.py","--registry",registry,"--candidate-id",candidate,
                     "--out",a.runs/"native"/candidate,"--mode",a.native_mode,
                     "--maximum-cycles",str(a.maximum_cycles)]
            if a.loads:command.extend(("--loads",a.loads))
            call(*command)
    if a.stage in ("ledger","all"):write_ledger(a.evidence,a.runs)
    return 0

if __name__=="__main__":raise SystemExit(main())
