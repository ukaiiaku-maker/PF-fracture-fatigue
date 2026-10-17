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
    state_summaries=[json.loads(path.read_text()) for path in
                     evidence.glob("state_aware_execution*/state_aware_summary.json")]
    kinetic_analyses=list((evidence/"kinetic_czm_execution").glob("*/analysis_summary.json"))
    active_comparisons=list((evidence/"kinetic_czm_execution").glob("*/active_comparison_summary.json"))
    timescales=[json.loads(path.read_text()) for path in evidence.glob("kinetic_czm_timescale_acquisition*/timescale_acquisition.json")]
    timescale_summaries=[row for payload in timescales for row in payload.get("summaries",[])]
    observed_controls=[row for payload in timescales for row in payload.get("observed_advancing_controls",[])]
    resource_path=evidence/"kinetic_czm_timescale_acquisition/resource_dispositions.json"
    resource=(json.loads(resource_path.read_text()) if resource_path.exists() else {"materials":[]})
    developed=0;censored=0;restartable=0
    for path in fatigue:
        result=json.loads(path.read_text())
        if result.get("status")=="developed_target_completed":developed+=1
        else:censored+=1
        if (result.get("status")!="developed_target_completed" and
                (path.parent/"restart/high_cycle_live_checkpoint.json").exists()):restartable+=1
    executed_native_ids=set()
    for path in fatigue:
        result=json.loads(path.read_text());candidate=result.get("candidate_id")
        if candidate:executed_native_ids.add(candidate)
    total_native_queue=protocol["native_queued_not_executed"]+expansion["native_queued_not_executed"]
    queued_ids=[]
    for registry in (evidence/"native_batch_registry.csv",expansion_path.parent/"native_batch_registry.csv"):
        if registry.exists():
            with registry.open(newline="") as stream:queued_ids.extend(r["candidate_id"] for r in csv.DictReader(stream))
    screen_closed_ids=set()
    for disposition in evidence.glob("state_aware_execution*/state_aware_dispositions.csv"):
        with disposition.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if (row.get("mechanism_screen_disposition")=="OPENING_INSENSITIVE_SCREEN" and
                    row.get("all_adjacent_primary_target","").lower()=="false" and
                    int(row.get("state_aware_loads",0))>=5 and int(row.get("cycles_per_load",0))>=12):
                    screen_closed_ids.add(row["candidate_id"].removesuffix("_J"))
    qualified_kinetic=0
    for path in kinetic_analyses:
        result=json.loads(path.read_text())
        qualified_kinetic+=int(bool(result.get("moderate_slope_pass")))
    qualified_portfolio=0
    portfolio=evidence/"portfolio.csv"
    if portfolio.exists():
        with portfolio.open(newline="") as stream:
            for row in csv.DictReader(stream):
                qualified_portfolio+=sum(str(value).startswith("QUALIFIED") for key,value in row.items()
                                         if key!="fracture_class")
    payload={"schema":"r10.cohesive-campaign-ledger/2","family":protocol["family"],
        "allocated":protocol["allocated"]+expansion["allocated"],
        "analytically_evaluated":protocol["analytically_evaluated"]+expansion["analytically_evaluated"],
        "analytical_domain_valid":protocol["domain_valid"]+expansion["domain_valid"],
        "state_aware_allocated":protocol["state_aware_allocated_not_executed"],
        "state_aware_execution_blocks":len(state_summaries),
        "state_aware_materials_evaluated":sum(int(x.get("executed_complete_materials",0)) for x in state_summaries),
        "state_aware_load_conditions_evaluated":sum(int(x.get("executed_load_conditions",0)) for x in state_summaries),
        "state_aware_informative":sum(int(x.get("informative_count",0)) for x in state_summaries),
        "native_queued":max(total_native_queue-len(executed_native_ids)-len(set(queued_ids)&screen_closed_ids),0),
        "native_queue_deprioritized_by_finite_12_cycle_screen":len(set(queued_ids)&screen_closed_ids),
        "native_fatigue_executed":len(fatigue),"native_fracture_executed":len(fracture),
        "developed":developed,"physically_censored":censored,"restartable":restartable,
        "kinetic_czm_spatial_qualifications":len(spatial),
        "kinetic_czm_advancing_materials_analyzed":len({json.loads(path.read_text())["candidate_id"] for path in kinetic_analyses+active_comparisons}|{x["candidate_id"] for x in observed_controls}),
        "kinetic_czm_active_dislocation_comparisons":len(active_comparisons),
        "kinetic_czm_timescale_materials_evaluated":len(timescale_summaries),
        "kinetic_czm_same_material_thermal_panels":len(list((evidence/"kinetic_czm_execution").glob("*/thermal_analysis_summary.json"))),
        "numerically_unresolved":0,"resource_limited_active_trajectory_materials":len(resource["materials"]),
        "qualified_joint_materials":qualified_portfolio+qualified_kinetic,
        "classification_status":"ACQUISITION_AND_EXECUTION_IN_PROGRESS",
        "open_cells":"reported in portfolio.csv; unexecuted cells remain NOT_YET_FOUND",
        "counts_are_execution_scoped":True}
    path=evidence/"continuation_ledger.json";tmp=path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");tmp.replace(path)
    checkpoints=sorted(str(path.relative_to(ROOT)) for path in runs.glob("**/restart/high_cycle_live_checkpoint.json"))
    remaining_order=[]
    for candidate in queued_ids:
        if candidate not in executed_native_ids and candidate not in screen_closed_ids and candidate not in remaining_order:
            remaining_order.append(candidate)
    next_native=remaining_order[0] if remaining_order else None
    advancing_ids={json.loads(path.read_text())["candidate_id"] for path in kinetic_analyses+active_comparisons}
    advancing_ids.update(x["candidate_id"] for x in observed_controls)
    queue_status={"schema":"r10.cohesive-queue-status/1","active_writers":[],
        "active_job_status":"NONE_AT_LEDGER_SNAPSHOT","native_queue_total":len(set(queued_ids)),
        "native_completed_material_ids":sorted(executed_native_ids),
        "native_state_screen_deprioritized_material_ids":sorted(set(queued_ids)&screen_closed_ids),
        "native_state_screen_scope":"FINITE_12_CYCLE_PROMOTION_SCREEN__NOT_LIFETIME_INACTIVITY",
        "native_remaining_material_ids":sorted(set(queued_ids)-executed_native_ids-screen_closed_ids),
        "accepted_checkpoint_paths":checkpoints,
        "advancing_kinetic_completed_material_ids":sorted(advancing_ids),
        "phase_alias_history":"RETAINED_BUT_EXCLUDED; 64/128-PHASE CHECK RESOLVED",
        "next_native_information_candidate":next_native,"next_native_command_identity":"COHESIVE_BARRIER_ONLY",
        "kinetic_czm_starting_hypothesis_verified":"R10D_COH_19672" if timescales else None,
        "kinetic_czm_starting_hypothesis_disposition":"OPENING_BURST_BEFORE_STATE_DEVELOPMENT__NOT_A_DEVELOPED_RATE" if timescales else None,
        "kinetic_czm_active_comparison_completed":"R10D_COH_30017" if active_comparisons else None,
        "kinetic_czm_active_comparison_disposition":"SOURCE_ACTIVE_BLUNTING_SUPPRESSES_RECURRENT_OPENING_OVER_99_CYCLES" if active_comparisons else None,
        "kinetic_czm_next_acquisition_requirement":"NEW_JOINT_VARIATION_OUTSIDE_THE_NARROW_OPENING_TRANSITION__CURRENT_ACTIVE_ROW_HAS_NO_DEVELOPED_FULL_STATE_WINDOW" if active_comparisons else None,
        "barrier_only_resume_command":(None if next_native is None else
            f"python scripts/run_r10_cohesive_campaign.py native --candidate-id {next_native} --maximum-cycles 100000"),
        "kinetic_czm_reproduce_active_comparison_command":("python scripts/run_r10_kinetic_czm_coupled.py --registry integration_evidence/r10_cohesive_arrhenius_joint_sweep/kinetic_czm_execution/kinetic_czm_registry.csv --candidate-id R10D_COH_30017 --out runs/r10_cohesive_arrhenius_joint_sweep/kinetic_czm_timescale/R10D_COH_30017 --loads 28.914041499941035 --target-events 5 --target-extension-um 20 --maximum-cycles 100 --phase-count 12" if active_comparisons else None)}
    queue_path=evidence/"queue_status.json";queue_tmp=queue_path.with_suffix(".json.tmp")
    queue_tmp.write_text(json.dumps(queue_status,indent=2,sort_keys=True)+"\n");queue_tmp.replace(queue_path)
    print(json.dumps(payload,indent=2,sort_keys=True))


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("qualification","acquisition","expansion","state-aware","spatial","native","kinetic","ledger","continuation","all"))
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
    state_out=a.evidence/"state_aware_execution"
    if a.stage in ("state-aware","continuation","all") and not (state_out/"state_aware_summary.json").exists():
        call(py,"scripts/run_r10_cohesive_state_aware.py","--allocation",a.evidence/"state_aware_allocation.csv",
             "--original-registry",ORIGINAL,"--parent-registry",PARENTS,"--out",state_out,
             "--count-per-stratum","3","--cycles","12","--phase-count","32")
    state_out2=a.evidence/"state_aware_execution_block2"
    if a.stage in ("state-aware","continuation","all") and not (state_out2/"state_aware_summary.json").exists():
        call(py,"scripts/run_r10_cohesive_state_aware.py","--allocation",a.evidence/"state_aware_allocation.csv",
             "--original-registry",ORIGINAL,"--parent-registry",PARENTS,"--out",state_out2,
             "--count-per-stratum","2","--cycles","12","--phase-count","32",
             "--exclude-registry",state_out/"state_aware_registry.csv")
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
    if a.stage in ("kinetic","continuation","all"):
        kinetic_registry=a.evidence/"kinetic_czm_execution/kinetic_czm_registry.csv"
        if not kinetic_registry.exists():raise FileNotFoundError("kinetic CZM selection registry is absent")
        candidate=a.candidate_id or "R10D_COH_10980"
        run64=a.runs/"kinetic_czm"/f"{candidate}_phase64_developed"
        call(py,"scripts/run_r10_kinetic_czm_coupled.py","--registry",kinetic_registry,
             "--candidate-id",candidate,"--out",run64,"--target-events","15",
             "--target-extension-um","70","--maximum-cycles",str(a.maximum_cycles),"--phase-count","64")
        summary=json.loads((run64/"campaign_summary.json").read_text());middle=float(summary["loads"][1])
        run128=a.runs/"kinetic_czm"/f"{candidate}_phase128_selected_mid"
        call(py,"scripts/run_r10_kinetic_czm_coupled.py","--registry",kinetic_registry,
             "--candidate-id",candidate,"--out",run128,"--loads",str(middle),
             "--target-events","15","--target-extension-um","70",
             "--maximum-cycles",str(a.maximum_cycles),"--phase-count","128")
        call(py,"scripts/analyze_r10_kinetic_czm_coupled.py","--run",run64,
             "--phase128",run128,"--out",a.evidence/"kinetic_czm_execution"/candidate)
        thermal=a.runs/"kinetic_czm_thermal"/candidate
        call(py,"scripts/run_r10_kinetic_czm_thermal.py","--registry",kinetic_registry,
             "--candidate-id",candidate,"--out",thermal)
        call(py,"scripts/analyze_r10_kinetic_czm_thermal.py","--run",thermal,
             "--out",a.evidence/"kinetic_czm_execution"/candidate)
    if a.stage in ("ledger","continuation","all"):write_ledger(a.evidence,a.runs)
    return 0

if __name__=="__main__":raise SystemExit(main())
