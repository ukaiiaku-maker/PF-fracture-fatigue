#!/usr/bin/env python3
"""Prepare and supervise the final 32-task dense-DeltaK production campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

try:
    from scripts import v10230_qualification_supervisor as q
    from scripts.v10230_qualification_family import validate as validate_family
except ModuleNotFoundError:
    import v10230_qualification_supervisor as q
    from v10230_qualification_family import validate as validate_family

from arrhenius_fracture.run_state_checkpoint_v10230 import load_combined_checkpoint, validate_cross_layer

FRACTIONS = (0.925, 0.900, 0.875, 0.850, 0.825, 0.800, 0.775, 0.750)
CYCLES_MAX = 1e14
TARGET_UM = 100.0
SOURCE_NAME = "dense_restart_source.json"
BASE_MATRIX = q.matrix
BASE_CLASSIFY = q.classify


def matrix() -> list[dict]:
    rows=[]
    for fraction in FRACTIONS:
        for label,(option,seed,critical) in q.OPTIONS.items():
            rows.append({"case":f"{label}_f{fraction:.3f}".replace(".","p")+f"_seed{seed}",
                "label":label,"parameter_option":option,"seed":seed,"fraction":fraction,
                "deltaK_MPa_sqrt_m":critical*fraction,"Kmax_MPa_sqrt_m":critical*fraction/0.9,
                "mode":"resumed" if fraction==.75 else "fresh","cycle_horizon":CYCLES_MAX,
                "target_extension_um":TARGET_UM})
    return rows


def inspect_resume(source_root: Path, row: dict) -> dict:
    source_case=source_root/f'{row["label"]}_f0p75_seed{row["seed"]}'; output=source_case/"output"
    if not q.checkpoint_valid(source_case,row): raise RuntimeError(f'{row["case"]}: invalid source checkpoint')
    outer,kinetic,_=load_combined_checkpoint(output); validate_cross_layer(outer,kinetic)
    control=q.read_json(output/"v10_2_30_fixed_deltaK_control.json")
    descriptor=q.read_json(output/"run_state_checkpoint.json"); stochastic=kinetic.get("stochastic",{})
    selection=q.read_json(output/"v10_2_22_parameter_selection.json")
    option=outer.get("case",{}).get("parameter_option") or selection.get("exact_registry_row",{}).get("option_key")
    checks={"option":option==row["parameter_option"],"temperature":outer["case"].get("temperature_K")==300.0,
        "deltaK":outer["case"].get("deltaK_MPa_sqrt_m")==row["deltaK_MPa_sqrt_m"],"R":outer["case"].get("R")==.1,
        "frequency":outer["case"].get("frequency_Hz")==1000.0,"seed":outer["case"].get("seed")==row["seed"],
        "da_phys":outer["case"].get("da_phys_m")==5e-6,"old_horizon":control.get("cycles_max")==1e12,
        "restored_cycles":outer.get("cycles_total")==1e12,"events":outer.get("geometry",{}).get("committed_event_count")==stochastic.get("hazard_event_index")}
    if not all(checks.values()): raise RuntimeError(f'{row["case"]}: resume invariant failed: {checks}')
    return {"source_case":str(source_case.resolve()),"source_checkpoint":str(output.resolve()),
        "source_checkpoint_generation":descriptor["generation"],"starting_cycles":outer["cycles_total"],
        "starting_event_count":outer["geometry"]["committed_event_count"],"starting_extension_um":0.0,
        "old_cycle_horizon":1e12,"new_cycle_horizon":CYCLES_MAX,"stochastic":stochastic,"checks":checks}


def preflight(source_root: Path, destination: Path, minimum_free_gib: float=10.0) -> dict:
    repo=Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git","status","--porcelain"],cwd=repo,text=True).strip(): raise RuntimeError("clean worktree required")
    if destination.exists(): raise RuntimeError(f"destination already exists: {destination}")
    rows=matrix()
    if len(rows)!=32 or len({r["case"] for r in rows})!=32: raise RuntimeError("dense launch matrix must contain 32 unique tasks")
    if any(r["fraction"] in {.55,.95} for r in rows): raise RuntimeError("0.55/0.95 must not be launched")
    resumes={r["case"]:inspect_resume(source_root,r) for r in rows if r["mode"]=="resumed"}
    family=validate_family(repo/"runtime_inputs/v10_2_30/qualification_family_manifest.json")
    free=q.free_gib(destination.parent.resolve())
    if free<minimum_free_gib: raise RuntimeError("minimum free-space preflight failed")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    cases=[]
    for row in rows:
        source=resumes.get(row["case"],{})
        cases.append({**row,**source,"output_directory":str((destination/row["case"]/"output").resolve()),
            "expected_final_analysis_inclusion":True})
    return {"schema":"v10.2.30_dense_deltaK_preflight_v1","launch_git_head":head,
        "qualified_simulation_head":q.QUALIFIED_SIMULATION_HEAD,"family":family,"family_hash":family["observed_sha256"],
        "source_qualification_root":str(source_root.resolve()),"source_0p95_root":str((repo/"runs/v10_2_30_four_class_developed_extension_from_7a5133f_cfbdee4_20260804").resolve()),
        "maximum_concurrency":2,"minimum_free_gib":minimum_free_gib,"available_free_gib":free,"cases":cases}


def prepare(source_root: Path,destination: Path,minimum_free_gib:float=10.0) -> dict:
    payload=preflight(source_root,destination,minimum_free_gib); destination.mkdir(parents=True)
    for row in payload["cases"]:
        case=destination/row["case"]
        if row["mode"]=="resumed":
            shutil.copytree(Path(row["source_case"]),case)
            (case/"output/exit_code.txt").unlink(missing_ok=True)
            q.atomic_json(case/SOURCE_NAME,{k:v for k,v in row.items() if k.startswith("source_") or k.startswith("starting_") or k.endswith("horizon") or k in {"stochastic","checks"}})
            q.set_status(case,"restartable",pid=None,restart_count=0,monotonic_horizon_extension=True)
        else:
            case.mkdir(); q.set_status(case,"pending",pid=None,restart_count=0)
    q.atomic_json(destination/"dense_deltaK_matrix.json",payload)
    return payload


def validate_staged(root:Path)->dict:
    payload=q.read_json(root/"dense_deltaK_matrix.json")
    if payload.get("schema")!="v10.2.30_dense_deltaK_preflight_v1" or len(payload.get("cases",[]))!=32: raise RuntimeError("invalid dense staging manifest")
    for row in payload["cases"]:
        case=root/row["case"]
        if row["mode"]=="resumed":
            source=q.read_json(case/SOURCE_NAME)
            if source.get("source_checkpoint_generation")!=q.read_json(case/"output/run_state_checkpoint.json").get("generation") or not q.checkpoint_valid(case,row):
                raise RuntimeError(f'{row["case"]}: staged resume changed')
        elif q.checkpoint_valid(case,row): raise RuntimeError(f'{row["case"]}: fresh case points to checkpoint')
    return payload


def classify(case: Path, row: dict | None = None) -> str:
    """Recognize both no-growth and after-growth physical horizon censors."""
    old=q.read_json(case/"qualification_status.json")
    if old.get("status") in q.TERMINAL: return old["status"]
    output=q.artifacts(case)
    if (output/"exit_code.txt").is_file():
        try: code=int((output/"exit_code.txt").read_text().strip())
        except ValueError: code=1
        summary=q.read_json(output/"developed_fatigue_growth_summary.json")
        control=q.read_json(output/"v10_2_30_fixed_deltaK_control.json")
        if code==0 and summary.get("target_reached") is True: return "completed"
        cycles=summary.get("cycles_consumed")
        if code==0 and ("right_censored" in str(control.get("censor_status","")) or
                        (cycles is not None and float(cycles)>=CYCLES_MAX)):
            return "censored"
    return BASE_CLASSIFY(case,row)


def run(root:Path,args)->int:
    validate_staged(root); q.matrix=matrix; q.classify=classify
    q.EXPECTED_MATRIX={(row["label"],row["fraction"]):row["deltaK_MPa_sqrt_m"] for row in matrix()}
    qargs=q.parser().parse_args(["run",str(root),"--max-jobs","2","--target-extension-um","100","--cycles-max","1e14",
        "--minimum-free-gib",str(args.minimum_free_gib),"--no-progress-seconds",str(args.no_progress_seconds),
        *( ["--recover-stale-lock"] if args.recover_stale_lock else [] )])
    return q.run(qargs)


def monitor(root:Path)->int:
    print(f"disk_free_GiB={q.free_gib(root):.2f}")
    for row in matrix():
        case=root/row["case"]; status=q.read_json(case/"qualification_status.json"); status.update(q.progress(case))
        print(row["case"],json.dumps({k:status.get(k) for k in ("status","pid","cycles_reached","event_count","crack_extension_um","current_mode","current_phase","latest_physical_progress_timestamp","latest_liveness_timestamp","restart_count")},sort_keys=True))
    return 0


def parser():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    for command in ("preflight","prepare"):
        c=sub.add_parser(command); c.add_argument("source",type=Path); c.add_argument("destination",type=Path); c.add_argument("--minimum-free-gib",type=float,default=10)
    r=sub.add_parser("run"); r.add_argument("root",type=Path); r.add_argument("--minimum-free-gib",type=float,default=10); r.add_argument("--no-progress-seconds",type=float,default=900); r.add_argument("--recover-stale-lock",action="store_true")
    for command in ("monitor","stop"): sub.add_parser(command).add_argument("root",type=Path)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if a.command=="preflight": print(json.dumps(preflight(a.source,a.destination,a.minimum_free_gib),indent=2,sort_keys=True)); return 0
    if a.command=="prepare": print(json.dumps(prepare(a.source,a.destination,a.minimum_free_gib),indent=2,sort_keys=True)); return 0
    if a.command=="run": return run(a.root.resolve(),a)
    if a.command=="monitor": return monitor(a.root.resolve())
    return q.stop_launcher(a.root.resolve())

if __name__=="__main__": raise SystemExit(main())
