#!/usr/bin/env python3
"""Run/resume, analyze, plot, and verify the R10 inverse-shape campaign."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.coupled_opening_reduction_r10 import score_recurrence


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_files(root: Path, name: str) -> list[Path]:
    return sorted(root.rglob(name)) if root.exists() else []


def registry_metadata(root: Path) -> dict[str, dict]:
    metadata = {}
    for path in sorted(root.rglob("projected_registry.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                identity = row.get("complete_row_sha256")
                if not identity:
                    continue
                metadata[identity] = {key: row.get(key) for key in (
                    "source_material_class", "cohesive_fold_thermal_prediction",
                    "thermal_prediction", "mechanism_prediction",
                    "predicted_hcf_cyclic_action_per_cycle",
                    "predicted_hcf_local_slope",
                    "predicted_hcf_accessible_phase_fraction")}
    return metadata


def hcf_rows(root: Path) -> list[dict]:
    rows = []
    metadata = registry_metadata(root)
    for path in json_files(root, "result.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("schema") != "r10.czm-owned-hcf-trajectory/1":
            continue
        evidence_state = "TERMINAL_RESULT"
        progress_path = path.with_name("progress.json")
        if progress_path.is_file():
            try:
                progress = json.loads(progress_path.read_text())
            except (OSError, json.JSONDecodeError):
                progress = {}
            if float(progress.get("accepted_cycles") or 0.) > float(
                    record.get("accepted_cycles") or 0.):
                record = {**record, **progress}
                frequency = float(record.get("frequency_Hz") or 1.)
                record["accepted_seconds"] = float(record["accepted_cycles"])/frequency
                events = record.get("events") or []
                scored = score_recurrence(events,
                    float(record.get("observation_path_m") or 120e-6),
                    bool(record.get("unsupported") or
                         record.get("unsupported_validities")))
                record["physical_event_count"] = len(events)
                record["advance_m"] = (float(events[-1]["extension_m"])
                                       if events else float(record.get("advance_m") or 0.))
                record["developed_rate_m_per_cycle"] = scored["developed_rate_m_per_cycle"]
                record["developed_window"] = scored["developed_window"]
                record["recurrence_outcome"] = scored["outcome"]
                record["disposition"] = "ACCEPTED_RESTART_IN_PROGRESS"
                evidence_state = "ACCEPTED_PROGRESS_NEWER_THAN_RESULT"
        identity = record.get("complete_row_sha256")
        rows.append({"path": str(path.relative_to(root)),
            "evidence_state": evidence_state,
            **metadata.get(identity, {}), **{key: record.get(key) for key in (
            "candidate_id", "complete_row_sha256", "shape", "mode", "temperature_K",
            "R", "frequency_Hz", "Kmax_MPa_sqrt_m", "DeltaK_MPa_sqrt_m",
            "accepted_cycles", "accepted_seconds", "physical_event_count", "advance_m",
            "first_event_cycle", "all_waiting_intervals_cycles", "inter_event_cycles",
            "developed_rate_m_per_cycle", "developed_window", "disposition",
            "recurrence_outcome", "numeric_failure", "resource_stop",
            "unsupported_validities", "restart_checkpoint", "plastic_feedback_reference")}})
    return rows


def developed_slope_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        rate = row.get("developed_rate_m_per_cycle")
        amplitude = row.get("DeltaK_MPa_sqrt_m")
        if rate in (None, "", 0, 0.) or amplitude in (None, ""):
            continue
        key = (row.get("candidate_id"), row.get("complete_row_sha256"),
               row.get("temperature_K"), row.get("R"), row.get("frequency_Hz"),
               row.get("plastic_feedback_reference"))
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in grouped.items():
        amplitudes: dict[float, list[dict]] = {}
        for row in group:
            amplitudes.setdefault(float(row["DeltaK_MPa_sqrt_m"]), []).append(row)
        points = []
        for amplitude, replicates in sorted(amplitudes.items()):
            rates = sorted(float(row["developed_rate_m_per_cycle"])
                           for row in replicates)
            points.append((amplitude, median(rates), len(rates)))
        for low, high in zip(points, points[1:]):
            output.append({
                "candidate_id": key[0], "complete_row_sha256": key[1],
                "temperature_K": key[2], "R": key[3], "frequency_Hz": key[4],
                "plastic_feedback_reference": key[5],
                "DeltaK_low_MPa_sqrt_m": low[0],
                "DeltaK_high_MPa_sqrt_m": high[0],
                "rate_low_m_per_cycle": low[1],
                "rate_high_m_per_cycle": high[1],
                "local_log_slope": math.log(high[1]/low[1])/math.log(high[0]/low[0]),
                "low_replicates": low[2], "high_replicates": high[2],
                "connected_rate_width_log10": math.log10(high[1]/low[1]),
            })
    return output


def native_rows(root: Path) -> list[dict]:
    rows = []
    for directory in sorted(root.glob("native*")):
        for path in directory.glob("*/*.json"):
            try:
                record = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record.get("cycle_history"), list):
                continue
            history = record["cycle_history"]
            rows.append({"path": str(path.relative_to(root)), **{key: record.get(key) for key in (
                "candidate_id", "same_complete_material_sha256", "paired_plasticity_sha256",
                "shape", "mode", "temperature_K",
                "Kmax_MPa_sqrt_m", "DeltaK_MPa_sqrt_m", "stop_category",
                "unsupported_crossover_encountered")}, "cycles_completed": len(history),
                "event_count": len(record.get("events") or []),
                "extension_m": history[-1].get("extension_m", 0.) if history else 0.})
    return rows


def csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(value, sort_keys=True) if isinstance(value, (list, dict))
                                   else value) for key, value in row.items()})


def analytical_frontier(root: Path) -> dict:
    banks = sorted(root.rglob("analytical_bank_32768.csv.gz"))
    if not banks:
        return {"bank": None, "valid": 0, "strict_passes": 0, "best": None}
    bank = next((path for path in banks if "round_4_targeted_tail" in str(path)), banks[-1])
    valid = 0
    strict = 0
    best = None
    with gzip.open(bank, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("domain_pass") != "True":
                continue
            valid += 1
            strict += row.get("strict_internal_reference_pass") == "True"
            score = float(row.get("moderate_connected_width_log10rate") or 0.)
            if best is None or score > best[0]:
                best = (score, row)
    keep = None if best is None else {key: best[1].get(key) for key in (
        "candidate_id", "cohesive_shape", "source_material_candidate_id",
        "mechanism_prediction", "thermal_prediction", "moderate_connected_width_log10K",
        "moderate_connected_width_log10rate", "strict_connected_width_log10rate",
        "hcf_slope_median", "hcf_slope_min", "metastable_load_ratio_300K",
        "completion_cutoff_m", "initial_stiffness_Pa_per_m", "Gamma_ref_J_m2",
        "t_hat_ref_Pa", "activation_area_m2", "zero_traction_barrier_eV_300K",
        "evaluated_cohesive_json", "curve_json")}
    return {"bank": str(bank.relative_to(root)), "bank_sha256": sha256(bank),
            "valid": valid, "strict_passes": strict, "best": keep}


def cyclic_inverse_frontier(root: Path) -> dict:
    tables=sorted(root.rglob("cyclic_inverse_starts.csv"))
    executions=[];best=None
    for path in tables:
        valid=0;strict=0;local_best=None
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("domain_pass") != "True":continue
                valid+=1;strict+=row.get("cyclic_strict_internal_reference_pass")=="True"
                score=float(row.get("objective") or math.inf)
                if local_best is None or score<local_best[0]:local_best=(score,row)
                if best is None or score<best[0]:best=(score,row,path)
        executions.append({"table":str(path.relative_to(root)),"valid":valid,
                           "strict_passes":strict})
    keep=None
    if best is not None:
        keep={key:best[1].get(key) for key in (
            "candidate_id","source_material_candidate_id","mechanism_prediction",
            "thermal_prediction","objective","cyclic_target_K_MPa_sqrt_m",
            "cyclic_target_action_per_cycle","cyclic_target_local_slope",
            "cyclic_moderate_connected_width_log10K",
            "cyclic_moderate_connected_width_log10rate",
            "cyclic_strict_connected_width_log10rate",
            "optimized_cohesive_json")}
        keep["table"]=str(best[2].relative_to(root))
    return {"executions":executions,"best":keep,
            "strict_passes":sum(row["strict_passes"] for row in executions)}


def analyze(args) -> int:
    out = args.root / "campaign_delivery"
    hcf = hcf_rows(args.root)
    native = native_rows(args.root)
    frontier = analytical_frontier(args.root)
    fractures = []
    for path in sorted(args.root.rglob("ramp_dispositions.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                fractures.append({"path": str(path.relative_to(args.root)), **row})
    two_d = []
    for path in sorted(args.root.rglob("qualification.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("schema") == "r10.opening-alternatives-2d-transfer/1":
            two_d.append({"path": str(path.relative_to(args.root)),
                "candidate_id": record.get("candidate_id"), "scope": record.get("scope"),
                "same_complete_material_sha256": record.get("same_complete_material_sha256"),
                "cases": len(record.get("cases") or []),
                "triggers": [case.get("trigger") for case in record.get("cases") or []]})
    csv_write(out / "hcf_results.csv", hcf)
    csv_write(out / "native_developed_slopes.csv", developed_slope_rows(hcf))
    csv_write(out / "native_results.csv", native)
    csv_write(out / "fracture_results.csv", fractures)
    atomic(out / "two_d_results.json", two_d)
    restarts=[]
    for row in hcf:
        checkpoint=row.get("restart_checkpoint")
        if not checkpoint:continue
        restarts.append({key:row.get(key) for key in (
            "candidate_id","complete_row_sha256","accepted_cycles",
            "accepted_seconds","physical_event_count","advance_m","disposition",
            "resource_stop","restart_checkpoint")})
        restarts[-1]["checkpoint_exists"]=(ROOT/checkpoint).exists()
        restarts[-1]["result_path"]=row["path"]
    csv_write(out/"restart_inventory.csv",restarts)
    analytical_ids=set()
    for bank in args.root.rglob("analytical_bank_32768.csv.gz"):
        with gzip.open(bank,"rt",newline="") as stream:
            analytical_ids.update(row.get("candidate_id") for row in csv.DictReader(stream))
    summary = {"schema": "r10.finite-compliance-campaign-analysis/1",
        "analytical_frontier": frontier,
        "cyclic_inverse_frontier":cyclic_inverse_frontier(args.root),
        "coverage": {"analytical_allocated": sum(json.loads(path.read_text()).get("allocated_vectors", 0)
                for path in args.root.rglob("analytical_summary.json")),
            "analytical_bank_executions":len(list(args.root.rglob("analytical_bank_32768.csv.gz"))),
            "analytical_unique_candidate_labels":len(analytical_ids),
            "state_informed_results": sum(1 for path in args.root.glob("state_informed*/*/*.json")),
            "native_full_state_results": len(native),
            "native_unique_complete_materials": len({row.get("same_complete_material_sha256") for row in native
                                                       if row.get("same_complete_material_sha256")}),
            "hcf_results": len(hcf),
            "hcf_unique_complete_materials": len({row.get("complete_row_sha256") for row in hcf
                                                   if row.get("complete_row_sha256")}),
            "hcf_accepted_cycles": sum(float(row.get("accepted_cycles") or 0.) for row in hcf),
            "hcf_accepted_seconds": sum(float(row.get("accepted_seconds") or 0.) for row in hcf),
            "hcf_events": sum(int(row.get("physical_event_count") or 0) for row in hcf),
            "fracture_rows": len(fractures), "two_d_receipts": len(two_d),
            "accepted_restart_records":len(restarts)},
        "hcf_dispositions": {}, "native_stop_categories": {}}
    for row in hcf:
        key = str(row.get("disposition")); summary["hcf_dispositions"][key] = summary["hcf_dispositions"].get(key, 0) + 1
    for row in native:
        key = str(row.get("stop_category")); summary["native_stop_categories"][key] = summary["native_stop_categories"].get(key, 0) + 1
    atomic(out / "campaign_summary.json", summary)
    print(json.dumps(summary, indent=2)); return 0


def verify(args) -> int:
    failures = []
    summaries = []
    for path in sorted(args.root.rglob("analytical_summary.json")):
        record = json.loads(path.read_text()); summaries.append(str(path.relative_to(args.root)))
        if record.get("algebraic_tail_registered") is not False:
            failures.append(f"algebraic diagnostic registration flag invalid: {path}")
    hcf = hcf_rows(args.root)
    for row in hcf:
        checkpoint = row.get("restart_checkpoint")
        if checkpoint and not (ROOT / checkpoint).exists():
            failures.append(f"missing restart checkpoint: {checkpoint}")
        if row.get("numeric_failure"):
            failures.append(f"numeric failure in {row['path']}: {row['numeric_failure']}")
    payload = {"schema": "r10.finite-compliance-campaign-verification/1",
        "passed": not failures, "failures": failures,
        "analytical_summaries_checked": summaries, "hcf_results_checked": len(hcf),
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                text=True).strip()}
    atomic(args.root / "campaign_delivery" / "verification.json", payload)
    print(json.dumps(payload, indent=2)); return 0 if not failures else 1


def plot(args) -> int:
    import matplotlib.pyplot as plt
    import numpy as np
    from arrhenius_fracture.cohesive_arrhenius import (
        BILINEAR, COMPACT_INDEPENDENT, SMOOTH_EXPONENTIAL,
        cohesive_shape_response,
    )
    from arrhenius_fracture.finite_compliance_inverse_shape_r10 import (
        connected_barrier, imposed_from_K, shape_coordinates,
    )
    from scripts.run_r10_finite_compliance_inverse_shape import _thermal
    from scripts.run_r10_opening_alternatives import mechanics

    out=args.root/"campaign_delivery"/"figures";out.mkdir(parents=True,exist_ok=True)
    frontier=analytical_frontier(args.root);best=frontier.get("best")
    if best:
        curve=json.loads(best["curve_json"]);K=np.asarray([p["K_MPa_sqrt_m"] for p in curve])
        rate=np.asarray([p["events_per_cycle_peak_rate"] for p in curve])
        slope=np.asarray([np.nan if p["local_log_slope"] is None else p["local_log_slope"] for p in curve])
        fig,axes=plt.subplots(2,1,figsize=(6.2,6.2),sharex=True)
        axes[0].semilogy(K,rate,"o-",color="#1f4e79",ms=3,lw=1.4)
        axes[0].set_ylabel("Peak opening action (cycle$^{-1}$)")
        axes[1].plot(K,slope,"o-",color="#a23b2a",ms=3,lw=1.4)
        axes[1].axhspan(3.4,4.6,color="#4c956c",alpha=.18)
        axes[1].set_xlabel(r"$K_{\max}$ (MPa$\sqrt{\mathrm{m}}$)")
        axes[1].set_ylabel(r"Local $d\ln v/d\ln K$")
        fig.tight_layout();fig.savefig(out/"analytical_rate_slope.png",dpi=240);fig.savefig(out/"analytical_rate_slope.pdf");plt.close(fig)

        cohesive=json.loads(best["evaluated_cohesive_json"]);thermal=_thermal(cohesive)
        fig,axes=plt.subplots(2,1,figsize=(6.2,6.2),sharex=True)
        colors={SMOOTH_EXPONENTIAL:"#1f77b4",BILINEAR:"#ff7f0e",COMPACT_INDEPENDENT:"#2ca02c"}
        labels={SMOOTH_EXPONENTIAL:"Smooth exponential",BILINEAR:"Bilinear",COMPACT_INDEPENDENT:"Compact research"}
        maximum=0.
        data=[]
        for shape in (SMOOTH_EXPONENTIAL,BILINEAR,COMPACT_INDEPENDENT):
            coords=shape_coordinates(shape,thermal,300.);maximum=max(maximum,coords["completion_cutoff_m"])
            cutoff=float(coords["completion_cutoff_m"]);peak=float(coords["peak_separation_m"])
            d=np.unique(np.r_[0.,peak,np.geomspace(max(cutoff*1e-9,1e-20),cutoff,1400)])
            traction,_,potential=cohesive_shape_response(shape,d,thermal,300.)
            data.append((shape,d,np.asarray(traction),np.asarray(potential)))
        for shape,d,traction,potential in data:
            axes[0].plot(d*1e9,traction/1e9,color=colors[shape],lw=1.5,label=labels[shape])
            axes[1].plot(d*1e9,potential,color=colors[shape],lw=1.5)
        axes[0].set_ylabel("Traction (GPa)");axes[0].legend(frameon=False)
        axes[1].set_xlabel("Separation (nm)");axes[1].set_ylabel(r"Potential (J m$^{-2}$)")
        fig.tight_layout();fig.savefig(out/"potential_traction.png",dpi=240);fig.savefig(out/"potential_traction.pdf");plt.close(fig)

        _,kenv=mechanics(thermal,300.);shape=best["cohesive_shape"]
        sample=[float(curve[i]["K_MPa_sqrt_m"]) for i in (0,len(curve)//3,2*len(curve)//3)]
        prepared=[];largest_root=0.
        for index,Kvalue in enumerate(sample):
            imposed=imposed_from_K(Kvalue,kenv)
            points=connected_barrier(imposed,kenv,thermal,300.,shape)["stationary"]
            largest_root=max(largest_root,max((float(point["delta_m"]) for point in points),default=0.))
            prepared.append((index,Kvalue,imposed,points))
        dmax=max(2.*largest_root,5.*shape_coordinates(shape,thermal,300.)["peak_separation_m"])
        d=np.unique(np.r_[0.,np.geomspace(max(dmax*1e-8,1e-20),dmax,1600)])
        fig,ax=plt.subplots(figsize=(6.2,4.4))
        for index,Kvalue,imposed,points in prepared:
            _,_,potential=cohesive_shape_response(shape,d,thermal,300.)
            landscape=np.asarray(potential)+.5*kenv*(imposed-d)**2
            landscape=landscape-float(np.min(landscape))
            ax.plot(d*1e9,landscape,label=fr"{Kvalue:.1f} MPa$\sqrt{{\mathrm{{m}}}}$",lw=1.4)
            for point in points:
                marker="o" if point["kind"]=="MINIMUM" else "x"
                raw_min=float(np.min(np.asarray(potential)+.5*kenv*(imposed-d)**2))
                ax.plot(float(point["delta_m"])*1e9,float(point["energy_J_m2"])-raw_min,
                        marker,color=f"C{index}",ms=5)
        ax.set_xlabel("Separation (nm)");ax.set_ylabel(r"Loaded potential (J m$^{-2}$)")
        ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/"finite_compliance_landscape.png",dpi=240);fig.savefig(out/"finite_compliance_landscape.pdf");plt.close(fig)

    records=[row for row in hcf_rows(args.root) if row.get("candidate_id")=="R10_INV_21321__SM" and
             float(row.get("temperature_K") or 0.)==300. and float(row.get("R") or 0.)==.1 and
             float(row.get("frequency_Hz") or 0.)==1000. and
             row.get("plastic_feedback_reference")=="EVOLVING_FULL_STATE"]
    if records:
        byK={}
        for row in records:
            K=float(row["Kmax_MPa_sqrt_m"]);byK[K]=row
        fig,ax=plt.subplots(figsize=(6.2,4.4))
        floor=1e-14
        for K,row in sorted(byK.items()):
            rate=row.get("developed_rate_m_per_cycle");disposition=row.get("disposition")
            if rate is not None:
                ax.semilogy(K,float(rate),"o",color="#a23b2a")
            elif disposition=="PHYSICAL_RUNOUT_TO_DECLARED_HORIZON":
                ax.semilogy(K,floor,"v",mfc="none",mec="#1f4e79")
            elif disposition in {"TARGET_TRUNCATED_CASCADE","UNRESOLVED_OPENING_PHYSICS"}:
                ax.semilogy(K,1e-4,"^",mfc="none",mec="#6a3d9a")
            else:
                ax.semilogy(K,max(float(row.get("advance_m") or 0.)/max(float(row.get("accepted_cycles") or 1.),1.),floor),"s",mfc="none",mec="#444444")
        ax.set_xlabel(r"$K_{\max}$ (MPa$\sqrt{\mathrm{m}}$)")
        ax.set_ylabel(r"Developed $da/dN$ (m cycle$^{-1}$)")
        fig.tight_layout();fig.savefig(out/"native_amplitude_outcomes.png",dpi=240);fig.savefig(out/"native_amplitude_outcomes.pdf");plt.close(fig)

    print(json.dumps({"figure_directory":str(out),"files":sorted(p.name for p in out.iterdir())},indent=2));return 0


def execute_plan(args) -> int:
    plan = json.loads(args.plan.read_text())
    receipts = []
    for task in plan["tasks"]:
        sentinel = ROOT / task["sentinel"]
        if args.resume and sentinel.exists():
            receipts.append({"id": task["id"], "status": "SKIPPED_COMPLETE",
                             "sentinel": task["sentinel"]}); continue
        log = args.root / "campaign_delivery" / "logs" / f"{task['id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            result = subprocess.run(task["argv"], cwd=ROOT, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        receipts.append({"id": task["id"], "status": "COMPLETE" if result.returncode == 0 else "FAILED",
            "returncode": result.returncode, "sentinel": task["sentinel"],
            "sentinel_exists": sentinel.exists(), "log": str(log.relative_to(args.root))})
        atomic(args.root / "campaign_delivery" / "plan_receipts.json", receipts)
        if result.returncode and not args.keep_going:
            break
    print(json.dumps(receipts, indent=2)); return 0 if all(r["status"] != "FAILED" for r in receipts) else 1


def status(args) -> int:
    payload = {"analytical_summaries": len(json_files(args.root, "analytical_summary.json")),
        "state_results": sum(1 for _ in args.root.glob("state_informed*/*/*.json")),
        "native_results": len(native_rows(args.root)), "hcf_results": len(hcf_rows(args.root)),
        "hcf_progress_files": len(json_files(args.root, "progress.json")),
        "fracture_results": sum(1 for _ in args.root.rglob("ramp_dispositions.csv")),
        "two_d_receipts": sum(1 for _ in args.root.rglob("qualification.json"))}
    print(json.dumps(payload, indent=2)); return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
        default=Path("runs/r10_inverse_shape_12h"))
    sub = parser.add_subparsers(dest="command", required=True)
    for name, function in (("status", status), ("analyze", analyze),
                           ("verify", verify), ("plot", plot)):
        child = sub.add_parser(name); child.set_defaults(function=function)
    child = sub.add_parser("run")
    child.add_argument("--plan", type=Path, required=True)
    child.add_argument("--resume", action="store_true")
    child.add_argument("--keep-going", action="store_true")
    child.set_defaults(function=execute_plan)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
