#!/usr/bin/env python3
"""Rebuild campaign coverage and scientific tables from accepted artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.coupled_opening_reduction_r10 import score_recurrence


def read_json(path: Path):
    return json.loads(path.read_text())


def write_csv(path: Path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_results(campaign: Path):
    native = []
    hcf = []
    qualifications = []
    for path in campaign.rglob("result.json"):
        try:
            result = read_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if result.get("schema") == "r10.czm-owned-hcf-trajectory/1":
            if "eventful_adapter_qualification" in path.parts:
                qualifications.append((path, result))
            else:
                progress_path = path.with_name("progress.json")
                if progress_path.is_file():
                    progress = read_json(progress_path)
                    if float(progress.get("accepted_cycles", 0.)) > float(result.get("accepted_cycles", 0.)):
                        result = {**result, **progress, "live_checkpoint_ahead": True}
                        result["acceleration_blocks"] = result.get(
                            "blocks", result.get("acceleration_blocks", []))
                        result["accepted_seconds"] = (float(result["accepted_cycles"]) /
                                                      float(result["frequency_Hz"]))
                        scored = score_recurrence(
                            result.get("events", []),
                            float(result.get("observation_path_m", 120e-6)),
                            bool(result.get("unsupported_validities", [])))
                        result["physical_event_count"] = len(result.get("events", []))
                        result["advance_m"] = (result["events"][-1].get("extension_m", 0.)
                                               if result.get("events") else 0.)
                        result["developed_rate_m_per_cycle"] = scored["developed_rate_m_per_cycle"]
                        result["developed_window"] = scored["developed_window"]
                hcf.append((path, result))
        elif "events" in result and "cycle_history" in result:
            native.append((path, result))
    return native, hcf, qualifications


def native_rows(items):
    rows = []
    for path, result in items:
        events = result.get("events", [])
        scored = score_recurrence(
            events, float(result.get("cohesive_path_observation_limit_m", 120e-6)),
            bool(result.get("unsupported_crossover_encountered", False)))
        history = result.get("cycle_history", [])
        rows.append({
            "candidate_id": result.get("candidate_id"),
            "source_material_candidate_id": result.get("source_material_candidate_id"),
            "shape": result.get("shape"),
            "mode": result.get("mode"),
            "temperature_K": result.get("temperature_K"),
            "R": result.get("R", .1),
            "frequency_Hz": result.get("frequency_Hz", 1000.),
            "reference_factor": result.get("reference_factor"),
            "Kmax_MPa_sqrt_m": result.get("Kmax_MPa_sqrt_m"),
            "DeltaK_MPa_sqrt_m": result.get("DeltaK_MPa_sqrt_m"),
            "accepted_cycles": len(history),
            "accepted_seconds": history[-1].get("physical_time_s") if history else 0.,
            "events": len(events),
            "advance_m": events[-1].get("extension_m", 0.) if events else 0.,
            "disposition": result.get("stop_category"),
            "recurrence_outcome": scored["outcome"],
            "developed_rate_m_per_cycle": scored["developed_rate_m_per_cycle"],
            "developed_window": json.dumps(scored["developed_window"], sort_keys=True),
            "unsupported": result.get("unsupported_crossover_encountered", False),
            "result_path": str(path),
        })
    return rows


def hcf_rows(items):
    rows = []
    for path, result in items:
        rows.append({
            "candidate_id": result.get("candidate_id"),
            "source_material_candidate_id": result.get("source_material_candidate_id"),
            "shape": result.get("shape"),
            "mode": result.get("mode"),
            "temperature_K": result.get("temperature_K"),
            "R": result.get("R"),
            "frequency_Hz": result.get("frequency_Hz"),
            "reference_factor": result.get("reference_factor"),
            "Kmax_MPa_sqrt_m": result.get("Kmax_MPa_sqrt_m"),
            "DeltaK_MPa_sqrt_m": result.get("DeltaK_MPa_sqrt_m"),
            "accepted_cycles": result.get("accepted_cycles", 0.),
            "accepted_seconds": result.get("accepted_seconds", 0.),
            "events": result.get("physical_event_count", 0),
            "advance_m": result.get("advance_m", 0.),
            "disposition": result.get("disposition"),
            "recurrence_outcome": result.get("recurrence_outcome"),
            "developed_rate_m_per_cycle": result.get("developed_rate_m_per_cycle"),
            "developed_window": json.dumps(result.get("developed_window"), sort_keys=True),
            "plastic_feedback_reference": result.get("plastic_feedback_reference", "EVOLVING_FULL_STATE"),
            "resource_stop": result.get("resource_stop", False),
            "live_checkpoint_ahead": result.get("live_checkpoint_ahead", False),
            "cpu_seconds": result.get("cpu_seconds"),
            "wall_seconds": result.get("wall_seconds"),
            "acceleration_blocks": len(result.get("acceleration_blocks", [])),
            "numeric_failure": result.get("numeric_failure"),
            "result_path": str(path),
        })
    return rows


def thermal_rows(campaign: Path):
    rows = []
    for path in campaign.glob("thermal_ramps_*/ramp_dispositions.csv"):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                rows.append({**row, "source_path": str(path)})
    return rows


def finalist_rows(campaign: Path, source_ids: set[str]):
    found = {}
    registry_paths = list(campaign.rglob("*registry.csv"))
    for path in registry_paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                source = row.get("source_material_candidate_id")
                if source in source_ids:
                    key = (source, row.get("candidate_id"))
                    found.setdefault(key, {**row, "registry_path": str(path)})
    return [found[key] for key in sorted(found)]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--finalist-source-ids", default="")
    args = parser.parse_args(argv)
    native_items, hcf_items, qualification_items = load_results(args.campaign)
    native = native_rows(native_items)
    hcf = hcf_rows(hcf_items)
    thermal = thermal_rows(args.campaign)
    developed = [r for r in native + hcf if r["developed_rate_m_per_cycle"] not in (None, "", 0, 0.)]
    state_sources = set()
    state_rows = 0
    for path in args.campaign.rglob("*state_map*.csv"):
        with path.open(newline="") as stream:
            mapped = list(csv.DictReader(stream))
        state_rows += len(mapped)
        state_sources.update(r.get("source_material_candidate_id") for r in mapped)
    native_sources = {r["source_material_candidate_id"] for r in native if r["source_material_candidate_id"]}
    protocols = {(r["temperature_K"], r["R"], r["frequency_Hz"]) for r in hcf}
    cheap = read_json(args.campaign / "acquisition_round_0_1024" / "analytical_summary.json")
    coverage = {
        "schema": "r10.12h-campaign-analysis/1",
        "cheap_allocated_materials": cheap["allocated_materials"],
        "cheap_domain_valid_materials": cheap["domain_valid_materials"],
        "cheap_opening_arms": cheap["evaluated_opening_arms"],
        "state_map_records": state_rows,
        "state_informed_physical_vectors": len(state_sources),
        "native_result_files": len(native),
        "native_physical_vectors": len(native_sources),
        "native_execution_identities": len({r["candidate_id"] for r in native}),
        "native_eventful_cases": sum(r["events"] > 0 for r in native),
        "native_event_commits": sum(int(r["events"] or 0) for r in native),
        "native_accepted_cycle_records": sum(float(r["accepted_cycles"] or 0.) for r in native),
        "native_accepted_seconds": sum(float(r["accepted_seconds"] or 0.) for r in native),
        "native_unsupported_cases": sum(bool(r["unsupported"]) for r in native),
        "hcf_result_files": len(hcf),
        "adapter_qualification_result_files": len(qualification_items),
        "hcf_physical_vectors": len({r["source_material_candidate_id"] for r in hcf}),
        "hcf_total_accepted_cycles": sum(float(r["accepted_cycles"] or 0.) for r in hcf),
        "hcf_total_accepted_seconds": sum(float(r["accepted_seconds"] or 0.) for r in hcf),
        "hcf_eventful_cases": sum(int(r["events"] or 0) > 0 for r in hcf),
        "hcf_event_commits": sum(int(r["events"] or 0) for r in hcf),
        "hcf_acceleration_blocks": sum(int(r["acceleration_blocks"] or 0) for r in hcf),
        "hcf_reported_cpu_seconds": sum(float(r["cpu_seconds"] or 0.) for r in hcf),
        "hcf_cpu_measurement_files": sum(r["cpu_seconds"] not in (None, "") for r in hcf),
        "hcf_worker_wall_seconds": sum(float(r["wall_seconds"] or 0.) for r in hcf),
        "hcf_protocols": len(protocols),
        "developed_rate_points": len(developed),
        "thermal_fracture_rows": len(thermal),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    write_csv(args.out / "native_dispositions.csv", native)
    write_csv(args.out / "hcf_dispositions.csv", hcf)
    write_csv(args.out / "developed_fatigue.csv", developed)
    write_csv(args.out / "thermal_fracture.csv", thermal)
    wanted = {x.strip() for x in args.finalist_source_ids.split(",") if x.strip()}
    if wanted:
        write_csv(args.out / "finalist_parameterizations.csv",
                  finalist_rows(args.campaign, wanted))
    print(json.dumps(coverage, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
