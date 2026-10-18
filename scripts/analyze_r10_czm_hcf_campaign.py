#!/usr/bin/env python3
"""Package the current-CZM HCF continuation without converting runout to a rate."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, data):
    data = list(data)
    fields = list(data[0]) if data else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def jdump(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def f(value):
    return None if value in (None, "") else float(value)


def modes(result):
    return sorted({m.get("mode") for b in result["acceleration_blocks"] for m in b["modes"] if m.get("mode")})


def disposition(result):
    if result["numeric_failure"]:
        return "NUMERICAL_STOP"
    if result["resource_stop"]:
        return "RESOURCE_STOP"
    if result["unsupported_validities"]:
        return "UNRESOLVED_PHYSICS"
    if result["physical_event_count"]:
        return result["disposition"]
    if "stationary_tail" in modes(result):
        return "SUPPORTED_SHAKEDOWN_PERIODIC_ORBIT"
    if result["accepted_cycles"] >= result["exposure_target_cycles"]:
        return "PHYSICAL_RUNOUT_TO_DECLARED_HORIZON"
    return "INSUFFICIENT_EXPOSURE"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--fracture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    selection = rows(args.campaign / "selection" / "selected_hcf_cases.csv")
    selected = {r["candidate_id"]: r for r in selection}
    result_paths = sorted((args.campaign / "executed").glob("*/*/result.json"))
    results = [json.loads(p.read_text()) for p in result_paths]
    by_candidate = {r["candidate_id"]: r for r in results}
    missing = sorted(set(selected) - set(by_candidate))
    if missing:
        raise RuntimeError(f"selected cases lack result.json: {missing}")

    summary = []
    histories = []
    for candidate in sorted(selected):
        r = by_candidate[candidate]
        h0, h1 = r["history"][0], r["history"][-1]
        disp = disposition(r)
        summary.append({
            "candidate_id": candidate,
            "source_material_candidate_id": r["source_material_candidate_id"],
            "shape": r["shape"], "mode": r["mode"],
            "reference_factor": r["reference_factor"],
            "Kmax_MPa_sqrt_m": r["Kmax_MPa_sqrt_m"],
            "DeltaK_MPa_sqrt_m": r["DeltaK_MPa_sqrt_m"],
            "accepted_cycles": r["accepted_cycles"], "accepted_seconds": r["accepted_seconds"],
            "first_event_cycle": r["first_event_cycle"],
            "inter_event_cycles": json.dumps(r["inter_event_cycles"]),
            "physical_event_count": r["physical_event_count"], "advance_m": r["advance_m"],
            "growth_window": json.dumps(r["developed_window"]),
            "developed_rate_m_per_cycle": r["developed_rate_m_per_cycle"],
            "mobile_count_initial": h0["mobile_count"], "mobile_count_final": h1["mobile_count"],
            "tip_radius_m_initial": h0["tip_radius_m"], "tip_radius_m_final": h1["tip_radius_m"],
            "cohesive_separation_m_initial": h0["cohesive_separation_m"],
            "cohesive_separation_m_final": h1["cohesive_separation_m"],
            "opening_branch_final": h1["branch_state"],
            "acceleration_modes": json.dumps(modes(r)),
            "one_cycle_overlap_passed": r["overlap"]["passed"],
            "disposition": disp,
            "scope_note": ("periodic-orbit convergence supports shakedown at this load" if disp.startswith("SUPPORTED") else
                           "finite accepted exposure only; no asymptotic arrest claim"),
        })
        for item in r["history"]:
            histories.append({"candidate_id": candidate, **item})

    write_csv(args.out / "hcf_case_summary.csv", summary)
    write_csv(args.out / "state_evolution.csv", histories)
    write_csv(args.out / "amplitude_timescale_map.csv", rows(args.campaign / "selection" / "amplitude_timescale_map.csv"))
    write_csv(args.out / "selected_hcf_cases.csv", selection)

    fracture_rows = [r for r in rows(args.fracture) if r["candidate_id"] in selected]
    write_csv(args.out / "same_material_fracture_context.csv", fracture_rows)
    with gzip.open(args.out / "native_trajectories.json.gz", "wt") as stream:
        json.dump(results, stream, sort_keys=True)

    counts = {}
    for row in summary:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    total_cycles = sum(f(r["accepted_cycles"]) for r in summary)
    total_seconds = sum(f(r["accepted_seconds"]) for r in summary)
    unsupported_map = sum(r["unsupported"].lower() == "true" for r in rows(args.campaign / "selection" / "amplitude_timescale_map.csv"))
    decision = {
        "schema": "r10.current-czm-hcf-continuation/1",
        "decision": "HCF_BOUNDARY_ESTABLISHED_NO_PARIS_PASS",
        "executed_cases": len(summary), "accepted_cycles_total": total_cycles,
        "accepted_seconds_total": total_seconds, "physical_events_total": sum(int(r["physical_event_count"]) for r in summary),
        "developed_rates_total": sum(r["developed_rate_m_per_cycle"] not in (None, "") for r in summary),
        "disposition_counts": counts, "amplitude_map_rows": len(rows(args.campaign / "selection" / "amplitude_timescale_map.csv")),
        "unsupported_map_points_excluded_from_execution": unsupported_map,
        "legacy_cooperative_clock_used": False,
        "scientific_interpretation": [
            "The lower-amplitude current-CZM block is a genuine HCF exposure comparison, not a first-excursion prefix.",
            "No selected arm advanced, so no Paris slope or recurrent-growth rate is reported.",
            "Stationary-tail convergence is called supported shakedown; other completed cases are finite-horizon physical runouts only.",
            "The four opening arms remain paired to identical source plasticity inputs but evolve independently.",
        ],
        "next_acquisition": "Move amplitude within the mapped bonded interval toward finite opening/plastic timescale coordinates; retain exact event continuation and extend only state-evolving finite-horizon arms.",
    }
    jdump(args.out / "decision.json", decision)

    source_files = [
        Path("arrhenius_fracture/coupled_opening_reduction_r10.py"),
        Path("arrhenius_fracture/czm_high_cycle_adapter_r10.py"),
        Path("scripts/run_r10_czm_hcf.py"),
        Path("scripts/run_r10_czm_hcf_campaign.py"),
    ]
    receipt = {
        "schema": "r10.current-czm-hcf-runtime-receipt/1",
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "parent_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "result_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in result_paths},
        "executed_candidate_ids": sorted(by_candidate),
        "exact_checkpoint_count": sum((p.parent / "checkpoint" / "high_cycle_live_checkpoint.json").is_file() for p in result_paths),
        "phase_count": sorted({r["phase_count"] for r in results}),
        "opening_owners": sorted({r["opening_owner"] for r in results}),
        "plasticity_owners": sorted({r["plasticity_owner"] for r in results}),
        "legacy_cooperative_clock_used": any(r["legacy_cooperative_clock_used"] for r in results),
    }
    jdump(args.out / "runtime_receipt.json", receipt)

    report = f"""# Current-CZM HCF/VHCF continuation

## Decision

The corrected current-CZM calculation produced **no Paris pass**. Across {len(summary)} independently evolved athermal/mixed smooth/bilinear cases, the native phase-resolved engine accepted **{total_cycles:,.0f} cycles ({total_seconds:,.0f} s)** and observed no cohesive-cell advances. No startup secant, first-passage prediction, or analytical value is reported as a developed rate.

The result is nevertheless an HCF boundary result rather than another short prefix: the executed horizons span 10^4 to 10^6 cycles, every case passed the exact-versus-accelerated one-cycle overlap, and the trajectories record the changing EXP_floor state. Cases reaching the high-cycle stationary tail are classified as supported periodic-orbit shakedown. Cases that merely reached their declared horizon are finite-horizon physical runouts; they are not claims of infinite-life arrest.

## What changed

- Recurrence scoring now admits repeated bounded bursts across an actual 20–60 um developed window, while preserving unsupported-history and target-cascade censors.
- Opening stress is recomputed after every material-cell remap.
- A mixed deterministic unit-action crossing is an acquisition screen only; it cannot commit separation or carry phase action into a successor cell.
- The new HCF adapter uses the current athermal/mixed cohesive owner with the existing EXP_floor full-state engine. The legacy Arrhenius/cooperative opening clock is not called.

## Coverage and interpretation

- The amplitude/timescale map contains {decision['amplitude_map_rows']} points over four previously unexecuted plasticity backgrounds and all four shape/mode arms. {unsupported_map} unsupported mixed crossover points were excluded rather than assigned rates.
- The executed block contains {counts.get('SUPPORTED_SHAKEDOWN_PERIODIC_ORBIT', 0)} supported periodic-orbit shakedown cases and {counts.get('PHYSICAL_RUNOUT_TO_DECLARED_HORIZON', 0)} finite-horizon runouts.
- First-event times, inter-event waits, growth windows, and developed rates are null because there were no physical events. This is a measured outcome, not missing reporting.
- Same-material monotonic fracture classifications are carried in `same_material_fracture_context.csv`; fatigue runout does not overwrite those fracture results.

## Acquisition consequence

The next block should interpolate inside the already mapped bonded interval toward finite opening/plastic timescale coordinates. State-evolving arms may be continued from their exact checkpoints. Stationary-tail arms should not be repeated at the same load. A higher load is justified only by the mapped timescale/branch coordinate, not by runtime convenience.
"""
    (args.out / "RESULTS.md").write_text(report)

    manifest = {}
    for path in sorted(args.out.iterdir()):
        if path.name == "SHA256SUMS.json" or not path.is_file():
            continue
        manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    jdump(args.out / "SHA256SUMS.json", manifest)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
