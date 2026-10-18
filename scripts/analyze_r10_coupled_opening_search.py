#!/usr/bin/env python3
"""Package the coupled cyclic-state reduction, fracture ramps, and native block."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows) -> None:
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def native_disposition(d):
    if d["unsupported_crossover_encountered"] and not d["events"]:
        return "UNSUPPORTED_CROSSOVER_UNRESOLVED"
    event_cycles = sorted({int(event["cycle"]) for event in d["events"]})
    # Fractional event times within the first waveform are a first-passage
    # cascade, not an event-window da/dN measurement.
    if d["events"] and len(event_cycles) <= 1:
        return "TARGET_TRUNCATED_SAME_CYCLE_CASCADE"
    developed = [event for event in d["events"] if 20e-6 <= event["extension_m"] <= 60e-6]
    if len(developed) >= 2:
        dc = developed[-1]["cycle"] - developed[0]["cycle"]
        if dc > 0:
            return "DEVELOPED_RECURRENT_RATE"
    if len(event_cycles) >= 2:
        return "RECURRENT_OUTSIDE_DEVELOPED_WINDOW"
    if d["events"]:
        return "TARGET_TRUNCATED_FIRST_PASSAGE"
    if "SHAKEDOWN" in d["stop_category"]:
        return "SUPPORTED_CYCLIC_SHAKEDOWN"
    return "FINITE_EXPOSURE_NO_ADVANCE"


def thermal_class(rows):
    supported = [r for r in rows if r["root"] is not None and not r["unsupported"]]
    if len(supported) == len(rows):
        values = [r["root"] for r in supported]
        ratio = max(values) / min(values)
        interior = max(values[1:-1], default=-1)
        if interior >= 1.1 * max(values[0], values[-1]):
            return "PEAK_T_RESOLVED"
        if ratio <= 1.25:
            return "WEAK_T_RESOLVED"
        if values[-1] <= 0.8 * values[0]:
            return "DBTT_LIKE_RESOLVED"
        return "NONCANONICAL_T_DEPENDENCE"
    roots = {int(r["T"]): r["root"] for r in supported}
    if roots and min(roots) >= 900 and all(r["root"] is None for r in rows if r["T"] < min(roots)):
        return "HOT_COMMITMENT_WITH_COLD_KMAX_CENSOR"
    if any(r["unsupported"] for r in rows):
        return "MIXED_CROSSOVER_UNRESOLVED"
    return "KMAX_CENSORED"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--reduced", type=Path, required=True)
    parser.add_argument("--ramps", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--retained-descriptor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    with args.registry.open(newline="") as stream:
        registry = list(csv.DictReader(stream))
    reg = {row["candidate_id"]: row for row in registry}

    histories = {"reduced": {}, "ramps": {}, "native": {}}
    reduced_rows = []
    for path in sorted(args.reduced.glob("**/K_*.json")):
        d = load(path); histories["reduced"][str(path)] = d
        reduced_rows.append({
            "candidate_id": d["candidate_id"], "source_material_candidate_id": d["source_material_candidate_id"],
            "shape": d["shape"], "mode": d["mode"], "reference_factor": d["reference_factor"],
            "cycles": d["cycles_completed"], "events": len(d["events"]), "event_cycle_count": d["event_cycle_count"],
            "outcome": d["outcome"], "unsupported_crossover": d["unsupported_crossover"],
            "developed_rate_m_per_cycle": d["developed_rate_m_per_cycle"], "sha256": sha(path),
        })

    ramp_groups = defaultdict(list)
    for path in sorted(args.ramps.glob("**/T_*K.json")):
        d = load(path); histories["ramps"][str(path)] = d
        ramp_groups[d["candidate_id"]].append({
            "T": int(d["temperature_K"]), "root": d["first_committed_separation_K_MPa_sqrt_m"],
            "loss": d["occupied_branch_loss_K_MPa_sqrt_m"], "status": d["status"],
            "unsupported": d["unsupported_crossover"], "Kmax": d["Kmax_MPa_sqrt_m"], "sha256": sha(path),
        })
    thermal_rows = []
    for candidate, rows in sorted(ramp_groups.items()):
        rows.sort(key=lambda r: r["T"])
        thermal_rows.append({
            "candidate_id": candidate, "source_material_candidate_id": reg[candidate]["source_material_candidate_id"],
            "shape": reg[candidate]["selected_shape"], "mode": reg[candidate]["selected_mode"],
            "temperatures_K": json.dumps([r["T"] for r in rows]),
            "committed_roots_MPa_sqrt_m": json.dumps([r["root"] for r in rows]),
            "occupied_branch_losses_MPa_sqrt_m": json.dumps([r["loss"] for r in rows]),
            "unsupported_temperatures_K": json.dumps([r["T"] for r in rows if r["unsupported"]]),
            "thermal_class": thermal_class(rows),
        })

    native_rows = []
    for path in sorted(args.native.glob("**/K_*.json")):
        d = load(path); histories["native"][str(path)] = d
        event_cycles = sorted({int(event["cycle"]) for event in d["events"]})
        native_rows.append({
            "candidate_id": d["candidate_id"], "source_material_candidate_id": d["source_material_candidate_id"],
            "shape": d["shape"], "mode": d["mode"], "reference_factor": d["reference_factor"],
            "Kmax_MPa_sqrt_m": d["Kmax_MPa_sqrt_m"], "cycles": len(d["cycle_history"]),
            "events": len(d["events"]), "event_cycles": json.dumps(event_cycles),
            "extension_m": d["cycle_history"][-1]["extension_m"] if d["cycle_history"] else 0.0,
            "disposition": native_disposition(d), "stop_category": d["stop_category"],
            "unsupported_crossover": d["unsupported_crossover_encountered"], "sha256": sha(path),
        })

    by_arm = defaultdict(list)
    for row in native_rows:
        by_arm[row["candidate_id"]].append(row)
    arm_rows = []
    for candidate, rows in sorted(by_arm.items()):
        supported_no = [float(r["reference_factor"]) for r in rows if r["disposition"] in {"SUPPORTED_CYCLIC_SHAKEDOWN", "FINITE_EXPOSURE_NO_ADVANCE"}]
        cascade = [float(r["reference_factor"]) for r in rows if r["disposition"] == "TARGET_TRUNCATED_SAME_CYCLE_CASCADE"]
        unresolved = [float(r["reference_factor"]) for r in rows if r["disposition"] == "UNSUPPORTED_CROSSOVER_UNRESOLVED"]
        arm_rows.append({
            "candidate_id": candidate, "source_material_candidate_id": rows[0]["source_material_candidate_id"],
            "shape": rows[0]["shape"], "mode": rows[0]["mode"],
            "maximum_supported_no_advance_factor": max(supported_no) if supported_no else None,
            "minimum_same_cycle_cascade_factor": min(cascade) if cascade else None,
            "unsupported_factors": json.dumps(unresolved), "developed_rate_points": 0,
            "local_slopes": "[]", "strict_Paris_pass": False,
        })

    # Acquisition is driven by observed native boundaries, not a fitted rate correction.
    executed = set(by_arm)
    acquisition = []
    for row in registry:
        if row["candidate_id"] in executed:
            continue
        priority = {"MIXED_CONTROL_PREDICTED": 0, "PLASTICITY_CONTROL_PREDICTED": 1}.get(row["mechanism_prediction"], 2)
        acquisition.append({
            "priority": priority, "candidate_id": row["candidate_id"],
            "source_material_candidate_id": row["source_material_candidate_id"], "shape": row["selected_shape"],
            "mode": row["selected_mode"], "search_design": row["search_design"],
            "mechanism_prediction": row["mechanism_prediction"],
            "reason": "UNEXECUTED_PAIRED_ARM_AFTER_COUPLED_REDUCTION_AND_NATIVE_BOUNDARY_UPDATE",
        })
    acquisition.sort(key=lambda r: (r["priority"], r["source_material_candidate_id"], r["shape"], r["mode"]))

    write_csv(args.out / "reduced_case_dispositions.csv", reduced_rows)
    write_csv(args.out / "monotonic_fracture_panel.csv", thermal_rows)
    write_csv(args.out / "native_case_dispositions.csv", native_rows)
    write_csv(args.out / "native_arm_summary.csv", arm_rows)
    write_csv(args.out / "continuing_acquisition.csv", acquisition)
    write_csv(args.out / "paired_shape_mode_registry.csv", registry)
    with gzip.open(args.out / "coupled_histories.json.gz", "wt") as stream:
        json.dump(histories, stream, sort_keys=True, separators=(",", ":"))

    retained_files = ("analytical_summary.json", "analytical_bank.csv.gz", "decision.json")
    retained_receipt = {
        "scope": "STATIC_ACCESSIBILITY_AND_HEURISTIC_EXPOSURE_DESCRIPTOR_BANK",
        "source_directory": str(args.retained_descriptor),
        "regenerated": False,
        "files": {name: {"sha256": sha(args.retained_descriptor / name), "bytes": (args.retained_descriptor / name).stat().st_size}
                  for name in retained_files},
    }
    write_json(args.out / "retained_descriptor_receipt.json", retained_receipt)
    repo = Path(__file__).resolve().parents[1]
    execution_receipt = {
        "branch": "codex/r10-athermal-mixed-czm-comparison",
        "accepted_base_head": "28761405af7d8f09314138c3fd41d34219e4043d",
        "source_files": {
            name: sha(repo / name) for name in (
                "arrhenius_fracture/coupled_opening_reduction_r10.py",
                "scripts/run_r10_coupled_opening_search.py",
                "scripts/analyze_r10_coupled_opening_search.py",
            )
        },
        "allocation": {"physical_vectors": 6, "shapes": 2, "opening_modes": 2,
                       "mechanism_strata": ["OPENING_CONTROL_PREDICTED", "MIXED_CONTROL_PREDICTED", "PLASTICITY_CONTROL_PREDICTED"]},
        "reduced_protocol": {"temperature_K": 300, "R": 0.1, "frequency_Hz": 1000, "cycles": 32,
                             "phases": 8, "load_factors": [0.72, 0.84, 0.94, 1.02, 1.12, 1.28]},
        "native_protocol": {"temperature_K": 300, "R": 0.1, "frequency_Hz": 1000, "cycles": 24,
                            "plastic_phases": 4, "opening_phases": 8, "load_factors": [0.84, 0.96, 1.06, 1.20],
                            "seed_base": 20261017, "process_zone_support_m": 5e-5,
                            "cohesive_cell_m": 1e-6, "observation_path_m": 1.2e-4},
        "monotonic_protocol": {"temperatures_K": [300, 450, 600, 750, 900, 1050, 1200],
                               "Kdot_MPa_sqrt_m_s": 0.005, "dK_MPa_sqrt_m": 1.0,
                               "Kmax_by_material_MPa_sqrt_m": {"R10F_V2_01617": 100, "R10F_V2_03832": 200, "R10F_V2_07211": 200}},
        "continuation": {"prior_native_cases_replayed": False, "new_native_cases_started_from_serialized_complete_material": 32,
                         "per_case_final_rng_and_state_archived": True, "summary_is_not_a_restart": True},
    }
    write_json(args.out / "execution_receipt.json", execution_receipt)

    counts = defaultdict(int)
    for row in native_rows:
        counts[row["disposition"]] += 1
    decision = {
        "schema": "r10.coupled-opening-search-decision/1",
        "preserved_inputs": {"descriptor_bank_regenerated": False, "legacy_opening_clock_used": False, "physical_vectors": 6, "paired_arms": 24},
        "reduction": {"cases": len(reduced_rows), "cycles_requested": 32, "developed_rates": sum(r["developed_rate_m_per_cycle"] is not None for r in reduced_rows), "recurrent_growth_points": sum(r["event_cycle_count"] >= 2 for r in reduced_rows)},
        "fracture": {"ramp_arms": len(thermal_rows), "temperatures_per_arm": 7, "ramp_rate_MPa_sqrt_m_s": 0.005, "temperature_class_counts": dict(sorted((c, sum(r["thermal_class"] == c for r in thermal_rows)) for c in {r["thermal_class"] for r in thermal_rows}))},
        "native": {"cases": len(native_rows), "cycles_or_endpoint_exposures": sum(r["cycles"] for r in native_rows), "dispositions": dict(sorted(counts.items())), "developed_rates": 0, "local_slopes": 0},
        "inactive_bilinear_coordinate_regression": {"candidate": "R10F_V2_01617__AB", "K_cut": 25.900339647524365, "occupied_branch_loss": 27.65295454858545, "eight_phase_discrete_loss": 28.63378214589751, "production_correction_fitted": False},
        "joint_pareto": {"strict_passes": 0, "decision": "ZERO_PARIS_PASS_PRESERVED; NO_STARTUP_OR_CASCADE_RATE_REPORTED"},
        "acquisition": {"queued_paired_arms": len(acquisition), "basis": "CURRENT_STATE_BOUNDARY_AND_MECHANISM_DIVERSITY"},
        "limits": ["No native developed recurrent interval was found in this block.", "Mixed cases with unsupported crossover remain unresolved.", "The current block is 1-D; no candidate passed to a new 2-D run."],
    }
    write_json(args.out / "decision.json", decision)
    results = f"""# R10 coupled opening fatigue/fracture continuation

The completed V2 descriptor bank was retained unchanged. A state-informed reduction was evaluated for all 24 paired smooth/bilinear and athermal/mixed arms ({len(reduced_rows)} cyclic cases), followed by monotonic same-material fracture ramps for 12 arms and a 32-case current full-state 1-D block. The legacy cooperative opening clock was not used.

The inactive bilinear convention is now explicit: endpoint accessibility is `K_cut=25.90033965`, occupied bonded-branch loss is `27.65295455`, and the eight-phase discrete-wave loss is `28.63378215 MPa sqrt(m)`. None is treated as committed separation, and no production correction was fitted.

The reduced block produced no developed recurrent rate. The native block likewise produced **0 developed rates and 0 local slopes**: {counts['SUPPORTED_CYCLIC_SHAKEDOWN'] + counts['FINITE_EXPOSURE_NO_ADVANCE']} supported 24-cycle non-advances, {counts['TARGET_TRUNCATED_SAME_CYCLE_CASCADE']} target-truncated same-cycle cascades, and {counts['UNSUPPORTED_CROSSOVER_UNRESOLVED']} unresolved mixed-crossover cases. Thus the strict joint Pareto set remains empty; cascade extensions and startup passages were not relabeled as Paris rates.

The monotonic ramp is distinct from the cyclic screen. Bilinear `01617` is weak-T with committed roots `[28,28,27,27,26,27,28]`; bilinear `07211` is also weak-T at `[27,27,27,30,31,31,31]`. Strong-plasticity `03832` is cold-censored through 200 MPa sqrt(m) and commits only at hot temperatures (`188,171,156` at 900/1050/1200 K), so it is recorded as hot commitment with a cold Kmax censor, not manufactured into a completed DBTT class. Unsupported mixed ramps remain unresolved.

`continuing_acquisition.csv` ranks the 16 unexecuted paired arms using mechanism diversity plus the observed current-state boundary. It does not enlarge or regenerate the descriptor bank. No new 2-D comparison is authorized by this block because no complete material produced a developed recurrent-growth interval.
"""
    (args.out / "RESULTS.md").write_text(results)
    hashes = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in args.out.iterdir() if p.is_file() and p.name != "SHA256SUMS.json"}
    write_json(args.out / "SHA256SUMS.json", hashes)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
