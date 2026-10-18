#!/usr/bin/env python3
"""V2-style acquisition for the corrected mechanical/mixed cohesive families.

This is an additive search driver.  It reuses the retained V2 source portfolio,
Sobol allocation, and EXP_floor mutation rules, but all opening descriptors are
computed from the current finite-compliance cohesive coordinate.  Native
promotion calls the transaction-qualified recurrence consumer directly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cohesive_arrhenius import (
    BILINEAR, SMOOTH_EXPONENTIAL, CohesiveThermalLaw,
    FAMILY, OPENING_ALTERNATIVES_FAMILY, cohesive_shape_response, mechanical_completion,
)
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.opening_alternatives_r10 import OpeningAlternative, evaluate_opening
from scripts.run_r10_cohesive_arrhenius_search import SEED, generate, gamma_T
from scripts.run_r10_cohesive_state_aware import mutate_dislocations, valid_transport
from scripts.run_r10_global_mechanism_search import (
    canonical_hash, phase_action, read_sources, transport_rate,
)
from scripts.run_r10_opening_alternatives import mechanics, thermal_from
from scripts.run_r10_opening_recurrence import run_case

SCHEMA = "r10.v2-current-opening-joint-search/1"
TEMPERATURES = (300., 450., 600., 750., 900., 1050., 1200.)
MODES = ("cohesive_athermal", "cohesive_mixed")
SHAPES = (SMOOTH_EXPONENTIAL, BILINEAR)
R_RATIO = .1
FREQUENCY_HZ = 1000.


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict], compressed: bool = False) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compressed else open
    with opener(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def opening_arm(mode: str, shape: str, cohesive: dict) -> OpeningAlternative:
    return OpeningAlternative(
        mode, shape,
        "quasistatic_mechanical" if mode == "cohesive_athermal"
        else "time_dependent_first_passage",
        None if mode == "cohesive_athermal" else float(cohesive["activation_area_m2"]),
        None if mode == "cohesive_athermal" else float(cohesive["attempt_frequency_s"]),
    )


def waveform_descriptor(cohesive: dict, shape: str, mode: str, Kmax: float,
                        temperature: float, phases: int = 128) -> dict:
    """Follow the occupied branch over one frozen-state waveform.

    The mixed action is accumulated only in the qualified high-barrier domain.
    Static saddle MFPTs are recorded as uncertainty and never consumed as a
    cyclic hazard.
    """
    thermal = thermal_from(cohesive)
    _, kenv = mechanics(thermal, temperature)
    alt = opening_arm(mode, shape, cohesive)
    wave = FatigueWaveform(Kmax=Kmax * 1e6, R=R_RATIO, frequency_Hz=FREQUENCY_HZ)
    state = {"branch": "BONDED", "delta_m": 0.}
    action = 0.
    validity: dict[str, int] = {}
    completion = False
    first_completion_phase = None
    maximum_separation = 0.
    dt = wave.period_s / phases
    for phase in range(phases):
        angle = 2 * math.pi * (phase + .5) / phases
        K = max(float(wave.K_phase(angle)), 0.)
        stress = K / math.sqrt(2 * math.pi * 1e-6)
        opened = evaluate_opening(alt, stress / kenv, kenv, thermal, temperature,
                                  state, phase_duration_s=dt)
        state = opened["connected_state"]
        tag = opened["kinetic_validity"]
        validity[tag] = validity.get(tag, 0) + 1
        point = opened.get("connected")
        if point is not None:
            maximum_separation = max(maximum_separation, float(point["delta_m"]))
        if tag == "QUALIFIED_HIGH_BARRIER" and opened.get("elementary_rate_s-1") is not None:
            action += float(opened["elementary_rate_s-1"]) * dt
        if opened["state"] == "SEPARATION_COMPLETED":
            completion = True
            first_completion_phase = phase
            break
    return {"mechanical_completion": completion,
            "first_completion_phase": first_completion_phase,
            "qualified_mixed_action": action,
            "maximum_separation_m": maximum_separation,
            "kinetic_validity_counts": validity}


def access_load(cohesive: dict, shape: str, mode: str, temperature: float) -> tuple[float, dict]:
    """Exact finite-compliance load at the existing completion cutoff.

    This is a mechanical accessibility coordinate for both modes, not a mixed
    fatigue-rate prediction.  Mixed thermal action is evaluated separately and
    uncertainty is left for native promotion.
    """
    thermal = thermal_from(cohesive)
    _, kenv = mechanics(thermal, temperature)
    cutoff = float(mechanical_completion(shape, 0., thermal, temperature)["cutoff_m"])
    traction = float(cohesive_shape_response(shape, cutoff, thermal, temperature)[0])
    imposed = cutoff + traction / kenv
    free_stress = kenv * imposed
    K = free_stress * math.sqrt(2 * math.pi * 1e-6) / 1e6
    return K, {"threshold_kind": "FINITE_COMPLIANCE_COMPLETION_COORDINATE",
               "completion_cutoff_m": cutoff, "traction_at_cutoff_Pa": traction,
               "mixed_thermal_advance_included": False}


def plastic_exposure(parent: dict, K: float) -> tuple[float, str]:
    """EXP_floor source/transport exposure descriptor, not a fatigue rate."""
    opening_scale = max(K, .05)
    emission = max(float(phase_action(parent, "emit", opening_scale, 300.)), 1e-300)
    stress = min(opening_scale * 1e6 / math.sqrt(2 * math.pi * 1e-6), 30e9)
    peierls = max(float(transport_rate(parent, "peierls", stress, 300.)), 1e-300)
    taylor = max(float(transport_rate(parent, "taylor", stress, 300.)), 1e-300)
    source_count = max(float(parent["rho_source0_m2"]) *
                       float(parent["reference_source_area_um2"]) * 1e-12, 1.)
    score = math.log10(max(emission * source_count * math.sqrt(peierls * taylor) /
                           FREQUENCY_HZ, 1e-300))
    label = "OPENING_CONTROL_PREDICTED" if score < -1 else (
        "MIXED_CONTROL_PREDICTED" if score <= 2 else "PLASTICITY_CONTROL_PREDICTED")
    return score, label


def thermal_class(roots: list[float]) -> str:
    a = np.asarray(roots, float)
    if not np.all(np.isfinite(a)):
        return "THERMAL_ACCESS_CENSORED"
    if np.max(a) / max(np.min(a), 1e-300) <= 1.25:
        return "WEAK_T_PREDICTED"
    imax = int(np.argmax(a))
    if 0 < imax < len(a)-1 and a[imax] >= 1.10 * max(a[0], a[-1]):
        return "PEAK_T_PREDICTED"
    if a[-1] > 1.25 * a[0]:
        return "DBTT_PREDICTED"
    return "OTHER_T_PREDICTED"


def consumed_parent(parent: dict) -> dict:
    metadata = {"candidate_id", "option_key", "material_class", "role",
                "mechanism_summary", "validation_status", "search_source_id",
                "coordinate_source_Tref_K", "coordinate_rereference_only"}
    return {k: v for k, v in parent.items()
            if k not in metadata and not k.startswith("cleave_")}


def material_row(cid: str, design: str, cohesive: dict, parent: dict,
                 source_parent_id: str) -> dict:
    ph = canonical_hash(parent)
    ch = canonical_hash(cohesive)
    complete = canonical_hash({"family": FAMILY,
                               "candidate_id": cid,
                               "parent_sha256": ph,
                               "cohesive_sha256": ch})
    return {"candidate_id": cid, "family": FAMILY,
            "additive_family": OPENING_ALTERNATIVES_FAMILY,
            "complete_row_sha256": complete,
            "parent_candidate_id": source_parent_id,
            "parent_complete_row_sha256": ph,
            "plasticity_physics_sha256": canonical_hash(consumed_parent(parent)),
            "cohesive_parameter_sha256": ch,
            "cleavage_hits": 3.0, "cleavage_tau_s": 1e-6,
            "search_design": design,
            "cohesive_parameter_vector": json.dumps(cohesive, sort_keys=True,
                                                      separators=(",", ":")),
            "parent_complete_parameter_vector": json.dumps(parent, sort_keys=True,
                                                             separators=(",", ":"))}


def acquire(args) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    sources = read_sources(args.original_registry, args.parent_registry)
    points = qmc.Sobol(d=5, scramble=True, seed=SEED).random_base2(args.log2_bank)
    mutations = qmc.Sobol(d=11, scramble=True, seed=20261017).random_base2(args.log2_bank)
    anchors: list[dict] = []
    with args.anchor_registry.open(newline="") as stream:
        for row in csv.DictReader(stream):
            cohesive = json.loads(row["cohesive_parameter_vector"])
            if cohesive not in anchors:
                anchors.append(cohesive)
    fixed_parents = [s for s in sources if s["candidate_id"] in {
        "P25_TRANSFER_V1_RANK1", "P40_TRANSFER_CALIBRATED_GEN2",
        "v913_zeroD_sobol_0202500", "v913_zeroD_sobol_0129902"}]
    bank: list[dict] = []
    registry_by_id: dict[str, dict] = {}
    for index, (u, mutation) in enumerate(zip(points, mutations)):
        item = generate(index, u, sources)
        design_index = index % 3
        if design_index == 0:
            design = "FIXED_COHESION_VARY_PLASTICITY"
            cohesive = dict(anchors[(index // 3) % len(anchors)])
            base = sources[(index // (3 * len(anchors))) % len(sources)]
            parent = mutate_dislocations(base, mutation)
        elif design_index == 1:
            design = "FIXED_PLASTICITY_VARY_COHESION"
            cohesive = dict(item["cohesive"])
            base = fixed_parents[(index // 3) % len(fixed_parents)]
            parent = dict(base)
        else:
            design = "JOINT_COHESION_PLASTICITY"
            cohesive = dict(item["cohesive"])
            base = item["parent"]
            parent = mutate_dislocations(base, mutation)
        if min(gamma_T(cohesive, T) for T in TEMPERATURES) <= .2 or not valid_transport(parent):
            continue
        cid = f"R10F_V2_{index:05d}"
        registry_by_id[cid] = material_row(cid, design, cohesive, parent,
                                               str(base["candidate_id"]))
        for shape in SHAPES:
            for mode in MODES:
                roots = []
                details = []
                for temperature in TEMPERATURES:
                    root, detail = access_load(cohesive, shape, mode, temperature)
                    roots.append(root)
                    details.append(detail)
                score, mechanism = plastic_exposure(parent, roots[0] if math.isfinite(roots[0]) else 20.)
                bank.append({"candidate_id": cid, "search_design": design,
                    "source_parent_id": base["candidate_id"], "cohesive_shape": shape,
                    "opening_mode": mode, "mechanism_prediction": mechanism,
                    "thermal_prediction": thermal_class(roots),
                    "plastic_exposure_log10": score,
                    "delta0_m": cohesive["delta0_m"],
                    "cohesive_zone_scale_m": mechanics(thermal_from(cohesive), 300.)[0],
                    **{f"Kaccess_{int(T)}K": root for T, root in zip(TEMPERATURES, roots)},
                    "Kaccess_ratio_1200_over_300": (roots[-1] / roots[0]
                        if math.isfinite(roots[0]) and math.isfinite(roots[-1]) else math.nan),
                    "access_detail_300K": json.dumps(details[0], sort_keys=True,
                                                       separators=(",", ":"))})
        if index and index % 512 == 0:
            print(json.dumps({"allocated": index, "bank_rows": len(bank)}), flush=True)

    # Balanced farthest-from-threshold ordering within every requested cross.
    chosen: list[dict] = []
    per_cell = max(args.native_count // (3 * 2 * 2), 1)
    for design in ("FIXED_COHESION_VARY_PLASTICITY",
                   "FIXED_PLASTICITY_VARY_COHESION", "JOINT_COHESION_PLASTICITY"):
        for mode in MODES:
            for family in ("OPENING_OR_MIXED", "PLASTICITY"):
                pool = [r for r in bank if r["search_design"] == design and
                        r["opening_mode"] == mode and
                        ((family == "PLASTICITY") ==
                         (r["mechanism_prediction"] == "PLASTICITY_CONTROL_PREDICTED")) and
                        math.isfinite(float(r["Kaccess_300K"])) and
                        1. <= float(r["Kaccess_300K"]) <= 250.]
                pool.sort(key=lambda r: (abs(math.log10(max(float(r["Kaccess_300K"]), 1e-12) / 25.)),
                                         abs(float(r["plastic_exposure_log10"])), r["candidate_id"],
                                         r["cohesive_shape"]))
                chosen.extend(pool[:per_cell])
    unique: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for row in chosen + sorted(bank, key=lambda r: (
            abs(float(r["plastic_exposure_log10"])), r["candidate_id"])):
        key = (row["candidate_id"], row["cohesive_shape"], row["opening_mode"])
        if key not in seen:
            unique.append(row); seen.add(key)
        if len(unique) >= args.native_count:
            break
    selected_registry = []
    for row in unique:
        material = dict(registry_by_id[row["candidate_id"]])
        source_id = material["candidate_id"]
        arm_code = ("A" if row["opening_mode"] == "cohesive_athermal" else "M") + (
            "S" if row["cohesive_shape"] == SMOOTH_EXPONENTIAL else "B")
        execution_id = f"{source_id}__{arm_code}"
        material["source_material_candidate_id"] = source_id
        material["candidate_id"] = execution_id
        material["complete_row_sha256"] = canonical_hash({
            "family": FAMILY, "additive_family": OPENING_ALTERNATIVES_FAMILY,
            "source_complete_row_sha256":
            registry_by_id[source_id]["complete_row_sha256"],
            "opening_mode": row["opening_mode"], "cohesive_shape": row["cohesive_shape"]})
        material.update({"selected_shape": row["cohesive_shape"],
                         "selected_mode": row["opening_mode"],
                         "predicted_Kaccess_300K": row["Kaccess_300K"],
                         "mechanism_prediction": row["mechanism_prediction"],
                         "thermal_prediction": row["thermal_prediction"],
                         "plastic_exposure_log10": row["plastic_exposure_log10"]})
        selected_registry.append(material)
    write_csv(args.out / "analytical_bank.csv.gz", bank, True)
    write_csv(args.out / "native_information_registry.csv", selected_registry)
    write_csv(args.out / "native_information_descriptors.csv", unique)
    counts: dict[str, int] = {}
    for row in bank:
        key = "__".join((row["search_design"], row["opening_mode"],
                         row["mechanism_prediction"], row["thermal_prediction"]))
        counts[key] = counts.get(key, 0) + 1
    payload = {"schema": SCHEMA, "stage": "ANALYTICAL_COMPLETE",
        "allocated_materials": len(points), "domain_valid_materials": len(registry_by_id),
        "evaluated_opening_arms": len(bank), "native_information_material_arms": len(unique),
        "designs": sorted({r["search_design"] for r in bank}),
        "counts": counts, "legacy_opening_clock_used": False,
        "opening_transaction_semantics": "COMMITTED_SEPARATION_ONLY",
        "source_registry_sha256": hashlib.sha256(args.anchor_registry.read_bytes()).hexdigest()}
    atomic(args.out / "analytical_summary.json", payload)
    print(json.dumps(payload, indent=2)); return 0


def _native_task(task: dict) -> tuple[dict, dict]:
    result = run_case(Path(task["registry"]), task["candidate_id"], task["shape"],
        task["mode"], task["Kmax"], task["temperature_K"], task["cycles"],
        task["plastic_phases"], task["opening_phases"], task["patch_m"], task["seed"],
        task["path_limit_m"], task["transport_substeps"])
    return task, result


def execute(args) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    with args.registry.open(newline="") as stream:
        available = list(csv.DictReader(stream))
    if args.candidate_ids:
        wanted = set(args.candidate_ids.split(","))
        available = [row for row in available if row["candidate_id"] in wanted]
    # Round-robin across design and opening mode; CSV ordering must not collapse
    # a limited information block onto the first acquisition cell.
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in available:
        groups.setdefault((row["search_design"], row["selected_mode"]), []).append(row)
    selected: list[dict] = []
    while len(selected) < min(args.materials, len(available)):
        advanced = False
        for key in sorted(groups):
            if groups[key]:
                selected.append(groups[key].pop(0)); advanced = True
                if len(selected) >= args.materials:
                    break
        if not advanced:
            break
    tasks: list[dict] = []
    for material_index, row in enumerate(selected):
        K0 = float(row["predicted_Kaccess_300K"])
        factors = [float(x) for x in args.load_factors.split(",")]
        for load_index, factor in enumerate(factors):
            K = factor * K0
            tasks.append({"kind":"fatigue", "registry":str(args.registry),
                "candidate_id":row["candidate_id"], "search_design":row["search_design"],
                "shape":row["selected_shape"], "mode":row["selected_mode"], "Kmax":K,
                "temperature_K":300., "cycles":args.cycles,
                "plastic_phases":args.plastic_phases, "opening_phases":args.opening_phases,
                "patch_m":args.patch_um*1e-6, "path_limit_m":args.path_limit_um*1e-6,
                "transport_substeps":args.transport_substeps,
                "seed":args.seed+1000*material_index+load_index})
        # Same complete material thermal accessibility checks at its predicted endpoints.
        cohesive = json.loads(row["cohesive_parameter_vector"])
        for temperature in (300., 1200.):
            predicted, _ = access_load(cohesive, row["selected_shape"], row["selected_mode"], temperature)
            for factor in (.98, 1.02):
                K = factor * predicted
                tasks.append({"kind":"thermal", "registry":str(args.registry),
                    "candidate_id":row["candidate_id"], "search_design":row["search_design"],
                    "shape":row["selected_shape"], "mode":row["selected_mode"], "Kmax":K,
                    "temperature_K":temperature, "cycles":1, "predicted_Kaccess":predicted,
                    "thermal_bracket_factor":factor, "plastic_phases":args.plastic_phases,
                    "opening_phases":args.opening_phases, "patch_m":args.patch_um*1e-6,
                    "path_limit_m":args.path_limit_um*1e-6,
                    "transport_substeps":args.transport_substeps,
                    "seed":args.seed+100000+1000*material_index+int(temperature)+int(100*factor)})
    cases: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_native_task, task) for task in tasks]
        for future in as_completed(futures):
            task, result = future.result()
            arm_name = f'{task["mode"]}__{task["shape"]}'
            if task["kind"] == "fatigue":
                case_dir = (args.out/"fatigue"/task["candidate_id"]/arm_name/
                            f'K_{task["Kmax"]:.8g}')
            else:
                case_dir = (args.out/"thermal"/task["candidate_id"]/arm_name/
                            f'T_{int(task["temperature_K"])}_f_{task["thermal_bracket_factor"]:.2f}')
            atomic(case_dir/"result.json", result)
            row = {"candidate_id":task["candidate_id"], "search_design":task["search_design"],
                "shape":task["shape"], "mode":task["mode"], "case_kind":task["kind"],
                "temperature_K":task["temperature_K"], "Kmax_MPa_sqrt_m":task["Kmax"],
                "DeltaK_MPa_sqrt_m":.9*task["Kmax"],
                "cycles":len(result["cycle_history"]), "events":len(result["events"]),
                "extension_m":result["cycle_history"][-1]["extension_m"] if result["cycle_history"] else 0.,
                "stop_category":result["stop_category"],
                "unsupported_crossover":result["unsupported_crossover_encountered"]}
            if task["kind"] == "thermal":
                row.update({"predicted_Kaccess":task["predicted_Kaccess"],
                            "thermal_bracket_factor":task["thermal_bracket_factor"]})
            cases.append(row); print(json.dumps(row), flush=True)
    write_csv(args.out / "native_case_dispositions.csv", cases)
    atomic(args.out / "native_execution_summary.json", {"schema": SCHEMA,
        "stage": "NATIVE_INFORMATION_BLOCK_COMPLETE", "materials": len(selected),
        "fatigue_cases": len(selected)*len([float(x) for x in args.load_factors.split(",")]),
        "thermal_cases": len(selected)*4, "cases": len(cases),
        "cycles_per_fatigue_case": args.cycles, "transaction_consumer":
        "scripts.run_r10_opening_recurrence.run_case"})
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("acquire")
    a.add_argument("--original-registry", type=Path, required=True)
    a.add_argument("--parent-registry", type=Path, required=True)
    a.add_argument("--anchor-registry", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--log2-bank", type=int, default=13)
    a.add_argument("--native-count", type=int, default=24)
    a.set_defaults(func=acquire)
    e = sub.add_parser("execute")
    e.add_argument("--registry", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--materials", type=int, default=6)
    e.add_argument("--candidate-ids")
    e.add_argument("--load-factors", default=".88,.96,1.04,1.14")
    e.add_argument("--cycles", type=int, default=48)
    e.add_argument("--plastic-phases", type=int, default=8)
    e.add_argument("--opening-phases", type=int, default=64)
    e.add_argument("--transport-substeps", type=int, default=1)
    e.add_argument("--patch-um", type=float, default=1.)
    e.add_argument("--path-limit-um", type=float, default=120.)
    e.add_argument("--seed", type=int, default=20261017)
    e.add_argument("--workers", type=int, default=1)
    e.set_defaults(func=execute)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
