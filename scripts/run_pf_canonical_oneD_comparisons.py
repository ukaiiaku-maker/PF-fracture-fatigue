#!/usr/bin/env python3
"""Run matched current V2 PF-provider reductions for the canonical PF matrix."""
from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


CLASS_ALIASES = {"peak": "Peak", "DBTT": "DBTT", "weakT": "weak-T", "ceramic": "ceramic-like"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonable(value: Any):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(type(value).__name__)


def summarize(result: dict[str, Any], plan: dict[str, str]) -> dict[str, Any]:
    events = result["events"]
    onsets = [event for event in events if event["event_index"] == 0 or event["reload_separated"]]
    sizes = [float(event["event_length_m"]) * 1e6 for event in events]
    onset_k = [float(event["native_KJ_MPa_sqrt_m"]) for event in onsets]
    return {
        "case_id": plan["case_id"], "matrix": plan["matrix"],
        "material_class": plan["material_class"], "candidate_id": result["candidate_id"],
        "temperature_K": float(plan["temperature_K"]), "theta_deg": float(plan["theta_deg"]),
        "rate_tag": plan["rate_tag"], "loading_rate_factor": float(plan["loading_rate_factor"]),
        "nominal_dt_s": float(plan["nominal_dt_s"]), "seed": int(plan["seed"]),
        "target_um": float(plan["target_extension_um"]), "status": result["status"],
        "event_count": result["event_count"], "physical_avalanche_count": result["physical_avalanche_count"],
        "precursor_reinitiation_count": result["precursor_reinitiation_count"],
        "largest_avalanche_fraction": result["largest_avalanche_fraction"],
        "first_event_opening_um": None if result["first_event_opening_m"] is None else result["first_event_opening_m"] * 1e6,
        "first_event_native_KJ_MPa_sqrt_m": result["first_event_native_KJ_MPa_sqrt_m"],
        "onset_envelope_min_MPa_sqrt_m": min(onset_k, default=np.nan),
        "onset_envelope_max_MPa_sqrt_m": max(onset_k, default=np.nan),
        "mean_event_size_um": float(np.mean(sizes)) if sizes else np.nan,
        "median_event_size_um": float(np.median(sizes)) if sizes else np.nan,
        "terminal_extension_um": result["terminal_extension_m"] * 1e6,
        "terminal_opening_um": result["terminal_opening_m"] * 1e6,
        "max_tip_radius_um": result["max_tip_radius_m"] * 1e6,
        "max_backstress_GPa": result["max_backstress_Pa"] * 1e-9,
        "minimum_front_width_um": result["min_front_width_m"] * 1e6,
        "max_source_multiplicity": result["max_source_multiplicity"],
        "model_contract": result["model_contract"],
        "mechanics_coordinate": "PROJECTED_X_EXTENSION",
        "mechanics_semantics": "PF_MODEL_NATIVE_PRODUCTION_DISCRETE_SHARP_WAKE_NOT_CONTINUUM_G",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reduced-repo", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reduced = args.reduced_repo.resolve(); sys.path.insert(0, str(reduced))
    from arrhenius_fracture.emergent_gnd_campaign_v913 import candidate_from_registry_row
    from arrhenius_fracture.emergent_gnd_types_v913 import CommonPhysics
    from reduced_fracture_v2.predictive import (
        PF_PREDICTIVE_LIFECYCLE_V2, ProviderMechanicsMap, SourceDriveMap,
        provider_loading_map, run_zero_d_predictive,
    )

    physics_path = reduced / "mpz_v9_13_v10222_transfer_common_physics.json"
    physics = CommonPhysics(**json.loads(physics_path.read_text())["common_physics"])
    registry = pd.read_csv(args.registry)
    registry["canonical_class"] = registry.material_class.map(CLASS_ALIASES).fillna(registry.material_class)
    rows_by_class = {row.canonical_class: row for _, row in registry.iterrows()}
    with args.plan.open(newline="") as stream:
        plans = list(csv.DictReader(stream))
    outputs = []; summaries = []
    for index, plan in enumerate(plans):
        theta = float(plan["theta_deg"])
        mechanics_path = args.maps / f"pf_v2_theta{theta:g}_mechanics_map.csv"
        drive_path = args.maps / f"pf_v2_theta{theta:g}_source_drive_map.csv"
        mechanics = ProviderMechanicsMap.from_csv("PF", mechanics_path)
        drive = SourceDriveMap.from_csv(drive_path)
        projected_advance = 5e-6 * np.cos(np.deg2rad(theta))
        lifecycle = replace(PF_PREDICTIVE_LIFECYCLE_V2, nominal_advance_m=float(projected_advance))
        loading = provider_loading_map(
            mechanics, seed=int(plan["seed"]), target_extension_m=float(plan["target_extension_um"]) * 1e-6,
            lifecycle=lifecycle, nominal_dU_m=float(plan["nominal_dU_m"]), nominal_dt_s=float(plan["nominal_dt_s"]),
        )
        candidate = candidate_from_registry_row(rows_by_class[plan["material_class"]])
        result = run_zero_d_predictive(
            candidate, physics, mechanics, drive, loading, float(plan["temperature_K"]),
            target_extension_m=float(plan["target_extension_um"]) * 1e-6,
            lifecycle=lifecycle, maximum_intervals=500_000, maximum_opening_m=500e-6,
        )
        plan = dict(plan); plan["case_id"] = (
            f"{plan['matrix'].lower()}__{plan['material_class'].replace('-', '').lower()}__"
            f"T{int(float(plan['temperature_K'])):04d}K__theta{theta:g}__{plan['rate_tag']}__seed{plan['seed']}"
        )
        result["canonical_case_identity"] = plan
        result["angle_conditioned_mechanics_map_sha256"] = sha256(mechanics_path)
        result["angle_conditioned_source_drive_map_sha256"] = sha256(drive_path)
        outputs.append(result); summaries.append(summarize(result, plan))
        print(f"ONED_COMPLETE {index + 1}/{len(plans)} {plan['case_id']} status={result['status']}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    cases_path = args.out / "pf_canonical_1D_cases.json"
    cases_path.write_text(json.dumps(outputs, indent=2, sort_keys=True, default=jsonable) + "\n")
    frame = pd.DataFrame(summaries)
    frame.to_csv(args.out / "pf_canonical_1D_comparison_results.csv", index=False)
    manifest = {
        "schema": "pf_canonical_matched_oneD_v2_manifest_v1", "case_count": len(outputs),
        "reduced_source_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=reduced, text=True).strip(),
        "registry_sha256": sha256(args.registry), "plan_sha256": sha256(args.plan),
        "physics_sha256": sha256(physics_path), "all_target_right_censored": all(
            result["status"] == "TARGET_RIGHT_CENSORED" for result in outputs),
        "theta_conditioned": True, "rate_conditioned": True, "seed_matched": True,
        "provider": "PF", "comparison_warning": "ONSET_AND_AVALANCHE_SUMMARIES_NOT_EVENTWISE_R_CURVE",
    }
    (args.out / "pf_canonical_1D_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if manifest["all_target_right_censored"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
