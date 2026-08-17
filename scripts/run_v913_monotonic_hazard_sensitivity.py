#!/usr/bin/env python3
"""Evaluate local derivatives of the integrated monotonic opening hazard.

The operator replays the authoritative v9.13 first-event displacement path to
its saved peak displacement.  It uses the production adaptive increment,
state-resolution, coupled state advance, and hazard quadrature functions.  A
baseline replay is audited against the saved full pre-passage state; centered
K and T perturbations then give total (state-evolved) and frozen-path direct
hazard derivatives.  No stochastic, constitutive, or geometry law is changed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--fracture-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--candidate-ids", nargs="*", default=())
    parser.add_argument("--temperatures-K", nargs="*", type=float, default=())
    parser.add_argument("--relative-K-perturbation", type=float, default=1.0e-4)
    parser.add_argument("--temperature-perturbation-K", type=float, default=0.1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_external(v913_root: Path):
    if str(v913_root) not in sys.path:
        sys.path.insert(0, str(v913_root))
    from arrhenius_fracture.emergent_gnd_campaign_v913 import candidate_from_registry_row
    from arrhenius_fracture.emergent_gnd_rcurve_v913 import RCurveLoadingMap
    import arrhenius_fracture.emergent_gnd_rcurve_v913 as rcurve
    from scripts.run_mpz_v9_13_persistent_top5 import load_physics
    return candidate_from_registry_row, RCurveLoadingMap, rcurve, load_physics


def replay_to_peak(
    row: dict[str, str], settings: dict[str, Any], temperature_K: float,
    endpoint_U_m: float, geometry_scale: float, direct_perturbations: tuple[float, float] | None,
) -> dict[str, Any]:
    v913_root = Path(settings["v913_root"])
    candidate_from_registry_row, loading_class, rcurve, load_physics = load_external(v913_root)
    candidate = candidate_from_registry_row(row)
    physics, _ = load_physics(Path(settings["physics_json"]))
    loading = loading_class.from_dict(json.loads(Path(settings["loading_map"]).read_text()))
    state = rcurve.EmergentGNDState(candidate, physics)
    threshold = float(loading.threshold_actions[0])
    geometry = float(loading.K_per_U_MPa_sqrt_m_per_m[0]) * geometry_scale
    event_path_advance = float(loading.path_advances_m[0])
    displacement_rate = float(loading.displacement_rate_m_s)
    exponent = float(settings["translation_action_exponent"])
    max_increment = float(settings["max_hazard_increment"])
    minimum_activations = 1.0e-4
    displacement = accumulated = path_advance = elapsed = 0.0
    substeps = 0
    direct = {"Kminus": 0.0, "Kplus": 0.0, "Tminus": 0.0, "Tplus": 0.0}
    eps_K, eps_T = direct_perturbations or (0.0, 0.0)
    tolerance = max(1e-16, abs(endpoint_U_m) * 1e-13)
    while displacement < endpoint_U_m - tolerance:
        substeps += 1
        if substeps > 200_000:
            raise RuntimeError("fixed-peak replay exceeded production substep cap")
        K0 = geometry * displacement
        rate0 = state.cleavage_rate_s(K0, temperature_K)
        if direct_perturbations:
            direct_rate0 = {
                "Kminus": state.cleavage_rate_s(K0 * (1.0 - eps_K), temperature_K),
                "Kplus": state.cleavage_rate_s(K0 * (1.0 + eps_K), temperature_K),
                "Tminus": state.cleavage_rate_s(K0, temperature_K - eps_T),
                "Tplus": state.cleavage_rate_s(K0, temperature_K + eps_T),
            }
        dU_trial = rcurve._adaptive_displacement_increment(
            state=state, geometry_factor=geometry, displacement_m=displacement,
            temperature_K=temperature_K, displacement_rate_m_s=displacement_rate,
            nominal_dU_m=loading.nominal_dU_m, max_hazard_increment=max_increment,
        )
        final_segment = displacement + dU_trial >= endpoint_U_m
        trial_dt = dU_trial / displacement_rate
        trial_K1 = geometry * (displacement + dU_trial)
        trial_Kmid = 0.5 * (K0 + trial_K1)
        trial_predictor = state.cleavage_rate_s(trial_K1, temperature_K)
        trial_dH_predictor = 0.5 * (rate0 + trial_predictor) * trial_dt
        dU = min(dU_trial, endpoint_U_m - displacement)
        dt = dU / displacement_rate
        K1 = geometry * (displacement + dU)
        Kmid = 0.5 * (K0 + K1)
        predictor = state.cleavage_rate_s(K1, temperature_K)
        dH_predictor = (
            trial_dH_predictor if final_segment else 0.5 * (rate0 + predictor) * dt
        )
        desired = max(min((accumulated + dH_predictor) / threshold, 1.0), 0.0) ** exponent
        da_step = min(event_path_advance - path_advance, max(event_path_advance * desired - path_advance, 0.0))
        production_crossing = (
            final_segment
            and geometry_scale == 1.0
            and abs(temperature_K - float(settings["reference_temperature_K"])) < 1e-12
        )
        if production_crossing:
            # This is the production crossing segment: the production operator
            # first advances the complete candidate transaction, evaluates its
            # post-state hazard, restores the pre-trial state, and only then
            # commits the localized fractional segment.  Reproduce that trial
            # even though it is rolled back: resolution and localization must
            # follow the identical production call sequence.
            da_step = event_path_advance - path_advance
        resolved = rcurve._state_advance_is_resolved(
            state, trial_Kmid if final_segment else Kmid, temperature_K,
            trial_dt if final_segment else dt,
            minimum_expected_activations=minimum_activations,
        )
        if production_crossing:
            state_before_trial = copy.deepcopy(state)
            if resolved and da_step > 0.0:
                state.advance_coupled_segment(
                    duration_s=trial_dt, da_m=da_step,
                    K_start_MPa_sqrt_m=K0, K_end_MPa_sqrt_m=trial_K1,
                    T_K=temperature_K, geometry_extension_override_m=0.0,
                )
            elif resolved:
                state.advance_time(trial_dt, trial_Kmid, temperature_K)
            elif da_step > 0.0:
                state.time_s += trial_dt
                state.translate_tip(da_step)
            trial_rate1 = state.cleavage_rate_s(trial_K1, temperature_K)
            trial_dH = 0.5 * (rate0 + trial_rate1) * trial_dt
            fraction = float(np.clip(
                (threshold - accumulated) / max(trial_dH, 1.0e-300),
                0.0, 1.0,
            ))
            state = state_before_trial
            localized_dU = dU_trial * fraction
            if not math.isclose(
                displacement + localized_dU, endpoint_U_m,
                rel_tol=2.0e-12, abs_tol=max(1.0e-18, abs(endpoint_U_m) * 2.0e-12),
            ):
                raise RuntimeError("production crossing replay disagrees with archived endpoint")
            dU = localized_dU
            dt = trial_dt * fraction
            K1 = geometry * (displacement + dU)
            Kmid = 0.5 * (K0 + K1)
        if resolved and da_step > 0.0:
            state.advance_coupled_segment(
                duration_s=dt, da_m=da_step, K_start_MPa_sqrt_m=K0,
                K_end_MPa_sqrt_m=K1, T_K=temperature_K,
                geometry_extension_override_m=0.0,
            )
        elif resolved:
            state.advance_time(dt, Kmid, temperature_K)
        elif da_step > 0.0:
            state.time_s += dt
            state.translate_tip(da_step)
        else:
            state.time_s += dt
        rate1 = state.cleavage_rate_s(K1, temperature_K)
        accumulated += 0.5 * (rate0 + rate1) * dt
        if direct_perturbations:
            direct_rate1 = {
                "Kminus": state.cleavage_rate_s(K1 * (1.0 - eps_K), temperature_K),
                "Kplus": state.cleavage_rate_s(K1 * (1.0 + eps_K), temperature_K),
                "Tminus": state.cleavage_rate_s(K1, temperature_K - eps_T),
                "Tplus": state.cleavage_rate_s(K1, temperature_K + eps_T),
            }
            for key in direct:
                direct[key] += 0.5 * (direct_rate0[key] + direct_rate1[key]) * dt
        displacement += dU
        elapsed += dt
        path_advance += da_step
    return {
        "hazard_action": accumulated, "substeps": substeps, "elapsed_time_s": elapsed,
        "path_advance_m": path_advance, "state": state, "direct_hazards": direct,
    }


def state_error(state: Any, archive: Path) -> tuple[float, float]:
    comparisons = {
        "mobile_m2": np.asarray(state.mobile_m2),
        "retained_m2": np.asarray(state.retained_m2),
        "accumulated_slip_m2": np.asarray(state.accumulated_slip_m2),
    }
    maximum_relative = maximum_absolute = 0.0
    with np.load(archive) as saved:
        positions = np.flatnonzero(np.asarray(saved["event_index"]) == 0)
        if len(positions) != 1:
            raise RuntimeError(f"saved event zero is not unique: {archive}")
        index = int(positions[0])
        for key, current in comparisons.items():
            expected = np.asarray(saved[key][index])
            difference = np.abs(current - expected)
            maximum_absolute = max(maximum_absolute, float(np.max(difference)))
            scale = np.maximum(np.maximum(np.abs(current), np.abs(expected)), 1.0)
            maximum_relative = max(maximum_relative, float(np.max(difference / scale)))
    return maximum_relative, maximum_absolute


def one_case(task: tuple[dict[str, str], Path, dict[str, Any], float, float]) -> dict[str, Any]:
    row, case_path, settings, eps_K, eps_T = task
    payload = json.loads(case_path.read_text())
    event = payload["events"][0]
    temperature = float(payload["temperature_K"])
    endpoint = float(event["applied_displacement_m"])
    Kpeak = float(event["K_MPa_sqrt_m"])
    local_settings = {**settings, "reference_temperature_K": temperature}
    baseline = replay_to_peak(row, local_settings, temperature, endpoint, 1.0, (eps_K, eps_T))
    variants = {
        "Kminus": replay_to_peak(row, local_settings, temperature, endpoint, 1.0 - eps_K, None),
        "Kplus": replay_to_peak(row, local_settings, temperature, endpoint, 1.0 + eps_K, None),
        "Tminus": replay_to_peak(row, local_settings, temperature - eps_T, endpoint, 1.0, None),
        "Tplus": replay_to_peak(row, local_settings, temperature + eps_T, endpoint, 1.0, None),
    }
    for key, value in variants.items():
        if value["hazard_action"] <= 0.0:
            raise RuntimeError(f"nonpositive shadow hazard {case_path}:{key}")
    A_K = math.log(variants["Kplus"]["hazard_action"] / variants["Kminus"]["hazard_action"]) / (2.0 * eps_K * Kpeak)
    A_T = math.log(variants["Tplus"]["hazard_action"] / variants["Tminus"]["hazard_action"]) / (2.0 * eps_T)
    direct = baseline["direct_hazards"]
    A_K_direct = math.log(direct["Kplus"] / direct["Kminus"]) / (2.0 * eps_K * Kpeak)
    A_T_direct = math.log(direct["Tplus"] / direct["Tminus"]) / (2.0 * eps_T)
    relative, absolute = state_error(baseline["state"], Path(payload["event_state_npz"]))
    threshold = float(event["threshold_action"])
    return {
        "candidate_id": payload["candidate_id"], "temperature_K": temperature,
        "K_first_MPa_sqrt_m": Kpeak, "endpoint_displacement_m": endpoint,
        "threshold_action": threshold, "baseline_replayed_hazard_action": baseline["hazard_action"],
        "baseline_hazard_relative_error": abs(baseline["hazard_action"] - threshold) / threshold,
        "baseline_state_max_relative_error": relative, "baseline_state_max_absolute_error": absolute,
        "baseline_state_replay_valid": relative <= 5e-9,
        "A_K_total_per_MPa_sqrt_m": A_K, "A_T_total_per_K": A_T,
        "hazard_predicted_dK_dT_MPa_sqrt_m_per_K": -A_T / A_K,
        "A_K_direct_frozen_path_per_MPa_sqrt_m": A_K_direct,
        "A_T_direct_frozen_path_per_K": A_T_direct,
        "A_K_state_correction_per_MPa_sqrt_m": A_K - A_K_direct,
        "A_T_state_correction_per_K": A_T - A_T_direct,
        "relative_K_perturbation": eps_K, "temperature_perturbation_K": eps_T,
        "baseline_substeps": baseline["substeps"],
        "Kminus_hazard_action": variants["Kminus"]["hazard_action"],
        "Kplus_hazard_action": variants["Kplus"]["hazard_action"],
        "Tminus_hazard_action": variants["Tminus"]["hazard_action"],
        "Tplus_hazard_action": variants["Tplus"]["hazard_action"],
        "sensitivity_operator": "EXACT_V913_FIXED_PEAK_COUPLED_TRAJECTORY_REPLAY_CENTERED_DIFFERENCE",
        "physics_changed": False,
    }


def main() -> int:
    args = parse_args()
    contract = json.loads((args.fracture_root / "run_contract.json").read_text())
    settings = dict(contract["settings"])
    settings["max_hazard_increment"] = 0.05
    settings["translation_action_exponent"] = 0.95
    with args.registry.open(newline="") as stream:
        registry = {row["prospective_candidate_id"]: row for row in csv.DictReader(stream)}
    tasks = []
    selected_candidates = set(args.candidate_ids)
    selected_temperatures = {float(value) for value in args.temperatures_K}
    for path in sorted((args.fracture_root / "cases").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or not payload.get("events"):
            continue
        if selected_candidates and str(payload["candidate_id"]) not in selected_candidates:
            continue
        if selected_temperatures and float(payload["temperature_K"]) not in selected_temperatures:
            continue
        tasks.append((registry[str(payload["candidate_id"])], path, settings, args.relative_K_perturbation, args.temperature_perturbation_K))
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(one_case, task): task[1] for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"V913_MONOTONIC_HAZARD_CASE_COMPLETE candidate={row['candidate_id']} T={row['temperature_K']:g} AK={row['A_K_total_per_MPa_sqrt_m']:.6g} AT={row['A_T_total_per_K']:.6g}", flush=True)
    output = pd.DataFrame(rows).sort_values(["candidate_id", "temperature_K"])
    args.out.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out / "prospective_monotonic_integrated_hazard_sensitivity.csv", index=False)
    manifest = {
        "schema": "v913_monotonic_integrated_hazard_sensitivity_v1",
        "case_count": len(output), "all_baseline_state_replays_valid": bool(output.baseline_state_replay_valid.all()),
        "maximum_baseline_state_relative_error": float(output.baseline_state_max_relative_error.max()),
        "maximum_baseline_hazard_relative_error": float(output.baseline_hazard_relative_error.max()),
        "all_baseline_hazard_replays_within_1e_minus4_relative": bool((output.baseline_hazard_relative_error <= 1.0e-4).all()),
        "relative_K_perturbation": args.relative_K_perturbation,
        "temperature_perturbation_K": args.temperature_perturbation_K,
        "fracture_run_contract_sha256": sha256(args.fracture_root / "run_contract.json"),
        "physics_changed": False,
    }
    (args.out / "prospective_monotonic_hazard_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not manifest["all_baseline_state_replays_valid"] or not manifest["all_baseline_hazard_replays_within_1e_minus4_relative"]:
        raise RuntimeError("baseline full-state or hazard replay audit failed")
    print(f"V913_MONOTONIC_HAZARD_COMPLETE cases={len(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
