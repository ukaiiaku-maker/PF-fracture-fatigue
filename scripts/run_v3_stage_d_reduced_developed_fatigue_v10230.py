#!/usr/bin/env python3
"""Reduced event-to-event developed-fatigue campaign for six V3 finalists.

This is the v9.14 exact-row lineage requested by the protocol.  It is labeled
reduced because it translates the 1-D process-zone state directly and does not
replace the later sparse spatial energy-gated transfer.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from arrhenius_fracture.canonical_v2_registry_v10230 import ThermodynamicBarrierAdapter
from scripts.fatigue_informed_joint_barrier_v3_v10230 import deserialize_surface
from scripts import select_v2_3_exact_row_cycle_hazard_loads_v10230 as exact_row


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-registry", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1720)
    parser.add_argument("--loads-per-candidate", type=int, default=5)
    parser.add_argument("--target-extension-um", type=float, default=100.0)
    parser.add_argument("--maximum-cycles", type=float, default=1.0e12)
    return parser.parse_args()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def load_candidate(
    fatigue, complete: dict, *, canonical_registry=None, canonical_manifest=None,
):
    exact_row.CANDIDATE_ID = complete["source_candidate_id"]
    source, physics, row = exact_row.exact_candidate_and_physics(
        fatigue, registry_path=canonical_registry, manifest_path=canonical_manifest,
    )
    surface = deserialize_surface(complete["opening_surface"])
    parent = json.loads(row["parent_material_manifest_json"])["cleavage"]
    adapter = ThermodynamicBarrierAdapter(surface, parent)
    candidate = replace(source, candidate_id=complete["candidate_id"], cleavage=adapter)
    if float(physics.cleavage_hits) != float(complete["renewal"]["hits"]):
        physics = replace(physics, cleavage_hits=float(complete["renewal"]["hits"]))
    if float(physics.cleavage_correlation_time_s) != float(complete["renewal"]["tau_s"]):
        physics = replace(
            physics, cleavage_correlation_time_s=float(complete["renewal"]["tau_s"])
        )
    return candidate, physics, row


def load_grid(complete: dict, count: int) -> list[float]:
    window = complete["reference_window"]
    low = float(window["window_K_low"])
    high = float(window["window_K_high"])
    if high <= low:
        raise ValueError(f"candidate {complete['candidate_id']} has no positive reference window")
    return np.geomspace(low, high, int(count)).tolist()


def run_case(fatigue, complete: dict, candidate, physics, Kmax: float, seed: int,
             run_root: Path, target_extension_m: float, maximum_cycles: float) -> dict:
    label = f"Kmax_{Kmax:.8f}".replace(".", "p")
    case = run_root / complete["candidate_id"] / f"seed_{seed}" / label
    checkpoint = case / "live_checkpoint.json"
    result_path = case / "result.json"
    loading = fatigue.FatigueLoading(
        deltaK_MPa_sqrt_m=0.9 * Kmax, R=0.1, frequency_Hz=1000.0,
        temperature_K=300.0, phase_steps=64,
    )
    numerics = fatigue.FatigueNumerics(
        maximum_cycles=maximum_cycles, target_extension_m=target_extension_m,
        base_event_length_m=5.0e-6, event_length_factor_min=0.5,
        event_length_factor_max=4.0, checkpoint_wait_cycles=1.0e8,
    )
    contract = {
        "candidate_id": complete["candidate_id"],
        "complete_v3_row_sha256": complete["complete_v3_row_sha256"],
        "loading": asdict(loading), "numerics": asdict(numerics), "seed": seed,
        "driver": "IMMUTABLE_V9_14_EXACT_ROW_REDUCED_EVENT_TO_EVENT",
    }
    contract_hash = canonical_hash(contract)
    if result_path.is_file():
        saved = json.loads(result_path.read_text())
        if saved.get("v3_case_contract_sha256") != contract_hash:
            raise RuntimeError(f"refuse mismatched result at {result_path}")
        return saved
    case.mkdir(parents=True, exist_ok=True)
    result = fatigue.run_cyclic_fatigue(
        candidate, physics, loading, seed=seed, numerics=numerics,
        checkpoint_path=checkpoint, restart_from=checkpoint if checkpoint.is_file() else None,
    )
    payload = result.as_dict()
    payload.update({
        "v3_case_contract": contract, "v3_case_contract_sha256": contract_hash,
        "complete_v3_row_sha256": complete["complete_v3_row_sha256"],
        "fidelity": "REDUCED_1D_EVENT_TO_EVENT_NOT_SPATIAL_ENERGY_GATED_TRANSFER",
        "raw_run_output_not_added_to_git": True,
    })
    atomic_json(result_path, payload)
    return payload


def summarize(result: dict) -> dict:
    events = result["events"]
    developed = [x for x in events if float(x["cumulative_extension_m"]) > 20.0e-6]
    rate = math.nan
    first_window = math.nan
    second_window = math.nan
    if len(developed) >= 2:
        da = float(developed[-1]["cumulative_extension_m"]) - float(developed[0]["cumulative_extension_m"])
        dN = float(developed[-1]["cycles"]) - float(developed[0]["cycles"])
        rate = da / dN if dN > 0.0 else math.nan
    # Two nonoverlapping developed windows, 20-60 and 60-100 um.
    for name, low, high in (("first", 20e-6, 60e-6), ("second", 60e-6, 100e-6)):
        q = [x for x in events if low <= float(x["cumulative_extension_m"]) <= high]
        value = math.nan
        if len(q) >= 2:
            value = (float(q[-1]["cumulative_extension_m"]) - float(q[0]["cumulative_extension_m"])) / (
                float(q[-1]["cycles"]) - float(q[0]["cycles"])
            )
        if name == "first": first_window = value
        else: second_window = value
    loading = result["loading"]
    return {
        "candidate_id": result["candidate_id"], "seed": result["seed"],
        "Kmax_MPa_sqrt_m": loading["deltaK_MPa_sqrt_m"] / (1.0 - loading["R"]),
        "DeltaK_MPa_sqrt_m": loading["deltaK_MPa_sqrt_m"], "R": loading["R"],
        "frequency_Hz": loading["frequency_Hz"], "temperature_K": loading["temperature_K"],
        "status": result["status"], "event_count": len(events),
        "final_cycles": result["final_cycles"], "final_extension_m": result["final_extension_m"],
        "developed_da_dN_m_per_cycle": rate,
        "developed_window_20_60um_da_dN": first_window,
        "developed_window_60_100um_da_dN": second_window,
        "two_nonoverlapping_developed_windows": bool(np.isfinite(first_window) and np.isfinite(second_window)),
        "periodic_state": result["periodic_state"],
        "validated_cycle_map_state": result["validated_cycle_map_state"],
        "fidelity": result["fidelity"],
    }


def main() -> int:
    a = args()
    registry = json.loads(a.candidate_registry.read_text())
    by_id = {row["candidate_id"]: row for row in registry["rows"]}
    selection = pd.read_csv(a.selection)
    # The widest feasible row from each of the six source backgrounds.
    finalists = selection[selection.selection_role == "FEASIBLE_WIDEST_PER_SOURCE"].candidate_id.tolist()
    if len(finalists) != 6:
        raise SystemExit(f"expected six widest-per-source finalists, found {len(finalists)}")
    fatigue = exact_row.load_external_fatigue()
    results = []
    for cid in finalists:
        complete = by_id[cid]
        candidate, physics, _ = load_candidate(fatigue, complete)
        for Kmax in load_grid(complete, a.loads_per_candidate):
            result = run_case(
                fatigue, complete, candidate, physics, float(Kmax), a.seed,
                a.run_root.resolve(), a.target_extension_um * 1.0e-6, a.maximum_cycles,
            )
            summary = summarize(result)
            results.append(summary)
            print(json.dumps({"stage": "D_REDUCED", **{k: summary[k] for k in (
                "candidate_id", "Kmax_MPa_sqrt_m", "status", "event_count",
                "final_cycles", "final_extension_m")}}, sort_keys=True), flush=True)
    frame = pd.DataFrame(results)
    a.out_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out_root / "stage_D_reduced_developed_case_table.csv", index=False)
    slopes = []
    for cid, q in frame.groupby("candidate_id"):
        q = q.sort_values("Kmax_MPa_sqrt_m")
        valid = q[np.isfinite(q.developed_da_dN_m_per_cycle) & (q.developed_da_dN_m_per_cycle > 0.0)]
        local = np.diff(np.log(valid.developed_da_dN_m_per_cycle)) / np.diff(np.log(valid.Kmax_MPa_sqrt_m)) if len(valid) >= 2 else np.array([])
        for index, value in enumerate(local):
            slopes.append({
                "candidate_id": cid,
                "Kmax_low": float(valid.Kmax_MPa_sqrt_m.iloc[index]),
                "Kmax_high": float(valid.Kmax_MPa_sqrt_m.iloc[index + 1]),
                "local_log_slope": float(value),
                "within_predeclared_15_percent_band": bool(3.4 <= value <= 4.6),
            })
    pd.DataFrame(slopes).to_csv(a.out_root / "stage_D_reduced_local_slopes.csv", index=False)
    atomic_json(a.out_root / "stage_D_reduced_decision.json", {
        "schema": "v10.2.30_v3_stage_D_reduced_decision_v1",
        "candidate_count": len(finalists), "trajectory_count": len(frame),
        "completed_growth_targets": int((frame.status == "growth_target_reached").sum()),
        "cycle_censors": int((frame.status == "cycle_censored").sum()),
        "failed_trajectories": int(frame.status.str.startswith("failed").sum()),
        "two_window_trajectories": int(frame.two_nonoverlapping_developed_windows.sum()),
        "classification": "REDUCED_DEVELOPED_SCREEN_REQUIRES_SPARSE_SPATIAL_TRANSFER",
        "event_length_is_existing_stochastic_law": True,
        "paris_law_fitted": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
