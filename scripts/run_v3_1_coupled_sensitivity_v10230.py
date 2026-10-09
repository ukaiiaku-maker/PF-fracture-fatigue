#!/usr/bin/env python3
"""Paired native-trajectory sensitivity matrix for the V3.1 continuation."""
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
from scripts.fatigue_informed_joint_barrier_v3_v10230 import (
    KineticSurface, SmoothBroadBasis, canonical_hash, deserialize_surface,
    serialize_surface,
)
from scripts.thermodynamic_joint_barrier_v10230 import Basis, ThermodynamicSurface
from scripts import run_v3_stage_d_reduced_developed_fatigue_v10230 as stage_d
from scripts import select_v2_3_exact_row_cycle_hazard_loads_v10230 as exact_row


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--candidate-registry", type=Path, required=True)
    p.add_argument("--v3-cases", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    return p.parse_args()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def serialize_thermodynamic(surface: ThermodynamicSurface) -> dict:
    bases = []
    for basis in surface.bases:
        if isinstance(basis, SmoothBroadBasis):
            bases.append({"kind": "SMOOTH_NORMALIZED_BROAD_V1", **asdict(basis)})
        else:
            bases.append({"kind": "LEGACY_EXP", **asdict(basis)})
    return {
        "process": surface.process, "attempt_frequency_s": surface.attempt_frequency_s,
        "reference_floor_eV": surface.reference_floor_eV,
        "reference_amplitudes_eV": list(surface.reference_amplitudes_eV),
        "bases": bases, "entropy_infinity_kB": surface.entropy_infinity_kB,
        "entropy_amplitudes_kB": list(surface.entropy_amplitudes_kB),
        "heat_capacity_infinity_kB": surface.heat_capacity_infinity_kB,
        "heat_capacity_amplitudes_kB": list(surface.heat_capacity_amplitudes_kB),
    }


def deserialize_thermodynamic(data: dict) -> ThermodynamicSurface:
    bases = []
    for raw in data["bases"]:
        item = dict(raw); kind = item.pop("kind")
        bases.append(SmoothBroadBasis(**item) if kind == "SMOOTH_NORMALIZED_BROAD_V1" else Basis(**item))
    return ThermodynamicSurface(
        process=data["process"], attempt_frequency_s=float(data["attempt_frequency_s"]),
        reference_floor_eV=float(data["reference_floor_eV"]),
        reference_amplitudes_eV=tuple(data["reference_amplitudes_eV"]), bases=tuple(bases),
        entropy_infinity_kB=float(data["entropy_infinity_kB"]),
        entropy_amplitudes_kB=tuple(data["entropy_amplitudes_kB"]),
        heat_capacity_infinity_kB=float(data["heat_capacity_infinity_kB"]),
        heat_capacity_amplitudes_kB=tuple(data["heat_capacity_amplitudes_kB"]),
    )


def solve_amplitude_floor(old_floor: float, old_amplitude: float,
                          old_basis, new_basis, anchor: float) -> tuple[float, float]:
    old_b = float(old_basis.value(anchor)); new_b = float(new_basis.value(anchor))
    if abs(new_b - 1.0) < 1e-12:
        raise ValueError("perturbed basis is singular at constraint anchor")
    amplitude = old_amplitude * (old_b - 1.0) / (new_b - 1.0)
    floor = old_floor + old_amplitude - amplitude
    if floor <= 0.0 or amplitude <= 0.0:
        raise ValueError("barrier perturbation violates positive coefficient bounds")
    return floor, amplitude


def reconstruct_entropy(old, bases, anchors: np.ndarray) -> tuple[float, tuple[float, ...]]:
    matrix = np.column_stack([np.ones(len(anchors))] + [b.value(anchors) for b in bases])
    target = np.asarray(old.entropy_kB(anchors, 300.0), dtype=float)
    coefficients = np.linalg.solve(matrix, target)
    return float(coefficients[0]), tuple(float(x) for x in coefficients[1:])


def variant_surfaces(complete: dict, source_emission, u: np.ndarray):
    """Construct declared barrier perturbations; u is in protocol step units."""
    opening = deserialize_surface(complete["opening_surface"])
    active = 18.0e6 / math.sqrt(2.0 * math.pi * 1.0e-6)
    broad, guard = opening.bases
    new_broad = SmoothBroadBasis(
        broad.sigma_b_Pa, broad.exponent * math.exp(0.05 * float(u[0])),
        broad.smoothing_ratio,
    )
    floor, amplitude = solve_amplitude_floor(
        opening.reference_floor_eV, opening.reference_amplitudes_eV[0],
        broad, new_broad, active,
    )
    entropy_floor, entropy_amplitudes = reconstruct_entropy(
        opening, (new_broad, guard), np.array([0.0, active, 30.0e9])
    )
    opening_new = KineticSurface(
        process=opening.process, attempt_frequency_s=opening.attempt_frequency_s,
        reference_floor_eV=floor,
        reference_amplitudes_eV=(amplitude, opening.reference_amplitudes_eV[1]),
        bases=(new_broad, guard), entropy_infinity_kB=entropy_floor,
        entropy_amplitudes_kB=entropy_amplitudes,
        heat_capacity_infinity_kB=opening.heat_capacity_infinity_kB,
        heat_capacity_amplitudes_kB=opening.heat_capacity_amplitudes_kB,
        hits=opening.hits, tau_s=opening.tau_s,
    )

    emission = source_emission
    old_basis = emission.bases[0]
    new_basis = Basis(old_basis.alpha, old_basis.sigma_Pa,
                      old_basis.exponent * math.exp(0.05 * float(u[1])))
    e_floor, e_amplitude = solve_amplitude_floor(
        emission.reference_floor_eV, emission.reference_amplitudes_eV[0],
        old_basis, new_basis, active,
    )
    e_entropy_floor, e_entropy_amplitudes = reconstruct_entropy(
        emission, (new_basis,), np.array([0.0, active])
    )
    shift = 0.01 * float(u[2])
    emission_new = ThermodynamicSurface(
        process=emission.process, attempt_frequency_s=emission.attempt_frequency_s,
        reference_floor_eV=e_floor + shift,
        reference_amplitudes_eV=(e_amplitude,), bases=(new_basis,),
        entropy_infinity_kB=e_entropy_floor,
        entropy_amplitudes_kB=e_entropy_amplitudes,
        heat_capacity_infinity_kB=emission.heat_capacity_infinity_kB,
        heat_capacity_amplitudes_kB=emission.heat_capacity_amplitudes_kB,
    )
    return opening_new, emission_new


def run_case(fatigue, candidate, physics, parent_id: str, variant_id: str,
             coordinates: list[float], Kmax: float, seed: int, run_root: Path,
             target_extension_m: float, maximum_cycles: float) -> dict:
    label = f"Kmax_{Kmax:.8f}".replace(".", "p")
    case = run_root / variant_id / f"seed_{seed}" / label
    result_path = case / "result.json"; checkpoint = case / "live_checkpoint.json"
    loading = fatigue.FatigueLoading(deltaK_MPa_sqrt_m=0.9*Kmax, R=0.1,
        frequency_Hz=1000.0, temperature_K=300.0, phase_steps=64)
    numerics = fatigue.FatigueNumerics(maximum_cycles=maximum_cycles,
        target_extension_m=target_extension_m, base_event_length_m=5e-6,
        event_length_factor_min=0.5, event_length_factor_max=4.0,
        checkpoint_wait_cycles=1e8)
    contract = {
        "parent_candidate_id": parent_id, "candidate_id": variant_id,
        "normalized_coordinates": coordinates, "candidate": asdict(candidate),
        "physics": asdict(physics), "loading": asdict(loading),
        "numerics": asdict(numerics), "seed": seed,
        "driver": "IMMUTABLE_V9_14_EXACT_ROW_REDUCED_EVENT_TO_EVENT",
    }
    digest = canonical_hash(contract)
    if result_path.is_file():
        saved = json.loads(result_path.read_text())
        if saved.get("v3_1_case_contract_sha256") != digest:
            raise RuntimeError(f"stale or mismatched result: {result_path}")
        return saved
    case.mkdir(parents=True, exist_ok=True)
    result = fatigue.run_cyclic_fatigue(candidate, physics, loading, seed=seed,
        numerics=numerics, checkpoint_path=checkpoint,
        restart_from=checkpoint if checkpoint.is_file() else None)
    payload = result.as_dict()
    payload.update({"v3_1_case_contract": contract,
                    "v3_1_case_contract_sha256": digest,
                    "fidelity": "REDUCED_1D_NATIVE_EVENT_TO_EVENT_SENSITIVITY",
                    "raw_run_output_not_added_to_git": True})
    atomic_json(result_path, payload)
    return payload


def summarize(result: dict, parent: str, variant: str, coordinates: list[float]) -> dict:
    row = stage_d.summarize(result)
    developed = [e for e in result["events"] if e["cumulative_extension_m"] > 20e-6]
    row.update({
        "parent_candidate_id": parent, "variant_id": variant,
        "opening_shape_coordinate": coordinates[0],
        "emission_shape_coordinate": coordinates[1],
        "relative_emission_barrier_coordinate": coordinates[2],
        "median_developed_radius_m": float(np.median([e["tip_radius_m"] for e in developed])),
        "median_developed_shielding_MPa_sqrt_m": float(np.median([e["shielding_MPa_sqrt_m"] for e in developed])),
        "median_developed_mobile_m2": float(np.median([e["mobile_m2"] for e in developed])),
        "median_developed_retained_m2": float(np.median([e["retained_m2"] for e in developed])),
        "mean_accepted_to_proposed_ratio": float(np.mean([e["committed_advance_m"]/e["proposed_advance_m"] for e in result["events"]])),
    })
    return row


def main() -> int:
    a = args(); protocol = json.loads(a.protocol.read_text())
    registry = json.loads(a.candidate_registry.read_text())
    by_id = {x["candidate_id"]: x for x in registry["rows"]}
    old_cases = pd.read_csv(a.v3_cases)
    fatigue = exact_row.load_external_fatigue()
    perturbations = []
    names = ["opening_log_derivative_curvature", "emission_log_derivative",
             "relative_opening_emission_barrier"]
    for j, name in enumerate(names):
        for sign in (-1.0, 1.0):
            u = np.zeros(3); u[j] = sign
            perturbations.append((name, sign, u))
    records, variant_registry = [], []
    for parent_id in protocol["mechanism_representatives"]:
        complete = by_id[parent_id]
        base_candidate, physics, row = stage_d.load_candidate(fatigue, complete)
        loads = old_cases[old_cases.candidate_id == parent_id].sort_values("Kmax_MPa_sqrt_m").Kmax_MPa_sqrt_m.to_numpy()
        diagnostic_loads = loads[np.asarray(protocol["diagnostic_load_indices"], dtype=int)]
        for direction, sign, u in perturbations:
            opening, emission = variant_surfaces(complete, base_candidate.emission.surface, u)
            variant_id = f"{parent_id}__{direction}__{'plus' if sign > 0 else 'minus'}"
            cleavage_adapter = ThermodynamicBarrierAdapter(opening, base_candidate.cleavage.parent)
            emission_adapter = ThermodynamicBarrierAdapter(emission, base_candidate.emission.parent)
            candidate = replace(base_candidate, candidate_id=variant_id,
                                cleavage=cleavage_adapter, emission=emission_adapter)
            variant_registry.append({
                "variant_id": variant_id, "parent_candidate_id": parent_id,
                "direction": direction, "sign": sign,
                "normalized_coordinates": u.tolist(),
                "candidate_sha256": canonical_hash(asdict(candidate)),
                "opening_surface": serialize_surface(opening),
                "emission_surface": serialize_thermodynamic(emission),
                "attempt_frequencies_modified": False, "network_law_modified": False,
            })
            for Kmax in diagnostic_loads:
                result = run_case(fatigue, candidate, physics, parent_id, variant_id,
                    u.tolist(), float(Kmax), int(protocol["sensitivity"]["paired_seed"]),
                    a.run_root.resolve(), protocol["sensitivity"]["target_extension_um"]*1e-6,
                    protocol["sensitivity"]["maximum_cycles"])
                record = summarize(result, parent_id, variant_id, u.tolist())
                records.append(record)
                print(json.dumps({k: record[k] for k in ("variant_id", "Kmax_MPa_sqrt_m", "status", "event_count")}), flush=True)
    frame = pd.DataFrame(records)
    a.out_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out_root / "native_paired_sensitivity_trajectories.csv", index=False)
    atomic_json(a.out_root / "v3_1_sensitivity_variant_registry.json", {
        "schema": "v10.2.30_v3_1_sensitivity_variant_registry_v1",
        "rows": variant_registry,
    })

    matrix = []
    output_columns = ["developed_da_dN_m_per_cycle", "median_developed_radius_m",
        "median_developed_shielding_MPa_sqrt_m", "median_developed_mobile_m2",
        "median_developed_retained_m2", "mean_accepted_to_proposed_ratio"]
    for parent in protocol["mechanism_representatives"]:
        for direction in names:
            minus = frame[(frame.parent_candidate_id == parent) & frame.variant_id.str.endswith(f"{direction}__minus")].sort_values("Kmax_MPa_sqrt_m")
            plus = frame[(frame.parent_candidate_id == parent) & frame.variant_id.str.endswith(f"{direction}__plus")].sort_values("Kmax_MPa_sqrt_m")
            for (_, lo), (_, hi) in zip(minus.iterrows(), plus.iterrows()):
                for output in output_columns:
                    scale = max(abs(float(lo[output])), abs(float(hi[output])), 1e-300)
                    derivative = (float(hi[output]) - float(lo[output])) / (2.0 * scale)
                    matrix.append({"parent_candidate_id": parent, "coordinate": direction,
                        "Kmax_MPa_sqrt_m": lo.Kmax_MPa_sqrt_m, "output": output,
                        "dimensionless_symmetric_sensitivity": derivative,
                        "minus_value": lo[output], "plus_value": hi[output],
                        "paired_seed": protocol["sensitivity"]["paired_seed"],
                        "status": "EVALUATED_NATIVE_TRAJECTORIES"})
            # Physical slope across low-middle and middle-high diagnostic loads.
            for j in range(2):
                slo = math.log(minus.developed_da_dN_m_per_cycle.iloc[j+1] / minus.developed_da_dN_m_per_cycle.iloc[j]) / math.log(minus.Kmax_MPa_sqrt_m.iloc[j+1] / minus.Kmax_MPa_sqrt_m.iloc[j])
                shi = math.log(plus.developed_da_dN_m_per_cycle.iloc[j+1] / plus.developed_da_dN_m_per_cycle.iloc[j]) / math.log(plus.Kmax_MPa_sqrt_m.iloc[j+1] / plus.Kmax_MPa_sqrt_m.iloc[j])
                matrix.append({"parent_candidate_id": parent, "coordinate": direction,
                    "Kmax_MPa_sqrt_m": math.sqrt(minus.Kmax_MPa_sqrt_m.iloc[j]*minus.Kmax_MPa_sqrt_m.iloc[j+1]),
                    "output": f"developed_secant_slope_{j}",
                    "dimensionless_symmetric_sensitivity": (shi-slo)/2.0,
                    "minus_value": slo, "plus_value": shi,
                    "paired_seed": protocol["sensitivity"]["paired_seed"],
                    "status": "EVALUATED_NATIVE_TRAJECTORIES"})
    for fixed in protocol["sensitivity"]["fixed_coordinates"]:
        matrix.append({"parent_candidate_id": "ALL", "coordinate": fixed,
            "Kmax_MPa_sqrt_m": math.nan, "output": "NOT_EVALUATED",
            "dimensionless_symmetric_sensitivity": math.nan,
            "minus_value": math.nan, "plus_value": math.nan,
            "paired_seed": protocol["sensitivity"]["paired_seed"],
            "status": "FIXED_BY_PROTOCOL_NOT_CALLED_INSENSITIVE"})
    pd.DataFrame(matrix).to_csv(a.out_root / "dimensionless_coupled_sensitivity_matrix.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
