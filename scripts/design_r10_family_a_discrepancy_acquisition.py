#!/usr/bin/env python3
"""Rank strict-Family-A Weak-T acquisition rows from the completed V2 benchmark.

The Family-B V2-P25 result supplies an empirical A0-to-current transfer only.
Every proposed row is rebuilt directly from the original Weak-T Family-A row.
The scan changes reference-state barrier levels in the existing EXP_floor law;
all temperature slopes, activation entropies, transport laws, and mechanics are
copied verbatim.  Consequently this is an acquisition ranking, not a Family-B
promotion or an independent endpoint fit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.material_manifest import ExpFloorBarrier
from scripts.design_r10_joint_level_iteration import action_curve, frozen_onset


BENCHMARK = "P25_TJBSV2_S_026704"
PARENT = "v913_zeroD_sobol_0129902"
DELTAS_EV = (-0.12, -0.06, 0.0, 0.06, 0.12)
TEMPERATURES_K = (300, 450, 600, 750, 900, 1050, 1200)


def canonical_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def developed_rate(path: Path) -> tuple[float, float]:
    result = json.loads(path.read_text())
    events = result["events"]
    left = next((event for event in events if float(event["extension_m"]) >= 20e-6), None)
    right = next((event for event in events if float(event["extension_m"]) >= 60e-6), None)
    if left is None or right is None:
        raise RuntimeError(f"benchmark is not developed through 60 um: {path}")
    rate = (float(right["extension_m"]) - float(left["extension_m"])) / (
        float(right["cycles"]) - float(left["cycles"])
    )
    return float(result["DeltaK_MPa_sqrt_m"]), rate


def surface(row: dict, prefix: str, nu0: float) -> ExpFloorBarrier:
    return ExpFloorBarrier(
        G00_eV=float(row[f"{prefix}_G00_eV"]),
        gT_eV_per_K=float(row[f"{prefix}_gT_eV_per_K"]),
        sigc0_Pa=1e9 * float(row[f"{prefix}_sigc0_GPa"]),
        sT_Pa_per_K=1e9 * float(row[f"{prefix}_sT_GPa_per_K"]),
        alpha=float(row[f"{prefix}_exp_a"]),
        exponent=float(row[f"{prefix}_exp_n"]),
        floor_fraction=float(row[f"{prefix}_floor_frac"]),
        Tref_K=float(row["Tref_K"]),
        attempt_frequency_s=nu0,
    )


def reference_energy_shift(row: dict, prefix: str, delta_eV: float) -> dict:
    child = dict(row)
    base = surface(row, prefix, 1e12 if prefix == "cleave" else 1e11)
    G300 = base.G00_eV + base.gT_eV_per_K * (300.0 - base.Tref_K)
    if G300 + delta_eV <= 0.0:
        raise ValueError("reference-state shift leaves the positive barrier domain")
    child[f"{prefix}_G00_eV"] = float(base.G00_eV + delta_eV)
    return child


def constant_300k_opening_shift(row: dict, delta_eV: float) -> dict:
    """Shift the whole 300 K opening surface using existing G00/floor fields."""
    child = dict(row)
    base = surface(row, "cleave", 1e12)
    G300 = base.G00_eV + base.gT_eV_per_K * (300.0 - base.Tref_K)
    floor300 = base.floor_fraction * G300
    child["cleave_G00_eV"] = float(base.G00_eV + delta_eV)
    child["cleave_floor_frac"] = float((floor300 + delta_eV) / (G300 + delta_eV))
    return child


def log_interp_action(screen: list[dict], delta_k: np.ndarray) -> np.ndarray:
    rows = sorted((r for r in screen if r["candidate_id"] == BENCHMARK),
                  key=lambda r: float(r["DeltaK"]))
    x = np.log([float(r["DeltaK"]) for r in rows])
    y = np.log([float(r["action_per_cycle"]) for r in rows])
    return np.exp(np.interp(np.log(delta_k), x, y))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-registry", type=Path, required=True)
    parser.add_argument("--analytical-screen", type=Path, required=True)
    parser.add_argument("--projection-frontier", type=Path, required=True)
    parser.add_argument("--low-result", type=Path, required=True)
    parser.add_argument("--mid-result", type=Path, required=True)
    parser.add_argument("--high-result", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)

    benchmark = [developed_rate(path) for path in
                 (args.low_result, args.mid_result, args.high_result)]
    benchmark.sort()
    delta_k = np.array([point[0] for point in benchmark])
    current_rate = np.array([point[1] for point in benchmark])
    a0_action = log_interp_action(read_csv(args.analytical_screen), delta_k)
    transfer_slope, transfer_intercept = np.polyfit(np.log(a0_action), np.log(current_rate), 1)
    fitted = np.exp(transfer_intercept) * a0_action ** transfer_slope
    transfer_rms = float(np.sqrt(np.mean((np.log10(fitted) - np.log10(current_rate)) ** 2)))

    parent = next(row for row in read_csv(args.original_registry) if row["candidate_id"] == PARENT)
    protected = (
        "cleave_gT_eV_per_K", "cleave_sT_GPa_per_K", "emit_gT_eV_per_K",
        "emit_sT_GPa_per_K", "peierls_activation_entropy_kB",
        "taylor_activation_entropy_kB", "peierls_H0_eV", "taylor_H0_eV",
    )
    grid = np.geomspace(3.0, 30.0, 81)
    proposals = []
    for lane, prefix in (("opening_local", "cleave"), ("dislocation_barrier_local", "emit")):
        for delta in DELTAS_EV:
            proposals.append((lane, prefix, delta, reference_energy_shift(parent, prefix, delta), "original_local_scan"))

    # Reuse the P25/P40 opening design at 300 K only.  The withdrawn projected
    # hot anchor is explicitly discarded: G00/sigc0 are recalculated so that
    # the desired 300 K surface is paired with the original Weak-T gT/sT.
    projections = [row for row in read_csv(args.projection_frontier)
                   if row["parent_candidate_id"] == PARENT and row["target"] in ("P25", "P40")]
    for projection in projections:
        projected = json.loads(projection["complete_parameter_vector"])
        target = projection["target"]
        Tref = float(parent["Tref_K"])
        desired_G300 = float(projected["cleave_G00_eV"]) + float(projected["cleave_gT_eV_per_K"]) * (300.0 - Tref)
        desired_sigc300 = float(projected["cleave_sigc0_GPa"]) + float(projected["cleave_sT_GPa_per_K"]) * (300.0 - Tref)
        base = dict(parent)
        base["cleave_G00_eV"] = desired_G300 - float(parent["cleave_gT_eV_per_K"]) * (300.0 - Tref)
        base["cleave_sigc0_GPa"] = desired_sigc300 - float(parent["cleave_sT_GPa_per_K"]) * (300.0 - Tref)
        for key in ("cleave_exp_a", "cleave_exp_n", "cleave_floor_frac"):
            base[key] = float(projected[key])
        opening_shift = 0.036 if target == "P40" else 0.0
        base = constant_300k_opening_shift(base, opening_shift)
        proposals.append((f"opening_{target}_reference_projection", "cleave", opening_shift, base,
                          f"{projection['candidate_id']}:300K_surface_only"))
        for delta in (-0.06, 0.06):
            child = reference_energy_shift(base, "emit", delta)
            proposals.append((f"coupled_{target}_dislocation_barrier", "emit", delta, child,
                              f"{projection['candidate_id']}:300K_surface_only"))

    records = []
    candidates = []
    for lane, prefix, delta, child, design_source in proposals:
            target = "P25" if "P25" in lane else "P40" if "P40" in lane else "LOCAL"
            tag = f"M{abs(int(round(delta * 1000))):03d}" if delta < 0 else f"P{int(round(delta * 1000)):03d}"
            mechanism = "OPEN" if prefix == "cleave" else "EMIT"
            cid = f"R10A_WEAKT_{target}_{mechanism}_{tag}MEV_V1"
            child["candidate_id"] = cid
            opening = surface(child, "cleave", 1e12)
            q = 3.0
            tau = 1e-6
            action = action_curve(opening, q, tau, grid / 0.9)
            corrected = np.exp(transfer_intercept) * action ** transfer_slope
            corrected_slope = float(np.polyfit(np.log(grid), np.log(corrected), 1)[0])
            corrected_width = float(np.log10(corrected.max() / corrected.min()))
            onsets = {str(T): frozen_onset(opening, q, tau, float(T)) for T in TEMPERATURES_K}
            emission = surface(child, "emit", 1e11)
            probes = {str(stress): float(emission.rate(stress * 1e9, 300.0))
                      for stress in (3.0, 5.0, 8.0)}
            changed = {key: {"parent": parent[key], "offspring": child[key]}
                       for key in child if parent.get(key) != child[key] and key != "candidate_id"}
            protected_equal = all(str(child[key]) == str(parent[key]) for key in protected)
            score = (
                abs(corrected_slope - 4.0)
                + max(0.0, 2.0 - corrected_width)
                + (0.0 if onsets["300"] is not None and 5.0 <= onsets["300"] <= 30.0 else 10.0)
            )
            record = {
                "candidate_id": cid,
                "parent_candidate_id": PARENT,
                "family": "A_ORIGINAL_SINGLE_EXP_FLOOR",
                "target": target,
                "cleavage_hits": q,
                "cleavage_tau_s": tau,
                "lane": lane,
                "design_source": design_source,
                "delta_reference_level_eV": delta,
                "complete_row_sha256": canonical_hash(child),
                "corrected_A0_global_slope": corrected_slope,
                "corrected_A0_rate_width_decades": corrected_width,
                "frozen_Kinit_by_T_json": json.dumps(onsets, sort_keys=True),
                "emission_rate_probes_300K_json": json.dumps(probes, sort_keys=True),
                "acquisition_score": score,
                "protected_thermal_and_entropy_fields_equal": protected_equal,
                "changed_fields_json": json.dumps(changed, sort_keys=True),
                "complete_parameter_vector": json.dumps(child, sort_keys=True),
                "status": "ANALYTICAL_ACQUISITION_ONLY",
            }
            records.append(record)
            candidates.append(child)

    ranked = sorted(records, key=lambda row: (float(row["acquisition_score"]), row["lane"],
                                               abs(float(row["delta_reference_level_eV"]))))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    write_csv(args.out_root / "strict_family_a_acquisition.csv", ranked)
    write_csv(args.out_root / "strict_family_a_candidate_rows.csv", [{key: row[key] for key in parent} for row in candidates])

    discrepancy = [{
        "DeltaK_MPa_sqrt_m": float(dk),
        "A0_action_per_cycle": float(a0),
        "developed_20_60_rate_m_per_cycle": float(rate),
        "fitted_rate_m_per_cycle": float(pred),
        "log10_current_over_A0": float(math.log10(rate / a0)),
    } for dk, a0, rate, pred in zip(delta_k, a0_action, current_rate, fitted)]
    write_csv(args.out_root / "family_b_A0_to_current_discrepancy.csv", discrepancy)
    decision = {
        "schema": "r10.family-a-discrepancy-acquisition/1",
        "benchmark": {
            "candidate_id": BENCHMARK,
            "family": "B_INHERITED_V2_GUARD_ENTROPY_EXTENSION",
            "role": "empirical_acquisition_transfer_only",
            "fracture_and_2D_receipts": "historical",
            "points": discrepancy,
        },
        "transfer": {
            "formula": "developed_rate=exp(intercept)*A0_action**slope",
            "slope": float(transfer_slope),
            "intercept_natural_log": float(transfer_intercept),
            "rms_log10_rate": transfer_rms,
            "scope": "ranking surrogate; not a constitutive law or qualification result",
        },
        "primary_objective": "strict-Family-A complete EXP_floor material evaluated jointly in fracture and fatigue",
        "candidate_count": len(ranked),
        "ranked_candidate_ids": [row["candidate_id"] for row in ranked],
        "recommended_opening": next(row["candidate_id"] for row in ranked
                                    if row["lane"].startswith("opening_P40")),
        "recommended_dislocation_bracket": [row["candidate_id"] for row in ranked
                                             if row["lane"] == "coupled_P40_dislocation_barrier"],
        "constraints": [
            "all candidates descend directly from the original Weak-T Family-A row",
            "only existing EXP_floor opening-shape/reference coordinates and emission reference energy are varied",
            "original gT/sT coefficients and Peierls/Taylor activation entropies are unchanged",
            "no cold/hot endpoint is selected or fitted independently",
            "P25/P40 projected rows contribute their 300 K opening shape only; their projected thermal slopes are discarded",
            "Family-B benchmark does not replace the primary strict-Family-A objective",
            "full-state fracture and developed fatigue are required before promotion; 2-D follows only after 1-D gates",
        ],
    }
    (args.out_root / "acquisition_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
