#!/usr/bin/env python3
"""Physics-informed Paris continuation and candidate-A cleavage diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammainc, gammaln

KB_EV_PER_K = 8.617333262145e-5
CLEAVAGE = (
    "cleave_G00_eV", "cleave_sigc0_GPa", "cleave_exp_a",
    "cleave_exp_n", "cleave_floor_frac",
)
A_ID = "v914_endurance_knee_0462"
A_LOADS = np.asarray([10.24, 10.7388310162, 11.7972560356, 12.96])


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, data: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in data for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(data)


def finite(value: Any) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else math.nan
    except (TypeError, ValueError):
        return math.nan


def transform(value: float, mode: str) -> float:
    return math.log10(value) if mode == "log10" else value


def inverse(value: float, mode: str) -> float:
    return 10.0**value if mode == "log10" else value


def log_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log10(x), np.log10(np.maximum(y, 1.0e-300)), 1)[0])


def surface(row: dict[str, Any], delta_k: np.ndarray, common: dict[str, Any], hits: float = 3.0) -> dict[str, np.ndarray]:
    temperature, R = 300.0, 0.1
    g0 = finite(row["cleave_G00_eV"]) + finite(row.get("cleave_gT_eV_per_K", 0.0)) * (temperature - finite(row.get("Tref_K", 300.0)))
    sigc = (finite(row["cleave_sigc0_GPa"]) + finite(row.get("cleave_sT_GPa_per_K", 0.0)) * (temperature - finite(row.get("Tref_K", 300.0)))) * 1.0e9
    floor = min(0.95 * g0, max(1.0e-4, finite(row["cleave_floor_frac"]) * g0))
    radius = finite(common.get("r0_m", 1.0e-6))
    sigma = delta_k / (1.0 - R) * 1.0e6 / math.sqrt(2.0 * math.pi * radius)
    a, n = finite(row["cleave_exp_a"]), finite(row["cleave_exp_n"])
    z = a * (sigma / sigc) ** n
    decay = np.exp(-z)
    barrier = floor + (g0 - floor) * decay
    dG_dsigma = -(g0 - floor) * decay * n * z / sigma
    raw = finite(common.get("cleavage_nu0_s", 1.0e12)) * np.exp(-barrier / (KB_EV_PER_K * temperature))
    tau = finite(common.get("cleavage_correlation_time_s", 1.0e-6))
    exposure = np.minimum(raw * tau, 1.0e12)
    production = raw if hits <= 1.0 + 1.0e-12 else gammainc(hits, exposure) / tau
    if hits <= 1.0 + 1.0e-12:
        amplification = np.ones_like(raw)
    else:
        p = np.maximum(gammainc(hits, exposure), 1.0e-300)
        amplification = np.exp(hits * np.log(np.maximum(exposure, 1.0e-300)) - exposure - gammaln(hits) - np.log(p))
    return {
        "stress_GPa": sigma / 1.0e9, "Gstar_eV": barrier, "dG_dsigma_eV_Pa": dG_dsigma,
        "raw_rate_s": raw, "production_rate_s": production, "exposure": exposure,
        "production_over_raw": production / np.maximum(raw, 1.0e-300),
        "dln_production_dln_raw": amplification,
    }


def perturbed(row: dict[str, Any], field: str, value: float, bound: dict[str, Any]) -> dict[str, Any]:
    result = dict(row); result[field] = value
    lo, hi = finite(bound["lower"]), finite(bound["upper"])
    u = (transform(value, bound["transform"]) - transform(lo, bound["transform"])) / (
        transform(hi, bound["transform"]) - transform(lo, bound["transform"])
    )
    result[f"u__{field}"] = u
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--bounds", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--causal-decomposition", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    registry = rows(args.registry); arow = next(row for row in registry if row["candidate_id"] == A_ID)
    bounds_payload = json.loads(args.bounds.read_text())
    bounds = {item["field"]: item for item in bounds_payload["historical_bounds_in_generator_order"]}
    physics_payload = json.loads(args.physics.read_text()); common = physics_payload.get("common_physics", physics_payload)

    dense = np.geomspace(A_LOADS.min(), A_LOADS.max(), 200)
    base = surface(arow, dense, common, 3.0)
    base_raw_m = log_slope(dense, base["raw_rate_s"])
    bound_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = [{
        **arow, "source_candidate_id": A_ID, "paris_design_role": "CLEAVAGE_EXTENSION_OAT_REFERENCE",
        "cleavage_extension_parameter": "reference", "cleavage_extension_stage": "historical_A",
        "barrier_surface_valid": True, "diagnostic_raw_local_m": base_raw_m,
        "diagnostic_three_hit_local_m": log_slope(dense, base["production_rate_s"]),
    }]
    oat: list[dict[str, Any]] = [{
        "candidate_id": A_ID, "parameter": "reference", "role": "A_REFERENCE",
        "parameter_value": "", "developed_m": 20.565928009211895,
        "C": 4.879719067443904e-30, "R2": 0.9411906931337134,
        "feasibility": "MEASURED", "diagnostic_raw_local_m": base_raw_m,
        "diagnostic_three_hit_local_m": log_slope(dense, base["production_rate_s"]),
        "barrier_surface_valid": True,
    }]
    for field in CLEAVAGE:
        bound = bounds[field]; lo, hi = finite(bound["lower"]), finite(bound["upper"])
        value = finite(arow[field]); tlo, thi, tv = (transform(x, bound["transform"]) for x in (lo, hi, value))
        span = thi - tlo
        eps = 0.01 * span
        minus = max(tlo, tv - eps); plus = min(thi, tv + eps)
        m_minus = log_slope(dense, surface(perturbed(arow, field, inverse(minus, bound["transform"]), bound), dense, common)["raw_rate_s"])
        m_plus = log_slope(dense, surface(perturbed(arow, field, inverse(plus, bound["transform"]), bound), dense, common)["raw_rate_s"])
        flatten = "lower" if m_minus < m_plus else "higher"
        shell_t = tlo - 0.25 * span if flatten == "lower" else thi + 0.25 * span
        shell2_t = tlo - 0.50 * span if flatten == "lower" else thi + 0.50 * span
        shell = inverse(shell_t, bound["transform"]); shell2 = inverse(shell2_t, bound["transform"])
        if field == "cleave_floor_frac":
            shell = min(hi + 0.25 * (hi - lo), 0.90)
            shell2 = min(hi + 0.50 * (hi - lo), 0.90)
        if field in {"cleave_G00_eV", "cleave_sigc0_GPa"}:
            # A 25% linear-range extension crosses zero for these two old
            # bounds. Retain a finite, positive shell rather than admitting a
            # mathematically clipped near-zero activation scale.
            shell = max(shell, 0.75 * lo); shell2 = max(shell2, 0.50 * lo)
        if field in {"cleave_exp_a", "cleave_exp_n"}:
            shell = max(shell, 1.0e-8); shell2 = max(shell2, 1.0e-8)
        bound_rows.append({
            "parameter": field, "A_value": value, "historical_lower": lo, "historical_upper": hi,
            "transform": bound["transform"], "A_normalized": (tv-tlo)/span,
            "distance_to_lower_normalized": (tv-tlo)/span, "distance_to_upper_normalized": (thi-tv)/span,
            "causal_flattening_direction": flatten, "A_raw_local_m": base_raw_m,
            "minus_1pct_range_raw_m": m_minus, "plus_1pct_range_raw_m": m_plus,
            "shell1_extended_lower": shell if flatten == "lower" else lo,
            "shell1_extended_upper": hi if flatten == "lower" else shell,
            "shell2_diagnostic_value_not_yet_authorized": shell2,
        })
        center_t = 0.5 * (tlo + thi)
        for role, proposed_t in (("INWARD_REFERENCE", 0.5*(tv+center_t)), ("OUTWARD_SHELL1", shell_t)):
            proposed = shell if role == "OUTWARD_SHELL1" else inverse(proposed_t, bound["transform"])
            if field == "cleave_floor_frac": proposed = min(max(proposed, 0.0), 0.90)
            candidate = perturbed(arow, field, proposed, bound)
            candidate_id = f"v914_A_cleave_oat_{field.removeprefix('cleave_')}_{role.lower()}"
            candidate.update({
                "candidate_id": candidate_id, "source_candidate_id": A_ID,
                "paris_design_role": f"CLEAVAGE_EXTENSION_OAT_{role}",
                "cleavage_extension_parameter": field, "cleavage_extension_stage": "shell1",
            })
            diagnostic = surface(candidate, dense, common)
            valid = bool(
                finite(candidate["cleave_G00_eV"]) > 0.0
                and finite(candidate["cleave_sigc0_GPa"]) > 0.0
                and finite(candidate["cleave_exp_a"]) > 0.0
                and finite(candidate["cleave_exp_n"]) > 0.0
                and 0.0 <= finite(candidate["cleave_floor_frac"]) < 0.95
                and np.all(np.isfinite(diagnostic["Gstar_eV"]))
                and np.all(diagnostic["Gstar_eV"] > 0.0)
                and np.all(np.diff(diagnostic["Gstar_eV"]) < 0.0)
            )
            candidate["barrier_surface_valid"] = valid
            candidate["diagnostic_raw_local_m"] = log_slope(dense, diagnostic["raw_rate_s"])
            candidate["diagnostic_three_hit_local_m"] = log_slope(dense, diagnostic["production_rate_s"])
            candidates.append(candidate)
            oat.append({
                "candidate_id": candidate_id, "parameter": field, "role": role,
                "parameter_value": proposed, "developed_m": "", "C": "", "R2": "",
                "feasibility": "NOT_RUN", "diagnostic_raw_local_m": candidate["diagnostic_raw_local_m"],
                "diagnostic_three_hit_local_m": candidate["diagnostic_three_hit_local_m"],
                "barrier_surface_valid": valid,
            })
    write_csv(args.out / "historical_vs_extended_cleavage_bounds.csv", bound_rows)
    write_csv(args.out / "cleavage_extension_candidates.csv", candidates)
    write_csv(args.out / "cleavage_extension_OAT.csv", oat)

    decomposition = []
    for load in A_LOADS:
        for hits in (1.0, 2.0, 3.0):
            item = surface(arow, np.asarray([load]), common, hits)
            decomposition.append({
                "DeltaK_MPa_sqrt_m": load, "n_hits_diagnostic": int(hits),
                "tau_c_s": finite(common["cleavage_correlation_time_s"]),
                "Gstar_eV": item["Gstar_eV"][0], "dG_dsigma_eV_Pa": item["dG_dsigma_eV_Pa"][0],
                "Gamma_raw_s": item["raw_rate_s"][0], "Gamma_prod_s": item["production_rate_s"][0],
                "x_Gamma_raw_tau_c": item["exposure"][0], "Gamma_prod_over_Gamma_raw": item["production_over_raw"][0],
                "dln_Gamma_prod_dln_Gamma_raw": item["dln_production_dln_raw"][0],
                "low_exposure_asymptote_Gamma3_s": item["raw_rate_s"][0]**3 * finite(common["cleavage_correlation_time_s"])**2 / 6.0,
            })
    write_csv(args.out / "A_raw_vs_threehit_slope_decomposition.csv", decomposition)
    causal = rows(args.causal_decomposition)[0]
    state_rows = [{
        "scenario": "A_only_DeltaK_changes_internal_state_frozen",
        "m": finite(causal["intrinsic_barrier_stress_sensitivity_contribution"]),
        "method": "analytical finite-tip raw activation surface",
    }, {
        "scenario": "B_direct_plus_three_hit_same_state",
        "m": finite(causal["intrinsic_barrier_stress_sensitivity_contribution"]) + finite(causal["three_hit_transformation_contribution"]),
        "method": "non-feedback three-hit transform of identical raw rates",
    }, {
        "scenario": "C_with_internal_state_and_waiting",
        "m": finite(causal["measured_full_m"]) - finite(causal["event_size_variation_contribution"]),
        "method": "saved developed event waits with normal v7 state/history; event-size contribution removed",
    }, {
        "scenario": "D_full",
        "m": finite(causal["measured_full_m"]),
        "method": "saved developed trajectories",
    }]
    write_csv(args.out / "A_state_waiting_slope_decomposition.csv", state_rows)

    fig, ax = plt.subplots(figsize=(7.2, 5.2)); ax.plot(dense, base["Gstar_eV"])
    ax.set(xlabel="DeltaK (MPa sqrt(m))", ylabel="G* (eV)", title="Candidate A cleavage barrier"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(args.out/"Gstar_vs_DeltaK_A.png",dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 5.2)); ax.semilogy(dense,base["raw_rate_s"],label="raw"); ax.semilogy(dense,base["production_rate_s"],label="three-hit production")
    ax.set(xlabel="DeltaK (MPa sqrt(m))",ylabel="hazard rate (1/s)",title="Candidate A raw and production hazard"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(args.out/"raw_vs_production_hazard_A.png",dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2,1,figsize=(7.2,7),sharex=True)
    axes[0].semilogy(dense,base["production_over_raw"]); axes[0].set(ylabel="Gamma_prod / Gamma_raw")
    axes[1].plot(dense,base["dln_production_dln_raw"]); axes[1].axhline(3,color="k",ls="--",lw=1); axes[1].set(xlabel="DeltaK (MPa sqrt(m))",ylabel="d ln Gamma_prod / d ln Gamma_raw")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(args.out/"three_hit_amplification_vs_DeltaK.png",dpi=180); plt.close(fig)

    fits = rows(args.fits); measured = [row for row in fits if math.isfinite(finite(row.get("m")))]
    stages = [
        {"search_stage":"historical_initial","best_m":min(finite(r["m"]) for r in measured if r["candidate_id"] in {A_ID,"v914_endurance_knee_0849","v914_endurance_knee_0917"})},
        {"search_stage":"historical_completed_budget","best_m":min(finite(r["m"]) for r in measured)},
        {"search_stage":"physics_informed_continuation","best_m":""},
        {"search_stage":"cleavage_extension","best_m":""},
    ]
    write_csv(args.out/"best_m_vs_search_iteration.csv",stages)
    fig,ax=plt.subplots(figsize=(7.2,5.2)); known=[r for r in stages if finite(r["best_m"])>0]; ax.plot(range(len(known)),[finite(r["best_m"]) for r in known],"o-"); ax.set_xticks(range(len(known)),[r["search_stage"] for r in known],rotation=20,ha="right"); ax.set(ylabel="best observed developed m",title="Paris-slope search progress"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(args.out/"best_m_vs_search_iteration.png",dpi=180); plt.close(fig)
    print(f"V914_PHYSICS_INFORMED_DIAGNOSTICS_COMPLETE out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
