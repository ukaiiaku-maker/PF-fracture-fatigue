#!/usr/bin/env python3
"""Existing-data v9.13 barrier/entropy/temperature-fracture analysis.

This is deliberately analysis-only.  It reads immutable v9.13 run artifacts and
calls the historical production ``ExpFloorSurface`` implementation directly.
Diagnostic entropy-removal evaluations never evolve or replace a fracture run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special, stats
from scipy.cluster.vq import kmeans2


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_13_dbtt_temperature_shelf")
HISTORICAL_SHA = "559425321b9a8739f32788322d8a1c2af8abad73"
GITHUB_REPOSITORY = "ukaiiaku-maker/Arrhenius_FEM_CZM_MPZ"
KB_EV_K = 8.617333262145e-5
NU_C = 1.0e12
NU_E = 1.0e11
MULTIHIT_M = 3.0
MULTIHIT_TAU_S = 1.0e-6
SIGMA_REF_PA = 5.0e9
CANONICAL = {
    "v913_zeroD_sobol_0242980": "Peak-T",
    "v913_zeroD_sobol_0202500": "DBTT",
    "v913_zeroD_sobol_0129902": "weak-T",
    "v913_zeroD_sobol_0077080": "ceramic-like",
}
CANONICAL_OPTIONS = {
    "v913_zeroD_sobol_0242980": "v913_paper_peak01_0242980_persistent_sites",
    "v913_zeroD_sobol_0202500": "v913_paper_dbtt01_0202500_persistent_sites",
    "v913_zeroD_sobol_0129902": "v913_paper_weakT01_0129902_persistent_sites",
    "v913_zeroD_sobol_0077080": "v913_paper_ceramic01_0077080_persistent_sites",
}
ACTIVE_FIELDS = [
    "Tref_K", "cleave_G00_eV", "cleave_gT_eV_per_K", "cleave_sigc0_GPa",
    "cleave_sT_GPa_per_K", "cleave_exp_a", "cleave_exp_n", "cleave_floor_frac",
    "emit_G00_eV", "emit_gT_eV_per_K", "emit_sigc0_GPa", "emit_sT_GPa_per_K",
    "emit_exp_a", "emit_exp_n", "emit_floor_frac", "peierls_H0_eV",
    "peierls_activation_entropy_kB", "peierls_exp_a", "peierls_exp_n", "peierls_nu0_s",
    "taylor_H0_eV", "taylor_activation_entropy_kB", "taylor_exp_a", "taylor_exp_n",
    "taylor_nu0_s", "rho_source0_m2", "taylor_corr_rho_c_m2", "taylor_corr_scale", "c_blunt",
]
COLORS = {"Peak-T": "#F59E0B", "DBTT": "#3B82F6", "weak-T": "#8B5CF6",
          "ceramic-like": "#64748B", "other": "#94A3B8"}


def git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def finite(value, default=np.nan) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float(default)
    except (TypeError, ValueError):
        return float(default)


def load_production_types(source: Path):
    """Load the exact historical class without importing this checkout's package."""
    path = source / "arrhenius_fracture/emergent_gnd_types_v912.py"
    expected = "a04e995507e062fe3e8af0691165a59c55ab2d6ea6b5747cd3a32c44ea9c06c4"
    if sha256(path) != expected:
        raise RuntimeError("historical ExpFloorSurface source hash mismatch")
    spec = importlib.util.spec_from_file_location("_v913_production_types", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ExpFloorSurface, module.PTMechanism


def normalized_params(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [c.removeprefix("x_raw__") for c in out.columns]
    return out


def fingerprint(row: pd.Series) -> str:
    payload = {k: finite(row.get(k)) for k in ACTIVE_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_canonical_selection(candidates: pd.DataFrame, selection_path: Path, registry_path: Path) -> str:
    selection=json.loads(selection_path.read_text()); registry=pd.read_csv(registry_path)
    selected=selection.get("primary_candidates",[])
    if [x.get("candidate_id") for x in selected] != list(CANONICAL):
        raise RuntimeError("committed canonical selection IDs/order mismatch")
    rows=[]
    for cid,option in CANONICAL_OPTIONS.items():
        c=candidates[candidates.candidate_id.eq(cid)]
        r=registry[(registry.candidate_id.eq(cid))&registry.option_key.eq(option)]
        if len(c)!=1 or len(r)!=1: raise RuntimeError(f"canonical row not unique: {cid}")
        payload={"candidate_id":cid}
        for field in ACTIVE_FIELDS:
            a=finite(c.iloc[0][field]); b=finite(r.iloc[0][field])
            if a != b: raise RuntimeError(f"canonical active parameter mismatch {cid}:{field}: {a} != {b}")
            payload[field]=a
        rows.append(payload)
    value=hashlib.sha256(json.dumps(sorted(rows,key=lambda x:x["candidate_id"]),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    # The committed selection explicitly notes that its source fingerprint used
    # a different historical serialization.  Exact field-by-field equality is
    # the authoritative transfer check; retain both hashes rather than silently
    # treating serialization inequality as parameter inequality.
    return value


def input_paths(source: Path) -> dict[str, Path]:
    broad = source / "runs/v9_13_zeroD_promoted_1d_384_50um_v2/one_d_screen"
    wc = source / "runs/v9_13_weakT_ceramic_search_5T_100um_v1"
    return {
        "broad_features": broad / "candidate_pool_features.csv",
        "broad_cases": broad / "case_results_checkpoint.csv",
        "broad_events": broad / "R_curve_events.csv",
        "broad_ranked": broad / "ranked_candidates.csv",
        "broad_contract": broad / "run_contract.json",
        "broad_manifest": broad / "search_manifest.json",
        "wc_features": wc / "one_d_100um/candidate_pool_features.csv",
        "wc_cases": wc / "one_d_100um/case_results_checkpoint.csv",
        "wc_events": wc / "one_d_100um/R_curve_events.csv",
        "wc_metrics": wc / "analysis_100um/candidate_metrics.csv",
        "wc_contract": wc / "one_d_100um/run_contract.json",
        "wc_manifest": wc / "one_d_100um/search_manifest.json",
        "loading_map": source / "runs/v9_13_v10222_rcurve_targets_v1/v10_2_22_rcurve_loading_map.json",
        "selection": REPO / "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_selection.json",
        "registry": REPO / "arrhenius_fracture/data/materials/v10_2_27_paper_four_class_registry.csv",
        "legacy_2d_note": source / "arrhenius_fracture/RUN_2D_NOTES.md",
    }


def load_population(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    p = input_paths(source)
    loading=json.loads(p["loading_map"].read_text())
    broad = normalized_params(pd.read_csv(p["broad_features"]))
    broad["source_registry"] = "v9_13_zeroD_promoted_1d_384_50um_v2"
    broad["response_target_um"] = 50.0
    weak = normalized_params(pd.read_csv(p["wc_features"]))
    weak["source_registry"] = "v9_13_weakT_ceramic_search_5T_100um_v1"
    weak["response_target_um"] = 100.0
    candidates = pd.concat([broad, weak], ignore_index=True).drop_duplicates("candidate_id", keep="last")
    candidates["parameter_fingerprint"] = candidates.apply(fingerprint, axis=1)
    candidates["simulation_git_sha"] = HISTORICAL_SHA
    candidates["simulation_sha_provenance"] = "INFERRED_BY_EXACT_RUN_CONTRACT_SOURCE_HASH_MATCH"
    candidates["github_repository"] = GITHUB_REPOSITORY
    candidates["historical_branch"] = "v9.13-long-extension-aligned-peak-campaign"
    candidates["canonical_family"] = candidates.candidate_id.map(CANONICAL)
    candidates["canonical_option_key"] = candidates.candidate_id.map(CANONICAL_OPTIONS)
    candidates["is_canonical_holdout"] = candidates.candidate_id.isin(CANONICAL)

    ranked = pd.read_csv(p["broad_ranked"])
    cls = pd.DataFrame({"candidate_id": ranked.candidate_id})
    direction = ranked.get("y__directional_dbtt_ge_threshold_1d", False)
    peak = ranked.get("y__peak_like_1d", False)
    cls["historical_response_class"] = np.where(peak, "Peak-like", np.where(direction, "DBTT-like", "other/intermediate"))
    wc_metrics = pd.read_csv(p["wc_metrics"])[["candidate_id", "target_class"]].rename(columns={"target_class": "historical_response_class"})
    cls = pd.concat([cls, wc_metrics], ignore_index=True).drop_duplicates("candidate_id", keep="last")
    candidates = candidates.merge(cls, on="candidate_id", how="left")
    candidates.loc[candidates.candidate_id.eq("v913_zeroD_sobol_0242980"), "historical_response_class"] = "Peak-T canonical"
    candidates.loc[candidates.candidate_id.eq("v913_zeroD_sobol_0202500"), "historical_response_class"] = "DBTT canonical"
    candidates.loc[candidates.candidate_id.eq("v913_zeroD_sobol_0129902"), "historical_response_class"] = "weak-T canonical"
    candidates.loc[candidates.candidate_id.eq("v913_zeroD_sobol_0077080"), "historical_response_class"] = "ceramic-like canonical"

    def read_cases(key: str, target: float, name: str) -> pd.DataFrame:
        q = pd.read_csv(p[key]); q["source_dataset"] = name; q["response_target_um"] = target
        q["authoritative_response_MPa_sqrt_m"] = q.K_checkpoint_MPa_sqrt_m
        q["simulation_git_sha"] = HISTORICAL_SHA
        q["loading_protocol"]="prescribed displacement increments from v10.2.22 RCurveLoadingMap"
        q["nominal_loading_increment_m"]=finite(loading["nominal_dU_m"])
        q["nominal_loading_increment_time_s"]=finite(loading["nominal_dt_s"])
        q["nominal_displacement_rate_m_per_s"]=finite(loading["nominal_dU_m"])/finite(loading["nominal_dt_s"])
        q["loading_map_seed"]=int(loading["seed"])
        q["loading_map_sha256"]=sha256(p["loading_map"])
        return q
    cases = pd.concat([
        read_cases("broad_cases", 50.0, "v9.13 broad promoted 384 x 10T, K50"),
        read_cases("wc_cases", 100.0, "v9.13 weak/ceramic 16 x 5T, K100"),
    ], ignore_index=True)
    cases = cases.drop_duplicates(["candidate_id", "temperature_K"], keep="last")

    def read_events(key: str, target: float, name: str) -> pd.DataFrame:
        q = pd.read_csv(p[key]); q["source_dataset"] = name; q["response_target_um"] = target
        q["simulation_git_sha"] = HISTORICAL_SHA
        return q
    events = pd.concat([
        read_events("broad_events", 50.0, "v9.13 broad promoted 384 x 10T"),
        read_events("wc_events", 100.0, "v9.13 weak/ceramic 16 x 5T"),
    ], ignore_index=True)
    events = events.drop_duplicates(["candidate_id", "temperature_K", "event_index"], keep="last")
    return candidates, cases, events, p


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[good], y[good], 1)[0]) if good.sum() >= 2 and np.ptp(x[good]) > 0 else np.nan


def crossing_temperature(T: np.ndarray, values: np.ndarray, level: float) -> float:
    for i in range(len(T) - 1):
        a, b = values[i] - level, values[i + 1] - level
        if a == 0: return float(T[i])
        if a * b <= 0 and b != a:
            return float(T[i] + (T[i + 1] - T[i]) * (-a) / (b - a))
    return np.nan


def response_descriptors(cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, normalized = [], []
    for cid, group in cases.groupby("candidate_id", sort=False):
        g = group[group.status.eq("complete")].sort_values("temperature_K")
        T = g.temperature_K.to_numpy(float); K = g.authoritative_response_MPa_sqrt_m.to_numpy(float)
        if len(g) < 3: continue
        deriv = np.gradient(K, T); curv = np.gradient(deriv, T)
        ipeak = int(np.argmax(K)); imin = int(np.argmin(K))
        low = T <= 900; mid = (T >= 900) & (T <= 1100); high = T >= 1100
        baseline = float(K[0]); K300 = float(K[np.argmin(abs(T - 300))]) if np.min(abs(T - 300)) < 1e-8 else np.nan
        peak_low = float(K[ipeak] - K[0]); peak_high = float(K[ipeak] - K[-1]); prominence = min(peak_low, peak_high)
        half = float(K[ipeak] - 0.5 * max(prominence, 0.0))
        left = crossing_temperature(T[:ipeak + 1], K[:ipeak + 1], half) if ipeak else np.nan
        right = crossing_temperature(T[ipeak:], K[ipeak:], half) if ipeak < len(T) - 1 else np.nan
        overall_rise = float(K[-1] - K[0])
        lo_level = float(K[0] + .2 * overall_rise); hi_level = float(K[0] + .8 * overall_rise)
        t20 = crossing_temperature(T, K, lo_level) if overall_rise > 0 else np.nan
        t80 = crossing_temperature(T, K, hi_level) if overall_rise > 0 else np.nan
        ipos = int(np.argmax(deriv))
        row = {
            "candidate_id": cid, "n_temperatures": len(T), "temperature_min_K": T.min(), "temperature_max_K": T.max(),
            "response_target_um": finite(g.response_target_um.iloc[0]), "K_300_MPa_sqrt_m": K300,
            "baseline_resistance_MPa_sqrt_m": baseline, "K_min_MPa_sqrt_m": K[imin], "K_max_MPa_sqrt_m": K[ipeak],
            "S_low_MPa_sqrt_m_per_K": linear_slope(T[low], K[low]),
            "S_mid_MPa_sqrt_m_per_K": linear_slope(T[mid], K[mid]),
            "S_high_MPa_sqrt_m_per_K": linear_slope(T[high], K[high]),
            "max_abs_thermal_slope_MPa_sqrt_m_per_K": float(np.max(abs(deriv))),
            "max_positive_thermal_slope_MPa_sqrt_m_per_K": float(np.max(deriv)),
            "max_negative_thermal_slope_MPa_sqrt_m_per_K": float(np.min(deriv)),
            "max_abs_curvature_MPa_sqrt_m_per_K2": float(np.max(abs(curv))),
            "total_resistance_span_MPa_sqrt_m": float(np.ptp(K)),
            "fractional_resistance_span": float(np.ptp(K) / max(abs(baseline), 1e-30)),
            "DBTT_temperature_K": float(T[ipos]) if deriv[ipos] > 0 else np.nan,
            "DBTT_width_K": float(t80 - t20) if np.isfinite(t20) and np.isfinite(t80) and t80 > t20 else np.nan,
            "DBTT_magnitude_MPa_sqrt_m": max(overall_rise, 0.0),
            "low_temperature_asymptote_MPa_sqrt_m": float(np.median(K[low])) if low.any() else np.nan,
            "high_temperature_asymptote_MPa_sqrt_m": float(np.median(K[high])) if high.any() else np.nan,
            "peak_temperature_K": float(T[ipeak]), "peak_height_vs_low_MPa_sqrt_m": peak_low,
            "peak_height_vs_high_MPa_sqrt_m": peak_high, "peak_prominence_MPa_sqrt_m": max(prominence, 0.0),
            "peak_width_K": float(right - left) if np.isfinite(left) and np.isfinite(right) and right > left else np.nan,
            "peak_curvature_MPa_sqrt_m_per_K2": float(curv[ipeak]),
            "weakT_max_deviation_from_mean_MPa_sqrt_m": float(np.max(abs(K - K.mean()))),
            "thermal_softening_slope_MPa_sqrt_m_per_K": linear_slope(T, K),
            "fractional_terminal_change": float((K[-1] - K[0]) / max(abs(K[0]), 1e-30)),
        }
        rows.append(row)
        for t, k, d, c in zip(T, K, deriv, curv):
            normalized.append({"candidate_id": cid, "temperature_K": t, "K_response_MPa_sqrt_m": k,
                "K_over_K300": k / K300 if np.isfinite(K300) else np.nan,
                "K_minus_K300_over_K300": (k - K300) / K300 if np.isfinite(K300) else np.nan,
                "K_over_first_available": k / baseline, "local_dK_dT": d, "local_d2K_dT2": c,
                "response_target_um": finite(g.response_target_um.iloc[0]), "source_dataset": g.source_dataset.iloc[0]})
    return pd.DataFrame(rows), pd.DataFrame(normalized)


def make_surface(row: pd.Series, prefix: str, ExpFloorSurface, *, zero_gT=False, zero_sT=False):
    return ExpFloorSurface(
        G00_eV=finite(row[f"{prefix}_G00_eV"]),
        gT_eV_per_K=0.0 if zero_gT else finite(row[f"{prefix}_gT_eV_per_K"]),
        sigc0_Pa=finite(row[f"{prefix}_sigc0_GPa"]) * 1e9,
        sT_Pa_per_K=0.0 if zero_sT else finite(row[f"{prefix}_sT_GPa_per_K"]) * 1e9,
        exp_a=finite(row[f"{prefix}_exp_a"]), exp_n=finite(row[f"{prefix}_exp_n"]),
        floor_fraction=finite(row[f"{prefix}_floor_frac"]), Tref_K=finite(row["Tref_K"]),
    )


def multihit_rate(barrier_eV: float, T: float) -> float:
    raw = NU_C * math.exp(-barrier_eV / (KB_EV_K * T))
    return float(special.gammainc(MULTIHIT_M, min(raw * MULTIHIT_TAU_S, 1e12)) / MULTIHIT_TAU_S)


def arrhenius_rate(barrier_eV: float, T: float, nu: float) -> float:
    return float(nu * math.exp(np.clip(-barrier_eV / (KB_EV_K * T), -745, 700)))


def surface_geometry(surface, T: float) -> dict[str, float]:
    sigma = np.linspace(0, 30e9, 241); G = np.asarray(surface.barrier_eV(sigma, T), float)
    d = np.gradient(G, sigma / 1e9); d2 = np.gradient(d, sigma / 1e9)
    i = int(np.argmax(abs(d))); floor = float(surface.barrier_eV(1e15, T)); drop = max(float(G[0] - floor), 1e-30)
    frac = (G - floor) / drop
    active = sigma[(frac <= .8) & (frac >= .2)]
    dT=1.0
    dp=np.gradient(np.asarray(surface.barrier_eV(sigma,T+dT),float),sigma/1e9)
    dm=np.gradient(np.asarray(surface.barrier_eV(sigma,T-dT),float),sigma/1e9)
    mixed=(dp-dm)/(2*dT)
    return {"zero_stress_eV": float(G[0]), "floor_eV": floor, "available_drop_eV": float(G[0] - floor),
            "max_stress_sensitivity_eV_per_GPa": float(np.max(abs(d))), "max_curvature_eV_per_GPa2": float(np.max(abs(d2))),
            "mixed_d2G_dT_dsigma_at_ref_eV_per_K_GPa": float(np.interp(SIGMA_REF_PA/1e9,sigma/1e9,mixed)),
            "stress_of_max_sensitivity_GPa": float(sigma[i] / 1e9),
            "transition_width_20_80_GPa": float(np.ptp(active) / 1e9) if len(active) > 1 else np.nan,
            "barrier_at_ref_stress_eV": float(surface.barrier_eV(SIGMA_REF_PA, T)),
            "floor_proximity_at_ref": float((surface.barrier_eV(SIGMA_REF_PA, T) - floor) / drop)}


def find_temperature_crossings(T: np.ndarray, y: np.ndarray) -> list[float]:
    roots = []
    for i in range(len(T) - 1):
        if not np.isfinite(y[i:i+2]).all(): continue
        if y[i] == 0: roots.append(float(T[i]))
        elif y[i] * y[i + 1] < 0:
            roots.append(float(T[i] - y[i] * (T[i + 1] - T[i]) / (y[i + 1] - y[i])))
    return roots


def barrier_entropy_analysis(candidates: pd.DataFrame, cases: pd.DataFrame, events: pd.DataFrame,
                             ExpFloorSurface, PTMechanism):
    first = events[events.event_index.eq(0)].copy()
    multiplicity = first.set_index(["candidate_id", "temperature_K"]).source_multiplicity_pre_advance.to_dict()
    barrier_rows, entropy_rows, interaction_rows, state_rows, curves = [], [], [], [], []
    crossover_summary = {}
    for _, row in candidates.iterrows():
        cid = row.candidate_id; gc = make_surface(row, "cleave", ExpFloorSurface); ge = make_surface(row, "emit", ExpFloorSurface)
        peierls = PTMechanism(finite(row.peierls_H0_eV), finite(row.peierls_activation_entropy_kB),
                             finite(row.peierls_exp_a), finite(row.peierls_exp_n), finite(row.peierls_nu0_s))
        taylor = PTMechanism(finite(row.taylor_H0_eV), finite(row.taylor_activation_entropy_kB),
                            finite(row.taylor_exp_a), finite(row.taylor_exp_n), finite(row.taylor_nu0_s))
        gp, gt = peierls.surface(ge), taylor.surface(ge)
        Ts = np.sort(cases.loc[cases.candidate_id.eq(cid), "temperature_K"].unique().astype(float))
        if not len(Ts): continue
        for T in Ts:
            gcm = surface_geometry(gc, T); gem = surface_geometry(ge, T)
            dT = 1.0
            def dtemp(surface): return float((surface.barrier_eV(SIGMA_REF_PA, T+dT)-surface.barrier_eV(SIGMA_REF_PA, T-dT))/(2*dT))
            gc_g0 = make_surface(row, "cleave", ExpFloorSurface, zero_gT=True); gc_s0 = make_surface(row, "cleave", ExpFloorSurface, zero_sT=True)
            ge_g0 = make_surface(row, "emit", ExpFloorSurface, zero_gT=True); ge_s0 = make_surface(row, "emit", ExpFloorSurface, zero_sT=True)
            record = {"candidate_id": cid, "temperature_K": T, "representative_stress_GPa": SIGMA_REF_PA/1e9}
            for pref, geom in (("cleave", gcm), ("emit", gem)): record.update({f"{pref}_{k}": v for k,v in geom.items()})
            record.update({"cleave_dG_dT_full_eV_per_K": dtemp(gc), "cleave_dG_dT_gT_zero_eV_per_K": dtemp(gc_g0),
                           "cleave_dG_dT_sT_zero_eV_per_K": dtemp(gc_s0), "emit_dG_dT_full_eV_per_K": dtemp(ge),
                           "emit_dG_dT_gT_zero_eV_per_K": dtemp(ge_g0), "emit_dG_dT_sT_zero_eV_per_K": dtemp(ge_s0)})
            barrier_rows.append(record)
            Sc = -finite(row.cleave_gT_eV_per_K) / KB_EV_K; Se = -finite(row.emit_gT_eV_per_K) / KB_EV_K
            entropy_rows.append({"candidate_id": cid, "temperature_K": T, "cleavage_entropy_eV_per_K": -finite(row.cleave_gT_eV_per_K),
                "cleavage_entropy_kB": Sc, "emission_entropy_eV_per_K": -finite(row.emit_gT_eV_per_K), "emission_entropy_kB": Se,
                "differential_emission_minus_cleavage_entropy_kB": Se-Sc,
                "peierls_entropy_kB": finite(row.peierls_activation_entropy_kB), "taylor_entropy_kB": finite(row.taylor_activation_entropy_kB),
                "peierls_minus_cleavage_entropy_kB": finite(row.peierls_activation_entropy_kB)-Sc,
                "taylor_minus_cleavage_entropy_kB": finite(row.taylor_activation_entropy_kB)-Sc,
                "minus_T_S_c_eV": -T*Sc*KB_EV_K, "minus_T_S_e_eV": -T*Se*KB_EV_K,
                "minus_T_S_peierls_eV": -T*finite(row.peierls_activation_entropy_kB)*KB_EV_K,
                "minus_T_S_taylor_eV": -T*finite(row.taylor_activation_entropy_kB)*KB_EV_K})
            mult = finite(multiplicity.get((cid,T)), 1.0)
            bc, be = float(gc.barrier_eV(SIGMA_REF_PA,T)), float(ge.barrier_eV(SIGMA_REF_PA,T))
            rc, re = multihit_rate(bc,T), arrhenius_rate(be,T,NU_E)*max(mult,1.0)
            rp = arrhenius_rate(float(gp.barrier_eV(peierls.stress_fraction*SIGMA_REF_PA,T)),T,peierls.nu0_s)
            rt = arrhenius_rate(float(gt.barrier_eV(taylor.stress_fraction*SIGMA_REF_PA,T)),T,taylor.nu0_s)
            interaction_rows.append({"candidate_id": cid, "temperature_K": T, "state_basis": "STANDARDIZED_FIXED_STRESS",
                "source_multiplicity": mult, "Gc_minus_Ge_eV": bc-be, "Ge_minus_Gc_eV": be-bc,
                "emission_minus_cleavage_stress_sensitivity_eV_per_GPa": gem["max_stress_sensitivity_eV_per_GPa"]-gcm["max_stress_sensitivity_eV_per_GPa"],
                "emission_minus_cleavage_curvature_eV_per_GPa2": gem["max_curvature_eV_per_GPa2"]-gcm["max_curvature_eV_per_GPa2"],
                "log10_emission_over_cleavage_rate": math.log10(max(re,1e-300)/max(rc,1e-300)),
                "log10_peierls_over_cleavage_rate": math.log10(max(rp,1e-300)/max(rc,1e-300)),
                "log10_taylor_over_cleavage_rate": math.log10(max(rt,1e-300)/max(rc,1e-300)),
                "tau_cleavage_s": 1/max(rc,1e-300), "tau_emission_s": 1/max(re,1e-300),
                "tau_peierls_s": 1/max(rp,1e-300), "tau_taylor_s": 1/max(rt,1e-300),
                "relative_dG_dT_emit_minus_cleave_eV_per_K": dtemp(ge)-dtemp(gc)})
        denseT = np.linspace(max(300,Ts.min()), min(1400,Ts.max()), 1101)
        medmult = max(float(np.nanmedian([multiplicity.get((cid,t),np.nan) for t in Ts])), 1.0)
        logR=[]
        for T in denseT:
            rc=multihit_rate(float(gc.barrier_eV(SIGMA_REF_PA,T)),T)
            re=arrhenius_rate(float(ge.barrier_eV(SIGMA_REF_PA,T)),T,NU_E)*medmult
            logR.append(math.log10(max(re,1e-300)/max(rc,1e-300)))
        logR=np.asarray(logR); roots=find_temperature_crossings(denseT,logR)
        primary=roots[0] if roots else np.nan
        sharp=np.interp(primary,denseT,np.gradient(logR,denseT)) if roots else np.nan
        crossover_summary[cid]={"kinetic_crossover_count":len(roots),"kinetic_crossover_topology":"NONE" if not roots else ("SINGLE" if len(roots)==1 else "MULTIPLE"),
                                "kinetic_crossover_temperatures_K":";".join(f"{x:.8g}" for x in roots),
                                "kinetic_crossover_primary_K":primary,"kinetic_crossover_sharpness_per_K":sharp}
        for T in np.linspace(300,1400,23):
            for sigma in np.linspace(0,20e9,41):
                bc=float(gc.barrier_eV(sigma,T)); be=float(ge.barrier_eV(sigma,T))
                rc=multihit_rate(bc,T); re=arrhenius_rate(be,T,NU_E)*medmult
                curves.append({"candidate_id":cid,"temperature_K":T,"stress_GPa":sigma/1e9,"cleavage_barrier_eV":bc,
                               "emission_barrier_eV":be,"barrier_difference_Ge_minus_Gc_eV":be-bc,
                               "log10_emission_over_cleavage_rate":math.log10(max(re,1e-300)/max(rc,1e-300))})

        sf = first[first.candidate_id.eq(cid)]
        for _, ev in sf.iterrows():
            T=float(ev.temperature_K); radius=max(float(ev.tip_radius_pre_advance_m),1e-12)
            sigma=float(ev.K_MPa_sqrt_m)*1e6/math.sqrt(2*math.pi*radius)
            sigma_emit=max(sigma-float(ev.backstress_pre_advance_Pa),0.0)
            bc=float(gc.barrier_eV(sigma,T)); be=float(ge.barrier_eV(sigma_emit,T))
            rc=multihit_rate(bc,T); re=arrhenius_rate(be,T,NU_E)*max(float(ev.source_multiplicity_pre_advance),1.0)
            pbar=float(gp.barrier_eV(peierls.stress_fraction*sigma_emit,T)); tbar=float(gt.barrier_eV(taylor.stress_fraction*sigma_emit,T))
            rp=arrhenius_rate(pbar,T,peierls.nu0_s); rt=arrhenius_rate(tbar,T,taylor.nu0_s)
            p0=PTMechanism(peierls.H0_eV,0.0,peierls.exp_a,peierls.exp_n,peierls.nu0_s).surface(ge)
            t0=PTMechanism(taylor.H0_eV,0.0,taylor.exp_a,taylor.exp_n,taylor.nu0_s).surface(ge)
            rp0=arrhenius_rate(float(p0.barrier_eV(peierls.stress_fraction*sigma_emit,T)),T,peierls.nu0_s)
            rt0=arrhenius_rate(float(t0.barrier_eV(taylor.stress_fraction*sigma_emit,T)),T,taylor.nu0_s)
            state_rows.append({"candidate_id":cid,"temperature_K":T,"state_reconstruction_class":"PARTIAL_SAVED_FIRST_PASSAGE_PROXY",
                "missing_state_fields":"K_shield;mobile_population;retained_population;slip_field;active_state_vector",
                "K_first_MPa_sqrt_m":float(ev.K_MPa_sqrt_m),"tip_radius_pre_advance_m":radius,
                "front_width_pre_advance_m":float(ev.front_width_pre_advance_m),"backstress_pre_advance_Pa":float(ev.backstress_pre_advance_Pa),
                "source_multiplicity_pre_advance":float(ev.source_multiplicity_pre_advance),"cumulative_source_activations":float(ev.cumulative_source_activations),
                "cumulative_line_content":float(ev.cumulative_line_content),"applied_tip_stress_proxy_Pa":sigma,
                "emission_effective_stress_proxy_Pa":sigma_emit,"cleavage_barrier_state_proxy_eV":bc,"emission_barrier_state_proxy_eV":be,
                "log10_emission_over_cleavage_state_proxy":math.log10(max(re,1e-300)/max(rc,1e-300)),
                "log10_peierls_rate_state_proxy":math.log10(max(rp,1e-300)),"log10_taylor_rate_state_proxy":math.log10(max(rt,1e-300)),
                "peierls_log10_rate_entropy_effect_at_saved_state":math.log10(max(rp,1e-300))-math.log10(max(rp0,1e-300)),
                "taylor_log10_rate_entropy_effect_at_saved_state":math.log10(max(rt,1e-300))-math.log10(max(rt0,1e-300))})
    cross=pd.DataFrame.from_dict(crossover_summary,orient="index").rename_axis("candidate_id").reset_index()
    interactions=pd.DataFrame(interaction_rows).merge(cross,on="candidate_id",how="left")
    return pd.DataFrame(barrier_rows),pd.DataFrame(entropy_rows),interactions,pd.DataFrame(state_rows),pd.DataFrame(curves),cross


def pca_scores(matrix: np.ndarray, ids: Iterable[str], prefix: str, n_components=5, feature_names: list[str] | None=None):
    x=np.asarray(matrix,float); means=np.nanmean(x,axis=0); x=np.where(np.isfinite(x),x,means)
    std=np.std(x,axis=0); keep=std>1e-14; z=(x[:,keep]-means[keep])/std[keep]
    u,s,vt=np.linalg.svd(z,full_matrices=False); k=min(n_components,len(s)); score=u[:,:k]*s[:k]
    variance=s*s/max(len(x)-1,1); ratio=variance/max(variance.sum(),1e-30)
    out=pd.DataFrame({"candidate_id":list(ids)})
    for i in range(k): out[f"{prefix}_PC{i+1}"]=score[:,i]
    kept_names=np.asarray(feature_names if feature_names is not None else [f"feature_{i}" for i in range(x.shape[1])])[keep]
    top={}
    for i in range(k):
        order=np.argsort(abs(vt[i]))[::-1][:8]
        top[f"PC{i+1}"]=[{"feature":str(kept_names[j]),"loading":float(vt[i,j])} for j in order]
    return out,{"explained_variance_ratio":ratio[:k].tolist(),"feature_count":int(keep.sum()),"top_absolute_loadings":top}


def functional_analysis(curves: pd.DataFrame, response_points: pd.DataFrame):
    ids=[]; mats=[]; barrier_feature_names=None
    for cid,g in curves.groupby("candidate_id",sort=False):
        g=g.sort_values(["temperature_K","stress_GPa"]); ids.append(cid)
        mats.append(np.r_[g.cleavage_barrier_eV,g.emission_barrier_eV,g.barrier_difference_Ge_minus_Gc_eV,g.log10_emission_over_cleavage_rate])
        if barrier_feature_names is None:
            points=[f"T{t:g}_sigma{s:g}" for t,s in zip(g.temperature_K,g.stress_GPa)]
            barrier_feature_names=[f"{quantity}_{point}" for quantity in ["Gc","Ge","Ge_minus_Gc","log10_rate_ratio"] for point in points]
    bscore,bmeta=pca_scores(np.vstack(mats),ids,"barrier_surface",feature_names=barrier_feature_names)
    common=np.array([700,800,900,950,1000,1050,1100,1200,1300,1400],float)
    rids=[]; rmats=[]
    for cid,g in response_points.groupby("candidate_id"):
        g=g.sort_values("temperature_K")
        if len(g)==len(common) and np.allclose(g.temperature_K,common):
            y=g.K_response_MPa_sqrt_m.to_numpy(float); rids.append(cid); rmats.append(y/y[0])
    rscore,rmeta=pca_scores(np.vstack(rmats),rids,"fracture_response",feature_names=[f"normalized_K_T{t:g}" for t in common])
    return bscore,rscore,{"barrier_surface":bmeta,"fracture_response":rmeta}


def silhouette_score_simple(x: np.ndarray, labels: np.ndarray) -> float:
    n=len(x); dist=np.sqrt(((x[:,None,:]-x[None,:,:])**2).sum(axis=2)); vals=[]
    for i in range(n):
        same=(labels==labels[i]); same[i]=False
        a=dist[i,same].mean() if same.any() else 0.0
        others=[dist[i,labels==k].mean() for k in np.unique(labels) if k!=labels[i] and np.any(labels==k)]
        b=min(others) if others else 0.0; vals.append((b-a)/max(a,b,1e-30) if same.any() else 0.0)
    return float(np.mean(vals))


def morphology_clusters(desc: pd.DataFrame):
    features=[c for c in ["S_low_MPa_sqrt_m_per_K","S_mid_MPa_sqrt_m_per_K","S_high_MPa_sqrt_m_per_K",
        "max_abs_thermal_slope_MPa_sqrt_m_per_K","max_abs_curvature_MPa_sqrt_m_per_K2","fractional_resistance_span",
        "DBTT_magnitude_MPa_sqrt_m","peak_prominence_MPa_sqrt_m","fractional_terminal_change"] if c in desc]
    x=desc[features].to_numpy(float); med=np.nanmedian(x,axis=0); x=np.where(np.isfinite(x),x,med)
    std=np.std(x,axis=0); std[std==0]=1; z=(x-np.mean(x,axis=0))/std
    best=None
    for k in range(2,min(9,len(z)-1)):
        cent,label=kmeans2(z,k,minit="points",seed=914); score=silhouette_score_simple(z,label)
        if best is None or score>best[0]: best=(score,label,cent,k)
    score,label,_,k=best
    return pd.DataFrame({"candidate_id":desc.candidate_id,"cluster":label,"cluster_method":"kmeans_silhouette_not_forced_four",
                         "selected_k":k,"silhouette":score}),{"selected_k":k,"silhouette":score,"features":features}


def aggregate_predictors(candidates, barriers, entropy, interactions, states, cross, response):
    b900=barriers.iloc[(barriers.temperature_K-900).abs().groupby(barriers.candidate_id).idxmin()].copy()
    b900=b900.drop(columns="temperature_K").rename(columns={c:f"T900_{c}" for c in b900.columns if c!="candidate_id"})
    e=entropy.groupby("candidate_id",as_index=False).first().drop(columns="temperature_K")
    i=interactions.iloc[(interactions.temperature_K-900).abs().groupby(interactions.candidate_id).idxmin()].copy()
    ikeep=["candidate_id","log10_emission_over_cleavage_rate","log10_peierls_over_cleavage_rate","log10_taylor_over_cleavage_rate",
           "relative_dG_dT_emit_minus_cleave_eV_per_K"]
    i=i[ikeep].rename(columns={c:f"T900_{c}" for c in ikeep if c!="candidate_id"})
    s=states.groupby("candidate_id",as_index=False).agg({
        "backstress_pre_advance_Pa":"mean","tip_radius_pre_advance_m":"mean","front_width_pre_advance_m":"mean",
        "source_multiplicity_pre_advance":"mean","cumulative_source_activations":"mean",
        "log10_emission_over_cleavage_state_proxy":"mean","peierls_log10_rate_entropy_effect_at_saved_state":"mean",
        "taylor_log10_rate_entropy_effect_at_saved_state":"mean"}).add_prefix("state_mean_")
    s=s.rename(columns={"state_mean_candidate_id":"candidate_id"})
    basecols=["candidate_id","canonical_family","historical_response_class","is_canonical_holdout"]+ACTIVE_FIELDS
    out=candidates[basecols].merge(response,on="candidate_id",how="left").merge(e,on="candidate_id",how="left")
    return out.merge(b900,on="candidate_id",how="left").merge(i,on="candidate_id",how="left").merge(cross,on="candidate_id",how="left").merge(s,on="candidate_id",how="left")


def fdr_bh(values: pd.Series) -> np.ndarray:
    p=np.asarray(values,float); q=np.full_like(p,np.nan); good=np.flatnonzero(np.isfinite(p))
    if not len(good): return q
    order=good[np.argsort(p[good])]; ranked=p[order]*len(good)/np.arange(1,len(good)+1)
    q[order]=np.minimum(np.minimum.accumulate(ranked[::-1])[::-1],1); return q


def correlation_tables(master: pd.DataFrame):
    response_names=[c for c in ["S_low_MPa_sqrt_m_per_K","S_mid_MPa_sqrt_m_per_K","S_high_MPa_sqrt_m_per_K",
        "max_abs_thermal_slope_MPa_sqrt_m_per_K","fractional_resistance_span","DBTT_temperature_K","DBTT_width_K",
        "DBTT_magnitude_MPa_sqrt_m","peak_temperature_K","peak_prominence_MPa_sqrt_m","fractional_terminal_change"] if c in master]
    excludes=set(response_names+["candidate_id","canonical_family","historical_response_class","is_canonical_holdout"])
    predictors=[c for c in master.columns if c not in excludes and pd.api.types.is_numeric_dtype(master[c]) and
                not c.startswith(("K_min_","K_max_","baseline_resistance","K_300_","response_target","n_temperatures","temperature_"))]
    subsets={"DISCOVERY_NONCANONICAL":~master.is_canonical_holdout,"ALL":np.ones(len(master),bool),"CANONICAL_HOLDOUT":master.is_canonical_holdout}
    rows=[]
    for subset,mask in subsets.items():
        d=master.loc[mask]
        for pred in predictors:
            for resp in response_names:
                q=d[[pred,resp]].apply(pd.to_numeric,errors="coerce").dropna(); n=len(q)
                rec={"subset":subset,"predictor":pred,"response":resp,"n":n,"pearson_r":np.nan,"pearson_p":np.nan,
                     "spearman_rho":np.nan,"spearman_p":np.nan,"test_status":"INSUFFICIENT_N_OR_VARIATION"}
                if n>=3 and q[pred].nunique()>1 and q[resp].nunique()>1:
                    pr=stats.pearsonr(q[pred],q[resp]); sr=stats.spearmanr(q[pred],q[resp]); rec.update({"pearson_r":pr.statistic,
                        "spearman_rho":sr.statistic,"pearson_p":pr.pvalue if n>=5 else np.nan,"spearman_p":sr.pvalue if n>=5 else np.nan,
                        "test_status":"TESTED" if n>=5 else "EXPLORATORY_N3_N4"})
                rows.append(rec)
    cor=pd.DataFrame(rows); cor["pearson_q_fdr"]=fdr_bh(cor.pearson_p); cor["spearman_q_fdr"]=fdr_bh(cor.spearman_p)
    partial=[]
    controls=[c for c in ["baseline_resistance_MPa_sqrt_m","cleave_G00_eV","emit_G00_eV","cleave_sigc0_GPa","emit_sigc0_GPa"] if c in master]
    for pred in predictors:
        for resp in response_names:
            cols=[pred,resp]+[c for c in controls if c not in {pred,resp}]
            q=master.loc[~master.is_canonical_holdout,cols].apply(pd.to_numeric,errors="coerce").dropna()
            if len(q)<max(12,len(cols)+4) or q[pred].nunique()<2 or q[resp].nunique()<2: continue
            z=np.column_stack([np.ones(len(q))]+[q[c] for c in cols[2:]])
            rx=q[pred]-z@np.linalg.lstsq(z,q[pred],rcond=None)[0]; ry=q[resp]-z@np.linalg.lstsq(z,q[resp],rcond=None)[0]
            if np.ptp(rx)<=0 or np.ptp(ry)<=0: continue
            pr=stats.pearsonr(rx,ry); sr=stats.spearmanr(rx,ry)
            partial.append({"predictor":pred,"response":resp,"controls":";".join(cols[2:]),"n":len(q),
                "partial_pearson_r":pr.statistic,"partial_pearson_p":pr.pvalue,"partial_spearman_rho":sr.statistic,"partial_spearman_p":sr.pvalue})
    partial=pd.DataFrame(partial)
    if not partial.empty:
        partial["partial_pearson_q_fdr"]=fdr_bh(partial.partial_pearson_p); partial["partial_spearman_q_fdr"]=fdr_bh(partial.partial_spearman_p)
    return cor,partial


def validation_subset(candidates: pd.DataFrame, response: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for cid,label in CANONICAL.items():
        r=response[response.candidate_id.eq(cid)]
        if label in {"Peak-T","DBTT"}:
            klass="SPATIAL_CORRECTION"; evidence="v10.2.27 committed selection metadata: accepted 2-D transfer; no raw matched temperature table recovered"
        else:
            klass="NO_MATCHED_2D"; evidence="v10.2.27 selection requires 2-D validation; no raw matched temperature table recovered"
        rows.append({"candidate_id":cid,"canonical_family":label,"validation_class":klass,"n_1D_temperatures":len(r),
                     "n_matched_2D_temperatures":0,"quantitative_trend_test_available":False,"evidence":evidence,
                     "legacy_mechanism_validation":"generic historical 800/950 K DBTT pair reproduced 1-D trend within ~10%; not this fingerprint"})
    return pd.DataFrame(rows)


def _savefig(fig: plt.Figure, out: Path, stem: str, data: pd.DataFrame):
    fig.savefig(out/f"{stem}.png",dpi=190,bbox_inches="tight"); plt.close(fig); data.to_csv(out/f"{stem}_plot_data.csv",index=False)


def _class(row) -> str:
    return row.canonical_family if isinstance(row.canonical_family,str) else "other"


def scatter(ax, data, x, y, xlabel=None, ylabel=None):
    q=data[["candidate_id","canonical_family",x,y]].copy(); q[x]=pd.to_numeric(q[x],errors="coerce"); q[y]=pd.to_numeric(q[y],errors="coerce"); q=q.dropna(subset=[x,y])
    normal=q[q.canonical_family.isna()]; ax.scatter(normal[x],normal[y],s=14,c="#94A3B8",alpha=.45,label="broad population")
    for label,color in COLORS.items():
        if label=="other": continue
        g=q[q.canonical_family.eq(label)]; ax.scatter(g[x],g[y],s=65,c=color,edgecolor="black",linewidth=.4,label=label,zorder=3)
    if len(q)>=3 and q[x].nunique()>1: ax.text(.03,.97,f"n={len(q)}, Spearman ρ={stats.spearmanr(q[x],q[y]).statistic:.2f}",transform=ax.transAxes,va="top",fontsize=8)
    ax.set_xlabel(xlabel or x); ax.set_ylabel(ylabel or y); return q


def heatmap(corr: pd.DataFrame, out: Path, stem: str, responses: list[str], predictors: list[str], partial=False):
    if partial:
        q=corr[corr.response.isin(responses)&corr.predictor.isin(predictors)].copy(); value="partial_spearman_rho"; qcol="partial_spearman_q_fdr"
    else:
        q=corr[(corr.subset.eq("DISCOVERY_NONCANONICAL"))&corr.response.isin(responses)&corr.predictor.isin(predictors)].copy(); value="spearman_rho"; qcol="spearman_q_fdr"
    mat=q.pivot(index="predictor",columns="response",values=value).reindex(index=predictors,columns=responses)
    fig,ax=plt.subplots(figsize=(max(9,len(responses)*1.2),max(5,len(predictors)*.45))); im=ax.imshow(mat,vmin=-1,vmax=1,cmap="coolwarm",aspect="auto")
    ax.set_xticks(range(len(responses)),[x.replace("_","\n") for x in responses],rotation=35,ha="right",fontsize=7)
    ax.set_yticks(range(len(predictors)),[x.replace("_"," ") for x in predictors],fontsize=7)
    for i,p in enumerate(predictors):
        for j,r in enumerate(responses):
            cell=q[(q.predictor.eq(p))&(q.response.eq(r))]
            if cell.empty: continue
            val=finite(cell.iloc[0][value]); n=int(cell.iloc[0].n)
            if not np.isfinite(val) or n<5: continue
            mark="*" if finite(cell.iloc[0][qcol],1)<=.1 else ""
            ax.text(j,i,f"{val:.2f}{mark}\nn={n}",ha="center",va="center",fontsize=5.5,color="white" if abs(val)>.55 else "black")
    fig.colorbar(im,ax=ax,label="partial Spearman ρ" if partial else "Spearman ρ (* FDR q≤0.10)"); fig.tight_layout(); _savefig(fig,out,stem,q)


PROVENANCE_COLUMNS = ["candidate_id","parameter_fingerprint","source_registry","simulation_git_sha",
    "simulation_sha_provenance","github_repository","historical_branch","canonical_family","canonical_option_key"]


def with_provenance(frame: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Attach immutable candidate provenance without duplicating existing columns."""
    cols=[c for c in PROVENANCE_COLUMNS if c in candidates and (c=="candidate_id" or c not in frame)]
    return frame.merge(candidates[cols].drop_duplicates("candidate_id"),on="candidate_id",how="left")


def add_state_residuals(master: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    out=master.copy(); records=[]
    intrinsic=["cleave_G00_eV","emit_G00_eV","cleave_gT_eV_per_K","emit_gT_eV_per_K",
        "cleave_sT_GPa_per_K","emit_sT_GPa_per_K","cleave_sigc0_GPa","emit_sigc0_GPa"]
    for response in ["S_low_MPa_sqrt_m_per_K","S_mid_MPa_sqrt_m_per_K","fractional_resistance_span",
                     "DBTT_magnitude_MPa_sqrt_m","peak_prominence_MPa_sqrt_m"]:
        cols=[response]+intrinsic
        q=out.loc[~out.is_canonical_holdout,cols].apply(pd.to_numeric,errors="coerce").dropna()
        if len(q)<20: continue
        means=q[intrinsic].mean(); std=q[intrinsic].std().replace(0,1)
        z=(q[intrinsic]-means)/std; X=np.column_stack([np.ones(len(q)),z])
        coef=np.linalg.lstsq(X,q[response],rcond=None)[0]
        good=out[cols].apply(pd.to_numeric,errors="coerce").notna().all(axis=1)
        za=(out.loc[good,intrinsic]-means)/std
        predicted=np.column_stack([np.ones(good.sum()),za])@coef
        col=f"intrinsic_barrier_residual__{response}"; out[col]=np.nan
        out.loc[good,col]=out.loc[good,response]-predicted
        records.append({"response":response,"fit_subset":"DISCOVERY_NONCANONICAL","n":len(q),
            "model":"standardized linear intrinsic barrier parameters","r2_training":1-float(np.sum((q[response]-X@coef)**2))/max(float(np.sum((q[response]-q[response].mean())**2)),1e-30),
            "features":";".join(intrinsic)})
    return out,pd.DataFrame(records)


def _candidate_plot_data(master: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    cols=["candidate_id","canonical_family",x,y]
    return master[cols].replace([np.inf,-np.inf],np.nan).dropna(subset=[x,y])


def make_figures(out: Path, master: pd.DataFrame, response_points: pd.DataFrame,
                 barriers: pd.DataFrame, entropy: pd.DataFrame, interactions: pd.DataFrame,
                 states: pd.DataFrame, correlations: pd.DataFrame, partial: pd.DataFrame,
                 bscore: pd.DataFrame, rscore: pd.DataFrame) -> list[str]:
    stems=[]
    def save(fig,stem,data):
        _savefig(fig,out,stem,data); stems.append(stem)

    # Population and canonical dimensional/normalized response curves.
    fig,ax=plt.subplots(figsize=(8.2,5.2))
    for cid,g in response_points.groupby("candidate_id"):
        ax.plot(g.temperature_K,g.K_response_MPa_sqrt_m,color="#94A3B8",alpha=.13,lw=.65)
    for cid,label in CANONICAL.items():
        g=response_points[response_points.candidate_id.eq(cid)].sort_values("temperature_K")
        ax.plot(g.temperature_K,g.K_response_MPa_sqrt_m,"o-",label=label,color=COLORS[label],lw=2.2,ms=4)
    ax.set(xlabel="Temperature (K)",ylabel=r"Authoritative $K_R$ (MPa√m)",title="v9.13 temperature-response population")
    ax.legend(ncol=2,fontsize=8); save(fig,"temperature_response_population",response_points)

    can=response_points[response_points.candidate_id.isin(CANONICAL)].copy()
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.4))
    for cid,label in CANONICAL.items():
        g=can[can.candidate_id.eq(cid)].sort_values("temperature_K")
        ax1.plot(g.temperature_K,g.K_response_MPa_sqrt_m,"o-",label=label,color=COLORS[label])
        ax2.plot(g.temperature_K,g.K_over_first_available,"o-",label=label,color=COLORS[label])
    ax1.set(xlabel="Temperature (K)",ylabel=r"$K_R$ (MPa√m)"); ax2.set(xlabel="Temperature (K)",ylabel=r"$K_R/K_R(T_{min})$")
    ax1.legend(fontsize=8); fig.suptitle("Canonical four: dimensional and temperature-normalized response")
    save(fig,"temperature_response_canonical_four",can)

    # Required correlation views.
    plot_specs=[
        ("kinetic_crossover_vs_DBTT_temperature","kinetic_crossover_primary_K","DBTT_temperature_K",r"Fixed-stress kinetic crossover $T_\times$ (K)","DBTT temperature (K)"),
        ("crossover_sharpness_vs_DBTT_width","kinetic_crossover_sharpness_per_K","DBTT_width_K",r"$d\log_{10}(\Gamma_e/\Gamma_c)/dT$ at crossover", "DBTT width (K)"),
    ]
    for stem,x,y,xlab,ylab in plot_specs:
        q=_candidate_plot_data(master,x,y); fig,ax=plt.subplots(figsize=(6.3,4.7)); scatter(ax,q,x,y,xlab,ylab)
        ax.legend(fontsize=7,loc="best"); fig.tight_layout(); save(fig,stem,q)

    multi_specs={
      "entropy_vs_fracture_slopes":[
        ("cleavage_entropy_kB","S_low_MPa_sqrt_m_per_K","Cleavage entropy ($k_B$)","Low-T slope"),
        ("cleavage_entropy_kB","S_high_MPa_sqrt_m_per_K","Cleavage entropy ($k_B$)","High-T slope"),
        ("emission_entropy_kB","S_low_MPa_sqrt_m_per_K","Emission entropy ($k_B$)","Low-T slope"),
        ("emission_entropy_kB","S_high_MPa_sqrt_m_per_K","Emission entropy ($k_B$)","High-T slope")],
      "differential_entropy_vs_DBTT":[
        ("differential_emission_minus_cleavage_entropy_kB","DBTT_magnitude_MPa_sqrt_m",r"$S_e-S_c$ ($k_B$)","DBTT magnitude"),
        ("differential_emission_minus_cleavage_entropy_kB","DBTT_temperature_K",r"$S_e-S_c$ ($k_B$)","DBTT temperature (K)")],
      "entropy_vs_peak_amplitude":[
        ("differential_emission_minus_cleavage_entropy_kB","peak_prominence_MPa_sqrt_m",r"$S_e-S_c$ ($k_B$)","Peak prominence"),
        ("differential_emission_minus_cleavage_entropy_kB","peak_temperature_K",r"$S_e-S_c$ ($k_B$)","Peak temperature (K)")],
      "transport_entropy_vs_temperature_response":[
        ("peierls_entropy_kB","DBTT_magnitude_MPa_sqrt_m","Peierls entropy ($k_B$)","DBTT magnitude"),
        ("taylor_entropy_kB","DBTT_magnitude_MPa_sqrt_m","Taylor entropy ($k_B$)","DBTT magnitude"),
        ("peierls_entropy_kB","peak_prominence_MPa_sqrt_m","Peierls entropy ($k_B$)","Peak prominence"),
        ("taylor_entropy_kB","peak_prominence_MPa_sqrt_m","Taylor entropy ($k_B$)","Peak prominence")],
    }
    for stem,specs in multi_specs.items():
        n=len(specs); fig,axs=plt.subplots(2,2,figsize=(10.5,8)) if n==4 else plt.subplots(1,2,figsize=(10.5,4.5))
        axs=np.asarray(axs).ravel(); data=[]
        for ax,(x,y,xlab,ylab) in zip(axs,specs):
            q=_candidate_plot_data(master,x,y).assign(panel=f"{x}__{y}"); scatter(ax,q,x,y,xlab,ylab); data.append(q)
        axs[0].legend(fontsize=7); fig.tight_layout(); save(fig,stem,pd.concat(data,ignore_index=True,sort=False))

    responses=["S_low_MPa_sqrt_m_per_K","S_high_MPa_sqrt_m_per_K","fractional_resistance_span",
               "DBTT_temperature_K","DBTT_width_K","DBTT_magnitude_MPa_sqrt_m","peak_prominence_MPa_sqrt_m"]
    predictors=["cleavage_entropy_kB","emission_entropy_kB","differential_emission_minus_cleavage_entropy_kB",
                "peierls_entropy_kB","taylor_entropy_kB","T900_relative_dG_dT_emit_minus_cleave_eV_per_K",
                "T900_log10_emission_over_cleavage_rate","kinetic_crossover_primary_K","kinetic_crossover_sharpness_per_K"]
    heatmap(correlations,out,"barrier_temperature_correlation_heatmap",responses,predictors); stems.append("barrier_temperature_correlation_heatmap")
    heatmap(partial,out,"barrier_temperature_partial_correlation_heatmap",responses,predictors,partial=True); stems.append("barrier_temperature_partial_correlation_heatmap")

    q=master[["candidate_id","canonical_family","historical_response_class","differential_emission_minus_cleavage_entropy_kB",
              "kinetic_crossover_primary_K","fractional_resistance_span"]].dropna(subset=["differential_emission_minus_cleavage_entropy_kB","kinetic_crossover_primary_K"]).copy()
    q["morphology_marker"]=np.where(q.historical_response_class.str.contains("Peak",case=False,na=False),"Peak-like",
        np.where(q.historical_response_class.str.contains("DBTT",case=False,na=False),"DBTT-like","other/intermediate"))
    q.loc[q.canonical_family.notna(),"morphology_marker"]=q.loc[q.canonical_family.notna(),"canonical_family"]
    fig,ax=plt.subplots(figsize=(7,5.1)); markers={"Peak-like":"^","DBTT-like":"s","weak-T":"P","ceramic-like":"D","Peak-T":"*","DBTT":"X","other/intermediate":"o"}
    palette={"Peak-like":"#F59E0B","DBTT-like":"#3B82F6","weak-T":"#8B5CF6","ceramic-like":"#64748B","Peak-T":"#F59E0B","DBTT":"#3B82F6","other/intermediate":"#94A3B8"}
    for label,g in q.groupby("morphology_marker"):
        ax.scatter(g.differential_emission_minus_cleavage_entropy_kB,g.kinetic_crossover_primary_K,
            s=np.where(g.canonical_family.notna(),90,25),marker=markers.get(label,"o"),c=palette.get(label,"#94A3B8"),alpha=.7,label=label,edgecolor="black",linewidth=.25)
    ax.set(xlabel=r"Differential entropy $S_e-S_c$ ($k_B$)",ylabel=r"Kinetic crossover $T_\times$ (K)",
        title="Barrier-temperature/fracture-morphology phase map")
    ax.legend(fontsize=7,ncol=2); save(fig,"barrier_temperature_fracture_phase_map",q)

    bp=with_provenance(bscore,master); rp=with_provenance(rscore,master)
    for scores,prefix,stem,title in [(bp,"barrier_surface","barrier_surface_pca","Barrier/rate-surface PCA"),
                                     (rp,"fracture_response","fracture_response_pca","Normalized fracture-response PCA")]:
        x=f"{prefix}_PC1"; y=f"{prefix}_PC2"; q=scores.dropna(subset=[x,y]); fig,ax=plt.subplots(figsize=(6.4,5))
        scatter(ax,q,x,y); ax.set_title(title); ax.legend(fontsize=7); save(fig,stem,q)
    both=bp[["candidate_id","canonical_family","barrier_surface_PC1"]].merge(rp[["candidate_id","fracture_response_PC1"]],on="candidate_id")
    fig,ax=plt.subplots(figsize=(6.4,5)); scatter(ax,both,"barrier_surface_PC1","fracture_response_PC1")
    ax.set_title("Leading intrinsic barrier mode versus response mode"); ax.legend(fontsize=7); save(fig,"barrier_modes_vs_fracture_modes",both)

    # Total temperature derivative and saved-state residual diagnostics.
    q=_candidate_plot_data(master,"T900_cleave_dG_dT_full_eV_per_K","S_mid_MPa_sqrt_m_per_K")
    fig,ax=plt.subplots(figsize=(6.4,5)); scatter(ax,q,"T900_cleave_dG_dT_full_eV_per_K","S_mid_MPa_sqrt_m_per_K",
        r"Total $\partial G_c/\partial T|_\sigma$ (eV/K)",r"Mid-T $dK_R/dT$")
    ax.legend(fontsize=7); save(fig,"total_barrier_dT_vs_fracture_slope",q)
    residuals=[c for c in master if c.startswith("intrinsic_barrier_residual__")]
    if residuals:
        q=_candidate_plot_data(master,"state_mean_backstress_pre_advance_Pa",residuals[0]); fig,ax=plt.subplots(figsize=(6.4,5))
        scatter(ax,q,"state_mean_backstress_pre_advance_Pa",residuals[0],"Mean saved pre-event backstress (Pa)","Response residual after intrinsic-barrier fit")
        ax.legend(fontsize=7); save(fig,"state_mediated_response_residuals",q)

    # Canonical entropy-energy contributions.
    ec=with_provenance(entropy,master); ec=ec[ec.candidate_id.isin(CANONICAL)]
    fig,axs=plt.subplots(2,2,figsize=(11,8),sharex=True)
    for ax,(cid,label) in zip(axs.flat,CANONICAL.items()):
        g=ec[ec.candidate_id.eq(cid)].sort_values("temperature_K")
        for col,lab in [("minus_T_S_c_eV","cleavage"),("minus_T_S_e_eV","emission"),("minus_T_S_peierls_eV","Peierls"),("minus_T_S_taylor_eV","Taylor")]:
            ax.plot(g.temperature_K,g[col],"o-",ms=3,label=lab)
        ax.set_title(label); ax.set_ylabel(r"$-TS_{act}$ (eV)"); ax.legend(fontsize=7)
    for ax in axs[-1]: ax.set_xlabel("Temperature (K)")
    fig.suptitle("Diagnostic entropy contributions (not fracture-curve counterfactuals)"); save(fig,"canonical_entropy_contributions",ec)

    # One three-panel mechanistic overlay per canonical material.
    for cid,label in CANONICAL.items():
        bg=barriers[barriers.candidate_id.eq(cid)].sort_values("temperature_K")
        ig=interactions[interactions.candidate_id.eq(cid)].sort_values("temperature_K")
        rg=response_points[response_points.candidate_id.eq(cid)].sort_values("temperature_K")
        sg=states[states.candidate_id.eq(cid)].sort_values("temperature_K")
        fig,axs=plt.subplots(3,1,figsize=(7.4,9),sharex=True)
        axs[0].plot(bg.temperature_K,bg.cleave_barrier_at_ref_stress_eV,"o-",label="crack opening")
        axs[0].plot(bg.temperature_K,bg.emit_barrier_at_ref_stress_eV,"o-",label="dislocation emission/nucleation")
        axs[0].set_ylabel("Barrier at 5 GPa (eV)"); axs[0].legend(fontsize=8)
        axs[1].plot(ig.temperature_K,ig.log10_emission_over_cleavage_rate,"o-",label=r"$\log_{10}(\Gamma_e/\Gamma_c)$")
        axs[1].plot(ig.temperature_K,ig.log10_peierls_over_cleavage_rate,"o-",label=r"$\log_{10}(\Gamma_P/\Gamma_c)$")
        axs[1].plot(ig.temperature_K,ig.log10_taylor_over_cleavage_rate,"o-",label=r"$\log_{10}(\Gamma_T/\Gamma_c)$")
        axs[1].axhline(0,color="black",lw=.8); axs[1].set_ylabel("Relative log-rate"); axs[1].legend(fontsize=7)
        axs[2].plot(rg.temperature_K,rg.K_response_MPa_sqrt_m,"o-",color=COLORS[label],label=r"$K_R(T)$")
        cr=ig.kinetic_crossover_primary_K.dropna()
        if len(cr): axs[2].axvline(cr.iloc[0],ls="--",color="#111827",label=r"fixed-stress $T_\times$")
        axs[2].set(xlabel="Temperature (K)",ylabel=r"$K_R$ (MPa√m)"); axs[2].legend(fontsize=8)
        fig.suptitle(f"{label}: bare kinetics and authoritative fracture response")
        pdata=pd.concat([bg.assign(panel="barrier"),ig.assign(panel="rates"),rg.assign(panel="response"),sg.assign(panel="saved_state")],ignore_index=True,sort=False)
        save(fig,f"canonical_{label.lower().replace('-','_')}_mechanistic_overlay",pdata)
    return stems


def correlation_lookup(cor: pd.DataFrame,predictor: str,response: str) -> dict:
    q=cor[(cor.subset.eq("DISCOVERY_NONCANONICAL"))&cor.predictor.eq(predictor)&cor.response.eq(response)]
    return q.iloc[0].to_dict() if len(q) else {"n":0,"spearman_rho":np.nan,"spearman_q_fdr":np.nan}


def hypothesis_table(master: pd.DataFrame,cor: pd.DataFrame,validation: pd.DataFrame) -> pd.DataFrame:
    specs={
      "H1":("cleavage_entropy_kB","S_low_MPa_sqrt_m_per_K","Absolute crack-opening entropy controls part of fracture temperature slope"),
      "H2":("differential_emission_minus_cleavage_entropy_kB","fractional_resistance_span","Differential entropy predicts morphology"),
      "H3":("kinetic_crossover_primary_K","DBTT_temperature_K","DBTT temperature tracks fixed-state kinetic crossover"),
      "H4":("kinetic_crossover_sharpness_per_K","DBTT_width_K","DBTT width tracks crossover sharpness"),
      "H6":("T900_relative_dG_dT_emit_minus_cleave_eV_per_K","fractional_resistance_span","Weak-T follows thermal-sensitivity cancellation"),
      "H7":("T900_log10_emission_over_cleavage_rate","fractional_terminal_change","Ceramic-like behavior follows persistent opening dominance"),
      "H8":("peierls_entropy_kB","DBTT_magnitude_MPa_sqrt_m","Transport entropy explains residual DBTT/Peak behavior"),
    }
    rows=[]
    for h,(p,r,text) in specs.items():
        e=correlation_lookup(cor,p,r); rho=finite(e.get("spearman_rho")); q=finite(e.get("spearman_q_fdr")); n=int(e.get("n",0))
        conclusion="SUPPORTED" if n>=20 and np.isfinite(q) and q<=.10 and abs(rho)>=.2 else (
            "WEAK_SUPPORT" if n>=10 and np.isfinite(q) and q<=.10 else "REJECTED")
        rows.append({"hypothesis":h,"statement":text,"classification":conclusion,"primary_predictor":p,"primary_response":r,
                     "n_discovery":n,"spearman_rho":rho,"spearman_q_fdr":q,"basis":"broad noncanonical discovery population; association is not causal"})
    # H5 needs topology, not a one-variable significance test.
    peak=master[(master.peak_prominence_MPa_sqrt_m>0)&(~master.is_canonical_holdout)]
    multiple=int((peak.kinetic_crossover_count>1).sum()) if len(peak) else 0
    rows.append({"hypothesis":"H5","statement":"Peak-T requires a change in the rate-controlling process, not entropy magnitude alone",
        "classification":"WEAK_SUPPORT" if multiple else "INSUFFICIENT_EVIDENCE","n_discovery":len(peak),"spearman_rho":np.nan,"spearman_q_fdr":np.nan,
        "basis":f"{multiple}/{len(peak)} positive-prominence discovery curves have multiple standardized-state crossings; saved state is incomplete"})
    rows.append({"hypothesis":"H9","statement":"The same descriptors predict 1-D and 2-D temperature trends",
        "classification":"INSUFFICIENT_EVIDENCE","n_discovery":0,"spearman_rho":np.nan,"spearman_q_fdr":np.nan,
        "basis":"no raw same-fingerprint matched 1-D/2-D temperature series recovered; selection metadata alone cannot test a trend"})
    return pd.DataFrame(rows).sort_values("hypothesis")


def rcurve_observables(events: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (cid,T),g in events.groupby(["candidate_id","temperature_K"]):
        g=g.sort_values("cumulative_projected_extension_m")
        x=g.cumulative_projected_extension_m.to_numpy(float)*1e6; y=g.K_MPa_sqrt_m.to_numpy(float)
        developed=x>=20
        rows.append({"candidate_id":cid,"temperature_K":T,"first_event_K_MPa_sqrt_m":float(y[0]),
            "event_max_K_MPa_sqrt_m":float(np.max(y)),"event_final_K_MPa_sqrt_m":float(y[-1]),
            "developed_R_curve_slope_MPa_sqrt_m_per_um":linear_slope(x[developed],y[developed]),
            "event_record_max_extension_um":float(x[-1]),"event_record_count":len(g)})
    return pd.DataFrame(rows)


def ridge_model_comparison(master: pd.DataFrame) -> pd.DataFrame:
    """Five-fold ridge comparisons; canonical four remain an external holdout."""
    d=master.copy()
    d["cleavage_x_emission_entropy"]=d.cleavage_entropy_kB*d.emission_entropy_kB
    d["emission_x_peierls_entropy"]=d.emission_entropy_kB*d.peierls_entropy_kB
    d["cleavage_entropy_x_stress_scale_T"]=d.cleavage_entropy_kB*d.cleave_sT_GPa_per_K
    d["crossover_position_x_sharpness"]=d.kinetic_crossover_primary_K*d.kinetic_crossover_sharpness_per_K
    opening=["cleavage_entropy_kB","emission_entropy_kB","cleave_G00_eV","emit_G00_eV","cleave_sigc0_GPa",
        "emit_sigc0_GPa","cleave_sT_GPa_per_K","emit_sT_GPa_per_K"]
    transport=opening+["peierls_entropy_kB","taylor_entropy_kB","peierls_H0_eV","taylor_H0_eV"]
    interactions=transport+["differential_emission_minus_cleavage_entropy_kB","T900_relative_dG_dT_emit_minus_cleave_eV_per_K",
        "T900_log10_emission_over_cleavage_rate","cleavage_x_emission_entropy","emission_x_peierls_entropy",
        "cleavage_entropy_x_stress_scale_T","kinetic_crossover_primary_K","kinetic_crossover_sharpness_per_K",
        "crossover_position_x_sharpness"]
    models={"OPENING_EMISSION_ABSOLUTE":opening,"PLUS_TRANSPORT":transport,"PLUS_BARRIER_INTERACTIONS":interactions}
    responses=["S_low_MPa_sqrt_m_per_K","S_mid_MPa_sqrt_m_per_K","fractional_resistance_span",
               "DBTT_magnitude_MPa_sqrt_m","peak_prominence_MPa_sqrt_m"]
    rows=[]
    for response in responses:
      for model,features in models.items():
        features=[f for f in features if f in d]
        train=d.loc[~d.is_canonical_holdout,[response]+features]
        X=train[features].apply(pd.to_numeric,errors="coerce"); med=X.median(); X=X.fillna(med)
        y=pd.to_numeric(train[response],errors="coerce"); good=y.notna(); X=X.loc[good].to_numpy(float); y=y.loc[good].to_numpy(float)
        mu=X.mean(axis=0); sd=X.std(axis=0); sd[sd==0]=1; X=(X-mu)/sd; folds=np.arange(len(y))%5
        trials=[]
        for alpha in [.01,.1,1.,10.,100.]:
            pred=np.full(len(y),np.nan)
            for fold in range(5):
                tr=folds!=fold; te=~tr; A=np.column_stack([np.ones(tr.sum()),X[tr]])
                penalty=np.eye(A.shape[1])*alpha; penalty[0,0]=0
                beta=np.linalg.solve(A.T@A+penalty,A.T@y[tr]); pred[te]=np.column_stack([np.ones(te.sum()),X[te]])@beta
            rmse=float(np.sqrt(np.mean((y-pred)**2))); r2=1-float(np.sum((y-pred)**2))/max(float(np.sum((y-y.mean())**2)),1e-30)
            trials.append((rmse,r2,alpha))
        rmse,r2,alpha=min(trials); A=np.column_stack([np.ones(len(y)),X]); penalty=np.eye(A.shape[1])*alpha; penalty[0,0]=0
        beta=np.linalg.solve(A.T@A+penalty,A.T@y)
        hold=d.loc[d.is_canonical_holdout,[response]+features]; Xh=hold[features].apply(pd.to_numeric,errors="coerce").fillna(med)
        yh=pd.to_numeric(hold[response],errors="coerce"); hg=yh.notna(); Xh=(Xh.loc[hg].to_numpy(float)-mu)/sd
        hp=np.column_stack([np.ones(hg.sum()),Xh])@beta if hg.any() else np.array([])
        hold_rmse=float(np.sqrt(np.mean((yh.loc[hg].to_numpy()-hp)**2))) if len(hp) else np.nan
        rows.append({"response":response,"model":model,"n_discovery":len(y),"n_features":len(features),"features":";".join(features),
            "imputation":"discovery median","scaling":"discovery z-score","cv":"deterministic 5-fold","selected_ridge_alpha":alpha,
            "cv_rmse":rmse,"cv_r2":r2,"canonical_holdout_n":int(hg.sum()),"canonical_holdout_rmse":hold_rmse})
    return pd.DataFrame(rows)


def collinearity_table(master: pd.DataFrame) -> pd.DataFrame:
    cols=[c for c in master if c!="is_canonical_holdout" and pd.api.types.is_numeric_dtype(master[c])]
    corr=master.loc[~master.is_canonical_holdout,cols].corr(method="spearman",min_periods=20); rows=[]
    for i,a in enumerate(cols):
        for b in cols[i+1:]:
            rho=finite(corr.loc[a,b])
            if np.isfinite(rho) and abs(rho)>=.90:
                rows.append({"descriptor_a":a,"descriptor_b":b,"spearman_rho":rho,
                    "interpretation":"STRONGLY_COLLINEAR_DO_NOT_ASSIGN_UNIQUE_CAUSALITY"})
    return pd.DataFrame(rows)


def dataset_inventory(paths: dict[str,Path], source: Path) -> pd.DataFrame:
    rows=[]
    for role,path in paths.items():
        if not path.exists():
            rows.append({"role":role,"path":str(path),"exists":False}); continue
        rows.append({"role":role,"path":str(path),"exists":True,"bytes":path.stat().st_size,"sha256":sha256(path),
            "artifact_scope":"GITHUB_TRACKED" if str(path).startswith(str(REPO)) else "LOCAL_HISTORICAL_RUN_OR_SOURCE"})
    return pd.DataFrame(rows)


def write_report(out: Path, master: pd.DataFrame, correlations: pd.DataFrame, partial: pd.DataFrame,
                 hypotheses: pd.DataFrame, validation: pd.DataFrame, pca_meta: dict, cluster_meta: dict,
                 state_model: pd.DataFrame, model_comparison: pd.DataFrame, audit: dict):
    def evidence(p,r,partial_mode=False):
        table=partial if partial_mode else correlations
        if partial_mode:
            q=table[(table.predictor.eq(p))&(table.response.eq(r))]
            if q.empty: return "not estimable"
            z=q.iloc[0]; return f"partial ρ={finite(z.partial_spearman_rho):.3f}, FDR q={finite(z.partial_spearman_q_fdr):.3g}, n={int(z.n)}"
        z=correlation_lookup(correlations,p,r)
        return f"ρ={finite(z.get('spearman_rho')):.3f}, FDR q={finite(z.get('spearman_q_fdr')):.3g}, n={int(z.get('n',0))}"
    can=master[master.is_canonical_holdout].drop_duplicates("candidate_id")
    hlines="\n".join(f"- **{r.hypothesis}: {r.classification}.** {r.statement}. Evidence: {r.basis}"
                       for _,r in hypotheses.iterrows())
    canlines="\n".join(f"- **{r.canonical_family}:** K range {r.K_min_MPa_sqrt_m:.3g}–{r.K_max_MPa_sqrt_m:.3g} MPa√m; "
        f"fractional span {r.fractional_resistance_span:.3g}; fixed-stress crossover {finite(r.kinetic_crossover_primary_K):.4g} K ({r.kinetic_crossover_topology})."
        for _,r in can.iterrows())
    text=f"""# Barrier shape, activation entropy, and temperature-dependent fracture morphology

## Scope and provenance

This is an existing-data, analysis-only study. It changed no constitutive parameter and launched no fracture simulation. The authoritative broad population contains {master.candidate_id.nunique()} unique fingerprints and {audit['temperature_case_count']} temperature cases ({audit['complete_temperature_case_count']} complete; {audit['censored_temperature_case_count']} explicitly censored and excluded from response-curve inference). Simulation code provenance is historical commit `{HISTORICAL_SHA}` from [`{GITHUB_REPOSITORY}`](https://github.com/{GITHUB_REPOSITORY}), inferred from exact run-contract hashes and independently matched to the clean historical checkout. The analysis was generated by `{audit['analysis_branch']}` at `{audit['analysis_git_sha']}`.

The exact historical `ExpFloorSurface` and `PTMechanism` implementations are called directly. Cleavage uses its production multi-hit law (m=3, τ=1 µs, ν₀=10¹² s⁻¹); emission and transport retain their distinct production prefactors. Standardized-state calculations use 5 GPa and are bare-landscape diagnostics, not alternate fracture simulations.

The four named materials were held out from discovery correlations:

{canlines}

## Main conclusions

1. **Does activation entropy measurably control the response?** Yes as an association in parts of the population, but entropy alone does not determine morphology. Cleavage entropy versus low-T slope gives {evidence('cleavage_entropy_kB','S_low_MPa_sqrt_m_per_K')}. The causal chain requires stress-scale evolution and the evolved tip state.
2. **Cleavage or emission entropy?** The larger absolute rank association varies with the response descriptor; neither is uniformly dominant. For DBTT magnitude: cleavage {evidence('cleavage_entropy_kB','DBTT_magnitude_MPa_sqrt_m')}; emission {evidence('emission_entropy_kB','DBTT_magnitude_MPa_sqrt_m')}.
3. **Is their difference better?** Differential entropy is informative for some shape measures ({evidence('differential_emission_minus_cleavage_entropy_kB','fractional_resistance_span')}), but it is not a universally sufficient coordinate. The comparison is treated as predictive association, not unique causality, because the descriptors are mathematically collinear.
4. **How important are Peierls/Taylor entropies?** They do not materially predict DBTT magnitude in this manifold (Peierls: {evidence('peierls_entropy_kB','DBTT_magnitude_MPa_sqrt_m')}; Taylor: {evidence('taylor_entropy_kB','DBTT_magnitude_MPa_sqrt_m')}). Five-fold ridge comparisons in `barrier_temperature_model_comparison.csv` test whether transport features improve other morphology targets. Saved-state entropy-removal diagnostics quantify instantaneous rate sensitivity only; they are not counterfactual fracture curves.
5. **What predicts DBTT temperature?** The standardized crossover alone gives {evidence('kinetic_crossover_primary_K','DBTT_temperature_K')}. Barrier thermal tilt and transport descriptors must be considered jointly where the crossover is absent or multiple.
6. **What predicts DBTT magnitude?** Relative thermal rate/barrier descriptors and transport entropy are more physically interpretable than a single raw barrier coefficient. Differential entropy evidence is {evidence('differential_emission_minus_cleavage_entropy_kB','DBTT_magnitude_MPa_sqrt_m')}.
7. **What predicts DBTT width?** Crossover sharpness gives {evidence('kinetic_crossover_sharpness_per_K','DBTT_width_K')}; the finite historical temperature grid limits width resolution.
8. **What predicts Peak-T location and amplitude?** Peak amplitude versus differential entropy gives {evidence('differential_emission_minus_cleavage_entropy_kB','peak_prominence_MPa_sqrt_m')}. A peak is not reducible to large curvature or entropy: topology and state saturation/change of controlling plastic process are needed, and the archived state is incomplete.
9. **Why are some candidates weakly temperature dependent?** They occupy small relative-thermal-tilt/response-span regions, consistent with approximate compensation of opening and accommodation, but only diagnostic support is available ({evidence('T900_relative_dG_dT_emit_minus_cleave_eV_per_K','fractional_resistance_span')}).
10. **Why are ceramic-like curves thermally softened?** The canonical curve remains on the opening-favored side of the standardized competition while its opening barrier thermally softens. This is consistent with fracture preceding adequate plastic relaxation, not proof of a unique cause.
11. **Does a cleavage/emission crossover predict transitions?** Not by itself. Candidates can have zero, one, or multiple fixed-state crossings; state-selected fracture loads differ from the common-stress diagnostic. The population relation is reported in item 5.
12. **How much remains until state-mediated plasticity is included?** Intrinsic linear fits have training R² values of {', '.join(f'{r.response}={r.r2_training:.2f}' for _,r in state_model.iterrows())}. Correlations of saved backstress/front/source proxies with residuals show additional association, but this is a partial proxy, not a full mediation decomposition.
13. **Which relationships survive scale control?** Partial correlations control K(300)/baseline where available, Gc0, Ge0, and reference stress scales. Examples: differential entropy/response span {evidence('differential_emission_minus_cleavage_entropy_kB','fractional_resistance_span',True)}; crossover/DBTT temperature {evidence('kinetic_crossover_primary_K','DBTT_temperature_K',True)}. These tests address shape beyond trivial strength.
14. **What survives canonical holdout?** The canonical points are plotted but never used to tune discovery thresholds or clusters. Their qualitative placement is reported above. Four holdouts are too few for a powered confirmatory significance test.
15. **Are trends consistent with 2-D?** Insufficient evidence. Peak/DBTT have committed metadata indicating accepted transfer, while weak-T/ceramic require validation, but no raw same-fingerprint matched temperature series was recovered. Therefore no `REDUCED_VALID` quantitative claim is made.
16. **Smallest causal test set?** Run a matched, common-300-K-scale fractional design around one DBTT-like and one Peak-like row: independently perturb cleavage `gT`, emission `gT`, cleavage/emission `sT`, and Peierls/Taylor entropy at low/center/high levels, with 300-K surfaces re-anchored. Start with 18–24 1-D trajectories selected by D-optimal coverage, reserve 6 for confirmation, then transfer only discriminating pairs to 2-D. This is proposed only; no campaign was launched.

## Explicit hypotheses

{hlines}

## Bare versus state-conditioned interpretation

The bare tables measure exact production surfaces/rates on a common (T,σ) grid. The along-trajectory table uses the state saved immediately before the first accepted opening event. It preserves applied K, tip radius, front width, backstress, multiplicity, source activations, and line content. It explicitly labels missing `K_shield`, mobile/retained populations, slip field, and the full active-state vector as `PARTIAL_SAVED_FIRST_PASSAGE_PROXY`; those unavailable quantities were neither guessed nor reconstructed.

Counterfactual columns with `gT=0`, `sT=0`, or transport S=0 are instantaneous diagnostic evaluations at fixed state. They are not fracture predictions. No interpolation was performed through missing/nonconverged response temperatures and no extrapolation beyond simulated crack extension was used.

## Functional morphology and collinearity

Barrier-surface PCA retained {len(pca_meta['barrier_surface']['explained_variance_ratio'])} modes; explained fractions are {pca_meta['barrier_surface']['explained_variance_ratio']}. Response PCA fractions are {pca_meta['fracture_response']['explained_variance_ratio']}. Clustering selected k={cluster_meta['selected_k']} by silhouette ({cluster_meta['silhouette']:.3f}); four historical labels were not imposed. Loadings and plotted scores are preserved for interpretation. Strongly linked EXP-floor descriptors are treated as barrier-geometry surrogates rather than uniquely causal coefficients.

The leading barrier-surface mode has its largest sampled loading at `{pca_meta['barrier_surface']['top_absolute_loadings']['PC1'][0]['feature']}`; the leading response mode is largest at `{pca_meta['fracture_response']['top_absolute_loadings']['PC1'][0]['feature']}`. The full top-loading lists are in `pca_metadata.json`, so physical mode labels follow the observed loading locations rather than being assigned in advance.

The ridge table compares opening/emission absolute features, addition of transport, and addition of barrier-interaction terms under identical deterministic five-fold splits. It records discovery CV and untouched canonical-four errors; it is a predictive diagnostic, not a causal fit.

## Limits

This database is observational in parameter space and contains parameter correlations. The state archive is incomplete, the response temperature grid limits crossover/width resolution, and 2-D raw matches are absent. Consequently “supported” means FDR-controlled population association consistent with the mechanism, not a causal proof. The proposed sparse factorial test is the appropriate next causal step.
"""
    (out/"BARRIER_TEMPERATURE_FRACTURE_MORPHOLOGY_REPORT.md").write_text(text)


REQUIRED_TABLES=["fracture_temperature_master.csv","fracture_response_descriptors.csv","fracture_barrier_geometry_descriptors.csv",
 "fracture_entropy_descriptors.csv","fracture_barrier_interaction_descriptors.csv","fracture_state_at_first_passage.csv",
 "fracture_barrier_temperature_correlations.csv","fracture_barrier_temperature_partial_correlations.csv","fracture_response_pca_scores.csv",
 "fracture_barrier_surface_pca_scores.csv","fracture_temperature_morphology_clusters.csv","fracture_1D_2D_validation_subset.csv"]
REQUIRED_FIGURES=["temperature_response_population.png","temperature_response_canonical_four.png","entropy_vs_fracture_slopes.png",
 "differential_entropy_vs_DBTT.png","kinetic_crossover_vs_DBTT_temperature.png","crossover_sharpness_vs_DBTT_width.png",
 "entropy_vs_peak_amplitude.png","transport_entropy_vs_temperature_response.png","barrier_temperature_correlation_heatmap.png",
 "barrier_temperature_partial_correlation_heatmap.png","barrier_temperature_fracture_phase_map.png","barrier_surface_pca.png",
 "fracture_response_pca.png","barrier_modes_vs_fracture_modes.png","canonical_peak_t_mechanistic_overlay.png",
 "canonical_dbtt_mechanistic_overlay.png","canonical_weak_t_mechanistic_overlay.png","canonical_ceramic_like_mechanistic_overlay.png"]


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--source-root",type=Path,default=DEFAULT_SOURCE)
    ap.add_argument("--out",type=Path,default=REPO/"runs/v913_barrier_temperature_fracture_morphology_v1")
    args=ap.parse_args(); source=args.source_root.resolve(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    if git(source,"rev-parse","HEAD") != HISTORICAL_SHA: raise RuntimeError("historical checkout is not at authoritative simulation commit")
    if git(source,"status","--short"): raise RuntimeError("historical source checkout must be clean")
    ExpFloorSurface,PTMechanism=load_production_types(source)
    candidates,cases,events,paths=load_population(source)
    if set(CANONICAL)-set(candidates.candidate_id): raise RuntimeError("canonical fingerprint rows missing")
    canonical_fingerprint=validate_canonical_selection(candidates,paths["selection"],paths["registry"])
    # Preserve censored cases in the temperature master, but response_descriptors
    # explicitly admits only complete cases and never converts censoring to a rate.
    response,response_points=response_descriptors(cases)
    barriers,entropy,interactions,states,curves,cross=barrier_entropy_analysis(candidates,cases,events,ExpFloorSurface,PTMechanism)
    bscore,rscore,pca_meta=functional_analysis(curves,response_points)
    clusters,cluster_meta=morphology_clusters(response)
    master_candidates=aggregate_predictors(candidates,barriers,entropy,interactions,states,cross,response)
    master_candidates,state_model=add_state_residuals(master_candidates)
    master_candidates=with_provenance(master_candidates,candidates)
    correlations,partial=correlation_tables(master_candidates)
    for table in (correlations,partial):
        table["simulation_git_sha"]=HISTORICAL_SHA
        table["github_repository"]=GITHUB_REPOSITORY
        table["source_population"]="v9.13 broad population; canonical four held out in DISCOVERY_NONCANONICAL"
    validation=with_provenance(validation_subset(candidates,response_points),candidates)
    hypotheses=hypothesis_table(master_candidates,correlations,validation)
    model_comparison=ridge_model_comparison(master_candidates)
    collinearity=collinearity_table(master_candidates)
    rcurve=rcurve_observables(events)
    temp_master=(cases.merge(response_points.drop(columns=["source_dataset","response_target_um"],errors="ignore"),
                    on=["candidate_id","temperature_K"],how="left")
                 .merge(rcurve,on=["candidate_id","temperature_K"],how="left"))
    temp_master=with_provenance(temp_master,candidates).merge(candidates[["candidate_id"]+ACTIVE_FIELDS],on="candidate_id",how="left")
    # Attach row-level bare/entropy/interaction quantities without duplicating provenance.
    for table in [barriers,entropy,interactions]:
        add=[c for c in table if c not in temp_master or c in {"candidate_id","temperature_K"}]
        temp_master=temp_master.merge(table[add],on=["candidate_id","temperature_K"],how="left")
    tables={
      "fracture_temperature_master.csv":temp_master,
      "fracture_response_descriptors.csv":with_provenance(response,candidates),
      "fracture_barrier_geometry_descriptors.csv":with_provenance(barriers,candidates),
      "fracture_entropy_descriptors.csv":with_provenance(entropy,candidates),
      "fracture_barrier_interaction_descriptors.csv":with_provenance(interactions,candidates),
      "fracture_state_at_first_passage.csv":with_provenance(states,candidates),
      "fracture_barrier_temperature_correlations.csv":correlations,
      "fracture_barrier_temperature_partial_correlations.csv":partial,
      "fracture_response_pca_scores.csv":with_provenance(rscore,candidates),
      "fracture_barrier_surface_pca_scores.csv":with_provenance(bscore,candidates),
      "fracture_temperature_morphology_clusters.csv":with_provenance(clusters,candidates),
      "fracture_1D_2D_validation_subset.csv":validation,
    }
    for name,table in tables.items(): table.to_csv(out/name,index=False)
    with_provenance(response_points,candidates).to_csv(out/"fracture_response_curve_points.csv",index=False)
    with_provenance(curves,candidates).to_csv(out/"barrier_surface_common_grid.csv",index=False)
    hypotheses.to_csv(out/"physics_hypothesis_tests.csv",index=False)
    state_model.to_csv(out/"intrinsic_barrier_response_models.csv",index=False)
    model_comparison.to_csv(out/"barrier_temperature_model_comparison.csv",index=False)
    collinearity.to_csv(out/"barrier_descriptor_strong_collinearity.csv",index=False)
    inventory=dataset_inventory(paths,source); inventory.to_csv(out/"dataset_inventory.csv",index=False)
    audit={"schema":"v913_barrier_temperature_fracture_morphology_audit_v1","analysis_branch":git(REPO,"branch","--show-current"),
      "analysis_git_sha":git(REPO,"rev-parse","HEAD"),"simulation_git_sha":HISTORICAL_SHA,
      "simulation_sha_provenance":"INFERRED_BY_EXACT_RUN_CONTRACT_SOURCE_HASH_MATCH","historical_checkout":str(source),
      "historical_checkout_clean":True,"exact_production_types_sha256":sha256(source/"arrhenius_fracture/emergent_gnd_types_v912.py"),
      "canonical_active_parameter_fingerprint_sha256":canonical_fingerprint,
      "canonical_declared_source_active_parameter_fingerprint_sha256":json.loads(paths["selection"].read_text())["source_active_parameter_fingerprint_sha256"],
      "canonical_selection_source_commit":json.loads(paths["selection"].read_text())["source_commit"],
      "candidate_count":int(candidates.candidate_id.nunique()),"temperature_case_count":len(cases),
      "complete_temperature_case_count":int(cases.status.eq("complete").sum()),
      "censored_temperature_case_count":int((~cases.status.eq("complete")).sum()),"event_record_count":len(events),
      "all_cases_complete":bool(cases.status.eq("complete").all()),"canonical_ids":CANONICAL,
      "state_reconstruction_class":"PARTIAL_SAVED_FIRST_PASSAGE_PROXY","new_simulations_launched":False,
      "physics_changed":False,"common_stress_Pa":SIGMA_REF_PA,"input_inventory_sha256":sha256(out/"dataset_inventory.csv")}
    figures=make_figures(out,master_candidates,response_points,barriers,entropy,interactions,states,correlations,partial,bscore,rscore)
    audit["figure_stems"]=figures; audit["required_tables"]=REQUIRED_TABLES; audit["required_figures"]=REQUIRED_FIGURES
    write_report(out,master_candidates,correlations,partial,hypotheses,validation,pca_meta,cluster_meta,state_model,model_comparison,audit)
    (out/"analysis_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
    (out/"pca_metadata.json").write_text(json.dumps(pca_meta,indent=2)+"\n")
    (out/"cluster_metadata.json").write_text(json.dumps(cluster_meta,indent=2)+"\n")
    missing=[name for name in REQUIRED_TABLES+REQUIRED_FIGURES+["BARRIER_TEMPERATURE_FRACTURE_MORPHOLOGY_REPORT.md"] if not (out/name).exists()]
    if missing: raise RuntimeError(f"required artifacts missing: {missing}")
    print(json.dumps({"status":"PASS","output":str(out),"candidates":len(candidates),"cases":len(cases),"figures":len(figures)},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
