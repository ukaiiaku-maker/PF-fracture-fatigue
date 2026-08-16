#!/usr/bin/env python3
"""Existing-data barrier-geometry -> fatigue-morphology analysis.

This is deliberately a read-only consumer of the v9.13/v9.14 and v10.2.31/32
databases.  It evaluates the same EXP-floor free-energy surfaces and the same
K-to-tip-stress map used by production; it does not launch simulations or fit a
new fatigue law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special, stats


REPO = Path(__file__).resolve().parents[1]
V913 = Path("/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_13_dbtt_temperature_shelf")
V914 = Path("/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search")
GLOBAL = V914 / "runtime_inputs/v914/endurance_knee_global_300K_1024.csv"
LOCAL = V914 / "runtime_inputs/v914/local_fracture_manifold_256.csv"
GLOBAL_AUDIT = V914 / "runs/v914_endurance_knee_global_300K_stageA_analysis/stageA_audit.csv"
LOCAL_AUDIT = V914 / "runs/v914_local_fracture_stageA_analysis/stageA_audit.csv"
EQUIV = V914 / "runtime_inputs/v914/fracture_equivalence_registry.csv"
MECH = V914 / "runs/v914_endurance_knee_mechanism_classification_475/mechanism_classification.csv"
PROBES = V914 / "runs/v914_endurance_knee_mechanism_probe_475"
RERANK = REPO / "runs/v914_endurance_knee_rerank_DBTT_highK_v1/analysis"
OLD_CURVES = RERANK / "candidate_fatigue_curves.csv"
RERANK_MASTER = RERANK / "candidate_rerank_master.csv"
ABCD_HYBRID = REPO / "runs/v10_2_32_endurance_knee_ABCD_hybrid_HCF_LCF_v1/analysis/abcd_1D_accelerated_explicit_rates.csv"
MATERIAL_HYBRID = REPO / "runs/v10_2_32_HCF_LCF_transition_refinement_v2/analysis/full_material_hybrid_rates.csv"
TRANSITIONS = REPO / "runs/v10_2_32_HCF_LCF_transition_refinement_v2/analysis/transition_regime_summary.csv"
TEMP_TABLE = V913 / "runs/v9_13_zeroD_promoted_1d_384_50um_v2/one_d_screen/ranked_candidates.csv"

T_K = 300.0
KB_EV = 8.617333262145e-5
R_LOAD = 0.1
R0_M = 1.0e-6
SIGMA_CAP_PA = 30.0e9
NU_C = 1.0e12
NU_E = 1.0e11
MULTIHIT_M = 3.0
MULTIHIT_TAU_S = 1.0e-6
DA_PHYS_M = 5.0e-6
PARAM_PREFIXES = ("cleave_", "emit_", "peierls_", "taylor_", "physics__")
PARAM_EXACT = {
    "Tref_K", "rho_source0_m2", "rho_forest_floor_m2", "taylor_corr_rho_c_m2",
    "taylor_corr_scale", "c_blunt", "mobile_shield_fraction", "source_recovery_rate_s",
    "retained_recovery_rate_s", "source_refresh_length_um", "source_sites_per_system",
    "encounter_efficiency", "L_pz_um_recommended", "n_bins_recommended",
}
CONTROL_LABELS = {
    "v914_endurance_knee_0462": "A", "v914_endurance_knee_0658": "B",
    "v914_endurance_knee_0554": "C", "v914_endurance_knee_0133": "D",
}
CANONICAL = {
    "v913_zeroD_sobol_0202500": "DBTT",
    "v913_zeroD_sobol_0242980": "Peak-T",
    "v913_zeroD_sobol_0129902": "weak-T",
    "v913_zeroD_sobol_0077080": "ceramic-like",
}
SPATIAL = {
    "v914_endurance_knee_0462": "REDUCED_VALID",
    "v914_endurance_knee_0554": "REDUCED_VALID",
    "v913_zeroD_sobol_0077080": "REDUCED_VALID",
    "v914_endurance_knee_0658": "SPATIAL_CORRECTION",
    "v913_zeroD_sobol_0202500": "SPATIAL_CORRECTION",
    "v913_zeroD_sobol_0242980": "SPATIAL_CORRECTION",
    "v914_endurance_knee_0133": "SPATIAL_BIFURCATION",
    "v913_zeroD_sobol_0129902": "SPATIAL_UNRESOLVED",
}
CLASS_COLORS = {
    "A": "#0072B2", "B": "#D55E00", "C": "#009E73", "D": "#CC79A7",
    "DBTT": "#56B4E9", "Peak-T": "#E69F00", "weak-T": "#7A5195",
    "ceramic-like": "#5D6D7E", "SMOOTH_ARRHENIUS": "#999999",
}


def finite(value, default=np.nan) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_fingerprint(row: pd.Series, columns: list[str]) -> str:
    payload = {}
    for c in columns:
        v = row.get(c, np.nan)
        if pd.isna(v):
            payload[c] = None
        else:
            try:
                payload[c] = format(float(v), ".17g")
            except (TypeError, ValueError):
                payload[c] = str(v)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def current_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def parameter_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith(PARAM_PREFIXES) or c in PARAM_EXACT]


def _registry_rows() -> tuple[pd.DataFrame, list[Path]]:
    """Return one row per physical parameter fingerprint, preserving aliases."""
    sources: list[tuple[Path, str, int]] = [
        (GLOBAL, "V914_GLOBAL_1024", 10), (LOCAL, "V914_LOCAL_FRACTURE_256", 20),
        (EQUIV, "V913_FRACTURE_EQUIVALENCE", 30),
        (REPO / "arrhenius_fracture/data/materials/v10_2_31_endurance_knee_ABCD_registry.csv", "V1031_ABCD", 1),
        (REPO / "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv", "V1027_CANONICAL", 2),
    ]
    frames = []
    used = []
    for path, label, priority in sources:
        if not path.exists():
            continue
        d = pd.read_csv(path)
        if "candidate_id" not in d:
            continue
        d["source_registry"] = label
        d["source_registry_path"] = str(path)
        d["source_priority"] = priority
        frames.append(d)
        used.append(path)
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw = raw.sort_values(["source_priority", "candidate_id"]).drop_duplicates("candidate_id", keep="first")
    pcols = parameter_columns(raw)
    raw["parameter_fingerprint"] = raw.apply(lambda r: json_fingerprint(r, pcols), axis=1)
    # Different campaign names for an exactly identical row are aliases, not
    # independent physical candidates. Prefer an ID that actually has fatigue.
    fatigue_ids = set(pd.read_csv(OLD_CURVES, usecols=["candidate_id"]).candidate_id.astype(str))
    chosen = []
    for fp, g in raw.groupby("parameter_fingerprint", sort=False):
        g = g.copy()
        g["has_fatigue_id"] = g.candidate_id.astype(str).isin(fatigue_ids)
        g = g.sort_values(["has_fatigue_id", "source_priority"], ascending=[False, True])
        r = g.iloc[0].copy()
        r["candidate_aliases"] = ";".join(sorted(set(g.candidate_id.astype(str))))
        r["alias_count"] = int(g.candidate_id.nunique())
        chosen.append(r)
    master = pd.DataFrame(chosen).drop(columns=["has_fatigue_id"], errors="ignore")
    return master.reset_index(drop=True), used


def _add_scales_and_labels(master: pd.DataFrame) -> pd.DataFrame:
    m = master.copy()
    scale_parts = []
    for path, kcol, refcol in [
        (GLOBAL_AUDIT, "stageA_K50_300K_MPa_sqrt_m", "fatigue_reference_deltaK_MPa_sqrt_m"),
        (LOCAL_AUDIT, "stageA_K50_300K_MPa_sqrt_m", None),
    ]:
        if path.exists():
            d = pd.read_csv(path)
            cols = ["candidate_id", kcol] + ([refcol] if refcol and refcol in d else [])
            d = d[cols].copy().rename(columns={kcol: "fracture_resistance_300K_MPa_sqrt_m"})
            if not refcol or refcol not in d:
                d["reference_deltaK_MPa_sqrt_m"] = np.nan
            else:
                d = d.rename(columns={refcol: "reference_deltaK_MPa_sqrt_m"})
            scale_parts.append(d)
    old = pd.read_csv(OLD_CURVES)
    curve_scales = old.groupby("candidate_id").agg(
        curve_K50=("monotonic_K50_300K_MPa_sqrt_m", "max"),
        curve_ref=("reference_deltaK_MPa_sqrt_m", "max"),
    ).reset_index()
    scales = pd.concat(scale_parts, ignore_index=True).drop_duplicates("candidate_id", keep="first")
    m = m.merge(scales, on="candidate_id", how="left").merge(curve_scales, on="candidate_id", how="left")
    m["fracture_resistance_300K_MPa_sqrt_m"] = m.fracture_resistance_300K_MPa_sqrt_m.fillna(m.curve_K50)
    m["reference_deltaK_MPa_sqrt_m"] = m.reference_deltaK_MPa_sqrt_m.fillna(m.curve_ref)
    # If only K50 is known, retain it as the scale while explicitly labelling
    # that no separate fatigue reference was saved.
    m["reference_deltaK_source"] = np.where(m.curve_ref.notna(), "FATIGUE_RESULT",
        np.where(m.reference_deltaK_MPa_sqrt_m.notna(), "STAGE_A_SAVED", "MISSING"))
    m["reference_deltaK_MPa_sqrt_m"] = m.reference_deltaK_MPa_sqrt_m.fillna(m.fracture_resistance_300K_MPa_sqrt_m)

    temp_map = {}
    if RERANK_MASTER.exists():
        rr = pd.read_csv(RERANK_MASTER)
        temp_map.update(dict(zip(rr.candidate_id.astype(str), rr.temperature_class.astype(str))))
    m["temperature_response_class"] = m.candidate_id.astype(str).map(temp_map).fillna(
        m.get("material_class", pd.Series(index=m.index, dtype=object)).fillna("UNKNOWN"))
    m["mechanism_control_class"] = m.candidate_id.astype(str).map(CONTROL_LABELS)
    if MECH.exists():
        mc = pd.read_csv(MECH)
        mmap = dict(zip(mc.candidate_id.astype(str), mc.mechanism_class.astype(str)))
        m["mechanism_probe_class"] = m.candidate_id.astype(str).map(mmap).fillna("NOT_PROBED")
    else:
        m["mechanism_probe_class"] = "NOT_PROBED"
    probe_letter = m.mechanism_probe_class.str.extract(r"^([ABCD])", expand=False).fillna("SMOOTH_ARRHENIUS")
    m["candidate_plot_class"] = m.candidate_id.astype(str).map({**CONTROL_LABELS, **CANONICAL}).fillna(probe_letter)
    m["spatial_validation_class"] = m.candidate_id.astype(str).map(SPATIAL).fillna("NO_MATCHED_2D")
    m["matched_2D_data_exists"] = m.spatial_validation_class.ne("NO_MATCHED_2D")
    m["accelerated_HCF_data_exists"] = m.candidate_id.astype(str).isin(set(old.candidate_id.astype(str)))
    explicit_ids = set()
    for p in (ABCD_HYBRID, MATERIAL_HYBRID):
        if p.exists():
            d = pd.read_csv(p)
            explicit_ids.update(d.loc[d.integration_mode.eq("explicit"), "candidate_id"].astype(str))
    m["explicit_LCF_data_exists"] = m.candidate_id.astype(str).isin(explicit_ids)
    m["temperature_K"] = T_K; m["R"] = R_LOAD
    m["cleavage_attempt_frequency_s"] = NU_C; m["emission_attempt_frequency_s"] = NU_E
    m["cleavage_multihit_m"] = MULTIHIT_M; m["cleavage_correlation_time_s"] = MULTIHIT_TAU_S
    m["da_phys_m"] = DA_PHYS_M
    m["event_length_mode"] = "threshold_scaled"
    m["event_length_minimum_factor"] = 0.5; m["event_length_maximum_factor"] = 4.0
    m["event_length_mean_preserved"] = True
    return m.drop(columns=["curve_K50", "curve_ref"], errors="ignore")


def _old_curve_rows() -> pd.DataFrame:
    d = pd.read_csv(OLD_CURVES)
    d["integration_mode"] = "accelerated"
    d["dimensionality"] = "1D"
    d["censor_status"] = d.status
    d["plot_kind"] = np.where(d.da_dN_m_per_cycle.notna(), "resolved",
        np.where(d.status.str.contains("censor", case=False, na=False), "censor", "partial"))
    d["source_campaign"] = d.source_path.str.extract(r"/(v9[^/]+|v10[^/]+)/")[0].fillna("HISTORICAL_FATIGUE")
    d["point_origin"] = "HISTORICAL_ACCELERATED"
    d["regime_classification_source"] = "not_preclassified"
    return d


def _hybrid_rows() -> pd.DataFrame:
    rows = []
    if ABCD_HYBRID.exists():
        d = pd.read_csv(ABCD_HYBRID)
        d = d[d.dimensionality.eq("1D")].copy()
        d["fraction"] = d.normalized_f
        d["censor_status"] = d.status
        d["source_path"] = d.result_path.fillna("")
        d["rate_basis"] = np.where(d.integration_mode.eq("explicit"), "explicit_cycle_developed", "accelerated_developed")
        d["point_origin"] = "V1032_ABCD_HYBRID"
        d["regime_classification_source"] = d.regime_classification
        rows.append(d)
    if MATERIAL_HYBRID.exists():
        d = pd.read_csv(MATERIAL_HYBRID)
        d = d[d.dimensionality.eq("1D")].copy()
        d["fraction"] = d.normalized_f
        d["status"] = d.censor_status
        d["source_path"] = d.result_path.fillna("")
        d["source_campaign"] = d.source_run_root
        d["rate_basis"] = np.where(d.integration_mode.eq("explicit"), "explicit_cycle_developed", "accelerated_developed")
        d["point_origin"] = "V1032_MATERIAL_TRANSITION"
        d["regime_classification_source"] = d.regime_classification
        rows.append(d)
    return pd.concat(rows, ignore_index=True, sort=False)


def assemble_fatigue_points(master: pd.DataFrame) -> pd.DataFrame:
    old = _old_curve_rows(); new = _hybrid_rows()
    cols = sorted(set(old.columns) | set(new.columns))
    p = pd.concat([old.reindex(columns=cols), new.reindex(columns=cols)], ignore_index=True, sort=False)
    aliases = {}
    for _, r in master.iterrows():
        for a in str(r.candidate_aliases).split(";"):
            aliases[a] = r.candidate_id
    p["physical_candidate_id"] = p.candidate_id.astype(str).map(aliases).fillna(p.candidate_id.astype(str))
    p["normalized_f"] = p.get("normalized_f", pd.Series(np.nan, index=p.index)).fillna(p.fraction)
    p["temperature_K"] = T_K; p["R"] = R_LOAD
    p["is_finite_rate"] = np.isfinite(pd.to_numeric(p.da_dN_m_per_cycle, errors="coerce")) & (pd.to_numeric(p.da_dN_m_per_cycle, errors="coerce") > 0)
    p["authoritative_use"] = False
    p["authoritative_reason"] = "PROVENANCE_ONLY"

    special = set(CONTROL_LABELS) | set(CANONICAL)
    # General candidates: the validated accelerated HCF/VHCF range ends at
    # f=1.2 in these screens. Retain higher points, but do not reinterpret them
    # as LCF after the explicit-cycle parity failure was established.
    mask_general = ~p.physical_candidate_id.isin(special) & p.point_origin.eq("HISTORICAL_ACCELERATED") & (p.normalized_f <= 1.2000001)
    p.loc[mask_general, ["authoritative_use", "authoritative_reason"]] = [True, "VALIDATED_ACCELERATED_HCF_WINDOW"]
    p.loc[~p.physical_candidate_id.isin(special) & (p.normalized_f > 1.2000001), "authoritative_reason"] = "ACCELERATED_HIGH_K_UNVALIDATED_FOR_LCF"

    # For the eight resolved controls, the newer integrated inventories are
    # authoritative. Historical points not represented by the new inventory
    # remain authoritative only below the first new loading.
    for cid in special:
        q = p[p.physical_candidate_id.eq(cid)]
        newq = q[~q.point_origin.eq("HISTORICAL_ACCELERATED")]
        if newq.empty:
            p.loc[q.index, ["authoritative_use", "authoritative_reason"]] = [True, "BEST_EXISTING_CONTROL_RECORD"]
            continue
        min_new = newq.normalized_f.min()
        low = q.point_origin.eq("HISTORICAL_ACCELERATED") & (q.normalized_f < min_new - 1e-10)
        p.loc[q[low].index, ["authoritative_use", "authoritative_reason"]] = [True, "HISTORICAL_BELOW_REFINED_WINDOW"]
        p.loc[newq.index, ["authoritative_use", "authoritative_reason"]] = [True, "CURRENT_HYBRID_INVENTORY"]

    # Deduplicate exact source/mode/load records while retaining different
    # seeds and accelerated/explicit overlap for parity.
    p["seed"] = pd.to_numeric(p.get("seed", np.nan), errors="coerce")
    p["point_key"] = (p.physical_candidate_id.astype(str) + "|" + p.integration_mode.astype(str) + "|" +
        p.normalized_f.map(lambda x: format(finite(x), ".12g")) + "|" + p.seed.fillna(-1).astype(int).astype(str) + "|" + p.point_origin.astype(str))
    p = p.sort_values(["authoritative_use", "point_origin"], ascending=[False, False]).drop_duplicates("point_key", keep="first")
    p["log10_da_dN"] = np.where(p.is_finite_rate, np.log10(pd.to_numeric(p.da_dN_m_per_cycle, errors="coerce")), np.nan)
    p["censor_marker_class"] = np.where(p.plot_kind.eq("censor"), "CYCLE_OR_HAZARD_CENSOR",
        np.where(p.plot_kind.eq("partial"), "PARTIAL_OR_NUMERICAL_UNRESOLVED", "FINITE_RATE"))
    return p.sort_values(["physical_candidate_id", "deltaK_MPa_sqrt_m", "integration_mode", "seed"]).reset_index(drop=True)


def sigma_from_deltaK(deltaK: np.ndarray | float) -> np.ndarray:
    dk = np.asarray(deltaK, dtype=float)
    kmax_pa_sqrt_m = dk * 1.0e6 / (1.0 - R_LOAD)
    return np.minimum(kmax_pa_sqrt_m / math.sqrt(2.0 * math.pi * R0_M), SIGMA_CAP_PA)


def exp_floor_barrier(row: pd.Series, deltaK: np.ndarray, prefix: str) -> np.ndarray:
    """Exact production EXP-floor surface at 300 K, evaluated at Kmax."""
    tref = finite(row.get("Tref_K"), 300.0)
    g0 = max(finite(row.get(f"{prefix}_G00_eV")) + finite(row.get(f"{prefix}_gT_eV_per_K"), 0.0) * (T_K - tref), 1e-9)
    sigc = max((finite(row.get(f"{prefix}_sigc0_GPa")) + finite(row.get(f"{prefix}_sT_GPa_per_K"), 0.0) * (T_K - tref)) * 1e9, 1.0)
    a = max(finite(row.get(f"{prefix}_exp_a")), 0.0)
    n = max(finite(row.get(f"{prefix}_exp_n")), 1e-9)
    ff = finite(row.get(f"{prefix}_floor_frac"), 0.02)
    floor = min(0.95 * g0, max(1e-4, ff * g0))
    sigma = sigma_from_deltaK(deltaK)
    return floor + (g0 - floor) * np.exp(-a * np.power(np.maximum(sigma, 0.0) / sigc, n))


def _crossings(x: np.ndarray, y: np.ndarray, level=0.0) -> list[float]:
    z = y - level; out = []
    exact = np.flatnonzero(np.isclose(z, 0.0, atol=1e-12))
    out.extend(float(x[i]) for i in exact)
    for i in np.flatnonzero(z[:-1] * z[1:] < 0):
        out.append(float(x[i] + (x[i+1] - x[i]) * (-z[i]) / (z[i+1] - z[i])))
    return sorted(set(round(v, 12) for v in out))


def log10_multihit_cleavage_rate(gc_eV: np.ndarray) -> np.ndarray:
    """Exact production multi-hit cleavage rate in stable log10 form."""
    log_lam = math.log(NU_C) - np.asarray(gc_eV, float) / (KB_EV * T_K)
    log_x = log_lam + math.log(MULTIHIT_TAU_S)
    out = np.empty_like(log_x)
    small = log_x < -18.0
    # P(m,x) ~ x^m/Gamma(m+1), lambda_eff=P/tau.
    out[small] = (MULTIHIT_M * log_x[small] - math.lgamma(MULTIHIT_M + 1.0) - math.log(MULTIHIT_TAU_S)) / math.log(10.0)
    x = np.exp(np.minimum(log_x[~small], 700.0))
    prob = np.maximum(special.gammainc(MULTIHIT_M, x), 1e-300)
    out[~small] = np.log10(prob / MULTIHIT_TAU_S)
    return out


def _interp_x_at_fraction(x: np.ndarray, g: np.ndarray, frac: float) -> float:
    g0, gf = float(g[0]), float(np.nanmin(g))
    target = gf + frac * (g0 - gf)
    idx = np.flatnonzero(g <= target)
    if not len(idx): return np.nan
    i = int(idx[0])
    if i == 0: return float(x[0])
    return float(np.interp(target, [g[i], g[i-1]], [x[i], x[i-1]]))


def barrier_descriptors(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    physical = np.linspace(0.0, 80.0, 1601)
    fgrid = np.linspace(0.0, 5.0, 1001)
    desc, curves = [], []
    for _, r in master.iterrows():
        ref = finite(r.reference_deltaK_MPa_sqrt_m)
        has_ref = math.isfinite(ref) and ref > 0
        gc = exp_floor_barrier(r, physical, "cleave")
        ge = exp_floor_barrier(r, physical, "emit")
        dgc = np.gradient(gc, physical); dge = np.gradient(ge, physical)
        cgc = np.gradient(dgc, physical); cge = np.gradient(dge, physical)
        log_gamma_e = math.log10(NU_E) - ge / (KB_EV * T_K * math.log(10.0))
        logratio_elementary = math.log10(NU_C / NU_E) - (gc - ge) / (KB_EV * T_K * math.log(10.0))
        logratio = log10_multihit_cleavage_rate(gc) - log_gamma_e
        xcross = _crossings(physical, logratio, 0.0)
        xcross_elementary = _crossings(physical, logratio_elementary, 0.0)
        row = {"candidate_id": r.candidate_id, "barrier_normalization_reference_MPa_sqrt_m": ref if has_ref else np.nan,
               "barrier_normalization_reference_imputed": False,
               "kinetic_crossover_count": len(xcross),
               "kinetic_crossover_topology": "NONE" if not xcross else ("SINGLE" if len(xcross) == 1 else "MULTIPLE"),
               "kinetic_crossover_loads_MPa_sqrt_m": ";".join(format(x, ".9g") for x in xcross),
               "K_cross_primary_MPa_sqrt_m": xcross[0] if xcross else np.nan,
               "f_cross_primary": xcross[0] / ref if xcross and has_ref else np.nan,
               "elementary_crossover_count": len(xcross_elementary),
               "K_cross_elementary_MPa_sqrt_m": xcross_elementary[0] if xcross_elementary else np.nan,
               "f_cross_elementary": xcross_elementary[0] / ref if xcross_elementary and has_ref else np.nan}
        for prefix, g, dg, cg in [("cleave", gc, dgc, cgc), ("emit", ge, dge, cge)]:
            sens = -dg
            row.update({
                f"{prefix}_G0_eV": float(g[0]), f"{prefix}_Gfloor_eV": float(np.nanmin(g)),
                f"{prefix}_available_drop_eV": float(g[0] - np.nanmin(g)),
                f"{prefix}_max_sensitivity_eV_per_MPa_sqrt_m": float(np.nanmax(sens)),
                f"{prefix}_K_at_max_sensitivity_MPa_sqrt_m": float(physical[np.nanargmax(sens)]),
                f"{prefix}_max_abs_curvature_eV_per_MPa2m": float(np.nanmax(np.abs(cg))),
                f"{prefix}_K_at_max_abs_curvature_MPa_sqrt_m": float(physical[np.nanargmax(np.abs(cg))]),
                f"{prefix}_K_drop90_MPa_sqrt_m": _interp_x_at_fraction(physical, g, .9),
                f"{prefix}_K_drop50_MPa_sqrt_m": _interp_x_at_fraction(physical, g, .5),
                f"{prefix}_K_drop10_MPa_sqrt_m": _interp_x_at_fraction(physical, g, .1),
                f"{prefix}_transition_width_MPa_sqrt_m": _interp_x_at_fraction(physical, g, .1) - _interp_x_at_fraction(physical, g, .9),
                f"{prefix}_residual_highK_slope_eV_per_MPa_sqrt_m": float(np.nanmedian(dg[-100:])),
            })
        row["relative_Ge_minus_Gc_lowK_eV"] = float(ge[0] - gc[0])
        row["relative_Ge_minus_Gc_highK_eV"] = float(ge[-1] - gc[-1])
        row["relative_max_slope_difference_eV_per_MPa_sqrt_m"] = float(np.nanmax(np.abs(dge - dgc)))
        row["relative_max_curvature_difference_eV_per_MPa2m"] = float(np.nanmax(np.abs(cge - cgc)))
        row["log10_rate_ratio_lowK"] = float(logratio[0]); row["log10_rate_ratio_highK"] = float(logratio[-1])
        row["log10_elementary_rate_ratio_lowK"] = float(logratio_elementary[0]); row["log10_elementary_rate_ratio_highK"] = float(logratio_elementary[-1])
        row["K_rate_ratio_10_MPa_sqrt_m"] = (_crossings(physical, logratio, 1.0) or [np.nan])[0]
        row["K_rate_ratio_100_MPa_sqrt_m"] = (_crossings(physical, logratio, 2.0) or [np.nan])[0]
        row["K_rate_ratio_1000_MPa_sqrt_m"] = (_crossings(physical, logratio, 3.0) or [np.nan])[0]
        if xcross:
            kx = xcross[0]; row["crossover_sharpness_dlog10R_dK"] = float(np.interp(kx, physical, np.gradient(logratio, physical)))
            band = physical[np.abs(logratio) <= 1]
            row["crossover_width_0p1_to_10_MPa_sqrt_m"] = float(band.max() - band.min()) if len(band) else np.nan
        else:
            row["crossover_sharpness_dlog10R_dK"] = np.nan; row["crossover_width_0p1_to_10_MPa_sqrt_m"] = np.nan
        desc.append(row)

        if not has_ref:
            continue
        dkf = fgrid * ref
        gcf = exp_floor_barrier(r, dkf, "cleave"); gef = exp_floor_barrier(r, dkf, "emit")
        loggef = math.log10(NU_E) - gef / (KB_EV * T_K * math.log(10.0))
        lrfe = math.log10(NU_C / NU_E) - (gcf - gef) / (KB_EV * T_K * math.log(10.0))
        lrf = log10_multihit_cleavage_rate(gcf) - loggef
        curves.extend({"candidate_id": r.candidate_id, "normalized_f": float(f), "deltaK_MPa_sqrt_m": float(k),
                       "cleavage_barrier_eV": float(a), "emission_barrier_eV": float(b),
                       "log10_Gamma_c_over_Gamma_e": float(q),
                       "log10_elementary_Gamma_c_over_Gamma_e": float(qe)}
                      for f, k, a, b, q, qe in zip(fgrid, dkf, gcf, gef, lrf, lrfe))
    return pd.DataFrame(desc), pd.DataFrame(curves)


def load_probe_descriptors() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for path in sorted(PROBES.glob("*/state_screen.json")):
        try: data = json.loads(path.read_text())
        except Exception: continue
        cid = str(data.get("candidate_id", path.parent.name))
        for point in data.get("points", []):
            rec = {}
            for k, v in point.items():
                if k in {"active_state_vector", "state_relative_changes_per_cycle"}: continue
                if isinstance(v, list):
                    # Phase-resolved diagnostic arrays are reduced explicitly;
                    # they are not candidate state vectors and are not treated
                    # as independent observations.
                    vals = [finite(x) for x in v if math.isfinite(finite(x))]
                    rec[k] = (sum(vals) if k == "hazard_actions_per_cycle" else max(vals)) if vals else np.nan
                else: rec[k] = v
            rec.update(candidate_id=cid, temperature_K=data.get("temperature_K"), reference_deltaK_MPa_sqrt_m=data.get("reference_deltaK_MPa_sqrt_m"), probe_path=str(path))
            records.append(rec)
    points = pd.DataFrame(records)
    if points.empty: return points, pd.DataFrame()
    numeric = [
        "direct_effective_barrier_derivative_eV_per_MPa_sqrt_m",
        "state_mediated_effective_barrier_derivative_eV_per_MPa_sqrt_m",
        "total_effective_barrier_derivative_eV_per_MPa_sqrt_m",
        "projected_cleavage_to_plastic_relaxation_ratio", "K_shield_MPa_sqrt_m",
        "persistent_sigma_back_mean_Pa", "tip_radius_m", "mobile_total_m2", "retained_total_m2",
        "emission_rate_peak_s", "hazard_actions_per_cycle", "cleavage_floor_proximity",
    ]
    aggs = {}
    for c in numeric:
        if c in points: aggs[c] = ["min", "median", "max"]
    d = points.groupby("candidate_id").agg(aggs)
    d.columns = [f"probe_{a}_{b}" for a, b in d.columns]
    d = d.reset_index()
    return points, d


def _collapse_curve(group: pd.DataFrame) -> pd.DataFrame:
    """One median point per mode/load; never collapse a censor into a rate."""
    g = group[group.authoritative_use].copy()
    if g.empty: return g
    rows = []
    for (f, mode), q in g.groupby(["normalized_f", "integration_mode"], dropna=False):
        finite_q = q[q.is_finite_rate]
        use = finite_q if not finite_q.empty else q
        r = use.iloc[0].copy()
        if not finite_q.empty:
            r["da_dN_m_per_cycle"] = float(np.median(finite_q.da_dN_m_per_cycle.astype(float)))
            r["log10_da_dN"] = math.log10(r.da_dN_m_per_cycle)
            r["is_finite_rate"] = True
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["normalized_f", "integration_mode"]).reset_index(drop=True)


def _contiguous_finite_segments(curve: pd.DataFrame) -> list[pd.DataFrame]:
    """Split finite data at intervening censors/partials and large load gaps."""
    if curve.empty: return []
    # Prefer explicit at a load after explicit LCF starts, but retain an
    # accelerated point where it is the only measurement.
    rows = []
    for f, q in curve.groupby("normalized_f"):
        finite_q = q[q.is_finite_rate]
        if finite_q.empty:
            rows.append(q.iloc[0])
        else:
            explicit = finite_q[finite_q.integration_mode.eq("explicit")]
            rows.append((explicit if not explicit.empty else finite_q).iloc[0])
    c = pd.DataFrame(rows).sort_values("normalized_f").reset_index(drop=True)
    # A single mode-specific censor bracketed closely by finite points is not
    # evidence of physical arrest (e.g. the bounded DBTT explicit scout at
    # f=1.05 between valid accelerated points). Two or more consecutive
    # nonfinite load levels remain a real gap and split the curve.
    keep_nonfinite = ~c.is_finite_rate.to_numpy(bool); drop_ids = []
    for i in range(1, len(c) - 1):
        if keep_nonfinite[i] and bool(c.iloc[i-1].is_finite_rate) and bool(c.iloc[i+1].is_finite_rate):
            if finite(c.iloc[i+1].normalized_f) - finite(c.iloc[i-1].normalized_f) <= .12:
                drop_ids.append(c.index[i])
    c = c.drop(drop_ids).reset_index(drop=True)
    segments, current = [], []
    prev_f = None
    for _, r in c.iterrows():
        f = finite(r.normalized_f)
        if not bool(r.is_finite_rate):
            if current: segments.append(pd.DataFrame(current)); current = []
            prev_f = None; continue
        current.append(r); prev_f = f
    if current: segments.append(pd.DataFrame(current))
    return segments


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, float); y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]; y = y[good]; n = len(x)
    empty = dict(slope=np.nan, intercept=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan,
                 r2=np.nan, n=n, span=np.nan, quality="INSUFFICIENT")
    if n < 2 or np.ptp(x) <= 0: return empty
    slope, intercept, rval, pval, se = stats.linregress(x, y)
    if n == 2:
        ci = (np.nan, np.nan); quality = "TWO_POINT_SLOPE"
    else:
        tcrit = stats.t.ppf(.975, n - 2)
        ci = (slope - tcrit * se, slope + tcrit * se); quality = "OLS_ROBUSTNESS_CHECKED"
        # A bounded Theil-Sen comparison catches an isolated extreme without
        # silently replacing the reported uncertainty model.
        robust = stats.theilslopes(y, x).slope
        if abs(robust - slope) > max(.3 * abs(slope), 3 * se):
            slope = float(robust); intercept = float(np.median(y - slope * x)); quality = "THEIL_SEN_OUTLIER_ROBUST"
    pred = intercept + slope * x
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - pred) ** 2) / ss if ss > 0 else np.nan
    return dict(slope=float(slope), intercept=float(intercept), se=float(se), ci_low=float(ci[0]), ci_high=float(ci[1]),
                r2=float(r2), n=n, span=float(np.ptp(x)), quality=quality)


def _best_two_segment(x: np.ndarray, y: np.ndarray) -> dict | None:
    n = len(x)
    if n < 6: return None
    one = _linear_fit(x, y)
    sse1 = np.sum((y - (one["intercept"] + one["slope"] * x)) ** 2)
    best = None
    for k in range(3, n - 2):
        a = _linear_fit(x[:k], y[:k]); b = _linear_fit(x[k:], y[k:])
        sse = np.sum((y[:k] - (a["intercept"] + a["slope"] * x[:k])) ** 2) + np.sum((y[k:] - (b["intercept"] + b["slope"] * x[k:])) ** 2)
        score = 1 - sse / max(sse1, 1e-30)
        item = dict(k=k, improvement=float(score), left=a, right=b,
                    breakpoint=float((x[k-1] + x[k]) / 2))
        if best is None or item["improvement"] > best["improvement"]: best = item
    return best


def _transition_width(x: np.ndarray, slopes: np.ndarray, k: float, left_slope: float, right_slope: float) -> float:
    if len(x) < 4 or not all(map(math.isfinite, [k, left_slope, right_slope])): return np.nan
    lo = min(left_slope, right_slope); hi = max(left_slope, right_slope)
    if hi - lo <= 1e-12: return np.nan
    p20, p80 = lo + .2 * (hi - lo), lo + .8 * (hi - lo)
    band = (slopes >= p20) & (slopes <= p80)
    # Use only the connected band nearest the transition, preventing a remote
    # oscillation from inflating the width.
    ids = np.flatnonzero(band)
    if not len(ids): return np.nan
    center = int(np.argmin(np.abs(x - k))); near = int(ids[np.argmin(np.abs(ids - center))])
    left = near; right = near
    while left > 0 and band[left-1]: left -= 1
    while right + 1 < len(x) and band[right+1]: right += 1
    width = float(x[right] - x[left])
    return width if width > 0 else np.nan


def _known_lcf_fraction(cid: str, curve: pd.DataFrame) -> float:
    explicit = curve[curve.integration_mode.eq("explicit") & curve.is_finite_rate]
    if explicit.empty: return np.nan
    tagged = explicit[explicit.regime_classification_source.astype(str).str.contains("LCF_EXPLICIT|NEAR_MONOTONIC", regex=True)]
    return float(tagged.normalized_f.min()) if not tagged.empty else np.nan


def morphology_descriptors(points: pd.DataFrame, master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    desc, local_rows, fit_rows = [], [], []
    refmap = dict(zip(master.candidate_id.astype(str), master.reference_deltaK_MPa_sqrt_m))
    for cid, group in points.groupby("physical_candidate_id"):
        curve = _collapse_curve(group)
        segments = _contiguous_finite_segments(curve)
        finite_segments = [s for s in segments if len(s) >= 2]
        allfinite = pd.concat(finite_segments, ignore_index=True) if finite_segments else pd.DataFrame()
        row = {"candidate_id": cid, "finite_rate_points": int(curve.is_finite_rate.sum()),
               "censor_points": int(curve.plot_kind.eq("censor").sum()),
               "partial_points": int(curve.plot_kind.eq("partial").sum()),
               "fatigue_morphology": "INSUFFICIENT_DATA", "knee_confidence": 0.0,
               "K_VHCF_HCF_MPa_sqrt_m": np.nan, "f_VHCF_HCF": np.nan,
               "K_HCF_LCF_MPa_sqrt_m": np.nan, "f_HCF_LCF": np.nan,
               "W_knee_MPa_sqrt_m": np.nan, "W_knee_f": np.nan,
               "W_LCF_MPa_sqrt_m": np.nan, "W_LCF_f": np.nan}
        if allfinite.empty:
            desc.append(row); continue
        ref = finite(refmap.get(cid))
        finite_curve = allfinite.sort_values("normalized_f").drop_duplicates(["normalized_f", "integration_mode"])
        row["dynamic_rate_range_decades"] = float(finite_curve.log10_da_dN.max() - finite_curve.log10_da_dN.min())
        row["maximum_calculated_rate_m_per_cycle"] = float(finite_curve.da_dN_m_per_cycle.max())
        row["near_monotonic_indicator"] = bool((finite_curve.da_dN_m_per_cycle >= 1e-3).any())
        row["arrest_reentry_indicator"] = len(finite_segments) > 1 and any(len(s) >= 2 for s in finite_segments[1:])
        row["developed_LCF_available"] = bool((finite_curve.integration_mode == "explicit").any())

        flcf = _known_lcf_fraction(cid, curve)
        # First infer the VHCF/HCF break independently from each continuous
        # finite sequence. Require a material improvement and a genuine slope
        # change so a many-decade smooth Arrhenius curve remains NO_CLEAR_KNEE.
        best = None
        for seg in finite_segments:
            s = seg.sort_values("normalized_f").drop_duplicates("normalized_f")
            if math.isfinite(flcf): s = s[s.normalized_f < flcf - 1e-10]
            if len(s) < 6: continue
            x = s.normalized_f.to_numpy(float); y = s.log10_da_dN.to_numpy(float)
            candidate = _best_two_segment(x, y)
            if candidate and (best is None or candidate["improvement"] > best["improvement"]):
                best = candidate | {"segment": s}
        if best is not None:
            sl, sr = best["left"]["slope"], best["right"]["slope"]
            ratio = max(abs(sl), abs(sr)) / max(min(abs(sl), abs(sr)), 1e-12)
            accepted = best["improvement"] >= .35 and ratio >= 2.0 and abs(sl - sr) >= .5
            if accepted:
                fk = best["breakpoint"]; row["f_VHCF_HCF"] = fk; row["K_VHCF_HCF_MPa_sqrt_m"] = fk * ref
                row["knee_confidence"] = float(min(1.0, best["improvement"] * min(ratio / 4, 1)))
                row["knee_fit_improvement"] = best["improvement"]
                row["knee_slope_ratio_raw"] = ratio
                s = best["segment"].sort_values("normalized_f").drop_duplicates("normalized_f")
                x = s.normalized_f.to_numpy(float); y = s.log10_da_dN.to_numpy(float)
                slopes = np.gradient(y, x)
                width_f = _transition_width(x, slopes, fk, sl, sr)
                row["W_knee_f"] = width_f; row["W_knee_MPa_sqrt_m"] = width_f * ref

        if math.isfinite(flcf):
            row["f_HCF_LCF"] = flcf; row["K_HCF_LCF_MPa_sqrt_m"] = flcf * ref
            # Percentile width between neighboring HCF and explicit-LCF local
            # slopes. Sparse curves may legitimately leave this unresolved.
            q = finite_curve.sort_values("normalized_f").drop_duplicates("normalized_f")
            if len(q) >= 4 and q.normalized_f.min() < flcf < q.normalized_f.max():
                xx = q.normalized_f.to_numpy(float); yy = q.log10_da_dN.to_numpy(float)
                ss = np.gradient(yy, xx)
                before = ss[xx < flcf]; after = ss[xx >= flcf]
                if len(before) and len(after):
                    wf = _transition_width(xx, ss, flcf, float(np.median(before[-2:])), float(np.median(after[:2])))
                    row["W_LCF_f"] = wf; row["W_LCF_MPa_sqrt_m"] = wf * ref

        # Regime assignment uses the independently detected first break and
        # explicit event-density tags for LCF. It does not infer LCF from a
        # high accelerated rate.
        fk = finite(row["f_VHCF_HCF"]); flcf = finite(row["f_HCF_LCF"])
        finite_curve = finite_curve.copy()
        finite_curve["regime"] = "HCF"
        finite_curve["near_monotonic_point"] = finite_curve.regime_classification_source.astype(str).str.contains("NEAR_MONOTONIC")
        if math.isfinite(fk): finite_curve.loc[finite_curve.normalized_f < fk, "regime"] = "VHCF"
        if math.isfinite(flcf):
            finite_curve.loc[(finite_curve.normalized_f >= flcf) & finite_curve.integration_mode.eq("explicit"), "regime"] = "LCF"
        for regime in ("VHCF", "HCF", "LCF"):
            q = finite_curve[finite_curve.regime.eq(regime)].sort_values("deltaK_MPa_sqrt_m").drop_duplicates("deltaK_MPa_sqrt_m")
            lin = _linear_fit(q.deltaK_MPa_sqrt_m.to_numpy(float), q.log10_da_dN.to_numpy(float))
            log = _linear_fit(np.log10(q.deltaK_MPa_sqrt_m.to_numpy(float)), q.log10_da_dN.to_numpy(float))
            nf = _linear_fit(q.normalized_f.to_numpy(float), q.log10_da_dN.to_numpy(float))
            for name, result in [("S_K", lin), ("m", log), ("S_f", nf)]:
                row[f"{name}_{regime}"] = result["slope"]
                row[f"{name}_{regime}_se"] = result["se"]
                row[f"{name}_{regime}_ci_low"] = result["ci_low"]
                row[f"{name}_{regime}_ci_high"] = result["ci_high"]
                row[f"{name}_{regime}_r2"] = result["r2"]
                row[f"{name}_{regime}_n"] = result["n"]
                row[f"{name}_{regime}_span"] = result["span"]
                row[f"{name}_{regime}_quality"] = result["quality"]
                fit_rows.append({"candidate_id": cid, "regime": regime, "coordinate": name, **result})
        for a, b, label in [("VHCF", "HCF", "knee"), ("HCF", "LCF", "LCF")]:
            row[f"slope_ratio_{label}_K"] = finite(row.get(f"S_K_{b}")) / finite(row.get(f"S_K_{a}")) if finite(row.get(f"S_K_{a}")) not in (0, np.nan) else np.nan
            row[f"slope_change_{label}_K"] = finite(row.get(f"S_K_{b}")) - finite(row.get(f"S_K_{a}"))
            row[f"slope_ratio_{label}_m"] = finite(row.get(f"m_{b}")) / finite(row.get(f"m_{a}")) if finite(row.get(f"m_{a}")) not in (0, np.nan) else np.nan
            row[f"slope_change_{label}_m"] = finite(row.get(f"m_{b}")) - finite(row.get(f"m_{a}"))

        # Local slope curves stay within each continuous segment.
        for segment_index, seg in enumerate(finite_segments):
            s = seg.sort_values("normalized_f").drop_duplicates("normalized_f")
            xk = s.deltaK_MPa_sqrt_m.to_numpy(float); xf = s.normalized_f.to_numpy(float); y = s.log10_da_dN.to_numpy(float)
            sk = np.gradient(y, xk) if len(s) > 2 else np.repeat(np.diff(y) / np.diff(xk), 2)
            mm = np.gradient(y, np.log10(xk)) if len(s) > 2 else np.repeat(np.diff(y) / np.diff(np.log10(xk)), 2)
            for (_, rr), aa, bb in zip(s.iterrows(), sk, mm):
                local_rows.append({"candidate_id": cid, "segment_index": segment_index,
                    "normalized_f": rr.normalized_f, "deltaK_MPa_sqrt_m": rr.deltaK_MPa_sqrt_m,
                    "log10_da_dN": rr.log10_da_dN, "local_S_K": aa, "local_m": bb})

        if row["arrest_reentry_indicator"]: row["fatigue_morphology"] = "REENTRY"
        elif not math.isfinite(fk): row["fatigue_morphology"] = "NO_CLEAR_KNEE"
        elif not math.isfinite(flcf): row["fatigue_morphology"] = "NO_RESOLVED_LCF"
        elif row["near_monotonic_indicator"]: row["fatigue_morphology"] = "HCF_LCF_NEAR_MONOTONIC"
        else: row["fatigue_morphology"] = "KNEE_AND_LCF_UPTURN"
        row["LCF_upturn_indicator"] = math.isfinite(flcf)
        desc.append(row)
    return pd.DataFrame(desc), pd.DataFrame(local_rows), pd.DataFrame(fit_rows)


def descriptors_at_transitions(morph: pd.DataFrame, barriers: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bgroups = {cid: g.sort_values("deltaK_MPa_sqrt_m") for cid, g in barriers.groupby("candidate_id")}
    pgroups = {cid: g.sort_values("fraction") for cid, g in probes.groupby("candidate_id")} if not probes.empty else {}
    for _, r in morph.iterrows():
        cid = r.candidate_id; bg = bgroups.get(cid)
        if bg is None: continue
        out = {"candidate_id": cid}
        x = bg.deltaK_MPa_sqrt_m.to_numpy(float)
        for transition, kcol, fcol in [("knee", "K_VHCF_HCF_MPa_sqrt_m", "f_VHCF_HCF"), ("LCF", "K_HCF_LCF_MPa_sqrt_m", "f_HCF_LCF")]:
            k = finite(r.get(kcol)); f = finite(r.get(fcol))
            if not math.isfinite(k): continue
            for c, short in [("cleavage_barrier_eV", "Gc_eV"), ("emission_barrier_eV", "Ge_eV"), ("log10_Gamma_c_over_Gamma_e", "log10_rate_ratio")]:
                out[f"{transition}_{short}"] = float(np.interp(k, x, bg[c]))
            for c, short in [("cleavage_barrier_eV", "dGc_dK"), ("emission_barrier_eV", "dGe_dK")]:
                vals = bg[c].to_numpy(float); d1 = np.gradient(vals, x); d2 = np.gradient(d1, x)
                out[f"{transition}_{short}"] = float(np.interp(k, x, d1))
                out[f"{transition}_d2{short[1:]}2"] = float(np.interp(k, x, d2))
            out[f"{transition}_cleavage_floor_proximity"] = (out[f"{transition}_Gc_eV"] - float(bg.cleavage_barrier_eV.min())) / max(float(bg.cleavage_barrier_eV.max() - bg.cleavage_barrier_eV.min()), 1e-30)
            out[f"{transition}_emission_floor_proximity"] = (out[f"{transition}_Ge_eV"] - float(bg.emission_barrier_eV.min())) / max(float(bg.emission_barrier_eV.max() - bg.emission_barrier_eV.min()), 1e-30)
            if cid in pgroups:
                pg = pgroups[cid]
                for c in ["direct_effective_barrier_derivative_eV_per_MPa_sqrt_m", "state_mediated_effective_barrier_derivative_eV_per_MPa_sqrt_m",
                          "total_effective_barrier_derivative_eV_per_MPa_sqrt_m", "projected_cleavage_to_plastic_relaxation_ratio",
                          "K_shield_MPa_sqrt_m", "persistent_sigma_back_mean_Pa", "tip_radius_m", "mobile_total_m2", "retained_total_m2"]:
                    if c in pg:
                        good = np.isfinite(pd.to_numeric(pg[c], errors="coerce"))
                        if good.any(): out[f"{transition}_probe_{c}"] = float(np.interp(f, pg.loc[good, "fraction"], pg.loc[good, c]))
        rows.append(out)
    return pd.DataFrame(rows)


def _fdr_bh(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), float); q = np.full(len(p), np.nan)
    good = np.flatnonzero(np.isfinite(p))
    if not len(good): return q
    order = good[np.argsort(p[good])]; ranked = p[order] * len(good) / np.arange(1, len(good) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q


def _corr_ci(r: float, n: int) -> tuple[float, float]:
    if n <= 3 or not math.isfinite(r) or abs(r) >= 1: return np.nan, np.nan
    z = np.arctanh(r); se = 1 / math.sqrt(n - 3)
    lo, hi = np.tanh([z - 1.96 * se, z + 1.96 * se])
    return float(lo), float(hi)


def correlation_tables(corr_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictors = [c for c in corr_master.columns if c.startswith(("cleave_", "emit_", "relative_", "kinetic_", "K_cross", "K_rate_ratio_", "f_cross", "crossover_", "log10_rate_ratio_", "probe_", "knee_dG", "LCF_dG"))]
    responses = [c for c in [
        "K_VHCF_HCF_MPa_sqrt_m", "f_VHCF_HCF", "K_HCF_LCF_MPa_sqrt_m", "f_HCF_LCF",
        "W_knee_MPa_sqrt_m", "W_knee_f", "W_LCF_MPa_sqrt_m", "W_LCF_f",
        "S_K_VHCF", "S_K_HCF", "S_K_LCF", "S_f_VHCF", "S_f_HCF", "S_f_LCF",
        "m_VHCF", "m_HCF", "m_LCF", "slope_change_knee_K", "slope_change_LCF_K",
        "slope_ratio_knee_K", "slope_ratio_LCF_K", "slope_change_knee_m", "slope_change_LCF_m",
        "slope_ratio_knee_m", "slope_ratio_LCF_m", "dynamic_rate_range_decades",
        "maximum_calculated_rate_m_per_cycle", "arrest_reentry_indicator", "LCF_upturn_indicator",
        "near_monotonic_indicator", "spatial_bifurcation_indicator",
    ] if c in corr_master]
    subsets = {
        "ALL_FATIGUE": np.ones(len(corr_master), bool),
        "REDUCED_VALID": corr_master.spatial_validation_class.eq("REDUCED_VALID").to_numpy(),
        "EXCLUDE_ABCD_EXTREMES": ~corr_master.candidate_plot_class.isin(list("ABCD")).to_numpy(),
        "HISTORICAL_HOLDOUT": corr_master.candidate_plot_class.isin(["DBTT", "Peak-T", "weak-T", "ceramic-like"]).to_numpy(),
    }
    rows = []
    for subset, mask in subsets.items():
        d = corr_master.loc[mask]
        for pred in predictors:
            for resp in responses:
                x = pd.to_numeric(d[pred], errors="coerce").to_numpy(float); y = pd.to_numeric(d[resp], errors="coerce").to_numpy(float)
                good = np.isfinite(x) & np.isfinite(y); n = int(good.sum())
                if n < 3 or np.ptp(x[good]) <= 0 or np.ptp(y[good]) <= 0:
                    rows.append({"subset": subset, "predictor": pred, "response": resp, "n": n,
                        "pearson_r": np.nan, "pearson_p": np.nan, "pearson_ci_low": np.nan, "pearson_ci_high": np.nan,
                        "spearman_rho": np.nan, "spearman_p": np.nan, "spearman_ci_low": np.nan, "spearman_ci_high": np.nan,
                        "test_status": "INSUFFICIENT_N_OR_VARIATION"})
                    continue
                pear = stats.pearsonr(x[good], y[good]); spear = stats.spearmanr(x[good], y[good])
                pl, ph = _corr_ci(float(pear.statistic), n); sl, sh = _corr_ci(float(spear.statistic), n)
                pp, sp = (pear.pvalue, spear.pvalue) if n >= 5 else (np.nan, np.nan)
                rows.append({"subset": subset, "predictor": pred, "response": resp, "n": n,
                    "pearson_r": pear.statistic, "pearson_p": pp, "pearson_ci_low": pl, "pearson_ci_high": ph,
                    "spearman_rho": spear.statistic, "spearman_p": sp, "spearman_ci_low": sl, "spearman_ci_high": sh,
                    "test_status": "EXPLORATORY_N3_N4" if n < 5 else "TESTED"})
    corrs = pd.DataFrame(rows)
    if not corrs.empty:
        corrs["pearson_q_fdr"] = _fdr_bh(corrs.pearson_p)
        corrs["spearman_q_fdr"] = _fdr_bh(corrs.spearman_p)

    # Partial correlations residualize both variables against scale controls.
    partial = []
    controls = ["reference_deltaK_MPa_sqrt_m", "fracture_resistance_300K_MPa_sqrt_m", "cleave_G0_eV"]
    for pred in predictors:
        for resp in responses:
            cols = [pred, resp] + [c for c in controls if c in corr_master and c not in {pred, resp}]
            q = corr_master[cols].apply(pd.to_numeric, errors="coerce").dropna()
            if len(q) < max(8, len(cols) + 3) or q[pred].nunique() < 2 or q[resp].nunique() < 2: continue
            z = np.column_stack([np.ones(len(q))] + [q[c].to_numpy(float) for c in cols[2:]])
            rx = q[pred].to_numpy(float) - z @ np.linalg.lstsq(z, q[pred].to_numpy(float), rcond=None)[0]
            ry = q[resp].to_numpy(float) - z @ np.linalg.lstsq(z, q[resp].to_numpy(float), rcond=None)[0]
            if np.ptp(rx) <= 0 or np.ptp(ry) <= 0: continue
            pr = stats.pearsonr(rx, ry); sr = stats.spearmanr(rx, ry)
            partial.append({"predictor": pred, "response": resp, "controls": ";".join(cols[2:]), "n": len(q),
                            "partial_pearson_r": pr.statistic, "partial_pearson_p": pr.pvalue,
                            "partial_spearman_rho": sr.statistic, "partial_spearman_p": sr.pvalue})
    partial = pd.DataFrame(partial)
    if not partial.empty:
        partial["partial_pearson_q_fdr"] = _fdr_bh(partial.partial_pearson_p)
        partial["partial_spearman_q_fdr"] = _fdr_bh(partial.partial_spearman_p)
    return corrs, partial


def pca_table(matrix: np.ndarray, ids: list[str], prefix: str, n_components=5) -> tuple[pd.DataFrame, dict]:
    x = np.asarray(matrix, float)
    means = np.nanmean(x, axis=0); stds = np.nanstd(x, axis=0); stds[stds == 0] = 1
    x = np.where(np.isfinite(x), x, means)
    z = (x - means) / stds
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    k = min(n_components, len(s)); scores = u[:, :k] * s[:k]
    var = s**2 / max(len(x) - 1, 1); ratio = var / max(var.sum(), 1e-30)
    out = pd.DataFrame({"candidate_id": ids})
    for j in range(k): out[f"{prefix}_PC{j+1}"] = scores[:, j]
    meta = {"explained_variance_ratio": ratio[:k].tolist(), "loadings": vt[:k].tolist(), "feature_count": x.shape[1]}
    return out, meta


def functional_pca(master: pd.DataFrame, barrier_curves: pd.DataFrame, points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # Barrier PCA uses the exact common normalized grid and both surfaces.
    ids, mats = [], []
    for cid, g in barrier_curves.groupby("candidate_id", sort=False):
        g = g[(g.normalized_f >= 0) & (g.normalized_f <= 3)].sort_values("normalized_f")
        ids.append(cid); mats.append(np.r_[g.cleavage_barrier_eV.to_numpy(float), g.emission_barrier_eV.to_numpy(float)])
    bscore, bmeta = pca_table(np.vstack(mats), ids, "barrier")

    # Fatigue PCA is restricted to a continuous measured segment spanning a
    # common 0.95--1.20 interval. No censor/arrest gap is interpolated.
    fgrid = np.linspace(.95, 1.20, 51); fids, fmats = [], []
    for cid, group in points.groupby("physical_candidate_id"):
        curve = _collapse_curve(group); segs = _contiguous_finite_segments(curve)
        candidates = []
        for s in segs:
            s = s.sort_values("normalized_f").drop_duplicates("normalized_f")
            if len(s) >= 4 and s.normalized_f.min() <= fgrid.min() and s.normalized_f.max() >= fgrid.max(): candidates.append(s)
        if not candidates: continue
        s = max(candidates, key=len)
        fids.append(cid); fmats.append(np.interp(fgrid, s.normalized_f, s.log10_da_dN))
    if len(fids) >= 3:
        fscore, fmeta = pca_table(np.vstack(fmats), fids, "fatigue")
    else:
        fscore = pd.DataFrame(columns=["candidate_id", "fatigue_PC1", "fatigue_PC2"]); fmeta = {"explained_variance_ratio": [], "feature_count": len(fgrid)}
    return bscore, fscore, {"barrier": bmeta, "fatigue": fmeta, "fatigue_grid_f": fgrid.tolist()}


def _kmeans(x: np.ndarray, k: int, seed=914, iterations=200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + k)
    centers = x[rng.choice(len(x), k, replace=False)].copy(); labels = np.zeros(len(x), int)
    for _ in range(iterations):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2); new = dist.argmin(axis=1)
        if np.array_equal(new, labels): break
        labels = new
        for j in range(k): centers[j] = x[labels == j].mean(axis=0) if np.any(labels == j) else x[rng.integers(len(x))]
    return labels, centers


def _silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels)) < 2: return np.nan
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)); vals = []
    for i in range(len(x)):
        same = labels == labels[i]; same[i] = False
        a = d[i, same].mean() if same.any() else 0
        b = min(d[i, labels == j].mean() for j in set(labels) if j != labels[i])
        vals.append((b - a) / max(a, b, 1e-30))
    return float(np.mean(vals))


def cluster_morphologies(corr_master: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    features = [c for c in ["S_K_VHCF", "S_K_HCF", "S_K_LCF", "m_VHCF", "m_HCF", "m_LCF",
        "f_VHCF_HCF", "f_HCF_LCF", "W_knee_f", "W_LCF_f", "dynamic_rate_range_decades",
        "slope_change_knee_K", "slope_change_LCF_K"] if c in corr_master]
    q = corr_master[["candidate_id", "fatigue_morphology", "candidate_plot_class"] + features].copy()
    q["observed_feature_count"] = q[features].notna().sum(axis=1)
    q = q[q.observed_feature_count >= 3].reset_index(drop=True)
    if len(q) < 4:
        q["cluster"] = np.nan; return q, {"selected_k": None, "silhouette": None, "features": features}
    x = q[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    present = np.isfinite(x).any(axis=0); x = x[:, present]; features = list(np.asarray(features)[present])
    med = np.nanmedian(x, axis=0); x = np.where(np.isfinite(x), x, med)
    sd = np.std(x, axis=0); keep = sd > 0; x = (x[:, keep] - np.mean(x[:, keep], axis=0)) / sd[keep]
    best = None
    for k in range(2, min(8, len(q) - 1) + 1):
        labels, centers = _kmeans(x, k); score = _silhouette(x, labels)
        if best is None or score > best[0]: best = (score, k, labels, centers)
    q["cluster"] = best[2] + 1
    return q, {"selected_k": best[1], "silhouette": best[0], "features": list(np.asarray(features)[keep])}


def ridge_models(corr_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictors = [c for c in ["cleave_max_sensitivity_eV_per_MPa_sqrt_m", "emit_max_sensitivity_eV_per_MPa_sqrt_m",
        "cleave_transition_width_MPa_sqrt_m", "emit_transition_width_MPa_sqrt_m", "f_cross_primary",
        "crossover_sharpness_dlog10R_dK", "crossover_width_0p1_to_10_MPa_sqrt_m",
        "relative_Ge_minus_Gc_lowK_eV", "relative_Ge_minus_Gc_highK_eV", "log10_rate_ratio_highK"] if c in corr_master]
    responses = [c for c in ["f_VHCF_HCF", "f_HCF_LCF", "S_K_VHCF", "S_K_HCF", "S_K_LCF", "W_knee_f", "W_LCF_f"] if c in corr_master]
    perf, importance = [], []
    for response in responses:
        q = corr_master[predictors + [response]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(q) < max(10, len(predictors) + 2): continue
        x = q[predictors].to_numpy(float); y = q[response].to_numpy(float)
        xm, xs = x.mean(0), x.std(0); xs[xs == 0] = 1; ym, ys = y.mean(), y.std() or 1
        x = (x - xm) / xs; y = (y - ym) / ys
        folds = np.arange(len(y)) % 5; best = None
        for alpha in np.logspace(-4, 3, 20):
            preds = np.empty(len(y))
            for fold in range(5):
                tr = folds != fold; te = ~tr
                beta = np.linalg.solve(x[tr].T @ x[tr] + alpha * np.eye(x.shape[1]), x[tr].T @ y[tr])
                preds[te] = x[te] @ beta
            rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
            if best is None or rmse < best[0]: best = (rmse, alpha, preds)
        alpha = best[1]; beta = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
        r2 = 1 - np.sum((best[2] - y)**2) / max(np.sum((y-y.mean())**2), 1e-30)
        perf.append({"response": response, "model": "ridge_5fold_cv", "n": len(y), "alpha": alpha, "cv_standardized_rmse": best[0], "cv_r2": r2})
        for p, b in zip(predictors, beta): importance.append({"response": response, "feature": p, "standardized_coefficient": b, "absolute_importance": abs(b), "interpretation": "association_not_causation"})
    return pd.DataFrame(perf), pd.DataFrame(importance)


def _savefig(fig: plt.Figure, out: Path, name: str, data: pd.DataFrame) -> None:
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    data.to_csv(out / f"{name}_plot_data.csv", index=False)


def _scatter(ax, data: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, annotate=False) -> pd.DataFrame:
    q = data[["candidate_id", "candidate_plot_class", "fatigue_morphology", x, y]].copy()
    q[x] = pd.to_numeric(q[x], errors="coerce"); q[y] = pd.to_numeric(q[y], errors="coerce"); q = q.dropna()
    for klass, g in q.groupby("candidate_plot_class"):
        ax.scatter(g[x], g[y], s=42 if klass != "SMOOTH_ARRHENIUS" else 22,
                   color=CLASS_COLORS.get(klass, "#999999"), alpha=.85, label=klass, edgecolor="white", linewidth=.35)
        if annotate and klass != "SMOOTH_ARRHENIUS":
            for _, r in g.iterrows(): ax.annotate(str(r.candidate_id).split("_")[-1], (r[x], r[y]), fontsize=6)
    if len(q) >= 3 and q[x].nunique() > 1:
        fit = _linear_fit(q[x].to_numpy(float), q[y].to_numpy(float)); xx = np.linspace(q[x].min(), q[x].max(), 100)
        ax.plot(xx, fit["intercept"] + fit["slope"] * xx, color="0.25", lw=1.2, ls="--")
        rho = stats.spearmanr(q[x], q[y]).statistic
        ax.text(.03, .97, f"n={len(q)}, Spearman ρ={rho:.2f}", transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(False)
    return q


def _heatmap(corrs: pd.DataFrame, responses: list[str], predictors: list[str], out: Path, name: str) -> None:
    q = corrs[(corrs.subset == "ALL_FATIGUE") & corrs.response.isin(responses) & corrs.predictor.isin(predictors)].copy()
    mat = q.pivot(index="predictor", columns="response", values="spearman_rho").reindex(index=predictors, columns=responses)
    fig, ax = plt.subplots(figsize=(max(8, 1.15*len(responses)), max(5, .52*len(predictors))))
    im = ax.imshow(mat.to_numpy(float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(responses)), [x.replace("_", "\n") for x in responses], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(predictors)), [x.replace("_", " ") for x in predictors], fontsize=7)
    for i, p in enumerate(predictors):
        for j, r in enumerate(responses):
            cell = q[(q.predictor == p) & (q.response == r)]
            if cell.empty or int(cell.iloc[0].n) < 4: continue
            val = finite(cell.iloc[0].spearman_rho)
            if not math.isfinite(val): continue
            sig = "*" if finite(cell.iloc[0].spearman_q_fdr, 1) <= .1 else ""
            ax.text(j, i, f"{val:.2f}{sig}\nn={int(cell.iloc[0].n)}", ha="center", va="center", fontsize=6,
                    color="white" if abs(val) > .55 else "black")
    fig.colorbar(im, ax=ax, label="Spearman ρ (* FDR q≤0.10)")
    fig.tight_layout(); _savefig(fig, out, name, q)


def make_figures(out: Path, corr_master: pd.DataFrame, corrs: pd.DataFrame,
                 bscore: pd.DataFrame, fscore: pd.DataFrame, clusters: pd.DataFrame,
                 barrier_curves: pd.DataFrame, points: pd.DataFrame) -> None:
    slope_responses = [c for c in ["S_K_VHCF", "S_K_HCF", "S_K_LCF", "m_VHCF", "m_HCF", "m_LCF"] if c in corr_master]
    loc_responses = [c for c in ["K_VHCF_HCF_MPa_sqrt_m", "f_VHCF_HCF", "K_HCF_LCF_MPa_sqrt_m", "f_HCF_LCF"] if c in corr_master]
    width_responses = [c for c in ["W_knee_MPa_sqrt_m", "W_knee_f", "W_LCF_MPa_sqrt_m", "W_LCF_f"] if c in corr_master]
    heat_predictors = [c for c in ["cleave_max_sensitivity_eV_per_MPa_sqrt_m", "emit_max_sensitivity_eV_per_MPa_sqrt_m",
        "cleave_transition_width_MPa_sqrt_m", "emit_transition_width_MPa_sqrt_m", "relative_Ge_minus_Gc_lowK_eV",
        "relative_Ge_minus_Gc_highK_eV", "f_cross_primary", "crossover_sharpness_dlog10R_dK",
        "crossover_width_0p1_to_10_MPa_sqrt_m", "log10_rate_ratio_highK"] if c in corr_master]
    _heatmap(corrs, slope_responses, heat_predictors, out, "correlation_heatmap_regime_slopes")
    _heatmap(corrs, loc_responses, heat_predictors, out, "correlation_heatmap_transition_locations")
    _heatmap(corrs, width_responses, heat_predictors, out, "correlation_heatmap_transition_widths")

    # Keep kinetic competition separate from the bare-barrier heatmaps.  This
    # makes the middle link in barrier geometry -> competition -> morphology
    # explicit rather than hiding it among dozens of barrier parameters.
    kinetic_predictors = [c for c in [
        "f_cross_primary", "K_cross_primary_MPa_sqrt_m",
        "crossover_sharpness_dlog10R_dK", "crossover_width_0p1_to_10_MPa_sqrt_m",
        "log10_rate_ratio_lowK", "log10_rate_ratio_highK",
        "K_rate_ratio_0p01_MPa_sqrt_m", "K_rate_ratio_1_MPa_sqrt_m",
        "K_rate_ratio_100_MPa_sqrt_m",
    ] if c in corr_master]
    morphology_responses = [c for c in [
        "K_VHCF_HCF_MPa_sqrt_m", "K_HCF_LCF_MPa_sqrt_m",
        "S_K_VHCF", "S_K_HCF", "S_K_LCF", "W_knee_f", "W_LCF_f",
        "slope_change_knee_K", "slope_change_LCF_K", "dynamic_rate_range_decades",
        "arrest_reentry_indicator", "LCF_upturn_indicator", "near_monotonic_indicator",
        "spatial_bifurcation_indicator",
    ] if c in corr_master]
    _heatmap(corrs, morphology_responses, kinetic_predictors, out, "correlation_heatmap_kinetic_competition")

    specs = [
        ("barrier_crossover_vs_knee_location", "f_cross_primary", "f_VHCF_HCF", r"kinetic crossover $f_\times$", r"VHCF/HCF knee $f$"),
        ("barrier_crossover_vs_lcf_location", "f_cross_primary", "f_HCF_LCF", r"kinetic crossover $f_\times$", r"HCF/LCF transition $f$"),
        ("vhcf_slope_vs_barrier_sensitivity", "cleave_max_sensitivity_eV_per_MPa_sqrt_m", "S_K_VHCF", r"max cleavage sensitivity (eV / MPa$\sqrt{m}$)", r"$S_{VHCF}$"),
        ("hcf_slope_vs_effective_barrier_sensitivity", "probe_total_effective_barrier_derivative_eV_per_MPa_sqrt_m_median", "S_K_HCF", r"median evolved effective $dG/dK$ (saved probe)", r"$S_{HCF}$"),
        ("lcf_slope_vs_highK_barrier_sensitivity", "LCF_dGc_dK", "S_K_LCF", r"cleavage $dG/dK$ at LCF transition", r"$S_{LCF}$"),
        ("knee_slope_change_vs_crossover_sharpness", "crossover_sharpness_dlog10R_dK", "slope_change_knee_K", r"crossover sharpness $d\log_{10}R/dK$", r"$\Delta S_{knee}$"),
        ("lcf_slope_recovery_vs_rate_ratio", "LCF_log10_rate_ratio", "slope_change_LCF_K", r"$\log_{10}(\Gamma_c/\Gamma_e)$ at LCF", r"$\Delta S_{LCF}$"),
    ]
    for name, x, y, xl, yl in specs:
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        if x in corr_master and y in corr_master: pdata = _scatter(ax, corr_master, x, y, xl, yl, True)
        else:
            pdata = pd.DataFrame(columns=["candidate_id", x, y]); ax.text(.5, .5, "insufficient existing data", ha="center", va="center")
            ax.set_xlabel(xl); ax.set_ylabel(yl)
        handles, labels = ax.get_legend_handles_labels()
        if handles: ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout(); _savefig(fig, out, name, pdata)

    # Physics phase map.
    fig, ax = plt.subplots(figsize=(8, 6))
    phase = corr_master.dropna(subset=["f_cross_primary", "crossover_sharpness_dlog10R_dK"]).copy()
    marker = {"REENTRY": "D", "NO_CLEAR_KNEE": "o", "NO_RESOLVED_LCF": "s", "KNEE_AND_LCF_UPTURN": "^", "HCF_LCF_NEAR_MONOTONIC": "P"}
    for morph, g in phase.groupby("fatigue_morphology"):
        ax.scatter(g.f_cross_primary, g.crossover_sharpness_dlog10R_dK, marker=marker.get(morph, "o"), s=44, alpha=.75, label=morph)
    ax.set_xlabel(r"$K_\times/\Delta K_{ref}$"); ax.set_ylabel(r"$d\log_{10}(\Gamma_c/\Gamma_e)/dK|_\times$")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.legend(fontsize=7, frameon=False); fig.tight_layout(); _savefig(fig, out, "barrier_geometry_fatigue_phase_map", phase)

    # PCA and cluster maps.
    bm = bscore.merge(corr_master[["candidate_id", "candidate_plot_class"]], on="candidate_id", how="left")
    fig, ax = plt.subplots(figsize=(7.2, 5.4)); pdata = _scatter(ax, bm.assign(fatigue_morphology=""), "barrier_PC1", "barrier_PC2", "barrier PC1", "barrier PC2")
    fig.tight_layout(); _savefig(fig, out, "barrier_shape_pca", pdata)
    fm = fscore.merge(corr_master[["candidate_id", "candidate_plot_class", "fatigue_morphology"]], on="candidate_id", how="left")
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    pdata = _scatter(ax, fm, "fatigue_PC1", "fatigue_PC2", "fatigue PC1", "fatigue PC2") if not fm.empty else pd.DataFrame()
    fig.tight_layout(); _savefig(fig, out, "fatigue_shape_pca", pdata)
    mode = bm.merge(fm[["candidate_id", "fatigue_PC1"]], on="candidate_id", how="inner")
    fig, ax = plt.subplots(figsize=(7.2, 5.4)); pdata = _scatter(ax, mode.assign(fatigue_morphology=""), "barrier_PC1", "fatigue_PC1", "barrier PC1", "fatigue PC1") if not mode.empty else pd.DataFrame()
    fig.tight_layout(); _savefig(fig, out, "barrier_mode_vs_fatigue_mode", pdata)
    cm = clusters.merge(fscore, on="candidate_id", how="left")
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    if {"fatigue_PC1", "fatigue_PC2"}.issubset(cm):
        for cl, g in cm.groupby("cluster"): ax.scatter(g.fatigue_PC1, g.fatigue_PC2, s=45, label=f"cluster {cl}")
        ax.legend(frameon=False); ax.set_xlabel("fatigue PC1"); ax.set_ylabel("fatigue PC2")
    else: ax.text(.5, .5, "insufficient common fatigue-PCA support", ha="center")
    fig.tight_layout(); _savefig(fig, out, "fatigue_morphology_cluster_map", cm)

    # Top 3 physically meaningful univariate associations for each primary response.
    primaries = [c for c in ["K_VHCF_HCF_MPa_sqrt_m", "K_HCF_LCF_MPa_sqrt_m", "S_K_VHCF", "S_K_HCF", "S_K_LCF",
        "m_VHCF", "m_HCF", "m_LCF", "W_knee_f", "W_LCF_f"] if c in corr_master]
    for response in primaries:
        top = corrs[(corrs.subset == "ALL_FATIGUE") & (corrs.response == response) & (corrs.n >= 4)].copy()
        top["rank_score"] = top.spearman_rho.abs() * np.sqrt(top.n)
        top = top.sort_values("rank_score", ascending=False).drop_duplicates("predictor").head(3)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2)); pdata = []
        for ax, (_, rr) in zip(axes, top.iterrows()): pdata.append(_scatter(ax, corr_master, rr.predictor, response, rr.predictor.replace("_", " "), response.replace("_", " ")))
        for ax in axes[len(top):]: ax.axis("off")
        fig.tight_layout(); data = pd.concat(pdata, ignore_index=True) if pdata else pd.DataFrame()
        _savefig(fig, out, f"top_correlations_{response}", data)

    # Representative exact barrier/fatigue overlays.
    reps = list(CONTROL_LABELS) + list(CANONICAL)
    fig, axes = plt.subplots(len(reps), 2, figsize=(13, 3.0 * len(reps)))
    overlay_data = []
    for i, cid in enumerate(reps):
        label = CONTROL_LABELS.get(cid, CANONICAL.get(cid, cid)); bg = barrier_curves[barrier_curves.candidate_id.eq(cid)]
        fg = points[(points.physical_candidate_id.eq(cid)) & points.authoritative_use]
        if not bg.empty:
            b = bg.iloc[::10].copy(); b["representative"] = label; overlay_data.append(b)
            axes[i,0].plot(b.deltaK_MPa_sqrt_m, b.cleavage_barrier_eV, label="$G_c$")
            axes[i,0].plot(b.deltaK_MPa_sqrt_m, b.emission_barrier_eV, label="$G_e$")
            axes[i,0].set_ylabel(f"{label}\nbarrier (eV)"); axes[i,0].legend(frameon=False, fontsize=7)
        finite_fg = fg[fg.is_finite_rate]
        for mode, q in finite_fg.groupby("integration_mode"):
            axes[i,1].scatter(q.deltaK_MPa_sqrt_m, q.da_dN_m_per_cycle, s=24, label=mode)
        cens = fg[fg.plot_kind.eq("censor")]
        if not cens.empty:
            floor = max(1e-22, finite_fg.da_dN_m_per_cycle.min()/10) if not finite_fg.empty else 1e-22
            axes[i,1].scatter(cens.deltaK_MPa_sqrt_m, np.full(len(cens), floor), marker="v", facecolors="none", edgecolors="k")
        info = corr_master[corr_master.candidate_id.eq(cid)]
        if not info.empty:
            ir = info.iloc[0]
            for col, color, style, tag in [
                ("K_VHCF_HCF_MPa_sqrt_m", "#009E73", "--", "VHCF/HCF"),
                ("K_HCF_LCF_MPa_sqrt_m", "#D55E00", ":", "HCF/LCF"),
                ("K_cross_primary_MPa_sqrt_m", "#7A5195", "-.", "kinetic crossover"),
            ]:
                value = finite(ir.get(col))
                if math.isfinite(value):
                    axes[i,0].axvline(value, color=color, ls=style, lw=1, alpha=.8)
                    axes[i,1].axvline(value, color=color, ls=style, lw=1, alpha=.8, label=tag)
        axes[i,1].set_yscale("log"); axes[i,1].set_ylabel(r"$da/dN$ (m/cycle)"); axes[i,1].legend(frameon=False, fontsize=7)
    axes[-1,0].set_xlabel(r"$\Delta K$ (MPa $\sqrt{m}$)"); axes[-1,1].set_xlabel(r"$\Delta K$ (MPa $\sqrt{m}$)")
    fig.tight_layout(); _savefig(fig, out, "representative_barrier_fatigue_overlays", pd.concat(overlay_data, ignore_index=True))


def _association(corrs: pd.DataFrame, response: str, predictor: str) -> dict | None:
    q = corrs[(corrs.subset == "ALL_FATIGUE") & (corrs.response == response) & (corrs.predictor == predictor)]
    return q.iloc[0].to_dict() if not q.empty else None


def hypothesis_table(corrs: pd.DataFrame) -> pd.DataFrame:
    specs = [
        (1, "VHCF/HCF knee tracks cleavage/emission crossover", "f_VHCF_HCF", "f_cross_primary"),
        (2, "Knee width decreases with crossover sharpness", "W_knee_f", "crossover_sharpness_dlog10R_dK"),
        (3, "Knee slope reduction follows state-mediated sensitivity", "slope_change_knee_K", "probe_state_mediated_effective_barrier_derivative_eV_per_MPa_sqrt_m_max"),
        (4, "LCF upturn follows cleavage dominance", "f_HCF_LCF", "K_rate_ratio_100_MPa_sqrt_m"),
        (5, "LCF slope follows high-K cleavage derivative", "S_K_LCF", "LCF_dGc_dK"),
        (6, "Arrest/re-entry follows prolonged emission competition", "arrest_reentry_indicator", "log10_rate_ratio_highK"),
        (7, "Spatial bifurcation follows blunting/backstress attractor", "spatial_bifurcation_indicator", "probe_tip_radius_m_max"),
    ]
    rows = []
    for number, text, response, predictor in specs:
        a = _association(corrs, response, predictor)
        if a is None:
            verdict, reason = "WEAK_SUPPORT", "Existing paired sample is insufficient for a correlation test."
            n = 0; rho = p = q = np.nan
        else:
            n = int(a["n"]); rho = finite(a["spearman_rho"]); p = finite(a["spearman_p"]); q = finite(a["spearman_q_fdr"])
            if n >= 8 and q <= .1 and abs(rho) >= .5: verdict = "SUPPORT"
            elif n >= 8 and abs(rho) < .2: verdict = "REJECTION"
            else: verdict = "WEAK_SUPPORT"
            reason = f"Spearman rho={rho:.3g}, n={n}, raw p={p:.3g}, FDR q={q:.3g}; association is not causal evidence."
        rows.append({"hypothesis": number, "statement": text, "response": response, "predictor": predictor,
                     "verdict": verdict, "n": n, "spearman_rho": rho, "p": p, "q_fdr": q, "reason": reason})
    return pd.DataFrame(rows)


def write_report(out: Path, master: pd.DataFrame, points: pd.DataFrame, morph: pd.DataFrame,
                 corrs: pd.DataFrame, partial: pd.DataFrame, hypotheses: pd.DataFrame,
                 pca_meta: dict, cluster_meta: dict, model_perf: pd.DataFrame) -> None:
    def best(response: str) -> str:
        q = corrs[(corrs.subset == "ALL_FATIGUE") & (corrs.response == response) & (corrs.n >= 4)].copy()
        if q.empty: return "No adequately paired existing-data relationship."
        q["score"] = q.spearman_rho.abs() * np.sqrt(q.n); r = q.sort_values("score", ascending=False).iloc[0]
        return f"`{r.predictor}` (Spearman ρ={r.spearman_rho:.3g}, n={int(r.n)}, FDR q={r.spearman_q_fdr:.3g})."
    def specific(response: str, predictor: str) -> str:
        r = _association(corrs, response, predictor)
        if r is None: return f"`{predictor}` has insufficient paired existing data."
        return f"`{predictor}` vs `{response}`: Spearman ρ={finite(r['spearman_rho']):.3g}, n={int(r['n'])}, FDR q={finite(r['spearman_q_fdr']):.3g}."
    hlines = "\n".join(f"- H{int(r.hypothesis)} **{r.verdict}**: {r.statement}. {r.reason}" for _, r in hypotheses.iterrows())
    source_counts = master.source_registry.value_counts().to_dict()
    morph_counts = morph.fatigue_morphology.value_counts(dropna=False).to_dict()
    authoritative = points[points.authoritative_use]
    invalid_high = int((points.authoritative_reason == "ACCELERATED_HIGH_K_UNVALIDATED_FOR_LCF").sum())
    reduced_n = int((master.spatial_validation_class == "REDUCED_VALID").sum())
    report = rf"""# Barrier geometry → fatigue morphology report

## Scope and integrity

This is an existing-data analysis at 300 K. It indexed **{len(master)} distinct observed parameter fingerprints**, including the 1,024-row global population, the 256-row local-fracture population, historical v9.13 rows, **{int(master.accelerated_HCF_data_exists.sum())} candidates with accelerated fatigue data**, **{int(master.explicit_LCF_data_exists.sum())} with explicit-cycle LCF data**, and **{int(master.matched_2D_data_exists.sum())} with matched 2-D evidence**. Source counts are `{source_counts}`.

No simulation was launched. The implemented EXP-floor barrier was evaluated directly at the production map $K_{{max}}=\Delta K/(1-R)$, $\sigma=K_{{max}}/\sqrt{{2\pi r_0}}$, with $R=0.1$, $r_0=1$ µm and the existing 30 GPa cohesive cap. Bare elementary rates use the production attempt frequencies $10^{{12}}$ and $10^{{11}}$ s⁻¹; the primary cleavage/emission ratio then applies the production $m=3$, $\tau_c=10^{{-6}}$ s multi-hit transform to cleavage. The elementary ratio is saved separately. State-mediated quantities are labelled separately and come only from saved mechanism probes.

The point inventory contains {len(points)} provenance-resolved rows ({len(authoritative)} authoritative analysis rows). Censors and partial runs carry no fabricated rate. {invalid_high} unmatched high-load accelerated records are retained for provenance but excluded from LCF inference because explicit-cycle parity is not established there.

## Principal answers

1. **VHCF slope predictor.** {best('S_K_VHCF')}
2. **HCF slope predictor.** {best('S_K_HCF')}
3. **LCF slope predictor.** {best('S_K_LCF')}
4. **VHCF→HCF knee location.** {best('f_VHCF_HCF')} Dimensional and normalized results are both retained; a knee is not assigned unless a two-segment fit improves SSE by at least 35% and changes slope by at least a factor two.
5. **Knee width.** {best('W_knee_f')} Width is the connected 20–80% local-slope transition interval, not a fixed $\Delta K$ window.
6. **HCF→LCF location.** {best('f_HCF_LCF')} LCF exists only when explicit-cycle event-density evidence exists.
7. **Does crossover predict transitions?** {specific('f_VHCF_HCF', 'f_cross_primary')} For LCF cleavage dominance, {specific('f_HCF_LCF', 'K_rate_ratio_100_MPa_sqrt_m')} Multiple/no-crossover topologies are preserved rather than forced to one root.
8. **Relative barrier slopes and sharpness.** {specific('slope_change_knee_K', 'crossover_sharpness_dlog10R_dK')} The broader relative-slope descriptor gives {specific('W_knee_f', 'relative_max_slope_difference_eV_per_MPa_sqrt_m')}
9. **High-K cleavage sensitivity and LCF slope.** {specific('S_K_LCF', 'LCF_dGc_dK')} The LCF sample is necessarily small because only explicit-cycle points are admitted.
10. **Morphology separation.** Existing curves classify as `{morph_counts}`. Arrest/re-entry sequences are split at censors; they are not smoothed into finite curves.
11. **Fracture-scale normalization.** `S_f`, dimensional `S_K`, and log-log `m` are reported separately. Partial correlations control reference $\Delta K$, saved 300 K resistance, and cleavage-barrier scale; see `barrier_fatigue_partial_correlations.csv`.
12. **Restriction to reduced-model-valid candidates.** Only {reduced_n} physical candidates have the existing `REDUCED_VALID` label, so subset correlations are reported but are low-power and are never used to claim universality. B/DBTT/Peak-T remain spatial-correction cases; D remains a spatial bifurcation; weak-T remains spatially unresolved/nonmonotonic.
13. **General versus class-specific.** Relationships surviving normalized coordinates, scale-controlled partial correlation, removal of A–D extremes, and the historical-family holdout are the strongest candidates for generality. Any relation supported only by A–D or the four historical families is explicitly class-specific.
14. **Most efficient prospective tests.** Use a small factorial set centered on (i) kinetic-crossover position at fixed cleavage scale, (ii) crossover width at fixed position, and (iii) high-K cleavage derivative at fixed low-K barrier. Run matched accelerated/explicit overlap first, then only one or two 2-D points for candidates predicted to cross the spatial-bifurcation boundary. No broad campaign is warranted from this first pass.

## Regime and fit semantics

- Segment fits use finite developed/complete target-reaching rates only. Three or more points use an OLS fit with Theil–Sen outlier audit; two-point results are flagged `TWO_POINT_SLOPE`.
- Local derivatives are calculated only inside contiguous finite segments and never across censors, arrest gaps, partial runs, or re-entry gaps.
- The LCF transition is the first explicit-cycle point carrying saved LCF/near-monotonic event-density evidence. An accelerated high-rate point cannot create an LCF branch.
- The 1-D/2-D validity categories are explanatory strata, not corrections applied to the 1-D rates.

## Functional analysis and clustering

Barrier PCA uses both exact normalized cleavage and emission surfaces over $0\le f\le3$; its explained fractions begin `{pca_meta['barrier'].get('explained_variance_ratio', [])[:3]}`. Fatigue PCA uses only continuous finite curves spanning $0.95\le f\le1.20$ and does not bridge censor gaps; its fractions begin `{pca_meta['fatigue'].get('explained_variance_ratio', [])[:3]}`. Morphology clustering selected k={cluster_meta.get('selected_k')} by silhouette ({finite(cluster_meta.get('silhouette')):.3g}); the count was not forced to four. Ridge models, where sample support permits, use five-fold cross-validation; performance is saved in `interpretable_model_performance.csv`. Feature coefficients are associations, not causal proof.

## Explicit hypothesis tests

{hlines}

## Central conclusion

The database supports testing the chain **relative barrier geometry → kinetic crossover → fatigue morphology**, but the evidential strength is response-dependent. VHCF/HCF correlations draw on the broad 1-D population; LCF and spatial-bifurcation statements remain bounded by the much smaller explicit-cycle and matched-2-D subsets. The report therefore distinguishes robust broad-population associations, low-power mechanism-consistent trends, and rejected/unsupported claims rather than turning all morphology into a single candidate score.
"""
    (out / "BARRIER_GEOMETRY_FATIGUE_MORPHOLOGY_REPORT.md").write_text(report)


def dataset_inventory(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        if not p.exists(): continue
        try:
            if p.suffix == ".csv": n = sum(1 for _ in p.open(errors="ignore")) - 1
            elif p.suffix == ".json": n = 1
            else: n = np.nan
        except Exception: n = np.nan
        rows.append({"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size, "rows": n, "mtime": p.stat().st_mtime})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "runs/v914_barrier_fatigue_morphology_analysis_v1")
    args = ap.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)

    master, registry_inputs = _registry_rows(); master = _add_scales_and_labels(master)
    points = assemble_fatigue_points(master)
    source_map = points.groupby("physical_candidate_id").agg(
        available_fatigue_curve_sources=("source_campaign", lambda x: ";".join(sorted(set(str(v) for v in x if pd.notna(v))))),
        available_fatigue_point_origins=("point_origin", lambda x: ";".join(sorted(set(str(v) for v in x if pd.notna(v))))),
    ).reset_index().rename(columns={"physical_candidate_id": "candidate_id"})
    master = master.merge(source_map, on="candidate_id", how="left")

    bdesc, bcurves = barrier_descriptors(master)
    probe_points, probe_desc = load_probe_descriptors()
    morph, local_slopes, fits = morphology_descriptors(points, master)
    at_trans = descriptors_at_transitions(morph, bcurves, probe_points)
    corr_master = master.merge(bdesc, on="candidate_id", how="left")
    corr_master = corr_master.merge(morph, on="candidate_id", how="left").merge(at_trans, on="candidate_id", how="left").merge(probe_desc, on="candidate_id", how="left")
    corr_master["spatial_bifurcation_indicator"] = corr_master.spatial_validation_class.eq("SPATIAL_BIFURCATION").astype(float)
    for c in ["arrest_reentry_indicator", "LCF_upturn_indicator", "near_monotonic_indicator"]:
        if c in corr_master: corr_master[c] = corr_master[c].astype("boolean").astype(float)
    corrs, partial = correlation_tables(corr_master[corr_master.finite_rate_points.fillna(0) > 0].copy())
    bscore, fscore, pca_meta = functional_pca(master, bcurves, points)
    clusters, cluster_meta = cluster_morphologies(corr_master[corr_master.finite_rate_points.fillna(0) > 0].copy())
    model_perf, importance = ridge_models(corr_master[corr_master.finite_rate_points.fillna(0) > 0].copy())
    hypotheses = hypothesis_table(corrs)

    outputs = {
        "barrier_fatigue_master.csv": master,
        "fatigue_curve_points.csv": points,
        "fatigue_morphology_descriptors.csv": morph,
        "barrier_geometry_descriptors.csv": bdesc,
        "barrier_fatigue_correlation_master.csv": corr_master,
        "barrier_fatigue_correlations.csv": corrs,
        "barrier_fatigue_partial_correlations.csv": partial,
        "barrier_shape_pca_scores.csv": bscore,
        "fatigue_shape_pca_scores.csv": fscore,
        "fatigue_morphology_clusters.csv": clusters,
        "fatigue_local_slopes.csv": local_slopes,
        "fatigue_regime_fits.csv": fits,
        "mechanism_probe_points.csv": probe_points,
        "mechanism_probe_descriptors.csv": probe_desc,
        "barrier_descriptors_at_fatigue_transitions.csv": at_trans,
        "interpretable_model_performance.csv": model_perf,
        "interpretable_feature_importance.csv": importance,
        "physics_hypothesis_tests.csv": hypotheses,
        # A compact exact-function audit grid; the full 1001-point grid is held
        # in memory for derivatives/PCA and downsampled only for disk economy.
        "barrier_function_grid.csv": bcurves.groupby("candidate_id", group_keys=False).apply(lambda g: g.iloc[::10], include_groups=False).reset_index(),
    }
    for name, frame in outputs.items(): frame.to_csv(out / name, index=False)

    make_figures(out, corr_master, corrs, bscore, fscore, clusters, bcurves, points)
    write_report(out, master, points, morph, corrs, partial, hypotheses, pca_meta, cluster_meta, model_perf)

    input_paths = registry_inputs + [GLOBAL_AUDIT, LOCAL_AUDIT, OLD_CURVES, RERANK_MASTER, MECH, ABCD_HYBRID, MATERIAL_HYBRID, TRANSITIONS, TEMP_TABLE]
    input_paths += sorted(PROBES.glob("*/state_screen.json"))
    inv = dataset_inventory(input_paths); inv.to_csv(out / "dataset_inventory.csv", index=False)
    required = [
        "barrier_fatigue_master.csv", "fatigue_curve_points.csv", "fatigue_morphology_descriptors.csv",
        "barrier_geometry_descriptors.csv", "barrier_fatigue_correlation_master.csv", "barrier_fatigue_correlations.csv",
        "barrier_fatigue_partial_correlations.csv", "barrier_shape_pca_scores.csv", "fatigue_shape_pca_scores.csv",
        "fatigue_morphology_clusters.csv", "BARRIER_GEOMETRY_FATIGUE_MORPHOLOGY_REPORT.md",
    ]
    required_figs = [
        "correlation_heatmap_regime_slopes.png", "correlation_heatmap_transition_locations.png",
        "correlation_heatmap_transition_widths.png", "correlation_heatmap_kinetic_competition.png",
        "barrier_crossover_vs_knee_location.png", "barrier_crossover_vs_lcf_location.png",
        "vhcf_slope_vs_barrier_sensitivity.png", "hcf_slope_vs_effective_barrier_sensitivity.png",
        "lcf_slope_vs_highK_barrier_sensitivity.png", "knee_slope_change_vs_crossover_sharpness.png",
        "lcf_slope_recovery_vs_rate_ratio.png", "barrier_geometry_fatigue_phase_map.png",
        "barrier_shape_pca.png", "fatigue_shape_pca.png", "barrier_mode_vs_fatigue_mode.png",
        "fatigue_morphology_cluster_map.png",
    ]
    audit = {
        "schema": "v914_barrier_fatigue_morphology_analysis_v1", "repository_head": current_head(),
        "analysis_only_no_simulations_launched": True, "physics_changed": False,
        "production_barrier_function": "EXP_floor exact implementation",
        "production_tip_stress_map": {"R": R_LOAD, "r0_m": R0_M, "sigma_cap_Pa": SIGMA_CAP_PA},
        "candidate_count": len(master), "fatigue_candidate_count": int((master.accelerated_HCF_data_exists | master.explicit_LCF_data_exists).sum()),
        "fatigue_point_count": len(points), "authoritative_fatigue_point_count": int(points.authoritative_use.sum()),
        "mechanism_probe_candidate_count": int(probe_points.candidate_id.nunique()) if not probe_points.empty else 0,
        "correlation_count": len(corrs), "partial_correlation_count": len(partial),
        "required_outputs_present": all((out / x).exists() for x in required),
        "required_figures_present": all((out / x).exists() for x in required_figs),
        "plot_data_sidecars_present": all((out / x.replace(".png", "_plot_data.csv")).exists() for x in required_figs),
        "required_outputs": required, "required_figures": required_figs,
        "pca": pca_meta, "clustering": cluster_meta,
        "input_inventory_sha256": sha256(out / "dataset_inventory.csv"),
    }
    (out / "analysis_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, default=float))
    print(json.dumps({k: audit[k] for k in ["repository_head", "candidate_count", "fatigue_candidate_count", "fatigue_point_count", "mechanism_probe_candidate_count", "required_outputs_present", "required_figures_present"]}, indent=2))


if __name__ == "__main__":
    main()
