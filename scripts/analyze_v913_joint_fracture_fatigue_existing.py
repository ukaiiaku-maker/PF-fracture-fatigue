#!/usr/bin/env python3
"""Joint v9.13 fracture / v9.14 fatigue existing-data analysis.

This script is deliberately analysis-only.  It joins candidates by a SHA256
fingerprint of the complete physical-parameter intersection shared by the two
authoritative archives, then uses candidate ID only as a consistency check.
It never turns a censor or partial trajectory into a fatigue rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
V1 = REPO / "runs/v913_barrier_temperature_fracture_morphology_v1"
V3 = REPO / "runs/v913_barrier_temperature_fracture_morphology_v3_focused"
VF = REPO / "runs/v914_barrier_fatigue_morphology_analysis_v1"
OUT = REPO / "runs/v913_joint_fracture_fatigue_causality_v1/joint_existing"

COMMON_PHYSICAL_PARAMETERS = [
    "cleave_G00_eV", "cleave_gT_eV_per_K", "cleave_sigc0_GPa",
    "cleave_sT_GPa_per_K", "cleave_exp_a", "cleave_exp_n",
    "cleave_floor_frac", "emit_G00_eV", "emit_gT_eV_per_K",
    "emit_sigc0_GPa", "emit_sT_GPa_per_K", "emit_exp_a", "emit_exp_n",
    "emit_floor_frac", "peierls_H0_eV", "peierls_activation_entropy_kB",
    "peierls_exp_a", "peierls_exp_n", "peierls_nu0_s", "taylor_H0_eV",
    "taylor_activation_entropy_kB", "taylor_exp_a", "taylor_exp_n",
    "taylor_nu0_s", "rho_source0_m2", "taylor_corr_rho_c_m2",
    "taylor_corr_scale", "c_blunt",
]

FRACTURE_RESPONSES = [
    "K_300_MPa_sqrt_m", "S_low_MPa_sqrt_m_per_K",
    "S_mid_MPa_sqrt_m_per_K", "S_high_MPa_sqrt_m_per_K",
    "fractional_resistance_span", "DBTT_magnitude_MPa_sqrt_m",
    "DBTT_temperature_K", "DBTT_width_K", "peak_prominence_MPa_sqrt_m",
    "peak_temperature_K", "weakT_max_deviation_from_mean_MPa_sqrt_m",
    "fractional_terminal_change",
]
FATIGUE_RESPONSES = [
    "f_VHCF_HCF", "W_knee_f", "f_HCF_LCF", "S_f_VHCF", "S_f_HCF",
    "S_f_LCF", "m_VHCF", "m_HCF", "m_LCF",
    "minimum_finite_rate_m_per_cycle", "developed_HCF_rate_m_per_cycle",
    "maximum_explicit_LCF_rate_m_per_cycle", "dynamic_rate_range_decades",
    "arrest_reentry_indicator", "LCF_upturn_indicator",
]

CANONICAL = {
    "v913_zeroD_sobol_0202500": "DBTT",
    "v913_zeroD_sobol_0242980": "Peak-T",
    "v913_zeroD_sobol_0129902": "weak-T",
    "v913_zeroD_sobol_0077080": "ceramic-like",
}
COLORS = {"DBTT": "#3B82F6", "Peak-T": "#F59E0B", "weak-T": "#8B5CF6",
          "ceramic-like": "#64748B", "historical intersection": "#9CA3AF"}


def finite(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except (TypeError, ValueError):
        return float(default)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def canonical_value(value):
    if pd.isna(value):
        return None
    try:
        return format(float(value), ".17g")
    except (TypeError, ValueError):
        return str(value)


def shared_physics_fingerprint(row: pd.Series, columns=COMMON_PHYSICAL_PARAMETERS) -> str:
    """Hash the full shared constitutive row, excluding IDs and provenance."""
    payload = {c: canonical_value(row.get(c, np.nan)) for c in columns}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def nondominated_mask(values: np.ndarray, maximize: list[bool]) -> np.ndarray:
    """Return the Pareto-nondominated mask, treating every row as feasible."""
    x = np.asarray(values, float).copy()
    for j, up in enumerate(maximize):
        if not up:
            x[:, j] *= -1
    keep = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        dominates_i = np.all(x >= x[i], axis=1) & np.any(x > x[i], axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            keep[i] = False
    return keep


def continuous_segment_eligible(curve: pd.DataFrame, low=.95, high=1.20, minimum=4) -> bool:
    """Require one uninterrupted finite segment spanning [low, high].

    Rows explicitly marked censor/partial split the finite sequence.  This
    predicate is also the regression guard against censor interpolation.
    """
    q = curve.sort_values("normalized_f").copy()
    finite_mask = q["plot_kind"].eq("resolved") & np.isfinite(q["da_dN_m_per_cycle"]) & (q["da_dN_m_per_cycle"] > 0)
    group = (~finite_mask).cumsum()
    for _, seg in q[finite_mask].groupby(group[finite_mask]):
        seg = seg.drop_duplicates("normalized_f")
        if len(seg) >= minimum and seg.normalized_f.min() <= low and seg.normalized_f.max() >= high:
            return True
    return False


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    mu = x.mean(axis=0); sd = x.std(axis=0, ddof=1); sd[sd == 0] = 1
    return (x - mu) / sd, mu, sd


def small_cca_pls(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Two-component, unregularized descriptive CCA and SVD-PLS.

    The caller is responsible for labeling small-n estimates exploratory.
    """
    zx, _, _ = standardize(x); zy, _, _ = standardize(y)
    n = len(zx); denom = max(n - 1, 1)
    cxx = zx.T @ zx / denom + np.eye(zx.shape[1]) * 1e-8
    cyy = zy.T @ zy / denom + np.eye(zy.shape[1]) * 1e-8
    cxy = zx.T @ zy / denom
    ex, ux = np.linalg.eigh(cxx); ey, uy = np.linalg.eigh(cyy)
    wx = ux @ np.diag(1 / np.sqrt(np.maximum(ex, 1e-10))) @ ux.T
    wy = uy @ np.diag(1 / np.sqrt(np.maximum(ey, 1e-10))) @ uy.T
    u, s, vt = np.linalg.svd(wx @ cxy @ wy)
    rows = []
    for k in range(min(2, len(s))):
        ax = wx @ u[:, k]; ay = wy @ vt.T[:, k]
        tx = zx @ ax; ty = zy @ ay
        rows.append({"method": "CCA", "component": k + 1, "n": n,
                     "association": float(np.corrcoef(tx, ty)[0, 1]),
                     "singular_value": float(s[k]),
                     "x_weights": json.dumps(ax.tolist()), "y_weights": json.dumps(ay.tolist()),
                     "interpretation_scope": "EXPLORATORY_SMALL_INTERSECTION_NONCAUSAL"})
    up, sp, vtp = np.linalg.svd(cxy)
    for k in range(min(2, len(sp))):
        tx = zx @ up[:, k]; ty = zy @ vtp.T[:, k]
        rows.append({"method": "PLS_SVD", "component": k + 1, "n": n,
                     "association": float(np.corrcoef(tx, ty)[0, 1]),
                     "singular_value": float(sp[k]),
                     "x_weights": json.dumps(up[:, k].tolist()),
                     "y_weights": json.dumps(vtp.T[:, k].tolist()),
                     "interpretation_scope": "EXPLORATORY_SMALL_INTERSECTION_NONCAUSAL"})
    return pd.DataFrame(rows)


def savefig(fig, out: Path, stem: str, data: pd.DataFrame) -> None:
    fig.savefig(out / f"{stem}.png", dpi=190, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    data.to_csv(out / f"{stem}_plot_data.csv", index=False)


def join_inventory() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    ft = pd.read_csv(V1 / "fracture_temperature_master.csv", low_memory=False)
    fr_params = ft.sort_values("temperature_K").drop_duplicates("candidate_id").copy()
    fatigue_all = pd.read_csv(VF / "barrier_fatigue_master.csv", low_memory=False)
    morph = pd.read_csv(VF / "fatigue_morphology_descriptors.csv", low_memory=False)
    fatigue = fatigue_all[fatigue_all.candidate_id.isin(morph.candidate_id)].copy()
    missing = [c for c in COMMON_PHYSICAL_PARAMETERS if c not in fr_params or c not in fatigue]
    if missing:
        raise RuntimeError(f"shared physical fingerprint fields missing: {missing}")
    fr_params["joint_parameter_fingerprint_sha256"] = fr_params.apply(shared_physics_fingerprint, axis=1)
    fatigue["joint_parameter_fingerprint_sha256"] = fatigue.apply(shared_physics_fingerprint, axis=1)
    if fr_params.joint_parameter_fingerprint_sha256.duplicated().any():
        raise RuntimeError("fracture archive contains duplicate shared-physics fingerprints")
    if fatigue.joint_parameter_fingerprint_sha256.duplicated().any():
        raise RuntimeError("fatigue response archive contains duplicate shared-physics fingerprints")
    matched = fr_params[["candidate_id", "parameter_fingerprint", "joint_parameter_fingerprint_sha256"]].merge(
        fatigue[["candidate_id", "parameter_fingerprint", "joint_parameter_fingerprint_sha256"]],
        on="joint_parameter_fingerprint_sha256", suffixes=("_fracture", "_fatigue"), validate="one_to_one")
    matched["candidate_id_crosscheck"] = np.where(
        matched.candidate_id_fracture.eq(matched.candidate_id_fatigue), "PASS_EXACT_ID", "ALIAS_ID_DIFFERENCE")
    matched["match_method"] = "PRIMARY_FULL_SHARED_PHYSICS_FINGERPRINT"
    matched["shared_physical_parameter_count"] = len(COMMON_PHYSICAL_PARAMETERS)
    audit = {
        "fracture_candidate_count": int(fr_params.candidate_id.nunique()),
        "fatigue_candidate_count_with_response": int(fatigue.candidate_id.nunique()),
        "exact_fingerprint_intersection_count": int(len(matched)),
        "exact_id_crosscheck_pass_count": int(matched.candidate_id_crosscheck.eq("PASS_EXACT_ID").sum()),
        "shared_physical_parameter_count": len(COMMON_PHYSICAL_PARAMETERS),
    }
    return matched, audit, fatigue


def build_master(matched: pd.DataFrame, fatigue_registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = matched.candidate_id_fracture.tolist()
    resp = pd.read_csv(V1 / "fracture_response_descriptors.csv", low_memory=False)
    fpca = pd.read_csv(V1 / "fracture_response_pca_scores.csv", low_memory=False)
    curves = pd.read_csv(V1 / "fracture_response_curve_points.csv", low_memory=False)
    morph = pd.read_csv(VF / "fatigue_morphology_descriptors.csv", low_memory=False)
    points = pd.read_csv(VF / "fatigue_curve_points.csv", low_memory=False)
    intrinsic = pd.read_csv(V3 / "focused_model_master.csv", low_memory=False)
    fscore = pd.read_csv(VF / "fatigue_shape_pca_scores.csv", low_memory=False)

    master = matched.rename(columns={"candidate_id_fracture": "candidate_id"}).drop(columns="candidate_id_fatigue")
    master = master.merge(resp, on="candidate_id", how="left", suffixes=("", "_response"))
    master = master.merge(fpca[["candidate_id", "fracture_response_PC1", "fracture_response_PC2"]], on="candidate_id", how="left")
    master = master.merge(morph, on="candidate_id", how="left")
    master = master.merge(fscore[["candidate_id", "fatigue_PC1", "fatigue_PC2"]], on="candidate_id", how="left")
    useful_intrinsic = ["candidate_id", "cleave_x50", "cleave_sstar_max", "delta_mu_emit_minus_cleave",
                        "normalized_center_separation_Dmu", "activation_window_overlap_Oce",
                        "width80_ratio_emit_over_cleave", "delta_Theta_sigma_900",
                        "delta_Theta_G_900", "Dgamma_span", "Dmin_span",
                        "B_P_log10_tauP_over_taue", "B_T_log10_tauT_over_taue",
                        "plastic_control_change_count"]
    master = master.merge(intrinsic[[c for c in useful_intrinsic if c in intrinsic]], on="candidate_id", how="left")
    registry_cols = ["candidate_id", "matched_2D_data_exists", "accelerated_HCF_data_exists",
                     "explicit_LCF_data_exists", "spatial_validation_class", "reference_deltaK_MPa_sqrt_m",
                     "fracture_resistance_300K_MPa_sqrt_m"]
    master = master.merge(fatigue_registry[registry_cols], on="candidate_id", how="left").copy()

    master["K_R_300_MPa_sqrt_m"] = master.K_300_MPa_sqrt_m.fillna(master.fracture_resistance_300K_MPa_sqrt_m)
    master["K_R_300_source"] = np.select(
        [master.K_300_MPa_sqrt_m.notna(), master.fracture_resistance_300K_MPa_sqrt_m.notna()],
        ["V913_DIRECT_300K_FRACTURE_RESPONSE", "V914_AUTHORITATIVE_STAGEA_300K_FRACTURE_SCALE"],
        default="UNAVAILABLE")

    # Explicit normalized fracture coordinates.  A missing 300 K datum remains NA.
    for T, g in curves[curves.candidate_id.isin(ids)].groupby("temperature_K"):
        if float(T).is_integer():
            kval = master.candidate_id.map(g.set_index("candidate_id")["K_response_MPa_sqrt_m"])
            master[f"K_over_KR300_T{int(T)}"] = kval / master.K_R_300_MPa_sqrt_m

    p = points[points.physical_candidate_id.isin(ids) & points.authoritative_use.fillna(False)].copy()
    p["valid_rate"] = p.plot_kind.eq("resolved") & np.isfinite(p.da_dN_m_per_cycle) & (p.da_dN_m_per_cycle > 0)
    summaries = []
    for cid, g in p.groupby("physical_candidate_id"):
        finitep = g[g.valid_rate]
        hcf = finitep[finitep.regime_classification.astype(str).str.contains("HCF", case=False, na=False) &
                      ~finitep.regime_classification.astype(str).str.contains("LCF", case=False, na=False)]
        explicit = finitep[finitep.integration_mode.eq("explicit")]
        summaries.append({
            "candidate_id": cid,
            "minimum_finite_rate_m_per_cycle": finitep.da_dN_m_per_cycle.min() if len(finitep) else np.nan,
            "developed_HCF_rate_m_per_cycle": hcf.da_dN_m_per_cycle.median() if len(hcf) else np.nan,
            "maximum_explicit_LCF_rate_m_per_cycle": explicit.da_dN_m_per_cycle.max() if len(explicit) else np.nan,
            "authoritative_finite_point_count": int(len(finitep)),
            "authoritative_cycle_censor_count": int(g.plot_kind.eq("censor").sum()),
            "authoritative_partial_count": int(g.plot_kind.eq("partial").sum()),
            "fatigue_evidence_semantics": ";".join(sorted(set(g.integration_mode.astype(str))))})
    master = master.merge(pd.DataFrame(summaries), on="candidate_id", how="left")
    master["explicit_cycle_LCF"] = master.explicit_LCF_data_exists.fillna(False).astype(bool)
    master["accelerated_HCF_only"] = master.accelerated_HCF_data_exists.fillna(False).astype(bool) & ~master.explicit_cycle_LCF
    master["matched_2D_validation"] = master.matched_2D_data_exists.fillna(False).astype(bool)
    master["canonical_family_joint"] = master.candidate_id.map(CANONICAL)
    master["ceramic_softening_metric"] = master["fractional_terminal_change"]
    return master, p


def response_correlations(master: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("fractional_resistance_span", "f_VHCF_HCF", "fracture temperature span vs fatigue knee"),
        ("DBTT_magnitude_MPa_sqrt_m", "S_f_HCF", "DBTT magnitude vs HCF slope"),
        ("DBTT_width_K", "W_knee_f", "DBTT width vs fatigue knee width"),
        ("peak_prominence_MPa_sqrt_m", "LCF_upturn_indicator", "Peak prominence vs LCF upturn"),
        ("fracture_response_PC1", "fatigue_PC1", "fracture PC1 vs fatigue PC1"),
        ("fracture_response_PC1", "fatigue_PC2", "fracture PC1 vs fatigue PC2"),
        ("fracture_response_PC2", "fatigue_PC1", "fracture PC2 vs fatigue PC1"),
        ("fracture_response_PC2", "fatigue_PC2", "fracture PC2 vs fatigue PC2"),
    ]
    rows = []
    for x, y, label in pairs:
        if x not in master or y not in master:
            continue
        q = master[[x, y]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(q) < 3 or q[x].nunique() < 2 or q[y].nunique() < 2:
            pr = sr = pp = sp = np.nan
        else:
            a = stats.pearsonr(q[x], q[y]); b = stats.spearmanr(q[x], q[y])
            pr, pp, sr, sp = a.statistic, a.pvalue, b.statistic, b.pvalue
        rows.append({"relationship": label, "fracture_response": x, "fatigue_response": y,
                     "n": len(q), "pearson_r": pr, "pearson_p": pp,
                     "spearman_rho": sr, "spearman_p": sp,
                     "evidence_class": "RETROSPECTIVE_RESPONSE_RESPONSE_NONCAUSAL"})
    return pd.DataFrame(rows)


FEATURES = {
    "cleave_x50": "CLEAVAGE_TRANSITION_POSITION",
    "cleave_sstar_max": "CLEAVAGE_STRESS_SENSITIVITY",
    "delta_mu_emit_minus_cleave": "RELATIVE_WINDOW_SEPARATION",
    "normalized_center_separation_Dmu": "RELATIVE_WINDOW_SEPARATION",
    "activation_window_overlap_Oce": "RELATIVE_WINDOW_WIDTH",
    "width80_ratio_emit_over_cleave": "RELATIVE_WINDOW_WIDTH",
    "delta_Theta_sigma_900": "THERMAL_BARRIER_MOTION",
    "delta_Theta_G_900": "THERMAL_BARRIER_MOTION",
    "Dgamma_span": "KINETIC_COMPETITION",
    "Dmin_span": "KINETIC_COMPETITION",
    "B_P_log10_tauP_over_taue": "PLASTIC_BOTTLENECK",
    "B_T_log10_tauT_over_taue": "PLASTIC_BOTTLENECK",
    "plastic_control_change_count": "PLASTIC_BOTTLENECK",
}


def shared_predictors(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, family in FEATURES.items():
        if feature not in master:
            continue
        for domain, responses in [("FRACTURE", FRACTURE_RESPONSES), ("FATIGUE", FATIGUE_RESPONSES)]:
            vals = []
            for response in responses:
                if response not in master:
                    continue
                q = master[[feature, response]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(q) < 5 or q[feature].nunique() < 3 or q[response].nunique() < 3:
                    continue
                s = stats.spearmanr(q[feature], q[response])
                vals.append((response, len(q), float(s.statistic), float(s.pvalue)))
            if vals:
                best = max(vals, key=lambda z: abs(z[2]))
                rows.append({"feature": feature, "feature_family": family, "domain": domain,
                             "best_response": best[0], "n": best[1], "best_abs_spearman": abs(best[2]),
                             "signed_spearman": best[2], "nominal_p": best[3],
                             "scope": "RETROSPECTIVE_ASSOCIATION_SMALL_INTERSECTION"})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    wide = out.pivot(index=["feature", "feature_family"], columns="domain", values="best_abs_spearman").reset_index()
    wide = wide.rename(columns={"FRACTURE": "max_abs_spearman_fracture", "FATIGUE": "max_abs_spearman_fatigue"})
    wide["predictive_scope_class"] = np.select([
        (wide.max_abs_spearman_fracture >= .3) & (wide.max_abs_spearman_fatigue >= .3),
        wide.max_abs_spearman_fracture >= .3, wide.max_abs_spearman_fatigue >= .3],
        ["BOTH_DOMAINS_DESCRIPTIVE", "FRACTURE_ONLY_DESCRIPTIVE", "FATIGUE_ONLY_DESCRIPTIVE"],
        default="NO_STRONG_DESCRIPTIVE_ASSOCIATION")
    detail = out.pivot(index=["feature", "feature_family"], columns="domain", values=["best_response", "n", "signed_spearman", "nominal_p"]).reset_index()
    detail.columns = ["_".join(str(x) for x in c if x).lower() if isinstance(c, tuple) else c for c in detail.columns]
    return wide.merge(detail, on=["feature", "feature_family"], how="left")


def build_pareto(master: pd.DataFrame) -> pd.DataFrame:
    p = master[["candidate_id", "canonical_family_joint", "n_temperatures", "finite_rate_points",
                "partial_points", "explicit_cycle_LCF", "matched_2D_validation", "K_300_MPa_sqrt_m",
                "K_R_300_MPa_sqrt_m", "K_R_300_source", "dynamic_rate_range_decades", "developed_LCF_available"]].copy()
    p["fracture_curve_completeness"] = p.n_temperatures / p.n_temperatures.max()
    reference = master.loc[master.candidate_id.isin(CANONICAL), "K_R_300_MPa_sqrt_m"].dropna().median()
    if not np.isfinite(reference):
        reference = master.K_R_300_MPa_sqrt_m.dropna().median()
    p["K300_scale_deviation_log10"] = abs(np.log10(p.K_R_300_MPa_sqrt_m / reference))
    p["numerical_resolution_quality"] = 1 / (1 + p.partial_points.fillna(0))
    p["distance_from_historical_parameter_manifold"] = 0.0
    # Keep raw objectives separate: no arbitrary aggregate "realism score".
    objectives = ["fracture_curve_completeness", "finite_rate_points", "explicit_cycle_LCF",
                  "matched_2D_validation", "K300_scale_deviation_log10", "numerical_resolution_quality"]
    x = p[objectives].astype(float).fillna({"fracture_curve_completeness": 0, "finite_rate_points": 0,
        "explicit_cycle_LCF": 0, "matched_2D_validation": 0,
        "K300_scale_deviation_log10": np.inf, "numerical_resolution_quality": 0}).to_numpy(float)
    # Replace infinity with a finite dominated sentinel for robust comparisons.
    x[~np.isfinite(x)] = np.nanmax(x[np.isfinite(x)]) + 10
    p["pareto_nondominated"] = nondominated_mask(x, [True, True, True, True, False, True])
    p["realism_basis"] = "MODEL_INTERNAL_PHYSICAL_PLAUSIBILITY"
    p["selection_category"] = np.where(p.pareto_nondominated, "BEST_JOINT_BALANCE_PARETO", "DOMINATED_EXISTING_DATA")
    p["selection_note"] = "multi-objective; no scalar realism score and no experimental-agreement claim"
    return p


def make_figures(out: Path, master: pd.DataFrame, shared: pd.DataFrame,
                 pareto: pd.DataFrame, points: pd.DataFrame) -> None:
    # 1. Fracture/fatigue PC associations (only censor-safe fatigue PCA rows).
    q = master.dropna(subset=["fracture_response_PC1", "fracture_response_PC2", "fatigue_PC1", "fatigue_PC2"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, x, y in [(axes[0], "fracture_response_PC1", "fatigue_PC1"),
                     (axes[1], "fracture_response_PC2", "fatigue_PC2")]:
        for r in q.itertuples():
            label = CANONICAL.get(r.candidate_id, "historical intersection")
            ax.scatter(getattr(r, x), getattr(r, y), c=COLORS[label], s=48, edgecolor="black", linewidth=.4)
            if r.candidate_id in CANONICAL:
                ax.annotate(CANONICAL[r.candidate_id], (getattr(r, x), getattr(r, y)), fontsize=8)
        ax.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " "))
    fig.suptitle(f"Censor-safe joint PCA intersection (n={len(q)}; exploratory)")
    pdata = q[["candidate_id", "fracture_response_PC1", "fracture_response_PC2", "fatigue_PC1", "fatigue_PC2"]]
    savefig(fig, out, "fracture_PC_vs_fatigue_PC", pdata)

    # 2. Shared-feature family heatmap.
    hm = shared.groupby("feature_family")[["max_abs_spearman_fracture", "max_abs_spearman_fatigue"]].max().sort_index()
    fig, ax = plt.subplots(figsize=(7, max(4, .45 * len(hm))))
    im = ax.imshow(hm.to_numpy(float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks([0, 1], ["fracture", "fatigue"]); ax.set_yticks(range(len(hm)), hm.index)
    for i in range(len(hm)):
        for j in range(2): ax.text(j, i, f"{hm.iloc[i, j]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, label="max |Spearman rho| (descriptive)")
    savefig(fig, out, "shared_barrier_feature_heatmap", hm.reset_index().melt("feature_family", var_name="domain", value_name="max_abs_spearman"))

    # 3. Common barrier phase map.
    phase = master[["candidate_id", "normalized_center_separation_Dmu", "activation_window_overlap_Oce",
                    "fatigue_morphology", "canonical_family_joint", "finite_rate_points"]].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(phase.normalized_center_separation_Dmu, phase.activation_window_overlap_Oce,
                    c=phase.finite_rate_points, cmap="plasma", s=55, edgecolor="black", linewidth=.4)
    for r in phase.dropna(subset=["canonical_family_joint"]).itertuples():
        ax.annotate(r.canonical_family_joint, (r.normalized_center_separation_Dmu, r.activation_window_overlap_Oce), fontsize=8)
    ax.set(xlabel="normalized activation-window separation", ylabel="activation-window overlap",
           title="Exact fingerprint intersection; color = finite fatigue points")
    fig.colorbar(sc, ax=ax, label="finite fatigue points")
    savefig(fig, out, "fracture_fatigue_barrier_phase_map", phase)

    # 4. Pareto map.
    fig, ax = plt.subplots(figsize=(7, 5))
    for flag, g in pareto.groupby("pareto_nondominated"):
        ax.scatter(g.K300_scale_deviation_log10, g.finite_rate_points,
                   s=50 + 90 * g.fracture_curve_completeness, marker="o" if flag else "x",
                   label="Pareto" if flag else "dominated", alpha=.85)
    ax.set(xlabel="|log10(K300 / canonical median)| (minimize)", ylabel="finite fatigue points (maximize)",
           title="Pareto projection; explicit-LCF and 2-D flags are separate objectives")
    ax.legend()
    savefig(fig, out, "joint_candidate_pareto_map", pareto)

    # 5. Canonical and (currently absent) prospective map.
    cmap = phase.copy(); cmap["candidate_role"] = np.where(cmap.candidate_id.isin(CANONICAL), "CANONICAL", "EXISTING_INTERSECTION")
    cmap["prospective_candidate_available"] = False
    fig, ax = plt.subplots(figsize=(7, 5))
    for role, g in cmap.groupby("candidate_role"):
        ax.scatter(g.normalized_center_separation_Dmu, g.activation_window_overlap_Oce,
                   s=95 if role == "CANONICAL" else 35, marker="*" if role == "CANONICAL" else "o", label=role)
    for r in cmap[cmap.candidate_role.eq("CANONICAL")].itertuples():
        ax.annotate(CANONICAL[r.candidate_id], (r.normalized_center_separation_Dmu, r.activation_window_overlap_Oce), fontsize=8)
    ax.set(xlabel="normalized activation-window separation", ylabel="activation-window overlap",
           title="Existing joint map (prospective Track-A rows not yet available)")
    ax.legend()
    savefig(fig, out, "canonical_and_prospective_joint_map", cmap)

    # 6. Five-panel joint mechanism atlas for the four canonical shared rows.
    temp_curves = pd.read_csv(V1 / "fracture_response_curve_points.csv", low_memory=False)
    kinetics = pd.read_csv(V3 / "whole_surface_kinetic_competition.csv", low_memory=False)
    raw = pd.read_csv(V1 / "fracture_temperature_master.csv", low_memory=False).sort_values("temperature_K").drop_duplicates("candidate_id")
    atlas_rows = []
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
    x = np.linspace(0, 3, 181)
    for cid, label in CANONICAL.items():
        if cid not in set(master.candidate_id): continue
        color = COLORS[label]; rr = raw[raw.candidate_id.eq(cid)].iloc[0]
        for T in [700., 1000., 1400.]:
            for mechanism, prefix, ls in [("cleavage", "cleave", "-"), ("emission", "emit", "--")]:
                G0 = max(1e-12, rr[f"{prefix}_G00_eV"] + rr[f"{prefix}_gT_eV_per_K"] * (T - rr.Tref_K))
                floor = min(.999999 * G0, max(0., rr[f"{prefix}_floor_frac"] * G0))
                G = floor + (G0 - floor) * np.exp(-rr[f"{prefix}_exp_a"] * x ** rr[f"{prefix}_exp_n"])
                axes[0].plot(x, G, color=color, ls=ls, alpha={700.: .35, 1000.: .9, 1400.: .6}[T],
                             lw={700.: 1., 1000.: 1.8, 1400.: 1.2}[T])
                for xx, yy in zip(x, G): atlas_rows.append({"candidate_id": cid, "canonical_family": label, "panel": "barrier", "temperature_K": T, "mechanism": mechanism, "x": xx, "y": yy})
                if T == 1000.:
                    d = np.gradient(G, x); axes[1].plot(x, d, color=color, ls=ls, alpha=.85)
                    for xx, yy in zip(x, d): atlas_rows.append({"candidate_id": cid, "canonical_family": label, "panel": "barrier_derivative", "temperature_K": T, "mechanism": mechanism, "x": xx, "y": yy})
        kg = kinetics[kinetics.candidate_id.eq(cid)].sort_values("temperature_K")
        axes[2].plot(kg.temperature_K, kg.Mgamma_mean_signed_log10_ratio, color=color, label=label)
        for r in kg.itertuples(): atlas_rows.append({"candidate_id": cid, "canonical_family": label, "panel": "kinetic_competition", "temperature_K": r.temperature_K, "mechanism": "signed_mean_log_rate_ratio", "x": r.temperature_K, "y": r.Mgamma_mean_signed_log10_ratio})
        fg = temp_curves[temp_curves.candidate_id.eq(cid)].sort_values("temperature_K")
        axes[3].plot(fg.temperature_K, fg.K_response_MPa_sqrt_m, color=color)
        for r in fg.itertuples(): atlas_rows.append({"candidate_id": cid, "canonical_family": label, "panel": "fracture_KT", "temperature_K": r.temperature_K, "mechanism": "fracture", "x": r.temperature_K, "y": r.K_response_MPa_sqrt_m})
        pg = points[points.physical_candidate_id.eq(cid)].sort_values(["integration_mode", "dimensionality", "normalized_f"]).copy()
        resolved = pg[pg.plot_kind.eq("resolved") & (pg.da_dN_m_per_cycle > 0)]
        cens = pg[pg.plot_kind.eq("censor")]
        # Scatter only: never draw a finite line across a censor/arrest gap.
        for (mode, dim), gg in resolved.groupby(["integration_mode", "dimensionality"]):
            marker = "s" if mode == "explicit" else ("x" if dim == "2D" else "o")
            axes[4].scatter(gg.normalized_f, gg.da_dN_m_per_cycle, marker=marker, color=color, s=20, alpha=.85)
        if len(cens):
            ymin = max(resolved.da_dN_m_per_cycle.min() / 5 if len(resolved) else 1e-20, 1e-30)
            axes[4].scatter(cens.normalized_f, np.full(len(cens), ymin), marker="v", facecolors="none", edgecolors=color)
        for r in pg.itertuples(): atlas_rows.append({"candidate_id": cid, "canonical_family": label, "panel": "fatigue", "temperature_K": 300., "mechanism": r.plot_kind, "x": r.normalized_f, "y": r.da_dN_m_per_cycle})
    axes[0].set(xlabel="normalized local stress", ylabel="barrier (eV)", title="A: barriers at 700/1000/1400 K")
    axes[1].set(xlabel="normalized local stress", ylabel="dG/dx", title="B: barrier derivatives")
    axes[2].set(xlabel="temperature (K)", ylabel="mean signed log-rate ratio", title="C: kinetic competition")
    axes[3].set(xlabel="temperature (K)", ylabel="fracture K", title="D: monotonic fracture")
    axes[4].set(xlabel="normalized f", ylabel="da/dN (m/cycle)", yscale="log",
                title="E: fatigue; triangles=censors\n(no lines across arrest gaps)")
    axes[2].legend(fontsize=7)
    fig.suptitle("Joint mechanism atlas: exact shared canonical fingerprints")
    savefig(fig, out, "joint_mechanism_atlas", pd.DataFrame(atlas_rows))


def write_dictionary(out: Path, master: pd.DataFrame) -> None:
    lines = ["# Unified joint barrier descriptor dictionary", "",
             "The join uses all 28 constitutive fields shared by the authoritative fracture and fatigue registries. Candidate names are a cross-check, never the primary key.", "",
             "| Common family | Fracture/focused names | Fatigue names | Physical meaning |", "|---|---|---|---|",
             "| CLEAVAGE_TRANSITION_POSITION | `cleave_x50` | `cleave_K_drop50_MPa_sqrt_m`, `cleave_K_drop90_MPa_sqrt_m` | Location of the cleavage activation-window transition; normalized versus dimensional coordinates. |",
             "| CLEAVAGE_STRESS_SENSITIVITY | `cleave_sstar_max` | `cleave_max_sensitivity_eV_per_MPa_sqrt_m` | Maximum stress sensitivity of the cleavage barrier. |",
             "| RELATIVE_WINDOW_SEPARATION | `delta_mu_emit_minus_cleave`, `normalized_center_separation_Dmu` | `relative_max_slope_difference_eV_per_MPa_sqrt_m` | Separation of emission and cleavage activation windows. |",
             "| RELATIVE_WINDOW_WIDTH | `activation_window_overlap_Oce`, `width80_ratio_emit_over_cleave` | `crossover_width_0p1_to_10_MPa_sqrt_m` | Overlap/relative width of competing barrier transitions. |",
             "| THERMAL_BARRIER_MOTION | `delta_Theta_sigma_900`, `delta_Theta_G_900` | no independent 300 K response analogue | Relative temperature motion of barrier scales. |",
             "| KINETIC_COMPETITION | `Dgamma_span`, `Dmin_span` | `log10_rate_ratio_lowK`, `crossover_sharpness_dlog10R_dK` | Whole-surface cleavage/emission rate competition. |",
             "| PLASTIC_BOTTLENECK | `B_P_log10_tauP_over_taue`, `B_T_log10_tauT_over_taue` | mechanism-probe relaxation ratios | Serial emission/Peierls/Taylor bottleneck. |",
             "| STATE_MEDIATION | saved first-passage state (partial) | HCF/LCF mechanism probes | Evolved state mediator, not an intrinsic predictor. |", "",
             "## Evidence semantics", "",
             "- `resolved` finite fatigue points remain rates.",
             "- `censor` points remain cycle/hazard censors and are never interpolated into rates.",
             "- `partial` points remain unresolved and are excluded from rate PCA.",
             "- `explicit` and `accelerated` integration modes remain separate columns.",
             "- All joint correlations are retrospective associations, not causal evidence.", "",
             f"Master columns ({len(master.columns)}):", "", ", ".join(f"`{c}`" for c in master.columns)]
    (out / "joint_barrier_descriptor_dictionary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    matched, audit, fatigue_registry = join_inventory()
    master, points = build_master(matched, fatigue_registry)
    correlations = response_correlations(master)
    shared = shared_predictors(master)
    pareto = build_pareto(master)

    pca = master[["candidate_id", "fracture_response_PC1", "fracture_response_PC2", "fatigue_PC1", "fatigue_PC2"]].copy()
    fatigue_meta = json.loads((VF / "analysis_audit.json").read_text())["pca"]["fatigue"]
    pca["fatigue_PC1_explained_variance_fraction"] = fatigue_meta["explained_variance_ratio"][0]
    pca["fatigue_PC2_explained_variance_fraction"] = fatigue_meta["explained_variance_ratio"][1]
    # Labels follow inspection of the saved 51-point f=0.95--1.20 loadings:
    # PC1 is weak/negative at onset and positive over the upper HCF branch;
    # PC2 changes sign later and is a pronounced low-to-high-f tilt.
    pca["fatigue_PC1_loading_interpretation"] = "UPPER_HCF_BRANCH_AMPLITUDE_RELATIVE_TO_ONSET"
    pca["fatigue_PC2_loading_interpretation"] = "LOW_TO_HIGH_F_TILT_AND_KNEE_POSITION_CONTRAST"
    pca["fatigue_PCA_eligibility"] = np.where(pca.fatigue_PC1.notna(), "CONTINUOUS_FINITE_F_0P95_TO_1P20", "INELIGIBLE_CENSOR_OR_COVERAGE_GAP")
    pca["censor_interpolation"] = False
    pca["PCA_scope"] = "fatigue PCA basis from authoritative v9.14 all-candidate finite continuous curves; scores intersected by exact fingerprint"
    q = pca.dropna(subset=["fracture_response_PC1", "fracture_response_PC2", "fatigue_PC1", "fatigue_PC2"])
    if len(q) >= 4:
        cca = small_cca_pls(q[["fracture_response_PC1", "fracture_response_PC2"]].to_numpy(float),
                            q[["fatigue_PC1", "fatigue_PC2"]].to_numpy(float))
        cca["x_variables"] = "fracture_response_PC1;fracture_response_PC2"
        cca["y_variables"] = "fatigue_PC1;fatigue_PC2"
    else:
        cca = pd.DataFrame([{"method": "NOT_ESTIMATED", "component": np.nan, "n": len(q),
                             "interpretation_scope": "INSUFFICIENT_INTERSECTION"}])

    master.to_csv(out / "joint_fracture_fatigue_candidate_master.csv", index=False)
    correlations.to_csv(out / "fracture_fatigue_response_correlations.csv", index=False)
    cca.to_csv(out / "fracture_fatigue_CCA_PLS.csv", index=False)
    shared.to_csv(out / "shared_barrier_predictor_summary.csv", index=False)
    pca.to_csv(out / "joint_response_pca_scores.csv", index=False)
    pareto.to_csv(out / "joint_candidate_pareto_front.csv", index=False)
    matched.to_csv(out / "joint_fingerprint_match_audit.csv", index=False)
    write_dictionary(out, master)
    make_figures(out, master, shared, pareto, points)

    audit.update({
        "analysis_git_head": git_head(), "analysis_type": "EXISTING_DATA_ONLY_NO_SIMULATIONS",
        "joint_master_rows": len(master), "fatigue_pca_eligible_intersection_count": int(pca.fatigue_PC1.notna().sum()),
        "cca_pls_complete_case_count": len(q), "canonical_intersection_count": int(master.candidate_id.isin(CANONICAL).sum()),
        "explicit_cycle_LCF_intersection_count": int(master.explicit_cycle_LCF.sum()),
        "accelerated_HCF_only_intersection_count": int(master.accelerated_HCF_only.sum()),
        "matched_2D_intersection_count": int(master.matched_2D_validation.sum()),
        "prospective_rows_available": 0,
        "quantitative_experimental_reference_arrays_found_in_repository": 0,
        "realism_label_reason": "no quantitative experimental envelope recovered; Pareto uses raw model-internal evidence objectives",
        "fatigue_semantics": "finite rates only; censors and partials preserved; explicit and accelerated modes not conflated",
        "physical_claim_scope": "MODEL_INTERNAL_PHYSICAL_PLAUSIBILITY",
    })
    (out / "joint_existing_analysis_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
