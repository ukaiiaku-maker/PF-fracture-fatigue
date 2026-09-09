"""Supporting Information figures. Each figure materially supports
reproducibility or interpretation of a main-text claim and is not a
duplicate of a main-text figure. All source data are the same governed
portable bundle used for the main-text figures.

Source: codex/v10.2.30-paper-evidence-independent-verifier @
cbf2cf8bf73c649c1b87acd6a6a94f322f936bcc,
artifacts/paper_simulation_completion/source_bundle_figures_2_4/
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2_contingency, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_common import (CLASS_COLOR, CLASS_LABEL, CLASS_ORDER, SYSTEM_COLOR, SYSTEM_LABEL, SYSTEM_ORDER,
                           FIGSIZE_SINGLE_COL, FIGSIZE_DOUBLE_COL, FIGSIZE_DOUBLE_COL_TALL,
                           add_panel_label, savefig_all, sha256_of, style_log_axis)

PKG = Path(__file__).resolve().parents[1]
BUNDLE = Path("/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_worktrees/"
              "v10230-final-manuscript-figure-package/artifacts/paper_simulation_completion/"
              "source_bundle_figures_2_4")
SRC_OUT = PKG / "04_SOURCE_DATA" / "SUPPORTING_INFORMATION"
PANELS_OUT = PKG / "02_SUPPORTING_INFORMATION" / "INDIVIDUAL_PANELS"
COMPOSITE_OUT = PKG / "02_SUPPORTING_INFORMATION" / "COMPOSITE_FIGURES"
PDF_OUT = PKG / "02_SUPPORTING_INFORMATION" / "PDF"
SVG_OUT = PKG / "02_SUPPORTING_INFORMATION" / "SVG"
PNG_OUT = PKG / "02_SUPPORTING_INFORMATION" / "PNG_600DPI"

CONTEXT_LABEL = {
    "ctx_FCC_like_case29": "FCC-like", "ctx_shifted_ductile_case64": "Shifted ductile",
    "ctx_steep_cleavage_case35": "Steep cleavage", "ctx_slow_threshold_case101": "Slow threshold",
    "ctx_higher_barrier_case171": "Higher barrier", "ctx_plastic_shielded_case64_M1": "Plastic shielded",
}
CONTEXT_ORDER = list(CONTEXT_LABEL)


def _read_csv(name):
    with (BUNDLE / name).open() as fh:
        return list(csv.DictReader(fh))


def sat_model(da, K0, dK, ell, p):
    return K0 + dK * (1 - np.exp(-(da / ell) ** p))


HASHES = {}
MANIFEST = {}


def _save(fig, stem):
    HASHES[stem] = savefig_all(fig, stem, PANELS_OUT, PANELS_OUT, PANELS_OUT)
    savefig_all(fig, stem, PDF_OUT, SVG_OUT, PNG_OUT)
    plt.close(fig)


def _archive(name):
    if name in MANIFEST:
        return
    dest = SRC_OUT / name
    dest.write_bytes((BUNDLE / name).read_bytes())
    MANIFEST[name] = sha256_of(dest)


# SI-1: full six-system fatigue atlas -----------------------------------
def si1_full_atlas():
    rows = _read_csv("fig5A_atlas_2d_paris_points.csv")
    _archive("fig5A_atlas_2d_paris_points.csv")
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_COL)
    for case in SYSTEM_ORDER:
        rs = sorted([r for r in rows if r["case_label"] == case], key=lambda r: float(r["DeltaK_MPa_sqrtm"]))
        dk = [float(r["DeltaK_MPa_sqrtm"]) for r in rs]
        rate = [float(r["plot_da_dN_m_per_cycle"]) for r in rs]
        censored = [r["is_censored_upper_bound"] == "True" for r in rs]
        ok = [(d, v) for d, v, c in zip(dk, rate, censored) if not c]
        cen = [(d, v) for d, v, c in zip(dk, rate, censored) if c]
        if ok:
            ax.plot(*zip(*ok), color=SYSTEM_COLOR[case], marker="o", markersize=4, linewidth=1.1,
                    label=SYSTEM_LABEL[case])
        if cen:
            ax.plot(*zip(*cen), color=SYSTEM_COLOR[case], marker="v", markersize=5, linestyle="none",
                    markerfacecolor="none")
    style_log_axis(ax, "both")
    ax.set_xlabel(r"$\Delta K$ (MPa$\sqrt{\mathrm{m}}$)"); ax.set_ylabel(r"$da/dN$ (m/cycle)")
    ax.set_title("SI Figure 1. Complete six-system fatigue-atlas da/dN vs. "
                 r"$\Delta K$ (triangle = upper-bound censor)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    _save(fig, "SI_Fig1_full_six_system_atlas")


# SI-2: seed-resolved R-curve metric strip plot --------------------------
def si2_seed_resolved_metrics():
    rows = _read_csv("fig3_seed_Rcurve_metrics_and_fits.csv")
    _archive("fig3_seed_Rcurve_metrics_and_fits.csv")
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_DOUBLE_COL)
    for ax, col, ylabel in zip(axes, ["K0_0_50_MPa_sqrt_m", "Kss_late_MPa_sqrt_m",
                                       "DeltaK_late_minus_early_MPa_sqrt_m"],
                                [r"$K_0$", r"$K_{ss}$", r"$\Delta K_R$"]):
        for i, cls in enumerate(CLASS_ORDER):
            vals = [float(r[col]) for r in rows if r["class"] == cls and r["complete"] == "True"]
            x = np.full(len(vals), i) + np.random.default_rng(0).uniform(-0.1, 0.1, len(vals))
            ax.scatter(x, vals, color=CLASS_COLOR[cls], s=30, edgecolor="black", linewidth=0.4, zorder=3)
            ax.scatter([i], [np.mean(vals)], color="black", marker="_", s=400, zorder=4)
        ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_ORDER)
        ax.set_ylabel(ylabel + r" (MPa$\sqrt{\mathrm{m}}$)")
    fig.suptitle("SI Figure 2. All 20 individual seed-level R-curve metrics "
                 "(5 seeds x 4 classes; black bar = class mean)", fontsize=9, y=1.03)
    fig.tight_layout()
    _save(fig, "SI_Fig2_seed_resolved_metrics")


# SI-3: saturation-fit residuals ------------------------------------------
def si3_fit_residuals():
    binned = _read_csv("sec2_15_seed_binned_Rcurves_long.csv")
    fits = _read_csv("fig3_class_mean_Rcurve_fits.csv")
    _archive("sec2_15_seed_binned_Rcurves_long.csv"); _archive("fig3_class_mean_Rcurve_fits.csv")
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COL)
    for cls in CLASS_ORDER:
        rows = [r for r in binned if r["class"] == cls and r["complete"] == "True"]
        by_bin = defaultdict(list)
        for r in rows:
            by_bin[round(float(r["bin_center_um"]), 2)].append(float(r["K_mean_MPa_sqrt_m"]))
        xb = np.array(sorted(by_bin)); yb = np.array([np.mean(by_bin[b]) for b in xb])
        fit = [f for f in fits if f["class"] == cls][0]
        pred = sat_model(xb, float(fit["sat_K0"]), float(fit["sat_dK"]), float(fit["sat_ell_um"]), float(fit["sat_p"]))
        ax.plot(xb, yb - pred, color=CLASS_COLOR[cls], marker="o", markersize=3, linewidth=1.0,
                label=CLASS_LABEL[cls])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(r"Crack extension, $\Delta a$ ($\mu$m)")
    ax.set_ylabel(r"Residual, $K_{\mathrm{mean}}-K_{\mathrm{fit}}$ (MPa$\sqrt{\mathrm{m}}$)")
    ax.set_title("SI Figure 3. Saturation-fit residuals (class-mean curve\nminus "
                 r"$K_R(\Delta a)$ fit), from an independent refit", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "SI_Fig3_saturation_fit_residuals")


# SI-4: complete rate-response RMSE grid ----------------------------------
def si4_rate_response_grid():
    rows = _read_csv("fig4_first_passage_comparison_with_analytic_all_rates.csv")
    _archive("fig4_first_passage_comparison_with_analytic_all_rates.csv")
    rate_order = ["0.1x", "1x", "10x", "100x"]
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COL)
    width = 0.2
    x = np.arange(len(rate_order))
    for i, cls in enumerate(CLASS_ORDER):
        vals = []
        for rate in rate_order:
            rs = [r for r in rows if r["class"] == cls and r["rate_label"] == rate]
            diffs = [float(r["Kc_first_MPa_sqrt_m"]) - float(r["K_analytic_interp_MPa_sqrt_m"]) for r in rs]
            vals.append(math.sqrt(sum(d * d for d in diffs) / len(diffs)) if diffs else np.nan)
        ax.bar(x + (i - 1.5) * width, vals, width, color=CLASS_COLOR[cls], label=CLASS_LABEL[cls])
    ax.set_xticks(x); ax.set_xticklabels(rate_order)
    ax.set_xlabel("Nominal rate factor"); ax.set_ylabel(r"RMSE(FEM/CZM, V1) (MPa$\sqrt{\mathrm{m}}$)")
    ax.set_title("SI Figure 4. Complete rate x class RMSE table\n(underlies Figure 4)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "SI_Fig4_complete_rate_response_rmse")


# SI-5: censor/admissibility summary --------------------------------------
def si5_censor_summary():
    rows = _read_csv("fig2_first_passage_comparison_with_analytic_1x.csv")
    _archive("fig2_first_passage_comparison_with_analytic_1x.csv")
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COL)
    frameworks = ["PF sharp-front", "FEM/CZM"]
    complete_frac, incomplete_frac = [], []
    for fw in frameworks:
        rs = [r for r in rows if r["framework"] == fw]
        n_complete = sum(1 for r in rs if r["is_complete"] == "True")
        complete_frac.append(n_complete); incomplete_frac.append(len(rs) - n_complete)
    x = np.arange(len(frameworks))
    ax.bar(x, complete_frac, color="0.25", label="Complete run")
    ax.bar(x, incomplete_frac, bottom=complete_frac, color="0.75",
           label="Incomplete run (first passage still recorded)")
    ax.set_xticks(x); ax.set_xticklabels(frameworks)
    ax.set_ylabel("Number of class x temperature runs (1x rate)")
    ax.set_title("SI Figure 5. Run-completeness / admissibility summary\n(underlies Figure 2)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "SI_Fig5_censor_admissibility_summary")


# SI-6: six context-specific Kc-DKth fits ---------------------------------
def si6_context_fits():
    rows = _read_csv("fig6B_compact_joined_1360.csv")
    _archive("fig6B_compact_joined_1360.csv")
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_DOUBLE_COL_TALL)
    for ax, ctx in zip(axes.flat, CONTEXT_ORDER):
        rs = [r for r in rows if r["fracture_context"] == ctx]
        kc = np.array([float(r["Kc_first_MPa_sqrtm"]) for r in rs])
        dkth = np.array([float(r["DeltaK_th_MPa_sqrtm"]) for r in rs])
        r_val, _ = pearsonr(np.log10(kc), np.log10(dkth))
        ax.scatter(kc, dkth, s=8, alpha=0.5, color="tab:blue")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{CONTEXT_LABEL[ctx]}\n" r"$r_{\log}$=" f"{r_val:.4f}, n={len(rs)}", fontsize=8)
        ax.set_xlabel(r"$K_c$", fontsize=7); ax.set_ylabel(r"$\Delta K_{th}$", fontsize=7)
    fig.suptitle("SI Figure 6. Six context-specific " r"$K_c$" "-" r"$\Delta K_{th}$"
                 " log-space fits (underlies Figure 6B)", fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, "SI_Fig6_context_specific_Kc_DKth_fits")


# SI-7: complete AUC/CI results -------------------------------------------
def si7_complete_auc():
    manuscript = _read_csv("fig6_manuscript_statistics_table.csv")
    _archive("fig6_manuscript_statistics_table.csv")
    panelC = [r for r in manuscript if r["figure_panel"] == "C" and r["estimate"]]
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE_COL)
    fam_marker = {"SN_endurance_vs_Kc_same_T": "o", "SN_endurance_vs_DKth_same_T": "s"}
    contexts = sorted(set(r["context"] for r in panelC if r["context"] != "strength_only"))
    x_positions = {}
    idx = 0
    for T in [100.0, 300.0, 500.0]:
        for ctx in contexts:
            x_positions[(T, ctx)] = idx; idx += 1
    fam_display = {"SN_endurance_vs_Kc_same_T": "Kc", "SN_endurance_vs_DKth_same_T": "DKth"}
    for fam, marker in fam_marker.items():
        xs, ys, los, his = [], [], [], []
        for r in panelC:
            if r["analysis_family"] != fam or r["context"] not in contexts:
                continue
            key = (float(r["temperature_K"]), r["context"])
            if key not in x_positions:
                continue
            xs.append(x_positions[key]); ys.append(float(r["estimate"]))
            los.append(float(r["estimate"]) - float(r["ci95_low"]) if r["ci95_low"] else 0)
            his.append(float(r["ci95_high"]) - float(r["estimate"]) if r["ci95_high"] else 0)
        ax.errorbar(xs, ys, yerr=[los, his], fmt=marker, markersize=4, capsize=2,
                    label=fam_display[fam], alpha=0.85)
    ax.axhline(0.5, color="0.4", linestyle="--", label="Chance (AUC=0.5)")
    tick_labels = [f"{CONTEXT_LABEL.get(ctx, ctx)}\n{int(T)}K" for T, ctx in x_positions]
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(tick_labels, fontsize=5, rotation=90)
    ax.set_ylabel("AUC")
    ax.set_title("SI Figure 7. Complete AUC + 95% bootstrap-CI results,\nall temperatures x contexts "
                 "(underlies Figure 6C)", fontsize=9)
    ax.legend(fontsize=7)
    fig.subplots_adjust(bottom=0.35, top=0.85, left=0.08, right=0.98)  # rotated labels: skip tight_layout
    _save(fig, "SI_Fig7_complete_AUC_CI_results")


# SI-8: synthetic-identifiability sparse-vs-complete --------------------
def si8_identifiability():
    inv = _read_csv("SI_inversion_summary.csv")
    _archive("SI_inversion_summary.csv")
    truth_rms = {}
    for regime in ["ceramic", "peak", "weakT", "dbtt"]:
        rows = _read_csv(f"SI_{regime}_truth_barrier_grid.csv")
        _archive(f"SI_{regime}_truth_barrier_grid.csv")
        vals = [float(r["G_emit_eV"]) for r in rows]
        truth_rms[regime] = math.sqrt(sum(v * v for v in vals) / len(vals))
    by_acq_regime = defaultdict(list)
    for r in inv:
        pct = 100.0 * float(r["rmse_G_emit_eV"]) / truth_rms[r["regime"]]
        by_acq_regime[(r["acquisition"], r["regime"])].append(pct)

    def median(xs):
        xs = sorted(xs); n = len(xs); mid = n // 2
        return xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])

    regimes = ["ceramic", "peak", "weakT", "dbtt"]
    sparse = [median(by_acq_regime[("A_sparse_fracture", r)]) for r in regimes]
    complete = [median(by_acq_regime[("G_full_universe", r)]) for r in regimes]
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COL)
    x = np.arange(len(regimes)); width = 0.35
    ax.bar(x - width / 2, sparse, width, color="tab:red", label="Sparse fracture-only\nacquisition")
    ax.bar(x + width / 2, complete, width, color="tab:blue", label="Complete 75-condition\nacquisition")
    ax.set_xticks(x); ax.set_xticklabels([CLASS_LABEL.get(r if r != "dbtt" else "DBTT", r) for r in regimes], fontsize=7)
    ax.set_ylabel("Emission-landscape\nrecovery error (%)")
    ax.set_title("SI Figure 8. Synthetic-identifiability recovery error:\nsparse vs. complete acquisition, "
                 "4 hidden regimes", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "SI_Fig8_identifiability_sparse_vs_complete")


# SI-9: V1 parameter-continuation map (Fig1's underlying parameter family) --
def si9_parameter_map():
    rows_a = _read_csv("fig1AB_panel_A_waterfall_path.csv")
    _archive("fig1AB_panel_A_waterfall_path.csv")
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COL)
    n_sat_finite = [r for r in rows_a if r["N_sat"] != "inf"]
    n_sat_inf = [r for r in rows_a if r["N_sat"] == "inf"]
    ax.scatter([float(r["H0_eV"]) for r in n_sat_inf], [float(r["chi_shield"]) for r in n_sat_inf],
               color="0.6", s=25, label=r"$N_{sat}=\infty$", marker="o")
    sc = ax.scatter([float(r["H0_eV"]) for r in n_sat_finite], [float(r["chi_shield"]) for r in n_sat_finite],
                    c=[float(r["N_sat"]) for r in n_sat_finite], cmap="plasma", s=30, marker="s",
                    label=r"$N_{sat}$ finite (color-coded)")
    cb = plt.colorbar(sc, ax=ax); cb.set_label(r"$N_{sat}$", fontsize=8)
    ax.set_xlabel(r"$H_{0,c}$ (eV)"); ax.set_ylabel(r"$\chi_{\mathrm{shield}}$")
    ax.set_title("SI Figure 9. Full V1 continuation-path parameter map\n(underlies Figure 1A/B)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "SI_Fig9_parameter_family_map")


def main():
    SRC_OUT.mkdir(parents=True, exist_ok=True)
    for fn in [si1_full_atlas, si2_seed_resolved_metrics, si3_fit_residuals, si4_rate_response_grid,
               si5_censor_summary, si6_context_fits, si7_complete_auc, si8_identifiability, si9_parameter_map]:
        fn()
    (SRC_OUT / "si_source_data_hashes.json").write_text(json.dumps(
        dict(source_files=MANIFEST, figure_hashes=HASHES), indent=2, default=str))
    print(f"SI figures: wrote {len(HASHES)} figures")


if __name__ == "__main__":
    main()
