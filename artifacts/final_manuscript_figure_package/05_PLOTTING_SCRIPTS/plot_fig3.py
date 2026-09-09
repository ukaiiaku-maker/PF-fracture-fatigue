"""Figure 3 -- FEM/CZM event-sampled propagation-resistance (R-)curves at
500 K, four canonical classes. QUALIFIED_SOURCE_RESULT_VERIFIED (Fig3, all
12 class-statistic rows + Sec2.15 saturation fits). Shows individual-seed
binned curves (not obscured behind the fit), the 5-seed class mean, and the
saturating fit K_R(Da) = K0 + DeltaK_R[1-exp{-(Da/ellR)^p}].

Source: codex/v10.2.30-paper-evidence-independent-verifier @
cbf2cf8bf73c649c1b87acd6a6a94f322f936bcc,
source_bundle_figures_2_4/sec2_15_seed_binned_Rcurves_long.csv (raw per-seed),
fig3_seed_Rcurve_metrics_and_fits.csv (raw per-seed scalar metrics),
fig3_class_mean_Rcurve_fits.csv (fitted saturation parameters).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_common import CLASS_COLOR, CLASS_LABEL, CLASS_ORDER, FIGSIZE_SINGLE_COL, FIGSIZE_DOUBLE_COL, \
    add_panel_label, savefig_all, sha256_of

PKG = Path(__file__).resolve().parents[1]
BUNDLE = Path("/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_worktrees/"
              "v10230-final-manuscript-figure-package/artifacts/paper_simulation_completion/"
              "source_bundle_figures_2_4")
SRC_OUT = PKG / "04_SOURCE_DATA" / "MAIN_TEXT"
PANELS_OUT = PKG / "01_MAIN_TEXT" / "INDIVIDUAL_PANELS"
COMPOSITE_OUT = PKG / "01_MAIN_TEXT" / "COMPOSITE_FIGURES"
PDF_OUT = PKG / "01_MAIN_TEXT" / "PDF"
SVG_OUT = PKG / "01_MAIN_TEXT" / "SVG"
PNG_OUT = PKG / "01_MAIN_TEXT" / "PNG_600DPI"


def _sat_model(da, K0, dK, ell, p):
    return K0 + dK * (1 - np.exp(-(da / ell) ** p))


def _read_csv(name):
    with (BUNDLE / name).open() as fh:
        return list(csv.DictReader(fh))


def _plot_class(ax, binned, fits, metrics, cls):
    rows = [r for r in binned if r["class"] == cls and r["complete"] == "True"]
    seeds = sorted(set(r["seed"] for r in rows))
    for i, seed in enumerate(seeds):
        srows = sorted([r for r in rows if r["seed"] == seed], key=lambda r: float(r["bin_center_um"]))
        x = [float(r["bin_center_um"]) for r in srows]
        y = [float(r["K_mean_MPa_sqrt_m"]) for r in srows]
        ax.plot(x, y, color=CLASS_COLOR[cls], alpha=0.35, linewidth=1.0,
                label="Individual seeds (n=5)" if i == 0 else None)

    # class-mean curve (mean of the 5 seeds at each shared bin)
    from collections import defaultdict
    by_bin = defaultdict(list)
    for r in rows:
        by_bin[round(float(r["bin_center_um"]), 2)].append(float(r["K_mean_MPa_sqrt_m"]))
    xb = sorted(by_bin)
    yb = [np.mean(by_bin[b]) for b in xb]
    ax.plot(xb, yb, color="black", linewidth=1.8, label="5-seed class mean")

    fit = [f for f in fits if f["class"] == cls][0]
    K0, dK, ell, p = float(fit["sat_K0"]), float(fit["sat_dK"]), float(fit["sat_ell_um"]), float(fit["sat_p"])
    xf = np.linspace(min(xb), max(xb), 200)
    ax.plot(xf, _sat_model(xf, K0, dK, ell, p), color=CLASS_COLOR[cls], linewidth=1.6, linestyle="--",
            label="Saturating fit")

    m = [mm for mm in metrics if mm["class"] == cls]
    K0_vals = [float(mm["K0_0_50_MPa_sqrt_m"]) for mm in m if mm["complete"] == "True"]
    Kss_vals = [float(mm["Kss_late_MPa_sqrt_m"]) for mm in m if mm["complete"] == "True"]
    dK_vals = [float(mm["DeltaK_late_minus_early_MPa_sqrt_m"]) for mm in m if mm["complete"] == "True"]
    txt = (f"$K_0$ = {np.mean(K0_vals):.2f}$\\pm${np.std(K0_vals, ddof=1):.2f}\n"
           f"$K_{{ss}}$ = {np.mean(Kss_vals):.2f}$\\pm${np.std(Kss_vals, ddof=1):.2f}\n"
           f"$\\Delta K_R$ = {np.mean(dK_vals):.2f}$\\pm${np.std(dK_vals, ddof=1):.2f}")
    ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9))

    ax.set_xlabel(r"Crack extension, $\Delta a$ ($\mu$m)")
    ax.set_ylabel(r"$K_R$ (MPa$\sqrt{\mathrm{m}}$)")
    ax.set_title(CLASS_LABEL[cls], fontsize=9)


def composite(binned, fits, metrics):
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_DOUBLE_COL)
    for ax, cls, lbl in zip(axes.flat, CLASS_ORDER, "ABCD"):
        _plot_class(ax, binned, fits, metrics, cls)
        add_panel_label(ax, lbl)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=8)
    fig.suptitle("Figure 3. FEM/CZM event-sampled propagation-resistance curves at 500 K "
                 r"(seed-resolved; $K_R(\Delta a)=K_0+\Delta K_R[1-\exp\{-(\Delta a/\ell_R)^p\}]$)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    return fig


def main():
    binned = _read_csv("sec2_15_seed_binned_Rcurves_long.csv")
    fits = _read_csv("fig3_class_mean_Rcurve_fits.csv")
    metrics = _read_csv("fig3_seed_Rcurve_metrics_and_fits.csv")

    SRC_OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in ["sec2_15_seed_binned_Rcurves_long.csv", "fig3_class_mean_Rcurve_fits.csv",
                 "fig3_seed_Rcurve_metrics_and_fits.csv"]:
        dest = SRC_OUT / name
        dest.write_bytes((BUNDLE / name).read_bytes())
        manifest[name] = sha256_of(dest)

    hashes = {}
    for cls, letter in zip(CLASS_ORDER, "ABCD"):
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COL)
        _plot_class(ax, binned, fits, metrics, cls)
        add_panel_label(ax, letter)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
        fig.tight_layout()
        stem = f"Fig3{letter}_{cls}"
        hashes[stem] = savefig_all(fig, stem, PANELS_OUT, PANELS_OUT, PANELS_OUT)
        savefig_all(fig, stem, PDF_OUT, SVG_OUT, PNG_OUT)
        plt.close(fig)

    fig_comp = composite(binned, fits, metrics)
    hashes["Figure3_composite"] = savefig_all(fig_comp, "Figure3_composite", COMPOSITE_OUT, COMPOSITE_OUT, COMPOSITE_OUT)
    savefig_all(fig_comp, "Figure3_composite", PDF_OUT, SVG_OUT, PNG_OUT)
    plt.close(fig_comp)

    (SRC_OUT / "fig3_source_data_hashes.json").write_text(json.dumps(
        dict(source_files=manifest, figure_hashes=hashes), indent=2, default=str))
    print("Figure 3: wrote 4 panels + 1 composite")


if __name__ == "__main__":
    main()
