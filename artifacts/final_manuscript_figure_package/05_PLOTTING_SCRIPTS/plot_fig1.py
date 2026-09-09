"""Figure 1 -- Reduced-model (V1) response families across a shared barrier
continuation. All four panels are SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED
in the evidence package (paper_claim_evidence_matrix_v4.json, Fig1A-D): the
recovered axes, parameter-family ordering, and topology are verified, but no
manuscript-stated NUMBER is reproduced quantitatively. Consistent with that,
these panels plot the verified parameter-continuation TOPOLOGY (regime
transitions, shared entropy-family grid) rather than a fabricated
quantitative response curve that is not present in the portable evidence
bundle.

Source: codex/v10.2.30-paper-evidence-independent-verifier @
cbf2cf8bf73c649c1b87acd6a6a94f322f936bcc,
artifacts/paper_simulation_completion/source_bundle_figures_2_4/
  fig1AB_panel_A_waterfall_path.csv
  fig1AB_panel_B_fatigue_waterfall_path.csv
  fig1CD_panelC_curve_summary_v3.csv
  fig1CD_panelD_curve_summary_v3.csv
  fig1CD_panels_CD_manifest_v3.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_common import (CLASS_COLOR, CLASS_LABEL, FIGSIZE_SINGLE_COL, FIGSIZE_DOUBLE_COL,
                           add_panel_label, savefig_all, sha256_of)

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

REGIME_MAP = {"ceramic": "ceramic", "peak": "peak", "weakT": "weakT", "dbtt": "DBTT"}


def _read_csv(name):
    with (BUNDLE / name).open() as fh:
        return list(csv.DictReader(fh))


def _plot_waterfall_path(ax, rows, title):
    h0 = [float(r["H0_eV"]) for r in rows]
    chi = [float(r["chi_shield"]) for r in rows]
    ax.plot(h0, chi, color="0.6", linewidth=1.0, zorder=1)
    seen = set()
    for r in rows:
        cls = REGIME_MAP[r["regime_hint"]]
        lbl = CLASS_LABEL[cls] if cls not in seen else None
        seen.add(cls)
        ax.scatter(float(r["H0_eV"]), float(r["chi_shield"]), color=CLASS_COLOR[cls],
                   s=28, zorder=2, label=lbl, edgecolor="black", linewidth=0.4)
    ax.set_xlabel(r"Crack-opening barrier scale, $H_{0,c}$ (eV)")
    ax.set_ylabel(r"Shielding coefficient, $\chi_{\mathrm{shield}}$")
    ax.set_title(title, fontsize=10)


def panel_A(rows_a):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COL)
    _plot_waterfall_path(ax, rows_a, "Monotonic first-passage continuation")
    add_panel_label(ax, "A")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return fig


def panel_B(rows_b):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COL)
    _plot_waterfall_path(ax, rows_b, "Cyclic first-passage continuation")
    add_panel_label(ax, "B")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return fig


def panel_C(rows_c):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COL)
    lam = [float(r["Lambda_S_ref"]) for r in rows_c]
    slope = [float(r["high_cycle_abs_slope_MPa_per_decade"]) for r in rows_c]
    ax.scatter(lam, slope, color=CLASS_COLOR["weakT"], s=26, edgecolor="black", linewidth=0.4)
    ax.set_xlabel(r"Entropy-family coordinate, $\Lambda_{S,\mathrm{ref}}$")
    ax.set_ylabel("High-cycle S-N slope\n(MPa / decade)")
    ax.set_title("S-N-type initiation response, entropy family", fontsize=10)
    add_panel_label(ax, "C")
    fig.tight_layout()
    return fig


def panel_D(rows_d):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COL)
    lam = [float(r["Lambda_S_ref"]) for r in rows_d]
    amp = [float(r["peak_amp_frac_vs_300"]) for r in rows_d]
    sc = ax.scatter(lam, amp, c=[float(r["T_peak_K"]) for r in rows_d], cmap="viridis",
                     s=26, edgecolor="black", linewidth=0.4)
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$T_{\mathrm{peak}}$ (K)", fontsize=8)
    ax.axhline(0.0, color="0.7", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"Entropy-family coordinate, $\Lambda_{S,\mathrm{ref}}$")
    ax.set_ylabel("Intermediate-T strength\nanomaly, fraction vs. 300 K")
    ax.set_title("Fixed-rate strength-temperature response,\nsame entropy family", fontsize=10)
    add_panel_label(ax, "D")
    fig.tight_layout()
    return fig


def composite(rows_a, rows_b, rows_c, rows_d):
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_DOUBLE_COL)
    _plot_waterfall_path(axes[0, 0], rows_a, "Monotonic first-passage continuation")
    add_panel_label(axes[0, 0], "A")
    _plot_waterfall_path(axes[0, 1], rows_b, "Cyclic first-passage continuation")
    add_panel_label(axes[0, 1], "B")
    lam_c = [float(r["Lambda_S_ref"]) for r in rows_c]
    slope_c = [float(r["high_cycle_abs_slope_MPa_per_decade"]) for r in rows_c]
    axes[1, 0].scatter(lam_c, slope_c, color=CLASS_COLOR["weakT"], s=22, edgecolor="black", linewidth=0.4)
    axes[1, 0].set_xlabel(r"$\Lambda_{S,\mathrm{ref}}$"); axes[1, 0].set_ylabel("High-cycle S-N slope\n(MPa/decade)")
    axes[1, 0].set_title("S-N-type initiation response", fontsize=10)
    add_panel_label(axes[1, 0], "C")
    lam_d = [float(r["Lambda_S_ref"]) for r in rows_d]
    amp_d = [float(r["peak_amp_frac_vs_300"]) for r in rows_d]
    sc = axes[1, 1].scatter(lam_d, amp_d, c=[float(r["T_peak_K"]) for r in rows_d], cmap="viridis",
                             s=22, edgecolor="black", linewidth=0.4)
    cb = plt.colorbar(sc, ax=axes[1, 1], pad=0.02)
    cb.set_label(r"$T_{\mathrm{peak}}$ (K)", fontsize=8)
    axes[1, 1].axhline(0.0, color="0.7", linewidth=0.8, linestyle="--")
    axes[1, 1].set_xlabel(r"$\Lambda_{S,\mathrm{ref}}$"); axes[1, 1].set_ylabel("Strength anomaly\n(frac. vs 300K)")
    axes[1, 1].set_title("Fixed-rate strength-T response", fontsize=10)
    add_panel_label(axes[1, 1], "D")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLOR[c],
                          markeredgecolor="black", label=CLASS_LABEL[c], markersize=7) for c in CLASS_COLOR]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Figure 1. Reduced-model (V1) response families across a shared barrier continuation "
                 "(structural-source-qualified: topology verified, not a manuscript-numeric reproduction)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    return fig


def main():
    rows_a = _read_csv("fig1AB_panel_A_waterfall_path.csv")
    rows_b = _read_csv("fig1AB_panel_B_fatigue_waterfall_path.csv")
    rows_c = _read_csv("fig1CD_panelC_curve_summary_v3.csv")
    rows_d = _read_csv("fig1CD_panelD_curve_summary_v3.csv")

    SRC_OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in ["fig1AB_panel_A_waterfall_path.csv", "fig1AB_panel_B_fatigue_waterfall_path.csv",
                 "fig1CD_panelC_curve_summary_v3.csv", "fig1CD_panelD_curve_summary_v3.csv",
                 "fig1CD_panels_CD_manifest_v3.json"]:
        dest = SRC_OUT / name
        dest.write_bytes((BUNDLE / name).read_bytes())
        manifest[name] = sha256_of(dest)

    hashes = {}
    figs = dict(Fig1A=panel_A(rows_a), Fig1B=panel_B(rows_b), Fig1C=panel_C(rows_c), Fig1D=panel_D(rows_d))
    for stem, fig in figs.items():
        hashes[stem] = savefig_all(fig, stem, PANELS_OUT, PANELS_OUT, PANELS_OUT)
        savefig_all(fig, stem, PDF_OUT, SVG_OUT, PNG_OUT)
        plt.close(fig)

    fig_comp = composite(rows_a, rows_b, rows_c, rows_d)
    hashes["Figure1_composite"] = savefig_all(fig_comp, "Figure1_composite", COMPOSITE_OUT, COMPOSITE_OUT, COMPOSITE_OUT)
    savefig_all(fig_comp, "Figure1_composite", PDF_OUT, SVG_OUT, PNG_OUT)
    plt.close(fig_comp)

    (SRC_OUT / "fig1_source_data_hashes.json").write_text(json.dumps(
        dict(source_files=manifest, figure_hashes=hashes), indent=2, default=str))
    print("Figure 1: wrote 4 panels + 1 composite")


if __name__ == "__main__":
    main()
