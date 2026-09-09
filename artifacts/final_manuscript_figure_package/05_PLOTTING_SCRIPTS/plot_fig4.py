"""Figure 4 -- Loading-rate dependence of FEM/CZM first passage and the
rate-matched V1 analytical prediction, nominal rate factors 0.1x/1x/10x/100x,
theta=45 degrees (parsed and verified from the run config, not hardcoded).
QUALIFIED_SOURCE_RESULT_VERIFIED (all 4 Fig4 claim rows).

Required wording (per the accepted manuscript corrections): "FEM/CZM shows
no resolved local maximum on the available temperature grid" -- NOT a
quantitatively established "muted shoulder" (the verifier's own zero-local-
maxima check establishes peak ABSENCE on the sampled grid, not a formally
defined and measured shoulder metric). The DBTT transition shift is shown
using BOTH independent transition-location definitions from the evidence
audit (shelf-midpoint crossing and maximum-|dK/dT|-gradient), each shown as
a distinct marker on the DBTT panel.

Source: codex/v10.2.30-paper-evidence-independent-verifier @
cbf2cf8bf73c649c1b87acd6a6a94f322f936bcc,
source_bundle_figures_2_4/fig4_first_passage_comparison_with_analytic_all_rates.csv,
fig4_analytical_predictions_by_rate_fine_grid.csv, fig4_comparison_config.json.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

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

RATE_ORDER = ["0.1x", "1x", "10x", "100x"]
RATE_LINESTYLE = {"0.1x": ":", "1x": "-", "10x": "--", "100x": "-."}
RATE_MARKER = {"0.1x": "v", "1x": "o", "10x": "s", "100x": "^"}

# Verified DBTT crossing temperatures (shelf-midpoint proxy, primary; from
# recompute_Fig4_DBTT_transition_shift) and the independent max-gradient
# cross-check (from the same function's secondary definition).
DBTT_SHELF_MIDPOINT_K = {"0.1x": 788.9, "1x": 898.2, "10x": 1031.4, "100x": 1125.1}
DBTT_MAX_GRADIENT_K = {"0.1x": 750.0, "1x": 850.0, "10x": 1050.0, "100x": 1150.0}


def _read_csv(name):
    with (BUNDLE / name).open() as fh:
        return list(csv.DictReader(fh))


def _plot_class(ax, rows, cls):
    for rate in RATE_ORDER:
        rs = sorted([r for r in rows if r["class"] == cls and r["rate_label"] == rate],
                    key=lambda r: float(r["T_K"]))
        T = [float(r["T_K"]) for r in rs]
        K_fem = [float(r["Kc_first_MPa_sqrt_m"]) for r in rs]
        K_an = [float(r["K_analytic_interp_MPa_sqrt_m"]) for r in rs]
        ax.plot(T, K_an, color="0.5", linestyle=RATE_LINESTYLE[rate], linewidth=1.1, zorder=1)
        ax.plot(T, K_fem, color=CLASS_COLOR[cls], linestyle="none", marker=RATE_MARKER[rate],
                markersize=4.5, markeredgecolor="black", markeredgewidth=0.3, label=rate, zorder=2)

    if cls == "DBTT":
        for rate in RATE_ORDER:
            ax.axvline(DBTT_SHELF_MIDPOINT_K[rate], color="0.75", linewidth=0.7, linestyle=":", zorder=0)
        ax.text(0.02, 0.95, "Vertical lines: shelf-midpoint transition\nproxy per rate (see panel note)",
                transform=ax.transAxes, fontsize=6.0, va="top", color="0.4")

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$K_c$ (MPa$\sqrt{\mathrm{m}}$)")
    ax.set_title(CLASS_LABEL[cls], fontsize=9)


def _dbtt_shift_inset(ax):
    """Small companion panel showing the two independent transition-
    temperature definitions both shift monotonically with rate."""
    x = list(range(len(RATE_ORDER)))
    y1 = [DBTT_SHELF_MIDPOINT_K[r] for r in RATE_ORDER]
    y2 = [DBTT_MAX_GRADIENT_K[r] for r in RATE_ORDER]
    ax.plot(x, y1, marker="o", color=CLASS_COLOR["DBTT"], label="Shelf-midpoint\ncrossing (primary)")
    ax.plot(x, y2, marker="s", color="0.4", linestyle="--", label="Max-$|dK/dT|$-gradient\n(independent check)")
    ax.set_xticks(x); ax.set_xticklabels(RATE_ORDER)
    ax.set_xlabel("Nominal rate factor")
    ax.set_ylabel("DBTT transition\ntemperature proxy (K)")
    ax.set_title("Directional shift confirmed by\ntwo independent definitions", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper left")


def composite(rows):
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_DOUBLE_COL)
    for ax, cls, lbl in zip(axes.flat, CLASS_ORDER, "ABCD"):
        _plot_class(ax, rows, cls)
        add_panel_label(ax, lbl)
    handles, labels = axes.flat[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=4, title="Nominal rate factor",
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=8)
    fig.suptitle("Figure 4. Loading-rate dependence of FEM/CZM first passage and the rate-matched V1 "
                 r"analytical prediction ($\theta$=45$^\circ$). FEM/CZM shows no resolved local maximum "
                 "on the available temperature grid for the peak class at any rate.", fontsize=8.5, y=1.03)
    fig.tight_layout()
    return fig


def main():
    rows = _read_csv("fig4_first_passage_comparison_with_analytic_all_rates.csv")
    cfg = json.loads((BUNDLE / "fig4_comparison_config.json").read_text())
    theta = float(re.search(r"theta(\d+(?:\.\d+)?)", cfg.get("root", "")).group(1))
    assert theta == 45.0, f"theta parsed as {theta}, expected 45.0"

    SRC_OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in ["fig4_first_passage_comparison_with_analytic_all_rates.csv",
                 "fig4_analytical_predictions_by_rate_fine_grid.csv", "fig4_comparison_config.json"]:
        dest = SRC_OUT / name
        dest.write_bytes((BUNDLE / name).read_bytes())
        manifest[name] = sha256_of(dest)

    hashes = {}
    for cls, letter in zip(CLASS_ORDER, "ABCD"):
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COL)
        _plot_class(ax, rows, cls)
        add_panel_label(ax, letter)
        ax.legend(title="Rate factor", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
        fig.tight_layout()
        stem = f"Fig4{letter}_{cls}"
        hashes[stem] = savefig_all(fig, stem, PANELS_OUT, PANELS_OUT, PANELS_OUT)
        savefig_all(fig, stem, PDF_OUT, SVG_OUT, PNG_OUT)
        plt.close(fig)

    fig_e, ax_e = plt.subplots(figsize=FIGSIZE_SINGLE_COL)
    _dbtt_shift_inset(ax_e)
    add_panel_label(ax_e, "E")
    fig_e.tight_layout()
    hashes["Fig4E_dbtt_shift_definitions"] = savefig_all(fig_e, "Fig4E_dbtt_shift_definitions",
                                                          PANELS_OUT, PANELS_OUT, PANELS_OUT)
    savefig_all(fig_e, "Fig4E_dbtt_shift_definitions", PDF_OUT, SVG_OUT, PNG_OUT)
    plt.close(fig_e)

    fig_comp = composite(rows)
    hashes["Figure4_composite"] = savefig_all(fig_comp, "Figure4_composite", COMPOSITE_OUT, COMPOSITE_OUT, COMPOSITE_OUT)
    savefig_all(fig_comp, "Figure4_composite", PDF_OUT, SVG_OUT, PNG_OUT)
    plt.close(fig_comp)

    (SRC_OUT / "fig4_source_data_hashes.json").write_text(json.dumps(
        dict(source_files=manifest, figure_hashes=hashes, theta_deg_verified=theta,
             dbtt_shelf_midpoint_K=DBTT_SHELF_MIDPOINT_K, dbtt_max_gradient_K=DBTT_MAX_GRADIENT_K),
        indent=2, default=str))
    print("Figure 4: wrote 4 panels + 1 inset (E) + 1 composite")


if __name__ == "__main__":
    main()
