"""Figure 2 -- Matched first-passage comparison at the 1x loading rate:
V1 analytical vs PF/sharp-front vs FEM/CZM, four canonical classes.
QUALIFIED_SOURCE_RESULT_VERIFIED (all 5 Fig2 claim rows). RMS deviations
(ceramic 0.23, weakT 0.23, peak 1.18, DBTT 1.33 MPa*sqrt(m)) are the
independently recomputed, verified values -- not re-derived here, only
displayed.

Source: codex/v10.2.30-paper-evidence-independent-verifier @
cbf2cf8bf73c649c1b87acd6a6a94f322f936bcc,
source_bundle_figures_2_4/fig2_first_passage_comparison_with_analytic_1x.csv
and fig2_four_class_analytical_prediction_final_fine_grid.csv
"""
from __future__ import annotations

import csv
import json
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

RMS_VERIFIED = {"ceramic": 0.23, "weakT": 0.23, "peak": 1.18, "DBTT": 1.33}


def _read_csv(name):
    with (BUNDLE / name).open() as fh:
        return list(csv.DictReader(fh))


def _plot_class(ax, rows, fine_rows, cls):
    fine = sorted([r for r in fine_rows if r["class"] == cls], key=lambda r: float(r["T_K"]))
    Tf = [float(r["T_K"]) for r in fine]
    Kf = [float(r["K_target_MPa_sqrt_m"]) for r in fine]
    ax.plot(Tf, Kf, color="black", linewidth=1.3, label="V1 analytical", zorder=1)

    for framework, marker, base_color in [("PF sharp-front", "o", "0.35"), ("FEM/CZM", "s", CLASS_COLOR[cls])]:
        rs = sorted([r for r in rows if r["framework"] == framework and r["class"] == cls],
                    key=lambda r: float(r["T_K"]))
        T_complete = [float(r["T_K"]) for r in rs if r["is_complete"] == "True"]
        K_complete = [float(r["Kc_first_MPa_sqrt_m"]) for r in rs if r["is_complete"] == "True"]
        T_incomplete = [float(r["T_K"]) for r in rs if r["is_complete"] != "True"]
        K_incomplete = [float(r["Kc_first_MPa_sqrt_m"]) for r in rs if r["is_complete"] != "True"]
        lbl = "PF/sharp-front (complete)" if framework == "PF sharp-front" else "FEM/CZM (complete)"
        ax.scatter(T_complete, K_complete, marker=marker, s=32, facecolor=base_color,
                   edgecolor="black", linewidth=0.5, label=lbl, zorder=3)
        if T_incomplete:
            lbl2 = "PF/sharp-front (incomplete run,\nfirst passage recorded)" if framework == "PF sharp-front" \
                else "FEM/CZM (incomplete run)"
            ax.scatter(T_incomplete, K_incomplete, marker=marker, s=32, facecolor="none",
                       edgecolor=base_color, linewidth=1.3, label=lbl2, zorder=3)

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$K_c$ (MPa$\sqrt{\mathrm{m}}$)")
    ax.set_title(f"{CLASS_LABEL[cls]}: RMS(FEM/CZM vs. V1) = {RMS_VERIFIED[cls]:.2f} " r"MPa$\sqrt{\mathrm{m}}$",
                 fontsize=8.5)


def composite(rows, fine_rows):
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_DOUBLE_COL)
    for ax, cls, lbl in zip(axes.flat, CLASS_ORDER, "ABCD"):
        _plot_class(ax, rows, fine_rows, cls)
        add_panel_label(ax, lbl)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=8)
    fig.suptitle("Figure 2. Matched first-passage comparison at the 1x loading rate: V1 analytical, "
                 "PF/sharp-front, and FEM/CZM", fontsize=9, y=1.02)
    fig.tight_layout()
    return fig


def main():
    rows = _read_csv("fig2_first_passage_comparison_with_analytic_1x.csv")
    fine_rows = _read_csv("fig2_four_class_analytical_prediction_final_fine_grid.csv")

    SRC_OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in ["fig2_first_passage_comparison_with_analytic_1x.csv",
                 "fig2_four_class_analytical_prediction_final_fine_grid.csv"]:
        dest = SRC_OUT / name
        dest.write_bytes((BUNDLE / name).read_bytes())
        manifest[name] = sha256_of(dest)

    hashes = {}
    for cls, letter in zip(CLASS_ORDER, "ABCD"):
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COL)
        _plot_class(ax, rows, fine_rows, cls)
        add_panel_label(ax, letter)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
        fig.tight_layout()
        stem = f"Fig2{letter}_{cls}"
        hashes[stem] = savefig_all(fig, stem, PANELS_OUT, PANELS_OUT, PANELS_OUT)
        savefig_all(fig, stem, PDF_OUT, SVG_OUT, PNG_OUT)
        plt.close(fig)

    fig_comp = composite(rows, fine_rows)
    hashes["Figure2_composite"] = savefig_all(fig_comp, "Figure2_composite", COMPOSITE_OUT, COMPOSITE_OUT, COMPOSITE_OUT)
    savefig_all(fig_comp, "Figure2_composite", PDF_OUT, SVG_OUT, PNG_OUT)
    plt.close(fig_comp)

    (SRC_OUT / "fig2_source_data_hashes.json").write_text(json.dumps(
        dict(source_files=manifest, figure_hashes=hashes, verified_rms=RMS_VERIFIED), indent=2, default=str))
    print("Figure 2: wrote 4 panels + 1 composite")


if __name__ == "__main__":
    main()
