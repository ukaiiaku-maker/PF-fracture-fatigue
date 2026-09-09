"""Figure 7 -- Cohesive-law mapping, peak class, 900 K. Compares the EXP-
floor barrier form's activation enthalpy H(sigma) and velocity proxy v(sigma)
against three standard cohesive-law fits (bilinear, exponential, polynomial).
QUALIFIED_SOURCE_RESULT_VERIFIED. The fitted mappings are clearly distinguished
from physical simulation (there is no new physical simulation here -- this is
a deterministic fit of standard cohesive forms to the EXP-floor curve).

Source: codex/v10.2.30-paper-evidence-independent-verifier @
cbf2cf8bf73c649c1b87acd6a6a94f322f936bcc, source_bundle_figures_2_4/
  fig7_peak_900K_comparison_curves.csv, fig7_peak_900K_fitted_parameters.csv
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_common import FIGSIZE_SINGLE_COL, FIGSIZE_DOUBLE_COL, add_panel_label, savefig_all, sha256_of

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

FORM_STYLE = {
    "EXPfloor": dict(color="black", linestyle="-", label="EXP-floor (physical barrier form)"),
    "bilinear": dict(color="tab:blue", linestyle="--", label="Bilinear cohesive fit"),
    "exponential": dict(color="tab:red", linestyle="-.", label="Exponential cohesive fit"),
    "polynomial": dict(color="tab:green", linestyle=":", label="Polynomial cohesive fit"),
}


def _read_csv(name):
    with (BUNDLE / name).open() as fh:
        return list(csv.DictReader(fh))


H_COL = {"EXPfloor": "H_EXPfloor_eV", "bilinear": "bilinear_H_eV",
         "exponential": "exponential_H_eV", "polynomial": "polynomial_H_eV"}
V_COL = {"EXPfloor": "v_EXPfloor_m_per_s", "bilinear": "bilinear_v_m_per_s",
         "exponential": "exponential_v_m_per_s", "polynomial": "polynomial_v_m_per_s"}


def panel_A(ax, rows):
    sigma = [float(r["sigma_GPa"]) for r in rows]
    for form in ["EXPfloor", "bilinear", "exponential", "polynomial"]:
        H = [float(r[H_COL[form]]) for r in rows]
        ax.plot(sigma, H, linewidth=1.6 if form == "EXPfloor" else 1.3, **FORM_STYLE[form])
    ax.set_xlabel(r"Local stress, $\sigma$ (GPa)")
    ax.set_ylabel("Activation enthalpy, H (eV)")
    ax.set_title("Peak class, 900 K: enthalpy mapping", fontsize=8.5)


def panel_B(ax, rows):
    sigma = [float(r["sigma_GPa"]) for r in rows]
    for form in ["EXPfloor", "bilinear", "exponential", "polynomial"]:
        v = [float(r[V_COL[form]]) for r in rows]
        ax.plot(sigma, v, linewidth=1.6 if form == "EXPfloor" else 1.3, **FORM_STYLE[form])
    ax.set_yscale("log")
    ax.set_xlabel(r"Local stress, $\sigma$ (GPa)")
    ax.set_ylabel("Velocity proxy, v (m/s)")
    v_all = [float(r["v_EXPfloor_m_per_s"]) for r in rows if float(r.get("v_EXPfloor_m_per_s", 0) or 0) > 0]
    orders = math.log10(max(v_all) / min(v_all))
    ax.set_title(f"Velocity-proxy dynamic range:\n{orders:.1f} orders of magnitude", fontsize=8.5)


def composite(rows, fitted):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE_COL)
    panel_A(axes[0], rows); add_panel_label(axes[0], "A")
    panel_B(axes[1], rows); add_panel_label(axes[1], "B")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=8)
    fig.suptitle("Figure 7. Cohesive-law mapping between the EXP-floor barrier form and standard "
                 "cohesive representations (peak class, 900 K; fitted mapping, not a new physical "
                 "simulation)", fontsize=8.5, y=1.03)
    fig.tight_layout()
    return fig


def main():
    rows = _read_csv("fig7_peak_900K_comparison_curves.csv")
    fitted = _read_csv("fig7_peak_900K_fitted_parameters.csv")

    SRC_OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in ["fig7_peak_900K_comparison_curves.csv", "fig7_peak_900K_fitted_parameters.csv"]:
        dest = SRC_OUT / name
        dest.write_bytes((BUNDLE / name).read_bytes())
        manifest[name] = sha256_of(dest)

    hashes = {}
    for letter, fn in zip("AB", [panel_A, panel_B]):
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COL)
        fn(ax, rows); add_panel_label(ax, letter)
        ax.legend(fontsize=6.5, loc="best")
        fig.tight_layout()
        stem = f"Fig7{letter}"
        hashes[stem] = savefig_all(fig, stem, PANELS_OUT, PANELS_OUT, PANELS_OUT)
        savefig_all(fig, stem, PDF_OUT, SVG_OUT, PNG_OUT)
        plt.close(fig)

    fig_comp = composite(rows, fitted)
    hashes["Figure7_composite"] = savefig_all(fig_comp, "Figure7_composite", COMPOSITE_OUT, COMPOSITE_OUT, COMPOSITE_OUT)
    savefig_all(fig_comp, "Figure7_composite", PDF_OUT, SVG_OUT, PNG_OUT)
    plt.close(fig_comp)

    (SRC_OUT / "fig7_source_data_hashes.json").write_text(json.dumps(
        dict(source_files=manifest, figure_hashes=hashes), indent=2, default=str))
    print("Figure 7: wrote 2 panels + 1 composite")


if __name__ == "__main__":
    main()
