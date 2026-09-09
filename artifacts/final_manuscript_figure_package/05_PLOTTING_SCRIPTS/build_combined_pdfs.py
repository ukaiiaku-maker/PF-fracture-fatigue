"""Build the four combined-review PDFs (08_COMBINED_REVIEW_FILES/) and the
contact sheet.

Implementation note (disclosed in the package README and QC report): no PDF-
merge library (pypdf/PyMuPDF/pikepdf) or command-line tool (qpdf/
ghostscript) is available in this environment. The combined multi-page
review PDFs are therefore built by placing each figure's already-generated
600-DPI PNG export (itself produced directly by matplotlib from the same
vector figure, not a screen capture) as a full-page image via matplotlib's
own PDF backend, one page per figure. The authoritative, fully vector PDF
export of every individual figure remains in
01_MAIN_TEXT/PDF, 02_SUPPORTING_INFORMATION/PDF, and 03_REBONDING_PART_X/PDF
-- the combined files here are a review convenience, not a replacement for
those.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

PKG = Path(__file__).resolve().parents[1]
OUT = PKG / "08_COMBINED_REVIEW_FILES"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_TEXT_ORDER = [f"Figure{n}_composite" for n in range(1, 8)]
SI_ORDER = [f"SI_Fig{n}_{name}" for n, name in [
    (1, "full_six_system_atlas"), (2, "seed_resolved_metrics"), (3, "saturation_fit_residuals"),
    (4, "complete_rate_response_rmse"), (5, "censor_admissibility_summary"),
    (6, "context_specific_Kc_DKth_fits"), (7, "complete_AUC_CI_results"),
    (8, "identifiability_sparse_vs_complete"), (9, "parameter_family_map"),
]]
# Per the mission spec, ALL_REBONDING_PART_X_FIGURES.pdf contains the CURATED
# main set only (~8-12 figures), not the complete 24-figure diagnostic set.
PART_X_CURATED_ORDER = [
    "fig01_pcb_phase_trajectory", "fig04_screen_S_h_vs_R", "fig05_screen_frequency_response",
    "fig14_reversible_vs_persistent_curves", "fig07_screen_passivation_chemistry_response",
    "fig08_screen_cohesive_strength_response", "fig09_absolute_developed_da_dN_vs_Kmax",
    "px6_S_h_developed_vs_Kmax_all_protocols", "fig11_local_slope_curvature",
    "px6_px5_static_shield_vs_dynamic", "px6_seed_robustness_slope_comparison",
    "fig06_dwell_invalidated_vs_corrected",
]
PART_X_COMPLETE_DIAGNOSTIC_ORDER = PART_X_CURATED_ORDER + [
    "fig02_transition_actions_fluxes", "fig03_screen_contact_exposure_vs_R",
    "fig12_reduced_frequency_developed_curves", "fig13_reversible_vs_passivation_curves",
    "fig17_mechanism_diagnostics_vs_Kmax", "fig18_admissibility_map",
    "developed_da_dN_vs_Kmax_zero_and_finite", "S_h_vs_Kmax_all_windows",
    "local_delta_m_and_secants", "eventwise_and_rolling_waiting_time_ratios",
    "pB_contact_time_action_weighted_Krebond_vs_Kmax", "true_terminal_window_persistence",
]


def _page(pdf, png_path: Path, title: str):
    img = mpimg.imread(png_path)
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.90])
    ax.imshow(img); ax.axis("off")
    fig.text(0.5, 0.96, title, ha="center", fontsize=11)
    pdf.savefig(fig)
    plt.close(fig)


def build(section_dir: str, order: list[str], out_name: str, title_prefix: str):
    png_dir = PKG / section_dir / "PNG_600DPI"
    out_path = OUT / out_name
    n_written = 0
    with PdfPages(out_path) as pdf:
        for stem in order:
            p = png_dir / f"{stem}.png"
            if not p.is_file():
                print(f"  MISSING for {out_name}: {p}")
                continue
            _page(pdf, p, f"{title_prefix}: {stem}")
            n_written += 1
    print(f"{out_name}: {n_written} pages")
    return n_written


def build_contact_sheet():
    import csv
    manifest_path = PKG / "07_QC_AND_PROVENANCE" / "figure_manifest.csv"
    rows = list(csv.DictReader(manifest_path.open()))
    out_path = OUT / "ALL_FIGURES_CONTACT_SHEET.pdf"
    per_page = 12
    with PdfPages(out_path) as pdf:
        for i in range(0, len(rows), per_page):
            chunk = rows[i:i + per_page]
            fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
            for ax, row in zip(axes.flat, chunk):
                png_path = PKG / row["png_path"] if row["png_path"] else None
                if png_path and png_path.is_file():
                    img = mpimg.imread(png_path)
                    ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"{row['figure_id']}\n[{row['section']}]", fontsize=5.5)
            for ax in axes.flat[len(chunk):]:
                ax.axis("off")
            fig.suptitle(f"Final Figure Package -- Contact Sheet (page {i//per_page + 1})", fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"ALL_FIGURES_CONTACT_SHEET.pdf: {len(rows)} figures, "
          f"{(len(rows) + per_page - 1)//per_page} pages")


def main():
    n_main = build("01_MAIN_TEXT", MAIN_TEXT_ORDER, "ALL_MAIN_TEXT_FIGURES.pdf", "Main text")
    n_si = build("02_SUPPORTING_INFORMATION", SI_ORDER, "ALL_SUPPORTING_FIGURES.pdf", "SI")
    n_px = build("03_REBONDING_PART_X", PART_X_CURATED_ORDER, "ALL_REBONDING_PART_X_FIGURES.pdf",
                 "Part X (curated)")
    build_contact_sheet()
    assert n_main == 7, f"expected 7 main-text composite pages, got {n_main}"
    assert n_si == 9, f"expected 9 SI pages, got {n_si}"
    assert n_px == 12, f"expected 12 curated Part X pages, got {n_px}"


if __name__ == "__main__":
    main()
