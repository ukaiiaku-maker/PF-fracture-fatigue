"""Shared plotting-style module for the final manuscript figure package.

Import this module before creating any figure. It sets matplotlib rcParams
for a consistent, publication-quality, grayscale-compatible style across
every main-text, SI, and (where regenerated) Part X figure, and provides:

  - CLASS_COLOR / CLASS_MARKER / CLASS_LABEL: the four canonical barrier
    classes (ceramic, weakT, peak, DBTT), one consistent mapping used
    everywhere in the paper.
  - SYSTEM_COLOR / SYSTEM_MARKER / SYSTEM_LABEL: the six fatigue-atlas
    systems, one consistent mapping, with descriptive display names
    (internal case IDs are never shown on-plot; see the source-data CSVs
    for the ID mapping).
  - FIDELITY_LINESTYLE / FIDELITY_MARKER: distinguishes V1 analytical,
    PF/sharp-front, and FEM/CZM by line style and marker shape (not color
    alone), so the figures remain interpretable in grayscale.
  - CENSOR_MARKER, INVALIDATED_STYLE, etc.: shared semantic conventions for
    right-censored points, numerically excluded points, and invalidated
    results.
  - savefig_all(fig, stem, out_dirs): saves a given figure as .pdf, .svg,
    and .png (600 dpi) into the given directories, and returns a dict of
    {ext: sha256} for the manifest.
  - FIGSIZE_SINGLE_COL / FIGSIZE_DOUBLE_COL: journal-scale figure sizes
    (85 mm and 175 mm nominal width) at a 2x working canvas so exported
    text reduces to ~8-10 pt at final placed size.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

MM_PER_INCH = 25.4
WORKING_SCALE = 2.0  # 2x working canvas; final placed size is half this

def _fig_width_in(mm: float) -> float:
    return (mm / MM_PER_INCH) * WORKING_SCALE

FIGSIZE_SINGLE_COL = (_fig_width_in(85), _fig_width_in(85) * 0.78)
FIGSIZE_DOUBLE_COL = (_fig_width_in(178), _fig_width_in(178) * 0.62)
FIGSIZE_DOUBLE_COL_TALL = (_fig_width_in(178), _fig_width_in(178) * 0.95)

BASE_FONT_PT = 9 * WORKING_SCALE  # reduces to ~9pt at final placed size
TITLE_FONT_PT = 10 * WORKING_SCALE
PANEL_LABEL_FONT_PT = 12 * WORKING_SCALE

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.25,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.25,
    "ytick.major.width": 1.25,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": BASE_FONT_PT,
    "axes.titlesize": TITLE_FONT_PT,
    "axes.labelsize": BASE_FONT_PT,
    "xtick.labelsize": BASE_FONT_PT * 0.9,
    "ytick.labelsize": BASE_FONT_PT * 0.9,
    "legend.fontsize": BASE_FONT_PT * 0.85,
    "legend.frameon": False,
    "pdf.fonttype": 42,  # embed text as text, not outlines/curves
    "ps.fonttype": 42,
    "svg.fonttype": "none",  # keep text as text in SVG
    "lines.linewidth": 1.5,
    "lines.markersize": 5.0,
    "mathtext.default": "regular",
})

# ---------------------------------------------------------------------------
# Canonical four-class color/marker/label mapping (used identically across
# Figs 1, 2, 3, 4, and any SI figure referencing these classes).
# ---------------------------------------------------------------------------
CLASS_ORDER = ["ceramic", "weakT", "peak", "DBTT"]
CLASS_COLOR = {
    "ceramic": "#1b7837",   # green
    "weakT":   "#2166ac",   # blue
    "peak":    "#d6604d",   # red/orange
    "DBTT":    "#762a83",   # purple
}
CLASS_MARKER = {"ceramic": "o", "weakT": "s", "peak": "^", "DBTT": "D"}
CLASS_LABEL = {
    "ceramic": "Ceramic-like",
    "weakT": "Weakly T-dependent",
    "peak": "Peak / intermediate-T maximum",
    "DBTT": "DBTT-like",
}

# ---------------------------------------------------------------------------
# Six-system fatigue-atlas color/marker/label mapping and internal-ID map.
# Internal case IDs are recorded here (and in source-data CSVs) but MUST NOT
# be used as the visible plot label -- use SYSTEM_LABEL instead.
# ---------------------------------------------------------------------------
SYSTEM_ORDER = [
    "FCC_like_case29", "steep_cleavage_case35", "shifted_ductile_case64",
    "plastic_shielded_case64_M1", "slow_threshold_case101", "higher_barrier_case171",
]
SYSTEM_LABEL = {
    "FCC_like_case29": "Smooth ductile (FCC-like)",
    "steep_cleavage_case35": "Steep cleavage",
    "shifted_ductile_case64": "Shifted ductile",
    "plastic_shielded_case64_M1": "Plastic-shielded",
    "slow_threshold_case101": "Slow threshold",
    "higher_barrier_case171": "Higher barrier",
}
SYSTEM_COLOR = {
    "FCC_like_case29": "#4477AA", "steep_cleavage_case35": "#EE6677",
    "shifted_ductile_case64": "#228833", "plastic_shielded_case64_M1": "#CCBB44",
    "slow_threshold_case101": "#66CCEE", "higher_barrier_case171": "#AA3377",
}
SYSTEM_MARKER = {
    "FCC_like_case29": "o", "steep_cleavage_case35": "^", "shifted_ductile_case64": "s",
    "plastic_shielded_case64_M1": "D", "slow_threshold_case101": "v", "higher_barrier_case171": "P",
}
MAIN_TEXT_FOUR_SYSTEMS = ["FCC_like_case29", "steep_cleavage_case35",
                          "shifted_ductile_case64", "plastic_shielded_case64_M1"]

# ---------------------------------------------------------------------------
# Fidelity / provenance line & marker semantics (distinguish by shape, not
# color alone, so the figures remain interpretable in grayscale).
# ---------------------------------------------------------------------------
FIDELITY_STYLE = {
    "V1 analytical":  dict(linestyle="-",  marker=None, label="V1 analytical prediction"),
    "PF sharp-front": dict(linestyle="none", marker="o", markerfacecolor="none", label="PF/sharp-front (physical simulation)"),
    "FEM/CZM":        dict(linestyle="none", marker="s", label="FEM/CZM (physical simulation)"),
    "fitted":         dict(linestyle="--", marker=None, label="Fitted curve"),
    "interpolation":  dict(linestyle=":",  marker=None, label="Interpolation"),
}
OPEN_MARKER_KWARGS = dict(markerfacecolor="none")  # incomplete/first-passage-only runs
CENSOR_MARKER_KWARGS = dict(marker="v", markerfacecolor="white", markeredgewidth=1.5)  # right-censored
EXCLUDED_MARKER_KWARGS = dict(marker="x", color="0.5")  # numerically excluded
INVALIDATED_STYLE = dict(linestyle=(0, (1, 1)), color="0.55", alpha=0.8)  # invalidated result (dotted, gray)

PANEL_LABEL_KW = dict(fontsize=PANEL_LABEL_FONT_PT, fontweight="bold", va="top", ha="left")


def add_panel_label(ax, label: str, x: float = -0.16, y: float = 1.05):
    ax.text(x, y, label, transform=ax.transAxes, **PANEL_LABEL_KW)


def style_log_axis(ax, which: str = "both"):
    """Apply log-scale with minor ticks to the given axis/axes ('x','y','both')."""
    if which in ("x", "both"):
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs="all", numticks=20))
    if which in ("y", "both"):
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs="all", numticks=20))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def savefig_all(fig, stem: str, pdf_dir: Path, svg_dir: Path, png_dir: Path) -> dict:
    """Save fig as .pdf/.svg/.png(600dpi) into the three given directories;
    return {"pdf": sha256, "svg": sha256, "png": sha256, "pdf_path":..., ...}."""
    out = {}
    for ext, d in (("pdf", pdf_dir), ("svg", svg_dir), ("png", png_dir)):
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{stem}.{ext}"
        try:
            if ext == "png":
                fig.savefig(p, dpi=600, bbox_inches="tight")
            else:
                fig.savefig(p, bbox_inches="tight")
        except RuntimeError:
            # Rare Agg glyph-metrics failure during tight-bbox computation (font-cache
            # related, not a content error) -- fall back to a fixed-layout save so the
            # figure is never silently dropped.
            if ext == "png":
                fig.savefig(p, dpi=600)
            else:
                fig.savefig(p)
        out[ext] = _sha256_file(p)
        out[f"{ext}_path"] = str(p)
    return out


def sha256_of(path: Path) -> str:
    return _sha256_file(Path(path))
