"""Shared style module for the focused da/dN-vs-K comparison package."""
from __future__ import annotations
import hashlib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "black", "axes.linewidth": 1.25, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 1.25, "ytick.major.width": 1.25,
    "xtick.minor.width": 1.0, "ytick.minor.width": 1.0,
    "xtick.major.size": 4.5, "ytick.major.size": 4.5,
    "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 13, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "lines.linewidth": 1.5, "lines.markersize": 6.0,
})

R_MARKER = {-0.95: "o", 0.1: "s", 0.5: "^"}
R_LABEL = {-0.95: "R=-0.95", 0.1: "R=0.1", 0.5: "R=0.5"}
OPTION_COLOR = {"A_NATIVE": "black", "PT03": "tab:blue", "PT08": "tab:red"}
PROTOCOL_COLOR = {"D1": "tab:blue", "D2": "tab:orange", "D3": "tab:green",
                  "D5": "tab:red", "D6_conditional_persistent": "tab:purple", "D6": "tab:purple"}
BASELINE_STYLE = dict(color="0.5", linestyle="--", marker="o", markerfacecolor="none")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def savefig_all(fig, stem: str, pdf_dir: Path, svg_dir: Path, png_dir: Path) -> dict:
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
            if ext == "png":
                fig.savefig(p, dpi=600)
            else:
                fig.savefig(p)
        out[ext] = sha256_of(p)
        out[f"{ext}_path"] = str(p)
    return out
