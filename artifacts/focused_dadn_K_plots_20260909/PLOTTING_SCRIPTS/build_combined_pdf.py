"""Assemble the 8 individual figure files into one combined PDF,
FOCUSED_DADN_K_PLOTS.pdf, one page per figure, in mission order. No PDF-merge
library is available in this environment, so each page is built by placing
the already-rendered 600-DPI PNG onto a matplotlib page sized to match the
PNG's aspect ratio (this preserves the already-QC'd pixel content exactly;
it does not re-render or re-lay-out any figure).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

PKG = Path(__file__).resolve().parents[1]
OUT = PKG / "FOCUSED_DADN_K_PLOTS.pdf"

PAGES = [
    "01_R_DEPENDENCE/PNG_600DPI/Figure1_A_NATIVE_R_dependence.png",
    "02_TRANSPORT_PARAMETERIZATIONS/PNG_600DPI/Figure2_A_NATIVE_PT03_PT08.png",
    "02_TRANSPORT_PARAMETERIZATIONS/PNG_600DPI/Figure2_ratio_companion.png",
    "03_REBONDING_PARAMETERIZATIONS/PNG_600DPI/Figure3_rebonding_parameterizations.png",
    "03_REBONDING_PARAMETERIZATIONS/PNG_600DPI/Figure3_Sh_companion.png",
    "04_LOCAL_SLOPES/PNG_600DPI/Figure4_local_slopes.png",
    "04_LOCAL_SLOPES/PNG_600DPI/Figure4_global_slope_summary.png",
    "05_MATERIAL_CLASS_CONTEXT/PNG_600DPI/Figure5_material_class_context.png",
]


def main() -> None:
    with PdfPages(OUT) as pdf:
        for rel in PAGES:
            path = PKG / rel
            img = mpimg.imread(path)
            h, w = img.shape[0], img.shape[1]
            dpi = 600.0
            fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)
            print(f"page added: {rel}")
    print(f"\nWrote {OUT} ({len(PAGES)} pages)")


if __name__ == "__main__":
    main()
