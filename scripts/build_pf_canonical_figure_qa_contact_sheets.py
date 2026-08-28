#!/usr/bin/env python3
"""Build compact contact sheets for visual QA of every generated PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("analysis_outputs/pf_canonical_full_trajectory_and_mechanism_audit"),
    )
    parser.add_argument("--mark-passed", action="store_true")
    args = parser.parse_args()
    images = sorted((args.output / "figures").rglob("*.png"))
    if len(images) != 91:
        raise ValueError(f"expected 91 PNG figures, found {len(images)}")
    qa = args.output / "figure_visual_QA"
    qa.mkdir(parents=True, exist_ok=True)
    width, height = 420, 320
    columns, rows = 4, 5
    per_sheet = columns * rows
    records = []
    for sheet_index, start in enumerate(range(0, len(images), per_sheet), start=1):
        selected = images[start:start + per_sheet]
        sheet = Image.new("RGB", (columns * width, rows * height), "white")
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(selected):
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((width - 16, height - 54), Image.Resampling.LANCZOS)
            x = (index % columns) * width + (width - image.width) // 2
            y = (index // columns) * height + 4
            sheet.paste(image, (x, y))
            label = path.stem
            if len(label) > 58:
                label = label[:55] + "..."
            draw.text(((index % columns) * width + 8,
                       (index // columns) * height + height - 42),
                      label, fill="black", font=ImageFont.load_default())
        out = qa / f"PF_CANONICAL_FIGURE_QA_CONTACT_SHEET_{sheet_index:02d}.png"
        sheet.save(out, dpi=(150, 150), optimize=True)
        records.append({"sheet": str(out), "figure_count": len(selected),
                        "first": str(selected[0]), "last": str(selected[-1])})
    manifest = {
        "schema": "pf_canonical_figure_visual_qa_v1",
        "source_png_count": len(images),
        "contact_sheet_count": len(records),
        "all_source_images_nonempty": all(path.stat().st_size > 0 for path in images),
        "records": records,
        "visual_review_status": (
            "PASSED_ALL_91_FIGURES_INSPECTED_VIA_CONTACT_SHEETS"
            if args.mark_passed else "PENDING_HUMAN_OR_AGENT_INSPECTION"
        ),
    }
    (qa / "pf_canonical_figure_visual_qa_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(records)} contact sheets for {len(images)} figures")


if __name__ == "__main__":
    main()
