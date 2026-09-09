"""Build figure_manifest.{csv,json}, source_data_manifest.{csv,json},
file_hashes.json, and validation_report.json for the final figure package.
Pure post-processing over the already-generated package directory; performs
no plotting and touches no scientific source table.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
QC_OUT = PKG / "07_QC_AND_PROVENANCE"
QC_OUT.mkdir(parents=True, exist_ok=True)

EVIDENCE_CLASS_BY_FIGURE = {
    "Fig1A": "SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED",
    "Fig1B": "SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED",
    "Fig1C": "SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED",
    "Fig1D": "SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED",
    "Fig2A_ceramic": "QUALIFIED_SOURCE_RESULT_VERIFIED", "Fig2B_weakT": "QUALIFIED_SOURCE_RESULT_VERIFIED",
    "Fig2C_peak": "QUALIFIED_SOURCE_RESULT_VERIFIED", "Fig2D_DBTT": "QUALIFIED_SOURCE_RESULT_VERIFIED",
    "Fig3A_ceramic": "QUALIFIED_SOURCE_RESULT_VERIFIED", "Fig3B_weakT": "QUALIFIED_SOURCE_RESULT_VERIFIED",
    "Fig3C_peak": "QUALIFIED_SOURCE_RESULT_VERIFIED", "Fig3D_DBTT": "QUALIFIED_SOURCE_RESULT_VERIFIED",
    "Fig4A_ceramic": "QUALIFIED_SOURCE_RESULT_VERIFIED", "Fig4B_weakT": "QUALIFIED_SOURCE_RESULT_VERIFIED",
    "Fig4C_peak": "QUALIFIED_SOURCE_RESULT_VERIFIED", "Fig4D_DBTT": "QUALIFIED_SOURCE_RESULT_VERIFIED",
    "Fig4E_dbtt_shift_definitions": "QUALIFIED_SOURCE_RESULT_VERIFIED",
    "Fig5A": "QUALIFIED_SOURCE_RESULT_VERIFIED",
    "Fig5B": "SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED",
    "Fig5C": "QUALIFIED_SOURCE_RESULT_VERIFIED",
    "Fig5D": "SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED",
    "Fig6A": "QUALIFIED_SOURCE_RESULT_VERIFIED", "Fig6B": "QUALIFIED_SOURCE_RESULT_VERIFIED",
    "Fig6C": "QUALIFIED_SOURCE_RESULT_VERIFIED",
    "Fig7A": "QUALIFIED_SOURCE_RESULT_VERIFIED", "Fig7B": "QUALIFIED_SOURCE_RESULT_VERIFIED",
}
for n in range(1, 10):
    EVIDENCE_CLASS_BY_FIGURE[f"SI_Fig{n}"] = "DERIVED_FROM_GOVERNED_PORTABLE_BUNDLE"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _evidence_class(stem: str) -> str:
    for key, cls in EVIDENCE_CLASS_BY_FIGURE.items():
        if stem.startswith(key):
            return cls
    if stem.startswith("Figure") and stem.endswith("_composite"):
        n = re.search(r"Figure(\d)", stem)
        if n and n.group(1) in ("1", "5"):
            return "MIXED (see individual panels)"
        return "QUALIFIED_SOURCE_RESULT_VERIFIED (see individual panels)"
    return "SEE_INDIVIDUAL_PANELS"


def build_figure_manifest():
    rows = []
    for section, subdir in [("01_MAIN_TEXT", "main text"), ("02_SUPPORTING_INFORMATION", "SI"),
                             ("03_REBONDING_PART_X", "Part X")]:
        pdf_dir = PKG / section / "PDF"
        if not pdf_dir.is_dir():
            continue
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            stem = pdf_path.stem
            svg_path = PKG / section / "SVG" / f"{stem}.svg"
            png_path = PKG / section / "PNG_600DPI" / f"{stem}.png"
            rows.append(dict(
                figure_id=stem, section=subdir,
                pdf_path=str(pdf_path.relative_to(PKG)), pdf_sha256=_sha256(pdf_path) if pdf_path.exists() else "",
                svg_path=str(svg_path.relative_to(PKG)) if svg_path.exists() else "",
                svg_sha256=_sha256(svg_path) if svg_path.exists() else "",
                png_path=str(png_path.relative_to(PKG)) if png_path.exists() else "",
                png_sha256=_sha256(png_path) if png_path.exists() else "",
                evidence_class=_evidence_class(stem) if subdir != "Part X" else "SEE_PART_X_LIMITATIONS",
            ))
    fields = list(rows[0].keys())
    with (QC_OUT / "figure_manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    (QC_OUT / "figure_manifest.json").write_text(json.dumps(dict(n_figures=len(rows), rows=rows), indent=2))

    ids = [r["figure_id"] for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    return rows, dupes


def build_source_data_manifest():
    rows = []
    for section, subdir in [("MAIN_TEXT", "main text"), ("SUPPORTING_INFORMATION", "SI"),
                             ("REBONDING_PART_X", "Part X")]:
        d = PKG / "04_SOURCE_DATA" / section
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file():
                rows.append(dict(section=subdir, path=str(p.relative_to(PKG)), sha256=_sha256(p), n_bytes=p.stat().st_size))
    fields = list(rows[0].keys())
    with (QC_OUT / "source_data_manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    (QC_OUT / "source_data_manifest.json").write_text(json.dumps(dict(n_files=len(rows), rows=rows), indent=2))

    hashes_by_content = {}
    dup_hashes = []
    for r in rows:
        hashes_by_content.setdefault(r["sha256"], []).append(r["path"])
    for h, paths in hashes_by_content.items():
        if len(paths) > 1:
            dup_hashes.append(dict(sha256=h, paths=paths))
    return rows, dup_hashes


def validate(figure_rows, dup_ids, source_rows, dup_hashes):
    checks = {}
    checks["all_figures_have_pdf"] = all(r["pdf_sha256"] for r in figure_rows)
    checks["no_duplicate_figure_ids"] = (len(dup_ids) == 0)
    checks["all_source_files_nonzero_size"] = all(r["n_bytes"] > 0 for r in source_rows)
    checks["figure_manifest_nonempty"] = len(figure_rows) > 0
    checks["source_data_manifest_nonempty"] = len(source_rows) > 0
    n_main = sum(1 for r in figure_rows if r["section"] == "main text")
    n_si = sum(1 for r in figure_rows if r["section"] == "SI")
    n_px = sum(1 for r in figure_rows if r["section"] == "Part X")
    checks["main_text_has_at_least_25_files"] = n_main >= 25  # 7 figures x (panels+composite)
    report = dict(
        schema="v1_figure_package_validation_report",
        n_main_text_figures=n_main, n_si_figures=n_si, n_part_x_figures=n_px,
        n_source_data_files=len(source_rows),
        duplicate_figure_ids=dup_ids,
        duplicate_source_data_hashes=dup_hashes,
        checks=checks,
        all_checks_pass=all(checks.values()),
    )
    (QC_OUT / "validation_report.json").write_text(json.dumps(report, indent=2))
    return report


def build_file_hashes():
    hashes = {}
    for p in sorted(PKG.rglob("*")):
        if p.is_file() and not p.name.endswith(".zip") and not p.name.endswith(".sha256"):
            hashes[str(p.relative_to(PKG))] = _sha256(p)
    (QC_OUT / "file_hashes.json").write_text(json.dumps(dict(n_files=len(hashes), files=hashes), indent=2))
    return len(hashes)


def main():
    figure_rows, dup_ids = build_figure_manifest()
    source_rows, dup_hashes = build_source_data_manifest()
    report = validate(figure_rows, dup_ids, source_rows, dup_hashes)
    n_hashed = build_file_hashes()
    print(f"figure_manifest: {len(figure_rows)} rows")
    print(f"source_data_manifest: {len(source_rows)} rows")
    print(f"validation_report.all_checks_pass = {report['all_checks_pass']}")
    print(f"file_hashes.json: {n_hashed} files")


if __name__ == "__main__":
    main()
