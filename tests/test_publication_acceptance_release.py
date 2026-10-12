import csv
import json
import re
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE = ROOT / "review_packet" / "publication_acceptance_0c72ca2b"
RELEASE = ROOT / "fracture_final_current"


def test_paper_families_bind_to_persistent_source_operator():
    payload = json.loads((ACCEPTANCE / "METHOD_TO_RESULT_MAP.json").read_text())
    records = payload["families"]
    assert records
    persistent = [row for row in records if row["family"] != "cavity_current_source"]
    assert {row["source_lifecycle"] for row in persistent} == {"persistent_reusable"}
    assert all(row["lifecycle_attribute"] == "install_persistent_site_source" for row in persistent)
    cavity = [row for row in records if row["family"] == "cavity_current_source"]
    assert {row["source_lifecycle"] for row in cavity} == {"current_source"}
    assert payload["bulk_and_cavity"]["cavity"].startswith("separate current-source")
    assert payload["impact"]["rerun_required"] is False


def test_common_load_and_seed_estimator_are_numerically_qualified():
    comparison = json.loads((ACCEPTANCE / "PF_FEM_PRODUCTION_COMMON_LOAD.json").read_text())
    assert comparison["common_mesh_elastic_control"]["elastic_common_problem"]["passed"] is True
    assert comparison["common_mesh_elastic_control"]["elastic_common_problem"]["maximum_relative_difference"] == 0.0
    rows = list(csv.DictReader((ACCEPTANCE / "PF_FEM_COMMON_LOAD.csv").open()))
    assert len(rows) == 10
    ratios = [float(row["reaction_ratio_PF_over_FEM"]) for row in rows]
    assert max(ratios) - min(ratios) < 1e-5
    assert 1.09 < ratios[0] < 1.10
    estimator = json.loads((ACCEPTANCE / "FIGURE_3_ESTIMATOR_CHECK.json").read_text())
    assert estimator["estimator"].startswith("paired enabled-minus-opening-only gap")
    assert {group["n"] for group in estimator["groups"].values()} == {8}


def test_reader_manuscript_excludes_stale_source_and_draft_chronology():
    text = (RELEASE / "MANUSCRIPT.md").read_text()
    forbidden = [
        r"uses a finite depleted inventory",
        r"advance-refresh",
        r"supersede the historical draft",
        r"shielding factor[^.]*=",
    ]
    for pattern in forbidden:
        assert not re.search(pattern, text, flags=re.IGNORECASE)


def test_docx_contains_structured_math_and_no_literal_tex_commands():
    with ZipFile(RELEASE / "manuscript_and_SI.docx") as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    assert "<m:f>" in xml
    assert "<m:rad>" in xml
    assert "<m:sSub>" in xml
    assert "<m:nary>" in xml
    assert not re.search(r"\\(?:frac|sqrt|int|sum|mathrm|mathcal)", xml)


def test_portable_archive_contains_reader_and_reproduction_contracts():
    with ZipFile(RELEASE / "reproducibility.zip") as archive:
        names = set(archive.namelist())
    prefix = "fracture_final_current/"
    for member in (
        "MANUSCRIPT.md",
        "SUPPORTING_INFORMATION.md",
        "manuscript_and_SI.docx",
        "reproducibility/SOURCE_AND_INPUT_MANIFEST.json",
        "reproducibility/DOCX_VALIDATION.json",
        "reproducibility/requirements-docx.txt",
        "methods/model_and_observer_contracts/METHOD_TO_RESULT_MAP.json",
    ):
        assert prefix + member in names
