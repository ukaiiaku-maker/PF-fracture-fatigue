import argparse
import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DONOR = Path("/private/tmp/taylor-peierls-spatial-coupling-paper-audit/analysis_outputs/taylor_peierls_spatial_coupling_paper_audit")
FAMILY = Path("/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_2_28_kernel_cache/4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/family.json")


def module():
    spec = importlib.util.spec_from_file_location("a8pt", ROOT / "scripts/build_v10_2_30_A_native_plus_8PT_registry.py")
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj


def build(tmp_path):
    m = module()
    args = argparse.Namespace(
        A_registry=ROOT / "runs/refined_candidate_search/historical_ABCD_refined_registry_v2.csv",
        donor_root=DONOR, out=tmp_path, repo=ROOT, qualified_head=m.QUALIFIED_HEAD,
        common_physics=ROOT / "arrhenius_fracture/data/materials/v10_2_27_paper_four_class_registry.csv",
        family_json=FAMILY,
    )
    return m, m.build(args)


def test_real_composites_prove_all_preservation_contracts(tmp_path):
    if not DONOR.is_dir(): pytest.skip("audited PR-67 worktree unavailable")
    _, result = build(tmp_path)
    rows = list(csv.DictReader((tmp_path / "A_native_plus_8PT_registry.csv").open()))
    audits = list(csv.DictReader((tmp_path / "A_native_plus_8PT_parameter_diff_audit.csv").open()))
    imports = list(csv.DictReader((tmp_path / "donor_field_import_audit.csv").open()))
    assert len(rows) == 9 and rows[0]["option_key"] == "A_NATIVE"
    assert len(result["PT_SUBSTITUTION_FIELDS"]) == 10
    assert all(row["PT_copy_exact"] == "True" and row["non_PT_preserved"] == "True" for row in audits)
    assert all(json.loads(row["outside_PT_changed_fields_json"]) == [] for row in audits)
    assert all(row["imported"] == "False" or row["classification"] == "PT_TRANSFERRED_MATERIAL_COORDINATE" for row in imports)
    assert result["donor_artifacts_preserved"]
    assert result["solver_preflight"]["solver_physics_preserved"]
    assert result["solver_preflight"]["common_physics_preserved"]
    evidence = result["variants"]
    assert len({x["cleavage_hash"] for x in evidence}) == 1
    assert len({x["emission_hash"] for x in evidence}) == 1
    assert len({x["non_PT_candidate_hash"] for x in evidence}) == 1
    assert len({x["source_and_blunting_hash"] for x in evidence}) == 1
    assert all(x["PT_subset_hash"] == x["donor_PT_subset_hash"] for x in evidence)


def test_composite_verifier_rejects_non_pt_change_and_omitted_pt_copy():
    m = module(); A = {"p": "1", "q": "2", "cleave": "3"}; donor = {"p": "4", "q": "5"}
    good = {"p": "4", "q": "5", "cleave": "3"}
    assert m.verify_composite(A, good, donor, ["p", "q"], list(A))["pt_equal"]
    bad_non_pt = dict(good, cleave="9")
    with pytest.raises(ValueError, match="non-PT"): m.verify_composite(A, bad_non_pt, donor, ["p", "q"], list(A))
    omitted = dict(good, q="2")
    with pytest.raises(ValueError, match="every exact donor PT"): m.verify_composite(A, omitted, donor, ["p", "q"], list(A))


def test_solver_preflight_detects_modified_production_source(tmp_path, monkeypatch):
    m = module()
    for name in m.PHYSICS_FILES:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"qualified")
    (tmp_path / m.PHYSICS_FILES[0]).write_bytes(b"modified")
    common = tmp_path / "common"; common.write_bytes(b"common")
    family = tmp_path / "family"; family.write_bytes(b"family")
    monkeypatch.setattr(m, "git_blob", lambda *_: b"qualified")
    result = m.solver_audit(tmp_path, "head", common, family)
    assert not result["solver_physics_preserved"]
