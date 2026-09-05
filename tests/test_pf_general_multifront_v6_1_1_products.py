import hashlib
import io
import json
from pathlib import Path
import zipfile

from scripts.build_pf_general_multifront_qualification_v6_1_1 import (
    ARCHIVE_NAME, MANIFEST_NAME, V6_1_COMMIT, verify_archive_bytes,
)


ROOT = Path(__file__).parents[1]
OUT = ROOT / "analysis_outputs/pf_current_source_general_multifront_v6_1_1"


def load(name):
    return json.loads((OUT / name).read_text())


def test_actual_v6_1_commit_tree_ancestry_patches_and_all_snapshots_are_bound():
    record = load("pf_general_multifront_v6_1_1_source_provenance.json")
    assert record["v6_1_implementation_commit"] == V6_1_COMMIT
    assert record["v6_1_implementation_tree"] == "ab60dce9a259ca736801444c00103c16a4ba2be5"
    assert record["v6_parent_commit"] == "d993dd933f52281b89cf22e3aa9cf0db61d64487"
    assert record["exact_ancestry_path"] == [
        "ae9a06d8c42287428e917baef143b0c1142cefd8",
        "d993dd933f52281b89cf22e3aa9cf0db61d64487",
        V6_1_COMMIT,
    ]
    assert len(record["v6_1_source_snapshots"]) == 17
    assert record["every_snapshot_matches_git_blob_bytes"]
    assert record["pre_v12_execution_physics_unchanged"]
    assert record["v12_execution_source_added_or_modified"]
    assert "physics_source_modified" not in record
    for row in record["v6_1_source_snapshots"].values():
        path = ROOT / row["snapshot_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert path.stat().st_size == row["size_bytes"]
        assert row["snapshot_matches_git_blob_bytes"]
    for row in record["patches"].values():
        path = ROOT / row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_dynamic_provider_source_audit_passes_without_n_greater_than_two_claim():
    record = load("pf_general_multifront_dynamic_provider_source_audit_v6_1_1.json")
    assert record["qualification"] == "PASS"
    assert all(record["gates"].values())
    assert not any(record["fixed_cardinality_pattern_matches"].values())
    assert record["front_resource_limit_none_source_cap"] is None
    assert record["production_pf_max_fronts_greater_than_two"] == "NOT_YET_EXECUTED"


def test_prefix_and_full_terminal_parity_are_separate_and_pass():
    prefix = load("pf_general_multifront_v5_2_prefix_parity_v6_1_1.json")
    terminal = load("pf_general_multifront_v5_4_2_full_terminal_parity_v6_1_1.json")
    assert prefix["qualification"] == terminal["qualification"] == "PASS"
    assert {row["accepted_event_count"] for row in prefix["cases"].values()} == {84}
    assert terminal["cases"]["control"]["accepted_event_count"] == 85
    assert terminal["cases"]["enabled"]["accepted_event_count"] == 86
    assert terminal["cases"]["enabled"]["terminal_step"] == 813


def _rewrite_archive(*, remove=None, modify=None):
    source = OUT / ARCHIVE_NAME
    target = io.BytesIO()
    with zipfile.ZipFile(source) as incoming, zipfile.ZipFile(
        target, "w", compression=zipfile.ZIP_DEFLATED,
    ) as outgoing:
        for name in incoming.namelist():
            if name == remove:
                continue
            data = incoming.read(name)
            if name == modify:
                data = data + b"tampered"
            outgoing.writestr(name, data)
    return target.getvalue()


def test_exact_compact_archive_positive_removed_and_modified_verification():
    archive = OUT / ARCHIVE_NAME
    ok, reason = verify_archive_bytes(archive.read_bytes())
    assert ok and reason == "exact_member_set_size_and_sha256"
    manifest = load(MANIFEST_NAME)
    member = manifest["members"][0]["path"]
    ok, reason = verify_archive_bytes(_rewrite_archive(remove=member))
    assert not ok and reason == "archive_member_set_mismatch"
    ok, reason = verify_archive_bytes(_rewrite_archive(modify=member))
    assert not ok and reason.startswith("member_")


def test_report_preserves_permanent_execution_boundary():
    report = (OUT / "PF_CURRENT_SOURCE_GENERAL_MULTIFRONT_V6_1_1.md").read_text()
    assert "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS" in report
    assert "step **813**, not 807" in report
    assert "Generic-driver N=1/N=2 PF reproduction remains" in report
    assert "Production PF mechanics at N>2" in report
    assert "predictive_recursive_branching_physics_validated" in report
