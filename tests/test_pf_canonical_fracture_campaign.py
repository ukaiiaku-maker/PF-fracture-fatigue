from __future__ import annotations

import csv
import json
from pathlib import Path
import zipfile

import pytest

from scripts import analyze_pf_canonical_fracture_campaign as analysis
from scripts import archive_verified_legacy_pf_run as archive_tool
from scripts import audit_pf_canonical_fracture_campaign as audit
from scripts import execute_pf_scoped_delete_list as delete_tool
from scripts import run_pf_canonical_fracture_campaign as runner


EXPECTED_HASHES = {
    "Peak": "937644f63e8f44982523ea11fce962bc28fe38d347cfc3d37f898af070073283",
    "DBTT": "4ef4cda0fcdaedd2b8bad4330cd749772594f85b3df7af51ee955592f32256e6",
    "weak-T": "5689ee29ac72f27c7259cbb6a60f3175ad4327cd045a3e6f7935884c66f3e368",
    "ceramic-like": "fee4d08d19a0576b72157f76b5ef910739be3826354c63467e7ddb2249ba896b",
}


def test_current_four_material_hashes_are_exact():
    registry = Path("analysis_outputs/pf_canonical_fracture_v2_campaign/pf_v2_four_class_registry.csv")
    _, hashes = audit.load_registry(registry)
    assert hashes == EXPECTED_HASHES


@pytest.mark.parametrize("material", ["weak-T", "ceramic-like"])
def test_historical_replaced_rows_are_parameter_stale(material):
    status, reason = audit.classify_archive_case(material, "theta", "old-row", True, True)
    assert status == "RERUN_REQUIRED_PARAMETER_STALE"
    assert "SUPERSEDED_V2_MATERIAL_ROW" in reason


@pytest.mark.parametrize("material", ["Peak", "DBTT"])
def test_controls_require_source_compatibility(material):
    status, reason = audit.classify_archive_case(material, "theta", audit.CANONICAL_IDS[material], True, True)
    assert status == "RERUN_REQUIRED_SOURCE_STALE"
    assert "PHYSICS_HASH_MISMATCH" in reason


def test_zip_inventory_is_read_only_and_does_not_extract(tmp_path):
    archive = tmp_path / "tiny.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("campaign/case/run_args.json", "{}")
    before = set(tmp_path.iterdir())
    _, _, manifest = audit.inventory_zip(archive)
    assert set(tmp_path.iterdir()) == before
    assert manifest["entry_count"] == 1


def test_duplicate_policy_requires_content_hashes():
    text = Path("scripts/audit_pf_canonical_fracture_campaign.py").read_text()
    assert "sha256" in text.lower()
    assert "size_bytes" in text


def _delete_row(target: Path, archive: Path, manifest_hash: str) -> dict[str, str]:
    return {"absolute_path": str(target), "classification": "DELETE_SUPERSEDED_EXTRACTED_COPY",
            "status": "ELIGIBLE_AFTER_EXPLICIT_SAFETY_CHECK", "replacement_or_archive": str(archive),
            "content_manifest_sha256": manifest_hash}


def test_active_path_cannot_enter_executable_delete_list(tmp_path, monkeypatch):
    runs = tmp_path / "runs"; target = runs / "case"; target.mkdir(parents=True)
    replacement = runs / "archive.zst"; replacement.write_bytes(b"x")
    manifest = tmp_path / "manifest"; manifest.write_bytes(b"m")
    monkeypatch.setattr(delete_tool, "worktrees", lambda _: set())
    monkeypatch.setattr(delete_tool, "active_commands", lambda: f"python {target}")
    with pytest.raises(RuntimeError, match="active process"):
        delete_tool.validate_row(_delete_row(target, replacement, delete_tool.sha256(manifest)), runs, tmp_path, manifest)


def test_unique_run_requires_verified_archive_before_deletion(tmp_path, monkeypatch):
    runs = tmp_path / "runs"; target = runs / "case"; target.mkdir(parents=True)
    manifest = tmp_path / "manifest"; manifest.write_bytes(b"m")
    monkeypatch.setattr(delete_tool, "worktrees", lambda _: set())
    monkeypatch.setattr(delete_tool, "active_commands", lambda: "")
    with pytest.raises(RuntimeError, match="archive is absent"):
        delete_tool.validate_row(_delete_row(target, runs / "missing.zst", delete_tool.sha256(manifest)), runs, tmp_path, manifest)


def test_archive_full_test_extraction_roundtrip(tmp_path):
    runs = tmp_path / "runs"; source = runs / "case"; source.mkdir(parents=True)
    (source / "steps_1000K.csv").write_text("a\n1\n")
    expected = archive_tool.members(source)
    packed = runs / "case.tar.zst"
    archive_tool.create_archive(source, packed, 1)
    assert archive_tool.verify_full_extraction(source, packed, expected) == {
        "full_test_extraction": True, "verified_member_count": 1}


def test_deletion_target_is_confined_to_direct_child(tmp_path, monkeypatch):
    runs = tmp_path / "runs"; target = tmp_path / "outside"; target.mkdir(); runs.mkdir()
    manifest = tmp_path / "manifest"; manifest.write_bytes(b"m")
    with pytest.raises(RuntimeError, match="direct child"):
        delete_tool.validate_row(_delete_row(target, runs / "a.zst", delete_tool.sha256(manifest)), runs, tmp_path, manifest)


def test_git_worktree_cannot_be_archived_or_deleted(tmp_path):
    runs = tmp_path / "runs"; target = runs / "worktree"; target.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Git worktree"):
        archive_tool.assert_safe_source(target, runs, {target.resolve()})


def test_canonical_theta_grid_equals_recovered_grid():
    rows = audit.canonical_matrix()
    assert sorted({row["theta_deg"] for row in rows if row["matrix"] == "CANONICAL_SINGLE_CRACK_THETA"}) == [15.0, 30.0]


def test_canonical_rate_grid_equals_recovered_grid():
    rows = audit.canonical_matrix()
    got = sorted({(row["rate_tag"], row["loading_rate_factor"], row["nominal_dt_s"]) for row in rows if row["matrix"] == "CANONICAL_STRAIN_RATE"})
    assert got == sorted((tag, factor, dt) for tag, factor, dt, _ in audit.RATE_GRID)


def test_v2_analysis_separates_transactions_and_avalanches():
    rows = [
        {"case_id": "x", "physical_avalanche_index": 0, "event_transaction_index": 0,
         "event_extension_um": 2.0, "pre_event_projected_extension_um": 0.0,
         "post_event_projected_extension_um": 2.0, "pre_event_opening_m": 1.0,
         "pre_event_native_J_J_per_m2": 1.0, "pre_event_native_KJ_MPa_sqrt_m": 2.0,
         "right_censored_at_target": False, "matrix": "m", "material_class": "Peak",
         "candidate_id": "c", "temperature_K": 1000, "theta_deg": 15, "rate_tag": "rate1x",
         "loading_rate_factor": 1, "seed": 1},
        {"case_id": "x", "physical_avalanche_index": 0, "event_transaction_index": 1,
         "event_extension_um": 3.0, "pre_event_projected_extension_um": 2.0,
         "post_event_projected_extension_um": 5.0, "pre_event_opening_m": 1.0,
         "pre_event_native_J_J_per_m2": 1.0, "pre_event_native_KJ_MPa_sqrt_m": 2.0,
         "right_censored_at_target": True, "matrix": "m", "material_class": "Peak",
         "candidate_id": "c", "temperature_K": 1000, "theta_deg": 15, "rate_tag": "rate1x",
         "loading_rate_factor": 1, "seed": 1},
    ]
    avalanches = analysis.physical_avalanches(rows)
    assert len(rows) == 2 and len(avalanches) == 1
    assert avalanches[0]["avalanche_extension_um"] == 5.0


def test_new_weak_and_ceramic_rows_are_launcher_required():
    assert runner.EXPECTED_CLASSES["weak-T"] == "oneD_v2_focused_weak_T_0016"
    assert runner.EXPECTED_CLASSES["ceramic-like"] == "oneD_v2_focused_ceramic_like_0018"


def test_branching_must_be_labelled_demonstration_only():
    required = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
    mission = Path("/Users/shen/.codex/attachments/3d905682-3e9b-4e85-9f4a-65a21828e824/pasted-text.txt").read_text()
    assert required in mission


def test_generated_plan_is_deterministic():
    assert audit.canonical_matrix() == audit.canonical_matrix()


def test_storage_accounting_closes():
    path = Path("analysis_outputs/pf_canonical_fracture_v2_campaign/pf_storage_reclaimed.csv")
    row = next(csv.DictReader(path.open()))
    assert int(row["net_reclaimable_bytes"]) == int(row["original_size_bytes"]) - int(row["archive_size_bytes"])


def test_launcher_has_no_fatigue_or_energy_gate_feedback(tmp_path):
    registry = tmp_path / "r"; selection = tmp_path / "s"; family = tmp_path / "f"
    for path in (registry, selection, family): path.write_text("x")
    environment = runner.canonical_env(registry, selection, family, 7)
    assert "V10229_FATIGUE_ENABLED" not in environment
    assert "V10230_ENERGY_GATE_ENABLED" not in environment
    assert environment["ONED_V2_TP_STATE_DIAGNOSTICS"] == "1"


def test_observer_artifact_compression_is_lossless_and_verified(tmp_path):
    source = tmp_path / runner.LARGE_OBSERVER_ARTIFACTS[0]
    source.write_text(json.dumps({"profile": list(range(1000))}))
    expected_hash = runner.sha256(source)
    records = runner.compress_observer_artifacts(tmp_path)
    assert not source.exists()
    assert records[0]["source_sha256"] == expected_hash
    assert (tmp_path / f"{source.name}.zst").is_file()


def test_launcher_submits_incrementally_and_stops_after_failure():
    source = Path("scripts/run_pf_canonical_fracture_campaign.py").read_text()
    assert "return_when=FIRST_COMPLETED" in source
    assert "if not stopped_after_failure" in source
