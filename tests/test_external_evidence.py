import json

from arrhenius_fracture.external_evidence import fingerprint_path, load_and_verify


def test_directory_digest_is_order_independent_and_detects_content_change(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "b.txt").write_text("beta")
    (root / "a.txt").write_text("alpha")
    first = fingerprint_path(root)
    assert first["file_count"] == 2
    assert first == fingerprint_path(root)
    (root / "a.txt").write_text("changed")
    assert fingerprint_path(root)["sha256"] != first["sha256"]


def test_missing_mandatory_evidence_is_incomplete_not_skipped(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [{
        "evidence_id": "required", "path": str(tmp_path / "missing"),
        "mandatory": True, "fingerprint": {},
    }]}))
    result = load_and_verify(manifest)
    assert result["status"] == "INCOMPLETE"
    assert result["mandatory_incomplete_count"] == 1
    assert result["items"][0]["status"] == "INCOMPLETE"


def test_optional_absence_is_reported_explicitly(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [{
        "evidence_id": "optional", "path": str(tmp_path / "missing"),
        "mandatory": False, "fingerprint": {},
    }]}))
    result = load_and_verify(manifest)
    assert result["status"] == "PASS"
    assert result["items"][0]["status"] == "OPTIONAL_ABSENT"


def test_relative_paths_resolve_from_manifest_directory(tmp_path):
    evidence = tmp_path / "raw.bin"
    evidence.write_bytes(b"native")
    expected = fingerprint_path(evidence)
    folder = tmp_path / "release"
    folder.mkdir()
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps({"items": [{
        "evidence_id": "relative", "path": "../raw.bin", "mandatory": True,
        "fingerprint": expected,
    }]}))
    assert load_and_verify(manifest)["status"] == "PASS"
