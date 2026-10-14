from __future__ import annotations

import json
from pathlib import Path

from arrhenius_fracture.external_evidence import EvidenceManifest, directory_fingerprint


def test_manifest_loader_verifies_and_reports_exact_restore_instruction(tmp_path: Path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "state.txt").write_text("accepted state\n")
    expected = directory_fingerprint(evidence)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [{
        "evidence_id": "state", "mandatory": True, "path": "evidence",
        "fingerprint": expected, "restore_command": "restore-state --exact",
    }]}))
    loader = EvidenceManifest(manifest)
    assert loader.audit()["decision"] == "COMPLETE"

    (evidence / "state.txt").write_text("changed\n")
    failed = loader.audit()
    assert failed["decision"] == "INCOMPLETE"
    assert failed["mandatory_incomplete"][0]["status"] == "FINGERPRINT_MISMATCH"
    assert failed["retrieval_instructions"] == ["restore-state --exact"]


def test_manifest_loader_can_select_named_subset(tmp_path: Path):
    for name in ("a", "b"):
        (tmp_path / name).write_text(name)
    items = [{
        "evidence_id": name, "mandatory": True, "path": name,
        "fingerprint": {
            "kind": "file", "bytes": (tmp_path / name).stat().st_size,
            "sha256": __import__("hashlib").sha256(name.encode()).hexdigest(),
        },
    } for name in ("a", "b")]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": items}))
    report = EvidenceManifest(manifest).audit(["b"])
    assert [row["evidence_id"] for row in report["records"]] == ["b"]
