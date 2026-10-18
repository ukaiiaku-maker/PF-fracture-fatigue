import hashlib
import json
import zipfile

import pytest

from scripts.verify_r10_12h_campaign_archive import verify


def write_archive(path, payload=b"accepted evidence", declared_digest=None):
    digest = declared_digest or hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema": "r10.12h-portable-evidence-manifest/1",
        "base_revision": "base",
        "head_revision": "head",
        "files": [{"path": "evidence/result.json", "bytes": len(payload),
                   "sha256": digest}],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("evidence/result.json", payload)
        archive.writestr("MANIFEST.json", json.dumps(manifest))


def test_verify_campaign_archive_checks_all_bytes(tmp_path):
    path = tmp_path / "evidence.zip"
    write_archive(path)
    result = verify(path)
    assert result["status"] == "PASS"
    assert result["verified_files"] == 1
    assert result["verified_uncompressed_bytes"] == len(b"accepted evidence")


def test_verify_campaign_archive_rejects_hash_mismatch(tmp_path):
    path = tmp_path / "evidence.zip"
    write_archive(path, declared_digest="0" * 64)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify(path)
