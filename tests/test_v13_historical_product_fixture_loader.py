"""Merge-infrastructure regressions only; no V13 mechanics or lifecycle edits."""
import hashlib
import json

import pytest

from scripts.historical_product_fixtures import (
    BUNDLED, MANIFEST_NAME, HistoricalProductUnavailable, load_product,
    verify_package,
)


OPTIONAL = (
    "tests/test_pf_general_multifront_v6_2_products.py::"
    "test_required_v6_2_products_exist_and_preserve_boundary"
)


def package(tmp_path):
    data = b'{"historical_decision": "unchanged"}\n'
    (tmp_path / "record.json").write_bytes(data)
    manifest = {"members": {"record.json": {
        "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
    }}, "tests": {}}
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest))
    return manifest


def test_v13_optional_historical_pack_absence_is_explicit():
    with pytest.raises(HistoricalProductUnavailable, match="HISTORICAL_PRODUCT_FIXTURE_UNAVAILABLE"):
        load_product(OPTIONAL, env={})


def test_v13_configured_missing_pack_is_failure_not_skip(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_product(OPTIONAL, env={
            "PF_HISTORICAL_PRODUCT_FIXTURE_ROOT": str(tmp_path / "missing"),
            "PF_HISTORICAL_PRODUCT_MANIFEST_SHA256": "0" * 64,
        })


def test_v13_external_pack_requires_independent_manifest_pin(tmp_path):
    package(tmp_path)
    with pytest.raises(ValueError, match="root AND published"):
        load_product(OPTIONAL, env={"PF_HISTORICAL_PRODUCT_FIXTURE_ROOT": str(tmp_path)})
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        load_product(OPTIONAL, env={
            "PF_HISTORICAL_PRODUCT_FIXTURE_ROOT": str(tmp_path),
            "PF_HISTORICAL_PRODUCT_MANIFEST_SHA256": "0" * 64,
        })


def test_v13_historical_member_sha_and_size_are_both_checked(tmp_path):
    manifest = package(tmp_path)
    verify_package(tmp_path, manifest)
    data = (tmp_path / "record.json").read_bytes()
    (tmp_path / "record.json").write_bytes(data.replace(b"unchanged", b"corrupted"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_package(tmp_path, manifest)
    (tmp_path / "record.json").write_bytes(data + b"tampered")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_package(tmp_path, manifest)


def test_v13_missing_manifested_member_is_not_skipped(tmp_path):
    manifest = package(tmp_path)
    (tmp_path / "record.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_package(tmp_path, manifest)


def test_v13_unmanifested_member_is_rejected(tmp_path):
    manifest = package(tmp_path)
    (tmp_path / "unexpected.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="member set mismatch"):
        verify_package(tmp_path, manifest)


def test_v13_manifest_cannot_escape_fixture_root(tmp_path):
    manifest = package(tmp_path)
    manifest["members"]["../escape.json"] = manifest["members"].pop("record.json")
    with pytest.raises(ValueError, match="escapes package"):
        verify_package(tmp_path, manifest)


def test_v13_unknown_historical_test_is_not_silently_skipped():
    with pytest.raises(KeyError):
        load_product("tests/not_an_approved_historical_test.py::test_new", env={})


def test_v13_bundled_fixture_inventory_is_exact():
    catalog = json.loads((BUNDLED / MANIFEST_NAME).read_text())
    assert len(catalog["tests"]) == 24
    assert sum(r["availability"] == "BUNDLED_COMMIT_VERIFIED"
               for r in catalog["tests"].values()) == 2
    assert len(catalog["members"]) == 3
    verify_package(BUNDLED, catalog)
