from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from arrhenius_fracture import v2_named_parameterizations as registry
from arrhenius_fracture.canonical_v2_registry_v10230 import surface_adapters


ROOT = Path(__file__).resolve().parents[1]
PARENT_FILES = {
    "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv": "4ba723c80abcfdd7101cee0afaa9b4104eebf2a8e847892528128664829c7760",
    "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_selection.json": "3267ac05595cee0800a81a6300e80cb0e71b216067c35cc7a50d82ac4c70e4a2",
    "arrhenius_fracture/sharp_front_v10_2_27.py": "ce1663e68be0695fb6616e886af8a1135ec6f774ff612956972ec9e813bae95a",
    "arrhenius_fracture/parameter_registry_v9111.py": "6946759c5786344a7111b9e25ce8b2513e5335d78fc590ba860fc730bb7d83a7",
    "arrhenius_fracture/persistent_site_high_cycle_checkpoint_v10230.py": "0af1a7964012047ade46e822906669f7778b699b36b68b8738cbe8d0c00c0d05",
}
OLD_ROW_HASHES = {
    "peak": "83245aa3a01450f08d13820f6a042d1c01518748fbf790b6082879a9ed6fdeb1",
    "DBTT": "6d2d454e3c79e2171b8547c895a0ee2d42fa4897af90354efe906e7b27d044d4",
    "weakT": "cfc86642687cac968223e4d3ac202c9ff14648c37982d3f188326f0ff25da1e5",
    "ceramic": "d6abae369475fa80722bc3232603815a73f5ebc19433339dd73a2a84c0a1c3db",
}
ALIASES = {
    "DBTT_V2": ("P25_TJBSV2_S_002987", "8a16415705c47c7708ec35385cde6c9a53bb6b9de6edf98b902a1f40626a9f72"),
    "DBTT_V2_P40": ("P40_TJBSV2_S_038503", "be0343fd982eda89f7f7240bf599298a0e07e06eea87570ee34fa9935dabcbfa"),
    "weakT_V2_P25": ("P25_TJBSV2_S_026704", "85f8d9d2ddbe1e7c7c0c34589883b886023507089d0fe423c1ce2c4eb60d0041"),
    "weakT_V2_P40": ("P40_TJBSV2_S_038278", "98908d7eecb5916a86dc886900b6e4fd687f4d67daa2bd77e4d75ebaef06b3d6"),
    "ceramic_V2_P25": ("P25_TJBSV2_S_004127", "c794af3a2912fa91d587d06ae9177ba87af1da6d30e4ad7310d5f673edf741e3"),
    "ceramic_V2_P40": ("P40_TJBSV2_S_031027", "d76e6a011ba83f762b5830c03c75571c851936677d213cfa1fb600a00a286cfa"),
}


def canonical_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_every_established_registry_and_checkpoint_source_is_parent_byte_identical():
    for relative, expected in PARENT_FILES.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_every_established_row_and_default_order_are_parent_identical():
    path = ROOT / "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["material_class"] for row in rows] == ["peak", "DBTT", "weakT", "ceramic"]
    assert {row["material_class"]: canonical_hash(row) for row in rows} == OLD_ROW_HASHES
    selection = json.loads((ROOT / "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_selection.json").read_text())
    assert selection["canonical_option_order"] == [row["option_key"] for row in rows]


def test_established_lookup_and_checkpoint_identifiers_resolve_as_before():
    from arrhenius_fracture.run_state_checkpoint_v10230 import validate_compatibility
    from arrhenius_fracture.sharp_front_v10_2_27 import VALID_OPTIONS

    assert VALID_OPTIONS == {
        "v913_paper_peak01_0242980_persistent_sites": "v913_zeroD_sobol_0242980",
        "v913_paper_dbtt01_0202500_persistent_sites": "v913_zeroD_sobol_0202500",
        "v913_paper_weakT01_0129902_persistent_sites": "v913_zeroD_sobol_0129902",
        "v913_paper_ceramic01_0077080_persistent_sites": "v913_zeroD_sobol_0077080",
    }
    for parameter_option in ("Peak", "DBTT", "weakT", "ceramic"):
        case = {"parameter_option": parameter_option, "seed": 1720}
        validate_compatibility({"case": case}, dict(case))


def test_overlay_does_not_intercept_established_names_or_add_legacy_aliases():
    for name in ("Peak", "DBTT", "weakT", "ceramic", "Peak_Legacy", "DBTT_Legacy", "weakT_Legacy", "ceramic_Legacy"):
        with pytest.raises(KeyError, match="unknown opt-in V2 parameterization"):
            registry.load(name)


def test_six_aliases_and_exact_ids_resolve_to_same_immutable_rows_and_hashes():
    for alias, (candidate, expected_hash) in ALIASES.items():
        by_alias = registry.load(alias)
        by_id = registry.load(candidate)
        assert by_alias is by_id is registry.load_exact_candidate(candidate)
        assert by_alias.complete_bound_row_sha256 == expected_hash
        assert by_alias.full_precision_row["candidate_id"] == candidate
        with pytest.raises(TypeError):
            by_alias.full_precision_row["candidate_id"] = "changed"


def test_exact_controls_have_no_family_alias_and_remain_exact_id_callable():
    for candidate in registry.EXACT_ONLY_CONTROLS:
        record = registry.load(candidate)
        assert record.registry_alias is None
        assert record.status == "ARCHIVAL_RESEARCH_CANDIDATE"


def test_reserved_and_unknown_names_fail_closed():
    for alias in (*registry.RESERVED_ABSENT_ALIASES, "unknown"):
        with pytest.raises(KeyError, match="unknown opt-in V2 parameterization"):
            registry.load(alias)


def test_all_cross_code_loaders_preserve_row_identity_and_surface_values():
    for alias in ALIASES:
        record = registry.load(alias)
        assert all(registry.load_for(alias, loader) is record for loader in registry.SUPPORTED_LOADERS)
        opening, emission = surface_adapters(dict(record.full_precision_row))
        for adapter in (opening, emission):
            assert adapter.parity([0.0, 1e9, 5e9], 300.0)
            assert adapter.parity([0.0, 1e9, 5e9], 900.0)


def test_checkpoint_round_trip_and_hash_mismatch_fail_closed():
    for alias in ALIASES:
        record = registry.load(alias)
        assert registry.load_checkpoint_reference(record.checkpoint_identity()) is record
        bad = record.checkpoint_identity()
        bad["complete_bound_row_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="hash mismatch"):
            registry.load_checkpoint_reference(bad)


def test_incomplete_v2_source_and_unsupported_loader_fail_closed(monkeypatch):
    record = registry.load("DBTT_V2")
    incomplete = dict(record.full_precision_row)
    incomplete.pop("physics__cleavage_hits")
    registry.clear_cache_for_testing()
    monkeypatch.setattr(registry, "load_rows", lambda: {record.source_candidate_id: incomplete})
    monkeypatch.setattr(registry, "load_row", lambda candidate: dict(incomplete))
    with pytest.raises(ValueError, match="lacks complete corrected V2 fields"):
        registry.load("DBTT_V2")
    registry.clear_cache_for_testing()
    with pytest.raises(ValueError, match="unsupported V2 parameterization loader"):
        registry.load_for("DBTT_V2", "generic_default")
