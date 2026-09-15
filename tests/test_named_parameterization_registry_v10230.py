import csv
import hashlib
import json
from pathlib import Path

import pytest

from arrhenius_fracture import named_parameterization_registry as registry


ROOT = Path(__file__).resolve().parents[1]
V2_ALIASES = {
    "DBTT_V2": ("P25_TJBSV2_S_002987", "8a16415705c47c7708ec35385cde6c9a53bb6b9de6edf98b902a1f40626a9f72"),
    "DBTT_V2_P40": ("P40_TJBSV2_S_038503", "be0343fd982eda89f7f7240bf599298a0e07e06eea87570ee34fa9935dabcbfa"),
    "weakT_V2_P25": ("P25_TJBSV2_S_026704", "85f8d9d2ddbe1e7c7c0c34589883b886023507089d0fe423c1ce2c4eb60d0041"),
    "weakT_V2_P40": ("P40_TJBSV2_S_038278", "98908d7eecb5916a86dc886900b6e4fd687f4d67daa2bd77e4d75ebaef06b3d6"),
    "ceramic_V2_P25": ("P25_TJBSV2_S_004127", "c794af3a2912fa91d587d06ae9177ba87af1da6d30e4ad7310d5f673edf741e3"),
    "ceramic_V2_P40": ("P40_TJBSV2_S_031027", "d76e6a011ba83f762b5830c03c75571c851936677d213cfa1fb600a00a286cfa"),
}


def test_every_new_alias_and_exact_candidate_resolve_to_one_immutable_record():
    for alias, (candidate, expected_hash) in V2_ALIASES.items():
        by_alias = registry.load(alias)
        by_exact = registry.load_exact_candidate(candidate)
        assert by_alias is by_exact
        assert by_alias.complete_bound_row_sha256 == expected_hash
        assert by_alias.full_precision_row["candidate_id"] == candidate
        with pytest.raises(TypeError):
            by_alias.full_precision_row["candidate_id"] = "changed"


def test_every_cross_code_loader_returns_the_same_exact_record_and_surface_contract():
    from arrhenius_fracture.canonical_v2_registry_v10230 import surface_adapters
    for alias in V2_ALIASES:
        record = registry.load(alias)
        for loader in registry.SUPPORTED_LOADERS:
            assert registry.load_for(alias, loader) is record
        dedicated = (
            registry.load_analytical, registry.load_monotonic_reduced_1d,
            registry.load_fatigue_reduced_1d, registry.load_pf_sharp_front,
            registry.load_fem_czm, registry.load_checkpoint_serialization,
        )
        assert all(loader(alias) is record for loader in dedicated)
        for adapter in surface_adapters(dict(record.full_precision_row)):
            assert adapter.parity([0.0, 1e9, 5e9], 300.0)
            assert adapter.parity([0.0, 1e9, 5e9], 900.0)


def test_serialization_round_trip_preserves_every_row_string_and_identity():
    for alias in (*registry.DEFAULT_FOUR_ALIASES, *V2_ALIASES):
        record = registry.load(alias)
        restored = json.loads(json.dumps(dict(record.full_precision_row), sort_keys=True, separators=(",", ":")))
        assert restored == dict(record.full_precision_row)
        assert registry.load_checkpoint_reference(record.checkpoint_identity()) is record


def test_legacy_aliases_and_default_four_are_unchanged():
    explicit = ("Peak_Legacy", "DBTT_Legacy", "weakT_Legacy", "ceramic_Legacy")
    for old, named in zip(registry.DEFAULT_FOUR_ALIASES, explicit):
        assert registry.load(old) is registry.load(named)
        assert registry.load(old).generation == "LEGACY"
    defaults = registry.default_four_classes()
    assert tuple(row.registry_alias for row in defaults) == registry.DEFAULT_FOUR_ALIASES
    assert all(row.canonical_default for row in defaults)
    assert not any(row.generation.endswith("V2") for row in defaults)
    source = ROOT / "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "4ba723c80abcfdd7101cee0afaa9b4104eebf2a8e847892528128664829c7760"
    selection = ROOT / "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_selection.json"
    assert hashlib.sha256(selection.read_bytes()).hexdigest() == "3267ac05595cee0800a81a6300e80cb0e71b216067c35cc7a50d82ac4c70e4a2"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [dict(row.full_precision_row) for row in defaults] == rows


def test_reserved_and_unknown_aliases_fail_closed():
    for alias in ("Peak_V2", "weakT_V2", "ceramic_V2", "unknown"):
        with pytest.raises(KeyError, match="unknown named parameterization alias"):
            registry.load(alias)
    with pytest.raises(KeyError, match="unknown exact parameterization candidate"):
        registry.load_exact_candidate("missing")
    with pytest.raises(ValueError, match="unsupported parameterization loader"):
        registry.load_for("DBTT_V2", "generic_default_loader")


def test_missing_fields_hash_mismatch_and_status_incompatibility_fail_closed():
    payload = json.loads(registry.CATALOG_PATH.read_text())
    entry = next(item for item in payload["entries"] if item["registry_alias"] == "DBTT_V2")
    incomplete = dict(entry)
    incomplete.pop("renewal_fingerprint")
    with pytest.raises(ValueError, match="lacks required fields"):
        registry.validate_entry(incomplete, entry["full_precision_row"])
    bad_hash = dict(entry)
    bad_hash["complete_bound_row_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        registry.validate_entry(bad_hash, entry["full_precision_row"])
    incomplete_source = dict(entry["full_precision_row"])
    incomplete_source.pop("physics__cleavage_hits")
    incomplete_entry = dict(entry)
    incomplete_entry["full_precision_row"] = incomplete_source
    with pytest.raises(ValueError, match="lacks complete source fields"):
        registry.validate_entry(incomplete_entry, incomplete_source)
    with pytest.raises(ValueError, match="not required status"):
        registry.load("DBTT_V2", required_status="LEGACY_CANONICAL_PARAMETERIZATION")


def test_old_legacy_checkpoint_identities_remain_valid():
    for record in registry.default_four_classes():
        assert registry.load_checkpoint_reference({"parameter_option": record.full_precision_row["option_key"]}) is record
        assert registry.load_checkpoint_reference({"candidate_id": record.source_candidate_id}) is record
    assert registry.load_checkpoint_reference({"material_class": "weakT"}) is registry.load("weakT")
    with pytest.raises(KeyError, match="no recognized parameterization identity"):
        registry.load_checkpoint_reference({"parameter_option": "missing"})
