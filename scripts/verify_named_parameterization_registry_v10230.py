#!/usr/bin/env python3
"""Strict static verifier for the additive named parameterization catalog."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture import named_parameterization_registry as registry


PARENT = "c1b3f08d956242844ee1206bf10576f08f8a2859"
LEGACY_SHA = "4ba723c80abcfdd7101cee0afaa9b4104eebf2a8e847892528128664829c7760"
LEGACY_SELECTION_SHA = "3267ac05595cee0800a81a6300e80cb0e71b216067c35cc7a50d82ac4c70e4a2"
V2_ALIASES = (
    "DBTT_V2", "DBTT_V2_P40", "weakT_V2_P25", "weakT_V2_P40",
    "ceramic_V2_P25", "ceramic_V2_P40",
)


def main() -> int:
    assert subprocess.run(["git", "merge-base", "--is-ancestor", PARENT, "HEAD"], cwd=ROOT).returncode == 0
    payload = json.loads(registry.CATALOG_PATH.read_text())
    assert payload["catalog_version"] == "2.0.0"
    assert payload["accepted_parent_commit"] == PARENT
    assert tuple(payload["default_four_aliases"]) == registry.DEFAULT_FOUR_ALIASES
    assert set(payload["status_definitions"]) == set(registry.ALLOWED_STATUSES)
    assert len(payload["entries"]) == 10 and len(payload["alias_index"]) == 14
    assert payload["absent_reserved_aliases"] == ["Peak_V2", "weakT_V2", "ceramic_V2"]
    assert sum(item["status"] == "LEGACY_CANONICAL_PARAMETERIZATION" for item in payload["entries"]) == 4
    assert sum(item["status"] == "NAMED_V2_PROSPECTIVE_PARAMETERIZATION" for item in payload["entries"]) == 1
    assert sum(item["status"] == "NAMED_V2_ALTERNATE_PARAMETERIZATION" for item in payload["entries"]) == 5
    legacy_path = ROOT / "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv"
    assert hashlib.sha256(legacy_path.read_bytes()).hexdigest() == LEGACY_SHA
    selection_path = ROOT / "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_selection.json"
    assert hashlib.sha256(selection_path.read_bytes()).hexdigest() == LEGACY_SELECTION_SHA
    defaults = registry.default_four_classes()
    assert tuple(item.registry_alias for item in defaults) == registry.DEFAULT_FOUR_ALIASES
    assert all(item.generation == "LEGACY" and item.canonical_default for item in defaults)
    for alias in V2_ALIASES:
        item = registry.load(alias)
        assert not item.canonical_default
        assert registry.load_exact_candidate(item.source_candidate_id) is item
        assert all(registry.load_for(alias, loader) is item for loader in registry.SUPPORTED_LOADERS)
    assert registry.load("DBTT_V2").complete_bound_row_sha256 == "8a16415705c47c7708ec35385cde6c9a53bb6b9de6edf98b902a1f40626a9f72"
    assert registry.load("DBTT_V2").rising_resistance_status == "1D_STRONG_RISING_RESISTANCE"
    assert registry.load("DBTT_V2").fatigue_status == "THREE_FINITE_EXACT_ROW_SHORT_GROWTH_POINTS_CURVED"
    assert registry.load("DBTT_V2_P40").rising_resistance_status == "UNRESOLVED_AT_80_MPa_sqrt_m_CENSOR"
    assert len(list((ROOT / "parameterization_evidence_cards").glob("*.md"))) == 10
    decision = (ROOT / "PARAMETERIZATION_CATALOG.md").read_text()
    assert "does not supersede Legacy" in decision
    assert "no Peak-T candidate passed F1B or F2R" in decision
    assert "changes no physics, runs no simulation" in decision
    print(json.dumps({
        "status": "PASS",
        "verifier": "STRICT_NAMED_PARAMETERIZATION_REGISTRY_V2",
        "entries": 10,
        "aliases": 14,
        "legacy_defaults_changed": 0,
        "v2_default_selections": 0,
        "cross_code_loader_paths": len(registry.SUPPORTED_LOADERS),
        "physics_changes": 0,
        "simulations_run": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
