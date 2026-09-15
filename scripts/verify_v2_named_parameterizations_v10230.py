#!/usr/bin/env python3
"""Strict static verifier for the six-alias opt-in V2 overlay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture import v2_named_parameterizations as registry


PARENT = "c1b3f08d956242844ee1206bf10576f08f8a2859"


def main() -> int:
    assert subprocess.run(["git", "merge-base", "--is-ancestor", PARENT, "HEAD"], cwd=ROOT).returncode == 0
    catalog = json.loads(registry.CATALOG_PATH.read_text())
    provenance = json.loads((ROOT / "v2_named_parameterization_provenance.json").read_text())
    assert catalog["accepted_parent_commit"] == PARENT
    assert tuple(catalog["named_aliases"]) == registry.NAMED_ALIASES
    assert tuple(catalog["reserved_absent_aliases"]) == registry.RESERVED_ABSENT_ALIASES
    assert len(catalog["entries"]) == 6
    assert not provenance["established_rows_copied_into_v2_catalog"]
    assert provenance["established_aliases_added"] == []
    assert not provenance["default_selection_changed"]
    assert not provenance["physics_changed"] and provenance["simulations_run"] == 0
    for relative, expected in provenance["unchanged_parent_files_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    for alias in registry.NAMED_ALIASES:
        record = registry.load(alias)
        assert record.registry_alias == alias and not alias.endswith("_Legacy")
        assert registry.load(record.source_candidate_id) is record
        assert all(registry.load_for(alias, loader) is record for loader in registry.SUPPORTED_LOADERS)
    assert registry.load("DBTT_V2").status == "NAMED_V2_PROSPECTIVE_PARAMETERIZATION"
    assert all(registry.load(alias).status == "NAMED_V2_ALTERNATE_PARAMETERIZATION" for alias in registry.NAMED_ALIASES[1:])
    assert len(list((ROOT / "v2_parameterization_evidence_cards").glob("*.md"))) == 6
    print(json.dumps({"status": "PASS", "verifier": "STRICT_OPT_IN_NAMED_V2_PARAMETERIZATIONS", "named_aliases": 6, "exact_only_controls": 2, "established_rows_touched": 0, "default_selection_changes": 0, "physics_changes": 0, "simulations_run": 0}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
