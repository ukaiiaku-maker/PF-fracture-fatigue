from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.external_evidence
def test_common_specimen_audit_detects_executed_accommodation_mismatch():
    audit = _load("audit_common_specimen_contract")
    payload = audit.classify_contract(audit.DEFAULT_PF, audit.DEFAULT_FEM,
                                      Path("/does/not/exist"))
    assert payload["metadata_defect"]["false_full_field_parity_claim"] is True
    mismatch = payload["specimen_contract"]["unmatched_fields"]
    assert mismatch["accommodation"]["pf"] == "tip_only"
    assert mismatch["accommodation"]["fem"] == "full_field"
    assert mismatch["process_zone_length_m"] == {"pf": 1e-6, "fem": 5e-5}
    assert payload["prior_first_row"]["pf_over_fem_reaction_ratio"] > 10.0


def test_elastic_comparison_uses_relative_differences(tmp_path):
    runner = _load("run_common_specimen_elastic_reference")
    row = {
        "total_opening_m": 2e-7,
        "top_reaction_N_per_m": 2.0,
        "stored_energy_J_per_m": 3.0,
        "common_domain_J_J_per_m2": 4.0,
        "common_domain_KJ_Pa_sqrt_m": 5.0,
        "ahead_8_16um_sigma_yy_Pa": 6.0,
    }
    payload = {"mesh": {"nodes_sha256": "a", "elements_sha256": "b", "damage_sha256": "c"},
               "rows": [row]}
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    out = tmp_path / "comparison.json"
    import json
    left.write_text(json.dumps(payload))
    right.write_text(json.dumps(payload))
    result = runner._combine(left, right, out)
    assert result["passed"] is True
    assert result["maximum_relative_difference"] == 0.0
