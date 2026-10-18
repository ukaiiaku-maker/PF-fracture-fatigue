import json

from scripts.r10_finite_compliance_campaign import hcf_rows, native_rows


def test_campaign_aggregation_preserves_complete_material_identity(tmp_path):
    native = tmp_path / "native_round_0" / "CAND" / "K.json"
    native.parent.mkdir(parents=True)
    native.write_text(json.dumps({
        "candidate_id": "CAND", "same_complete_material_sha256": "a" * 64,
        "paired_plasticity_sha256": "b" * 64, "shape": "compact_independent",
        "mode": "cohesive_mixed", "cycle_history": [{"extension_m": 1e-6}],
        "events": [{}], "stop_category": "TARGET_TRUNCATED_CASCADE",
    }))
    rows = native_rows(tmp_path)
    assert rows[0]["same_complete_material_sha256"] == "a" * 64
    assert rows[0]["paired_plasticity_sha256"] == "b" * 64
    assert rows[0]["event_count"] == 1


def test_campaign_hcf_aggregation_requires_native_schema(tmp_path):
    good = tmp_path / "hcf" / "CAND" / "m" / "result.json"
    good.parent.mkdir(parents=True)
    good.write_text(json.dumps({
        "schema": "r10.czm-owned-hcf-trajectory/1", "candidate_id": "CAND",
        "complete_row_sha256": "c" * 64, "accepted_cycles": 1e8,
        "physical_event_count": 0, "disposition": "PHYSICAL_RUNOUT_TO_DECLARED_HORIZON",
    }))
    ignored = tmp_path / "other" / "result.json"
    ignored.parent.mkdir(); ignored.write_text(json.dumps({"schema": "other"}))
    rows = hcf_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "CAND"
    assert rows[0]["accepted_cycles"] == 1e8
