from pathlib import Path

from scripts.audit_campaign_v3_acceptance import audit


ROOT = Path(__file__).resolve().parents[1]


def test_current_campaign_has_exact_domain_rows_and_static_bindings():
    report = audit(ROOT / "common_framework_readiness/PROSPECTIVE_CAMPAIGN_V3.jsonl")
    assert report["case_count"] == 699
    assert report["native_command_count"] == 699
    assert len(report["raw_domain_blocked_rows"]) == 3
    assert len(report["near_boundary_rows"]) == 20
    assert report["raw_blocked_near_boundary_overlap_case_ids"] == []
    assert report["fitted_bilinear_qualification_request_count"] == 36
    assert report["binding_errors"] == {}
