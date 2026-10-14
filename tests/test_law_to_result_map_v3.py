from scripts.build_law_to_result_map_v3 import FIGURES, build


def test_law_to_result_map_covers_every_retained_family_and_required_axis():
    payload = build()
    assert {row["family"] for row in payload["families"]} == set(FIGURES)
    required = {
        "completion_mapping", "density_order", "phi_policy", "raw_domain_and_clamps",
        "source_lifecycle", "operator", "mechanical_backend", "input_identity", "figures",
    }
    assert all(required <= set(row) for row in payload["families"])
    assert all(row["frozen_status"].startswith("HISTORICAL_") for row in payload["families"])
    assert "not predictions" in payload["frozen_status"]


def test_current_readiness_outputs_are_not_promoted_to_specimen_results():
    payload = build()
    assert len(payload["current_readiness_outputs"]) == 3
    assert all("scope" in row for row in payload["current_readiness_outputs"])
