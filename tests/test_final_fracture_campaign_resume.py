from __future__ import annotations

import json

from scripts import run_final_fracture_campaign as campaign


def test_successful_attempt_skips_failed_base_and_finds_verified_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    row = {"output_root": "runs/case"}
    failed = tmp_path / "runs" / "case"
    passed = tmp_path / "runs" / "case_run2"
    failed.mkdir(parents=True)
    passed.mkdir()
    (failed / "shared_runtime_attestation.post.json").write_text(
        json.dumps({"execution_verified": False})
    )
    (failed / "summary.json").write_text("[{}]")
    (passed / "shared_runtime_attestation.post.json").write_text(
        json.dumps({"execution_verified": True})
    )
    (passed / "summary.json").write_text("[{}]")
    assert campaign.successful_attempt(row) == passed


def test_close_matrix_row_distinguishes_event_and_physical_censor(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    for name, first, expected in (
        ("event", 42.0, "complete_validated"),
        ("censor", None, "complete_physical_censor"),
    ):
        result = tmp_path / name
        result.mkdir()
        (result / "summary.json").write_text(
            json.dumps([{"Kc_first_MPa_sqrt_m": first}])
        )
        (result / "shared_runtime_attestation.post.json").write_text(
            json.dumps({"source": {"commit": "abcdef1234567890"}})
        )
        row = {}
        campaign.close_matrix_row(row, result)
        assert row["status"] == expected
        assert row["producer_commit"] == "abcdef123456"


def test_nonzero_orientation_does_not_reuse_theta_zero_kernel():
    row = {
        "arm": "accommodation_enabled",
        "material_row": "DBTT",
        "temperature_K": "300",
        "orientation_deg": "45",
        "rate_factor": "1.0",
        "seed": "1733",
        "source_geography": "tip_surface",
    }
    _, native = campaign.configured_case(row)
    assert "--signed-kernel-family" not in native
    assert native[native.index("--crystal-theta-deg") + 1] == "45"
