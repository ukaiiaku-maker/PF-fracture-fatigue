from __future__ import annotations

import ast
import json
from pathlib import Path


def test_v10214_atlas_assembler_parses_and_is_fail_closed():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_v10_2_14_active_only_real_signed_atlas.py"
    text = script.read_text()
    ast.parse(text)
    assert "REQUIRED_STATES = (\"E000\", \"E200\", \"E500\", \"E800\")" in text
    assert "--independent-review" in text
    assert "--authorize-production-parameterization" in text
    assert "mechanics_rerun_performed\": False" in text
    assert "production_parameterization_allowed") is not True" in text
    assert "refusing active-only promotion" in text


def test_v10214_independent_review_template_starts_unapproved():
    root = Path(__file__).resolve().parents[1]
    path = root / "docs" / "v10_2_14_active_only_independent_review_template.json"
    payload = json.loads(path.read_text())
    assert payload["schema"] == (
        "v10.2.13_independent_extension_only_real_signed_atlas_review"
    )
    checks = [
        key
        for key in payload
        if key.endswith("_reviewed") or key.endswith("_passed")
    ]
    assert checks
    assert all(payload[key] is False for key in checks)
    assert payload["reviewer"] == ""
    assert payload["reviewed_utc"] == ""
