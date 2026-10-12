from pathlib import Path

from arrhenius_fracture.prospective_campaign_v2 import build_cases_v2, summary, validate_plan


ROOT = Path(__file__).resolve().parents[1]


def test_v2_preserves_v1_size_but_repairs_plan_semantics():
    rows = build_cases_v2(ROOT)
    assert len(rows) == 644
    assert validate_plan(rows)["valid"]
    assert all(r["goal_ids"] and r["dependencies"] for r in rows)
    assert all(r["gate"] is None for r in rows)
    assert all(not r["production_execution_authorized"] for r in rows)
    assert all(not r["native_args"] for r in rows if r["command_status"] != "RESOLVED_DISABLED")
    p1 = next(r for r in rows if r["block"] == "P1")
    assert p1["physical_endpoint"]["kind"] == "first_accepted_fracture_event"
    p7 = next(r for r in rows if r["block"] == "P7" and r["cavity_mode"] != "disabled")
    assert p7["cavity_geometry"]["unit"] in {"m", "process_zone_length"}


def test_v2_digest_distinguishes_semantic_and_raw_bytes():
    rows = build_cases_v2(ROOT)
    info = summary(rows, b"not-the-canonical-jsonl")
    assert info["canonical_semantic_sha256"] != info["raw_jsonl_sha256"]
