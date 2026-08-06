from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_v10_4_3_plastic_dominance_campaign.sh"
MONITOR = ROOT / "scripts" / "monitor_v10_4_3_campaign.py"


def test_fresh_launcher_defaults_to_new_v1043_root() -> None:
    text = LAUNCHER.read_text()
    assert "v10_4_3_theta0_rate1x_bulk_PT_positiveJ_plastic_dominance_fresh48_base3621_v1" in text
    assert "Usage: $0 {pilot|full}" in text
    assert "mkdir -p \"$OUTROOT\"" in text


def test_fresh_launcher_forbids_inherited_materializations() -> None:
    text = LAUNCHER.read_text()
    assert "v10_4_2_reuse_audit.json" in text
    assert "inherited-case reuse is forbidden" in text
    assert "symbolic link found under fresh campaign root" in text
    assert "SKIP_REUSED_VERIFIED" in text
    assert "inherited-case reuse occurred in a fresh48 campaign" in text


def test_fresh_launcher_records_commit_locked_campaign_intent() -> None:
    text = LAUNCHER.read_text()
    assert "v10.4.3_fresh48_campaign_intent_v1" in text
    assert '"planned_case_count": 48' in text
    assert '"inherited_reuse_permitted": False' in text
    assert '"all_cases_recomputed_with_v10_4_3": True' in text
    assert "campaign intent mismatch" in text


def test_full_launch_no_longer_requires_reuse_smoke() -> None:
    text = LAUNCHER.read_text()
    assert "APPROVE_FULL_CAMPAIGN=YES" in text
    assert "v10_4_3_reuse_smoke_ok.json" not in text
    assert "missing successful reuse smoke record" not in text
    assert "Campaign acceptance: planned=48 complete=48 failed_or_incomplete=0" in text


def test_monitor_default_tracks_fresh48_root() -> None:
    text = MONITOR.read_text()
    assert "plastic_dominance_fresh48_base3621_v1" in text
    assert "reuse17_base3621_v1" not in text
