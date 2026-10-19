from pathlib import Path

from scripts.package_r10_12h_campaign import campaign_files


def test_campaign_export_keeps_restart_state_but_not_append_only_history(tmp_path):
    campaign = tmp_path / "campaign"
    checkpoint = campaign / "case" / "checkpoint"
    checkpoint.mkdir(parents=True)
    keep = checkpoint / "high_cycle_live_checkpoint.json"
    state = checkpoint / "high_cycle_live_state.npz"
    drop = checkpoint / "high_cycle_live_history.jsonl"
    result = checkpoint.parent / "result.json"
    summary = campaign / "inverse" / "inverse_summary.json"
    delivery = campaign / "campaign_delivery" / "figures" / "curve.pdf"
    summary.parent.mkdir(parents=True)
    delivery.parent.mkdir(parents=True)
    for path in (keep, state, drop, result, summary, delivery):
        path.write_bytes(b"evidence")
    selected = set(campaign_files(campaign))
    assert {keep, state, result} <= selected
    assert {summary, delivery} <= selected
    assert drop not in selected
