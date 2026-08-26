import json
from pathlib import Path

from scripts.verify_v10_2_30_reversible_solver_qualification import verify


def _write(path: Path, name: str, value) -> None:
    path.mkdir()
    (path / name).write_text(json.dumps(value) + "\n")


def test_real_run_verifier_enforces_return_and_atomic_event_contract(tmp_path):
    positive, negative, one, multi = [tmp_path / name for name in ("p", "n", "e", "m")]
    _write(positive, "high_cycle_live_checkpoint.json", {"ledgers": {
        "mpz.cumulative_physical_returned_mobile[0,0]": 0.0}})
    _write(negative, "high_cycle_live_checkpoint.json", {"ledgers": {
        "mpz.cumulative_physical_returned_mobile[0,0]": 0.25,
        "mpz.cumulative_cancelled_source_slip[0,0]": 0.25,
        "mpz.cumulative_gross_source_activity": 2.0}})

    snapshot = {"complete_active_state": True}
    audit = {
        "threshold_action": 0.5,
        "hazard_action_completed": 0.5,
        "energy_admissible_advance_m": 2.0e-6,
        "geometry_committed_advance_m": 2.0e-6,
        "mpz_translated_advance_m": 2.0e-6,
        "pre_event_state": snapshot,
        "pre_geometry_commit_state": snapshot,
        "post_event_state": snapshot,
    }
    event = {"event_transaction_audit": audit}
    _write(one, "hazard_energy_gated_events_v10_2_30.json", [event])
    _write(multi, "hazard_energy_gated_events_v10_2_30.json", [event] * 3)
    (multi / "high_cycle_live_history.jsonl").write_text("\n".join(
        json.dumps({"reason": "outer_driver_geometry_committed",
                    "high_cycle_cache": {"invalidated_reason": "first_passage_event"}})
        for _ in range(3)) + "\n")

    result = verify(positive, negative, one, multi)
    assert result["passed"] is True
    assert result["metrics"]["multi_event_count"] == 3
