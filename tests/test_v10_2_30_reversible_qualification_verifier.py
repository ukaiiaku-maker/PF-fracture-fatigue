import json
from pathlib import Path

import numpy as np

from scripts.verify_v10_2_30_reversible_solver_qualification import verify


def _write(path: Path, name: str, value) -> None:
    path.mkdir()
    (path / name).write_text(json.dumps(value) + "\n")


def test_real_run_verifier_enforces_return_and_atomic_event_contract(tmp_path):
    positive, negative, one, multi, explicit, accelerated = [
        tmp_path / name for name in ("p", "n", "e", "m", "x", "a")]
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

    checkpoint = {
        "diagnostics": {
            "active_K_shield_Pa_sqrt_m": 2.0, "emission_hazard_s": 3.0,
            "mobile_count": 4.0, "retained_count": 5.0,
            "sigma_back_Pa": 6.0, "tip_radius_m": 7.0},
        "stochastic": {"hazard_action_current": 0.2},
        "ledgers": {"mpz.cumulative_gross_source_activity": 8.0,
                    "mpz.escaped_total": 1.0e-10},
    }
    for path, modes in ((explicit, {"exact_cycle_burst": 2}),
                        (accelerated, {"exact_cycle_burst": 1, "slow_projective": 1})):
        _write(path, "high_cycle_live_checkpoint.json", checkpoint)
        np.savez(path / "high_cycle_live_state.npz", active_vector=np.array([1.0, 2.0]))
        (path / "high_cycle_summary.json").write_text(json.dumps({"mode_counts": modes}) + "\n")

    result = verify(positive, negative, one, multi, explicit, accelerated)
    assert result["passed"] is True
    assert result["metrics"]["multi_event_count"] == 3
